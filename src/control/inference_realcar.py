"""
inference_realcar.py — Inférence temps réel sur la voiture physique.

Hardware : Jetson Nano + Luxonis OAK-D Lite + Flipsky FSESC Mini V6.7 Pro
Modèle   : RobocarSpatial v18 (ONNX) — input [23] : 20 rays Z-scorés + 3 derived features

Architecture multi-thread :
  Thread 1 (perception)  : OAK-D → depth frame → raycasts virtuels
  Thread 2 (contrôle)    : raycasts → ONNX → smoother → VESC
  Watchdog               : arrêt auto si pas de frame depuis WATCHDOG_S secondes

⚠️  AVANT PREMIER TEST :
  1. python3.8 -m src.tools.calibrate_servo  → valider center/range/invert
  2. python3.8 -m src.tools.calibrate_rays  → collecter Z-score réel (3 phases)
  3. VESC Tool → App Settings → General → Timeout = 200ms  (watchdog hardware)
  4. Tester ROUES EN L'AIR avant de poser la voiture au sol

Usage :
  python3.8 -m src.control.inference_realcar --duty-max 0.20
  python3.8 -m src.control.inference_realcar --duty-max 0.20 --servo-center 0.52 --invert-steer
  python3.8 -m src.control.inference_realcar --duty-max 0.20 --perception-mode visual
"""

import csv
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("[WARNING] onnxruntime non installé")

try:
    import depthai as dai
    _DAI_AVAILABLE = True
except ImportError:
    _DAI_AVAILABLE = False
    print("[WARNING] depthai non installé")

from src.mask.depth_rays import DepthToRays, create_depthai_pipeline, add_stereo_depth
from src.mask.visual_rays   import VisualRays, create_color_pipeline
from src.control.vesc_interface import VESCInterface
from src.features import derive_features
from src.control_post import SmoothingFilter, apply_steer_offset, accel_from_geometry

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH          = "models/v18/best.onnx"
STATS_PATH          = "models/real_ray_stats.json"
FALLBACK_STATS_PATH = "models/ray_stats.json"
VESC_PORT           = "/dev/ttyACM0"
CURRENT_MAX         = 5.0    # A — courant max moteur (8A normal, 5A conservateur 1er test)
WATCHDOG_S          = 0.50   # 500ms — tolérance OAK-D démarrage
CONTROL_HZ          = 30
# Arrêt d'urgence obstacle (modes depth/fusion, direct-device ET hub)
EMERGENCY_NEAR_MM       = 500.0   # obstacle mesuré plus proche que ça (zone centrale) -> stop
EMERGENCY_VALID_FRAC_MIN = 0.003  # "blackout" : objet collé qui occulte la lentille.
                                  # Seuil bas car la stéréo OAK-D Lite est déjà éparse
                                  # (~1% de pixels valides en usage normal -> marge x3).
EMERGENCY_CENTER_BAND   = (0.40, 0.60)  # fraction de colonnes (centre image) scrutée
DEPTH_DEAD_MAX_MM       = 200    # frame.max() en deçà -> depth map nulle (capteur/hub perdu)


class RealCarInference:
    def __init__(
        self,
        model_path: str   = MODEL_PATH,
        stats_path: str   = STATS_PATH,
        vesc_port: str    = VESC_PORT,
        current_max: float = CURRENT_MAX,
        duty_max: float    = 0.20,
        servo_center: float = 0.50,
        servo_range: float  = 0.35,
        invert_steer: bool  = False,
        invert_motor: bool  = False,   # duty mode : accel positif -> marche avant
        steer_gain: float   = 1.0,
        log_csv: bool       = True,
        perception_mode: str = "depth",
        mask_mode: str       = "hsv",
        source: str          = "device",
    ):
        print("\n[RealCar] ══════════════════════════════════════════")
        print("[RealCar]  Behavioral Cloning — Voiture Réelle v18  ")
        print("[RealCar] ══════════════════════════════════════════")

        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime non installé.")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.sess = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[RealCar] ONNX chargé — provider: {self.sess.get_providers()[0]}")

        # ── Z-score ───────────────────────────────────────────────────────────
        real_stats = Path(stats_path)
        sim_stats  = Path(FALLBACK_STATS_PATH)
        if real_stats.exists():
            with open(real_stats) as f:
                stats = json.load(f)
            print(f"[RealCar] Z-score RÉEL chargé ✅")
        elif sim_stats.exists():
            with open(sim_stats) as f:
                stats = json.load(f)
            print(f"[RealCar] ⚠️  Z-score SIMULATION utilisé (recalibrer avec calibrate_ray_stats.py)")
        else:
            raise FileNotFoundError("Aucun ray_stats.json trouvé")

        self.ray_mu    = np.array(stats["mean"], dtype=np.float32)
        self.ray_sigma = np.array(stats["std"],  dtype=np.float32)

        self.perception_mode = perception_mode
        self.source          = source
        self.depth_bridge    = DepthToRays()
        self.visual_bridge   = VisualRays(mode=mask_mode)

        # ── VESC avec params calibrés ─────────────────────────────────────────
        self.vesc = VESCInterface(
            port=vesc_port,
            current_max=current_max,
            servo_center=servo_center,
            servo_range=servo_range,
            invert_steer=invert_steer,
            invert_motor=invert_motor,
            throttle_mode="duty",   # duty = plus smooth que current au démarrage sensorless
            max_duty=duty_max,
        )
        print(f"[RealCar] servo_center={servo_center:.3f} | range=±{servo_range:.3f} | invert={invert_steer} | duty_max={duty_max:.2f}")

        self.smoother   = SmoothingFilter()
        self.steer_gain = steer_gain

        self._lock              = threading.Lock()
        self._latest_rays: Optional[np.ndarray] = None
        self._last_frame_t      = time.time()
        self._perception_ready  = False   # True quand OAK-D envoie des frames
        self._proximity_blocked = False   # True = obstacle trop proche (arrêt d'urgence)
        self._running           = False
        self._step         = 0
        self._fps_times    = deque(maxlen=50)
        self._start_time   = None

        # ── CSV log ───────────────────────────────────────────────────────────
        self._log_csv  = log_csv
        self._csv_file = None
        self._csv_writer = None
        if log_csv:
            log_path = Path(f"logs/run_{int(time.time())}.csv")
            log_path.parent.mkdir(exist_ok=True)
            self._csv_file   = open(log_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["t", "steer", "steer_raw", "asym", "accel", "front_raw", "min_ray", "fps"])
            print(f"[RealCar] Log CSV → {log_path}")

        print(f"[RealCar] CURRENT_MAX = {current_max:.1f}A | WATCHDOG = {WATCHDOG_S*1000:.0f}ms")
        print("[RealCar] ✅ Prêt. Lance run() pour démarrer.\n")

    def _check_proximity(self, depth_frame) -> bool:
        """Obstacle trop proche dans la bande centrale ? (depth uint16, mm)

        Deux déclencheurs, car la stéréo a un angle mort sous ~35cm (MinZ) :
          1. obstacle MESURÉ proche : médiane des pixels valides < EMERGENCY_NEAR_MM
          2. "blackout"             : quasi aucun pixel valide (objet collé qui
             occulte la lentille) -> sinon l'obstacle "disparaît" et la voiture réaccélère.
        On teste la fraction de pixels VALIDES (pas l'inverse) : la depth OAK-D Lite
        est déjà éparse (~1% valides en normal), donc seul un blackout total discrimine.
        """
        b = self.depth_bridge
        c0, c1 = int(b.W * EMERGENCY_CENTER_BAND[0]), int(b.W * EMERGENCY_CENTER_BAND[1])
        roi = depth_frame[b.row_start:b.row_end, c0:c1]
        valid = roi[(roi >= b.min_valid_mm) & (roi <= b.max_dist_mm)]
        if valid.size and float(np.median(valid)) < EMERGENCY_NEAR_MM:
            return True
        return (valid.size / float(roi.size)) < EMERGENCY_VALID_FRAC_MIN

    def _apply_calib_fov(self, device, socket, bridge, label):
        """Applique le FOV usine du capteur au bridge (fallback = constante codée)."""
        try:
            fov = device.readCalibration().getFov(socket)
            bridge.set_fov(fov)
            print(f"[Perception] FOV {label} = {fov:.1f}° (calibration usine)")
        except Exception as e:
            print(f"[Perception] FOV {label} : getFov indisponible ({e}) — fallback {bridge.fov_deg:.1f}°")

    def _perception_thread(self):
        if not _DAI_AVAILABLE:
            print("[Perception] depthai non disponible")
            return
        if self.perception_mode == "visual":
            self._perception_visual()
        elif self.perception_mode == "fusion":
            self._perception_fusion()
        else:
            self._perception_depth()

    def _publish(self, rays, prox: bool = False) -> None:
        """Publie atomiquement la dernière frame perçue : rays + flag proximité + timestamp.

        Invariant unique des 6 boucles de perception. prox=False pour les modes
        sans depth (visual) — l'arrêt d'urgence y est inactif par construction.
        """
        with self._lock:
            self._latest_rays       = rays
            self._proximity_blocked = prox
            self._last_frame_t      = time.time()

    def _perception_depth(self):
        if self.source == "hub":
            self._perception_depth_hub(); return
        pipeline = create_depthai_pipeline()
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("depth", maxSize=1, blocking=False)
            print("[Perception] OAK-D connectée — flux DEPTH démarré")
            self._apply_calib_fov(device, dai.CameraBoardSocket.CAM_B, self.depth_bridge, "depth")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True
            while self._running:
                msg = q.tryGet()
                if msg is not None:
                    frame = msg.getFrame()
                    if frame.max() < DEPTH_DEAD_MAX_MM:
                        print("\n[Perception] ⚠️  Depth map nulle — perte OAK-D !")
                        self.vesc.stop()
                    else:
                        rays = self.depth_bridge(frame)
                        self._publish(rays, self._check_proximity(frame))
                time.sleep(0.005)

    def _perception_depth_hub(self):
        """Mode depth alimenté par le hub (SHM robocar_cam_depth, zéro-copie)."""
        from src.cam.hub import FrameClient, SHM_DEPTH
        client = FrameClient(stream=SHM_DEPTH)
        print("[Perception] source = hub (SHM robocar_cam_depth) — flux DEPTH")
        with self._lock:
            self._last_frame_t     = time.time()
            self._perception_ready = True
        while self._running:
            try:
                frame = client.getCvFrame()
            except (ConnectionError, OSError):
                print("[Perception] hub depth indisponible — reconnexion…")
                client.close(); time.sleep(0.5); continue
            if frame.max() < DEPTH_DEAD_MAX_MM:
                print("\n[Perception] ⚠️  Depth map nulle — hub/OAK-D ?")
                self.vesc.stop()
            else:
                rays = self.depth_bridge(frame)
                self._publish(rays, self._check_proximity(frame))
        client.close()

    def _perception_visual_hub(self):
        """Mode visual alimenté par le hub (frames en mémoire partagée, zéro-copie) — pas
        d'ouverture caméra ici. Permet à la preview (mask_stream) et à l'inférence de
        partager l'OAK-D."""
        from src.cam.hub import FrameClient
        client = FrameClient()
        print("[Perception] source = hub (SHM robocar_cam_color) — flux VISUAL (masque)")
        with self._lock:
            self._last_frame_t     = time.time()
            self._perception_ready = True
        while self._running:
            try:
                bgr = client.getCvFrame()
            except (ConnectionError, OSError):
                print("[Perception] hub indisponible — reconnexion…")
                client.close()
                time.sleep(0.5)
                continue
            rays = self.visual_bridge(bgr)
            self._publish(rays)
        client.close()

    def _perception_visual(self):
        if self.source == "hub":
            self._perception_visual_hub()
            return
        pipeline = create_color_pipeline()
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("color", maxSize=2, blocking=False)
            print("[Perception] OAK-D connectée — flux VISUAL (masque) démarré")
            self._apply_calib_fov(device, dai.CameraBoardSocket.CAM_A, self.visual_bridge, "visual")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True
            while self._running:
                msg = q.tryGet()
                if msg is not None:
                    rays = self.visual_bridge(msg.getCvFrame())
                    self._publish(rays)
                time.sleep(0.005)

    def _perception_fusion_hub(self):
        """Fusion depth+visual alimentée par le hub (SHM, zéro-copie). La couleur cadence la
        boucle (getCvFrame bloque sur une nouvelle frame) ; le depth est lu en 'dernière dispo'
        (latest, non bloquant) pour ne pas freiner la couleur."""
        from src.cam.hub import FrameClient, SHM_COLOR, SHM_DEPTH
        c_color = FrameClient(stream=SHM_COLOR)
        c_depth = FrameClient(stream=SHM_DEPTH)
        print("[Perception] source = hub (SHM color+depth) — flux FUSION")
        with self._lock:
            self._last_frame_t     = time.time()
            self._perception_ready = True
        while self._running:
            try:
                bgr = c_color.getCvFrame()
            except (ConnectionError, OSError):
                print("[Perception] hub indisponible — reconnexion…")
                c_color.close(); c_depth.close(); time.sleep(0.5); continue
            rays_visual = self.visual_bridge(bgr)
            depth = c_depth.latest()
            prox = False
            if depth is not None and depth.max() >= DEPTH_DEAD_MAX_MM:
                prox = self._check_proximity(depth)
                rays_fused = np.minimum(self.depth_bridge(depth), rays_visual)
            else:
                rays_fused = rays_visual   # fallback si depth absent/KO
            self._publish(rays_fused, prox)
        c_color.close(); c_depth.close()

    def _perception_fusion(self):
        """
        Fusion depth + visual : np.minimum(rays_depth, rays_visual)
        → conserve le signal le plus contraignant (obstacle le plus proche).
        Depth détecte les murs/obstacles en volume.
        Visual détecte les bords de piste (lignes blanches au sol).
        """
        if self.source == "hub":
            self._perception_fusion_hub(); return
        # Un seul pipeline v2 : couleur (CAM_A) + depth stéréo via la config canonique partagée
        # (add_stereo_depth = même config que le mode depth et le hub → DepthToRays reste calibré)
        pipeline = create_color_pipeline()
        stereo = add_stereo_depth(pipeline, dai)
        xout_d = pipeline.create(dai.node.XLinkOut)
        xout_d.setStreamName("depth")
        stereo.depth.link(xout_d.input)

        with dai.Device(pipeline) as device:
            q_color = device.getOutputQueue("color", maxSize=2, blocking=False)
            q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
            print("[Perception] OAK-D connectée — flux FUSION (depth + masque) démarré")
            self._apply_calib_fov(device, dai.CameraBoardSocket.CAM_B, self.depth_bridge,  "depth")
            self._apply_calib_fov(device, dai.CameraBoardSocket.CAM_A, self.visual_bridge, "visual")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True

            while self._running:
                bgr_msg   = q_color.get()
                depth_msg = q_depth.tryGet()

                rays_visual = self.visual_bridge(bgr_msg.getCvFrame())

                prox = False
                if depth_msg is not None:
                    frame = depth_msg.getFrame()
                    if frame.max() >= DEPTH_DEAD_MAX_MM:
                        rays_depth = self.depth_bridge(frame)
                        prox = self._check_proximity(frame)
                        # Fusion : minimum → le signal le plus contraignant
                        rays_fused = np.minimum(rays_depth, rays_visual)
                    else:
                        rays_fused = rays_visual   # fallback si depth KO
                else:
                    rays_fused = rays_visual       # fallback si pas de frame depth

                self._publish(rays_fused, prox)
                time.sleep(0.005)

    def _control_thread(self):
        print(f"[Control] Boucle démarrée @ {CONTROL_HZ} Hz")
        interval = 1.0 / CONTROL_HZ

        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                rays       = self._latest_rays
                last_frame = self._last_frame_t
                blocked    = self._proximity_blocked

            # ── Watchdog : inactif tant que OAK-D pas prête ───────────────
            if not self._perception_ready:
                time.sleep(0.05)
                continue

            if time.time() - last_frame > WATCHDOG_S:
                self.vesc.stop()
                time.sleep(0.1)
                continue

            if rays is None:
                time.sleep(0.01)
                continue

            # ── Arrêt d'urgence : obstacle trop proche (depth/fusion) ─────
            if blocked:
                self.vesc.stop()
                if self._step % 30 == 0:
                    print("\r[Control] ⛔ OBSTACLE PROCHE — arrêt d'urgence            ", end="", flush=True)
                time.sleep(0.02)
                continue

            # ── Z-score ───────────────────────────────────────────────────
            rays_z = (rays - self.ray_mu[:20]) / (self.ray_sigma[:20] + 1e-8)

            # ── Features dérivées ─────────────────────────────────────────
            half     = len(rays_z) // 2   # conservé pour front_raw (heuristique accel)
            derived  = derive_features(rays_z)
            features = np.concatenate([rays_z, derived]).astype(np.float32)

            # ── Inférence ONNX ────────────────────────────────────────────
            pred      = self.sess.run(None, {self.input_name: features[np.newaxis, :]})[0][0]
            steer_raw = float(pred[0])
            accel_raw = float(1.0 / (1.0 + np.exp(-pred[1])))  # sigmoid sur logit

            # ── Smoothing ─────────────────────────────────────────────────
            pred_s   = self.smoother.update(np.array([steer_raw, accel_raw], dtype=np.float32))
            steer    = apply_steer_offset(pred_s[0])
            steering = float(np.clip(steer * self.steer_gain, -1.0, 1.0))

            # ── Heuristique accel (front_raw) ─────────────────────────────
            # réel : corner_damp ON + plancher 0.50 (diverge du sim — voir control_post)
            front_raw    = float(rays[half - 1: half + 1].mean())
            acceleration = accel_from_geometry(steering, front_raw,
                                               corner_damp=True, accel_floor=0.50)

            # ── Envoi VESC ────────────────────────────────────────────────
            self.vesc.send(steering, acceleration)

            # ── Métriques + CSV ───────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            self._fps_times.append(elapsed)
            self._step += 1
            fps = 1.0 / (np.mean(self._fps_times) + 1e-9)

            if self._csv_writer and self._step % 3 == 0:
                self._csv_writer.writerow([
                    round(time.time() - self._start_time, 3),
                    round(steering, 4), round(steer_raw, 4), round(float(derived[0]), 4),
                    round(acceleration, 4),
                    round(front_raw, 4), round(float(rays.min()), 4),
                    round(fps, 1),
                ])

            if self._step % 50 == 0:
                side = "L" if steering < -0.05 else ("R" if steering > 0.05 else "=")
                bar  = "#" * int(abs(steering) * 12)
                print(
                    f"\r  [{side}] {bar:<12} "
                    f"steer={steering:+.3f} accel={acceleration:.3f} "
                    f"front={front_raw:.2f} {fps:.0f}fps    ",
                    end="", flush=True,
                )

            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def run(self):
        self._running    = True
        self._start_time = time.time()

        t_perc = threading.Thread(target=self._perception_thread, daemon=True, name="perception")
        t_ctrl = threading.Thread(target=self._control_thread,   daemon=True, name="control")
        t_perc.start()
        t_ctrl.start()

        print("[RealCar] ▶  En marche — Ctrl+C pour arrêter\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[RealCar] Arrêt demandé...")
        finally:
            self._running = False
            # Teardown atomique : un 2e Ctrl+C ne doit ni sauter la coupure moteur
            # ni laisser /dev/ttyACM0 ouvert. On ignore SIGINT le temps de fermer.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.vesc.stop()          # sécurité d'abord : moteur coupé
            time.sleep(0.2)           # laisse perception/control voir _running=False
            self.vesc.close()
            if self._csv_file:
                self._csv_file.close()
                print(f"[RealCar] Log CSV sauvegardé ✅")
            elapsed = time.time() - self._start_time
            print(f"[RealCar] Terminé — {self._step} steps | {elapsed:.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inférence voiture réelle v18")
    parser.add_argument("--model",        default=MODEL_PATH)
    parser.add_argument("--stats",        default=STATS_PATH)
    parser.add_argument("--port",         default=VESC_PORT)
    parser.add_argument("--current-max",  type=float, default=CURRENT_MAX,
                        help="Courant max moteur en A (defaut 5A pour 1er test, 8A normal)")
    parser.add_argument("--duty-max",     type=float, default=0.20,
                        help="Duty cycle max [0-1] en mode duty (defaut 0.20 = 20%%)")
    parser.add_argument("--servo-center", type=float, default=0.50)
    parser.add_argument("--servo-range",  type=float, default=0.35)
    parser.add_argument("--invert-steer", action="store_true")
    parser.add_argument("--invert-motor",    dest="invert_motor", action="store_true",  default=False,
                        help="inverse le sens moteur (à utiliser si la voiture recule au défaut)")
    parser.add_argument("--no-invert-motor", dest="invert_motor", action="store_false",
                        help="accel positif -> marche avant (défaut)")
    parser.add_argument("--steer-gain", type=float, default=1.0,
                        help="multiplicateur du steering (compense l'atténuation sim-to-real ; 1.0 = brut)")
    parser.add_argument("--no-log",          action="store_true")
    parser.add_argument("--perception-mode", choices=["depth", "visual", "fusion"], default="depth",
                        help="depth=depth map stéreo | visual=masque HSV/Canny couleur")
    parser.add_argument("--mask-mode",       choices=["hsv", "canny"], default="hsv",
                        help="Mode masque (si --perception-mode visual)")
    parser.add_argument("--source",   choices=["device", "hub"], default="hub",
                        help="hub=lit le hub caméra en mémoire partagée (défaut, partage l'OAK-D) | device=ouvre l'OAK-D")
    args = parser.parse_args()

    if args.source == "hub":
        from src.cam.hub import ensure_hub_or_prompt, SHM_COLOR, SHM_DEPTH
        # Préflight sur le flux RÉELLEMENT consommé selon le mode (sinon faux vert sur la
        # couleur alors que le mode depth lit le depth). Fusion : color est le flux bloquant,
        # le depth est best-effort (latest() → None si absent).
        stream = SHM_DEPTH if args.perception_mode == "depth" else SHM_COLOR
        if not ensure_hub_or_prompt(stream):
            sys.exit(1)

    RealCarInference(
        model_path   = args.model,
        stats_path   = args.stats,
        vesc_port    = args.port,
        current_max  = args.current_max,
        duty_max     = args.duty_max,
        servo_center = args.servo_center,
        servo_range  = args.servo_range,
        invert_steer     = args.invert_steer,
        invert_motor     = args.invert_motor,
        steer_gain       = args.steer_gain,
        log_csv          = not args.no_log,
        perception_mode  = args.perception_mode,
        mask_mode        = args.mask_mode,
        source           = args.source,
    ).run()


if __name__ == "__main__":
    main()

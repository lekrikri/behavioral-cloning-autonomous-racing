"""
inference_realcar.py — Inférence temps réel sur la voiture physique.

Hardware : Jetson Nano + Luxonis OAK-D Lite + Flipsky FSESC Mini V6.7 Pro
Modèle   : RobocarSpatial v18 (ONNX) — input [23] : 20 rays Z-scorés + 3 derived features

Architecture multi-thread :
  Thread 1 (perception)  : OAK-D → depth frame → raycasts virtuels
  Thread 2 (contrôle)    : raycasts → ONNX → smoother → VESC
  Watchdog               : arrêt auto si pas de frame depuis WATCHDOG_S secondes

⚠️  AVANT PREMIER TEST :
  1. python3.8 src/calibrate_servo.py  → valider center/range/invert
  2. python3.8 src/calibrate_ray_stats.py  → collecter Z-score réel (3 phases)
  3. VESC Tool → App Settings → General → Timeout = 200ms  (watchdog hardware)
  4. Tester ROUES EN L'AIR avant de poser la voiture au sol

Usage :
  python3.8 src/inference_realcar.py --duty-max 0.20
  python3.8 src/inference_realcar.py --duty-max 0.20 --servo-center 0.52 --invert-steer
  python3.8 src/inference_realcar.py --duty-max 0.20 --perception-mode visual
"""

import csv
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

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

from src.depth_to_rays import DepthToRays, create_depthai_pipeline
from src.visual_rays   import VisualRays, create_color_pipeline_v3
from src.vesc_interface import VESCInterface

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH          = "models/v18/best.onnx"
STATS_PATH          = "models/real_ray_stats.json"
FALLBACK_STATS_PATH = "models/ray_stats.json"
VESC_PORT           = "/dev/ttyACM0"
CURRENT_MAX         = 5.0    # A — courant max moteur (8A normal, 5A conservateur 1er test)
WATCHDOG_S          = 0.50   # 500ms — tolérance OAK-D démarrage
CONTROL_HZ          = 30


class SmoothingFilter:
    def __init__(self, alpha: float = 0.57, alpha_max: float = 0.92, deadzone: float = 0.06):
        self.alpha_base = alpha
        self.alpha_max  = alpha_max
        self.deadzone   = deadzone
        self._smoothed: Optional[np.ndarray] = None

    def reset(self):
        self._smoothed = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        if self._smoothed is None:
            self._smoothed = raw.copy()
        else:
            delta = abs(raw[0] - self._smoothed[0])
            alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * min(delta, 1.0)
            self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed
        result = self._smoothed.copy()
        if abs(result[0]) < self.deadzone:
            result[0] = 0.0
        return result

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.update(raw)


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
        log_csv: bool       = True,
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

        self.perception_mode = kwargs.get("perception_mode", "depth")
        self.depth_bridge    = DepthToRays()
        self.visual_bridge   = VisualRays(mode=kwargs.get("mask_mode", "hsv"))

        # ── VESC avec params calibrés ─────────────────────────────────────────
        self.vesc = VESCInterface(
            port=vesc_port,
            current_max=current_max,
            servo_center=servo_center,
            servo_range=servo_range,
            invert_steer=invert_steer,
            throttle_mode="duty",   # duty = plus smooth que current au démarrage sensorless
            max_duty=duty_max,
        )
        print(f"[RealCar] servo_center={servo_center:.3f} | range=±{servo_range:.3f} | invert={invert_steer} | duty_max={duty_max:.2f}")

        self.smoother = SmoothingFilter()

        self._lock              = threading.Lock()
        self._latest_rays: Optional[np.ndarray] = None
        self._last_frame_t      = time.time()
        self._perception_ready  = False   # True quand OAK-D envoie des frames
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
            self._csv_writer.writerow(["t", "steer", "accel", "front_raw", "min_ray", "fps"])
            print(f"[RealCar] Log CSV → {log_path}")

        print(f"[RealCar] CURRENT_MAX = {current_max:.1f}A | WATCHDOG = {WATCHDOG_S*1000:.0f}ms")
        print("[RealCar] ✅ Prêt. Lance run() pour démarrer.\n")

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

    def _perception_depth(self):
        pipeline = create_depthai_pipeline()
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("depth", maxSize=1, blocking=False)
            print("[Perception] OAK-D connectée — flux DEPTH démarré")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True
            while self._running:
                msg = q.tryGet()
                if msg is not None:
                    frame = msg.getFrame()
                    if frame.max() < 200:
                        print("\n[Perception] ⚠️  Depth map nulle — perte OAK-D !")
                        self.vesc.stop()
                    else:
                        rays = self.depth_bridge(frame)
                        with self._lock:
                            self._latest_rays  = rays
                            self._last_frame_t = time.time()
                time.sleep(0.005)

    def _perception_visual(self):
        device_info = None
        for d in dai.Device.getAllConnectedDevices():
            device_info = d
            break
        if device_info is None:
            print("[Perception] Aucun device OAK-D trouvé")
            return
        with dai.Device(device_info) as device:
            pipeline, q = create_color_pipeline_v3(device)
            pipeline.start()
            print("[Perception] OAK-D connectée — flux VISUAL (masque) démarré")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True
            while self._running:
                msg = q.get()
                rays = self.visual_bridge(msg.getCvFrame())
                with self._lock:
                    self._latest_rays  = rays
                    self._last_frame_t = time.time()
                time.sleep(0.005)

    def _perception_fusion(self):
        """
        Fusion depth + visual : np.minimum(rays_depth, rays_visual)
        → conserve le signal le plus contraignant (obstacle le plus proche).
        Depth détecte les murs/obstacles en volume.
        Visual détecte les bords de piste (lignes blanches au sol).
        """
        device_info = None
        for d in dai.Device.getAllConnectedDevices():
            device_info = d
            break
        if device_info is None:
            print("[Perception] Aucun device OAK-D trouvé")
            return

        import depthai as dai as dai_mod
        # Pipeline couleur + depth sur le même device
        with dai_mod.Device(device_info) as device:
            # Couleur
            pipeline_color, q_color = create_color_pipeline_v3(device)

            # Depth (ajout au même pipeline)
            mono_l = pipeline_color.create(dai_mod.node.MonoCamera)
            mono_r = pipeline_color.create(dai_mod.node.MonoCamera)
            stereo  = pipeline_color.create(dai_mod.node.StereoDepth)
            xout_d  = pipeline_color.create(dai_mod.node.XLinkOut)
            mono_l.setBoardSocket(dai_mod.CameraBoardSocket.CAM_B)
            mono_r.setBoardSocket(dai_mod.CameraBoardSocket.CAM_C)
            mono_l.setResolution(dai_mod.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_r.setResolution(dai_mod.MonoCameraProperties.SensorResolution.THE_400_P)
            stereo.setDefaultProfilePreset(dai_mod.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setMedianFilter(dai_mod.MedianFilter.KERNEL_7x7)
            mono_l.out.link(stereo.left)
            mono_r.out.link(stereo.right)
            stereo.depth.link(xout_d.input)
            xout_d.setStreamName("depth")
            q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)

            pipeline_color.start()
            print("[Perception] OAK-D connectée — flux FUSION (depth + masque) démarré")
            with self._lock:
                self._last_frame_t     = time.time()
                self._perception_ready = True

            while self._running:
                bgr_msg   = q_color.get()
                depth_msg = q_depth.tryGet()

                rays_visual = self.visual_bridge(bgr_msg.getCvFrame())

                if depth_msg is not None:
                    frame = depth_msg.getFrame()
                    if frame.max() >= 200:
                        rays_depth = self.depth_bridge(frame)
                        # Fusion : minimum → le signal le plus contraignant
                        rays_fused = np.minimum(rays_depth, rays_visual)
                    else:
                        rays_fused = rays_visual   # fallback si depth KO
                else:
                    rays_fused = rays_visual       # fallback si pas de frame depth

                with self._lock:
                    self._latest_rays  = rays_fused
                    self._last_frame_t = time.time()
                time.sleep(0.005)

    def _control_thread(self):
        print(f"[Control] Boucle démarrée @ {CONTROL_HZ} Hz")
        interval = 1.0 / CONTROL_HZ

        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                rays       = self._latest_rays
                last_frame = self._last_frame_t

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

            # ── Z-score ───────────────────────────────────────────────────
            rays_z = (rays - self.ray_mu[:20]) / (self.ray_sigma[:20] + 1e-8)

            # ── Features dérivées ─────────────────────────────────────────
            half      = len(rays_z) // 2
            asymmetry = (rays_z[half:].sum() - rays_z[:half].sum()) / (rays_z.sum() + 1e-8)
            front_ray = float(rays_z[half - 1: half + 1].mean())
            min_ray   = float(rays_z.min())
            derived   = np.array([asymmetry, front_ray, min_ray], dtype=np.float32)
            features  = np.concatenate([rays_z, derived]).astype(np.float32)

            # ── Inférence ONNX ────────────────────────────────────────────
            pred      = self.sess.run(None, {self.input_name: features[np.newaxis, :]})[0][0]
            steer_raw = float(pred[0])
            accel_raw = float(1.0 / (1.0 + np.exp(-pred[1])))  # sigmoid sur logit

            # ── Smoothing ─────────────────────────────────────────────────
            pred_s  = self.smoother.update(np.array([steer_raw, accel_raw], dtype=np.float32))
            steer   = pred_s[0]
            if 0.05 < abs(steer) < 0.35:
                steer -= 0.02 * np.sign(steer)
            steering = float(np.clip(steer, -1.0, 1.0))

            # ── Heuristique accel (front_raw) ─────────────────────────────
            front_raw = float(rays[half - 1: half + 1].mean())
            geo_base  = max(0.35, 1.0 - 1.2 * abs(steering))
            front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
            # Sécurité virage (recommandation Grok) : réduire accel si steer élevé
            corner_damp = 1.0 - 0.5 * abs(steering)
            acceleration = float(np.clip(min(geo_base, front_cap) * corner_damp, 0.50, 0.95))

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
                    round(steering, 4), round(acceleration, 4),
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
            time.sleep(0.2)
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
    parser.add_argument("--no-log",          action="store_true")
    parser.add_argument("--perception-mode", choices=["depth", "visual", "fusion"], default="depth",
                        help="depth=depth map stéreo | visual=masque HSV/Canny couleur")
    parser.add_argument("--mask-mode",       choices=["hsv", "canny"], default="hsv",
                        help="Mode masque (si --perception-mode visual)")
    args = parser.parse_args()

    RealCarInference(
        model_path   = args.model,
        stats_path   = args.stats,
        vesc_port    = args.port,
        current_max  = args.current_max,
        duty_max     = args.duty_max,
        servo_center = args.servo_center,
        servo_range  = args.servo_range,
        invert_steer     = args.invert_steer,
        log_csv          = not args.no_log,
        perception_mode  = args.perception_mode,
        mask_mode        = args.mask_mode,
    ).run()


if __name__ == "__main__":
    main()

"""
inference_realcar.py — Inférence temps réel sur la voiture physique.

Hardware : Jetson Nano + Luxonis OAK-D Lite + Flipsky FSESC Mini V6.7 Pro
Modèle   : RobocarSpatial v18 (ONNX) — input [23] : 20 rays Z-scorés + 3 derived features

Architecture multi-thread :
  Thread 1 (perception)  : OAK-D → depth frame → raycasts virtuels
  Thread 2 (contrôle)    : raycasts → ONNX → smoother → VESC
  Watchdog               : arrêt auto si pas de frame depuis WATCHDOG_S secondes

⚠️  AVANT PREMIER TEST :
  1. VESC Tool → App Settings → General → Timeout = 200ms  (watchdog hardware)
  2. DUTY_MAX = 0.15 (15% de puissance maximum)
  3. Tester ROUES EN L'AIR avant de poser la voiture au sol
  4. Vérifier le sens du servo (invert_steer si nécessaire)
"""

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Imports optionnels (pas disponibles sur PC de dev) ───────────────────
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
from src.vesc_interface import VESCInterface

# ─── Configuration ────────────────────────────────────────────────────────
MODEL_PATH   = "models/v18/best.onnx"
STATS_PATH   = "models/real_ray_stats.json"   # recalculé via calibrate_ray_stats.py
FALLBACK_STATS_PATH = "models/ray_stats.json" # stats simulation (fallback si réel absent)
VESC_PORT    = "/dev/ttyACM0"                 # vérifier avec : ls /dev/ttyACM* /dev/ttyUSB*
DUTY_MAX     = 0.15    # ⚠️  15% max pour premiers tests — augmenter progressivement
WATCHDOG_S   = 0.5     # arrêt si pas de frame depuis 500ms
CONTROL_HZ   = 30      # fréquence boucle de contrôle


class SmoothingFilter:
    """
    Filtre adaptatif identique à inference.py simulation.
    alpha élevé en virage (réactif), bas en ligne droite (stable).
    Deadzone sur le steering pour supprimer le micro-zigzag.
    """

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
    """
    Boucle d'inférence temps réel pour la voiture physique.
    Identique à inference.py simulation mais avec OAK-D + VESC à la place de RobocarEnv.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        stats_path: str = STATS_PATH,
        vesc_port: str  = VESC_PORT,
        duty_max: float = DUTY_MAX,
    ):
        print("\n[RealCar] ══════════════════════════════════════════")
        print("[RealCar]  Behavioral Cloning — Voiture Réelle v18  ")
        print("[RealCar] ══════════════════════════════════════════")

        # ── Modèle ONNX ──────────────────────────────────────────────────
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime non installé. Sur Jetson: pip install onnxruntime-gpu")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.sess = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        active_provider = self.sess.get_providers()[0]
        print(f"[RealCar] ONNX chargé — provider: {active_provider}")

        # ── Z-score ───────────────────────────────────────────────────────
        # Priorité : stats réelles > stats simulation
        real_stats = Path(stats_path)
        sim_stats  = Path(FALLBACK_STATS_PATH)
        if real_stats.exists():
            with open(real_stats) as f:
                stats = json.load(f)
            print(f"[RealCar] Z-score RÉEL chargé depuis {real_stats}")
        elif sim_stats.exists():
            with open(sim_stats) as f:
                stats = json.load(f)
            print(f"[RealCar] ⚠️  Z-score SIMULATION utilisé (recalibrer avec calibrate_ray_stats.py)")
        else:
            raise FileNotFoundError(f"Aucun ray_stats.json trouvé")

        self.ray_mu    = np.array(stats["mean"], dtype=np.float32)
        self.ray_sigma = np.array(stats["std"],  dtype=np.float32)

        # ── Depth bridge ──────────────────────────────────────────────────
        self.depth_bridge = DepthToRays()

        # ── VESC ──────────────────────────────────────────────────────────
        self.vesc = VESCInterface(port=vesc_port, duty_max=duty_max)

        # ── Smoother (identique simulation) ──────────────────────────────
        self.smoother = SmoothingFilter()

        # ── État partagé entre threads ────────────────────────────────────
        self._lock         = threading.Lock()
        self._latest_rays: Optional[np.ndarray] = None
        self._last_frame_t = time.time()
        self._running      = False

        # ── Métriques ─────────────────────────────────────────────────────
        self._step       = 0
        self._fps_times  = deque(maxlen=50)
        self._start_time = None

        print(f"[RealCar] DUTY_MAX = {duty_max*100:.0f}% | WATCHDOG = {WATCHDOG_S}s")
        print("[RealCar] ✅ Prêt. Lance run() pour démarrer.\n")

    # ── Thread perception ─────────────────────────────────────────────────
    def _perception_thread(self):
        """Lit les frames depth OAK-D et met à jour _latest_rays."""
        if not _DAI_AVAILABLE:
            print("[Perception] depthai non disponible — thread perception inactif")
            return

        pipeline = create_depthai_pipeline()
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("depth", maxSize=1, blocking=False)
            print("[Perception] OAK-D connectée — flux depth démarré")
            while self._running:
                msg = q.tryGet()
                if msg is not None:
                    depth_frame = msg.getFrame()   # uint16, mm
                    rays = self.depth_bridge(depth_frame)
                    with self._lock:
                        self._latest_rays  = rays
                        self._last_frame_t = time.time()
                time.sleep(0.005)

    # ── Thread contrôle ───────────────────────────────────────────────────
    def _control_thread(self):
        """Lit les raycasts, fait l'inférence, envoie les commandes VESC."""
        print(f"[Control] Boucle démarrée @ {CONTROL_HZ} Hz")
        interval = 1.0 / CONTROL_HZ

        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                rays       = self._latest_rays
                last_frame = self._last_frame_t

            # ── Watchdog ──────────────────────────────────────────────────
            if time.time() - last_frame > WATCHDOG_S:
                self.vesc.stop()
                time.sleep(0.1)
                continue

            if rays is None:
                time.sleep(0.01)
                continue

            # ── Z-score ───────────────────────────────────────────────────
            mu    = self.ray_mu[:20]
            sigma = self.ray_sigma[:20]
            rays_z = (rays - mu) / (sigma + 1e-8)

            # ── Features dérivées (identiques à inference.py) ─────────────
            n    = len(rays_z)
            half = n // 2
            asymmetry = (rays_z[half:].sum() - rays_z[:half].sum()) / (rays_z.sum() + 1e-8)
            front_ray = float(rays_z[half - 1: half + 1].mean())
            min_ray   = float(rays_z.min())
            derived   = np.array([asymmetry, front_ray, min_ray], dtype=np.float32)

            features = np.concatenate([rays_z, derived]).astype(np.float32)  # (23,)

            # ── Inférence ONNX ────────────────────────────────────────────
            pred = self.sess.run(None, {self.input_name: features[np.newaxis, :]})[0][0]

            # bimodal_accel=True → pred[1] est un logit → appliquer sigmoid
            steer_raw = float(pred[0])
            accel_raw = float(1.0 / (1.0 + np.exp(-pred[1])))  # sigmoid

            # ── Smoothing ─────────────────────────────────────────────────
            pred_smooth = self.smoother.update(np.array([steer_raw, accel_raw], dtype=np.float32))

            # ── Offset biais droite (identique simulation) ────────────────
            steer = pred_smooth[0]
            if 0.05 < abs(steer) < 0.35:
                steer = steer - 0.02 * np.sign(steer)
            steering = float(np.clip(steer, -1.0, 1.0))

            # ── Heuristique accel avec front_raw (identique simulation) ───
            # Utiliser les rays BRUTS (non Z-scorés) pour l'anticipation
            front_raw = float(rays[half - 1: half + 1].mean())
            geo_base  = max(0.35, 1.0 - 1.2 * abs(steering))
            if front_raw >= 0.65:
                front_cap = 1.0
            else:
                front_cap = 0.45 + 0.70 * front_raw
            acceleration = float(np.clip(min(geo_base, front_cap), 0.35, 0.95))

            # ── Envoi VESC ────────────────────────────────────────────────
            self.vesc.send(steering, acceleration)

            # ── Métriques ─────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            self._fps_times.append(elapsed)
            self._step += 1

            if self._step % 50 == 0:
                fps = 1.0 / (np.mean(self._fps_times) + 1e-9)
                side = "L" if steering < -0.05 else ("R" if steering > 0.05 else "=")
                bar = "#" * int(abs(steering) * 12)
                print(
                    f"\r  [{side}] {bar:<12} "
                    f"steer={steering:+.3f} accel={acceleration:.3f} "
                    f"front={front_raw:.2f} {fps:.0f}fps    ",
                    end="", flush=True,
                )

            # ── Cadence ───────────────────────────────────────────────────
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Point d'entrée ────────────────────────────────────────────────────
    def run(self):
        """Démarre les threads et bloque jusqu'à Ctrl+C."""
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
            elapsed = time.time() - self._start_time
            print(f"[RealCar] Terminé — {self._step} steps | {elapsed:.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inférence voiture réelle v18")
    parser.add_argument("--model",    default=MODEL_PATH,   help="Chemin modèle .onnx")
    parser.add_argument("--stats",    default=STATS_PATH,   help="Chemin ray_stats.json réel")
    parser.add_argument("--port",     default=VESC_PORT,    help="Port USB FSESC (ex: /dev/ttyACM0)")
    parser.add_argument("--duty-max", type=float, default=DUTY_MAX, help="Puissance max moteur (0.0-1.0)")
    args = parser.parse_args()

    RealCarInference(
        model_path=args.model,
        stats_path=args.stats,
        vesc_port=args.port,
        duty_max=args.duty_max,
    ).run()


if __name__ == "__main__":
    main()

"""
inference.py — Inférence temps réel avec SmoothingFilter robuste (v2).

Fixes Grok:
- SmoothingFilter: gestion premier step (None init), reset épisode, NaN
- PerformanceTracker: FPS, avg_speed, jerk (smoothness)
- Support ONNX Runtime (Jetson Nano: TensorRT via onnxruntime-gpu)
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import RobocarEnv


class SmoothingFilter:
    """
    Filtre adaptatif : alpha élevé en virage (réactif), bas en ligne droite (stable).
    Deadzone sur le steering pour supprimer le micro-zigzag.
    Recommandé Grok/Gemini v2.
    """

    def __init__(self, alpha: float = 0.57, alpha_max: float = 0.92, deadzone: float = 0.06):
        self.alpha_base = alpha
        self.alpha_max = alpha_max
        self.deadzone = deadzone
        self._smoothed: Optional[np.ndarray] = None

    def reset(self):
        self._smoothed = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)

        if self._smoothed is None:
            self._smoothed = raw.copy()
        else:
            # Alpha adaptatif : plus réactif si changement de steering fort (virage)
            delta = abs(raw[0] - self._smoothed[0])
            alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * min(delta, 1.0)
            self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed

        result = self._smoothed.copy()
        # Deadzone steering : supprime le micro-zigzag en ligne droite
        if abs(result[0]) < self.deadzone:
            result[0] = 0.0
        return result

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.update(raw)


class JerkTracker:
    """
    Calcule le jerk (dérivée du steering) = métrique de smoothness.
    Un bon modèle BC a un jerk faible → conduite fluide.
    """

    def __init__(self):
        self._prev_steering: Optional[float] = None
        self._jerks: deque = deque(maxlen=200)

    def update(self, steering: float):
        if self._prev_steering is not None:
            self._jerks.append(abs(steering - self._prev_steering))
        self._prev_steering = steering

    @property
    def avg_jerk(self) -> float:
        return float(np.mean(self._jerks)) if self._jerks else 0.0

    def reset(self):
        self._prev_steering = None
        self._jerks.clear()


class PerformanceTracker:
    """Suivi des métriques de performance en temps réel (v2 + jerk)."""

    def __init__(self, window: int = 100):
        self.step_count = 0
        self.start_time = time.time()
        self._frame_times: deque = deque(maxlen=window)
        self._speeds: deque = deque(maxlen=window)
        self.jerk = JerkTracker()

    def record(self, frame_time: float, speed: float, steering: float):
        self._frame_times.append(frame_time)
        self._speeds.append(speed)
        self.jerk.update(steering)
        self.step_count += 1

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / (np.mean(self._frame_times) + 1e-9)

    @property
    def avg_speed(self) -> float:
        return float(np.mean(self._speeds)) if self._speeds else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        return (
            f"Steps={self.step_count} | "
            f"FPS={self.fps:.1f} | "
            f"AvgSpeed={self.avg_speed:.3f} | "
            f"AvgJerk={self.jerk.avg_jerk:.4f} (↓ = plus fluide) | "
            f"Elapsed={self.elapsed:.1f}s"
        )


def run_inference_pytorch(
    model_path: str,
    config_path: str = "config.json",
    port: int = 5005,
    smoothing_alpha: float = 0.7,
    max_steps: int = 0,
):
    """Inférence avec modèle PyTorch."""
    import torch
    import json
    from src.model import load_model

    print(f"\n[Inference] Chargement: {model_path}")
    model = load_model(model_path)
    model.eval()
    print(f"[Inference] {model}")
    print(f"[INFO] Lancer RacingSimulator.x86_64 avant ce script!\n")

    # Charger les stats Z-score si disponibles
    ray_mu = ray_sigma = None
    stats_path = Path(model_path).parent / "ray_stats.json"
    if not stats_path.exists():
        stats_path = Path("models/ray_stats.json")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        ray_mu = np.array(stats["mean"], dtype=np.float32)
        ray_sigma = np.array(stats["std"], dtype=np.float32)
        print(f"[Inference] Z-score chargé depuis {stats_path}")

    smoother = SmoothingFilter(alpha=smoothing_alpha)
    tracker = PerformanceTracker()
    prev_rays = None  # Pour delta_rays (v3)

    with RobocarEnv(config_path=config_path, port=port) as env:
        observations = env.reset()
        smoother.reset()
        prev_rays = None
        print(f"[Inference] Connecté — {len(observations)} agent(s). Démarrage...\n")

        try:
            while True:
                if max_steps > 0 and tracker.step_count >= max_steps:
                    break

                t0 = time.perf_counter()

                if not observations:
                    observations = env.step()
                    continue

                obs = observations[0]
                rays = obs.rays
                if ray_mu is not None:
                    rays = (rays - ray_mu) / ray_sigma

                # Delta rays : contexte temporel (v3)
                if hasattr(model, "use_delta") and model.use_delta:
                    delta_rays = np.zeros_like(rays) if prev_rays is None else (rays - prev_rays)
                    prev_rays = rays.copy()
                    x = torch.from_numpy(
                        np.concatenate([rays, delta_rays]).astype(np.float32)
                    ).unsqueeze(0)
                else:
                    x = torch.from_numpy(rays).unsqueeze(0)

                with torch.no_grad():
                    raw = model(x).squeeze(0)
                    # forward() retourne logits bruts pour accel → Sigmoid ici à l'inférence
                    if hasattr(model, "bimodal_accel") and model.bimodal_accel:
                        raw = torch.stack([raw[0], torch.sigmoid(raw[1])])
                    pred = raw.numpy()

                pred_smooth = smoother.update(pred)
                steering = float(np.clip(pred_smooth[0], -0.7, 0.7))
                if hasattr(model, "bimodal_accel") and model.bimodal_accel:
                    acceleration = float(np.clip(pred_smooth[1], 0.3, 1.0))
                else:
                    acceleration = 0.5  # fixe — ancien comportement

                env.send_actions(steering=steering, acceleration=acceleration)
                observations = env.step()

                elapsed = time.perf_counter() - t0
                tracker.record(elapsed, obs.speed, steering)

                if tracker.step_count % 50 == 0:
                    side = "L" if steering < -0.05 else ("R" if steering > 0.05 else "=")
                    bar = "#" * int(abs(steering) * 12)
                    print(
                        f"\r  [{side}] {bar:<12} "
                        f"steer={steering:+.3f} accel={acceleration:+.3f} "
                        f"spd={obs.speed:.2f} {tracker.fps:.0f}fps "
                        f"jerk={tracker.jerk.avg_jerk:.4f}    ",
                        end="", flush=True
                    )

        except KeyboardInterrupt:
            print("\n[Inference] Arrêté.")

    print(f"\n\n[Inference] {tracker.summary()}")


def run_inference_onnx(
    model_path: str,
    config_path: str = "config.json",
    port: int = 5005,
    smoothing_alpha: float = 0.7,
):
    """
    Inférence ONNX Runtime — pour Jetson Nano.
    Utiliser CUDAExecutionProvider si disponible (onnxruntime-gpu).
    Pour TensorRT: convertir le .onnx avec trtexec --fp16 sur le Jetson.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] pip install onnxruntime")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    active_provider = sess.get_providers()[0]
    print(f"[ONNX] Provider actif: {active_provider}")

    smoother = SmoothingFilter(alpha=smoothing_alpha)
    tracker = PerformanceTracker()

    with RobocarEnv(config_path=config_path, port=port) as env:
        observations = env.reset()
        smoother.reset()

        try:
            while True:
                t0 = time.perf_counter()
                if not observations:
                    observations = env.step()
                    continue

                obs = observations[0]
                x = obs.as_vector.reshape(1, -1)
                pred = sess.run([output_name], {input_name: x})[0][0]
                pred_smooth = smoother.update(pred)
                steering = float(np.clip(pred_smooth[0], -1.0, 1.0))
                acceleration = 0.5  # fixe — accel bimodale non apprise

                env.send_actions(steering=steering, acceleration=acceleration)
                observations = env.step()

                elapsed = time.perf_counter() - t0
                tracker.record(elapsed, obs.speed, steering)

                if tracker.step_count % 50 == 0:
                    print(
                        f"\r  steer={steering:+.3f} accel={acceleration:+.3f} "
                        f"spd={obs.speed:.2f} {tracker.fps:.0f}fps    ",
                        end="", flush=True
                    )

        except KeyboardInterrupt:
            pass

    print(f"\n[ONNX] {tracker.summary()}")


def main():
    parser = argparse.ArgumentParser(description="Inférence Robocar v2")
    parser.add_argument("--model", required=True, help=".pth ou .onnx")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()

    use_onnx = args.onnx or args.model.endswith(".onnx")
    if use_onnx:
        run_inference_onnx(args.model, args.config, args.port, args.smoothing)
    else:
        run_inference_pytorch(args.model, args.config, args.port, args.smoothing, args.max_steps)


if __name__ == "__main__":
    main()

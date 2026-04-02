"""
calibrate_ray_stats.py — Recalibrer le Z-score sur des données depth réelles.

⚠️  Les ray_stats.json de simulation NE FONCTIONNERONT PAS sur la depth map réelle.
    La distribution est trop différente (raycasts Unity parfaits vs depth stéréo bruitée).

Usage :
    1. Poser la voiture sur la piste réelle (ou tenir la caméra à la bonne hauteur)
    2. La pousser/déplacer manuellement sur 2-5 minutes en couvrant tout le circuit
    3. Lancer ce script : python3 src/calibrate_ray_stats.py
    4. Le fichier models/real_ray_stats.json est généré automatiquement
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import depthai as dai
    _DAI_AVAILABLE = True
except ImportError:
    _DAI_AVAILABLE = False

from src.depth_to_rays import DepthToRays, create_depthai_pipeline


OUTPUT_PATH = "models/real_ray_stats.json"
MIN_FRAMES  = 200    # minimum pour des stats fiables
TARGET_SEC  = 120    # durée cible : 2 minutes


def calibrate():
    if not _DAI_AVAILABLE:
        print("[ERROR] depthai non installé. Sur Jetson: pip install depthai")
        sys.exit(1)

    bridge = DepthToRays()
    all_rays = []

    print("\n[Calibration] ══════════════════════════════════════════")
    print("[Calibration]  Recalibrage Z-score — Données RÉELLES   ")
    print("[Calibration] ══════════════════════════════════════════")
    print(f"[Calibration] Objectif : {TARGET_SEC}s de données réelles ({MIN_FRAMES} frames min)")
    print("[Calibration] Déplace la voiture manuellement sur la piste.")
    print("[Calibration] Ctrl+C pour terminer et sauvegarder.\n")

    pipeline = create_depthai_pipeline()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("depth", maxSize=1, blocking=True)
        t_start = time.time()

        try:
            while True:
                msg = q.get()
                depth_frame = msg.getFrame()
                rays = bridge(depth_frame)
                all_rays.append(rays)

                n = len(all_rays)
                elapsed = time.time() - t_start

                if n % 30 == 0:
                    print(f"\r  {n} frames | {elapsed:.0f}s | "
                          f"ray_min={rays.min():.3f} ray_mean={rays.mean():.3f} ray_max={rays.max():.3f}   ",
                          end="", flush=True)

                if elapsed >= TARGET_SEC:
                    print(f"\n[Calibration] {TARGET_SEC}s atteintes — arrêt auto.")
                    break

        except KeyboardInterrupt:
            print(f"\n[Calibration] Arrêt manuel — {len(all_rays)} frames collectées.")

    n_frames = len(all_rays)
    if n_frames < MIN_FRAMES:
        print(f"[Calibration] ⚠️  Seulement {n_frames} frames ({MIN_FRAMES} min recommandées).")
        print("[Calibration] Stats calculées quand même — relancer pour plus de précision.")

    rays_arr = np.stack(all_rays)   # (N, 20)
    mean = rays_arr.mean(axis=0).tolist()
    std  = np.maximum(rays_arr.std(axis=0), 1e-6).tolist()

    # Les 3 derived features (asymmetry, front_ray, min_ray) ont leur propre distribution
    # On peut les laisser non normalisées (mean=0, std=1) — elles sont déjà dans [-1,1]
    # Alternativement : calculer leurs stats aussi depuis les rays réels
    mean_full = mean + [0.0, 0.0, 0.0]
    std_full  = std  + [1.0, 1.0, 1.0]

    stats = {"mean": mean_full, "std": std_full, "n_frames": n_frames}

    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[Calibration] ✅ Sauvegardé → {output}")
    print(f"[Calibration] {n_frames} frames | mean[0..4] = {[f'{v:.3f}' for v in mean[:5]]}")
    print(f"[Calibration] Relancer inference_realcar.py — il utilisera ces stats automatiquement.")


if __name__ == "__main__":
    calibrate()

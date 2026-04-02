"""
calibrate_ray_stats.py — Recalibrer le Z-score sur données depth réelles.

Stratégie 3 phases recommandée (Grok/Gemini) :
  Phase 1 (~1 min) : Lignes droites  → baseline
  Phase 2 (~1 min) : Virages doux   → asymmetry
  Phase 3 (~1 min) : Virages serrés → min/max rays

⚠️  Les ray_stats.json de simulation NE FONCTIONNERONT PAS sur la depth map réelle.

Usage :
    python3.8 src/calibrate_ray_stats.py
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
MIN_FRAMES  = 200
PHASE_SEC   = 60   # 1 min par phase

PHASES = [
    ("1/3 LIGNES DROITES",  "Pousse la voiture en ligne droite, couvre toute la piste"),
    ("2/3 VIRAGES DOUX",    "Slalom large, expose les asymétries gauche/droite"),
    ("3/3 VIRAGES SERRÉS",  "Virages courts/angles extrêmes, couvre min_ray"),
]


def validate_stats(rays_arr: np.ndarray):
    """Vérifie que les stats sont exploitables."""
    stds = rays_arr.std(axis=0)
    dead = np.sum(stds < 0.01)
    if dead > 0:
        print(f"  ⚠️  {dead} rays avec std≈0 (ray mort) → forcé à 0.10")
    z_check = (rays_arr - rays_arr.mean(axis=0)) / (stds + 1e-8)
    spikes = np.sum(np.abs(z_check) > 5)
    if spikes > 0:
        print(f"  ⚠️  {spikes} valeurs Z>5 détectées (bruit depth) → normal en sim-to-real")
    print(f"  ✅ mean[0..4]  = {[f'{v:.3f}' for v in rays_arr.mean(axis=0)[:5]]}")
    print(f"  ✅ std[0..4]   = {[f'{v:.3f}' for v in stds[:5]]}")


def calibrate():
    if not _DAI_AVAILABLE:
        print("[ERROR] depthai non installé.")
        sys.exit(1)

    bridge = DepthToRays()
    all_rays = []

    print("\n[Calibration] ══════════════════════════════════════════")
    print("[Calibration]  Z-score Réel — Stratégie 3 Phases       ")
    print("[Calibration] ══════════════════════════════════════════\n")

    pipeline = create_depthai_pipeline()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("depth", maxSize=1, blocking=True)

        for phase_label, phase_desc in PHASES:
            print(f"\n  ━━━ Phase {phase_label} ━━━")
            print(f"  {phase_desc}")
            input("  → Appuie sur Entrée quand tu es prêt...")

            t_phase = time.time()
            n_phase = 0
            try:
                while time.time() - t_phase < PHASE_SEC:
                    msg   = q.get()
                    rays  = bridge(msg.getFrame())
                    # clip avant stockage (recommandation Gemini)
                    rays  = np.clip(rays, 0.0, 1.0)
                    all_rays.append(rays)
                    n_phase += 1
                    elapsed = time.time() - t_phase
                    if n_phase % 30 == 0:
                        print(f"\r    {elapsed:.0f}s/{PHASE_SEC}s | {n_phase} frames "
                              f"| mean={rays.mean():.3f} min={rays.min():.3f}   ",
                              end="", flush=True)
            except KeyboardInterrupt:
                print(f"\n  Arrêt manuel phase ({n_phase} frames)")

            print(f"\n  ✅ Phase terminée — {n_phase} frames")

    # ── Calcul des stats ────────────────────────────────────────────────────
    n_frames = len(all_rays)
    if n_frames < MIN_FRAMES:
        print(f"\n⚠️  Seulement {n_frames} frames ({MIN_FRAMES} min recommandées) — stats quand même.")

    rays_arr = np.stack(all_rays)   # (N, 20)
    mean_r   = rays_arr.mean(axis=0)
    std_r    = rays_arr.std(axis=0)
    # Clamp std: si ray mort (std<0.01) → 0.10 pour éviter div/0 explosif
    std_r    = np.where(std_r < 0.01, 0.10, std_r)

    print("\n[Calibration] ── Validation des stats ──")
    validate_stats(rays_arr)

    # Derived features : asymmetry, front_ray, min_ray → non normalisées ([-1,1])
    mean_full = mean_r.tolist() + [0.0, 0.0, 0.0]
    std_full  = std_r.tolist()  + [1.0, 1.0, 1.0]

    stats = {"mean": mean_full, "std": std_full, "n_frames": int(n_frames)}

    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[Calibration] ✅ {n_frames} frames sauvegardées → {output}")
    print("[Calibration] Relancer inference_realcar.py — ces stats seront chargées automatiquement.")


if __name__ == "__main__":
    calibrate()

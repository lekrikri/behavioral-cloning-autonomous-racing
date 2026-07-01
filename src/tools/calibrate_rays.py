"""
calibrate_ray_stats.py — Recalibrer le Z-score sur données depth réelles.

Stratégie 3 phases recommandée (Grok/Gemini) :
  Phase 1 (~1 min) : Lignes droites  → baseline
  Phase 2 (~1 min) : Virages doux   → asymmetry
  Phase 3 (~1 min) : Virages serrés → min/max rays

⚠️  Les ray_stats.json de simulation NE FONCTIONNERONT PAS sur la depth map réelle.

Usage :
    python3.8 -m src.tools.calibrate_rays
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from src.mask.white_line import white_line_mask
from src.mask.perception_config import resolve_profile

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
    from src.cam.hub import FrameClient, SHM_COLOR, SHM_DEPTH, ensure_hub_or_prompt
    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    # Recalibre sur LA MÊME perception que l'inférence : le PROFIL ACTIF (celui de la prod),
    # pas un profil codé en dur — sinon les stats ne collent pas au masque utilisé pour rouler.
    profile, prof_name = resolve_profile(None)
    print(f"[Calibration] profil = {prof_name}")
    mask_kw = profile.mask_kwargs()
    geom = [None]        # construit à la première frame (taille réelle)
    polar = [None]

    def _rays(bgr, depth):
        if geom[0] is None or geom[0].H != bgr.shape[0] or geom[0].W != bgr.shape[1]:
            from src.mask.camera_ground import CameraGround
            geom[0] = CameraGround(bgr.shape[1], bgr.shape[0], profile.fov_deg,
                                   profile.cam_height_m, profile.cam_pitch_deg)
            polar[0] = profile.polar_rays(geom=geom[0])
        mask = white_line_mask(bgr, depth_mm=depth, geom=geom[0], **mask_kw)
        return polar[0].normalized(polar[0](mask)[0])

    all_rays = []

    print("\n[Calibration] ══════════════════════════════════════════")
    print("[Calibration]  Z-score Réel (POLAIRE) — Stratégie 3 Phases")
    print("[Calibration] ══════════════════════════════════════════\n")

    c_color = FrameClient(stream=SHM_COLOR)   # couleur cadence ; depth best-effort
    c_depth = FrameClient(stream=SHM_DEPTH)

    for phase_label, phase_desc in PHASES:
        print(f"\n  ━━━ Phase {phase_label} ━━━")
        print(f"  {phase_desc}")
        input("  → Appuie sur Entrée quand tu es prêt...")

        t_phase = time.time()
        n_phase = 0
        try:
            while time.time() - t_phase < PHASE_SEC:
                bgr = c_color.getCvFrame()
                try:
                    depth = c_depth.latest()
                except (ConnectionError, OSError):
                    depth = None
                rays  = _rays(bgr, depth)
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

    c_color.close(); c_depth.close()

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

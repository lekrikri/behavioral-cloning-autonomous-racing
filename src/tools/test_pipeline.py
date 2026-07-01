"""
test_pipeline.py — Validation du pipeline complet AVANT déploiement Jetson.

Lance ce script sur le PC de développement pour vérifier que tout fonctionne
SANS OAK-D ni VESC connectés.

Tests effectués :
  1. masque + faisceau polaire — scènes couleur synthétiques (lignes blanches)
  2. ONNX inference — modèle v18 tourne sur des rayons polaires
  3. SmoothingFilter — lissage adaptatif
  4. Heuristique accel — front_raw cap
  5. VESCInterface  — mode simulation (sans hardware)
  6. Pipeline complet — boucle entière sur 100 frames synthétiques

Usage : python3 -m src.tools.test_pipeline
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

import onnxruntime as ort
from src.mask.white_line import white_line_mask
from src.mask.perception_config import PerceptionProfile
from src.control.vesc_interface import VESCInterface
from src.features import derive_features
from src.control_post import SmoothingFilter, accel_from_geometry

# Charger Z-score
_STATS_PATH = Path("models/real_ray_stats.json")
if not _STATS_PATH.exists():
    _STATS_PATH = Path("models/ray_stats.json")

with open(_STATS_PATH) as f:
    _STATS = json.load(f)
_MU    = np.array(_STATS["mean"], dtype=np.float32)
_SIGMA = np.array(_STATS["std"],  dtype=np.float32)

# Perception polaire (même chaîne que l'inférence) — profil classic, taille synthétique.
_PROFILE = PerceptionProfile()
_GEOM    = _PROFILE.camera_ground()
_POLAR   = _PROFILE.polar_rays(geom=_GEOM)
_MASK_KW = _PROFILE.mask_kwargs()

# ─── Helpers ──────────────────────────────────────────────────────────────

def make_scene(line_left=False, line_right=False, line_center=False,
               W=512, H=256) -> np.ndarray:
    """Scène couleur synthétique : lignes blanches (bords de piste) sur fond sombre."""
    bgr = np.zeros((H, W, 3), np.uint8)

    def vline(xf):
        x = int(W * xf)
        cv2.line(bgr, (x, int(H * 0.55)), (x, H - 1), (255, 255, 255), 8)

    if line_left:   vline(0.20)
    if line_right:  vline(0.80)
    if line_center: vline(0.50)
    return bgr


def rays_from(bgr, depth=None) -> np.ndarray:
    """Frame couleur -> rayons polaires normalisés [0,1] (1 = libre)."""
    mask = white_line_mask(bgr, depth_mm=depth, geom=_GEOM, **_MASK_KW)
    return _POLAR.normalized(_POLAR(mask)[0])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(rays: np.ndarray) -> np.ndarray:
    """Z-score + derived features via le contrat partagé (cf. src/features.py)."""
    rays_z = (rays - _MU[:20]) / (_SIGMA[:20] + 1e-8)
    derived = derive_features(rays_z)
    return np.concatenate([rays_z, derived]).astype(np.float32)


def accel_heuristic(steering: float, front_raw: float) -> float:
    return accel_from_geometry(steering, front_raw)


# ─── Tests ────────────────────────────────────────────────────────────────

def ok(msg): print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}"); sys.exit(1)


def test_polar_rays():
    print("\n[1] masque + faisceau polaire — scénarios lignes")

    # Aucune ligne → tous les rayons libres (max range)
    rays = rays_from(make_scene())
    assert rays.shape == (20,)
    assert rays.min() > 0.9, f"scène vide: rayons non libres (min={rays.min():.3f})"
    ok(f"scène vide → rayons libres (min={rays.min():.3f}) ✓")

    # Ligne à gauche → le rayon le plus proche est dans la moitié gauche
    rays = rays_from(make_scene(line_left=True))
    assert int(np.argmin(rays)) < 10, f"ligne gauche: min au ray {int(np.argmin(rays))}"
    ok(f"ligne gauche : rayon min = #{int(np.argmin(rays))} (gauche) ✓")

    # Ligne à droite → rayon le plus proche dans la moitié droite
    rays = rays_from(make_scene(line_right=True))
    assert int(np.argmin(rays)) >= 10, f"ligne droite: min au ray {int(np.argmin(rays))}"
    ok(f"ligne droite : rayon min = #{int(np.argmin(rays))} (droite) ✓")

    # Ligne centrale → rayon le plus proche au centre
    rays = rays_from(make_scene(line_center=True))
    assert 6 <= int(np.argmin(rays)) <= 13, f"ligne centrale: min au ray {int(np.argmin(rays))}"
    ok(f"ligne centrale : rayon min = #{int(np.argmin(rays))} (centre) ✓")


def test_onnx():
    print("\n[2] ONNX v18 — tourne sur rayons polaires")
    sess = ort.InferenceSession("models/v18/best.onnx", providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name

    scenarios = [
        ("deux_lignes",     make_scene(line_left=True, line_right=True)),
        ("ligne_gauche",    make_scene(line_left=True)),
        ("ligne_droite",    make_scene(line_right=True)),
        ("ligne_centre",    make_scene(line_center=True)),
    ]

    for name, bgr in scenarios:
        rays  = rays_from(bgr)
        feat  = preprocess(rays)
        pred  = sess.run(None, {inp: feat[np.newaxis, :]})[0][0]
        steer = float(pred[0])
        accel = sigmoid(float(pred[1]))
        ok(f"{name:16s} steer={steer:+.3f}  accel={accel:.3f}")

    feat = preprocess(rays_from(make_scene()))
    assert feat.shape == (23,), f"feature shape: {feat.shape} ≠ (23,)"
    ok("input shape (23,) ✓")


def test_smoother():
    print("\n[3] SmoothingFilter — lissage adaptatif")
    sm = SmoothingFilter()

    out = sm.update(np.array([0.5, 0.6]))
    assert abs(out[0] - 0.5) < 1e-4
    ok("premier step = valeur brute ✓")

    sm.reset()
    out = sm.update(np.array([0.04, 0.5]))
    assert out[0] == 0.0, f"deadzone: {out[0]} ≠ 0"
    ok("deadzone 0.06 → steering forcé à 0 ✓")

    sm.reset()
    sm.update(np.array([0.0, 0.5]))
    out = sm.update(np.array([0.8, 0.5]))
    assert out[0] > 0.60, f"virage réactif: alpha trop bas ({out[0]:.3f})"
    ok(f"virage réactif : alpha élevé → {out[0]:.3f} ✓")


def test_accel_heuristic():
    print("\n[4] Heuristique accélération — front_raw")

    a = accel_heuristic(steering=0.0, front_raw=0.90)
    assert a >= 0.90, f"ligne droite: accel={a:.3f} attendu ≥ 0.90"
    ok(f"ligne droite (front=0.90) → accel={a:.3f} ✓")

    a = accel_heuristic(steering=0.7, front_raw=0.30)
    assert a < 0.65, f"virage serré: accel={a:.3f} attendu < 0.65"
    ok(f"virage serré  (steer=0.7, front=0.30) → accel={a:.3f} ✓")

    a = accel_heuristic(steering=0.1, front_raw=0.66)
    assert a >= 0.85, f"threshold 0.65: accel={a:.3f} attendu ≥ 0.85"
    ok(f"front=0.66 (≥0.65) → accel={a:.3f} (pleine vitesse) ✓")


def test_vesc_sim():
    print("\n[5] VESCInterface — mode simulation (sans hardware)")
    vesc = VESCInterface(port="/dev/ttyACM_FAKE")  # port inexistant → sim mode

    assert vesc._sim_mode, "doit être en mode simulation"
    ok("mode simulation activé ✓")

    vesc.send(steering=0.5, accel=0.8)
    vesc.send(steering=-1.0, accel=0.0)
    vesc.send(steering=0.0, accel=1.0)
    ok("commandes envoyées sans erreur ✓")

    vesc.stop()
    ok("stop() sans erreur ✓")


def test_full_pipeline():
    print("\n[6] Pipeline complet — 100 frames synthétiques")
    sess = ort.InferenceSession("models/v18/best.onnx", providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name
    vesc = VESCInterface(port="/dev/ttyACM_FAKE")
    smoother = SmoothingFilter()

    t0 = time.perf_counter()
    for i in range(100):
        bgr = make_scene(line_left=(i % 2 == 0), line_right=(i % 3 == 0),
                         line_center=(i % 5 == 0))
        rays   = rays_from(bgr)
        feat   = preprocess(rays)
        pred   = sess.run(None, {inp: feat[np.newaxis, :]})[0][0]
        steer  = float(pred[0])
        accel  = sigmoid(float(pred[1]))
        smooth = smoother.update(np.array([steer, accel]))
        accel_h = accel_heuristic(smooth[0], float(rays[9:11].mean()))
        vesc.send(smooth[0], accel_h)

    elapsed = time.perf_counter() - t0
    fps = 100 / elapsed
    ok(f"100 frames en {elapsed:.2f}s → {fps:.0f} FPS ✓")
    assert fps > 10, f"FPS trop bas sur PC: {fps:.0f}"
    vesc.close()


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  Test pipeline voiture réelle — v18 (perception polaire)")
    print("═" * 55)

    stats_used = "RÉEL" if Path("models/real_ray_stats.json").exists() else "SIMULATION"
    print(f"  Z-score : {stats_used} ({_STATS_PATH})")

    test_polar_rays()
    test_onnx()
    test_smoother()
    test_accel_heuristic()
    test_vesc_sim()
    test_full_pipeline()

    print("\n" + "═" * 55)
    print("  ✅ TOUS LES TESTS PASSÉS — Pipeline prêt pour Jetson")
    print("═" * 55)

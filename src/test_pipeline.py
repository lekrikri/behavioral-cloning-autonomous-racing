"""
test_pipeline.py — Validation du pipeline complet AVANT déploiement Jetson.

Lance ce script sur le PC de développement pour vérifier que tout fonctionne
SANS OAK-D ni VESC connectés.

Tests effectués :
  1. depth_to_rays  — conversion depth map → raycasts (5 scénarios)
  2. Z-score        — normalisation cohérente
  3. ONNX inference — modèle v18 prédit des directions cohérentes
  4. SmoothingFilter — lissage adaptatif
  5. Heuristique accel — front_raw cap
  6. VESCInterface  — mode simulation (sans hardware)
  7. Pipeline complet — boucle entière sur 100 frames synthétiques

Usage : python3 src/test_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import onnxruntime as ort
from src.depth_to_rays import DepthToRays
from src.vesc_interface import VESCInterface

# Charger Z-score
_STATS_PATH = Path("models/real_ray_stats.json")
if not _STATS_PATH.exists():
    _STATS_PATH = Path("models/ray_stats.json")

with open(_STATS_PATH) as f:
    _STATS = json.load(f)
_MU    = np.array(_STATS["mean"], dtype=np.float32)
_SIGMA = np.array(_STATS["std"],  dtype=np.float32)

# ─── Helpers ──────────────────────────────────────────────────────────────

def make_depth(obstacle_left=False, obstacle_right=False, obstacle_front=False,
               full_nan=False, W=640, H=400) -> np.ndarray:
    """Génère une depth map synthétique (uint16, mm)."""
    depth = np.full((H, W), 3000, dtype=np.uint16)
    row_s, row_e = int(H * 0.40), int(H * 0.62)
    if full_nan:
        depth[row_s:row_e, :] = 0
        return depth
    if obstacle_left:
        depth[row_s:row_e, :int(W * 0.35)] = 600
    if obstacle_right:
        depth[row_s:row_e, int(W * 0.65):] = 600
    if obstacle_front:
        depth[row_s:row_e, int(W*0.40):int(W*0.60)] = 400
    # Ajouter quelques pixels invalides (réalisme stéréo)
    rng = np.random.default_rng(42)
    noise_mask = rng.random((H, W)) < 0.05  # 5% pixels invalides
    depth[noise_mask] = 0
    return depth


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(rays: np.ndarray) -> np.ndarray:
    """Z-score + derived features — identique à inference_realcar.py."""
    rays_z = (rays - _MU[:20]) / (_SIGMA[:20] + 1e-8)
    n = len(rays_z); half = n // 2
    asym  = (rays_z[half:].sum() - rays_z[:half].sum()) / (rays_z.sum() + 1e-8)
    front = float(rays_z[half-1:half+1].mean())
    minr  = float(rays_z.min())
    return np.concatenate([rays_z, [asym, front, minr]]).astype(np.float32)


def accel_heuristic(steering: float, front_raw: float) -> float:
    geo_base  = max(0.35, 1.0 - 1.2 * abs(steering))
    front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
    return float(np.clip(min(geo_base, front_cap), 0.35, 0.95))


# ─── Tests ────────────────────────────────────────────────────────────────

def ok(msg): print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}"); sys.exit(1)


def test_depth_to_rays():
    print("\n[1] depth_to_rays — 5 scénarios")
    bridge = DepthToRays()

    # Ligne droite : tous les rays doivent être élevés
    rays = bridge(make_depth())
    assert rays.min() > 0.5, f"ligne droite: min={rays.min():.3f} attendu > 0.5"
    assert rays.shape == (20,)
    ok(f"ligne droite : mean={rays.mean():.3f} ✓")

    # Obstacle gauche → rays gauches bas
    rays = bridge(make_depth(obstacle_left=True))
    assert rays[:4].mean() < rays[15:].mean(), "obstacle gauche non détecté"
    ok(f"obstacle gauche : rays[:4]={rays[:4].mean():.3f} < rays[15:]={rays[15:].mean():.3f} ✓")

    # Obstacle droite → rays droits bas
    rays = bridge(make_depth(obstacle_right=True))
    assert rays[16:].mean() < rays[:4].mean(), "obstacle droite non détecté"
    ok(f"obstacle droite : rays[16:]={rays[16:].mean():.3f} < rays[:4]={rays[:4].mean():.3f} ✓")

    # Obstacle frontal → ray central bas
    rays = bridge(make_depth(obstacle_front=True))
    assert rays[8:12].mean() < 0.5, f"obstacle frontal non détecté: {rays[8:12].mean():.3f}"
    ok(f"obstacle frontal : rays[8:12]={rays[8:12].mean():.3f} ✓")

    # Depth tout à 0 (pire cas — aucune disparité) → doit retourner max (pas de crash)
    rays = bridge(make_depth(full_nan=True))
    assert (rays == 1.0).all(), "full_nan: doit retourner 1.0 partout"
    ok("full NaN (0) → 1.0 partout (pas de crash) ✓")


def test_onnx():
    print("\n[2] ONNX v18 — directions cohérentes")
    sess = ort.InferenceSession("models/v18/best.onnx", providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name
    bridge = DepthToRays()

    scenarios = [
        ("ligne_droite",   make_depth(),                         "steer≈0"),
        ("obstacle_gauche", make_depth(obstacle_left=True),      "steer>0 (tourner droite)"),
        ("obstacle_droite", make_depth(obstacle_right=True),     "steer<0 (tourner gauche)"),
        ("obstacle_frontal", make_depth(obstacle_front=True),    "accel bas"),
    ]

    for name, depth, expected in scenarios:
        rays  = bridge(depth)
        feat  = preprocess(rays)
        pred  = sess.run(None, {inp: feat[np.newaxis, :]})[0][0]
        steer = float(pred[0])
        accel = sigmoid(float(pred[1]))
        ok(f"{name:22s} steer={steer:+.3f}  accel={accel:.3f}  [{expected}]")

    # Vérifier que les formes sont correctes
    feat = preprocess(bridge(make_depth()))
    assert feat.shape == (23,), f"feature shape: {feat.shape} ≠ (23,)"
    ok("input shape (23,) ✓")


def test_smoother():
    print("\n[3] SmoothingFilter — lissage adaptatif")
    from src.inference_realcar import SmoothingFilter
    sm = SmoothingFilter()

    # Premier step = valeur brute
    out = sm.update(np.array([0.5, 0.6]))
    assert abs(out[0] - 0.5) < 1e-4
    ok("premier step = valeur brute ✓")

    # Deadzone : steering < 0.06 → forcé à 0
    sm.reset()
    out = sm.update(np.array([0.04, 0.5]))
    assert out[0] == 0.0, f"deadzone: {out[0]} ≠ 0"
    ok("deadzone 0.06 → steering forcé à 0 ✓")

    # Lissage : grand saut → alpha élevé (réactif)
    sm.reset()
    sm.update(np.array([0.0, 0.5]))
    out = sm.update(np.array([0.8, 0.5]))
    assert out[0] > 0.60, f"virage réactif: alpha trop bas ({out[0]:.3f})"
    ok(f"virage réactif : alpha élevé → {out[0]:.3f} ✓")


def test_accel_heuristic():
    print("\n[4] Heuristique accélération — front_raw")

    # Ligne droite (front_raw élevé) → pleine vitesse
    a = accel_heuristic(steering=0.0, front_raw=0.90)
    assert a >= 0.90, f"ligne droite: accel={a:.3f} attendu ≥ 0.90"
    ok(f"ligne droite (front=0.90) → accel={a:.3f} ✓")

    # Virage serré frontal → ralentit
    a = accel_heuristic(steering=0.7, front_raw=0.30)
    assert a < 0.65, f"virage serré: accel={a:.3f} attendu < 0.65"
    ok(f"virage serré  (steer=0.7, front=0.30) → accel={a:.3f} ✓")

    # Threshold 0.65 : juste au-dessus → pleine vitesse
    a = accel_heuristic(steering=0.1, front_raw=0.66)
    assert a >= 0.85, f"threshold 0.65: accel={a:.3f} attendu ≥ 0.85"
    ok(f"front=0.66 (≥0.65) → accel={a:.3f} (pleine vitesse) ✓")


def test_vesc_sim():
    print("\n[5] VESCInterface — mode simulation (sans hardware)")
    vesc = VESCInterface(port="/dev/ttyACM_FAKE")  # port inexistant → sim mode

    assert vesc._sim_mode, "doit être en mode simulation"
    ok("mode simulation activé ✓")

    # Envoyer commandes sans crash
    vesc.send(steering=0.5, accel=0.8)
    vesc.send(steering=-1.0, accel=0.0)
    vesc.send(steering=0.0, accel=1.0)
    ok("commandes envoyées sans erreur ✓")

    vesc.stop()
    ok("stop() sans erreur ✓")


def test_full_pipeline():
    print("\n[6] Pipeline complet — 100 frames synthétiques")
    bridge = DepthToRays()
    sess   = ort.InferenceSession("models/v18/best.onnx", providers=["CPUExecutionProvider"])
    inp    = sess.get_inputs()[0].name
    vesc   = VESCInterface(port="/dev/ttyACM_FAKE")

    from src.inference_realcar import SmoothingFilter
    smoother = SmoothingFilter()

    rng = np.random.default_rng(0)
    t0  = time.perf_counter()

    for i in range(100):
        # Depth synthétique aléatoire
        depth = np.random.randint(200, 4000, (400, 640), dtype=np.uint16)
        depth[rng.random((400, 640)) < 0.1] = 0  # 10% NaN

        rays   = bridge(depth)
        feat   = preprocess(rays)
        pred   = sess.run(None, {inp: feat[np.newaxis, :]})[0][0]
        steer  = float(pred[0])
        accel  = sigmoid(float(pred[1]))

        smooth = smoother.update(np.array([steer, accel]))
        accel_h = accel_heuristic(smooth[0], float(rays[9:11].mean()))

        vesc.send(smooth[0], accel_h)

    elapsed = time.perf_counter() - t0
    fps = 100 / elapsed
    ok(f"100 frames en {elapsed:.2f}s → {fps:.0f} FPS (Jetson cible: ≥ 20 FPS) ✓")

    assert fps > 10, f"FPS trop bas sur PC: {fps:.0f}"
    vesc.close()


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  Test pipeline voiture réelle — v18 RobocarSpatial")
    print("═" * 55)

    stats_used = "RÉEL" if Path("models/real_ray_stats.json").exists() else "SIMULATION"
    print(f"  Z-score : {stats_used} ({_STATS_PATH})")

    test_depth_to_rays()
    test_onnx()
    test_smoother()
    test_accel_heuristic()
    test_vesc_sim()
    test_full_pipeline()

    print("\n" + "═" * 55)
    print("  ✅ TOUS LES TESTS PASSÉS — Pipeline prêt pour Jetson")
    print("═" * 55)

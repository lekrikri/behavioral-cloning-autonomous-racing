"""P2 — non-régression : les fonctions canoniques (features/control_post)
reproduisent EXACTEMENT les implémentations inline d'origine (sim, réel, dataset).

Lancer depuis la racine repo :  PYTHONPATH=. python3 tests/p2_contract_regression.py
"""

import numpy as np

from src.features import derive_features
from src.control_post import (
    SmoothingFilter, apply_steer_offset, accel_from_geometry,
)

rng = np.random.default_rng(0)
TOL = dict(rtol=1e-6, atol=1e-6)   # niveau float32


# ── Reconstructions des implémentations d'origine ───────────────────────────
def old_derive_inference_1d(rays):
    n = len(rays); half = n // 2
    left_sum = rays[:half].sum(); right_sum = rays[half:].sum()
    asymmetry = (right_sum - left_sum) / (left_sum + right_sum + 1e-8)
    front_ray = float(rays[half - 1:half + 1].mean())
    min_ray = rays.min()
    return np.array([asymmetry, front_ray, min_ray], dtype=np.float32)


def old_derive_realcar_1d(rays_z):
    half = len(rays_z) // 2
    asymmetry = (rays_z[half:].sum() - rays_z[:half].sum()) / (rays_z.sum() + 1e-8)
    front_ray = float(rays_z[half - 1:half + 1].mean())
    min_ray = float(rays_z.min())
    return np.array([asymmetry, front_ray, min_ray], dtype=np.float32)


def old_derive_dataset_2d(rays):
    n = rays.shape[1]; half = n // 2
    left_sum = rays[:, :half].sum(axis=1, keepdims=True)
    right_sum = rays[:, half:].sum(axis=1, keepdims=True)
    asymmetry = (right_sum - left_sum) / (left_sum + right_sum + 1e-8)
    front_ray = rays[:, n // 2 - 1:n // 2 + 1].mean(axis=1, keepdims=True)
    min_ray = rays.min(axis=1, keepdims=True)
    return np.concatenate([asymmetry, front_ray, min_ray], axis=1).astype(np.float32)


def old_steer_offset(s):
    if 0.05 < abs(s) < 0.35:
        s = s - 0.02 * np.sign(s)
    return s


def old_accel_sim(steering, front_raw):
    geo_base = max(0.35, 1.0 - 1.2 * abs(steering))
    front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
    return float(np.clip(min(geo_base, front_cap), 0.35, 0.95))


def old_accel_real(steering, front_raw):
    geo_base = max(0.35, 1.0 - 1.2 * abs(steering))
    front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
    corner_damp = 1.0 - 0.5 * abs(steering)
    return float(np.clip(min(geo_base, front_cap) * corner_damp, 0.50, 0.95))


# ── Tests ───────────────────────────────────────────────────────────────────
def test_derive_matches_training_contract():
    # Le modèle est entraîné sur dataset.py (denom = left+right). L'inférence DOIT
    # reproduire ce contrat ; inference.py (sim) utilise déjà la même formule.
    for _ in range(2000):
        rays = rng.standard_normal(20).astype(np.float32)
        assert np.allclose(derive_features(rays), old_derive_inference_1d(rays), **TOL)


def test_realcar_asymmetry_realignment():
    # inference_realcar utilisait denom=rays.sum() — divergent du contrat d'entraînement.
    # La canonicalisation l'aligne : identique en régime stable (|sum|>1), ne diffère que
    # dans le régime dégénéré sum≈0 où l'asymétrie est de toute façon instable.
    max_delta_stable = max_delta_degen = 0.0
    for _ in range(20000):
        rays = rng.standard_normal(20).astype(np.float32)
        d = float(np.abs(derive_features(rays) - old_derive_realcar_1d(rays)).max())
        if abs(float(rays.sum())) > 1.0:
            max_delta_stable = max(max_delta_stable, d)
        else:
            max_delta_degen = max(max_delta_degen, d)
    assert max_delta_stable < 1e-4, f"divergence en régime stable: {max_delta_stable}"
    print(f"    [info] realcar realign — max Δ stable={max_delta_stable:.2e} "
          f"degen(sum≈0)={max_delta_degen:.2e}")


def test_derive_features_2d():
    for _ in range(500):
        rays = rng.standard_normal((64, 20)).astype(np.float32)
        assert np.allclose(derive_features(rays), old_derive_dataset_2d(rays), **TOL)


def test_steer_offset():
    for s in np.linspace(-1.2, 1.2, 4001):
        assert np.allclose(apply_steer_offset(float(s)), old_steer_offset(float(s)), **TOL)


def test_accel():
    for steering in np.linspace(-1.2, 1.2, 121):
        for front_raw in np.linspace(0.0, 1.0, 101):
            s, f = float(steering), float(front_raw)
            assert np.allclose(accel_from_geometry(s, f), old_accel_sim(s, f), **TOL)
            assert np.allclose(
                accel_from_geometry(s, f, corner_damp=True, accel_floor=0.50),
                old_accel_real(s, f), **TOL)


def test_smoothing_filter_runs():
    sf = SmoothingFilter()
    out = None
    for _ in range(50):
        out = sf.update(rng.standard_normal(2).astype(np.float32))
    assert out is not None and out.shape == (2,)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'} ({len(tests)} tests)")
    raise SystemExit(1 if failed else 0)

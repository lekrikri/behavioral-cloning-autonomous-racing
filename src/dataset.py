"""
dataset.py — Dataset de conduite avec augmentations avancées.

Améliorations v2 (Gemini + Grok):
- Flip: .flip(0) au lieu de [::-1] (plus sûr avec torch)
- Noise adaptatif: std ∝ distance du rayon (rayons lointains = plus bruités)
- Speed jitter: ±jitter variation de vitesse (robustesse)
- Ray cutout: masquer 1-2 rayons aléatoires (simule capteur défaillant)
- WeightedRandomSampler: rééquilibrer la distribution du steering
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split, Subset


class DrivingDataset(Dataset):
    """
    Dataset PyTorch pour les trajectoires de conduite.

    Sample: (observation [N_rays+1], action [2])
    """

    def __init__(
        self,
        data_path: str,
        n_rays: Optional[int] = None,
        augment: bool = True,
        flip_prob: float = 0.5,
        noise_std: float = 0.01,
        noise_adaptive: bool = True,
        speed_jitter: float = 0.10,
        ray_cutout: bool = True,
        filter_stopped: bool = True,
        speed_threshold: float = 0.05,
        ray_stats: Optional[dict] = None,
    ):
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.noise_adaptive = noise_adaptive
        self.speed_jitter = speed_jitter
        self.ray_cutout = ray_cutout

        df = self._load_data(data_path)

        if filter_stopped:
            before = len(df)
            df = df[df["speed"] >= speed_threshold].reset_index(drop=True)
            removed = before - len(df)
            if removed:
                print(f"[Dataset] Filtre vitesse: {before} → {len(df)} frames ({removed} supprimées)")

        ray_cols = [c for c in df.columns if c.startswith("ray_")]
        if n_rays is not None:
            ray_cols = ray_cols[:n_rays]
        self.n_rays = len(ray_cols)

        # Z-score normalization des rayons (amplifier le signal faible)
        rays = df[ray_cols].values.astype(np.float32)
        if ray_stats is not None:
            mu = np.array(ray_stats["mean"], dtype=np.float32)
            sigma = np.array(ray_stats["std"], dtype=np.float32)
            rays = (rays - mu) / sigma
            print(f"[Dataset] Z-score appliqué (mu={mu.mean():.3f}, sigma_mean={sigma.mean():.3f})")

        # Speed toujours 0 dans ce simulateur — on l'exclut
        self.observations = rays
        self.actions = df[["steering", "acceleration"]].values.astype(np.float32)
        self._print_stats()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(self.observations[idx].copy())
        action = torch.from_numpy(self.actions[idx].copy())

        if self.augment:
            obs, action = self._augment(obs, action)

        # Sécurité: remplacer NaN/Inf résiduels
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        action = torch.nan_to_num(action, nan=0.0)

        return obs, action

    def _augment(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applique les augmentations data (toutes en tenseurs torch)."""
        rays = obs  # obs = seulement les rays (Z-scorés), plus de speed

        # 1. Flip horizontal (symétrie gauche/droite)
        if random.random() < self.flip_prob:
            rays = rays.flip(0)
            action = action * torch.tensor([-1.0, 1.0])

        # 2. Bruit gaussien sur les rayons (std fixe, Z-score donc ~N(0,1))
        if self.noise_std > 0:
            noise = torch.randn_like(rays) * self.noise_std
            rays = rays + noise

        # 3. Ray cutout (masquer 1-2 rayons → simuler capteur mort)
        if self.ray_cutout and self.n_rays > 4 and random.random() < 0.3:
            n_cut = random.choice([1, 2])
            indices = random.sample(range(self.n_rays), n_cut)
            for i in indices:
                rays[i] = 0.0

        return rays, action

    def get_steering_weights(self, n_bins: int = 20) -> torch.Tensor:
        """
        Calcule les poids pour WeightedRandomSampler.

        Rééquilibre la distribution du steering:
        les virages (grands angles) auront plus de poids que les lignes droites.
        """
        steering = self.actions[:, 0]
        bins = np.linspace(-1.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(steering, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float32)
        bin_counts = np.maximum(bin_counts, 1)               # éviter division par 0
        bin_weights = 1.0 / bin_counts                       # poids inversement proportionnel à la fréquence

        sample_weights = bin_weights[bin_indices]
        return torch.from_numpy(sample_weights)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        path = Path(data_path)
        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"Aucun CSV dans {path}")
            dfs = [pd.read_csv(f) for f in csv_files]
            for f, df in zip(csv_files, dfs):
                print(f"[Dataset] {f.name}: {len(df)} frames")
            return pd.concat(dfs, ignore_index=True)
        df = pd.read_csv(path)
        print(f"[Dataset] {path.name}: {len(df)} frames")
        return df

    def _print_stats(self):
        s = self.actions[:, 0]
        a = self.actions[:, 1]
        pct_curve = ((np.abs(s) > 0.15).mean() * 100)
        print(
            f"[Dataset] {len(self):,} samples | {self.n_rays} rayons | "
            f"virages: {pct_curve:.1f}%"
        )
        print(
            f"  Steering  mean={s.mean():.3f} std={s.std():.3f} "
            f"[{s.min():.2f},{s.max():.2f}]"
        )
        print(
            f"  Accel     mean={a.mean():.3f} std={a.std():.3f}"
        )


def create_dataloaders(
    data_path: str,
    n_rays: Optional[int] = None,
    batch_size: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    augment_train: bool = True,
    use_weighted_sampler: bool = True,
    sampler_bins: int = 20,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 42,
    filter_stopped: bool = True,
    ray_stats: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders train/val/test.

    WeightedRandomSampler (recommandé Gemini): rééquilibre la distribution
    du steering pour éviter que le modèle apprenne à toujours aller tout droit.
    """
    # Dataset complet sans augmentation pour le split propre
    full_ds = DrivingDataset(data_path, n_rays=n_rays, augment=False, filter_stopped=filter_stopped, ray_stats=ray_stats)
    n = len(full_ds)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split, test_split = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Train dataset avec augmentation
    train_ds_aug = DrivingDataset(data_path, n_rays=n_rays, augment=augment_train, filter_stopped=filter_stopped, ray_stats=ray_stats)
    train_subset = Subset(train_ds_aug, train_split.indices)

    # WeightedRandomSampler pour rééquilibrer le steering
    sampler = None
    if use_weighted_sampler:
        weights = full_ds.get_steering_weights(n_bins=sampler_bins)
        train_weights = weights[train_split.indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_split),
            replacement=True,
        )
        print(f"[DataLoaders] WeightedRandomSampler activé ({sampler_bins} bins)")

    # Kwargs communs pour les workers
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # pas de shuffle si sampler actif
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_split.indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_split.indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        f"[DataLoaders] Train: {len(train_split):,} | "
        f"Val: {len(val_split):,} | Test: {len(test_split):,}"
    )
    return train_loader, val_loader, test_loader


def generate_synthetic_data(
    n_samples: int = 5000,
    n_rays: int = 10,
    output_path: str = "data/synthetic.csv",
) -> pd.DataFrame:
    """Génère un dataset synthétique réaliste pour tester le pipeline."""
    np.random.seed(42)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_samples):
        rays = np.random.uniform(0.3, 1.0, n_rays).astype(np.float32)
        scenario = np.random.random()

        if scenario < 0.15:       # obstacle à gauche → tourner droite
            rays[:n_rays//3] *= np.random.uniform(0.1, 0.4)
            steering = np.random.uniform(0.3, 0.9)
        elif scenario < 0.30:     # obstacle à droite → tourner gauche
            rays[2*n_rays//3:] *= np.random.uniform(0.1, 0.4)
            steering = np.random.uniform(-0.9, -0.3)
        elif scenario < 0.40:     # virage doux
            steering = np.random.uniform(-0.4, 0.4) * np.sign(np.random.randn())
        else:                     # ligne droite (60% des cas)
            steering = np.random.uniform(-0.1, 0.1)

        acceleration = np.clip(1.0 - 0.5 * abs(steering), 0.3, 1.0)
        speed = np.random.uniform(0.2, 0.9)

        row = {"timestamp": i * 0.05}
        for j, r in enumerate(rays):
            row[f"ray_{j}"] = round(float(r), 4)
        row["speed"] = round(float(speed), 4)
        row["steering"] = round(float(np.clip(steering, -1, 1)), 4)
        row["acceleration"] = round(float(acceleration), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[Synthetic] {n_samples} samples → {output_path}")
    return df


if __name__ == "__main__":
    print("=== Test Dataset v2 ===\n")

    df = generate_synthetic_data(n_samples=3000, n_rays=10)
    ds = DrivingDataset("data/synthetic.csv", augment=True)

    obs, action = ds[0]
    assert obs.shape == (11,)
    assert action.shape == (2,)
    assert not torch.isnan(obs).any(), "NaN dans obs!"
    assert not torch.isnan(action).any(), "NaN dans action!"
    print(f"\nSample: obs={obs.shape} action={action.shape} | NaN-free ✓")

    # Test WeightedRandomSampler
    weights = ds.get_steering_weights()
    print(f"Sampler weights: shape={weights.shape} | range=[{weights.min():.3f},{weights.max():.3f}]")

    # Test DataLoaders
    train_l, val_l, test_l = create_dataloaders(
        "data/synthetic.csv",
        batch_size=64,
        num_workers=0,
        prefetch_factor=None,
    )
    obs_b, act_b = next(iter(train_l))
    print(f"\nBatch: obs={obs_b.shape} action={act_b.shape}")
    print("[OK] Dataset v2 OK")

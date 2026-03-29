"""
dataset.py — Dataset de conduite avec augmentations avancées.

Améliorations v2 (Gemini + Grok):
- Flip: .flip(0) au lieu de [::-1] (plus sûr avec torch)
- Noise adaptatif: std ∝ distance du rayon (rayons lointains = plus bruités)
- Speed jitter: ±jitter variation de vitesse (robustesse)
- Ray cutout: masquer 1-2 rayons aléatoires (simule capteur défaillant)
- WeightedRandomSampler: rééquilibrer la distribution du steering

Améliorations v3:
- use_delta: ajoute Δrays (ray_t - ray_{t-1}) → contexte temporel, réduit zigzag
- Flip adapté pour obs=[rays, Δrays] (flip les deux moitiés séparément)
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
        noise_std: float = 0.02,
        noise_adaptive: bool = True,
        speed_jitter: float = 0.10,
        ray_cutout: bool = True,
        filter_stopped: bool = True,
        speed_threshold: float = 0.05,
        ray_stats: Optional[dict] = None,
        use_delta: bool = True,
        use_action_cond: bool = True,
        action_noise_std: float = 0.30,
    ):
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.noise_adaptive = noise_adaptive
        self.speed_jitter = speed_jitter
        self.ray_cutout = ray_cutout
        self.use_delta = use_delta
        self.use_action_cond = use_action_cond

        df = self._load_data(data_path)

        if filter_stopped:
            before = len(df)
            df = df[df["speed"] >= speed_threshold].reset_index(drop=True)
            removed = before - len(df)
            if removed:
                print(f"[Dataset] Filtre vitesse: {before} → {len(df)} frames ({removed} supprimées)")

        # Trier par (episode_id, step) si disponible → delta temporel valide
        self._has_episode_info = "episode_id" in df.columns and "step" in df.columns
        if self._has_episode_info:
            df = df.sort_values(["episode_id", "step"]).reset_index(drop=True)

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

        # Features dérivées : résumé spatial des rays pour aider le steering
        # asymétrie L/R = signe du virage, front = obstacle devant, min = obstacle le plus proche
        n = rays.shape[1]
        half = n // 2
        left_sum = rays[:, :half].sum(axis=1, keepdims=True)
        right_sum = rays[:, half:].sum(axis=1, keepdims=True)
        asymmetry = (right_sum - left_sum) / (left_sum + right_sum + 1e-8)  # (N,1) [-1,1]
        front_ray = rays[:, n // 2 - 1 : n // 2 + 1].mean(axis=1, keepdims=True)  # (N,1)
        min_ray = rays.min(axis=1, keepdims=True)  # (N,1)
        derived = np.concatenate([asymmetry, front_ray, min_ray], axis=1).astype(np.float32)

        # Delta rays : différence temporelle frame t - frame t-1
        # Valide uniquement si episode_id+step présents (df déjà trié en amont)
        # Sinon : désactiver avec --no-delta (données shufflées → delta = bruit)
        if use_delta:
            if self._has_episode_info:
                episode_ids = df["episode_id"].values
                steps = df["step"].values
                delta_rays = np.zeros_like(rays)
                for i in range(1, len(rays)):
                    if episode_ids[i] == episode_ids[i - 1] and steps[i] == steps[i - 1] + 1:
                        delta_rays[i] = rays[i] - rays[i - 1]
                n_valid = int((delta_rays.any(axis=1)).sum())
                print(f"[Dataset] Delta épisodique: {n_valid:,}/{len(rays):,} frames valides")
            else:
                delta_rays = np.concatenate([
                    np.zeros((1, rays.shape[1]), dtype=np.float32),
                    np.diff(rays, axis=0).astype(np.float32),
                ], axis=0)
                print("[Dataset] WARN: delta_rays naïf (pas d'episode_id) — bruit possible!")
            self.observations = np.concatenate([rays, delta_rays, derived], axis=1)
        else:
            self.observations = np.concatenate([rays, derived], axis=1)

        self.n_derived = derived.shape[1]  # 3 features dérivées

        actions = df[["steering", "acceleration"]].values.astype(np.float32)
        self.actions = actions

        # Action-conditioning : noisy prev_steering (trick pour dataset shuffled)
        # Training : fake_prev = steer_expert + N(0, σ)  → modèle apprend sa dynamique
        # Inference : on passe le vrai prev_steering du step précédent
        if use_action_cond:
            noise = np.random.randn(len(actions), 1).astype(np.float32) * action_noise_std
            noisy_steer = (actions[:, 0:1] + noise).clip(-1.0, 1.0)
            self.observations = np.concatenate([self.observations, noisy_steer], axis=1)

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
        """Applique les augmentations data (toutes en tenseurs torch).

        Layout obs (n_rays=10, n_derived=3) :
          [0:10]   = rays Z-scorés
          [10:13]  = derived (asymmetry, front_ray, min_ray)     si use_delta=False
          [10:20]  = delta_rays                                   si use_delta=True
          [20:23]  = derived                                      si use_delta=True
        """
        n = self.n_rays
        nd = self.n_derived
        # Slices
        rays_end = n + (n if self.use_delta else 0)  # fin des rays/delta
        derived_start = rays_end

        # 1. Flip horizontal (symétrie gauche/droite)
        if random.random() < self.flip_prob:
            obs = obs.clone()
            if self.use_delta:
                obs[:n] = obs[:n].flip(0)
                obs[n:n*2] = obs[n:n*2].flip(0)
            else:
                obs[:n] = obs[:n].flip(0)
            # Asymétrie : négation (gauche↔droite inversés)
            obs[derived_start] = -obs[derived_start]
            # prev_steering (dernière feature si action-cond) : aussi négatif
            if self.use_action_cond:
                obs[-1] = -obs[-1]
            action = action * torch.tensor([-1.0, 1.0])

        # 2. Bruit gaussien sur les rayons uniquement
        if self.noise_std > 0:
            noise = torch.randn(n) * self.noise_std
            obs = obs.clone()
            obs[:n] = obs[:n] + noise

        # 3. Ray cutout (masquer 1-2 rayons → simuler capteur mort)
        if self.ray_cutout and n > 4 and random.random() < 0.3:
            n_cut = random.choice([1, 2])
            indices = random.sample(range(n), n_cut)
            obs = obs.clone()
            for i in indices:
                obs[i] = 0.0
                if self.use_delta:
                    obs[n + i] = 0.0

        return obs, action

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
    use_delta: bool = True,
    use_action_cond: bool = True,
    temporal_split: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders train/val/test.

    temporal_split=True : split par position temporelle (frames 0→75%=train, 75→90%=val, 90→100%=test).
    Nécessaire pour PairwiseSmoothingLoss (frames consécutives dans le batch → pas de shuffle).

    temporal_split=False (défaut) : split aléatoire reproductible.
    """
    # Dataset complet sans augmentation pour le split propre
    full_ds = DrivingDataset(data_path, n_rays=n_rays, augment=False, filter_stopped=filter_stopped,
                              ray_stats=ray_stats, use_delta=use_delta, use_action_cond=use_action_cond)
    n = len(full_ds)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    if temporal_split:
        # Indices par position — préserve l'ordre temporel des frames
        train_indices = list(range(0, n_train))
        val_indices   = list(range(n_train, n_train + n_val))
        test_indices  = list(range(n_train + n_val, n))
        shuffle_train = False  # ordre temporel requis pour PairwiseSmoothingLoss
        print(f"[DataLoaders] Split TEMPOREL: Train 0→{n_train:,} | "
              f"Val {n_train:,}→{n_train+n_val:,} | Test →{n:,}")
    else:
        generator = torch.Generator().manual_seed(seed)
        train_split, val_split, test_split = random_split(
            full_ds, [n_train, n_val, n_test], generator=generator
        )
        train_indices = list(train_split.indices)
        val_indices   = list(val_split.indices)
        test_indices  = list(test_split.indices)
        shuffle_train = True

    # Train dataset avec augmentation
    train_ds_aug = DrivingDataset(data_path, n_rays=n_rays, augment=augment_train, filter_stopped=filter_stopped,
                                   ray_stats=ray_stats, use_delta=use_delta, use_action_cond=use_action_cond)
    train_subset = Subset(train_ds_aug, train_indices)

    # WeightedRandomSampler — désactivé si temporal_split (shuffle=False incompatible)
    sampler = None
    if use_weighted_sampler and shuffle_train:
        weights = full_ds.get_steering_weights(n_bins=sampler_bins)
        train_weights = weights[train_indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_indices),
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
        shuffle=shuffle_train if sampler is None else False,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        f"[DataLoaders] Train: {len(train_indices):,} | "
        f"Val: {len(val_indices):,} | Test: {len(test_indices):,}"
    )
    return train_loader, val_loader, test_loader


class EpisodeSequenceDataset(Dataset):
    """
    Dataset séquentiel pour LightGRUCar.

    Fenêtre glissante de seq_len frames consécutives dans le même épisode.
    → Le modèle reçoit (seq_len, features) et prédit l'action de la dernière frame.
    → Élimine le zigzag : le GRU a de la mémoire temporelle.

    Nécessite des colonnes episode_id + step dans le CSV.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 10,
        n_rays: Optional[int] = None,
        ray_stats: Optional[dict] = None,
        filter_stopped: bool = False,
        speed_threshold: float = 0.05,
    ):
        self.seq_len = seq_len

        df = self._load(data_path)

        if filter_stopped:
            df = df[df["speed"] >= speed_threshold].reset_index(drop=True)

        assert "episode_id" in df.columns and "step" in df.columns, \
            "EpisodeSequenceDataset nécessite des colonnes episode_id + step"

        # Trier par ordre temporel
        df = df.sort_values(["episode_id", "step"]).reset_index(drop=True)

        ray_cols = [c for c in df.columns if c.startswith("ray_")]
        if n_rays is not None:
            ray_cols = ray_cols[:n_rays]
        self.n_rays = len(ray_cols)

        # Z-score
        rays = df[ray_cols].values.astype(np.float32)
        if ray_stats is not None:
            mu = np.array(ray_stats["mean"], dtype=np.float32)
            sigma = np.array(ray_stats["std"], dtype=np.float32)
            rays = (rays - mu) / sigma

        # Features dérivées
        n = rays.shape[1]
        half = n // 2
        left_sum = rays[:, :half].sum(axis=1, keepdims=True)
        right_sum = rays[:, half:].sum(axis=1, keepdims=True)
        asymmetry = (right_sum - left_sum) / (left_sum + right_sum + 1e-8)
        front_ray = rays[:, n // 2 - 1 : n // 2 + 1].mean(axis=1, keepdims=True)
        min_ray = rays.min(axis=1, keepdims=True)
        derived = np.concatenate([asymmetry, front_ray, min_ray], axis=1).astype(np.float32)
        self.n_derived = 3

        features = np.concatenate([rays, derived], axis=1)  # (N, n_rays+3)
        actions = df[["steering", "acceleration"]].values.astype(np.float32)

        episode_ids = df["episode_id"].values
        steps = df["step"].values

        # Construire les indices valides : seq_len frames consécutives, même épisode
        self.valid_ends = []
        for i in range(seq_len - 1, len(df)):
            # Vérifier que toutes les frames de la fenêtre sont du même épisode ET consécutives
            start = i - seq_len + 1
            if episode_ids[i] == episode_ids[start] and steps[i] == steps[start] + seq_len - 1:
                self.valid_ends.append(i)

        self.features = torch.from_numpy(features)
        self.actions = torch.from_numpy(actions)

        n_valid = len(self.valid_ends)
        pct = (abs(actions[self.valid_ends, 0]) > 0.15).mean() * 100
        print(f"[EpisodeDataset] {n_valid:,} séquences valides ({seq_len} frames) | "
              f"virages: {pct:.1f}%")

    def __len__(self) -> int:
        return len(self.valid_ends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.valid_ends[idx]
        start = end - self.seq_len + 1
        seq = self.features[start : end + 1]   # (seq_len, features)
        target = self.actions[end]             # action de la dernière frame
        return seq, target

    def _load(self, data_path: str) -> "pd.DataFrame":
        path = Path(data_path)
        if path.is_dir():
            dfs = [pd.read_csv(f) for f in sorted(path.glob("*.csv"))]
            return pd.concat(dfs, ignore_index=True)
        return pd.read_csv(path)


def create_sequence_dataloaders(
    data_path: str,
    seq_len: int = 10,
    n_rays: Optional[int] = None,
    batch_size: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    ray_stats: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders pour LightGRUCar — séquences temporelles épisodiques."""
    ds = EpisodeSequenceDataset(
        data_path, seq_len=seq_len, n_rays=n_rays, ray_stats=ray_stats
    )
    n = len(ds)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split, test_split = random_split(
        ds, [n_train, n_val, n_test], generator=generator
    )

    kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_split,  batch_size=batch_size, shuffle=False, **kwargs)

    print(f"[SeqLoaders] Train: {len(train_split):,} | Val: {len(val_split):,} | "
          f"Test: {len(test_split):,}")
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

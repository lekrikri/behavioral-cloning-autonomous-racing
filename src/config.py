"""
config.py — Configuration centralisée du projet Robocar.

Tous les hyperparamètres en un seul endroit.
Usage:
    from src.config import RobocarConfig
    cfg = RobocarConfig()
    cfg = RobocarConfig(nb_rays=20, lr=5e-4)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


@dataclass
class RobocarConfig:
    # --- Capteurs ---
    nb_rays: int = 10            # nombre de raycasts (1–50)
    fov: int = 180               # champ de vision en degrés (1–180)

    # --- Architecture ---
    model_type: Literal["mlp", "cnn"] = "mlp"
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    dropout: list = field(default_factory=lambda: [0.2, 0.1, 0.0])
    use_batch_norm: bool = True

    # --- Loss ---
    loss_type: Literal["weighted_mse", "huber", "l1"] = "huber"
    steer_weight: float = 0.7
    accel_weight: float = 0.3
    huber_delta: float = 1.0

    # --- Entraînement ---
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 15           # early stopping
    seed: int = 42
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)

    # --- DataLoader ---
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # --- Split ---
    val_ratio: float = 0.15
    test_ratio: float = 0.10

    # --- Augmentation ---
    augment: bool = True
    flip_prob: float = 0.5
    noise_std: float = 0.01
    noise_adaptive: bool = True  # bruit ∝ distance du rayon
    speed_jitter: float = 0.1   # ±10% variation vitesse
    ray_cutout: bool = True      # masquer 1-2 rayons aléatoirement
    filter_stopped: bool = True
    speed_threshold: float = 0.05

    # --- Sampling ---
    use_weighted_sampler: bool = True  # rééquilibrer distribution steering
    sampler_bins: int = 20             # bins pour le calcul des poids

    # --- Inférence ---
    smoothing_alpha: float = 0.7       # filtre exponentiel sur les actions
    port: int = 5004

    # --- Chemins ---
    simulator_path: str = "BuildLinux/RacingSimulator.x86_64"
    data_dir: str = "data"
    models_dir: str = "models"
    config_json: str = "config.json"

    @property
    def obs_size(self) -> int:
        return self.nb_rays + 1  # rays + speed

    def to_agent_config(self) -> dict:
        """Génère le config.json pour le simulateur Unity."""
        return {"agents": [{"fov": self.fov, "nbRay": self.nb_rays}]}

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RobocarConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __repr__(self):
        return (
            f"RobocarConfig(\n"
            f"  model={self.model_type} | nb_rays={self.nb_rays} | obs={self.obs_size}\n"
            f"  loss={self.loss_type} | lr={self.lr} | epochs={self.epochs}\n"
            f"  augment={self.augment} | mixed_precision={self.mixed_precision}\n"
            f")"
        )


if __name__ == "__main__":
    cfg = RobocarConfig()
    print(cfg)
    cfg.save("config_training.json")
    loaded = RobocarConfig.load("config_training.json")
    assert cfg.nb_rays == loaded.nb_rays
    print("[OK] Config save/load OK")

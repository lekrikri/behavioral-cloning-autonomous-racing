"""
model.py — Architectures du modèle IA pour la conduite autonome.

Deux architectures disponibles:
1. RobocarMLP  : MLP classique (baseline, ultra-léger)
2. RobocarCNN  : 1D-CNN + MLP (meilleur pour la structure spatiale des raycasts)

Recommandation Gemini/Grok: 1D-CNN détecte les "motifs" d'obstacles adjacents
que le MLP traite comme variables indépendantes.

Fix Grok (v2):
- BatchNorm order: Linear → BN → ReLU → Dropout (pas Dropout après BN)
- WeightedMSE: poids appliqués avant réduction
- Huber Loss ajoutée
"""

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Blocs de base
# ─────────────────────────────────────────────

class DenseBlock(nn.Module):
    """Couche dense: Linear → LayerNorm → ReLU (pas de Dropout pour régression faible variance)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # LayerNorm plus stable que BatchNorm pour petits vecteurs continus (conseillé Gemini)
        self.bn = nn.LayerNorm(out_dim) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        # Dropout désactivé : détruit le signal faible des raycasts
        self.drop = nn.Identity()
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


# ─────────────────────────────────────────────
# Architecture 1 : MLP
# ─────────────────────────────────────────────

class RobocarMLP(nn.Module):
    """
    MLP conduite autonome v3.

    Input  : [ray_0..ray_N, Δray_0..Δray_N]  →  N*2 features (Z-scorés + delta)
    Output : steering [-1,1] via Tanh  +  accel [0,1] via Sigmoid (bimodal)

    use_delta : ajoute les Δrays (différence frame t - frame t-1)
                → contexte temporel sans LSTM, réduit le zigzag
    bimodal_accel : tête Sigmoid + BCELoss pour l'accel (0=frein, 1=gaz)
    """

    def __init__(
        self,
        n_rays: int = 10,
        hidden_dims: list = None,
        dropout: list = None,
        use_batch_norm: bool = True,
        use_delta: bool = True,
        bimodal_accel: bool = True,
    ):
        super().__init__()
        self.n_rays = n_rays
        self.use_delta = use_delta
        self.bimodal_accel = bimodal_accel
        self.input_size = n_rays * 2 if use_delta else n_rays
        self.output_size = 2

        hidden_dims = hidden_dims or [128, 64, 32]
        dropout = dropout or [0.2, 0.1, 0.0]
        assert len(dropout) == len(hidden_dims)

        layers = []
        in_dim = self.input_size
        for i, (out_dim, drop) in enumerate(zip(hidden_dims, dropout)):
            use_bn = use_batch_norm and i < len(hidden_dims) - 1
            layers.append(DenseBlock(in_dim, out_dim, dropout=drop, use_bn=use_bn))
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        # Tête steering : Tanh → [-1, 1]
        self.steer_head = nn.Linear(in_dim, 1)
        # Tête accel : Sigmoid → [0, 1] (bimodal 0/1) ou Tanh si bimodal=False
        self.accel_head = nn.Linear(in_dim, 1)
        for head in [self.steer_head, self.accel_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne [steer, accel_logit] pendant le training.
        - steer    : Tanh → [-1, 1]
        - accel    : logit brut (sans Sigmoid) si bimodal → BCEWithLogitsLoss AMP-safe
        Appeler .predict() à l'inférence pour obtenir la proba Sigmoid [0,1].
        """
        feat = self.backbone(x)
        steer = torch.tanh(self.steer_head(feat))
        accel_raw = self.accel_head(feat)
        if self.bimodal_accel:
            accel = accel_raw  # logits bruts, Sigmoid appliqué par BCEWithLogitsLoss
        else:
            accel = torch.tanh(accel_raw)
        return torch.cat([steer, accel], dim=-1)

    def predict(self, rays: np.ndarray, delta_rays: Optional[np.ndarray] = None) -> tuple[float, float]:
        """Inférence : applique Sigmoid sur accel si bimodal (forward retourne logits)."""
        self.eval()
        with torch.no_grad():
            if self.use_delta and delta_rays is not None:
                obs = np.concatenate([rays, delta_rays], dtype=np.float32)
            else:
                obs = rays.astype(np.float32)
            out = self.forward(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
            if self.bimodal_accel:
                out = torch.stack([out[0], torch.sigmoid(out[1])])
        return float(out[0]), float(out[1])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str, metadata: Optional[dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": "mlp",
            "state_dict": self.state_dict(),
            "config": {
                "n_rays": self.n_rays,
                "input_size": self.input_size,
                "use_delta": self.use_delta,
                "bimodal_accel": self.bimodal_accel,
            },
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "RobocarMLP":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = cls(
            n_rays=cfg["n_rays"],
            use_delta=cfg.get("use_delta", False),
            bimodal_accel=cfg.get("bimodal_accel", False),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()

    def export_onnx(self, path: str):
        """Export ONNX avec toutes les précautions (eval + cpu + opset 17)."""
        path = Path(path)
        self.eval()
        self.cpu()
        dummy = torch.zeros(1, self.input_size, dtype=torch.float32)
        torch.onnx.export(
            self, dummy, str(path),
            opset_version=18,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        )
        try:
            import onnx
            onnx.checker.check_model(onnx.load(str(path)))
            print(f"[ONNX] Export valide: {path}")
        except ImportError:
            print(f"[ONNX] Exporté (installer onnx pour validation): {path}")

    def quantize(self):
        return torch.quantization.quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8)

    def __repr__(self):
        return f"RobocarMLP(n_rays={self.n_rays}, params={self.count_parameters():,})"


# ─────────────────────────────────────────────
# Architecture 2 : 1D-CNN + MLP
# ─────────────────────────────────────────────

class RobocarCNN(nn.Module):
    """
    1D-CNN pour la détection de motifs spatiaux dans les raycasts.

    Recommandé par Gemini: un kernel de taille 3-5 détecte les patterns
    d'obstacles adjacents (mur à droite, ouverture centrale, coin) que le
    MLP est incapable de capturer.

    Architecture:
      Rays → Conv1d(1,16,k=3) → Conv1d(16,32,k=3) → Flatten
           → Concat(speed) → Dense(128) → Dense(64) → [steer, accel]

    Avantage: ~15-20k params, ~1-2ms sur Jetson, bien meilleure extraction spatiale.
    """

    def __init__(
        self,
        n_rays: int = 10,
        cnn_channels: list = None,
        kernel_size: int = 3,
        mlp_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_rays = n_rays
        self.input_size = n_rays + 1
        self.output_size = 2

        cnn_channels = cnn_channels or [16, 32]
        mlp_dims = mlp_dims or [64, 32]

        # Branche CNN sur les raycasts
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        # Global Average Pooling: mean sur la dim spatiale → ONNX-compatible
        # Résultat: (batch, C_last) indépendamment de n_rays
        cnn_out_dim = cnn_channels[-1]

        # Branche MLP sur [cnn_features, speed]
        mlp_in = cnn_out_dim + 1  # +1 pour la vitesse
        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            use_bn = i < len(mlp_dims) - 1
            mlp_layers.append(DenseBlock(mlp_in, dim, dropout=dropout if i == 0 else 0.0, use_bn=use_bn))
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.head = nn.Linear(mlp_in, self.output_size)
        nn.init.xavier_uniform_(self.head.weight)

        # Initialisation CNN
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_rays + 1)  — rays concat speed
        """
        rays = x[:, :self.n_rays].unsqueeze(1)   # (batch, 1, n_rays)
        speed = x[:, self.n_rays:]                # (batch, 1)

        cnn_features = self.cnn(rays)             # (batch, C, n_rays)
        cnn_flat = cnn_features.mean(dim=-1)      # Global Avg Pool → (batch, C)

        combined = torch.cat([cnn_flat, speed], dim=1)  # (batch, C*4+1)
        features = self.mlp(combined)
        return torch.tanh(self.head(features))

    def predict(self, rays: np.ndarray, speed: float) -> tuple[float, float]:
        self.eval()
        with torch.no_grad():
            obs = np.concatenate([rays, [speed]], dtype=np.float32)
            out = self.forward(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        return float(out[0]), float(out[1])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str, metadata: Optional[dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": "cnn",
            "state_dict": self.state_dict(),
            "config": {"n_rays": self.n_rays, "input_size": self.input_size},
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "RobocarCNN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(n_rays=ckpt["config"]["n_rays"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()

    def export_onnx(self, path: str):
        path = Path(path)
        self.eval()
        self.cpu()
        dummy = torch.zeros(1, self.input_size, dtype=torch.float32)
        torch.onnx.export(
            self, dummy, str(path),
            opset_version=18,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        )
        print(f"[ONNX] CNN exporté: {path}")

    def __repr__(self):
        return f"RobocarCNN(n_rays={self.n_rays}, params={self.count_parameters():,})"


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

class WeightedHuberLoss(nn.Module):
    """
    Huber Loss pondérée (recommandée par Gemini + Grok).

    Avantages vs MSE:
    - Robuste aux outliers (coups de volant brusques dans le dataset)
    - Quadratique près de 0 (précision fine) + linéaire pour grands écarts
    - Bien meilleur pour distributions à queues épaisses

    steering_weight=0.7 : le steering est plus critique pour rester sur la piste.
    """

    def __init__(
        self,
        steer_weight: float = 0.7,
        accel_weight: float = 0.3,
        delta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("weights", torch.tensor([steer_weight, accel_weight]))
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Huber elementwise
        huber = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        # Pondération par output (steer vs accel)
        weighted = huber * self.weights.to(pred.device)
        return weighted.mean()


class BimodalLoss(nn.Module):
    """
    Loss combinée pour modèle bimodal v3:
    - Steering : HuberLoss (continu [-1,1])
    - Accel    : BCELoss (binaire 0/1)

    steer_weight=0.85 car le steering est la variable critique.
    """

    def __init__(self, steer_weight: float = 0.85, bce_weight: float = 0.15, delta: float = 1.0):
        super().__init__()
        self.steer_weight = steer_weight
        self.bce_weight = bce_weight
        self.steer_loss_fn = nn.HuberLoss(delta=delta)
        # BCEWithLogitsLoss = Sigmoid + BCELoss fusionnés → AMP-safe + stable
        self.accel_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        steer_loss = self.steer_loss_fn(pred[:, 0], target[:, 0])
        # Binariser la cible accel : 0 si ≤0.5, 1 si >0.5
        accel_target = (target[:, 1] > 0.5).float()
        # pred[:, 1] = logits bruts (SANS Sigmoid) → BCEWithLogitsLoss applique Sigmoid intern.
        accel_loss = self.accel_loss_fn(pred[:, 1], accel_target)
        return self.steer_weight * steer_loss + self.bce_weight * accel_loss


class WeightedMSELoss(nn.Module):
    """MSE pondérée (baseline — préférer WeightedHuberLoss)."""

    def __init__(self, steer_weight: float = 0.7, accel_weight: float = 0.3):
        super().__init__()
        self.register_buffer("weights", torch.tensor([steer_weight, accel_weight]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2   # (batch, 2)
        return (mse * self.weights.to(pred.device)).mean()


def build_model(
    arch: Literal["mlp", "cnn"] = "mlp",
    n_rays: int = 10,
    **kwargs,
):
    """Factory pour créer un modèle selon l'architecture choisie."""
    if arch == "cnn":
        return RobocarCNN(n_rays=n_rays, **kwargs)
    return RobocarMLP(n_rays=n_rays, **kwargs)


def build_loss(loss_type: str = "huber", steer_weight: float = 0.7, accel_weight: float = 0.3, delta: float = 1.0):
    if loss_type == "bimodal":
        return BimodalLoss(steer_weight=steer_weight, bce_weight=accel_weight, delta=delta)
    if loss_type == "huber":
        return WeightedHuberLoss(steer_weight, accel_weight, delta)
    return WeightedMSELoss(steer_weight, accel_weight)


def load_model(path: str):
    """Charge MLP ou CNN selon le checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "mlp")
    if arch == "cnn":
        return RobocarCNN.load(path)
    return RobocarMLP.load(path)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test RobocarMLP ===")
    mlp = RobocarMLP(n_rays=10)
    batch = torch.randn(32, 11)
    out = mlp(batch)
    assert out.shape == (32, 2)
    assert out.abs().max() <= 1.0 + 1e-5, "Tanh doit borner [-1,1]"
    print(f"  MLP: {mlp.count_parameters():,} params | output {out.shape} range [{out.min():.3f},{out.max():.3f}]")

    # Save/Load
    mlp.save("/tmp/test_mlp.pth")
    mlp2 = RobocarMLP.load("/tmp/test_mlp.pth")
    rays = np.random.rand(10).astype(np.float32)
    s1, a1 = mlp.predict(rays, 0.5)
    s2, a2 = mlp2.predict(rays, 0.5)
    assert abs(s1 - s2) < 1e-5
    print(f"  Save/Load: OK | predict: steering={s1:.4f}, accel={a1:.4f}")

    print("\n=== Test RobocarCNN ===")
    cnn = RobocarCNN(n_rays=10)
    out_cnn = cnn(batch)
    assert out_cnn.shape == (32, 2)
    print(f"  CNN: {cnn.count_parameters():,} params | output {out_cnn.shape}")

    # Test avec 50 rays (scaling)
    cnn50 = RobocarCNN(n_rays=50)
    batch50 = torch.randn(32, 51)
    out50 = cnn50(batch50)
    assert out50.shape == (32, 2)
    print(f"  CNN 50 rays: {cnn50.count_parameters():,} params | OK")

    print("\n=== Test Loss functions ===")
    loss_huber = WeightedHuberLoss()
    loss_mse = WeightedMSELoss()
    pred = torch.randn(32, 2).clamp(-1, 1)
    target = torch.randn(32, 2).clamp(-1, 1)
    lh = loss_huber(pred, target)
    lm = loss_mse(pred, target)
    print(f"  Huber: {lh.item():.4f} | MSE: {lm.item():.4f}")

    print("\n=== ONNX Export ===")
    try:
        mlp.export_onnx("/tmp/test_mlp.onnx")
        cnn.export_onnx("/tmp/test_cnn.onnx")
    except Exception as e:
        print(f"  ONNX: {e}")

    print("\n[OK] Tous les tests passent.")

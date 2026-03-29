"""
evaluate.py — Métriques d'évaluation offline et visualisations.

Métriques calculées:
- MAE steering / acceleration
- RMSE steering / acceleration
- R² steering / acceleration
- Distribution des erreurs
- Courbe d'apprentissage

Usage:
    python src/evaluate.py --model models/best.pth --data data/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """
    Calcule toutes les métriques de régression pour steering et acceleration.

    predictions : Tensor (N, 2) — [steering_pred, accel_pred]
    targets     : Tensor (N, 2) — [steering_true, accel_true]
    """
    pred = predictions.cpu().numpy()
    true = targets.cpu().numpy()

    errors = pred - true
    abs_errors = np.abs(errors)

    metrics = {}
    for i, name in enumerate(["steering", "acceleration"]):
        mae = abs_errors[:, i].mean()
        rmse = np.sqrt((errors[:, i] ** 2).mean())
        # R² = 1 - SS_res / SS_tot
        ss_res = (errors[:, i] ** 2).sum()
        ss_tot = ((true[:, i] - true[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        # Pourcentage de prédictions dans un seuil acceptable
        within_01 = (abs_errors[:, i] < 0.1).mean() * 100
        within_02 = (abs_errors[:, i] < 0.2).mean() * 100

        metrics[f"mae_{name}"] = float(mae)
        metrics[f"rmse_{name}"] = float(rmse)
        metrics[f"r2_{name}"] = float(r2)
        metrics[f"within_0.1_{name}"] = float(within_01)
        metrics[f"within_0.2_{name}"] = float(within_02)

    # Score global (objectif: MAE steering < 0.1)
    metrics["mae_combined"] = (metrics["mae_steering"] * 0.7 + metrics["mae_acceleration"] * 0.3)
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """Affiche les métriques de façon lisible."""
    p = f"[{prefix}] " if prefix else ""
    print(f"\n{p}{'─'*45}")
    print(f"{p} Steering:")
    print(f"{p}   MAE  = {metrics.get('mae_steering', 0):.4f}")
    print(f"{p}   RMSE = {metrics.get('rmse_steering', 0):.4f}")
    print(f"{p}   R²   = {metrics.get('r2_steering', 0):.4f}")
    print(f"{p}   Within ±0.1 = {metrics.get('within_0.1_steering', 0):.1f}%")
    print(f"{p}   Within ±0.2 = {metrics.get('within_0.2_steering', 0):.1f}%")
    print(f"{p} Acceleration:")
    print(f"{p}   MAE  = {metrics.get('mae_acceleration', 0):.4f}")
    print(f"{p}   RMSE = {metrics.get('rmse_acceleration', 0):.4f}")
    print(f"{p}   R²   = {metrics.get('r2_acceleration', 0):.4f}")
    print(f"{p} Combined MAE = {metrics.get('mae_combined', 0):.4f}")
    print(f"{p}{'─'*45}")

    # Verdict
    mae_s = metrics.get("mae_steering", 1.0)
    if mae_s < 0.05:
        verdict = "EXCELLENT"
    elif mae_s < 0.10:
        verdict = "BON"
    elif mae_s < 0.20:
        verdict = "PASSABLE"
    else:
        verdict = "A AMELIORER"
    print(f"{p} Verdict: {verdict} (MAE steering = {mae_s:.3f})")


def evaluate_model_on_dataset(
    model_path: str,
    data_path: str,
    batch_size: int = 256,
    plot: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Évalue un modèle sauvegardé sur un dataset.
    """
    sys.path.insert(0, str(Path(model_path).parent.parent))
    from src.model import RobocarMLP
    from src.dataset import DrivingDataset

    model = RobocarMLP.load(model_path)
    model.eval()

    dataset = DrivingDataset(data_path, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for obs, action in loader:
            pred = model(obs)
            all_preds.append(pred)
            all_targets.append(action)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    print_metrics(metrics, prefix="Evaluation")

    if plot:
        plot_evaluation(preds.numpy(), targets.numpy(), output_dir)

    return metrics


def plot_evaluation(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Optional[str] = None,
):
    """Génère les graphiques d'évaluation."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARNING] matplotlib non disponible pour les graphiques")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle("Évaluation du modèle Robocar", fontsize=14, fontweight="bold")

    labels = ["Steering", "Acceleration"]
    colors = ["#2196F3", "#FF5722"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        pred = predictions[:, i]
        true = targets[:, i]
        errors = pred - true

        # 1. Scatter: prédit vs réel
        ax = fig.add_subplot(gs[i, 0])
        ax.scatter(true, pred, alpha=0.3, s=5, color=color)
        lim = max(abs(true).max(), abs(pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="Idéal")
        ax.set_xlabel(f"{label} réel")
        ax.set_ylabel(f"{label} prédit")
        ax.set_title(f"{label}: Prédit vs Réel")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend()

        # 2. Distribution des erreurs
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(errors, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="black", linestyle="--", alpha=0.8)
        ax2.axvline(errors.mean(), color="red", linestyle="-", alpha=0.8, label=f"Mean={errors.mean():.3f}")
        ax2.set_xlabel("Erreur (prédit - réel)")
        ax2.set_ylabel("Fréquence")
        ax2.set_title(f"{label}: Distribution des erreurs")
        ax2.legend()

        # 3. Série temporelle (premier 500 samples)
        ax3 = fig.add_subplot(gs[i, 2])
        n = min(500, len(pred))
        t = np.arange(n)
        ax3.plot(t, true[:n], label="Réel", alpha=0.7)
        ax3.plot(t, pred[:n], label="Prédit", alpha=0.7)
        ax3.set_xlabel("Frame")
        ax3.set_ylabel(label)
        ax3.set_title(f"{label}: Série temporelle")
        ax3.legend()

    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "evaluation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Sauvegardé: {path}")
    else:
        plt.savefig("evaluation.png", dpi=150, bbox_inches="tight")
        print("[Plot] Sauvegardé: evaluation.png")

    plt.close()


def plot_training_history(history_path: str, output_dir: Optional[str] = None):
    """Affiche les courbes d'apprentissage depuis l'historique JSON."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Courbes d'apprentissage", fontsize=12)

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss (Weighted MSE)")
    axes[0].legend()

    # MAE
    axes[1].plot(history["val_mae_steer"], label="MAE Steering", color="blue")
    axes[1].plot(history["val_mae_accel"], label="MAE Acceleration", color="orange")
    axes[1].axhline(0.1, color="green", linestyle="--", alpha=0.5, label="Seuil 0.1")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Métriques validation")
    axes[1].legend()

    plt.tight_layout()
    save_path = (Path(output_dir) / "training_curves.png") if output_dir else Path("training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Courbes sauvegardées: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Évaluer le modèle Robocar")
    parser.add_argument("--model", required=True, help="Chemin vers le checkpoint .pth")
    parser.add_argument("--data", required=True, help="CSV ou répertoire de données")
    parser.add_argument("--no-plot", action="store_true", help="Désactiver les graphiques")
    parser.add_argument("--output-dir", default=None, help="Répertoire pour les graphiques")
    args = parser.parse_args()

    evaluate_model_on_dataset(
        model_path=args.model,
        data_path=args.data,
        plot=not args.no_plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

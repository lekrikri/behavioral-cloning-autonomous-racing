"""
train.py — Boucle d'entraînement optimisée (v2).

Améliorations Grok:
- Mixed precision (torch.amp.autocast + GradScaler)
- DataLoader optimisé (workers, pin_memory, prefetch, persistent_workers)
- ONNX export correct (model.eval() + cpu + opset 17 + onnx.checker)
- torch.compile optionnel

Usage:
    python src/train.py --data data/ --epochs 100
    python src/train.py --data data/ --arch cnn --epochs 100 --loss huber
    python src/train.py --data data/ --resume models/best.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model, build_loss, load_model
from src.dataset import create_dataloaders
from src.evaluate import compute_metrics, print_metrics


def train_epoch(model, loader, optimizer, loss_fn, device, clip_grad=1.0, use_amp=True):
    model.train()
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    total_loss = 0.0
    all_preds, all_targets = [], []
    bimodal = getattr(model, "bimodal_accel", False)

    for obs, action in loader:
        obs, action = obs.to(device, non_blocking=True), action.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # plus efficace que zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(obs)
            loss = loss_fn(pred, action)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(obs)
        # Sigmoid sur logits accel pour métriques cohérentes avec la cible [0,1]
        pred_m = pred.detach().float().cpu()
        if bimodal:
            pred_m = torch.stack([pred_m[:, 0], torch.sigmoid(pred_m[:, 1])], dim=1)
        all_preds.append(pred_m)
        all_targets.append(action.float().cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    bimodal = getattr(model, "bimodal_accel", False)

    for obs, action in loader:
        obs, action = obs.to(device, non_blocking=True), action.to(device, non_blocking=True)
        pred = model(obs)
        loss = loss_fn(pred, action)
        total_loss += loss.item() * len(obs)
        # Convertir logits accel → probabilité [0,1] pour les métriques
        pred_metrics = pred.float().cpu()
        if bimodal:
            pred_metrics = torch.stack([
                pred_metrics[:, 0],
                torch.sigmoid(pred_metrics[:, 1]),
            ], dim=1)
        all_preds.append(pred_metrics)
        all_targets.append(action.float().cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(
    data_path: str,
    output_dir: str = "models",
    arch: str = "mlp",
    loss_type: str = "bimodal",
    n_rays: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    steer_weight: float = 0.85,
    accel_weight: float = 0.15,
    patience: int = 15,
    resume_path: Optional[str] = None,
    seed: int = 42,
    mixed_precision: bool = True,
    compile_model: bool = False,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    use_delta: bool = True,
    bimodal_accel: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f" Robocar Training — Behavioral Cloning v2")
    print(f"{'='*62}")
    print(f" Device        : {device}")
    print(f" Architecture  : {arch.upper()}")
    print(f" Loss          : {loss_type}")
    print(f" Data          : {data_path}")
    print(f" Epochs        : {epochs} | BS: {batch_size} | LR: {lr}")
    print(f" Mixed FP16    : {mixed_precision and device.type == 'cuda'}")
    print(f" Sampler       : {'weighted' if use_weighted_sampler else 'uniform'}")
    print(f" Delta rays    : {use_delta}")
    print(f" Bimodal accel : {bimodal_accel}")
    print(f"{'='*62}\n")

    # Charger les stats Z-score si disponibles
    import json as _json
    ray_stats_path = Path("models/ray_stats.json")
    ray_stats = None
    if ray_stats_path.exists():
        with open(ray_stats_path) as f:
            ray_stats = _json.load(f)
        print(f"[Train] Z-score stats chargées depuis {ray_stats_path}")

    # DataLoaders optimisés
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path,
        n_rays=n_rays,
        batch_size=batch_size,
        augment_train=True,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        filter_stopped=False,
        ray_stats=ray_stats,
        use_delta=use_delta,
    )

    # Détecter n_rays depuis le dataset
    detected_n_rays = train_loader.dataset.dataset.n_rays

    # Modèle
    if resume_path:
        model = load_model(resume_path).to(device)
        print(f"[Resume] {resume_path}")
    else:
        model = build_model(arch=arch, n_rays=detected_n_rays, use_delta=use_delta, bimodal_accel=bimodal_accel).to(device)

    # torch.compile (PyTorch 2.0+)
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Compile] torch.compile activé")
        except Exception as e:
            print(f"[Compile] Ignoré: {e}")

    print(f"Modèle: {model}\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = build_loss(loss_type, steer_weight, accel_weight)

    history = {"train_loss": [], "val_loss": [], "val_mae_steer": [], "val_mae_accel": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = output_dir / "best.pth"

    header = (f"{'Epoch':>6} | {'Train':>8} | {'Val':>8} | "
              f"{'SteerMAE':>8} | {'AccelMAE':>8} | {'LR':>8} | {'Time':>5}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, loss_fn, device,
                              use_amp=mixed_precision)
        val_m = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_m["loss"])

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_mae_steer"].append(val_m["mae_steering"])
        history["val_mae_accel"].append(val_m["mae_acceleration"])

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        marker = ""

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            patience_counter = 0
            # Récupérer le modèle sans compile wrapper
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            raw_model.save(str(best_model_path), metadata={
                "epoch": epoch, "val_loss": val_m["loss"],
                "val_mae_steer": val_m["mae_steering"],
                "arch": arch, "loss_type": loss_type,
            })
            marker = " ←best"
        else:
            patience_counter += 1

        print(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {val_m['loss']:>8.4f} | "
            f"{val_m['mae_steering']:>8.4f} | {val_m['mae_acceleration']:>8.4f} | "
            f"{current_lr:>8.1e} | {elapsed:>4.1f}s{marker}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stop] {patience} epochs sans amélioration.")
            break

    # Évaluation finale
    print(f"\n{'='*62}")
    print(" Évaluation finale — Test Set")
    print(f"{'='*62}")
    best_model = load_model(str(best_model_path)).to(device)
    test_m = eval_epoch(best_model, test_loader, loss_fn, device)
    print_metrics(test_m, prefix="Test")

    # Historique JSON
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Export ONNX (avec toutes les précautions)
    onnx_path = output_dir / "best.onnx"
    try:
        # Récupérer le modèle raw (sans torch.compile wrapper)
        export_model = best_model
        if hasattr(export_model, "_orig_mod"):
            export_model = export_model._orig_mod
        export_model.export_onnx(str(onnx_path))
    except Exception as e:
        print(f"[WARNING] ONNX export: {e}")

    print(f"\n[OK] Best model : {best_model_path}")
    print(f"[OK] History    : {history_path}")
    return best_model, history


def main():
    parser = argparse.ArgumentParser(description="Entraîner le modèle Robocar v2")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models")
    parser.add_argument("--arch", default="mlp", choices=["mlp", "cnn"],
                        help="Architecture: mlp (baseline) ou cnn (recommandé Gemini)")
    parser.add_argument("--loss", default="bimodal", choices=["bimodal", "huber", "weighted_mse"],
                        help="Fonction de perte")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--steer-weight", type=float, default=0.7)
    parser.add_argument("--accel-weight", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Désactiver mixed precision")
    parser.add_argument("--compile", action="store_true", help="Activer torch.compile")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-sampler", action="store_true", help="Désactiver WeightedRandomSampler")
    parser.add_argument("--no-delta", action="store_true", help="Désactiver delta rays (v2 mode)")
    parser.add_argument("--no-bimodal", action="store_true", help="Désactiver tête accel bimodale")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        arch=args.arch,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steer_weight=args.steer_weight,
        accel_weight=args.accel_weight,
        patience=args.patience,
        resume_path=args.resume,
        seed=args.seed,
        mixed_precision=not args.no_amp,
        compile_model=args.compile,
        num_workers=args.workers,
        use_weighted_sampler=not args.no_sampler,
        use_delta=not args.no_delta,
        bimodal_accel=not args.no_bimodal,
    )


if __name__ == "__main__":
    main()

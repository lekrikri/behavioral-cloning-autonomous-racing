"""
Entraînement du Micro U-Net de segmentation de piste.

Usage :
  python src/train_mask.py                          # paramètres par défaut
  python src/train_mask.py --epochs 80 --batch 16   # personnalisé

Sorties (dans models/mask_v1/) :
  best.pth   — meilleurs poids PyTorch
  best.onnx  — modèle exporté pour Jetson Nano / ONNX Runtime
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent))
from dataset_masks import MaskDataset
from unet_model import DiceBCELoss, MicroUNet


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum().item()
    union = (pred_bin + target).clamp(0, 1).sum().item()
    return inter / (union + 1e-6)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_ds = MaskDataset(args.images, args.masks, augment=False)
    val_size   = max(1, int(0.2 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Réappliquer les augmentations uniquement sur le train split
    train_ds.dataset.augment = False  # sera géré par wrapper
    train_ds_aug = MaskDataset(args.images, args.masks, augment=True)
    # Sous-ensemble avec les mêmes indices
    train_ds_aug = torch.utils.data.Subset(train_ds_aug, train_ds.indices)

    train_loader = DataLoader(train_ds_aug, batch_size=args.batch,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,       batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train : {train_size} | Val : {val_size}")

    # ── Modèle ───────────────────────────────────────────────────────────────
    model     = MicroUNet().to(device)
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )
    print(f"Paramètres modèle : {model.count_params():,}")

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    # ── Boucle d'entraînement ───────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        val_loss = val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                val_iou  += iou_score(preds, masks)

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_iou    /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  IoU={val_iou:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path / "best.pth")
            print(f"  → Meilleur modèle sauvegardé (val={best_val:.4f})")

    # ── Export ONNX ─────────────────────────────────────────────────────────
    print("\nExport ONNX...")
    model.load_state_dict(torch.load(out_path / "best.pth", map_location="cpu"))
    model.eval().cpu()
    dummy = torch.randn(1, 3, 128, 256)
    onnx_path = str(out_path / "best.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
    )
    print(f"ONNX exporté : {onnx_path}")
    print(f"\nEntraînement terminé. Meilleur val_loss = {best_val:.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Micro U-Net segmentation piste")
    parser.add_argument("--images",  default="data/256_128/256_128",
                        help="Dossier images brutes")
    parser.add_argument("--masks",   default="data/masks_auto",
                        help="Dossier masques générés par mask_generator.py")
    parser.add_argument("--output",  default="models/mask_v1",
                        help="Dossier de sortie modèle")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=1e-3)
    args = parser.parse_args()
    train(args)

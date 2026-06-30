"""
Dataset PyTorch pour l'entraînement du Micro U-Net de segmentation.

Attend la structure :
  images_dir/  → *_original_image.png
  masks_dir/   → *_original_image_mask.png  (générés par mask_generator.py)

Augmentations (mode train) :
  - Flip horizontal (symétrie piste G/D)
  - Variation de luminosité  ×[0.6 – 1.4]
  - Variation de contraste   ×[0.8 – 1.2]
  - Bruit gaussien           σ=0.02
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.augment    = augment

        self.samples: list[tuple[Path, Path]] = []
        for mask_path in sorted(self.masks_dir.glob("*_mask.png")):
            # Retrouver l'image source (même préfixe, sans _mask)
            stem = mask_path.name.replace("_mask.png", ".png")
            img_path = self.images_dir / stem
            if img_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune paire image/masque trouvée.\n"
                f"  images : {self.images_dir}\n"
                f"  masques : {self.masks_dir}\n"
                f"Génère d'abord les masques : python -m src.mask.training.mask_generator"
            )
        print(f"MaskDataset ({'train+aug' if augment else 'val'}) : {len(self.samples)} paires")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"),  dtype=np.float32) / 255.0
        mask  = np.array(Image.open(mask_path).convert("L"),   dtype=np.float32) / 255.0

        if self.augment:
            image, mask = self._augment(image, mask)

        # (H, W, 3) → (3, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask  = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

    # ── Augmentations ───────────────────────────────────────────────────────

    def _augment(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Flip horizontal — symétrise les virages G/D
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask  = mask[:,  ::-1   ].copy()

        # Luminosité
        image = np.clip(image * random.uniform(0.6, 1.4), 0.0, 1.0)

        # Contraste
        mean  = image.mean()
        image = np.clip((image - mean) * random.uniform(0.8, 1.2) + mean, 0.0, 1.0)

        # Bruit gaussien (simule reflets variables)
        image = np.clip(
            image + np.random.normal(0, 0.02, image.shape).astype(np.float32),
            0.0, 1.0,
        )

        return image, mask

"""
Mask Generator — Génération automatique de masques de piste (vision classique).

Pipeline :
  1. CLAHE  — normalise l'éclairage, atténue les reflets
  2. HSV    — détection lignes blanches (S faible + V élevé)
  3. Threshold inverse — isole le sol sombre
  4. Flood fill multi-seed depuis le bas (la piste touche toujours le bas)
  5. Morphological closing — comble les trous dus aux reflets
  6. Largest connected component — supprime les artefacts isolés

Input  : image 256×128 RGB uint8
Output : masque binaire 256×128 uint8 (255 = piste navigable, 0 = hors-piste)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ── Paramètres calibrés sur le dataset 256_128 ─────────────────────────────
WHITE_V_MIN   = 195   # lignes blanches : luminosité minimale
WHITE_S_MAX   = 45    # lignes blanches : saturation maximale
DARK_THRESH   = 155   # sol sombre : tout pixel < seuil
FLOOD_LODIFF  = 35    # tolérance flood fill bas
FLOOD_UPDIFF  = 35    # tolérance flood fill haut
CLOSE_KSIZE   = (13, 9)   # noyau morphological close (comble les reflets)
OPEN_KSIZE    = (5, 5)    # noyau morphological open (supprime bruit)
MIN_AREA_PCT  = 0.02  # superficie minimale du masque (% de l'image)
# ────────────────────────────────────────────────────────────────────────────


def generate_mask(image: np.ndarray) -> np.ndarray:
    """
    Génère un masque de piste pour une image RGB 256×128.

    Args:
        image: np.ndarray shape (128, 256, 3) dtype uint8, RGB

    Returns:
        masque: np.ndarray shape (128, 256) dtype uint8, 255=piste 0=hors-piste
    """
    if not _CV2:
        raise ImportError("opencv-python requis : pip install opencv-python-headless")

    assert image.ndim == 3 and image.shape[2] == 3, "Image doit être (H, W, 3) RGB"
    h, w = image.shape[:2]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ── 1. CLAHE sur canal L (espace LAB) ──────────────────────────────────
    # Atténue les variations locales de luminosité (reflets, ombres)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── 2. Détection lignes blanches (HSV) ─────────────────────────────────
    hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV)
    white_lines = cv2.inRange(
        hsv,
        np.array([0,       0,        WHITE_V_MIN]),
        np.array([180, WHITE_S_MAX,        255]),
    )

    # ── 3. Masque sol sombre ───────────────────────────────────────────────
    gray = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    # Exclure les lignes blanches du masque sombre
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(white_lines))

    # ── 4. Flood fill multi-seed depuis le bas ─────────────────────────────
    # La piste est toujours connectée au bord inférieur (caméra frontale)
    seeds = [
        (w // 2,     h - 5),   # centre bas
        (w // 3,     h - 5),   # tiers gauche
        (2 * w // 3, h - 5),   # tiers droit
        (w // 2,     h - 15),  # centre légèrement plus haut
    ]

    work = dark.copy()
    for sx, sy in seeds:
        if 0 <= sx < w and 0 <= sy < h and work[sy, sx] == 255:
            ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(
                work, ff_mask, (sx, sy), 128,
                loDiff=(FLOOD_LODIFF,),
                upDiff=(FLOOD_UPDIFF,),
                flags=cv2.FLOODFILL_FIXED_RANGE | 4,
            )

    flooded = (work == 128).astype(np.uint8) * 255

    # Fallback : si flood fill trop restrictif, garder la moitié basse du dark mask
    if flooded.sum() < h * w * MIN_AREA_PCT * 255:
        flooded = dark.copy()
        flooded[: h // 2, :] = 0

    # ── 5. Morphological closing — comble les trous (reflets) ──────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KSIZE)
    flooded = cv2.morphologyEx(flooded, cv2.MORPH_CLOSE, k_close)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KSIZE)
    flooded = cv2.morphologyEx(flooded, cv2.MORPH_OPEN, k_open)

    # ── 6. Largest connected component ─────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flooded, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        flooded = (labels == largest).astype(np.uint8) * 255

    return flooded


def visualize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay vert semi-transparent sur la zone piste détectée."""
    overlay = image.copy()
    overlay[mask == 255, 1] = np.clip(overlay[mask == 255, 1].astype(int) + 80, 0, 255)
    return (overlay * 0.6 + image * 0.4).astype(np.uint8)


def batch_generate(input_dir: str, output_dir: str, preview_dir: str = None) -> None:
    """Génère les masques pour toutes les images d'un dossier."""
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if preview_dir:
        prev_path = Path(preview_dir)
        prev_path.mkdir(parents=True, exist_ok=True)

    images = sorted([
        f for f in in_path.iterdir()
        if f.suffix == ".png" and "Identifier" not in f.name
    ])

    ok = fail = 0
    print(f"Génération masques : {len(images)} images → {out_path}")

    for img_path in images:
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask  = generate_mask(image)

            mask_name = img_path.stem + "_mask.png"
            Image.fromarray(mask).save(out_path / mask_name)

            if preview_dir:
                vis = visualize(image, mask)
                Image.fromarray(vis).save(prev_path / (img_path.stem + "_preview.png"))

            ok += 1
        except Exception as e:
            print(f"  [ERREUR] {img_path.name}: {e}", file=sys.stderr)
            fail += 1

    print(f"Terminé : {ok} masques OK, {fail} erreurs.")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Generator — vision classique")
    parser.add_argument("--input",   default="data/256_128/256_128",
                        help="Dossier des images brutes")
    parser.add_argument("--output",  default="data/masks_auto",
                        help="Dossier de sortie des masques")
    parser.add_argument("--preview", default=None,
                        help="Dossier optionnel pour les overlays de vérification")
    parser.add_argument("--single",  default=None,
                        help="Traiter une seule image (chemin complet)")
    args = parser.parse_args()

    if args.single:
        img = np.array(Image.open(args.single).convert("RGB"))
        msk = generate_mask(img)
        out = args.single.replace(".png", "_mask.png")
        Image.fromarray(msk).save(out)
        print(f"Masque : {out}")
        if _CV2:
            vis = visualize(img, msk)
            Image.fromarray(vis).save(args.single.replace(".png", "_preview.png"))
    else:
        batch_generate(args.input, args.output, args.preview)

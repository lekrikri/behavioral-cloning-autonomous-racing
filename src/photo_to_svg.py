#!/usr/bin/env python3
"""
photo_to_svg.py — Génère un SVG top-view noir/blanc de la piste depuis une photo.

Usage:
  python3 src/photo_to_svg.py photo.jpg [output.svg]

Le script :
  1. Détecte (ou utilise) les 4 coins du tapis noir pour corriger la perspective
  2. Seuille les bandes blanches du ruban de piste
  3. Extrait et simplifie les contours
  4. Génère un SVG vectoriel propre (piste blanche sur fond noir)

Pour ajuster les coins manuellement si l'auto-détection est mauvaise :
  Modifier MANUAL_CORNERS ci-dessous avec les coordonnées pixel des 4 coins du tapis
  dans l'ordre : [bas-gauche, bas-droite, haut-droite, haut-gauche]
"""

import sys
import os
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SVG_W, SVG_H = 900, 600   # dimensions du SVG de sortie
WHITE_THRESH  = 190        # seuil luminosité pour "blanc" (0-255)
MIN_AREA      = 400        # aire min d'un contour à garder (px²)
SIMPLIFY_EPS  = 2.5        # epsilon approximation polygonale (px)

# Corners manuels : [bas-gauche, bas-droite, haut-droite, haut-gauche] en pixels photo
# Laisser None pour auto-détection.
# MANUAL_CORNERS = np.float32([[10, 530], [415, 390], [380, 50], [5, 200]])
MANUAL_CORNERS = None


# ── Détection automatique des coins du tapis ──────────────────────────────────
def auto_corners(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Le tapis = zone la plus sombre (noir) sur fond de salle
    _, dark = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((20, 20), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=3)

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    ratio = cv2.contourArea(cnt) / (h * w)
    print("[auto] Tapis détecté : {:.0f}% de l'image".format(ratio * 100))

    peri = cv2.arcLength(cnt, True)
    for eps in [0.01, 0.02, 0.04, 0.06, 0.10]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)

    # Fallback : 4 points extrêmes
    pts = cnt.reshape(-1, 2)
    corners = np.array([
        pts[np.argmax( pts[:, 0] + pts[:, 1])],   # bas-droite
        pts[np.argmin(-pts[:, 0] + pts[:, 1])],   # bas-gauche
        pts[np.argmin( pts[:, 0] + pts[:, 1])],   # haut-gauche
        pts[np.argmax(-pts[:, 0] + pts[:, 1])],   # haut-droite
    ], dtype=np.float32)
    return corners


def order_quad(pts):
    """[haut-gauche, haut-droite, bas-droite, bas-gauche]"""
    pts = pts.astype(np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def rectify(img, corners):
    src = order_quad(corners)
    dst = np.float32([[0, 0], [SVG_W, 0], [SVG_W, SVG_H], [0, SVG_H]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (SVG_W, SVG_H))


# ── Détection des bandes blanches ────────────────────────────────────────────
def white_mask(img_rect):
    lab = cv2.cvtColor(img_rect, cv2.COLOR_BGR2Lab)
    l   = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    _, m  = cv2.threshold(l, WHITE_THRESH, 255, cv2.THRESH_BINARY)

    k3 = np.ones((3, 3), np.uint8)
    k9 = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k3, iterations=1)
    return m


# ── Contours → SVG paths ──────────────────────────────────────────────────────
def to_svg_paths(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(cnt, SIMPLIFY_EPS, True)
        pts = approx.reshape(-1, 2)
        d = "M {} {}".format(*pts[0])
        for p in pts[1:]:
            d += " L {} {}".format(*p)
        d += " Z"
        paths.append(d)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    photo = sys.argv[1] if len(sys.argv) > 1 else "track_photo.jpg"
    out   = sys.argv[2] if len(sys.argv) > 2 else \
            os.path.splitext(photo)[0] + "_map.svg"

    if not os.path.exists(photo):
        print("ERREUR: fichier introuvable → {}".format(photo))
        print("Usage: python3 src/photo_to_svg.py photo.jpg [output.svg]")
        sys.exit(1)

    img = cv2.imread(photo)
    if img is None:
        print("ERREUR: impossible de lire l'image")
        sys.exit(1)
    h, w = img.shape[:2]
    print("[photo] {} — {}x{}px".format(photo, w, h))

    # ── Coins du tapis ──
    corners = MANUAL_CORNERS
    if corners is None:
        corners = auto_corners(img)

    if corners is None or len(corners) != 4:
        print("[WARN] Coins non trouvés — utilisation de toute l'image (sans correction)")
        corners = np.float32([[0, h], [w, h], [w, 0], [0, 0]])

    print("[corners] {}".format(corners.tolist()))

    # ── Rectification ──
    img_rect = rectify(img, corners)
    debug_rect = out.replace(".svg", "_rect.jpg")
    cv2.imwrite(debug_rect, img_rect)
    print("[debug] Vue rectifiée → {}".format(debug_rect))

    # ── Détection blanc ──
    mask = white_mask(img_rect)
    debug_mask = out.replace(".svg", "_mask.jpg")
    cv2.imwrite(debug_mask, mask)
    print("[debug] Masque → {}".format(debug_mask))

    # ── SVG ──
    paths = to_svg_paths(mask)
    print("[svg] {} contours extraits".format(len(paths)))

    path_elems = "\n  ".join(
        '<path d="{}" fill="white" stroke="white" stroke-width="0.5"/>'.format(p)
        for p in paths
    )

    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{w}" height="{h}" viewBox="0 0 {w} {h}"
     style="background:#111111">
  <!-- Piste extraite depuis photo (bandes blanches = ruban) -->
  {paths}
  <text x="8" y="{ty}" font-size="11" fill="#444" font-family="monospace">TRACK MAP — photo top-view</text>
</svg>""".format(w=SVG_W, h=SVG_H, paths=path_elems, ty=SVG_H - 6)

    with open(out, "w") as f:
        f.write(svg)
    print("[OK] SVG → {}".format(out))
    print()
    print("Si la perspective est mauvaise, éditer MANUAL_CORNERS dans le script")
    print("avec les 4 coins du tapis [bas-gauche, bas-droite, haut-droite, haut-gauche]")


if __name__ == "__main__":
    main()

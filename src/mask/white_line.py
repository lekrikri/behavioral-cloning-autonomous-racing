"""White-line mask: raw BGR frame -> binary image (255 = track white line, 0 = else).

Stacked artefact-rejection filters, applied in order so the mask survives glare,
surrounding objects and sensor noise across lighting conditions:

  1. CLAHE (LAB L channel)      flatten local illumination (glare, shadows)
  2. achromatic-bright gate     drop dark pixels (V floor) and colours (S ceiling)
  3. white top-hat              drop large bright areas (mats, walls), keep thin lines
  4. morphology                 close gaps, remove specks
  5. depth vertical filter      drop off-ground pixels (walls/objects) via ground plane
  6. component filter           drop small (noise) and non-rectilinear (blobs) groups

The output is the deliverable: an image with the white lines and the rest black.
Every stage is a keyword parameter (cf. rule 40 — nothing hard-coded in the logic).
"""

import cv2
import numpy as np

from src.mask.camera_ground import CameraGround


def white_line_mask(
    bgr: np.ndarray,
    depth_mm: np.ndarray = None,
    geom: CameraGround = None,
    *,
    clahe_clip: float = 2.0,
    s_max: int = 60,
    v_min_floor: int = 180,
    otsu_cap: int = 220,
    tophat_k: int = 15,
    tophat_thresh: int = 12,
    morph_k: int = 3,
    min_area: int = 40,
    min_elongation: float = 4.0,
    depth_tol: float = 0.35,
    bottom_ignore_frac: float = 0.0,
) -> np.ndarray:
    """Binary white-line mask (uint8, 0/255).

    depth_mm : optional depth map (mm) ALIGNED to the colour frame, same size as bgr.
               When given with `geom`, off-ground pixels (vertical surfaces) are removed.
    geom     : CameraGround, required for the depth vertical filter.
    """
    # 1. CLAHE on L (LAB): local contrast normalisation against glare/shadows.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Achromatic-bright gate. V floor drops dark pixels; the floor is raised toward
    # Otsu (scene-adaptive) but capped so a dim scene never opens the gate too wide.
    # S ceiling drops saturated colours -> only near-white survives.
    hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_min = max(int(v_min_floor), min(int(otsu), int(otsu_cap)))
    m = cv2.inRange(hsv, np.array([0, 0, v_min], np.uint8),
                    np.array([180, int(s_max), 255], np.uint8))

    # Carrosserie avant de la voiture : bande basse fixe de l'image, zérotée tôt pour
    # qu'elle ne survive pas ni ne se connecte à une vraie ligne au-dessus.
    if bottom_ignore_frac > 0.0:
        m[int(m.shape[0] * (1.0 - bottom_ignore_frac)):, :] = 0

    # 3. White top-hat on V: a line is a THIN bright structure; mats/walls/large glare
    # are WIDE -> near-zero top-hat response -> removed by the AND. Lines survive.
    if tophat_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_k, tophat_k))
        th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k)
        m = cv2.bitwise_and(m, (th >= tophat_thresh).astype(np.uint8) * 255)

    # 4. Morphology: close line gaps, open away isolated specks.
    if morph_k > 0:
        k = np.ones((morph_k, morph_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # 5. Vertical-surface rejection via depth: a pixel whose valid depth is well closer
    # than the ground plane at its row is off-ground (wall, chair leg) -> drop it.
    # Invalid depth (0, e.g. textureless asphalt) is kept — absence of depth is not proof.
    if depth_mm is not None and geom is not None:
        # Filet de sécurité : si le hub ne fournit pas encore la depth alignée à la taille
        # couleur (setDepthAlign/setOutputSize), on la redimensionne (NEAREST = pas d'interp
        # des mm). L'alignement exact reste préférable (hub), ceci évite juste le crash.
        if depth_mm.shape != m.shape:
            depth_mm = cv2.resize(depth_mm, (m.shape[1], m.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        thresh = geom.ground_depth_mm * (1.0 - depth_tol)   # (H,)
        reject = (depth_mm > 0) & (depth_mm.astype(np.float32) < thresh[:, None])
        m[reject] = 0

    # 6. Component filter: area (noise) + elongation (keep straight lines, drop blobs).
    if min_area > 0 or min_elongation > 1.0:
        m = _keep_line_components(m, min_area, min_elongation)

    return m


def _keep_line_components(mask: np.ndarray, min_area: int,
                          min_elongation: float) -> np.ndarray:
    """Keep components that are large enough AND rectilinear.

    Rectilinearity = sqrt(lambda_max / lambda_min) of the pixel covariance (PCA). A
    straight thin line has almost all variance along one axis -> large ratio; a round
    reflection or a blob spreads both ways -> small ratio. Rotation-invariant, so a
    diagonal line is treated the same as a vertical one (unlike a bbox fill ratio).

    Jetson-sober: per-label moments are accumulated with np.bincount (one C-level pass
    over the foreground pixels) and the 2x2 eigenvalues use the closed form, instead of
    a Python loop that slices `labels == i` for every component.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    keep = areas >= max(min_area, 1)
    keep[0] = False                                     # background label

    if min_elongation > 1.0:
        ys, xs = np.nonzero(labels)
        lab = labels[ys, xs]
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)
        cnt = np.bincount(lab, minlength=n).astype(np.float64)
        sx = np.bincount(lab, weights=xs,      minlength=n)
        sy = np.bincount(lab, weights=ys,      minlength=n)
        sxx = np.bincount(lab, weights=xs * xs, minlength=n)
        syy = np.bincount(lab, weights=ys * ys, minlength=n)
        sxy = np.bincount(lab, weights=xs * ys, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(cnt > 0, 1.0 / cnt, 0.0)
            mx, my = sx * inv, sy * inv
            var_x = sxx * inv - mx * mx                 # covariance entries
            var_y = syy * inv - my * my
            cov_xy = sxy * inv - mx * my
            half = (var_x + var_y) / 2.0
            disc = np.sqrt(np.maximum(((var_x - var_y) / 2.0) ** 2 + cov_xy ** 2, 0.0))
            lam_max = half + disc
            lam_min = half - disc
            elong = np.sqrt(np.maximum(lam_max, 1e-6) / np.maximum(lam_min, 1e-6))
        keep &= (cnt >= 2) & (elong >= min_elongation)

    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def _selftest():
    # Synthetic scene: a thin diagonal line (keep) + a round blob (drop, not rectilinear)
    # + a few isolated specks (drop, noise), all bright and achromatic.
    img = np.zeros((256, 512, 3), np.uint8)
    cv2.line(img, (100, 250), (260, 90), (255, 255, 255), 2)      # the line
    cv2.circle(img, (400, 180), 22, (255, 255, 255), -1)          # round reflection
    for x, y in [(50, 40), (480, 30), (300, 20)]:
        img[y, x] = (255, 255, 255)                              # specks

    m = white_line_mask(img, tophat_k=0, min_elongation=4.0, min_area=40)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    kept = n - 1
    assert kept == 1, "expected only the line to survive, got %d components" % kept
    # the survivor must be the diagonal line region, not the blob
    ys, xs = np.where(m > 0)
    assert xs.min() < 270 and xs.max() < 300, "wrong component kept (blob not rejected)"
    print("white_line_mask self-test OK (line kept, blob + specks rejected)")

    # Depth de taille différente du masque (hub non aligné) ne doit pas planter.
    from src.mask.camera_ground import CameraGround
    g = CameraGround(512, 256, 68.79, 0.33, 18.0)
    depth = np.full((400, 640), 3000, np.uint16)
    white_line_mask(img, depth_mm=depth, geom=g, tophat_k=0)
    print("white_line_mask depth-resize OK (400x640 depth -> 256x512 mask)")


if __name__ == "__main__":
    _selftest()

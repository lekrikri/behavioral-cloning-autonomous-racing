"""Lane detection from a binary mask — extracted from the controller_pd god file.

Pure vision: turns a white-line mask into lane-position / lateral-error estimates via
blobs, column histograms and horizontal scanlines. No control, no I/O, no globals — the
image geometry and thresholds that used to be module-level (and mutated at runtime) are
grouped in `LaneParams` and passed in explicitly, so these functions are unit-testable.

Ray convention is NOT used here; this is the classic PD controller's lane finder, kept
separate from the polar-rays perception fed to the model.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneParams:
    cam_w: int = 512
    cam_h: int = 256
    roi_far: float = 0.65          # runtime-tunable in the PD controller
    roi_mid: float = 0.80
    roi_near: float = 0.92
    min_blob_area: int = 800
    min_corner_area: int = 6000
    track_width_px: int = 280      # real track width ~280px at cam_w=512
    slide_win: int = 70            # +/- px search window around the previous position


_D = LaneParams()


def get_blobs(mask, p: LaneParams = _D):
    """Retourne (accepted_blobs, rejected_blobs) — les rejetés servent uniquement à la visu orange."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cy_min     = int(p.cam_h * 0.44)
    cy_max     = int(p.cam_h * 0.97)
    y_bot_min  = int(p.cam_h * 0.62)
    # 0.10 : pieds de chaises (area<<800) déjà éliminés par min_blob_area,
    # les lignes en perspective ont w/h ~0.10-0.30 selon distance
    aspect_min = 0.10
    w_min      = 20
    blobs    = []
    rejected = []
    for i in range(1, n):
        area   = stats[i, cv2.CC_STAT_AREA]
        x      = stats[i, cv2.CC_STAT_LEFT]
        y_top  = stats[i, cv2.CC_STAT_TOP]
        w      = stats[i, cv2.CC_STAT_WIDTH]
        h      = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        aspect = w / float(h)
        cx     = x + w // 2
        cy     = y_top + stats[i, cv2.CC_STAT_HEIGHT] // 2
        y_bot  = y_top + stats[i, cv2.CC_STAT_HEIGHT]
        rect   = (x, y_top, w, h)

        if cy < cy_min or cy > cy_max:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "cy", "rect": rect})
            continue
        if y_bot < y_bot_min:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "ybot", "rect": rect})
            continue
        if area < p.min_blob_area:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "area", "rect": rect})
            continue
        if aspect < aspect_min:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "asp", "rect": rect})
            continue
        if w < w_min:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "w", "rect": rect})
            continue
        # Blobs compacts et pleins (logo/flèche au sol) — solidity haute + pas extrêmement allongé.
        # Les lignes de piste ont solidity faible (<0.50) car fines dans leur bounding box.
        bbox_area = w * h
        solidity = float(area) / max(bbox_area, 1)
        if area > 3000 and solidity > 0.65 and aspect < 2.5:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "cmp", "rect": rect})
            continue
        # Rayons de soleil / reflets horizontaux : très larges et peu hauts (w>>h).
        if aspect > 5.0 and area > 1500:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "horiz", "rect": rect})
            continue
        blobs.append({"cx": cx, "cy": cy, "area": area, "aspect": round(aspect, 1)})

    blobs.sort(key=lambda b: b["area"], reverse=True)
    if len(blobs) >= 2:
        left  = min(blobs, key=lambda b: b["cx"])
        right = max(blobs, key=lambda b: b["cx"])
        if left is not right:
            return [left, right], rejected
    return blobs, rejected


def err_from_mask(mask, p: LaneParams = _D):
    M = cv2.moments(mask)
    if M["m00"] < 1:
        return None
    return int(M["m10"] / M["m00"]) - p.cam_w // 2


def err_from_scanlines(mask, p: LaneParams = _D):
    """3 scanlines à FAR/MID/NEAR : centre entre les blancs extrêmes de chaque ligne,
    médiane des 3. Retourne (err, scan_points)."""
    rows = [int(p.cam_h * p.roi_far), int(p.cam_h * p.roi_mid), int(p.cam_h * p.roi_near)]
    centers = []
    scan_points = []
    for r in rows:
        r = min(r, p.cam_h - 1)
        line = mask[r, :]
        whites = np.where(line > 0)[0]
        if len(whites) < 5:
            continue
        left  = int(whites[0])
        right = int(whites[-1])
        if right - left < 20:   # trop étroit = bruit
            continue
        center = (left + right) // 2
        centers.append(center)
        scan_points.append((center, r))
    if not centers:
        return None, []
    median_c = sorted(centers)[len(centers) // 2]
    return median_c - p.cam_w // 2, scan_points


def err_from_bands(mask, p: LaneParams = _D):
    row_near = int(p.cam_h * p.roi_near)
    row_mid  = int(p.cam_h * p.roi_mid)
    row_far  = int(p.cam_h * p.roi_far)
    mask_near = mask.copy(); mask_near[:row_near, :] = 0
    mask_mid  = mask.copy(); mask_mid[row_near:, :] = 0; mask_mid[:row_mid, :] = 0
    mask_far  = mask.copy(); mask_far[row_mid:, :] = 0;  mask_far[:row_far, :] = 0
    return err_from_mask(mask_near, p), err_from_mask(mask_mid, p), err_from_mask(mask_far, p)


def clean_mask_artifacts(mask, bgr=None, p: LaneParams = _D):
    """Filtre les composantes qui ne sont pas des lignes (géométrie + bords Sobel).
    Retourne (mask_clean, rejected_mask)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean    = np.zeros_like(mask)
    rejected = np.zeros_like(mask)

    sobel_mag = None
    if bgr is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sx * sx + sy * sy)

    for i in range(1, n):
        area  = stats[i, cv2.CC_STAT_AREA]
        bw    = stats[i, cv2.CC_STAT_WIDTH]
        bh    = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        by    = stats[i, cv2.CC_STAT_TOP]
        y_bot = by + bh
        blob_mask = (labels == i)

        reason = None
        if area < 350:
            reason = "small"
        elif y_bot < int(p.cam_h * 0.50):
            # Bas du blob entièrement en moitié haute → mur/fond de salle.
            reason = "high"
        else:
            asp = float(max(bw, bh)) / max(min(bw, bh), 1)
            if asp < 1.8 and area < 1500:
                reason = "compact"  # carré compact petit → reflet, logo, chaussure
            elif sobel_mag is not None and area < 4000:
                # Sobel sur les blobs MOYENS : cible les artefacts diffus (mur lointain, reflet).
                kernel3 = np.ones((3, 3), np.uint8)
                blob_u8 = blob_mask.astype(np.uint8) * 255
                border  = cv2.dilate(blob_u8, kernel3) - blob_u8
                border_pixels = sobel_mag[border > 0]
                if len(border_pixels) > 0 and float(np.mean(border_pixels)) < 10.0:
                    reason = "diffuse"

        if reason is not None:
            rejected[blob_mask] = 255
        else:
            clean[blob_mask] = 255

    return clean, rejected


def find_lane_histogram(mask, prev_left=None, prev_right=None, p: LaneParams = _D):
    """Détecte les deux lignes par histogramme de colonnes (zone basse), sliding windows.
    Returns: (left_cx, right_cx, left_conf, right_conf)."""
    y_start = int(p.cam_h * 0.62)
    y_end   = int(p.cam_h * 0.97)
    roi = mask[y_start:y_end, :]

    hist = np.sum(roi.astype(np.float32), axis=0)

    # Lissage gaussien 21px : consolide le pic d'une même ligne blanche (~30-60px de large).
    k = 21
    sigma = k / 4.0
    xs = np.arange(-(k // 2), k // 2 + 1, dtype=np.float32)
    gauss = np.exp(-0.5 * (xs / sigma) ** 2)
    gauss /= gauss.sum()
    hist = np.convolve(hist, gauss, mode='same')

    mid = p.cam_w // 2
    HIST_MIN = 12.0 * 255.0      # au moins ~12px blancs dans la colonne

    left_half  = hist[:mid]
    right_half = hist[mid:]

    def _best_peak(half_hist, offset, prev_cx, peer_cx):
        n = len(half_hist)
        if prev_cx is not None:
            local_center = prev_cx - offset
            i_min = max(2, local_center - p.slide_win)
            i_max = min(n - 2, local_center + p.slide_win)
        else:
            i_min, i_max = 2, n - 2
        best_cx = None
        best_score = -1.0
        for i in range(i_min, i_max + 1):
            v = half_hist[i]
            if v < HIST_MIN:
                continue
            if not (v > half_hist[i - 1] and v > half_hist[i + 1]):
                continue
            cx = i + offset
            score = v
            if prev_cx is not None:
                score += max(0.0, 3000.0 - abs(cx - prev_cx) * 60.0)
            if peer_cx is not None:
                dist = abs(cx - peer_cx)
                score += max(0.0, 4000.0 - abs(dist - p.track_width_px) * 80.0)
            if score > best_score:
                best_score = score
                best_cx = cx
        if best_cx is None:
            sub = half_hist[i_min:i_max + 1]
            if len(sub) > 0 and float(np.max(sub)) >= HIST_MIN:
                best_cx = int(np.argmax(sub)) + i_min + offset
        return best_cx

    left_cx  = _best_peak(left_half,  0,   prev_left,  None)
    right_cx = _best_peak(right_half, mid, prev_right, left_cx)
    left_cx  = _best_peak(left_half,  0,   prev_left,  right_cx)

    left_peak  = float(left_half[left_cx]) if left_cx is not None else 0.0
    right_peak = float(right_half[right_cx - mid]) if right_cx is not None else 0.0

    return left_cx, right_cx, left_peak, right_peak


def find_lane_scanlines(mask, n_lines=6, p: LaneParams = _D):
    """Raycasts horizontaux pour le bord INTÉRIEUR de chaque ligne (depuis chaque moitié
    vers le centre, zone morte centrale). Returns (left_cx, right_cx, rows, left_hits, right_hits)."""
    mid_x = p.cam_w // 2
    MARGIN = 30
    MIN_WHITES = 4

    rows = [int(p.cam_h * (0.65 + i * (0.25 / max(n_lines - 1, 1)))) for i in range(n_lines)]

    left_xs, right_xs = [], []
    left_hits, right_hits = [], []

    for r in rows:
        r = min(r, p.cam_h - 1)
        line = mask[r, :]

        whites_l = np.where(line[:mid_x - MARGIN] > 0)[0]
        if len(whites_l) >= MIN_WHITES:
            hit_l = int(whites_l[-1])   # le plus à droite = bord intérieur
            if r >= 3:
                above = int(np.sum(mask[r - 3:r, max(0, hit_l - 3):hit_l + 4]))
                if above >= 3 * 255:
                    left_xs.append(hit_l); left_hits.append((hit_l, r))
            else:
                left_xs.append(hit_l); left_hits.append((hit_l, r))

        whites_r = np.where(line[mid_x + MARGIN:] > 0)[0]
        if len(whites_r) >= MIN_WHITES:
            hit_r = int(mid_x + MARGIN + whites_r[0])  # le plus à gauche = bord intérieur
            if r >= 3:
                above = int(np.sum(mask[r - 3:r, hit_r - 3:min(p.cam_w, hit_r + 4)]))
                if above >= 3 * 255:
                    right_xs.append(hit_r); right_hits.append((hit_r, r))
            else:
                right_xs.append(hit_r); right_hits.append((hit_r, r))

    left_cx  = int(np.median(left_xs))  if left_xs  else None
    right_cx = int(np.median(right_xs)) if right_xs else None

    return left_cx, right_cx, rows, left_hits, right_hits


def fuse_lane_estimates(hist_left, hist_right, scan_left, scan_right):
    """Fusionne histogramme + scanlines : moyenne si proches (<40px), sinon histogramme."""
    MAX_DIVERGENCE = 40

    def _fuse_one(h, s):
        if h is not None and s is not None:
            return int(round(0.5 * h + 0.5 * s)) if abs(h - s) <= MAX_DIVERGENCE else h
        return h if h is not None else s

    return _fuse_one(hist_left, scan_left), _fuse_one(hist_right, scan_right)


def detect_corner_blob(mask, p: LaneParams = _D):
    """Marqueur de coin L : blob compact (area >= min_corner_area, aspect < 1.5, bas d'image).
    Retourne dict {cx, cy, area, aspect} ou None."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cy_min = int(p.cam_h * 0.62)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        asp  = w / float(h)
        cy   = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
        cx   = stats[i, cv2.CC_STAT_LEFT] + w // 2
        if area >= p.min_corner_area and asp < 1.5 and cy >= cy_min:
            if best is None or area > best["area"]:
                best = {"cx": cx, "cy": cy, "area": area, "aspect": round(asp, 1)}
    return best


def err_from_two_lines(blobs, track_width=None, p: LaneParams = _D):
    mid_x = p.cam_w // 2
    CLEAR_LEFT  = mid_x - 76
    CLEAR_RIGHT = mid_x + 76
    tw_est = int(track_width) if track_width is not None else p.track_width_px
    left_blobs  = [b for b in blobs if b["cx"] < CLEAR_LEFT]
    right_blobs = [b for b in blobs if b["cx"] > CLEAR_RIGHT]
    left  = max(left_blobs,  key=lambda b: b["area"]) if left_blobs  else None
    right = max(right_blobs, key=lambda b: b["area"]) if right_blobs else None
    if left and right:
        center = (left["cx"] + right["cx"]) // 2
        return center - mid_x, right["cx"] - left["cx"]
    if left:
        est_right = left["cx"] + tw_est
        return (left["cx"] + est_right) // 2 - mid_x, tw_est
    if right:
        est_left = right["cx"] - tw_est
        return (est_left + right["cx"]) // 2 - mid_x, tw_est
    return None, None


def _selftest():
    p = LaneParams()
    mask = np.zeros((p.cam_h, p.cam_w), np.uint8)
    # Two near-vertical lane lines in the lower band (wide enough to pass the aspect gate).
    cv2.line(mask, (120, 130), (110, 250), 255, 16)
    cv2.line(mask, (400, 130), (410, 250), 255, 16)
    l, r, lc, rc = find_lane_histogram(mask, p=p)
    assert l is not None and r is not None and l < p.cam_w // 2 < r, (l, r)
    sl, sr, rows, lh, rh = find_lane_scanlines(mask, p=p)
    assert sl is not None and sr is not None and sl < sr
    blobs, rej = get_blobs(mask, p=p)
    assert len(blobs) == 2
    err, _ = err_from_scanlines(mask, p=p)
    assert err is not None and abs(err) < 60          # roughly centred
    fl, fr = fuse_lane_estimates(l, r, sl, sr)
    assert fl is not None and fr is not None
    print("lane_detect self-test OK (histogram + scanlines + blobs + fuse)")


if __name__ == "__main__":
    _selftest()

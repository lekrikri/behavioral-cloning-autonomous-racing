"""
visual_rays.py — Bridge image couleur OAK-D → raycasts visuels [0,1]

Pipeline (idée Mathieu / LeCrabe) :
  Image BGR → masque binaire (HSV ou Canny) → 20 raycasts visuels

Sémantique identique à depth_to_rays.py :
  ray = 0.0 → bord de piste très proche dans cette direction
  ray = 1.0 → piste libre / pas de bord détecté

Remplacement direct de DepthToRays dans inference_realcar.py :
  --perception-mode visual
"""

import numpy as np
import cv2


def white_line_mask(
    bgr: np.ndarray,
    mode:       str = "hsv",
    hsv_low:    tuple = (0,   0, 175),
    hsv_high:   tuple = (180, 50, 255),
    canny_low:  int = 50,
    canny_high: int = 150,
    morph_k:    int = 5,
    blur_k:     int = 3,
    use_clahe:  bool = False,
    min_area:   int = 400,
) -> np.ndarray:
    """Masque binaire des lignes blanches (uint8, 0/255), sans ROI.

    Source unique du masquage, partagée par VisualRays (production) et
    live_mask_oak.py (outil de dev) — ne pas dupliquer ailleurs.

    blur_k    : taille kernel Gaussian blur avant HSV (0 = désactivé)
    use_clahe : normalisation CLAHE du channel V (éclairage variable)
    min_area  : surface minimale (px) d'un blob pour être gardé (filtre bruit)
    """
    if blur_k > 1:
        bgr = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0)

    kernel = np.ones((morph_k, morph_k), np.uint8)

    if mode == "canny":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        m    = cv2.Canny(gray, canny_low, canny_high)
        m    = cv2.dilate(m, kernel, iterations=2)
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if use_clahe:
            clahe        = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        # Seuil V adaptatif : Otsu sur channel V + garde V_MIN fixe comme plancher
        v_channel = hsv[:, :, 2]
        otsu_thresh, _ = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v_min_adaptive = max(int(hsv_low[2]), min(int(otsu_thresh), 220))
        hsv_low_adapted = (hsv_low[0], hsv_low[1], v_min_adaptive)
        m = cv2.inRange(hsv, np.asarray(hsv_low_adapted, np.uint8), np.asarray(hsv_high, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=1)

    # Filtrer les petits blobs (bruit, reflets, artefacts)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        filtered = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        m = filtered

    return m


class VisualRays:
    """
    Convertit une image BGR (caméra couleur OAK-D) en vecteur de raycasts [0,1]
    en passant par un masque binaire (lignes blanches).

    Paramètres :
      img_width / img_height : résolution caméra couleur (ex: 512x256)
      fov_deg   : FOV horizontal caméra couleur OAK-D Lite (68.8° mesuré)
      n_rays    : nombre de raycasts (doit correspondre au modèle — 20 pour v18)
      row_band  : bande verticale ROI (fraction 0-1 de img_height)
      mode      : "hsv" (eclairage homogene) | "canny" (eclairage variable)
    """

    def __init__(
        self,
        img_width:  int   = 512,
        img_height: int   = 256,
        fov_deg:    float = 68.8,       # FOV couleur OAK-D Lite (calibré)
        n_rays:     int   = 20,
        row_band:   tuple = (0.50, 1.0),
        mode:       str   = "hsv",
        hsv_low:    tuple = (0,   0, 180),
        hsv_high:   tuple = (180, 50, 255),
        canny_low:  int   = 50,
        canny_high: int   = 150,
        morph_k:    int   = 3,
    ):
        self.W          = img_width
        self.H          = img_height
        self.mode       = mode
        self.hsv_low    = np.array(hsv_low,  dtype=np.uint8)
        self.hsv_high   = np.array(hsv_high, dtype=np.uint8)
        self.canny_low  = canny_low
        self.canny_high = canny_high
        self.morph_k    = morph_k
        self.n_rays     = n_rays

        self.row_start = int(img_height * row_band[0])
        self.row_end   = int(img_height * row_band[1])
        self.roi_h     = self.row_end - self.row_start

        # fov_deg de la caméra COULEUR (CAM_A), distinct du FOV depth (mono).
        # Fallback ; remplacé au runtime par getFov(CAM_A) — cf. set_fov.
        self.set_fov(fov_deg)

    def set_fov(self, fov_deg: float) -> None:
        """Recalcule les colonnes d'échantillonnage pour un FOV donné.

        Appelé au runtime avec calib.getFov(CAM_A) pour coller au capteur
        couleur réel plutôt qu'au 68.8° codé en dur.
        """
        self.fov_deg = fov_deg
        focal_px   = self.W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        angles_deg = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, self.n_rays)
        self.cols  = np.clip(
            (self.W / 2.0 + np.tan(np.deg2rad(angles_deg)) * focal_px).astype(int),
            0, self.W - 1,
        )

    # ── Masque binaire ─────────────────────────────────────────────────────────
    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        return white_line_mask(
            bgr, mode=self.mode, hsv_low=self.hsv_low, hsv_high=self.hsv_high,
            canny_low=self.canny_low, canny_high=self.canny_high, morph_k=self.morph_k,
        )

    # ── Conversion principale ──────────────────────────────────────────────────
    def __call__(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        bgr_frame : image BGR (H, W, 3) de la caméra couleur OAK-D
        retourne  : float32 array (n_rays,) dans [0.0, 1.0]
                    0.0 = bord de piste très proche | 1.0 = direction libre
        """
        mask = self._mask(bgr_frame)
        roi  = mask[self.row_start:self.row_end, :]   # (roi_h, W)

        rays = np.ones(self.n_rays, dtype=np.float32)  # défaut = libre

        for i, col in enumerate(self.cols):
            column = roi[:, col]          # pixels de la colonne dans la ROI
            whites = np.where(column > 0)[0]

            if len(whites) == 0:
                rays[i] = 1.0             # pas de bord détecté = direction libre
            else:
                # Premier bord depuis le bas (le plus proche en perspective)
                first_from_bottom = whites[-1]
                # Normaliser : bas de ROI (proche) → 0.0 | haut de ROI (loin) → 1.0
                rays[i] = 1.0 - float(first_from_bottom) / self.roi_h

        return rays

    def get_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Retourne le masque binaire pour affichage/debug."""
        return self._mask(bgr_frame)


# ── Utilitaire : créer le pipeline depthai couleur (3.x) ─────────────────────
def create_color_pipeline_v3(device, width: int = 512, height: int = 256):
    """Pipeline depthai 3.x — caméra couleur uniquement."""
    import depthai as dai
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q   = cam.requestOutput(
        (width, height), dai.ImgFrame.Type.BGR888p
    ).createOutputQueue(maxSize=2, blocking=False)
    return pipeline, q


# ── Test standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hsv", "canny"], default="hsv")
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    try:
        import depthai as dai
    except ImportError:
        print("depthai non installe")
        sys.exit(1)

    vr = VisualRays(img_width=args.width, img_height=args.height, mode=args.mode)
    print(f"VisualRays mode={args.mode} | {args.width}x{args.height} | 20 rays")
    print(f"Colonnes angles : {vr.cols}")

    device_info = None
    for d in dai.Device.getAllConnectedDevices():
        device_info = d
        break
    if device_info is None:
        print("Aucun device OAK-D")
        sys.exit(1)

    with dai.Device(device_info) as device:
        pipeline, q = create_color_pipeline_v3(device, args.width, args.height)
        pipeline.start()
        print("Flux actif — Q pour quitter")

        while True:
            frame_msg = q.get()
            bgr  = frame_msg.getCvFrame()
            rays = vr(bgr)
            mask = vr.get_mask(bgr)

            # Affichage overlay
            vis  = bgr.copy()
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            h, w = bgr.shape[:2]

            # Dessiner les 20 raycasts
            for i, (col, ray) in enumerate(zip(vr.cols, rays)):
                length = int(ray * h * 0.4)
                color  = (0, int(255 * ray), int(255 * (1 - ray)))  # vert=libre, rouge=proche
                cv2.line(vis, (col, h), (col, h - length), color, 2)
                cv2.circle(vis, (col, h - length), 2, color, -1)

            info = f"mode={args.mode} | rays min={rays.min():.2f} max={rays.max():.2f} mean={rays.mean():.2f}"
            cv2.putText(vis, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

            display = np.hstack([vis, mask_bgr])
            cv2.imshow("VisualRays | G-CAR-000", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cv2.destroyAllWindows()

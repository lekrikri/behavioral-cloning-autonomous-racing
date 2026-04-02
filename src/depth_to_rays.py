"""
depth_to_rays.py — Bridge OAK-D Lite depth map → raycasts virtuels [0,1].

Corrections vs approche naïve (recommandations Gemini + Grok) :
- ROI band (40%→60%) au lieu d'une ligne unique → robustesse au tangage
- Distance min par colonne → simule un vrai raycast (premier obstacle)
- Gestion NaN/0 (pixels sans disparité stéréo) → remplacement par max_dist
- Filtre < 100mm (bruit sol/réflexions proches) → ignorés
- Focal length depuis FOV réel → projection correcte
"""

import numpy as np


class DepthToRays:
    """
    Convertit une depth map OAK-D Lite (uint16, mm) en vecteur de raycasts [0,1].

    Paramètres à calibrer selon ton montage physique :
      - fov_deg     : FOV horizontal réel de l'OAK-D Lite (~97° en 400p)
      - row_band    : bande ROI verticale (robustesse au tangage)
      - max_dist_mm : distance max considérée (au-delà = piste libre)
    """

    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 400,
        fov_deg: float = 97.0,          # OAK-D Lite FOV horizontal réel ~97°
        n_rays: int = 20,
        max_dist_mm: float = 3500.0,    # 3.5m — piste RC indoor/outdoor courte
        row_band: tuple = (0.40, 0.62), # 40%→62% hauteur (anti-tangage)
        min_valid_mm: float = 100.0,    # ignorer < 10cm (bruit sol / réflexions)
    ):
        self.W = img_width
        self.H = img_height
        self.max_dist_mm = max_dist_mm
        self.min_valid_mm = min_valid_mm
        self.n_rays = n_rays
        self.row_start = int(img_height * row_band[0])
        self.row_end   = int(img_height * row_band[1])

        # Focal length horizontal depuis FOV réel (intrinsics approximatifs)
        # Pour calibration précise : utiliser depthai getIntrinsics()
        self.focal = img_width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))

        # Angles uniformes sur le FOV — pré-calculés une fois
        angles_deg = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_rays)

        # Colonnes pixel correspondantes — clippées dans [0, W-1]
        self.cols = np.clip(
            (self.W / 2.0 + np.tan(np.deg2rad(angles_deg)) * self.focal).astype(int),
            0, self.W - 1,
        )

    def __call__(self, depth_frame: np.ndarray) -> np.ndarray:
        """
        depth_frame : uint16 array (H, W), valeurs en mm (stream 'depth' depthai)
        retourne    : float32 array (n_rays,) dans [0.0, 1.0]
                      0.0 = obstacle très proche | 1.0 = dégagé / loin
        """
        roi = depth_frame[self.row_start:self.row_end, :].astype(np.float32)

        # Masquer valeurs invalides : 0 (pas de disparité), trop proches, trop loin
        roi[(roi == 0) | (roi < self.min_valid_mm) | (roi > self.max_dist_mm)] = np.nan

        # Distance MINIMUM par colonne dans la ROI
        # → reproduit le comportement d'un raycast physique (1er obstacle rencontré)
        # np.nanmin ignore les NaN ; si toute la colonne est NaN → max_dist (piste libre)
        with np.errstate(all="ignore"):
            col_min = np.nanmin(roi, axis=0)                      # (W,)
        col_min = np.where(np.isnan(col_min), self.max_dist_mm, col_min)

        # Échantillonner aux angles des raycasts
        sampled = col_min[self.cols]                              # (n_rays,)

        # Normaliser [0, 1] et clipper
        rays = np.clip(sampled / self.max_dist_mm, 0.0, 1.0).astype(np.float32)

        return rays


def create_depthai_pipeline():
    """
    Crée et retourne un pipeline depthai configuré pour la depth map.
    Config optimisée pour réduire le bruit stéréo OAK-D Lite.
    """
    import depthai as dai

    pipeline = dai.Pipeline()

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    stereo  = pipeline.create(dai.node.StereoDepth)
    xout    = pipeline.create(dai.node.XLinkOut)

    # Caméras mono gauche/droite (OAK-D Lite : CAM_B = gauche, CAM_C = droite)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(30)
    mono_r.setFps(30)

    # Stéréo : config haute densité + filtres anti-bruit
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)          # réduit fortement les artefacts stéréo
    stereo.setSubpixel(False)               # désactiver subpixel → moins de bruit
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)   # lissage spatial
    stereo.setConfidenceThreshold(230)      # ignorer pixels à faible confiance

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    stereo.depth.link(xout.input)           # depth en mm (uint16)
    xout.setStreamName("depth")

    return pipeline

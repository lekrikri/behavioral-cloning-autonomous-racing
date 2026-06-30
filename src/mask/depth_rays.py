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


def fov_to_cols(width: int, fov_deg: float, n_rays: int) -> np.ndarray:
    """Colonnes pixel échantillonnées uniformément en angle sur le FOV horizontal.

    Source unique de la projection, partagée par DepthToRays et VisualRays — ne
    pas dupliquer (sinon depth et visual dérivent et se désalignent en fusion).
    """
    focal = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    angles_deg = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_rays)
    return np.clip(
        (width / 2.0 + np.tan(np.deg2rad(angles_deg)) * focal).astype(int),
        0, width - 1,
    )


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

        # fov_deg est un fallback ; en production on le remplace par getFov()
        # lu sur la calibration usine (cf. set_fov / inference_realcar).
        self.set_fov(fov_deg)

    def set_fov(self, fov_deg: float) -> None:
        """Recalcule les colonnes d'échantillonnage pour un FOV donné.

        Appelé au runtime avec calib.getFov(CAM_B) pour coller au capteur réel
        plutôt qu'au ~97° approximatif codé en dur.
        """
        self.fov_deg = fov_deg
        self.cols = fov_to_cols(self.W, fov_deg, self.n_rays)

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
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            col_min = np.nanmin(roi, axis=0)                      # (W,)
        col_min = np.where(np.isnan(col_min), self.max_dist_mm, col_min)

        # Échantillonner aux angles des raycasts
        sampled = col_min[self.cols]                              # (n_rays,)

        # Normaliser [0, 1] et clipper
        rays = np.clip(sampled / self.max_dist_mm, 0.0, 1.0).astype(np.float32)

        return rays


def add_stereo_depth(pipeline, dai):
    """Ajoute mono L/R + StereoDepth (config anti-bruit OAK-D Lite) à un pipeline EXISTANT
    et retourne le nœud stereo (le caller lie `stereo.depth` à sa sortie). SOURCE UNIQUE de
    la config depth : réutilisée par le hub (cam/hub.py) ET create_depthai_pipeline ci-dessous,
    pour que DepthToRays (calibré sur cette config) reste valide quelle que soit la source.
    """
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

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
    return stereo


def create_depthai_pipeline():
    """Pipeline depthai autonome pour la depth map (mode --source device). Réutilise la
    config canonique add_stereo_depth() ; le hub fait de même."""
    import depthai as dai

    pipeline = dai.Pipeline()
    stereo = add_stereo_depth(pipeline, dai)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("depth")
    stereo.depth.link(xout.input)           # depth en mm (uint16)
    return pipeline

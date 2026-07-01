"""stereo_depth.py — Config StereoDepth canonique de l'OAK-D Lite.

Source unique de la config depth (mono L/R + StereoDepth anti-bruit), partagée par le
hub (src/cam/hub.py) et l'inférence en mode device. La depth est alignée sur la couleur
(CAM_A) et consommée comme FILTRE du masque (surfaces verticales, cf. white_line_mask) —
il n'y a plus de raycasts issus de la depth : la seule représentation de sortie est le
faisceau polaire (src/mask/polar_rays.py).

Nom de fichier conservé (depth_rays.py) pour ne pas casser les imports existants.
"""


def add_stereo_depth(pipeline, dai, align_socket=None, output_size=None):
    """Ajoute mono L/R + StereoDepth (config anti-bruit OAK-D Lite) à un pipeline EXISTANT
    et retourne le nœud stereo (le caller lie `stereo.depth` à sa sortie).

    align_socket : si fourni (ex. CameraBoardSocket.CAM_A), la depth est reprojetée dans le
                   repère de cette caméra → pixel-à-pixel avec la couleur. Requis par le filtre
                   « surfaces verticales » de white_line_mask, qui indexe la depth avec la
                   même grille que le masque couleur.
    output_size  : (w, h) — si fourni, force la résolution de sortie depth = taille couleur,
                   pour que le tableau depth ait exactement la shape du masque.
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

    if align_socket is not None:
        stereo.setDepthAlign(align_socket)  # depth dans le repère couleur (CAM_A)
    if output_size is not None:
        stereo.setOutputSize(int(output_size[0]), int(output_size[1]))

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    return stereo

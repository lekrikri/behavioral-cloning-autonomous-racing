"""calibrate_geometry.py — Régler la géométrie caméra (hauteur + pitch) du profil.

La projection IPM du faisceau polaire (src/mask/camera_ground.py) dépend de deux valeurs
de montage : la hauteur de la caméra au-dessus du sol et son angle de plongée (pitch). Cet
outil affiche EN DIRECT la distance métrique mesurée par les rayons, pour la caler contre
un repère posé au sol à distance connue.

Méthode :
  1. Poser un repère (bande adhésive) au sol, bien centré devant la voiture, à distance
     connue D (ex. 1.00 m depuis l'axe des roues avant).
  2. Lancer cet outil en testant plusieurs valeurs de pitch :
       OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.calibrate_geometry --pitch 18
     Lire la distance du rayon qui tape le repère (le plus proche, angle ≈ 0).
  3. Ajuster --pitch jusqu'à ce que la distance lue ≈ D. Vérifier --height au mètre ruban.
  4. Reporter les valeurs retenues dans configs/profiles/classic.json
     (cam_pitch_deg, cam_height_m).

Ne commande jamais le moteur ; lecture seule du hub.
"""

import argparse
import sys
import time

import numpy as np

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from src.mask.white_line import white_line_mask
from src.mask.perception_config import resolve_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None,
                    help="profil ciblé ; par défaut le profil actif (celui de la prod)")
    ap.add_argument("--pitch",  type=float, default=None, help="override cam_pitch_deg (test)")
    ap.add_argument("--height", type=float, default=None, help="override cam_height_m (test)")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--no-depth", action="store_true",
                    help="ignore la depth (recommandé pour caler la seule géométrie couleur→sol)")
    args = ap.parse_args()

    profile, prof_name = resolve_profile(args.profile)
    print("[géométrie] profil = %s" % prof_name)
    if args.pitch is not None:
        profile.cam_pitch_deg = args.pitch
    if args.height is not None:
        profile.cam_height_m = args.height
    mask_kw = profile.mask_kwargs()

    from src.cam.hub import FrameClient, SHM_COLOR, SHM_DEPTH, ensure_hub_or_prompt
    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    c_color = FrameClient(stream=SHM_COLOR)
    c_depth = FrameClient(stream=SHM_DEPTH)
    geom = None
    polar = None

    print("[géométrie] height=%.3f m | pitch=%.1f°  (Ctrl+C pour quitter)"
          % (profile.cam_height_m, profile.cam_pitch_deg))
    print("[géométrie] pose un repère au sol à distance connue, cale la distance lue dessus.\n")

    try:
        while True:
            bgr = c_color.getCvFrame()
            if args.no_depth:
                depth = None
            else:
                try:
                    depth = c_depth.latest()
                except (ConnectionError, OSError):
                    depth = None
            if geom is None or geom.H != bgr.shape[0] or geom.W != bgr.shape[1]:
                from src.mask.camera_ground import CameraGround
                geom = CameraGround(bgr.shape[1], bgr.shape[0], profile.fov_deg,
                                    profile.cam_height_m, profile.cam_pitch_deg)
                polar = profile.polar_rays(geom=geom)

            mask = white_line_mask(bgr, depth_mm=depth, geom=geom, **mask_kw)
            dist, ang = polar(mask)
            hits = np.where(dist < profile.max_range_m)[0]
            n = len(dist)
            center = float(dist[n // 2])
            if len(hits):
                i = int(hits[np.argmin(dist[hits])])
                near = "%.2f m @ %+.1f°" % (dist[i], np.degrees(ang[i]))
            else:
                near = "aucun (masque vide ?)"
            print("\r height=%.3f pitch=%.1f | rayon proche = %-18s | centre = %.2f m   "
                  % (profile.cam_height_m, profile.cam_pitch_deg, near, center),
                  end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[géométrie] fin.")
    finally:
        c_color.close(); c_depth.close()


if __name__ == "__main__":
    main()

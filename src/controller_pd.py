"""
controller_pd.py — Contrôleur autonome PD + lookahead + vitesse adaptative

Architecture 4 niveaux (décommenter/commenter selon les tests) :
  Niveau 1 : PD basique sur err_now
  Niveau 2 : Lookahead (err_now + err_ahead pondérés)
  Niveau 3 : Deux lignes séparées (centre piste réel)
  Niveau 4 : Machine à états vitesse (droite / virage / récup / urgence)

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --port /dev/ttyACM0
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --port /dev/ttyACM0 --dry-run

  --dry-run : vision seule, VESC non commandé (tuning sécurisé)
  --level N : niveau contrôleur 1-4 (défaut : 2)
"""

import sys, time, argparse, threading, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from visual_rays import white_line_mask, VisualRays
from vesc_interface import VescInterface

try:
    import depthai as dai
except ImportError:
    print("[ctrl] depthai non installe")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES — ajuster pour le tuning
# ══════════════════════════════════════════════════════════════════════════════

# ── Caméra ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 512, 256
CAM_FPS      = 12

# ── Vision ────────────────────────────────────────────────────────────────────
HSV_LOW      = np.array([0,   0, 195], dtype=np.uint8)
HSV_HIGH     = np.array([180, 40, 255], dtype=np.uint8)
ROI_BOTTOM   = 1.00          # bande bas (proche voiture)
ROI_MID      = 0.55          # séparation now / ahead
ROI_TOP      = 0.35          # bande haute (loin, lookahead)
MIN_BLOB_AREA = 400          # px — filtre petits artefacts

# ── Contrôleur PD ─────────────────────────────────────────────────────────────
KP           = 0.004         # gain proportionnel (err px → steering [-1,1])
KD           = 0.002         # gain dérivé
W_NOW        = 0.35          # poids erreur zone basse  (Niveau 2)
W_AHEAD      = 0.65          # poids erreur zone haute  (Niveau 2)
STEERING_MAX = 0.8           # saturation steering (protège la mécanique)
STEERING_DEADZONE = 0.03     # zone morte (évite micro-oscillations)

# ── Vitesse ───────────────────────────────────────────────────────────────────
V_MAX        = 0.30          # duty cycle ligne droite (0-1)
V_TURN       = 0.18          # duty cycle virage
V_SLOW       = 0.12          # duty cycle récupération (1 seule ligne)
V_STOP       = 0.00          # arrêt d'urgence (0 ligne)
CURVE_THRESH_HIGH = 0.30     # std(rays) > seuil → virage
CURVE_THRESH_LOW  = 0.15     # std(rays) < seuil → ligne droite (hystérésis)

# ── VESC ─────────────────────────────────────────────────────────────────────
CURRENT_MAX  = 5.0           # ampères max (augmenter progressivement)


# ══════════════════════════════════════════════════════════════════════════════
# VISION
# ══════════════════════════════════════════════════════════════════════════════

def get_blobs(mask):
    """Retourne les composantes connexes triées par taille décroissante."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BLOB_AREA:
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            cy = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
            blobs.append({"cx": cx, "cy": cy, "area": area, "label": i})
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs


def err_from_mask(mask):
    """Erreur latérale depuis le centroïde global (Niveau 1)."""
    M = cv2.moments(mask)
    if M["m00"] < 1:
        return None
    return int(M["m10"] / M["m00"]) - CAM_W // 2


def err_from_bands(mask):
    """
    Erreur now (bas) et ahead (haut) pour le lookahead (Niveau 2).
    Retourne (err_now, err_ahead) — None si pas de ligne dans la bande.
    """
    row_mid   = int(CAM_H * ROI_MID)
    row_top   = int(CAM_H * ROI_TOP)

    mask_now   = mask.copy(); mask_now[:row_mid, :]  = 0
    mask_ahead = mask.copy(); mask_ahead[row_mid:, :] = 0; mask_ahead[:row_top, :] = 0

    err_now   = err_from_mask(mask_now)
    err_ahead = err_from_mask(mask_ahead)
    return err_now, err_ahead


def err_from_two_lines(blobs):
    """
    Centre de piste via ligne gauche + droite (Niveau 3).
    Retourne l'erreur et la largeur détectée, ou None si < 2 blobs.
    """
    if len(blobs) < 2:
        return None, None
    left  = min(blobs[:2], key=lambda b: b["cx"])
    right = max(blobs[:2], key=lambda b: b["cx"])
    center = (left["cx"] + right["cx"]) // 2
    width  = right["cx"] - left["cx"]
    return center - CAM_W // 2, width


# ══════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ══════════════════════════════════════════════════════════════════════════════

class PDController:
    def __init__(self, level=2):
        self.level    = level
        self.prev_err = 0.0
        self.state    = "STOP"   # STRAIGHT / TURN / RECOVER / STOP
        self.vr       = VisualRays(
            img_width=CAM_W, img_height=CAM_H,
            row_band=(ROI_TOP, ROI_BOTTOM),
            morph_k=5,
        )

    def _pd(self, err):
        """PD pur → steering [-1, 1]."""
        d_err     = err - self.prev_err
        self.prev_err = err
        raw       = KP * err + KD * d_err
        raw       = max(-STEERING_MAX, min(STEERING_MAX, raw))
        if abs(raw) < STEERING_DEADZONE:
            raw = 0.0
        return raw

    def _update_state(self, rays, n_blobs):
        """Machine à états vitesse (Niveau 4)."""
        curvature = float(np.std(rays))
        if n_blobs == 0:
            self.state = "STOP"
        elif n_blobs == 1:
            self.state = "RECOVER"
        elif curvature > CURVE_THRESH_HIGH:
            self.state = "TURN"
        elif curvature < CURVE_THRESH_LOW and self.state == "TURN":
            self.state = "STRAIGHT"   # hystérésis : ne sort de TURN que si vraiment droit
        elif self.state not in ("TURN", "STRAIGHT"):
            self.state = "STRAIGHT"

    def _speed(self):
        return {
            "STRAIGHT": V_MAX,
            "TURN":     V_TURN,
            "RECOVER":  V_SLOW,
            "STOP":     V_STOP,
        }.get(self.state, V_STOP)

    def compute(self, mask, bgr):
        """
        Entrée : masque binaire + frame BGR
        Sortie : (steering [-1,1], throttle [0,1], info_dict)
        """
        rays   = self.vr(bgr)
        blobs  = get_blobs(mask)
        n_blobs = len(blobs)

        # ── Erreur latérale selon niveau ──────────────────────────────────────
        err = None

        if self.level >= 3:
            err, width = err_from_two_lines(blobs)

        if err is None and self.level >= 2:
            err_now, err_ahead = err_from_bands(mask)
            if err_now is not None and err_ahead is not None:
                err = W_NOW * err_now + W_AHEAD * err_ahead
            elif err_now is not None:
                err = err_now
            elif err_ahead is not None:
                err = err_ahead

        if err is None and self.level >= 1:
            err = err_from_mask(mask)

        # ── Vitesse (Niveau 4 si level>=4, sinon adaptatif simple) ───────────
        if self.level >= 4:
            self._update_state(rays, n_blobs)
            throttle = self._speed()
        else:
            curvature = float(np.std(rays))
            if n_blobs == 0:
                throttle = V_STOP
            elif n_blobs == 1:
                throttle = V_SLOW
            else:
                throttle = V_MAX * (1.0 - min(curvature / CURVE_THRESH_HIGH, 1.0))
                throttle = max(V_TURN, throttle)

        # ── Steering ──────────────────────────────────────────────────────────
        if err is None:
            steering = 0.0
            throttle = V_STOP
        else:
            steering = self._pd(float(err))

        info = {
            "err":      err,
            "steering": steering,
            "throttle": throttle,
            "state":    self.state,
            "n_blobs":  n_blobs,
            "curvature": float(np.std(rays)),
        }
        return steering, throttle, info


# ══════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",    default="/dev/ttyACM0")
    p.add_argument("--baud",    type=int, default=115200)
    p.add_argument("--level",   type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--dry-run", action="store_true",
                   help="Vision seule — VESC non commandé")
    p.add_argument("--show",    action="store_true",
                   help="Afficher overlay (necessite display)")
    return p.parse_args()


def run(args):
    ctrl = PDController(level=args.level)

    # ── VESC ─────────────────────────────────────────────────────────────────
    vesc = None
    if not args.dry_run:
        vesc = VescInterface(port=args.port, baudrate=args.baud,
                             current_max=CURRENT_MAX)
        vesc.start()
        print("[ctrl] VESC connecté sur {}".format(args.port))
    else:
        print("[ctrl] DRY-RUN — VESC non commandé")

    # ── Pipeline OAK-D ───────────────────────────────────────────────────────
    attempt = 0
    while True:
        try:
            attempt += 1
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(CAM_W, CAM_H)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(CAM_FPS)
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("preview")
            cam.preview.link(xout.input)

            with dai.Device(pipeline, True) as device:
                q = device.getOutputQueue("preview", maxSize=1, blocking=False)
                print("[ctrl] Niveau {} | {}x{} @ {}fps".format(
                    args.level, CAM_W, CAM_H, CAM_FPS))
                attempt = 0

                t0 = time.time()
                frame_n = 0

                while True:
                    pkt = q.get()
                    bgr = pkt.getCvFrame()

                    mask = white_line_mask(
                        bgr, hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
                        morph_k=5, blur_k=3, use_clahe=True, min_area=MIN_BLOB_AREA,
                    )
                    roi_top_px = int(CAM_H * ROI_TOP)
                    mask[:roi_top_px, :] = 0

                    steering, throttle, info = ctrl.compute(mask, bgr)

                    # ── Envoi VESC ────────────────────────────────────────────
                    if vesc is not None:
                        vesc.set_steering(steering)
                        vesc.set_throttle(throttle)

                    frame_n += 1
                    if frame_n % (CAM_FPS * 5) == 0:
                        fps = frame_n / (time.time() - t0)
                        print("[ctrl] {:.0f}fps | err={} | steer={:.3f} | "
                              "throttle={:.2f} | state={} | blobs={}".format(
                                  fps,
                                  info["err"] if info["err"] is not None else "N/A",
                                  info["steering"], info["throttle"],
                                  info["state"], info["n_blobs"]))

                    if args.show:
                        vis = bgr.copy()
                        green = np.zeros_like(vis)
                        green[:, :, 1] = mask
                        vis = cv2.addWeighted(vis, 1.0, green, 0.6, 0)
                        cv2.putText(vis,
                            "L{} steer={:.2f} thr={:.2f} {}".format(
                                args.level, steering, throttle, info["state"]),
                            (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.imshow("Controller", vis)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("[ctrl] Arret.")
            break
        except Exception as e:
            if vesc:
                vesc.set_throttle(0.0)
            delay = min(3 * attempt, 15)
            print("[ctrl] Erreur OAK-D ({}) — reconnexion dans {}s".format(
                type(e).__name__, delay))
            time.sleep(delay)

    if vesc:
        vesc.set_throttle(0.0)
        vesc.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

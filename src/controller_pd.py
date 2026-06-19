"""
controller_pd.py — Contrôleur autonome PD + lookahead 3 bandes + séparation L/R

Architecture 4 niveaux :
  Niveau 1 : PD basique sur err_now
  Niveau 2 : Lookahead 3 bandes dynamiques (near/mid/far pondérées par forward_clearance)
  Niveau 3 : Séparation ligne gauche / droite (reconstruction si une seule visible)
  Niveau 4 : Machine à états vitesse (droite / virage / récup / urgence)

Améliorations v2 :
  - Dérivée filtrée (alpha=0.7) → élimine le bruit vision
  - 3 bandes lookahead (near/mid/far) au lieu de 2
  - Lookahead dynamique : poids far augmente avec la visibilité devant (rays centraux)
  - Séparation L/R par moitiés d'image → reconstruction si une seule ligne visible
  - Vitesse adaptative via forward_clearance (rays[8:12])

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --port /dev/ttyACM0
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --port /dev/ttyACM0 --dry-run

  --dry-run : vision seule, VESC non commandé (tuning sécurisé)
  --level N : niveau contrôleur 1-4 (défaut : 2)
"""

import sys, time, argparse, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from visual_rays import white_line_mask, VisualRays
try:
    from vesc_interface import VESCInterface as VescInterface
except ImportError:
    VescInterface = None

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
ROI_FAR      = 0.45          # bande haute (loin, lookahead)
ROI_MID      = 0.55          # bande milieu
ROI_NEAR     = 0.75          # bande basse proche (nouvelle)
ROI_BOTTOM   = 1.00          # bas de l'image
MIN_BLOB_AREA = 400          # px — filtre petits artefacts

# Largeur de piste estimée en pixels (60-80 cm réels sur 512px de large)
# Ajuster selon la distance caméra / hauteur de montage
TRACK_WIDTH_EST_PX = 385   # mesuré en live : cx≈[66,435] → 369-416px selon frame

# ── Contrôleur PD ─────────────────────────────────────────────────────────────
KP           = 0.004         # gain proportionnel (err px → steering [-1,1])
KD           = 0.002         # gain dérivé (sur dérivée filtrée)
ALPHA_D      = 0.7           # lissage dérivée (0=brut, 1=figé) — élimine bruit vision
STEERING_MAX = 0.8           # saturation steering (protège la mécanique)
STEERING_DEADZONE = 0.03     # zone morte (évite micro-oscillations)

# ── Vitesse ───────────────────────────────────────────────────────────────────
V_MAX        = 0.40          # duty cycle ligne droite (→ 20% duty réel avec max_duty=0.50)
V_TURN       = 0.28          # duty cycle virage      (→ 14% duty réel)
V_SLOW       = 0.20          # duty cycle récupération (→ 10% duty réel)
V_STOP       = 0.00          # arrêt d'urgence (0 ligne)
CURVE_THRESH_HIGH = 0.30     # std(rays) > seuil → virage
CURVE_THRESH_LOW  = 0.15     # std(rays) < seuil → ligne droite (hystérésis)

# ── VESC ─────────────────────────────────────────────────────────────────────
CURRENT_MAX  = 5.0           # ampères max (augmenter progressivement)


# ══════════════════════════════════════════════════════════════════════════════
# VISION
# ══════════════════════════════════════════════════════════════════════════════

def get_blobs(mask):
    """Composantes connexes triées par taille décroissante."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BLOB_AREA:
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            cy = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
            blobs.append({"cx": cx, "cy": cy, "area": area})
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
    Erreur sur 3 bandes verticales : near / mid / far (Niveau 2).
    Retourne (err_near, err_mid, err_far) — None si pas de ligne dans la bande.
    """
    row_near = int(CAM_H * ROI_NEAR)
    row_mid  = int(CAM_H * ROI_MID)
    row_far  = int(CAM_H * ROI_FAR)

    mask_near = mask.copy()
    mask_near[:row_near, :] = 0

    mask_mid = mask.copy()
    mask_mid[row_near:, :] = 0
    mask_mid[:row_mid, :]  = 0

    mask_far = mask.copy()
    mask_far[row_mid:, :] = 0
    mask_far[:row_far, :] = 0

    return err_from_mask(mask_near), err_from_mask(mask_mid), err_from_mask(mask_far)


def err_from_two_lines(blobs):
    """
    Centre de piste via ligne gauche + droite (Niveau 3).
    Séparation par moitié d'image — reconstruction si une seule ligne visible.
    Retourne (err, track_width) ou (None, None).
    """
    mid_x = CAM_W // 2

    left_blobs  = [b for b in blobs if b["cx"] < mid_x]
    right_blobs = [b for b in blobs if b["cx"] >= mid_x]

    left  = max(left_blobs,  key=lambda b: b["area"]) if left_blobs  else None
    right = max(right_blobs, key=lambda b: b["area"]) if right_blobs else None

    if left and right:
        center = (left["cx"] + right["cx"]) // 2
        width  = right["cx"] - left["cx"]
        return center - mid_x, width

    if left:
        # Reconstruction côté droit depuis la ligne gauche
        est_right = left["cx"] + TRACK_WIDTH_EST_PX
        center = (left["cx"] + est_right) // 2
        return center - mid_x, TRACK_WIDTH_EST_PX

    if right:
        # Reconstruction côté gauche depuis la ligne droite
        est_left = right["cx"] - TRACK_WIDTH_EST_PX
        center = (est_left + right["cx"]) // 2
        return center - mid_x, TRACK_WIDTH_EST_PX

    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ══════════════════════════════════════════════════════════════════════════════

class PDController:
    def __init__(self, level=2):
        self.level      = level
        self.prev_err   = 0.0
        self.d_filtered = 0.0    # dérivée lissée
        self.state      = "STOP"
        self.vr         = VisualRays(
            img_width=CAM_W, img_height=CAM_H,
            row_band=(ROI_FAR, ROI_BOTTOM),
            morph_k=5,
        )

    def _pd(self, err):
        """PD avec dérivée filtrée → steering [-1, 1]."""
        d_raw = err - self.prev_err
        self.d_filtered = ALPHA_D * self.d_filtered + (1.0 - ALPHA_D) * d_raw
        self.prev_err = err

        raw = KP * err + KD * self.d_filtered
        raw = max(-STEERING_MAX, min(STEERING_MAX, raw))
        if abs(raw) < STEERING_DEADZONE:
            raw = 0.0
        return raw

    def _combined_err(self, err_near, err_mid, err_far, rays):
        """
        Combine les 3 bandes avec poids dynamiques.
        Plus la voie devant est libre (rays centraux élevés), plus on regarde loin.
        """
        forward_clearance = float(np.mean(rays[8:12]))

        # w_far : 0.2 (virage serré) → 0.6 (ligne droite)
        w_far  = 0.2 + 0.4 * forward_clearance
        w_mid  = 0.30
        w_near = max(0.0, 1.0 - w_far - w_mid)

        pairs = [
            (err_near, w_near),
            (err_mid,  w_mid),
            (err_far,  w_far),
        ]
        valid = [(e, w) for e, w in pairs if e is not None]
        if not valid:
            return None

        total_w = sum(w for _, w in valid)
        return sum(e * w for e, w in valid) / total_w

    def _update_state(self, rays, n_blobs):
        """Machine à états vitesse (Niveau 4)."""
        curvature = float(np.std(rays))
        if n_blobs == 0:
            self.state = "STOP"
        elif n_blobs == 1 and self.state not in ("RECOVER",):
            self.state = "RECOVER"
        elif curvature > CURVE_THRESH_HIGH:
            self.state = "TURN"
        elif curvature < CURVE_THRESH_LOW and self.state == "TURN":
            self.state = "STRAIGHT"
        elif self.state not in ("TURN", "STRAIGHT"):
            self.state = "STRAIGHT"

    def _speed(self, forward_clearance=None):
        """Vitesse selon état + visibilité devant."""
        if self.state == "STOP":
            return V_STOP
        if self.state == "RECOVER":
            return V_SLOW
        if self.state == "TURN":
            return V_TURN
        # STRAIGHT : speed proportionnelle à la visibilité (0.6→1.0 de V_MAX)
        if forward_clearance is not None:
            factor = 0.6 + 0.4 * float(np.clip(forward_clearance, 0.0, 1.0))
            return V_MAX * factor
        return V_MAX

    def compute(self, mask, bgr):
        """
        Entrée : masque binaire + frame BGR
        Sortie : (steering [-1,1], throttle [0,1], info_dict)
        """
        rays    = self.vr(bgr)
        blobs   = get_blobs(mask)
        n_blobs = len(blobs)

        forward_clearance = float(np.mean(rays[8:12]))
        err = None

        # ── Niveau 3 : séparation L/R ────────────────────────────────────────
        if self.level >= 3:
            err, _ = err_from_two_lines(blobs)

        # ── Niveau 2 : lookahead 3 bandes dynamique ──────────────────────────
        if err is None and self.level >= 2:
            err_near, err_mid, err_far = err_from_bands(mask)
            err = self._combined_err(err_near, err_mid, err_far, rays)

        # ── Niveau 1 : centroïde global ───────────────────────────────────────
        if err is None and self.level >= 1:
            err = err_from_mask(mask)

        # ── Vitesse (Niveau 4 ou adaptatif simple) ───────────────────────────
        if self.level >= 4:
            self._update_state(rays, n_blobs)
            throttle = self._speed(forward_clearance)
        else:
            if n_blobs == 0:
                self.state = "STOP"
                throttle = V_STOP
            elif n_blobs == 1:
                self.state = "RECOVER"
                throttle = V_SLOW
            else:
                # forward_clearance = mean(rays[8:12]) : 1.0 = droit devant libre
                # plus fiable que std(rays) qui est toujours élevé (lignes sur les bords)
                throttle = V_TURN + (V_MAX - V_TURN) * forward_clearance
                self.state = "TURN" if forward_clearance < 0.5 else "STRAIGHT"

        # ── Steering ──────────────────────────────────────────────────────────
        if err is None:
            steering = 0.0
            throttle = V_STOP
            self.state = "STOP"
        else:
            steering = self._pd(float(err))

        info = {
            "err":               err,
            "steering":          steering,
            "throttle":          throttle,
            "state":             self.state,
            "n_blobs":           n_blobs,
            "curvature":         float(np.std(rays)),
            "forward_clearance": forward_clearance,
            "d_filtered":        self.d_filtered,
            "blobs_cx":          [b["cx"] for b in blobs[:4]],
            "blobs_area":        [b["area"] for b in blobs[:4]],
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

    vesc = None
    if not args.dry_run:
        if VescInterface is None:
            print("[ctrl] ERREUR : vesc_interface non disponible")
            sys.exit(1)
        vesc = VescInterface(port=args.port, baudrate=args.baud,
                             current_max=CURRENT_MAX,
                             throttle_mode="duty",
                             max_duty=0.50)
        print("[ctrl] VESC connecte sur {}".format(args.port))
    else:
        print("[ctrl] DRY-RUN — VESC non commande")

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
                print("[ctrl] Niveau {} | {}x{} @ {}fps | alpha_d={} | TRACK_W={}px".format(
                    args.level, CAM_W, CAM_H, CAM_FPS, ALPHA_D, TRACK_WIDTH_EST_PX))
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
                    roi_top_px = int(CAM_H * ROI_FAR)
                    mask[:roi_top_px, :] = 0

                    steering, throttle, info = ctrl.compute(mask, bgr)

                    if vesc is not None:
                        vesc.drive(steering, throttle)

                    frame_n += 1
                    if frame_n % (CAM_FPS * 2) == 0:
                        fps = frame_n / (time.time() - t0)
                        # Calcul track width si 2 blobs distincts
                        cx_list = info["blobs_cx"]
                        mid = CAM_W // 2
                        lefts  = [x for x in cx_list if x < mid]
                        rights = [x for x in cx_list if x >= mid]
                        tw_str = "?"
                        if lefts and rights:
                            tw_str = str(min(rights) - max(lefts))
                        print("[ctrl] {:.0f}fps | err={} | steer={:.3f} | "
                              "thr={:.2f} | state={} | blobs={} | fwd={:.2f} | "
                              "cx={} | track_w={}px".format(
                                  fps,
                                  info["err"] if info["err"] is not None else "N/A",
                                  info["steering"], info["throttle"],
                                  info["state"], info["n_blobs"],
                                  info["forward_clearance"],
                                  cx_list, tw_str))

                    if args.show:
                        vis = bgr.copy()
                        green = np.zeros_like(vis)
                        green[:, :, 1] = mask
                        vis = cv2.addWeighted(vis, 1.0, green, 0.6, 0)
                        # Lignes de bandes
                        for frac, color in [(ROI_NEAR, (255,200,0)), (ROI_MID, (0,200,255)), (ROI_FAR, (0,100,255))]:
                            y = int(CAM_H * frac)
                            cv2.line(vis, (0, y), (CAM_W, y), color, 1)
                        cv2.putText(vis,
                            "L{} s={:.2f} t={:.2f} {} fwd={:.2f}".format(
                                args.level, steering, throttle,
                                info["state"], info["forward_clearance"]),
                            (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                        cv2.imshow("Controller", vis)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("[ctrl] Arret.")
            break
        except Exception as e:
            if vesc:
                vesc.stop()
            delay = min(3 * attempt, 15)
            print("[ctrl] Erreur OAK-D ({}) — reconnexion dans {}s".format(
                type(e).__name__, delay))
            time.sleep(delay)

    if vesc:
        vesc.stop()
        vesc.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

"""
controller_pd.py — Contrôleur PD + stream MJPEG intégré (port 5601)

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --level 3
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --dry-run
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/controller_pd.py --fixed-speed 0.22

  --dry-run      : vision seule, VESC non commandé
  --fixed-speed  : vitesse constante (bypass machine à états) — mode calibration
  --level N      : niveau contrôleur 1-4 (défaut : 3)
  --stream-port  : port HTTP stream MJPEG (défaut : 5601, 0 = désactivé)
"""

import sys, time, argparse, os, threading, struct, socket, csv, glob, fcntl, math
import numpy as np
import cv2

from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

sys.path.insert(0, os.path.dirname(__file__))
from visual_rays import white_line_mask, VisualRays
try:
    from track_mapper    import TrackMapper
    from track_navigator import TrackNavigator
    _TRACK_MODULES_OK = True
except ImportError:
    _TRACK_MODULES_OK = False
try:
    from vesc_interface import VESCInterface as VescInterface
except ImportError:
    VescInterface = None

def _find_oak_sysfs():
    """Retourne (dir_path, dev_name) du device OAK-D dans sysfs, ou (None, None)."""
    for vendor_path in glob.glob('/sys/bus/usb/devices/*/idVendor'):
        try:
            with open(vendor_path) as f:
                if f.read().strip() != '03e7':
                    continue
        except Exception:
            continue
        dir_path = os.path.dirname(vendor_path)
        dev_name = os.path.basename(dir_path)
        return dir_path, dev_name
    return None, None


def _usb_reset_method1_authorized():
    """Méthode 1 : authorized 0→1 (soft power cycle)."""
    dir_path, _ = _find_oak_sysfs()
    if dir_path is None:
        return False
    auth = os.path.join(dir_path, 'authorized')
    if not os.path.exists(auth):
        return False
    try:
        with open(auth, 'w') as f: f.write('0')
        print("[ctrl] USB [1] authorized=0")
        time.sleep(4)
        with open(auth, 'w') as f: f.write('1')
        print("[ctrl] USB [1] authorized=1")
        time.sleep(5)
        return True
    except Exception as e:
        print("[ctrl] USB [1] err: {}".format(e))
    return False


def _usb_reset_method2_unbind_bind():
    """Méthode 2 : unbind + rebind driver USB (plus agressif)."""
    _, dev_name = _find_oak_sysfs()
    if dev_name is None:
        # Device peut ne plus être listé après crash — chercher dans unbind quand même
        return False
    try:
        with open('/sys/bus/usb/drivers/usb/unbind', 'w') as f:
            f.write(dev_name)
        print("[ctrl] USB [2] unbind {}".format(dev_name))
        time.sleep(5)
        with open('/sys/bus/usb/drivers/usb/bind', 'w') as f:
            f.write(dev_name)
        print("[ctrl] USB [2] bind {}".format(dev_name))
        time.sleep(6)
        return True
    except Exception as e:
        print("[ctrl] USB [2] err: {}".format(e))
    return False


def _usb_reset_method3_ioctl():
    """Méthode 3 : ioctl USBDEVFS_RESET (reset électrique bas niveau)."""
    USBDEVFS_RESET = 0x5514
    dir_path, _ = _find_oak_sysfs()
    if dir_path is None:
        return False
    try:
        with open(os.path.join(dir_path, 'busnum')) as f:
            bus = int(f.read().strip())
        with open(os.path.join(dir_path, 'devnum')) as f:
            dev = int(f.read().strip())
        dev_path = '/dev/bus/usb/{:03d}/{:03d}'.format(bus, dev)
        with open(dev_path, 'wb') as fd:
            fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        print("[ctrl] USB [3] ioctl reset {}".format(dev_path))
        time.sleep(4)
        return True
    except Exception as e:
        print("[ctrl] USB [3] err: {}".format(e))
    return False


# Compteur global pour alterner les méthodes de reset
_reset_attempt_total = 0


def _usb_reset_method4_parent_hub():
    """Méthode 4 : unbind/rebind du hub parent 1-2.1 (power cycle logique complet)."""
    # 1-2.1 = hub intermédiaire au-dessus de l'OAK-D (ne porte ni WiFi ni VESC)
    HUB_NAME = '1-2.1'
    hub_driver = '/sys/bus/usb/drivers/usb'
    try:
        with open(os.path.join(hub_driver, 'unbind'), 'w') as f:
            f.write(HUB_NAME)
        print("[ctrl] USB [4] hub {} unbind".format(HUB_NAME))
        time.sleep(3.0)
        with open(os.path.join(hub_driver, 'bind'), 'w') as f:
            f.write(HUB_NAME)
        print("[ctrl] USB [4] hub {} bind".format(HUB_NAME))
        time.sleep(5.0)
        return True
    except Exception as e:
        print("[ctrl] USB [4] err: {}".format(e))
    return False


def _usb_reset_oak():
    """Escalade automatique des méthodes de reset USB selon le nombre d'échecs."""
    global _reset_attempt_total
    _reset_attempt_total += 1
    n = _reset_attempt_total
    print("[ctrl] USB reset OAK-D (tentative {})".format(n))
    # Escalade : 1 → 2 → 4(hub) → 3 → cycle
    mod = n % 4
    if mod == 1:
        ok = _usb_reset_method1_authorized()
    elif mod == 2:
        ok = _usb_reset_method2_unbind_bind()
    elif mod == 3:
        ok = _usb_reset_method4_parent_hub()
    else:
        ok = _usb_reset_method3_ioctl()
    if not ok:
        for fn in [_usb_reset_method1_authorized, _usb_reset_method2_unbind_bind,
                   _usb_reset_method4_parent_hub, _usb_reset_method3_ioctl]:
            if fn():
                break
    return True


try:
    import depthai as dai
except ImportError:
    print("[ctrl] depthai non installe")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CAM_W, CAM_H = 640, 320
CAM_FPS      = 15

HSV_LOW      = np.array([0,   0, 150], dtype=np.uint8)   # V>=150 (adapté éclairage faible)
HSV_HIGH     = np.array([180, 45, 255], dtype=np.uint8)  # S<=45 (blanc incluant reflets tamisés)
ROI_FAR      = 0.65
ROI_MID      = 0.80
ROI_NEAR     = 0.92
ROI_BOTTOM   = 1.00
MIN_BLOB_AREA  = 1250   # prop. 800 * 640*320/(512*256)
MIN_CORNER_AREA = 9375  # prop. 6000 * 1.5625
CORNER_DURATION = 32   # frames de maintien virage (~2.5s @ 13fps) — virages serrés piste V
U_DETECT_ANGLE  = 1.35  # rad (~77°) → début détection virage en U
U_GYRO_ACCUM    = 2.20  # rad accumulés → U confirmé
U_ERR_FORCE     = 250.0 # erreur forcée en U (vs 220 virage simple)
U_THROTTLE      = 0.35  # throttle U (vs 0.50 virage simple)
U_CORNER_MAX    = 50    # frames max U-turn (vs 32 virage simple)
U_EXIT_FADE     = 8     # frames fading sortie U (vs 4 virage simple)
U_SEARCH_FRAMES = 18    # frames SEARCH post-U (vs 10)

TRACK_WIDTH_EST_PX = 350     # largeur réelle piste ~350px (CAM_W=640, prop. 512→640)
SLIDE_WIN    = 88            # fenêtre ±px pour sliding windows (prop. 70/512*640)

KP           = 0.006         # réduit : 6fps = 167ms par frame, évite sur-braquage
KD           = 0.007         # augmenté : amortit l'oscillation due au retard visuel
ALPHA_D      = 0.7
STEERING_MAX = 0.85
STEERING_DEADZONE = 0.05
CAMERA_OFFSET_PX = 0         # biais caméra — calibrer si la voiture dérive constamment

V_MAX        = 0.14          # vitesse max ligne droite (~7% duty)
V_TURN       = 0.08          # vitesse virage (réduit pour stabilité)
V_SLOW       = 0.10          # vitesse récupération b=1
V_STOP       = 0.00

CURVE_THRESH_HIGH = 0.30
CURVE_THRESH_LOW  = 0.15

CURRENT_MAX  = 5.0

# ══════════════════════════════════════════════════════════════════════════════
# STREAM MJPEG — partagé entre le thread caméra et le HTTP server
# ══════════════════════════════════════════════════════════════════════════════

_latest_jpeg = None
_frame_id    = 0
_stream_lock = threading.Lock()
_placeholder = None
_last_frame_time = [time.time()]   # watchdog : heure de la dernière frame reçue
_watchdog_trigger = [False]        # mis à True par le watchdog pour forcer un reset
_camera_restarted = [False]        # mis à True après chaque reconnexion OAK-D → reset Kalman
_coast_steer    = [0.0]            # dernier steering valide pour coast mode
_coast_throttle = [0.06]           # dernier throttle valide pour coast mode
_coast_crash_t  = [0.0]            # timestamp du début du crash caméra actuel
_drive_enabled = True   # contrôlé via HTTP /stop et /go
_go_reset = [False]         # mis à True par /go → PDController reset son état CORNER/Kalman
_calibrate_request = [False]       # mis à True par /calibrate → PDController applique l'offset
_calibrate_result  = [None]        # renseigné par PDController avec la valeur appliquée
_finish_map_request = [False]      # mis à True par /finish_map → sauvegarde track_map.json

def _make_placeholder():
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(img, "En attente camera...", (60, CAM_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return jpg.tobytes()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_text(self, body):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def do_GET(self):
        global _drive_enabled
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/stop":
            _drive_enabled = False
            self._send_text("STOPPED")
            print("[ctrl] /stop recu")
            return
        if path == "/go":
            _drive_enabled = True
            _go_reset[0] = True
            self._send_text("RUNNING")
            print("[ctrl] /go recu")
            return
        if path == "/status":
            self._send_text("running" if _drive_enabled else "stopped")
            return
        if path == "/calibrate":
            _calibrate_request[0] = True
            # Attendre max 2s que PDController applique l'offset
            for _ in range(40):
                if _calibrate_result[0] is not None:
                    break
                time.sleep(0.05)
            result = _calibrate_result[0]
            _calibrate_result[0] = None
            if result is not None:
                self._send_text("CAMERA_OFFSET_PX={:+d}px applique".format(result))
            else:
                self._send_text("ERREUR: b<2 ou pas assez de frames stables")
            return
        if path == "/finish_map":
            _finish_map_request[0] = True
            self._send_text("MAPPING_FINISH_REQUESTED")
            print("[track] /finish_map recu")
            return
        if path not in ("/", "/stream"):
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.connection.settimeout(10)
        last_id = -1
        try:
            while True:
                with _stream_lock:
                    cur_id = _frame_id
                    jpg    = _latest_jpeg or _placeholder
                if cur_id == last_id or jpg is None:
                    time.sleep(0.02)
                    continue
                last_id = cur_id
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write("Content-Length: {}\r\n\r\n".format(len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except Exception:
                    break
        except Exception:
            pass


def start_stream_server(port):
    global _placeholder
    _placeholder = _make_placeholder()
    srv = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print("[stream] http://{}:{}".format(
        socket.gethostbyname(socket.gethostname()), port))


def push_frame(bgr, mask, info, rejected_blobs=None):
    global _latest_jpeg, _frame_id
    vis = bgr.copy()
    mask_rej   = info.get("mask_rejected")
    mask_clean = info.get("mask_clean", mask)  # masque filtré pour overlay vert
    # overlay vert = pixels ACCEPTÉS (vraies lignes de piste après filtre IA)
    green = np.zeros_like(vis)
    green[:, :, 1] = mask_clean
    vis = cv2.addWeighted(vis, 1.0, green, 0.5, 0)
    # overlay rouge = pixels REJETÉS par filtre IA (artefacts, reflets, murs)
    if mask_rej is not None and mask_rej.any():
        red_ov = np.zeros_like(vis)
        red_ov[:, :, 2] = mask_rej
        vis = cv2.addWeighted(vis, 1.0, red_ov, 0.7, 0)
    # Blobs REJETÉS en orange — artefacts filtrés (debug, ne touche pas l'algo)
    if rejected_blobs:
        for rb in rejected_blobs:
            x, yt, w, h = rb["rect"]
            cv2.rectangle(vis, (x, yt), (x + w, yt + h), (0, 128, 255), 1)
            cv2.putText(vis, rb["reason"], (x, max(yt - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 128, 255), 1)
    # Ligne verticale blanche = cible (la voiture doit rester ici)
    cv2.line(vis, (CAM_W // 2, int(CAM_H * ROI_FAR)), (CAM_W // 2, CAM_H), (255, 255, 255), 1)
    # Lignes JAUNES verticales = positions détectées par histogramme
    y_hist = int(CAM_H * 0.62)
    if info.get("hist_left_cx") is not None:
        cv2.line(vis, (info["hist_left_cx"], y_hist), (info["hist_left_cx"], CAM_H), (0, 220, 255), 2)
    if info.get("hist_right_cx") is not None:
        cv2.line(vis, (info["hist_right_cx"], y_hist), (info["hist_right_cx"], CAM_H), (0, 220, 255), 2)
    # Lignes ORANGE POINTILLÉES = positions PRÉDITES (ligne disparue, extrapolée par velocity)
    for _pcx in [info.get("pred_left_cx"), info.get("pred_right_cx")]:
        if _pcx is not None:
            _pcx = max(0, min(CAM_W - 1, _pcx))
            for _ys in range(y_hist, CAM_H, 10):
                cv2.line(vis, (_pcx, _ys), (_pcx, min(_ys + 6, CAM_H - 1)), (0, 140, 255), 2)
            cv2.circle(vis, (_pcx, int(CAM_H * 0.75)), 6, (0, 140, 255), -1)
            cv2.putText(vis, "P", (_pcx + 4, int(CAM_H * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)
    # Raycasts HORIZONTAUX : tirets depuis le centre + cercle rouge sur la touche
    mid_x = CAM_W // 2
    for (hx, hy) in info.get("scan_left_hits", []):
        cv2.line(vis, (mid_x, hy), (hx, hy), (0, 80, 255), 1)   # trait rouge depuis centre
        cv2.circle(vis, (hx, hy), 4, (0, 0, 255), -1)            # cercle rouge = touche
    for (hx, hy) in info.get("scan_right_hits", []):
        cv2.line(vis, (mid_x, hy), (hx, hy), (0, 80, 255), 1)
        cv2.circle(vis, (hx, hy), 4, (0, 0, 255), -1)
    # Point VERT = midpoint de contrôle réel (ce que suit la voiture)
    # Le trait bleu = direction de steering, part du bas-centre vers le point vert
    if info["err"] is not None:
        ctrl_cx = int(CAM_W // 2 + info["err"])
        ctrl_cx = max(0, min(CAM_W - 1, ctrl_cx))
        ctrl_cy = int(CAM_H * 0.78)
        cv2.circle(vis, (ctrl_cx, ctrl_cy), 10, (0, 255, 0), -1)
        cv2.line(vis, (CAM_W // 2, CAM_H - 1), (ctrl_cx, ctrl_cy), (255, 0, 0), 2)
    # lignes bandes
    for frac, color in [(ROI_NEAR, (255, 200, 0)), (ROI_MID, (0, 200, 255)), (ROI_FAR, (0, 100, 255))]:
        cv2.line(vis, (0, int(CAM_H * frac)), (CAM_W, int(CAM_H * frac)), color, 1)
    # texte
    err_str = "{:+d}".format(int(info["err"])) if info["err"] is not None else "N/A"
    corner_flag = " [L]" if info.get("corner") else ""
    pred_flag  = " PRED{}".format(info.get("n_pred", 0)) if info.get("n_pred", 0) > 0 else ""
    cv2.putText(vis,
        "err={} steer={:.2f} thr={:.2f} {}{}{} b={} ray={:+.2f}".format(
            err_str, info["steering"], info["throttle"],
            info["state"], corner_flag, pred_flag, info["n_blobs"],
            info.get("ray_asym", 0.0)),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    # Panel droit : masque coloré — vert=accepté(ligne), rouge=rejeté(artefact IA)
    panel = np.zeros_like(vis)
    panel[mask > 0, 1] = 255          # vert = pixel accepté (vraie ligne)
    if mask_rej is not None:
        panel[mask_rej > 0, 2] = 255  # rouge = pixel rejeté par filtre IA
    display = np.hstack([vis, panel])
    _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
    with _stream_lock:
        _latest_jpeg = jpg.tobytes()
        _frame_id   += 1


# ══════════════════════════════════════════════════════════════════════════════
# VISION
# ══════════════════════════════════════════════════════════════════════════════

def get_blobs(mask):
    """Retourne (accepted_blobs, rejected_blobs) — les rejetés servent uniquement à la visu orange."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cy_min     = int(CAM_H * 0.44)
    cy_max     = int(CAM_H * 0.97)
    y_bot_min  = int(CAM_H * 0.62)
    # 0.10 : pieds de chaises (area<<800) déjà éliminés par MIN_BLOB_AREA,
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
        if area < MIN_BLOB_AREA:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "area", "rect": rect})
            continue
        if aspect < aspect_min:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "asp", "rect": rect})
            continue
        if w < w_min:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "w", "rect": rect})
            continue
        # Blobs compacts et pleins (logo/flèche au sol) — solidity haute + pas extrêmement allongé
        # Les lignes de piste ont solidity faible (<0.50) car fines dans leur bounding box
        # Les logos remplis ont solidity >0.65. Aspect < 2.5 évite de filtrer les lignes proches horizontales
        bbox_area = w * h
        solidity = float(area) / max(bbox_area, 1)
        if area > 4700 and solidity > 0.65 and aspect < 2.5:  # prop. 3000*1.5625
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "cmp", "rect": rect})
            continue
        # Rayons de soleil / reflets horizontaux : très larges et peu hauts (w>>h)
        # Les lignes de piste longitudinales ont aspect < 4, les rayons > 5
        if aspect > 5.0 and area > 2350:  # prop. 1500*1.5625
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


def err_from_mask(mask):
    M = cv2.moments(mask)
    if M["m00"] < 1:
        return None
    return int(M["m10"] / M["m00"]) - CAM_W // 2


def err_from_scanlines(mask):
    """3 scanlines à FAR/MID/NEAR : cherche pixel blanc le + à gauche et + à droite
    sur chaque ligne horizontale → centre entre elles → médiane des 3 centres.
    Robuste aux faux positifs : sol au milieu ignoré (on prend les extrêmes).
    Retourne (err, scan_points) où scan_points = [(cx, row), ...] pour affichage.
    """
    rows = [int(CAM_H * ROI_FAR), int(CAM_H * ROI_MID), int(CAM_H * ROI_NEAR)]
    centers = []
    scan_points = []
    for r in rows:
        r = min(r, CAM_H - 1)
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
    return median_c - CAM_W // 2, scan_points


def err_from_bands(mask):
    row_near = int(CAM_H * ROI_NEAR)
    row_mid  = int(CAM_H * ROI_MID)
    row_far  = int(CAM_H * ROI_FAR)
    mask_near = mask.copy(); mask_near[:row_near, :] = 0
    mask_mid  = mask.copy(); mask_mid[row_near:, :] = 0; mask_mid[:row_mid, :] = 0
    mask_far  = mask.copy(); mask_far[row_mid:, :] = 0;  mask_far[:row_far, :] = 0
    return err_from_mask(mask_near), err_from_mask(mask_mid), err_from_mask(mask_far)


def clean_mask_artifacts(mask, bgr=None):
    """
    Filtre intelligent du masque : supprime les composantes qui ne sont pas des lignes.

    Deux niveaux de filtrage :
    1. Géométrie : area, aspect ratio, position Y — rejette chaussures, logos, murs
    2. Sobel edges (si bgr fourni) : les vraies lignes ont des bords nets sur fond gris.
       Les reflets/artefacts diffus ont des bords flous → gradient Sobel faible.

    Retourne (mask_clean, rejected_mask) pour la visualisation.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean    = np.zeros_like(mask)
    rejected = np.zeros_like(mask)
    roi_top  = int(CAM_H * 0.52)

    # Pré-calcul Sobel sur image grise (une seule fois pour toutes les composantes)
    sobel_mag = None
    if bgr is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sx * sx + sy * sy)

    for i in range(1, n):
        area  = stats[i, cv2.CC_STAT_AREA]
        bx    = stats[i, cv2.CC_STAT_LEFT]
        by    = stats[i, cv2.CC_STAT_TOP]
        bw    = stats[i, cv2.CC_STAT_WIDTH]
        bh    = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        y_bot = by + bh   # bas du blob — clé : une ligne de piste touche la zone proche
        blob_mask = (labels == i)

        reason = None

        if area < 800:
            reason = "small"
        elif y_bot < int(CAM_H * 0.62):
            reason = "high"
        else:
            asp = float(max(bw, bh)) / max(min(bw, bh), 1)
            if asp < 2.5 and area < 4000:
                reason = "compact"  # trop carré → reflet, logo, chaussure, mur compact
            else:
                # PCA orientation : blobs MOYENS seulement (les vraies lignes ont area > 5000)
                if bw > 80 and 800 < area < 5000:
                    roi_m = blob_mask[by:by + bh, bx:bx + bw].astype(np.uint8)
                    _m = cv2.moments(roi_m)
                    if _m["m00"] > 0:
                        _mu20 = _m["mu20"] / _m["m00"]
                        _mu02 = _m["mu02"] / _m["m00"]
                        _mu11 = _m["mu11"] / _m["m00"]
                        _denom = _mu20 - _mu02
                        if abs(_denom) < 1e-6: _denom = 1e-6
                        _ang = abs(math.degrees(0.5 * math.atan2(2.0 * _mu11, _denom)))
                        if _ang < 15.0 and bw > 100:
                            reason = "horizontal"

                # Sobel uniquement sur blobs MOYENS non proches
                # Les lignes proches (y_bot bas, asp≈1) ont bords Sobel mous → faux rejets
                _proche = (y_bot > int(CAM_H * 0.75) or asp < 1.5)
                if reason is None and sobel_mag is not None and area < 8000 and not _proche:
                    kernel3 = np.ones((3, 3), np.uint8)
                    blob_u8 = blob_mask.astype(np.uint8) * 255
                    border  = cv2.dilate(blob_u8, kernel3) - blob_u8
                    border_pixels = sobel_mag[border > 0]
                    if len(border_pixels) > 0:
                        mean_grad = float(np.mean(border_pixels))
                        if mean_grad < 20.0:
                            reason = "diffuse"

        if reason is not None:
            rejected[blob_mask] = 255
        else:
            clean[blob_mask] = 255

    return clean, rejected


def find_lane_histogram(mask, prev_left=None, prev_right=None):
    """
    Détecte les deux lignes de piste par histogramme de colonnes.

    Au lieu de chercher des blobs (objets connectés), on somme les pixels blancs
    par colonne dans la zone basse. Les vraies lignes de piste traversent l'image
    en hauteur → gros pic. Les artefacts isolés (meubles, ombres) = pic faible.

    Returns: (left_cx, right_cx, left_conf, right_conf)
      left_cx / right_cx : position pixel du pic gauche/droit, None si non détecté
      left_conf / right_conf : énergie du pic (somme pixels blancs dans la colonne)
    """
    y_start = int(CAM_H * 0.62)
    y_end   = int(CAM_H * 0.97)
    roi = mask[y_start:y_end, :]

    # Histogramme : somme des pixels blancs par colonne (valeur max = 255 × hauteur_roi)
    hist = np.sum(roi.astype(np.float32), axis=0)

    # Lissage gaussien 21px : fusionne les pixels adjacents d'une même ligne blanche
    # Une ligne de piste fait ~30-60px de large → le pic se consolide
    k = 21
    sigma = k / 4.0
    xs = np.arange(-(k // 2), k // 2 + 1, dtype=np.float32)
    gauss = np.exp(-0.5 * (xs / sigma) ** 2)
    gauss /= gauss.sum()
    hist = np.convolve(hist, gauss, mode='same')

    mid = CAM_W // 2

    # Seuil minimum : au moins ~12px blancs dans la colonne
    HIST_MIN = 12.0 * 255.0

    left_half  = hist[:mid]
    right_half = hist[mid:]

    left_peak  = float(np.max(left_half))
    right_peak = float(np.max(right_half))

    # Multi-pics : chercher le pic le plus cohérent avec la track_width et la position précédente
    # Au lieu du simple argmax (vulnérable aux gros artefacts), on liste les pics locaux
    # et on choisit celui dont le score est le meilleur.
    def _best_peak(half_hist, offset, prev_cx, peer_cx, peer_side):
        """
        Retourne le meilleur cx dans half_hist (offset = décalage par rapport à la demi-image).
        Si prev_cx fourni : sliding windows ±SLIDE_WIN autour de la position précédente.
        peer_cx : position de la ligne de l'autre côté (pour contrainte de track_width).
        """
        n = len(half_hist)
        # Sliding windows : restreindre la zone de recherche si on a un historique
        if prev_cx is not None:
            local_center = prev_cx - offset
            i_min = max(2, local_center - SLIDE_WIN)
            i_max = min(n - 2, local_center + SLIDE_WIN)
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
                score += max(0.0, 4000.0 - abs(dist - TRACK_WIDTH_EST_PX) * 80.0)
            if score > best_score:
                best_score = score
                best_cx = cx
        # Fallback local argmax
        if best_cx is None:
            sub = half_hist[i_min:i_max + 1]
            if len(sub) > 0 and float(np.max(sub)) >= HIST_MIN:
                best_cx = int(np.argmax(sub)) + i_min + offset
        return best_cx

    left_cx  = _best_peak(left_half,  0,   prev_left,  None, +1)
    right_cx = _best_peak(right_half, mid, prev_right, left_cx, -1)
    # Deuxième passe : re-scorer left avec right connu
    left_cx  = _best_peak(left_half,  0,   prev_left,  right_cx, +1)

    left_peak  = float(left_half[left_cx]) if left_cx is not None else 0.0
    right_peak = float(right_half[right_cx - mid]) if right_cx is not None else 0.0

    return left_cx, right_cx, left_peak, right_peak


def find_lane_scanlines(mask, n_lines=6):
    """
    Raycasts horizontaux pour détecter le bord INTÉRIEUR de chaque ligne de piste.

    Stratégie : chercher depuis chaque moitié de l'image vers le centre.
      - Moitié gauche (x=0 à mid-30) : le blanc le plus proche du centre = bord intérieur ligne gauche
      - Moitié droite (x=mid+30 à CAM_W) : le blanc le plus proche du centre = bord intérieur ligne droite

    Avantage : insensible au bruit vert central (on ne part pas du centre, on évite cette zone).
    Les vraies lignes de piste sont en dehors de la zone centrale.

    Returns: (left_cx, right_cx, scanline_rows, left_hits, right_hits)
      left_cx / right_cx : médiane des touches, None si aucune touche
      scanline_rows : liste des y des scanlines (pour visu)
      left_hits / right_hits : liste des touches (x, y) pour visu
    """
    mid_x = CAM_W // 2
    MARGIN = 38   # zone morte autour du centre (prop. 30/512*640)
    MIN_WHITES = 4  # nombre minimum de pixels blancs pour valider une touche

    # Scanlines de 65% à 90% de la hauteur — zone proche, lignes larges et nettes
    rows = [int(CAM_H * (0.65 + i * (0.25 / max(n_lines - 1, 1)))) for i in range(n_lines)]

    left_xs   = []
    right_xs  = []
    left_hits  = []
    right_hits = []

    for r in rows:
        r = min(r, CAM_H - 1)
        line = mask[r, :]

        # Moitié gauche : bord intérieur de la ligne gauche = blanc le plus proche du centre
        whites_l = np.where(line[:mid_x - MARGIN] > 0)[0]
        if len(whites_l) >= MIN_WHITES:
            hit_l = int(whites_l[-1])   # le plus à droite = bord intérieur
            # Validation continuité verticale : la ligne doit exister aussi 3px au-dessus
            if r >= 3:
                above = int(np.sum(mask[r - 3:r, max(0, hit_l - 3):hit_l + 4]))
                if above >= 3 * 255:  # au moins 3 pixels blancs au-dessus
                    left_xs.append(hit_l)
                    left_hits.append((hit_l, r))
            else:
                left_xs.append(hit_l)
                left_hits.append((hit_l, r))

        # Moitié droite : bord intérieur de la ligne droite = blanc le plus proche du centre
        whites_r = np.where(line[mid_x + MARGIN:] > 0)[0]
        if len(whites_r) >= MIN_WHITES:
            hit_r = int(mid_x + MARGIN + whites_r[0])  # le plus à gauche = bord intérieur
            # Validation continuité verticale
            if r >= 3:
                above = int(np.sum(mask[r - 3:r, hit_r - 3:min(CAM_W, hit_r + 4)]))
                if above >= 3 * 255:
                    right_xs.append(hit_r)
                    right_hits.append((hit_r, r))
            else:
                right_xs.append(hit_r)
                right_hits.append((hit_r, r))

    left_cx  = int(np.median(left_xs))  if left_xs  else None
    right_cx = int(np.median(right_xs)) if right_xs else None

    return left_cx, right_cx, rows, left_hits, right_hits


def fuse_lane_estimates(hist_left, hist_right, scan_left, scan_right):
    """
    Fusionne les estimations histogramme + raycasts horizontaux.

    Si les deux méthodes sont proches (< 40px) → moyenne pondérée.
    Si divergence → préférer l'histogramme (vue globale plus robuste).
    Si une seule méthode donne un résultat → utiliser celle-là.

    Returns: (left_cx, right_cx) — valeurs fusionnées ou None
    """
    MAX_DIVERGENCE = 40   # px

    def _fuse_one(h, s):
        if h is not None and s is not None:
            if abs(h - s) <= MAX_DIVERGENCE:
                return int(round(0.5 * h + 0.5 * s))  # moyenne équipondérée
            else:
                return h  # divergence → histogramme prioritaire
        if h is not None:
            return h
        if s is not None:
            return s
        return None

    return _fuse_one(hist_left, scan_left), _fuse_one(hist_right, scan_right)


def detect_corner_blob(mask):
    """Détecte le marqueur de coin L : blob compact (area >= MIN_CORNER_AREA, aspect < 1.8).
    Appliqué sur mask_wide (ROI 45%) pour voir le L bien avant d'y arriver.
    Retourne dict {cx, cy, area} ou None.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cy_min = int(CAM_H * 0.62)  # coin L visible seulement dans le bas (60%+), élimine murs/plafond
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        asp  = w / float(h)
        cy   = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
        cx   = stats[i, cv2.CC_STAT_LEFT] + w // 2
        if area >= MIN_CORNER_AREA and asp < 1.5 and cy >= cy_min:
            if best is None or area > best["area"]:
                best = {"cx": cx, "cy": cy, "area": area, "aspect": round(asp, 1)}
    return best


def err_from_two_lines(blobs, track_width=None):
    mid_x = CAM_W // 2
    CLEAR_LEFT  = mid_x - 95   # prop. 76/512*640
    CLEAR_RIGHT = mid_x + 95
    # Largeur dynamique : utilise la mesure récente si dispo, sinon constante
    tw_est = int(track_width) if track_width is not None else TRACK_WIDTH_EST_PX
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


# ══════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ══════════════════════════════════════════════════════════════════════════════

class _Kalman1D(object):
    """Filtre de Kalman 1D pour l'erreur latérale. Robuste aux dropouts b=0/b=1."""
    def __init__(self, q=0.05, r=20.0):
        self.x = 0.0
        self.P = 1.0
        self.Q = q   # bruit processus (0.05 = lent à changer)
        self.R = r   # bruit mesure   (20 = on fait confiance à la vision à ~70%)

    def update(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

    def reset(self):
        self.x = 0.0
        self.P = 1.0


class PDController:
    def __init__(self, level=3, fixed_speed=None, no_corner=False):
        self.level        = level
        self.fixed_speed  = fixed_speed
        self.no_corner    = no_corner
        self.prev_err     = 0.0
        self.d_filtered   = 0.0
        self.state        = "STOP"
        self.err_history      = []    # dernières 6 erreurs pour tendance virage
        # ── Machine à états coin L ───────────────────────────────────────────
        self.corner_mode      = False
        self.corner_dir       = 0.0
        self.corner_count     = 0
        self.corner_accum     = 0     # accumulateur multi-signal (détection anticipée)
        self.prev_ray_asym    = 0.0   # dérivée ray_asym entre frames
        self.corner_imu_angle = 0.0   # angle intégré IMU depuis début virage (rad)
        self.corner_imu_t     = 0.0   # timestamp pour dt IMU
        self.corner_release   = 0     # fading sortie CORNER (frames restantes)
        self.search_frames    = 0     # frames restantes en mode SEARCH (post-CORNER_EXIT)
        self.last_corner_steer = 0.0  # dernier steering CORNER pour fading
        self.corner_sanity_ctr = 0    # watchdog contradiction direction (Q5)
        self.is_u_turn         = False  # virage en U détecté (angle > 77°)
        self.gyro_accum_corner = 0.0    # gyro cumulé depuis début CORNER (rad)
        self.u_exit_prepared   = False  # flag sortie U prête
        # ── Calibration biais gyroscope en ligne (Q1 — EMA conditionnelle) ──────
        self.gyro_bias_z      = 0.0   # biais estimé gyro_z (rad/s)
        self.gyro_calib_n     = 0     # compteur phases stables
        # ── Priorité 4 : mémoire de direction (IA suggestion) ────────────────
        self.last_turn_dir    = 0.0   # dernière direction forte mémorisée
        self.turn_memory_ctr  = 0     # frames restantes de mémoire
        # ── Dynamic track width : médiane des 20 dernières largeurs connues ───
        self.track_widths     = []    # largeurs réelles mesurées (n_blobs=2)
        # ── Blob proximity tracker (IA) ───────────────────────────────────────
        self.last_left_cx     = None  # cx du blob gauche frame précédente
        self.last_right_cx    = None  # cx du blob droit frame précédente
        # ── INERTIAL_COAST : maintien commande si vision perdue (IA) ──────────
        self.blind_frames     = 0     # compteur frames sans vision
        self.last_steering_cmd = 0.0  # dernier steering valide
        # ── Err smoothing exponentiel (IA) ────────────────────────────────────
        self.err_smooth       = 0.0
        # ── Kalman 1D sur err latérale (IA multi-sources) ─────────────────────
        self.kalman           = _Kalman1D(q=2.0, r=20.0)  # Q=2.0 → ~10 frames pour converger
        # ── Dérivée temporelle correcte ───────────────────────────────────────
        self.last_pd_time     = time.time()
        # ── CORNER multi-signal (IA) ──────────────────────────────────────────
        self.prev_n_blobs     = 0     # pour détecter transition b=2→1
        # ── Auto-calibration offset caméra ────────────────────────────────────
        self.calib_err_history = []   # err brutes récentes pour calibration
        # ── Prédiction ligne disparue : historique velocity ───────────────────
        self.left_cx_hist  = []   # N dernières positions ligne gauche détectées
        self.right_cx_hist = []   # N dernières positions ligne droite détectées
        self.left_age      = 0    # frames depuis dernière détection ligne gauche
        self.right_age     = 0    # frames depuis dernière détection ligne droite
        self.auto_offset       = 0.0  # offset appris en ligne (EMA)
        # ── Servo bias : biais mécanique châssis ──────────────────────────────
        self.servo_bias        = 0.0  # offset px appris (steering résiduel quand err≈0)
        self._bias_samples     = []   # échantillons steering récents quand err≈0
        # ── Sliding windows : positions précédentes pour restreindre la recherche
        self.hist_prev_left    = None  # cx ligne gauche frame précédente
        self.hist_prev_right   = None  # cx ligne droite frame précédente
        self.vr           = VisualRays(
            img_width=CAM_W, img_height=CAM_H,
            row_band=(ROI_FAR, ROI_BOTTOM), morph_k=5,
        )

    def _pd(self, err):
        now = time.time()
        dt  = max(now - self.last_pd_time, 0.01)
        self.last_pd_time = now
        d_raw = err - self.prev_err
        # Atténuer si frame droppée (dt > 2× nominal 1/6s) → évite spike dérivée
        if dt > 0.35:
            d_raw *= 0.5
        self.d_filtered = ALPHA_D * self.d_filtered + (1.0 - ALPHA_D) * d_raw
        self.prev_err = err
        # KP adaptatif : fort si loin du centre, doux si proche (anti-oscillation)
        abs_err = abs(err)
        if abs_err > 50:
            kp = 0.015
        elif abs_err < 15:
            kp = 0.008
        else:
            kp = 0.008 + (0.015 - 0.008) * (abs_err - 15.0) / (50.0 - 15.0)
        raw = kp * err + KD * self.d_filtered
        raw = max(-STEERING_MAX, min(STEERING_MAX, raw))
        if abs(raw) < STEERING_DEADZONE:
            raw = 0.0
        return raw

    def _combined_err(self, err_near, err_mid, err_far, rays):
        fwd = float(np.mean(rays[8:12]))
        w_far  = 0.2 + 0.4 * fwd
        w_mid  = 0.30
        w_near = max(0.0, 1.0 - w_far - w_mid)
        pairs = [(err_near, w_near), (err_mid, w_mid), (err_far, w_far)]
        valid = [(e, w) for e, w in pairs if e is not None]
        if not valid:
            return None
        total_w = sum(w for _, w in valid)
        return sum(e * w for e, w in valid) / total_w

    def compute(self, mask, bgr, mask_wide=None, gyro_z=0.0):
        global CAMERA_OFFSET_PX
        if _go_reset[0]:
            _go_reset[0] = False
            self.corner_mode      = False
            self.corner_count     = 0
            self.prev_n_blobs     = 0
            self.corner_accum     = 0
            self.corner_release   = 0
            self.prev_ray_asym    = 0.0
            self.corner_imu_angle = 0.0
            self.corner_sanity_ctr = 0
            self.is_u_turn         = False
            self.gyro_accum_corner = 0.0
            self.u_exit_prepared   = False
            self.left_cx_hist     = []
            self.right_cx_hist    = []
            self.left_age         = 0
            self.right_age        = 0
            self.kalman.reset()
            self.err_smooth   = 0.0
            self.prev_err     = 0.0
            print("[ctrl] /go reset CORNER+Kalman")
        if _camera_restarted[0]:
            _camera_restarted[0] = False
            self.kalman.reset()
            self.err_smooth      = 0.0
            self.prev_err        = 0.0
            self.track_widths    = []
            self.hist_prev_left  = None  # oublier positions sliding windows
            self.hist_prev_right = None
            print("[ctrl] camera restart → reset Kalman+track_widths+sliding_windows")
        rays    = self.vr(bgr)
        blobs, rejected_blobs = get_blobs(mask)
        n_blobs = len(blobs)
        forward_clearance = float(np.mean(rays[8:12]))
        err = None
        corner_blob = None
        scan_pts = []

        # ── Asymétrie raycasts : signal de virage très rapide ─────────────
        left_open  = float(np.mean(rays[:7]))
        right_open = float(np.mean(rays[13:]))
        ray_asym   = right_open - left_open  # >0 espace à droite → virage droite

        # ── Détection coin L dans mask_wide (ROI large 45%) ───────────────
        m_wide = mask_wide if mask_wide is not None else mask
        corner_blob = detect_corner_blob(m_wide)

        # ── Masque nettoyé : supprime artefacts (reflets, chaussures, murs) ──
        # Le masque brut reste pour la visu orange (blobs rejetés).
        # L'histogramme et les scanlines utilisent le masque propre.
        mask_clean, mask_rejected = clean_mask_artifacts(mask, bgr=bgr)

        # ── Détection lignes : Histogramme sliding + Scanlines + Fusion ─────
        hist_l, hist_r, hist_lconf, hist_rconf = find_lane_histogram(
            mask_clean, prev_left=self.hist_prev_left, prev_right=self.hist_prev_right)
        scan_l, scan_r, scan_rows, scan_left_hits, scan_right_hits = find_lane_scanlines(mask_clean)

        # Confiance scanlines : nb hits / 6 scanlines
        conf_scan_l = len(scan_left_hits)  / 6.0
        conf_scan_r = len(scan_right_hits) / 6.0
        HIST_PEAK_MAX = 12.0 * 255.0 * 20.0  # valeur de normalisation
        conf_hist_l = min(1.0, hist_lconf / HIST_PEAK_MAX)
        conf_hist_r = min(1.0, hist_rconf / HIST_PEAK_MAX)

        # Fusion pondérée histogramme + scanlines quand les deux existent et concordent
        def _fuse_cx(h, c_h, s, c_s):
            if h is None and s is None:
                return None
            if h is None:
                return s
            if s is None or c_s < 0.34:  # moins de 2 hits scanlines → ignorer
                return h
            if abs(h - s) > 50:          # divergence forte → histogramme prioritaire
                return h
            total_w = c_h + c_s
            if total_w < 1e-6:
                return h
            return int(round((h * c_h + s * c_s) / total_w))

        left_cx  = _fuse_cx(hist_l, conf_hist_l, scan_l, conf_scan_l)
        right_cx = _fuse_cx(hist_r, conf_hist_r, scan_r, conf_scan_r)

        # Mémoriser les positions pour la prochaine frame (sliding windows)
        if left_cx  is not None: self.hist_prev_left  = left_cx
        if right_cx is not None: self.hist_prev_right = right_cx

        # n_blobs : nombre de lignes détectées (0/1/2) — pour CORNER et COAST
        n_blobs = (1 if left_cx is not None else 0) + (1 if right_cx is not None else 0)

        # Ages de prédiction : toujours mis à jour (CORNER inclus)
        if left_cx is not None:
            self.left_age = 0
        else:
            self.left_age += 1
        if right_cx is not None:
            self.right_age = 0
        else:
            self.right_age += 1

        # ── Machine à états CORNER — accumulateur multi-signal + dérivée ray_asym ──
        d_asym = ray_asym - self.prev_ray_asym
        self.prev_ray_asym = ray_asym

        if not self.corner_mode and not self.no_corner:
            # Décrémentation naturelle : rémanence ~10 frames
            self.corner_accum = max(0, self.corner_accum - 1)
            # Accumulation multi-signal
            if corner_blob is not None:
                self.corner_accum += 3                      # blob compact L = signal fort
            if abs(ray_asym) > 0.28:
                self.corner_accum += 1                      # asymétrie raycasts
            if abs(d_asym) > 0.12:
                self.corner_accum += 1                      # dérivée : courbe s'amorce tôt
            if self.prev_n_blobs == 2 and n_blobs == 1:
                self.corner_accum += 2                      # perte soudaine d'une ligne
            if self.prev_n_blobs >= 1 and n_blobs == 0:
                self.corner_accum += 3                      # perte totale des lignes

            if self.corner_accum >= 5:
                if corner_blob is not None:
                    corner_dir_cx = corner_blob["cx"]
                elif ray_asym != 0:
                    corner_dir_cx = CAM_W // 2 + (1 if ray_asym > 0 else -1)
                elif err is not None:
                    corner_dir_cx = CAM_W // 2 + (1 if err > 0 else -1)
                else:
                    corner_dir_cx = CAM_W // 2 + 1
                self.corner_dir        = 1.0 if corner_dir_cx > CAM_W // 2 else -1.0
                self.corner_mode       = True
                self.corner_count      = CORNER_DURATION
                self.corner_imu_angle  = 0.0
                self.corner_imu_t      = time.time()
                self.corner_accum      = 0
                self.is_u_turn         = False
                self.gyro_accum_corner = 0.0
                self.u_exit_prepared   = False
                print("[ctrl] CORNER accum d_asym={:.2f} asym={:.2f} dir={:+.0f}".format(
                    d_asym, ray_asym, self.corner_dir))

        # Offset effectif calculé ici pour être appliqué à l'erreur BRUTE
        effective_offset = CAMERA_OFFSET_PX + int(self.auto_offset) + int(self.servo_bias)
        n_pred        = 0     # nombre de lignes prédites cette frame (0 si CORNER actif)
        pred_left_cx  = None  # cx prédit gauche pour visualisation
        pred_right_cx = None  # cx prédit droite pour visualisation

        # ── Q1 : Calibration biais gyroscope en ligne (EMA conditionnelle) ──────
        # N'apprend que sur phases b=2 stables (voiture droite, err faible, asym nulle)
        if n_blobs == 2 and abs(ray_asym) < 0.08 and err is not None and abs(err) < 15 and not self.corner_mode:
            if self.gyro_calib_n < 60:   # initialisation rapide : alpha=0.05
                self.gyro_bias_z = 0.95 * self.gyro_bias_z + 0.05 * gyro_z
                self.gyro_calib_n += 1
            else:                        # tracking lent : alpha=0.003
                self.gyro_bias_z = 0.997 * self.gyro_bias_z + 0.003 * gyro_z
        gyro_z_cal = gyro_z - self.gyro_bias_z

        if self.corner_mode:
            # ── IMU : intégration angle depuis début du virage ─────────────
            now_imu = time.time()
            dt_imu  = min(now_imu - self.corner_imu_t, 0.3)  # clamp si frame droppée
            self.corner_imu_t = now_imu
            self.corner_imu_angle  += abs(gyro_z_cal) * dt_imu
            self.gyro_accum_corner += abs(gyro_z_cal) * dt_imu

            # ── Détection virage en U (~77° ou gyro accumulé > 2.2 rad) ─────
            if (not self.is_u_turn and
                    (self.corner_imu_angle > U_DETECT_ANGLE or
                     self.gyro_accum_corner > U_GYRO_ACCUM)):
                self.is_u_turn = True
                self.corner_count = max(self.corner_count, U_CORNER_MAX)
                print("[ctrl] U-TURN détecté angle={:.0f}deg accum={:.2f}".format(
                    math.degrees(self.corner_imu_angle), self.gyro_accum_corner))

            # ── Q5 : Watchdog cohérence direction (désactivé en U-turn profond) ──
            _q5_active = not (self.is_u_turn and self.corner_imu_angle > 1.0)
            if _q5_active and ray_asym * self.corner_dir < -0.30:
                self.corner_sanity_ctr += 1
                if self.corner_sanity_ctr >= 5:
                    self.corner_dir = -self.corner_dir
                    self.corner_sanity_ctr = 0
                    print("[ctrl] CORNER watchdog inversion dir={:+.0f}".format(self.corner_dir))
            else:
                self.corner_sanity_ctr = max(0, self.corner_sanity_ctr - 1)

            # ── Condition de sortie : timer OU angle IMU (100° simple / 180° U) ──
            self.corner_count -= 1
            _exit_angle = math.pi if self.is_u_turn else 1.745  # 180° U / 100° simple
            imu_done = self.corner_imu_angle >= _exit_angle
            if self.corner_count <= 0 or imu_done:
                self.corner_mode = False
                self.corner_release = U_EXIT_FADE if self.is_u_turn else 4
                self.corner_sanity_ctr = 0
                self.last_corner_steer = self.last_steering_cmd
                self.left_age  = min(self.left_age, 3)
                self.right_age = min(self.right_age, 3)
                print("[ctrl] CORNER fin {} angle={:.0f}deg frames_restants={}".format(
                    "U" if self.is_u_turn else "S",
                    math.degrees(self.corner_imu_angle), self.corner_count))

            # ── Steering : centrage si b=2 visible, sinon err forcée ──────────
            # b=2 entre les lignes → utiliser le centre réel (voiture entre les lignes)
            _center_used = False
            if left_cx is not None and right_cx is not None:
                _tw_c = right_cx - left_cx
                _car_between_c = (left_cx < CAM_W // 2 and right_cx > CAM_W // 2
                                  and _tw_c > 100)
                if _car_between_c:
                    _center_c = (left_cx + right_cx) // 2
                    err = _center_c - CAM_W // 2 - effective_offset
                    _center_used = True
            if not _center_used:
                _err_force = U_ERR_FORCE if self.is_u_turn else 220.0
                base_err = abs(err) if (err is not None and
                                        (err * self.corner_dir) > 0) else 0.0
                err = self.corner_dir * max(base_err, _err_force)
            self.state = "CORNER"
        else:
            # ── Calcul erreur depuis positions fusionnées ──────────────────
            last_tw = float(np.median(self.track_widths[-10:])) if len(self.track_widths) >= 3 else float(TRACK_WIDTH_EST_PX)

            # Historique velocity : mémoriser positions brutes avant rejet (pour extrapolation)
            if left_cx is not None:
                self.left_cx_hist.append(left_cx)
                if len(self.left_cx_hist) > 5: self.left_cx_hist.pop(0)
            if right_cx is not None:
                self.right_cx_hist.append(right_cx)
                if len(self.right_cx_hist) > 5: self.right_cx_hist.pop(0)

            # Rejeter cx extrêmes : ligne à <25px du bord = perspective aberrante en virage
            if left_cx is not None and left_cx < 25:
                left_cx = None
                n_blobs = max(0, n_blobs - 1)
            if right_cx is not None and right_cx > CAM_W - 25:
                right_cx = None
                n_blobs = max(0, n_blobs - 1)

            # ── Prédiction ligne manquante par extrapolation velocity + fallback track_width ──
            # En virage droite : ligne droite se déplace vers la droite avant de sortir →
            # vel>0 → pred_vel sort de l'image → center estimé décalé à droite (correct).
            # Fusion : velocity fiable les premières frames, track_width prend le relais.
            _MAX_PRED_AGE = 7  # ~0.54s max de prédiction

            if left_cx is None and self.left_cx_hist and self.left_age <= _MAX_PRED_AGE:
                if len(self.left_cx_hist) >= 3:
                    _vel_l = (self.left_cx_hist[-1] - self.left_cx_hist[-3]) / 2.0
                elif len(self.left_cx_hist) == 2:
                    _vel_l = float(self.left_cx_hist[-1] - self.left_cx_hist[0])
                else:
                    _vel_l = 0.0
                _pred_l_vel = self.left_cx_hist[-1] + _vel_l * self.left_age
                _pred_l_vel = max(10, min(CAM_W // 2 - 5, int(_pred_l_vel)))
                _pred_l_tw  = (right_cx - int(last_tw)) if right_cx is not None else _pred_l_vel
                _pred_l_tw  = max(10, min(CAM_W // 2 - 5, _pred_l_tw))
                _w_vel_l    = max(0.0, 1.0 - float(self.left_age) / _MAX_PRED_AGE)
                left_cx = int(round(_w_vel_l * _pred_l_vel + (1.0 - _w_vel_l) * _pred_l_tw))
                pred_left_cx = left_cx
                n_blobs += 1
                n_pred  += 1

            if right_cx is None and self.right_cx_hist and self.right_age <= _MAX_PRED_AGE:
                if len(self.right_cx_hist) >= 3:
                    _vel_r = (self.right_cx_hist[-1] - self.right_cx_hist[-3]) / 2.0
                elif len(self.right_cx_hist) == 2:
                    _vel_r = float(self.right_cx_hist[-1] - self.right_cx_hist[0])
                else:
                    _vel_r = 0.0
                _pred_r_vel = self.right_cx_hist[-1] + _vel_r * self.right_age
                _pred_r_vel = max(CAM_W // 2 + 5, min(CAM_W - 10, int(_pred_r_vel)))
                _pred_r_tw  = (left_cx + int(last_tw)) if left_cx is not None else _pred_r_vel
                _pred_r_tw  = max(CAM_W // 2 + 5, min(CAM_W - 10, _pred_r_tw))
                _w_vel_r    = max(0.0, 1.0 - float(self.right_age) / _MAX_PRED_AGE)
                right_cx = int(round(_w_vel_r * _pred_r_vel + (1.0 - _w_vel_r) * _pred_r_tw))
                pred_right_cx = right_cx
                n_blobs += 1
                n_pred  += 1

            if left_cx is not None and right_cx is not None:
                tw = right_cx - left_cx
                # Seuil tw assoupli si une ou deux lignes sont prédites (velocity peut clamper près du centre)
                _tw_min = 80 if n_pred > 0 else 150
                car_between = (left_cx < CAM_W // 2 and right_cx > CAM_W // 2 and tw > _tw_min)
                if car_between:
                    center = (left_cx + right_cx) // 2
                    err = center - CAM_W // 2 - effective_offset
                    if tw < CAM_W - 20:
                        self.track_widths.append(tw)
                        if len(self.track_widths) > 20:
                            self.track_widths.pop(0)
                else:
                    # Config invalide (ligne parasite ou voiture hors piste) → b=1
                    n_blobs = 1
                    if abs(left_cx - CAM_W // 2) < abs(right_cx - CAM_W // 2):
                        est_right = min(left_cx + int(last_tw), CAM_W - 10)
                        center = (left_cx + est_right) // 2
                    else:
                        est_left = max(right_cx - int(last_tw), 10)
                        center = (est_left + right_cx) // 2
                    err = center - CAM_W // 2 - effective_offset
            elif left_cx is not None:
                est_right = min(left_cx + int(last_tw), CAM_W - 10)
                center = (left_cx + est_right) // 2
                err_line = center - CAM_W // 2 - effective_offset
                # Fusion ray_asym : en virage (fort asym), l'estimation de ligne est peu fiable
                # ray_asym>0 = espace à droite = voiture trop à gauche = err doit être positif
                if abs(ray_asym) > 0.20:
                    err_ray = ray_asym * 220.0
                    w_ray = min(0.80, abs(ray_asym) * 2.0)
                    err = w_ray * err_ray + (1.0 - w_ray) * err_line
                    if abs(ray_asym) > 0.30 and not self.corner_mode and not self.no_corner:
                        # Fort asym + b=1 = virage certain → CORNER immédiat
                        self.corner_accum = max(self.corner_accum, 5)
                else:
                    err = err_line
                n_blobs = 1
            elif right_cx is not None:
                est_left = max(right_cx - int(last_tw), 10)
                center = (est_left + right_cx) // 2
                err_line = center - CAM_W // 2 - effective_offset
                if abs(ray_asym) > 0.20:
                    err_ray = ray_asym * 220.0
                    w_ray = min(0.80, abs(ray_asym) * 2.0)
                    err = w_ray * err_ray + (1.0 - w_ray) * err_line
                    if abs(ray_asym) > 0.30 and not self.corner_mode and not self.no_corner:
                        self.corner_accum = max(self.corner_accum, 5)
                else:
                    err = err_line
                n_blobs = 1
            else:
                err = None

            # blobs=0 : pas de fallback err_from_mask (centroïde global bruité → pollue Kalman)
            # err reste None → steer=0 propre, Kalman non mis à jour

            # Mémoire de tendance : maintient la direction si vision perdue
            if err is not None:
                self.err_history.append(float(err))
                if len(self.err_history) > 6:
                    self.err_history.pop(0)
                if abs(err) > 60:
                    self.last_turn_dir = 1.0 if err > 0 else -1.0
                    self.turn_memory_ctr = 15
            # Trend et mémoire : seulement si aucune ligne visible
            if n_blobs == 0:
                trend = sum(self.err_history) / len(self.err_history) if self.err_history else 0.0
                if err is None:
                    err = trend * 0.6
                if self.turn_memory_ctr > 0:
                    self.turn_memory_ctr -= 1
                    if err is None or abs(err) < 30:
                        err = (err or 0.0) + self.last_turn_dir * 50.0

        # ── Vitesse adaptative selon courbure (Priorité 3 IA) ─────────────
        curvature = float(np.std(rays))  # std des rays = indicateur de courbure
        if self.fixed_speed is not None:
            if self.corner_mode:
                _t_ratio = U_THROTTLE if self.is_u_turn else 0.50
                throttle = self.fixed_speed * _t_ratio
                self.blind_frames = 0
            elif n_blobs == 0:
                self.blind_frames += 1
                if self.blind_frames <= 10:           # ~1.7s à 6fps : INERTIAL_COAST
                    throttle = self.fixed_speed * 0.65
                    self.state = "COAST"
                    steering = self.last_steering_cmd * 0.80   # décroissance rapide vers 0
                    self.prev_n_blobs = 0
                    info = {
                        "err": err, "steering": steering, "throttle": throttle,
                        "state": self.state, "n_blobs": 0,
                        "forward_clearance": forward_clearance,
                        "blobs_cx": [], "corner": False,
                        "ray_asym": round(ray_asym, 2), "scan_pts": [],
                        "hist_left_cx": hist_l, "hist_right_cx": hist_r,
                        "scan_left_hits": scan_left_hits,
                        "scan_right_hits": scan_right_hits,
                        "mask_rejected": mask_rejected,
                        "mask_clean": mask_clean,
                    }
                    push_frame(bgr, mask_clean, info, rejected_blobs)
                    return steering, throttle, info
                else:
                    throttle = V_STOP
                    self.state = "BLIND"
            elif curvature > 0.30 or (err is not None and abs(err) > 80):
                throttle = self.fixed_speed * 0.75   # courbe détectée → ralentir
                self.state = "FIXED"
                self.blind_frames = 0
            else:
                throttle = self.fixed_speed
                self.state = "FIXED"
                self.blind_frames = 0
        else:
            if self.corner_mode:
                throttle = V_TURN
            elif n_blobs == 0:
                throttle = V_STOP; self.state = "STOP"
            elif n_blobs == 1:
                throttle = V_SLOW; self.state = "RECOVER"
            else:
                throttle = V_TURN + (V_MAX - V_TURN) * forward_clearance
                self.state = "TURN" if forward_clearance < 0.5 else "STRAIGHT"

        # ── Err smoothing → Kalman → Deadband ─────────────────────────────
        if err is not None:
            self.err_smooth = 0.65 * self.err_smooth + 0.35 * float(err)
            err = self.err_smooth
            # Kalman 1D : lisse davantage, robuste aux artefacts et dropouts
            # Reset si changement de signe brutal (évite inertie après turn_boost/CORNER)
            if self.kalman.x * float(err) < -100.0:
                self.kalman.reset()
            err = self.kalman.update(float(err))
            # Deadband ±6px : élimine les micro-oscillations en ligne droite
            if abs(err) < 6.0:
                err = 0.0
        else:
            self.kalman.reset()
            self.err_smooth = 0.0  # reset pour éviter inertie au retour de vision

        self.prev_n_blobs = n_blobs   # pour CORNER score frame suivante

        # ── Auto-calibration offset caméra ────────────────────────────────
        # err ici est DÉJÀ corrigé de l'offset → on accumule l'erreur résiduelle
        if n_blobs == 2 and not self.corner_mode and err is not None:
            raw_for_calib = float(err)
            self.calib_err_history.append(raw_for_calib)
            if len(self.calib_err_history) > 90:  # ~15s à 6fps
                self.calib_err_history.pop(0)
            # EMA très lente : ne corrige que les biais persistants (pas les vraies erreurs)
            if len(self.calib_err_history) >= 30 and abs(self.auto_offset) < 15:
                recent_mean = sum(self.calib_err_history[-30:]) / 30.0
                if abs(recent_mean) > 8.0:  # biais > 8px → apprendre
                    self.auto_offset += 0.01 * recent_mean  # EMA encore plus lente

        # ── Calibration manuelle via HTTP /calibrate ──────────────────────
        if _calibrate_request[0] and n_blobs == 2 and len(self.calib_err_history) >= 10:
            _calibrate_request[0] = False
            residual = int(round(sum(self.calib_err_history[-20:]) / float(min(len(self.calib_err_history), 20))))
            # N'inclut PAS auto_offset : évite l'accumulation de bruit après plusieurs /calibrate
            new_offset = CAMERA_OFFSET_PX + residual
            CAMERA_OFFSET_PX = new_offset
            self.auto_offset  = 0.0   # reset : la calibration a absorbé le biais
            self.servo_bias   = 0.0   # reset : repart de zéro après calibration manuelle
            self.calib_err_history = []
            _calibrate_result[0] = new_offset
            print("[calib] CAMERA_OFFSET_PX={:+d}px (residuel={:+d}, auto+servo reset)".format(new_offset, residual))

        # ── Steering ───────────────────────────────────────────────────────
        if self.state == "BLIND":
            steering = 0.0          # en BLIND complet : ne pas dériver sur prev_err
            self.prev_err = 0.0     # reset pour éviter spike au retour de vision
            self.err_smooth = 0.0   # reset smooth aussi
        elif err is None:
            steering = 0.0
        else:
            steering = self._pd(float(err))  # offset déjà soustrait à la source
            # b=1 : estimation moins fiable → braquage max limité pour éviter crash mur
            if n_blobs == 1:
                steering = max(-0.65, min(0.65, steering))
        # Rate limiter anti-oscillation : max delta 0.13/frame hors CORNER/transitions
        if self.state not in ("CORNER", "CORNER_EXIT", "COAST", "BLIND"):
            _ds = steering - self.last_steering_cmd
            if abs(_ds) > 0.13:
                steering = self.last_steering_cmd + (0.13 if _ds > 0 else -0.13)
        self.last_steering_cmd = steering   # mémoriser pour INERTIAL_COAST

        # ── Fading sortie CORNER : blend progressif → PD normal ────────────
        if self.corner_release > 0:
            # Q6 : Cascade V/U — gyro encore fort → re-CORNER immédiat
            if abs(gyro_z_cal) > 1.0 and gyro_z_cal * self.corner_dir > 0:
                self.corner_mode      = True
                self.corner_count     = CORNER_DURATION
                self.corner_imu_angle = 0.0
                self.corner_imu_t     = time.time()
                self.corner_release   = 0
                print("[ctrl] CASCADE_V gyro={:.2f} dir={:+.0f}".format(gyro_z_cal, self.corner_dir))
            # Accum signal fort pendant le fading → épingle détectée → re-CORNER sans attendre
            elif self.corner_accum >= 5:
                self.corner_mode      = True
                self.corner_count     = CORNER_DURATION
                self.corner_imu_angle = 0.0
                self.corner_imu_t     = time.time()
                self.corner_release   = 0
                self.corner_accum     = 0
                print("[ctrl] CASCADE_ACCUM pendant fading dir={:+.0f}".format(self.corner_dir))
            else:
                blend = float(self.corner_release) / 6.0
                steering = blend * self.last_corner_steer + (1.0 - blend) * steering
                throttle = throttle * 0.70  # sortie virage lente → évite crash mur
                self.corner_release -= 1
                if self.corner_release == 0 and n_blobs < 2:
                    self.search_frames = U_SEARCH_FRAMES if self.is_u_turn else 10
                self.state = "CORNER_EXIT"
                self.last_steering_cmd = steering

        # ── SEARCH : après CORNER_EXIT si b<2, maintenir cap dans direction du virage ──
        if (not self.corner_mode and self.corner_release == 0
                and self.search_frames > 0 and n_blobs < 2):
            _s_factor = 0.35 if self.is_u_turn else 0.40
            _t_factor = 0.55 if self.is_u_turn else 0.60
            steering = self.last_corner_steer * _s_factor
            throttle = self.fixed_speed * _t_factor
            self.search_frames -= 1
            self.state = "SEARCH"
        elif self.search_frames > 0 and n_blobs >= 2:
            self.search_frames = 0  # 2 lignes visibles → sortir de SEARCH immédiatement
            self.is_u_turn = False  # reset U-turn après retour vision complète

        # ── Servo bias : apprentissage biais mécanique châssis ────────────────
        # Si err≈0 (voiture centrée) mais steer≠0, c'est un défaut physique du servo
        if n_blobs == 2 and err is not None and abs(float(err)) < 8:
            self._bias_samples.append(steering)
            if len(self._bias_samples) >= 80:
                mean_steer = sum(self._bias_samples[-40:]) / 40.0
                if abs(mean_steer) > 0.02:  # biais significatif
                    # Convertir steering résiduel en pixels d'offset (steer = kp * err_manquant)
                    bias_px = mean_steer / max(0.006, KP)
                    self.servo_bias = 0.95 * self.servo_bias + 0.05 * bias_px
                    self.servo_bias = max(-60.0, min(60.0, self.servo_bias))
                    print("[servo_bias] {:.1f}px (steer_moyen={:.3f})".format(self.servo_bias, mean_steer))
                self._bias_samples = []
        else:
            if len(self._bias_samples) > 10:
                self._bias_samples = []  # reset si on quitte la zone stable

        info = {
            "err": err, "steering": steering, "throttle": throttle,
            "state": self.state, "n_blobs": n_blobs,
            "n_pred": n_pred,
            "pred_left_cx": pred_left_cx,
            "pred_right_cx": pred_right_cx,
            "forward_clearance": forward_clearance,
            "blobs_cx": [],
            "corner": corner_blob is not None,
            "ray_asym": round(ray_asym, 2),
            "scan_pts": scan_pts,
            "rejected_blobs": rejected_blobs,
            "hist_left_cx": hist_l, "hist_right_cx": hist_r,
            "scan_left_hits": scan_left_hits,
            "scan_right_hits": scan_right_hits,
            "mask_rejected": mask_rejected,
            "mask_clean": mask_clean,
        }
        return steering, throttle, info


# ══════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",         default="/dev/ttyACM0")
    p.add_argument("--baud",         type=int, default=115200)
    p.add_argument("--level",        type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--fixed-speed",  type=float, default=None,
                   help="Vitesse constante [0-1] — bypass machine a etats (calibration)")
    p.add_argument("--stream-port",  type=int, default=5601,
                   help="Port stream MJPEG (0 = desactive)")
    p.add_argument("--record",       default=None, metavar="FILE",
                   help="Enregistre la piste dans un CSV (ex: /tmp/track.csv)")
    p.add_argument("--replay",       default=None, metavar="FILE",
                   help="Rejoue un CSV enregistré comme feedforward de trajectoire")
    p.add_argument("--replay-weight", type=float, default=0.70,
                   help="Poids du feedforward replay [0-1] (défaut: 0.70)")
    p.add_argument("--cam-crop-top", type=float, default=0.0,
                   help="Crop logiciel du haut de l'image [0.0-0.6] avant traitement "
                        "(simule inclinaison + zoom). Ex: 0.35 = enlever 35%% du haut.")
    p.add_argument("--roi-far", type=float, default=None,
                   help="Override ROI_FAR [0.0-0.9] — fraction du haut ignoree pour le masque. "
                        "Defaut: 0.65 sans crop, auto-ajuste si --cam-crop-top.")
    p.add_argument("--camera-offset-px", type=int, default=0,
                   help="Biais lateral camera en pixels (+ = camera decalee a droite, "
                        "- = camera decalee a gauche). Calibrer en posant la voiture "
                        "au centre de la piste et ajuster jusqu'a err=0.")
    p.add_argument("--steering-max", type=float, default=None,
                   help="Override STEERING_MAX [0.3-1.0] (defaut: 0.85).")
    p.add_argument("--no-corner",   action="store_true",
                   help="Desactive la detection de coin L (mode ligne droite / test).")
    p.add_argument("--mapping",     action="store_true",
                   help="Phase 1 : enregistre un tour de piste (IMU) -> track_map.json")
    p.add_argument("--racing",      action="store_true",
                   help="Phase 2 : charge track_map.json et pilote en anticipation IMU")
    p.add_argument("--map-file",    default="track_map.json",
                   help="Chemin du fichier carte (defaut: track_map.json)")
    return p.parse_args()


def load_replay(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "steering": float(row["steering"]),
                "throttle": float(row["throttle"]),
                "err":      float(row["err"]) if row["err"] != "None" else 0.0,
            })
    print("[replay] {} frames chargees depuis {}".format(len(data), path))
    return data


def run(args):
    global ROI_FAR, ROI_MID, ROI_NEAR
    if args.roi_far is not None:
        ROI_FAR = args.roi_far
    elif args.cam_crop_top > 0:
        # Avec crop logiciel, abaisser automatiquement le ROI
        # Ex: crop=0.35 → ROI_FAR=0.65-0.35=0.30 (voir le haut de la zone zoomée)
        ROI_FAR = max(0.20, ROI_FAR - args.cam_crop_top)
    ROI_MID  = min(0.95, ROI_FAR + 0.25)
    ROI_NEAR = min(0.97, ROI_FAR + 0.48)
    print("[ctrl] ROI_FAR={:.2f} ROI_MID={:.2f} ROI_NEAR={:.2f}".format(
        ROI_FAR, ROI_MID, ROI_NEAR))

    ctrl = PDController(level=args.level, fixed_speed=args.fixed_speed, no_corner=args.no_corner)

    # ── Modules cartographie / navigation prédictive ──────────────────
    mapper    = None
    navigator = None
    if _TRACK_MODULES_OK:
        if args.mapping:
            mapper = TrackMapper()
            mapper.start()
            print("[track] Mode MAPPING actif -> {}".format(args.map_file))
        elif args.racing:
            navigator = TrackNavigator()
            ok = navigator.load(args.map_file)
            if not ok:
                print("[track] WARN: carte non chargée — mode racing désactivé")
                navigator = None
            else:
                print("[track] Mode RACING actif")

    # Mode replay : charge le CSV de piste enregistrée
    replay_data = None
    if args.replay:
        replay_data = load_replay(args.replay)

    # Mode record : ouvre le CSV en écriture
    record_file = None
    record_writer = None
    if args.record:
        record_file = open(args.record, "w", newline="")
        record_writer = csv.DictWriter(record_file,
            fieldnames=["frame", "t", "err", "steering", "throttle", "state", "blobs"])
        record_writer.writeheader()
        print("[record] Enregistrement → {}".format(args.record))

    if args.stream_port > 0:
        start_stream_server(args.stream_port)

    vesc = None
    if not args.dry_run:
        if VescInterface is None:
            print("[ctrl] ERREUR : vesc_interface non disponible"); sys.exit(1)
        vesc = VescInterface(port=args.port, baudrate=args.baud,
                             current_max=CURRENT_MAX,
                             throttle_mode="duty", max_duty=0.50,
                             invert_motor=False)
        print("[ctrl] VESC connecte sur {}".format(args.port))
    else:
        print("[ctrl] DRY-RUN — VESC non commande")

    global CAMERA_OFFSET_PX, STEERING_MAX
    if args.camera_offset_px != 0:
        CAMERA_OFFSET_PX = args.camera_offset_px
    if args.steering_max is not None:
        STEERING_MAX = args.steering_max
    speed_str = "fixed={:.2f}".format(args.fixed_speed) if args.fixed_speed else "adaptatif"
    print("[ctrl] Niveau {} | {} | KP={} | offset={}px | steer_max={}".format(
        args.level, speed_str, KP, CAMERA_OFFSET_PX, STEERING_MAX))

    # Watchdog : thread qui surveille les frames et force un reset si caméra gelée
    def _watchdog():
        FRAME_TIMEOUT = 30.0  # secondes sans frame → reset forcé (boot MyriadX prend ~15-20s)
        while True:
            time.sleep(3)
            if time.time() - _last_frame_time[0] > FRAME_TIMEOUT:
                print("[watchdog] Aucune frame depuis {:.0f}s — reset USB force".format(
                    time.time() - _last_frame_time[0]))
                _watchdog_trigger[0] = True
                _usb_reset_oak()
                _last_frame_time[0] = time.time()  # reset le timer pour ne pas boucler

    wt = threading.Thread(target=_watchdog, daemon=True)
    wt.start()

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

            # IMU node (gyroscope pour mapping/racing)
            imu_node = pipeline.create(dai.node.IMU)
            imu_node.enableIMUSensor([dai.IMUSensor.GYROSCOPE_RAW], 100)
            imu_node.setBatchReportThreshold(1)
            imu_node.setMaxBatchReports(10)
            imu_xout = pipeline.create(dai.node.XLinkOut)
            imu_xout.setStreamName("imu")
            imu_node.out.link(imu_xout.input)

            # ── Détection + recovery automatique du device OAK-D ──────────────
            # Séquence : getAllConnectedDevices → si UNBOOTED → hub rebind →
            #            bootMemory → device passe en BOOTLOADER → pipeline OK
            _last_frame_time[0] = time.time()  # neutralise le watchdog pendant recovery

            _all_devs = dai.Device.getAllConnectedDevices()
            _dev_info = _all_devs[0] if _all_devs else None
            _did_boot_memory = False
            _forced_unbooted = False  # True si DeviceInfo créé manuellement (état inconnu mais UNBOOTED réel)

            if _dev_info is None or "UNBOOTED" in str(getattr(_dev_info, 'state', '')):
                # Device invisible ou UNBOOTED : tenter hub rebind pour forcer ré-énumération
                print("[ctrl] Device invisible/UNBOOTED — hub rebind 1-2.1...")
                _last_frame_time[0] = time.time()
                _usb_reset_method4_parent_hub()
                _all_devs = dai.Device.getAllConnectedDevices()
                if not _all_devs:
                    _all_devs = dai.DeviceBootloader.getAllAvailableDevices()
                _dev_info = _all_devs[0] if _all_devs else None

            if _dev_info is None:
                # XLink voit le device (warning "skipping UNBOOTED") mais ne le retourne pas.
                # Fallback : DeviceInfo hardcodé — le nom 1.2.1.4 est fixe sur ce hardware.
                try:
                    _dev_info = dai.DeviceInfo("1.2.1.4")
                    _forced_unbooted = True  # on sait qu'il est UNBOOTED (raison du skip)
                    print("[ctrl] Fallback DeviceInfo hardcode 1.2.1.4")
                except Exception as _fe:
                    raise RuntimeError("Device OAK-D introuvable: {}".format(_fe))

            print("[ctrl] Device: {0} state={1}".format(_dev_info.getMxId(), _dev_info.state))

            # Si UNBOOTED : bootMemory via subprocess Python isolé + os.execve pour XLink clean
            # XLink d'un long-running process est définitivement "corrodé" après resets.
            # Stratégie : subprocess bootMemory (XLink vierge) → device BOOTLOADER
            #             → os.execve pour relancer ce script avec XLink frais → pipeline OK
            # La var env OAKD_POST_RECOVERY=1 évite la boucle infinie :
            #   - 1er process (non marqué) : fait le bootMemory + execve
            #   - 2ème process (post-recovery) : trouve BOOTLOADER → ouvre normalement
            _state_str = str(getattr(_dev_info, 'state', ''))
            _recovery_count = int(os.environ.get('OAKD_POST_RECOVERY', '0'))
            _post_recovery = _recovery_count >= 1

            # execve seulement si : (a) on vient de faire bootMemory, OU
            # (b) device déjà BOOTLOADER sans recovery en cours (cas rare)
            # En post-recovery avec device BOOTLOADER → ouvrir directement, pas d'execve
            # En post-recovery avec device UNBOOTED → refaire bootMemory + execve (count < 3)
            _need_execve = False
            if "BOOTLOADER" in _state_str and not _post_recovery:
                _need_execve = True  # déjà BOOTLOADER avant toute tentative → XLink frais requis

            if ("UNBOOTED" in _state_str or _forced_unbooted) and _recovery_count < 3:
                # Device UNBOOTED → subprocess bootMemory pour le passer en BOOTLOADER
                print("[ctrl] UNBOOTED → bootMemory subprocess...")
                import subprocess as _sp
                _bootmem_code = (
                    "import depthai as dai, time, sys\n"
                    "bls = []\n"
                    "for _r in range(6):\n"
                    "    bls = dai.DeviceBootloader.getAllAvailableDevices()\n"
                    "    if not bls: bls = dai.Device.getAllConnectedDevices()\n"
                    "    if bls: break\n"
                    "    time.sleep(3)\n"
                    "if not bls:\n"
                    "    try:\n"
                    "        bls = [dai.DeviceInfo('1.2.1.4')]\n"
                    "        print('BL hardcode fallback')\n"
                    "    except: sys.exit(1)\n"
                    "bl = dai.DeviceBootloader(bls[0], allowFlashingBootloader=True)\n"
                    "print('BL v'+str(bl.getVersion()))\n"
                    "fw = dai.DeviceBootloader.getEmbeddedBootloaderBinary("
                    "dai.DeviceBootloader.Type.USB)\n"
                    "bl.bootMemory(fw)\n"
                    "del bl\n"
                    "print('bootMemory_OK')\n"
                    "time.sleep(2)\n"
                    "devs = dai.Device.getAllConnectedDevices()\n"
                    "print(str(devs[0].state) if devs else 'empty_after')\n"
                )
                _env_bm = dict(os.environ)
                _env_bm['OPENBLAS_CORETYPE'] = 'ARMV8'
                try:
                    _last_frame_time[0] = time.time()
                    _proc = _sp.Popen(['python3', '-c', _bootmem_code],
                                      stdout=_sp.PIPE, stderr=_sp.PIPE, env=_env_bm)
                    _out, _err = _proc.communicate(timeout=90)
                    _out_str = _out.decode().strip()
                    print("[ctrl] bootMem: {}".format(_out_str))
                    if _err:
                        _err_tail = _err.decode()[-200:]
                        if 'warning' not in _err_tail.lower():
                            print("[ctrl] bootMem stderr: {}".format(_err_tail))
                    _did_boot_memory = (b"bootMemory_OK" in _out
                                        or _proc.returncode == 0)
                    _need_execve = True
                except Exception as _bme:
                    print("[ctrl] bootMemory subprocess erreur: {0}".format(_bme))
                    _need_execve = False

            # execve seulement si le bootMemory a réussi OU si le device était déjà BOOTLOADER
            # (évite d'execve en boucle si le subprocess a échoué sans avoir booté quoi que ce soit)
            _can_execve = _did_boot_memory or ("BOOTLOADER" in _state_str)
            if _need_execve and _can_execve:
                # Device en BOOTLOADER post-bootMemory → XLink corrodé → execve pour XLink propre
                _env_exec = dict(os.environ)
                _env_exec['OAKD_POST_RECOVERY'] = str(_recovery_count + 1)
                _env_exec['OPENBLAS_CORETYPE'] = 'ARMV8'
                print("[ctrl] Restart process XLink-clean (OAKD_POST_RECOVERY={})...".format(
                    _recovery_count + 1))
                os.execve(sys.executable, [sys.executable, '-u'] + sys.argv, _env_exec)
                # os.execve ne revient pas

            # Ouverture pipeline — toujours passer _dev_info explicitement (USB 2.0)
            # En post-recovery _dev_info pointe sur le device BOOTLOADER trouvé ci-dessus
            if _post_recovery:
                print("[ctrl] Post-recovery pipeline (dev={0} state={1})...".format(
                    _dev_info.getMxId(), _dev_info.state))
            _device_ctx = dai.Device(pipeline, _dev_info, True)

            with _device_ctx as device:
                q        = device.getOutputQueue("preview", maxSize=1, blocking=False)
                imu_q    = device.getOutputQueue("imu",     maxSize=50, blocking=False)
                _last_gyro_z = [0.0]   # partagé entre lecture IMU et boucle vision
                if attempt > 1:
                    _camera_restarted[0] = True  # signale au controller de reset Kalman
                attempt = 0
                t0 = time.time(); frame_n = 0

                while True:
                    pkt = q.get()
                    bgr = pkt.getCvFrame()

                    # Crop logiciel + zoom : simule inclinaison + focale augmentée
                    if args.cam_crop_top > 0:
                        y0 = int(CAM_H * args.cam_crop_top)
                        bgr = cv2.resize(bgr[y0:, :], (CAM_W, CAM_H),
                                         interpolation=cv2.INTER_LINEAR)

                    mask = white_line_mask(
                        bgr, hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
                        morph_k=5, blur_k=3, use_clahe=True, min_area=MIN_BLOB_AREA,
                    )
                    # Masque large (ROI 45%) pour détection anticipée des coins
                    mask_wide = mask.copy()
                    mask_wide[:int(CAM_H * 0.45), :] = 0
                    # Masque normal (ROI 65%) pour lignes sans bruit fond
                    mask[:int(CAM_H * ROI_FAR), :] = 0

                    # ── Lecture IMU (gyro_z pour mapping/racing) ──────────────
                    imu_data = imu_q.tryGet()
                    if imu_data:
                        for pkt in imu_data.packets:
                            _last_gyro_z[0] = pkt.gyroscope.z

                    steering, throttle, info = ctrl.compute(mask, bgr, mask_wide,
                                                              gyro_z=_last_gyro_z[0])

                    # ── MODE MAPPING : enregistrement de la piste ─────────────
                    if mapper is not None:
                        mapper.process_frame(_last_gyro_z[0])
                        if _finish_map_request[0]:
                            _finish_map_request[0] = False
                            mapper.save_map(args.map_file)
                            mapper.summary()

                    # ── MODE RACING : anticipation IMU prédictive ─────────────
                    if navigator is not None:
                        nav = navigator.update(
                            _last_gyro_z[0], info["n_blobs"], info["err"])
                        # Vision stable → réduit drift odométrique
                        if info["n_blobs"] == 2 and info["err"] is not None:
                            navigator.notify_vision_stable(info["n_blobs"], info["err"])
                        # Sortir du fallback si vision récupérée
                        if info["n_blobs"] == 2:
                            navigator.recover_from_fallback()
                        # Fusion additive : PD + bias carte (jamais remplacement)
                        if nav["action"] != "FALLBACK" and nav["action"] != "STRAIGHT":
                            steering = steering + nav["steering_bias"]
                            steering = max(-STEERING_MAX, min(STEERING_MAX, steering))
                            throttle = throttle * nav["throttle_scale"]
                            info["state"] = "NAV_" + nav["action"].upper()[:8]

                    # ── MODE REPLAY : feedforward + correction PD ──────────────
                    if replay_data is not None:
                        idx = frame_n % len(replay_data)
                        ref = replay_data[idx]
                        w = args.replay_weight
                        # Feedforward (trajectoire mémorisée) + feedback (PD actuel)
                        steering = w * ref["steering"] + (1.0 - w) * steering
                        steering = max(-STEERING_MAX, min(STEERING_MAX, steering))
                        info["state"] = "REPLAY"

                    # ── MODE RECORD : enregistre la frame ─────────────────────
                    if record_writer is not None:
                        record_writer.writerow({
                            "frame":    frame_n,
                            "t":        round(time.time() - t0, 3),
                            "err":      info["err"],
                            "steering": round(steering, 4),
                            "throttle": round(throttle, 4),
                            "state":    info["state"],
                            "blobs":    info["n_blobs"],
                        })

                    # Mémoriser pour coast mode (crash caméra)
                    if abs(steering) < 0.5 and throttle > 0:
                        _coast_steer[0]    = steering
                        _coast_throttle[0] = throttle

                    if vesc is not None:
                        if _drive_enabled:
                            vesc.drive(steering, throttle)
                        else:
                            vesc.stop()

                    if args.stream_port > 0:
                        push_frame(bgr, info.get("mask_clean", mask), info, info.get("rejected_blobs"))
                    _last_frame_time[0] = time.time()

                    frame_n += 1
                    if frame_n % (CAM_FPS * 3) == 0:
                        fps = frame_n / (time.time() - t0)
                        mid = CAM_W // 2
                        cx_list = info["blobs_cx"]
                        lefts  = [x for x in cx_list if x < mid]
                        rights = [x for x in cx_list if x >= mid]
                        tw = str(min(rights) - max(lefts)) if lefts and rights else "?"
                        rec_str = " [REC {}f]".format(frame_n) if record_writer else ""
                        rep_str = " [REPLAY {}/{}]".format(frame_n % len(replay_data) if replay_data else 0,
                                                            len(replay_data) if replay_data else 0) if replay_data else ""
                        print("[ctrl] {:.0f}fps | err={} | steer={:.3f} | "
                              "thr={:.2f} | {} | blobs={} | cx={} | tw={}px | "
                              "off={:+.1f} sbias={:+.1f}{}{}".format(
                                  fps,
                                  int(info["err"]) if info["err"] is not None else "N/A",
                                  steering, throttle,
                                  info["state"], info["n_blobs"],
                                  cx_list, tw,
                                  ctrl.auto_offset, ctrl.servo_bias,
                                  rec_str, rep_str))

        except KeyboardInterrupt:
            print("[ctrl] Arret."); break
        except Exception as e:
            print("[ctrl] Erreur ({}) — {!s:.200}".format(type(e).__name__, e))
            print("[ctrl] coast mode + reset USB + reconnexion")
            _coast_crash_t[0] = time.time()
            _usb_reset_oak()
            delay = max(5, min(5 * attempt, 30))
            coast_s = float(_coast_steer[0])
            coast_t = float(_coast_throttle[0])
            # Coast mode : décroissance progressive pendant le reset USB
            elapsed = 0.0
            while elapsed < delay:
                elapsed = time.time() - _coast_crash_t[0]
                if vesc and _drive_enabled:
                    if elapsed < 0.5:
                        s = coast_s
                        t = coast_t
                    elif elapsed < 1.5:
                        s = coast_s * 0.90
                        t = coast_t * 0.90
                    elif elapsed < 2.5:
                        s = coast_s * 0.70
                        t = coast_t * 0.60
                    else:
                        try: vesc.stop()
                        except: pass
                        break
                    try: vesc.drive(s, t)
                    except: pass
                time.sleep(0.05)
            if vesc:
                try: vesc.stop()
                except: pass
            print("[ctrl] Reconnexion dans {}s restantes...".format(max(0, delay - elapsed)))

    if vesc:
        try: vesc.stop(); vesc.close()
        except: pass
    if record_file:
        record_file.flush(); record_file.close()
        print("[record] Sauvegarde terminee → {}".format(args.record))


if __name__ == "__main__":
    run(parse_args())

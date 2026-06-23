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

import sys, time, argparse, os, threading, struct, socket, csv, glob, fcntl
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

def _usb_reset_oak():
    """Escalade automatique des méthodes de reset USB selon le nombre d'échecs."""
    global _reset_attempt_total
    _reset_attempt_total += 1
    n = _reset_attempt_total
    print("[ctrl] USB reset OAK-D (tentative {})".format(n))
    # Alterner : 1, 2, 1, 3, 1, 2, 1, 3, ...
    if n % 4 == 2:
        ok = _usb_reset_method2_unbind_bind()
    elif n % 4 == 0:
        ok = _usb_reset_method3_ioctl()
    else:
        ok = _usb_reset_method1_authorized()
    if not ok:
        # Essayer les autres méthodes en cascade si la principale échoue
        for fn in [_usb_reset_method1_authorized, _usb_reset_method2_unbind_bind, _usb_reset_method3_ioctl]:
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

CAM_W, CAM_H = 512, 256
CAM_FPS      = 6

HSV_LOW      = np.array([0,   0, 150], dtype=np.uint8)   # V>=150 (adapté éclairage faible)
HSV_HIGH     = np.array([180, 45, 255], dtype=np.uint8)  # S<=45 (blanc incluant reflets tamisés)
ROI_FAR      = 0.65
ROI_MID      = 0.80
ROI_NEAR     = 0.92
ROI_BOTTOM   = 1.00
MIN_BLOB_AREA  = 800
MIN_CORNER_AREA = 1500
CORNER_DURATION = 15   # frames de maintien virage (~1.25s @ 12fps)

TRACK_WIDTH_EST_PX = 385

KP           = 0.012         # augmenté pour virages plus décisifs
KD           = 0.003
ALPHA_D      = 0.7
STEERING_MAX = 0.85
STEERING_DEADZONE = 0.03
CAMERA_OFFSET_PX = 0         # biais caméra — calibrer si la voiture dérive constamment

V_MAX        = 0.14          # → ~7% duty, minimum viable
V_TURN       = 0.14
V_SLOW       = 0.14
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
_drive_enabled = True   # contrôlé via HTTP /stop et /go

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
            self._send_text("RUNNING")
            print("[ctrl] /go recu")
            return
        if path == "/status":
            self._send_text("running" if _drive_enabled else "stopped")
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
    # overlay masque vert (blobs acceptés uniquement — lignes de piste)
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.5, 0)
    # Blobs REJETÉS en orange — artefacts filtrés (debug, ne touche pas l'algo)
    if rejected_blobs:
        for rb in rejected_blobs:
            x, yt, w, h = rb["rect"]
            cv2.rectangle(vis, (x, yt), (x + w, yt + h), (0, 128, 255), 1)
            cv2.putText(vis, rb["reason"], (x, max(yt - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 128, 255), 1)
    # Ligne verticale blanche = cible (la voiture doit rester ici)
    cv2.line(vis, (CAM_W // 2, int(CAM_H * ROI_FAR)), (CAM_W // 2, CAM_H), (255, 255, 255), 1)
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
    cv2.putText(vis,
        "err={} steer={:.2f} thr={:.2f} {}{} b={} ray={:+.2f}".format(
            err_str, info["steering"], info["throttle"],
            info["state"], corner_flag, info["n_blobs"],
            info.get("ray_asym", 0.0)),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    # vue droite : caméra masquée (seuls les pixels détectés visibles, reste noir)
    mask_bool = (mask > 0)
    panel = bgr.copy()
    panel[~mask_bool] = 0
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
        if area > 3000 and solidity > 0.65 and aspect < 2.5:
            rejected.append({"cx": cx, "cy": cy, "area": area, "reason": "cmp", "rect": rect})
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


def detect_corner_blob(mask):
    """Détecte le marqueur de coin L : blob compact (area >= MIN_CORNER_AREA, aspect < 1.8).
    Appliqué sur mask_wide (ROI 45%) pour voir le L bien avant d'y arriver.
    Retourne dict {cx, cy, area} ou None.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cy_min = int(CAM_H * 0.45)  # détecte coin L dans la moitié basse (élimine artefacts haut)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        asp  = w / float(h)
        cy   = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
        cx   = stats[i, cv2.CC_STAT_LEFT] + w // 2
        if area >= MIN_CORNER_AREA and asp < 1.8 and cy >= cy_min:
            if best is None or area > best["area"]:
                best = {"cx": cx, "cy": cy, "area": area, "aspect": round(asp, 1)}
    return best


def err_from_two_lines(blobs, track_width=None):
    mid_x = CAM_W // 2
    CLEAR_LEFT  = mid_x - 76
    CLEAR_RIGHT = mid_x + 76
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
    def __init__(self, level=3, fixed_speed=None):
        self.level        = level
        self.fixed_speed  = fixed_speed
        self.prev_err     = 0.0
        self.d_filtered   = 0.0
        self.state        = "STOP"
        self.err_history      = []    # dernières 6 erreurs pour tendance virage
        # ── Machine à états coin L ───────────────────────────────────────────
        self.corner_mode      = False
        self.corner_dir       = 0.0
        self.corner_count     = 0
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
        self.kalman           = _Kalman1D(q=0.05, r=20.0)
        # ── Dérivée temporelle correcte ───────────────────────────────────────
        self.last_pd_time     = time.time()
        # ── CORNER multi-signal (IA) ──────────────────────────────────────────
        self.prev_n_blobs     = 0     # pour détecter transition b=2→1
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
            kp = 0.020
        elif abs_err < 15:
            kp = 0.008
        else:
            kp = 0.008 + (0.020 - 0.008) * (abs_err - 15.0) / (50.0 - 15.0)
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

    def compute(self, mask, bgr, mask_wide=None):
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

        # ── Blob proximity tracker : rejette les sauts >80px vers des artefacts ─
        if blobs:
            lefts  = [b for b in blobs if b["cx"] < CAM_W // 2]
            rights = [b for b in blobs if b["cx"] >= CAM_W // 2]
            if lefts and self.last_left_cx is not None:
                best_l = min(lefts, key=lambda b: abs(b["cx"] - self.last_left_cx))
                if abs(best_l["cx"] - self.last_left_cx) > 80:
                    best_l = max(lefts, key=lambda b: b["area"])
                blobs = [best_l] + [b for b in blobs if b["cx"] >= CAM_W // 2]
            if rights and self.last_right_cx is not None:
                best_r = min(rights, key=lambda b: abs(b["cx"] - self.last_right_cx))
                if abs(best_r["cx"] - self.last_right_cx) > 80:
                    best_r = max(rights, key=lambda b: b["area"])
                blobs = [b for b in blobs if b["cx"] < CAM_W // 2] + [best_r]
            n_blobs = len(blobs)
            self.last_left_cx  = min(blobs, key=lambda b: b["cx"])["cx"] if blobs else None
            self.last_right_cx = max(blobs, key=lambda b: b["cx"])["cx"] if blobs else None
        else:
            self.last_left_cx = None
            self.last_right_cx = None

        # ── Machine à états CORNER — score multi-signal ≥ 3 requis (IA) ─────
        if not self.corner_mode:
            cscore = 0
            if corner_blob is not None:
                cscore += 2                                  # blob compact = signal fort
            if abs(ray_asym) > 0.35:
                cscore += 1                                  # asymétrie raycasts
            if self.prev_n_blobs == 2 and n_blobs <= 1:
                cscore += 1                                  # disparition soudaine d'une ligne

            if cscore >= 3:
                if corner_blob is not None:
                    corner_dir_cx = corner_blob["cx"]
                else:
                    corner_dir_cx = CAM_W // 2 + (1 if ray_asym > 0 else -1)
                self.corner_dir   = 1.0 if corner_dir_cx > CAM_W // 2 else -1.0
                self.corner_mode  = True
                self.corner_count = CORNER_DURATION
                print("[ctrl] CORNER score={} dir={:+.0f}".format(cscore, self.corner_dir))

        if self.corner_mode:
            self.corner_count -= 1
            if self.corner_count <= 0:
                self.corner_mode = False
                print("[ctrl] CORNER termine")
            err = self.corner_dir * 200.0
            self.state = "CORNER"
        else:
            # Mémoire de direction (mise à jour seulement, pas de boost)
            turn_boost = 0.0  # désactivé : causait des oscillations quand b=2

            # ── Méthode principale : deux lignes séparées (gauche/droite) ──
            # Largeur dynamique : médiane des 10 dernières mesures réelles (b=2)
            last_tw = None
            if len(self.track_widths) >= 3:
                last_tw = float(np.median(self.track_widths[-10:]))
            if self.level >= 3 and n_blobs >= 2:
                err_lines, tw = err_from_two_lines(blobs, track_width=last_tw)
                if tw is not None and tw > 100 and tw < CAM_W - 20:
                    self.track_widths.append(tw)
                    if len(self.track_widths) > 20:
                        self.track_widths.pop(0)
                err = err_lines
            elif n_blobs == 1:
                # b=1 : utilise la largeur dynamique pour mieux estimer la ligne manquante
                err_lines, _ = err_from_two_lines(blobs, track_width=last_tw)
                if err_lines is not None:
                    err = err_lines
            # ── Fallback : centroïde global du masque (point rouge) ────────
            if err is None:
                err = err_from_mask(mask)
            if err is None and n_blobs == 0:
                err = err_from_mask(m_wide)

            # Mémoire de tendance : maintient la direction si vision perdue
            if err is not None:
                self.err_history.append(float(err))
                if len(self.err_history) > 6:
                    self.err_history.pop(0)
                if abs(err) > 60:
                    self.last_turn_dir = 1.0 if err > 0 else -1.0
                    self.turn_memory_ctr = 15
            # Trend et mémoire : seulement si b=0 (pas de blob visible)
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
                throttle = self.fixed_speed * 0.60   # ralentir fort en virage
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
                    }
                    push_frame(bgr, mask, info, rejected_blobs)
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
            err = self.kalman.update(float(err))
            # Deadband ±6px : élimine les micro-oscillations en ligne droite
            if abs(err) < 6.0:
                err = 0.0
        else:
            self.kalman.reset()

        self.prev_n_blobs = n_blobs   # pour CORNER score frame suivante

        # ── Steering ───────────────────────────────────────────────────────
        if self.state == "BLIND":
            steering = 0.0          # en BLIND complet : ne pas dériver sur prev_err
            self.prev_err = 0.0     # reset pour éviter spike au retour de vision
            self.err_smooth = 0.0   # reset smooth aussi
        elif err is None:
            steering = 0.0
        else:
            steering = self._pd(float(err) - CAMERA_OFFSET_PX)
        self.last_steering_cmd = steering   # mémoriser pour INERTIAL_COAST

        info = {
            "err": err, "steering": steering, "throttle": throttle,
            "state": self.state, "n_blobs": n_blobs,
            "forward_clearance": forward_clearance,
            "blobs_cx": [b["cx"] for b in blobs[:4]],
            "corner": corner_blob is not None,
            "ray_asym": round(ray_asym, 2),
            "scan_pts": scan_pts,
            "rejected_blobs": rejected_blobs,
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

    ctrl = PDController(level=args.level, fixed_speed=args.fixed_speed)

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

    speed_str = "fixed={:.2f}".format(args.fixed_speed) if args.fixed_speed else "adaptatif"
    print("[ctrl] Niveau {} | {} | KP={} | offset={}px".format(
        args.level, speed_str, KP, CAMERA_OFFSET_PX))

    # Watchdog : thread qui surveille les frames et force un reset si caméra gelée
    def _watchdog():
        FRAME_TIMEOUT = 10.0  # secondes sans frame → reset forcé
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

            with dai.Device(pipeline, True) as device:
                q = device.getOutputQueue("preview", maxSize=1, blocking=False)
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

                    steering, throttle, info = ctrl.compute(mask, bgr, mask_wide)

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

                    if vesc is not None:
                        if _drive_enabled:
                            vesc.drive(steering, throttle)
                        else:
                            vesc.stop()

                    if args.stream_port > 0:
                        push_frame(bgr, mask, info, info.get("rejected_blobs"))
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
                              "thr={:.2f} | {} | blobs={} | cx={} | tw={}px{}{}".format(
                                  fps,
                                  int(info["err"]) if info["err"] is not None else "N/A",
                                  steering, throttle,
                                  info["state"], info["n_blobs"],
                                  cx_list, tw, rec_str, rep_str))

        except KeyboardInterrupt:
            print("[ctrl] Arret."); break
        except Exception as e:
            if vesc:
                try: vesc.stop()
                except: pass
            print("[ctrl] Erreur ({}) — reset USB + reconnexion".format(type(e).__name__))
            _usb_reset_oak()
            delay = max(5, min(5 * attempt, 30))
            print("[ctrl] Reconnexion dans {}s...".format(delay))
            time.sleep(delay)

    if vesc:
        try: vesc.stop(); vesc.close()
        except: pass
    if record_file:
        record_file.flush(); record_file.close()
        print("[record] Sauvegarde terminee → {}".format(args.record))


if __name__ == "__main__":
    run(parse_args())

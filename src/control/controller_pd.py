"""
controller_pd.py — Contrôleur PD + stream MJPEG intégré (port 5601)

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u -m src.control.controller_pd --level 3
  OPENBLAS_CORETYPE=ARMV8 python3 -u -m src.control.controller_pd --dry-run
  OPENBLAS_CORETYPE=ARMV8 python3 -u -m src.control.controller_pd --fixed-speed 0.22

  --dry-run      : vision seule, VESC non commandé
  --fixed-speed  : vitesse constante (bypass machine à états) — mode calibration
  --level N      : niveau contrôleur 1-4 (défaut : 3)
  --stream-port  : port HTTP stream MJPEG (défaut : 5601, 0 = désactivé)

Source vidéo : le hub caméra (SHM). controller_pd est un CLIENT du hub — il n'ouvre
jamais l'OAK-D lui-même (le hub possède la caméra et gère la reconnexion).
"""

import sys, time, argparse, os, threading, struct, socket, csv
import numpy as np
import cv2

from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))
from src.mask.white_line import white_line_mask
from src.mask.lane_detect import (LaneParams, get_blobs, detect_corner_blob,
                                  clean_mask_artifacts, find_lane_histogram,
                                  find_lane_scanlines)
from src.mask.perception_config import PerceptionProfile
try:
    from src.control.vesc_interface import VESCInterface as VescInterface
except ImportError:
    VescInterface = None



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
MIN_CORNER_AREA = 6000
CORNER_DURATION = 15   # frames de maintien virage (~1.25s @ 12fps)

TRACK_WIDTH_EST_PX = 280     # largeur réelle piste ~280px (CAM_W=512)
SLIDE_WIN    = 70            # fenêtre ±px pour sliding windows autour de la position précédente

KP           = 0.006         # réduit : 6fps = 167ms par frame, évite sur-braquage
KD           = 0.005         # augmenté : amortit l'oscillation due au retard visuel
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
_last_frame_time = [time.time()]   # heure de la dernière frame reçue (santé du flux)
_camera_restarted = [False]        # reconnexion flux → reset Kalman côté controller
_drive_enabled = True   # contrôlé via HTTP /stop et /go
_go_reset = [False]         # mis à True par /go → PDController reset son état CORNER/Kalman
_calibrate_request = [False]       # mis à True par /calibrate → PDController applique l'offset
_calibrate_result  = [None]        # renseigné par PDController avec la valeur appliquée

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
    cv2.putText(vis,
        "err={} steer={:.2f} thr={:.2f} {}{} b={} ray={:+.2f}".format(
            err_str, info["steering"], info["throttle"],
            info["state"], corner_flag, info["n_blobs"],
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

# --------------------------------------------------------------------------
# CONTRÔLEUR
# --------------------------------------------------------------------------


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
        self.auto_offset       = 0.0  # offset appris en ligne (EMA)
        # ── Servo bias : biais mécanique châssis ──────────────────────────────
        self.servo_bias        = 0.0  # offset px appris (steering résiduel quand err≈0)
        self._bias_samples     = []   # échantillons steering récents quand err≈0
        # ── Sliding windows : positions précédentes pour restreindre la recherche
        self.hist_prev_left    = None  # cx ligne gauche frame précédente
        self.hist_prev_right   = None  # cx ligne droite frame précédente
        # Paramètres de détection de lignes (globals runtime → objet explicite)
        self.lp = LaneParams(
            cam_w=CAM_W, cam_h=CAM_H, roi_far=ROI_FAR, roi_mid=ROI_MID, roi_near=ROI_NEAR,
            min_blob_area=MIN_BLOB_AREA, min_corner_area=MIN_CORNER_AREA,
            track_width_px=TRACK_WIDTH_EST_PX, slide_win=SLIDE_WIN,
        )
        # Signal d'espace libre : faisceau polaire (remplace le scan-colonne VisualRays).
        self.polar = PerceptionProfile(
            cam_width=CAM_W, cam_height=CAM_H,
            row_start_frac=ROI_FAR, row_end_frac=ROI_BOTTOM,
        ).polar_rays()

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

    def compute(self, mask, bgr, mask_wide=None):
        global CAMERA_OFFSET_PX
        if _go_reset[0]:
            _go_reset[0] = False
            self.corner_mode  = False
            self.corner_count = 0
            self.prev_n_blobs = 0
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
        rays    = self.polar.normalized(self.polar(mask)[0])
        blobs, rejected_blobs = get_blobs(mask, self.lp)
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
        corner_blob = detect_corner_blob(m_wide, self.lp)

        # ── Masque nettoyé : supprime artefacts (reflets, chaussures, murs) ──
        # Le masque brut reste pour la visu orange (blobs rejetés).
        # L'histogramme et les scanlines utilisent le masque propre.
        mask_clean, mask_rejected = clean_mask_artifacts(mask, bgr=bgr, p=self.lp)

        # ── Détection lignes : Histogramme sliding + Scanlines + Fusion ─────
        hist_l, hist_r, hist_lconf, hist_rconf = find_lane_histogram(
            mask_clean, prev_left=self.hist_prev_left, prev_right=self.hist_prev_right, p=self.lp)
        scan_l, scan_r, scan_rows, scan_left_hits, scan_right_hits = find_lane_scanlines(mask_clean, p=self.lp)

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

        # ── Machine à états CORNER — score multi-signal ≥ 4 requis ─────────
        if not self.corner_mode and not self.no_corner:
            cscore = 0
            if corner_blob is not None:
                cscore += 2                                  # blob compact = signal fort
            if abs(ray_asym) > 0.40:
                cscore += 1                                  # asymétrie raycasts
            if self.prev_n_blobs == 2 and n_blobs <= 1:
                cscore += 1                                  # disparition soudaine d'une ligne

            if cscore >= 4:
                if corner_blob is not None:
                    corner_dir_cx = corner_blob["cx"]
                else:
                    corner_dir_cx = CAM_W // 2 + (1 if ray_asym > 0 else -1)
                self.corner_dir   = 1.0 if corner_dir_cx > CAM_W // 2 else -1.0
                self.corner_mode  = True
                self.corner_count = CORNER_DURATION
                print("[ctrl] CORNER score={} dir={:+.0f}".format(cscore, self.corner_dir))

        # Offset effectif calculé ici pour être appliqué à l'erreur BRUTE
        effective_offset = CAMERA_OFFSET_PX + int(self.auto_offset) + int(self.servo_bias)

        if self.corner_mode:
            self.corner_count -= 1
            if self.corner_count <= 0:
                self.corner_mode = False
                print("[ctrl] CORNER termine")
            err = self.corner_dir * 200.0
            self.state = "CORNER"
        else:
            # ── Calcul erreur depuis positions fusionnées ──────────────────
            last_tw = float(np.median(self.track_widths[-10:])) if len(self.track_widths) >= 3 else float(TRACK_WIDTH_EST_PX)
            if left_cx is not None and right_cx is not None:
                center = (left_cx + right_cx) // 2
                err = center - CAM_W // 2 - effective_offset  # offset soustrait à la SOURCE
                tw = right_cx - left_cx
                if 100 < tw < CAM_W - 20:
                    self.track_widths.append(tw)
                    if len(self.track_widths) > 20:
                        self.track_widths.pop(0)
            elif left_cx is not None:
                est_right = min(left_cx + int(last_tw), CAM_W - 10)
                center = (left_cx + est_right) // 2
                err = center - CAM_W // 2 - effective_offset
                n_blobs = 1  # garde la trace pour limiter steer ensuite
            elif right_cx is not None:
                est_left = max(right_cx - int(last_tw), 10)
                center = (est_left + right_cx) // 2
                err = center - CAM_W // 2 - effective_offset
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
            # blobs=1 sans track_width fiable → estimation moins sûre → limiter steer
            if n_blobs == 1 and len(self.track_widths) < 3:
                steering = max(-0.40, min(0.40, steering))
        self.last_steering_cmd = steering   # mémoriser pour INERTIAL_COAST

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
    p.add_argument("--max-duty", type=float, default=0.50,
                   help="Duty cycle VESC maximal [0-1] (défaut 0.50).")
    p.add_argument("--gamepad", action="store_true",
                   help="Accepté pour compat profils ; la prise de main manette est gérée par "
                        "le superviseur core/, pas par ce worker (ignoré ici).")
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
                             throttle_mode="duty", max_duty=args.max_duty,
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

    # Pas de watchdog USB ici : le hub possède l'OAK-D et gère la reconnexion ; côté client,
    # la péremption des frames est gérée par le timeout du FrameClient.
    if args.gamepad:
        print("[ctrl] --gamepad ignoré : la prise de main manette est gérée par le superviseur core/.")

    def _drive_loop(get_frame):
        """Boucle de conduite indépendante de la source. get_frame() → prochaine frame BGR
        (lève sur échec : crash device en mode device, hub injoignable en mode hub).
        Tout le traitement (crop → masque → PD → VESC → stream) vit ICI, une seule fois ;
        les deux sources (OAK-D direct, hub SHM) ne font que fournir get_frame."""
        t0 = time.time(); frame_n = 0
        while True:
            bgr = get_frame()

            # Crop logiciel + zoom : simule inclinaison + focale augmentée
            if args.cam_crop_top > 0:
                y0 = int(CAM_H * args.cam_crop_top)
                bgr = cv2.resize(bgr[y0:, :], (CAM_W, CAM_H),
                                 interpolation=cv2.INTER_LINEAR)

            # Masque brut : les filtres agressifs (top-hat, rectilinéarité) sont OFF ici,
            # car ce contrôleur a son propre nettoyage en aval (clean_mask_artifacts).
            mask = white_line_mask(
                bgr, v_min_floor=int(HSV_LOW[2]), s_max=int(HSV_HIGH[1]),
                morph_k=5, min_area=MIN_BLOB_AREA, tophat_k=0, min_elongation=1.0,
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

    # Source unique = hub caméra (SHM). controller_pd est un client : il n'ouvre jamais
    # l'OAK-D. Le hub possède la caméra et gère la reconnexion.
    from src.cam.hub import FrameClient, ensure_hub_or_prompt
    if not ensure_hub_or_prompt():
        if vesc:
            try: vesc.stop(); vesc.close()
            except: pass
        return
    client = FrameClient()
    client.connect()
    print("[ctrl] source = hub (SHM /dev/shm/robocar_cam_color)")
    try:
        # copy=True : le traitement (CLAHE+morpho) peut dépasser la fenêtre de relap du ring
        # (~132 ms) ; on copie la frame pour ne jamais conduire sur une frame déchirée.
        _drive_loop(lambda: client.get(copy=True)[1])
    except KeyboardInterrupt:
        print("[ctrl] Arret.")
    except Exception as e:
        print("[ctrl] Hub injoignable ({}) — arret moteur.".format(type(e).__name__))
    finally:
        client.close()

    if vesc:
        try: vesc.stop(); vesc.close()
        except: pass
    if record_file:
        record_file.flush(); record_file.close()
        print("[record] Sauvegarde terminee → {}".format(args.record))


if __name__ == "__main__":
    run(parse_args())

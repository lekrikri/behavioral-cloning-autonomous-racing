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
# PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CAM_W, CAM_H = 512, 256
CAM_FPS      = 12

HSV_LOW      = np.array([0,   0, 155], dtype=np.uint8)   # 195→155 : éclairage salle
HSV_HIGH     = np.array([180, 55, 255], dtype=np.uint8)  # S 40→55 pour capturer lignes ternes
ROI_FAR      = 0.65   # ignorer 65% du haut
ROI_MID      = 0.75
ROI_NEAR     = 0.87
ROI_BOTTOM   = 1.00
MIN_BLOB_AREA = 700          # 300→700 : éliminer faux blobs (fenêtre, mur)

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

    def do_GET(self):
        global _drive_enabled
        if self.path == "/stop":
            _drive_enabled = False
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"STOPPED"); print("[ctrl] /stop recu"); return
        if self.path == "/go":
            _drive_enabled = True
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"RUNNING"); print("[ctrl] /go recu"); return
        if self.path not in ("/", "/stream"):
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


def push_frame(bgr, mask, info):
    global _latest_jpeg, _frame_id
    vis = bgr.copy()
    # overlay masque vert
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.5, 0)
    # centroïde rouge
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(vis, (CAM_W // 2, CAM_H - 1), (cx, cy), (255, 0, 0), 1)
    # lignes bandes
    for frac, color in [(ROI_NEAR, (255, 200, 0)), (ROI_MID, (0, 200, 255)), (ROI_FAR, (0, 100, 255))]:
        cv2.line(vis, (0, int(CAM_H * frac)), (CAM_W, int(CAM_H * frac)), color, 1)
    # texte
    err_str = "{:+d}".format(int(info["err"])) if info["err"] is not None else "N/A"
    cv2.putText(vis,
        "err={} steer={:.2f} thr={:.2f} {} blobs={}".format(
            err_str, info["steering"], info["throttle"],
            info["state"], info["n_blobs"]),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    # miniature masque
    mini_h = CAM_H // 2
    mini = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (CAM_W, mini_h))
    pad  = np.zeros((CAM_H - mini_h, CAM_W, 3), dtype=np.uint8)
    panel = np.vstack([mini, pad])
    display = np.hstack([vis, panel])
    _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
    with _stream_lock:
        _latest_jpeg = jpg.tobytes()
        _frame_id   += 1


# ══════════════════════════════════════════════════════════════════════════════
# VISION
# ══════════════════════════════════════════════════════════════════════════════

def get_blobs(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Les vrais blobs de lignes ont leur cy dans les 60% bas du masque
    # Les faux blobs (mur, fenêtre) ont leur cy dans les 40% hauts → filtrés
    cy_min = int(CAM_H * 0.60)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BLOB_AREA:
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            cy = stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] // 2
            if cy >= cy_min:   # ignorer blobs trop hauts dans l'image
                blobs.append({"cx": cx, "cy": cy, "area": area})
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs


def err_from_mask(mask):
    M = cv2.moments(mask)
    if M["m00"] < 1:
        return None
    return int(M["m10"] / M["m00"]) - CAM_W // 2


def err_from_bands(mask):
    row_near = int(CAM_H * ROI_NEAR)
    row_mid  = int(CAM_H * ROI_MID)
    row_far  = int(CAM_H * ROI_FAR)
    mask_near = mask.copy(); mask_near[:row_near, :] = 0
    mask_mid  = mask.copy(); mask_mid[row_near:, :] = 0; mask_mid[:row_mid, :] = 0
    mask_far  = mask.copy(); mask_far[row_mid:, :] = 0;  mask_far[:row_far, :] = 0
    return err_from_mask(mask_near), err_from_mask(mask_mid), err_from_mask(mask_far)


def err_from_two_lines(blobs):
    mid_x = CAM_W // 2
    # Zone "clairement gauche" : cx < 180 | "clairement droite" : cx > 332
    # Entre 180-332 (zone centrale ~150px) → blob ambigu, on ignore l'estimation TRACK_WIDTH
    CLEAR_LEFT  = mid_x - 76   # 180px
    CLEAR_RIGHT = mid_x + 76   # 332px
    left_blobs  = [b for b in blobs if b["cx"] < CLEAR_LEFT]
    right_blobs = [b for b in blobs if b["cx"] > CLEAR_RIGHT]
    left  = max(left_blobs,  key=lambda b: b["area"]) if left_blobs  else None
    right = max(right_blobs, key=lambda b: b["area"]) if right_blobs else None
    if left and right:
        center = (left["cx"] + right["cx"]) // 2
        return center - mid_x, right["cx"] - left["cx"]
    if left:
        est_right = left["cx"] + TRACK_WIDTH_EST_PX
        return (left["cx"] + est_right) // 2 - mid_x, TRACK_WIDTH_EST_PX
    if right:
        est_left = right["cx"] - TRACK_WIDTH_EST_PX
        return (est_left + right["cx"]) // 2 - mid_x, TRACK_WIDTH_EST_PX
    # Blob ambigu au centre → None, le fallback err_from_mask prend le relais
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ══════════════════════════════════════════════════════════════════════════════

class PDController:
    def __init__(self, level=3, fixed_speed=None):
        self.level        = level
        self.fixed_speed  = fixed_speed
        self.prev_err     = 0.0
        self.d_filtered   = 0.0
        self.state        = "STOP"
        self.err_history  = []   # dernières 6 erreurs pour tendance virage
        self.vr           = VisualRays(
            img_width=CAM_W, img_height=CAM_H,
            row_band=(ROI_FAR, ROI_BOTTOM), morph_k=5,
        )

    def _pd(self, err):
        d_raw = err - self.prev_err
        self.d_filtered = ALPHA_D * self.d_filtered + (1.0 - ALPHA_D) * d_raw
        self.prev_err = err
        raw = KP * err + KD * self.d_filtered
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

    def compute(self, mask, bgr):
        rays    = self.vr(bgr)
        blobs   = get_blobs(mask)
        n_blobs = len(blobs)
        forward_clearance = float(np.mean(rays[8:12]))
        err = None

        if self.level >= 3:
            err, _ = err_from_two_lines(blobs)
        if err is None and self.level >= 2:
            err_near, err_mid, err_far = err_from_bands(mask)
            err = self._combined_err(err_near, err_mid, err_far, rays)
        if err is None:
            err = err_from_mask(mask)

        # Mémoire de tendance : si vision perdue en virage, on maintient la direction
        if err is not None:
            self.err_history.append(float(err))
            if len(self.err_history) > 6:
                self.err_history.pop(0)
        trend = sum(self.err_history) / len(self.err_history) if self.err_history else 0.0
        # Si err proche de 0 mais tendance forte → utiliser la tendance
        if err is None or abs(err) < 25:
            err = trend * 0.6

        # Vitesse — fixe si --fixed-speed, sinon adaptatif
        # En fixed-speed : ne jamais s'arrêter complètement (garde le cap en virage)
        if self.fixed_speed is not None:
            throttle = self.fixed_speed   # avance toujours, même sans blob
            self.state = "FIXED" if n_blobs > 0 else "BLIND"
        else:
            if n_blobs == 0:
                throttle = V_STOP; self.state = "STOP"
            elif n_blobs == 1:
                throttle = V_SLOW; self.state = "RECOVER"
            else:
                throttle = V_TURN + (V_MAX - V_TURN) * forward_clearance
                self.state = "TURN" if forward_clearance < 0.5 else "STRAIGHT"

        # Steering avec correction offset caméra
        if err is None:
            # Pas de ligne visible : garde le dernier steering (inertie)
            steering = self.prev_err * KP  # steering résiduel depuis dernière erreur
        else:
            steering = self._pd(float(err) - CAMERA_OFFSET_PX)

        info = {
            "err": err, "steering": steering, "throttle": throttle,
            "state": self.state, "n_blobs": n_blobs,
            "forward_clearance": forward_clearance,
            "blobs_cx": [b["cx"] for b in blobs[:4]],
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

                    mask = white_line_mask(
                        bgr, hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
                        morph_k=5, blur_k=3, use_clahe=True, min_area=MIN_BLOB_AREA,
                    )
                    mask[:int(CAM_H * ROI_FAR), :] = 0

                    steering, throttle, info = ctrl.compute(mask, bgr)

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
                        push_frame(bgr, mask, info)

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
            delay = min(3 * attempt, 15)
            print("[ctrl] Erreur ({}) — reconnexion dans {}s".format(type(e).__name__, delay))
            time.sleep(delay)

    if vesc:
        try: vesc.stop(); vesc.close()
        except: pass
    if record_file:
        record_file.flush(); record_file.close()
        print("[record] Sauvegarde terminee → {}".format(args.record))


if __name__ == "__main__":
    run(parse_args())

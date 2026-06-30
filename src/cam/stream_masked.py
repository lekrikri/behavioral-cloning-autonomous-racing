"""
camera_stream_masked.py — Stream MJPEG HTTP résilient avec masque lignes blanches

OAK-D Lite → preview RAW → white_line_mask + overlay → MJPEG HTTP

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u -m src.cam.stream_masked --port 5601

VLC / navigateur :
  http://10.112.248.41:5601  (hotspot Krikri)
  http://100.112.10.119:5601 (Tailscale)

Améliorations anti-freeze :
  - JPEG pré-encodé dans le thread OAK-D (pas dans le handler HTTP)
  - frame_id : le handler n'envoie que les nouvelles frames (pas de flood)
  - flush() systématique après chaque frame
  - Content-Length dans le header MJPEG
  - Timeout socket (2s) pour détecter clients morts
  - Placeholder "OAK-D RECONNECTING..." à 1fps pendant les crashs
  - camera_online flag + reset _latest_jpeg = None dès un crash
  - ThreadingHTTPServer (multi-clients sans blocage)
"""

import sys, time, threading, argparse, socket
import os
import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from http.server import BaseHTTPRequestHandler
try:
    from socketserver import ThreadingTCPServer
    class ThreadingHTTPServer(ThreadingTCPServer):
        allow_reuse_address = True
        def server_bind(self):
            import socket as _s
            self.socket.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
            ThreadingTCPServer.server_bind(self)
except Exception:
    from http.server import HTTPServer as ThreadingHTTPServer

import cv2
import numpy as np
from src.mask.visual_rays import white_line_mask, VisualRays

try:
    import depthai as dai
except ImportError:
    print("[masked] depthai non installe")
    sys.exit(1)

# ── État partagé ──────────────────────────────────────────────────────────────
_lock          = threading.Lock()
_latest_jpeg   = None   # bytes JPEG pré-encodés (pas de frame brute)
_frame_id      = 0      # incrémenté à chaque nouvelle frame
_camera_online = False  # False = OAK-D en cours de reconnexion


def _make_placeholder(w, h):
    img = np.zeros((h, w * 2, 3), dtype=np.uint8)  # double largeur comme le stream normal
    cv2.putText(img, "OAK-D RECONNECTING...", (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return jpg.tobytes()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",      type=int, default=5601)
    p.add_argument("--width",     type=int, default=512)
    p.add_argument("--height",    type=int, default=256)
    p.add_argument("--fps",       type=int, default=8)
    p.add_argument("--mode",      choices=["hsv", "canny"], default="hsv")
    p.add_argument("--hsv-v-min", type=int, default=195,
                   help="Seuil V min HSV (175=permissif, 200=strict)")
    p.add_argument("--no-clahe",  action="store_true",
                   help="Desactiver la normalisation CLAHE")
    return p.parse_args()


def apply_overlay(bgr, mask, vr):
    H, W = bgr.shape[:2]
    vis  = bgr.copy()

    green          = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.6, 0)

    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(vis, (W // 2, H - 1), (cx, cy), (255, 0, 0), 1)
        err = cx - W // 2
        cv2.putText(vis, "err={:+d}px".format(err), (4, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    rays = vr(bgr)
    for col, ray in zip(vr.cols, rays):
        r = int(255 * (1.0 - ray))
        g = int(255 * ray)
        y_top = int(H - ray * H * 0.10)
        cv2.line(vis, (col, H), (col, y_top), (0, g, r), 1)

    whites = int(mask.sum() / 255)
    cv2.putText(vis, "{}x{} | {}px".format(W, H, whites), (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

    # Masque côté droit en miniature (moitié hauteur, même largeur) → hstack propre
    mask_small = cv2.resize(mask, (W, H // 2))
    mask_bgr   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    # Padding vertical pour aligner avec vis (H)
    pad = np.zeros((H - H // 2, W, 3), dtype=np.uint8)
    mask_panel = np.vstack([pad, mask_bgr])
    return np.hstack([vis, mask_panel])


# ── Handler MJPEG résilient ───────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        global _latest_jpeg, _frame_id, _camera_online

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        # Timeout socket : client mort détecté en 2s max
        try:
            self.connection.settimeout(2.0)
        except Exception:
            pass

        last_id = -1

        try:
            while True:
                # ── Lecture état partagé ──────────────────────────────────────
                with _lock:
                    online = _camera_online
                    cur_id = _frame_id
                    jpg    = _latest_jpeg

                # ── Mode dégradé : caméra déconnectée ────────────────────────
                if not online:
                    jpg_to_send = _placeholder
                    last_id = -1          # reset pour reprendre dès le retour
                    time.sleep(0.5)       # placeholder à ~2fps
                else:
                    if cur_id == last_id or jpg is None:
                        time.sleep(0.01)  # pas de nouvelle frame, on attend
                        continue
                    jpg_to_send = jpg
                    last_id = cur_id

                # ── Envoi MJPEG ───────────────────────────────────────────────
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        "Content-Length: {}\r\n\r\n".format(len(jpg_to_send)).encode()
                    )
                    self.wfile.write(jpg_to_send)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()   # push immédiat côté TCP
                except (BrokenPipeError, ConnectionResetError):
                    break
                except socket.timeout:
                    break
                except Exception:
                    break

        except Exception:
            pass


# ── Thread OAK-D ─────────────────────────────────────────────────────────────
def run_oak(args):
    global _latest_jpeg, _frame_id, _camera_online

    ROI_TOP    = int(args.height * 0.45)
    USE_CLAHE  = not args.no_clahe
    vr = VisualRays(
        img_width=args.width, img_height=args.height,
        mode=args.mode, row_band=(0.35, 1.0),
        morph_k=5,
    )
    HSV_LOW  = np.array([0,   0, args.hsv_v_min], dtype=np.uint8)
    HSV_HIGH = np.array([180, 40, 255],            dtype=np.uint8)  # S<=40 strict

    attempt = 0
    while True:
        try:
            attempt += 1
            if attempt > 1:
                print("[masked] Tentative {}...".format(attempt))

            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(args.width, args.height)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(args.fps)
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("preview")
            cam.preview.link(xout.input)

            with dai.Device(pipeline, True) as device:
                q = device.getOutputQueue("preview", maxSize=1, blocking=False)

                with _lock:
                    _camera_online = True
                attempt = 0   # reset backoff dès connexion réussie
                print("[masked] {}x{} @ {}fps | mode={} | port={}".format(
                    args.width, args.height, args.fps, args.mode, args.port))

                frame_count = 0
                t0 = time.time()

                while True:
                    pkt = q.get()
                    bgr = pkt.getCvFrame()

                    mask = white_line_mask(
                        bgr, mode=args.mode,
                        hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
                        morph_k=5, blur_k=3, use_clahe=USE_CLAHE,
                        min_area=400,
                    )
                    mask[:ROI_TOP, :] = 0

                    vis = apply_overlay(bgr, mask, vr)

                    # Encoder le JPEG ICI (pas dans le handler)
                    _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 65])

                    with _lock:
                        _latest_jpeg = jpg.tobytes()
                        _frame_id   += 1

                    frame_count += 1
                    if frame_count % (args.fps * 15) == 0:
                        fps_real = frame_count / (time.time() - t0)
                        print("[masked] {} frames | {:.1f} fps".format(frame_count, fps_real))

        except KeyboardInterrupt:
            print("[masked] Arret.")
            break
        except Exception as e:
            msg = str(e)
            with _lock:
                _camera_online = False
                _latest_jpeg   = None

            delay = min(5 * (attempt if attempt > 0 else 1), 30)
            print("[masked] OAK-D crash ({}) — reconnexion dans {}s...".format(
                type(e).__name__, delay))
            time.sleep(delay)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    _placeholder = _make_placeholder(args.width, args.height)

    t = threading.Thread(target=run_oak, args=(args,), daemon=True)
    t.start()

    print("[masked] Serveur MJPEG → http://0.0.0.0:{}".format(args.port))
    print("[masked] VLC : http://IP_JETSON:{}".format(args.port))

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

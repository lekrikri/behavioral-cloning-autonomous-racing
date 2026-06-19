"""
camera_stream_masked.py — Stream MJPEG HTTP avec masque lignes blanches

OAK-D Lite → preview RAW → white_line_mask + overlay → MJPEG HTTP

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream_masked.py --port 5601

VLC / navigateur :
  http://10.112.248.41:5601
"""

import sys, time, threading, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

# Import masque depuis visual_rays.py (même dossier)
import os
sys.path.insert(0, os.path.dirname(__file__))
from visual_rays import white_line_mask, VisualRays

try:
    import depthai as dai
except ImportError:
    print("[masked] depthai non installe : pip install depthai")
    sys.exit(1)

# Frame partagée entre thread OAK-D et serveur HTTP
_latest_frame = None
_frame_lock   = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",   type=int, default=5601)
    p.add_argument("--width",  type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--fps",    type=int, default=8)
    p.add_argument("--mode",   choices=["hsv", "canny"], default="hsv",
                   help="hsv=eclairage homogene | canny=bords (eclairage variable)")
    return p.parse_args()


def apply_overlay(bgr, mask, vr):
    H, W = bgr.shape[:2]
    vis  = bgr.copy()

    # Overlay vert sur les lignes blanches (alpha fort pour bien voir)
    green          = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.6, 0)

    # Centre de masse → direction estimée
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(vis, (W // 2, H - 1), (cx, cy), (255, 0, 0), 1)
        err = cx - W // 2
        cv2.putText(vis, f"err={err:+d}px", (4, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # Raycasts visuels — petites barres en bas (10% H) pour ne pas écraser le masque
    rays = vr(bgr)
    for col, ray in zip(vr.cols, rays):
        r = int(255 * (1.0 - ray))
        g = int(255 * ray)
        y_top = int(H - ray * H * 0.10)
        cv2.line(vis, (col, H), (col, y_top), (0, g, r), 1)

    # Info
    whites = int(mask.sum() / 255)
    cv2.putText(vis, f"{W}x{H} | {whites}px", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

    # Masque binaire côté droit (comme live_mask_oak.py)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([vis, mask_bgr])


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silencer logs HTTP

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                _, jpg = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except Exception:
            pass


def run_oak(args):
    global _latest_frame

    ROI_TOP  = int(args.height * 0.35)   # 35% — lignes de piste visibles dès 35% de haut
    vr = VisualRays(
        img_width=args.width, img_height=args.height,
        mode=args.mode, row_band=(0.35, 1.0),  # même bande que ROI_TOP
    )
    HSV_LOW  = np.array([0,   0, 180], dtype=np.uint8)
    HSV_HIGH = np.array([180, 50, 255], dtype=np.uint8)

    attempt = 0
    while True:
        try:
            attempt += 1
            if attempt > 1:
                print(f"[masked] Tentative {attempt}...")

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
                q = device.getOutputQueue("preview", maxSize=4, blocking=False)
                print(f"[masked] {args.width}x{args.height} @ {args.fps}fps "
                      f"| mode={args.mode} | port={args.port}")

                frame_count = 0
                t0 = time.time()
                while True:
                    pkt = q.get()
                    bgr = pkt.getCvFrame()

                    mask = white_line_mask(
                        bgr, mode=args.mode,
                        hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
                    )
                    mask[:ROI_TOP, :] = 0  # ignorer la moitié haute (fond/plafond)

                    vis = apply_overlay(bgr, mask, vr)

                    with _frame_lock:
                        _latest_frame = vis

                    frame_count += 1
                    if frame_count % (args.fps * 10) == 0:
                        fps = frame_count / (time.time() - t0)
                        print(f"[masked] {frame_count} frames | {fps:.1f} fps")

        except KeyboardInterrupt:
            print("[masked] Arret.")
            break
        except RuntimeError as e:
            msg = str(e)
            if any(k in msg for k in ("X_LINK_ERROR", "Device crashed", "Couldn't read",
                                       "No available devices", "Timeout")):
                delay = min(5 * attempt, 30)
                print(f"[masked] OAK-D crash — reconnexion dans {delay}s...")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            delay = min(5 * attempt, 30)
            print(f"[masked] Erreur inattendue ({type(e).__name__}) — reconnexion dans {delay}s...")
            time.sleep(delay)


if __name__ == "__main__":
    args = parse_args()

    t = threading.Thread(target=run_oak, args=(args,), daemon=True)
    t.start()

    print(f"[masked] Serveur MJPEG → http://0.0.0.0:{args.port}")
    print(f"[masked] VLC : http://IP_JETSON:{args.port}")
    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

"""
stream_view.py — Stream MJPEG caméra OAK-D (visualisation seule, sans controller)

Accessible dans le navigateur : http://JETSON_IP:5601

Usage (Jetson) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/stream_view.py
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/stream_view.py --port 5601 --crop-top 0.35 --show-mask

--port     : port HTTP MJPEG (défaut 5601)
--crop-top : fraction haute à couper [0-1] (défaut 0.35)
--show-mask: overlay masque blanc en vert (debug lignes blanches)
"""

import sys, os, time, argparse, threading, glob, socket
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

try:
    import depthai as dai
except ImportError:
    print("[stream] depthai non installé")
    sys.exit(1)

# ─── config ──────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 320
CAM_FPS      = 15
HSV_LOW  = np.array([0,   0, 150], dtype=np.uint8)
HSV_HIGH = np.array([180, 45, 255], dtype=np.uint8)

# ─── frame partagée ──────────────────────────────────────────────────────────
_lock = threading.Lock()
_jpg  = b""


def _placeholder():
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(img, "En attente camera...", (60, CAM_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    _, j = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return bytes(j)


_jpg = _placeholder()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path != "/":
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _lock:
                    data = _jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_server(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "?"
    print("[stream] http://{}:{}".format(ip, port))
    print("[stream] Ouvrir dans navigateur : http://10.41.58.41:{}".format(port))


# ─── recovery OAK-D ──────────────────────────────────────────────────────────
def _find_hub(name='1-2.1'):
    drv = '/sys/bus/usb/drivers/usb'
    try:
        with open(os.path.join(drv, 'unbind'), 'w') as f: f.write(name)
        print("[stream] hub {} unbind".format(name)); time.sleep(3)
        with open(os.path.join(drv, 'bind'),   'w') as f: f.write(name)
        print("[stream] hub {} bind".format(name));   time.sleep(5)
    except Exception as e:
        print("[stream] hub rebind: {}".format(e))


def _boot_memory(dev_info):
    try:
        with dai.Device(dai.OpenVINO.VERSION_2021_4, dev_info, dai.UsbSpeed.HIGH) as d:
            print("[stream] BL {}".format(d.getBootloaderVersion()))
            d.bootMemory()
            print("[stream] bootMemory OK")
    except Exception as e:
        print("[stream] bootMem: {}".format(e))


def _recovery():
    if os.environ.get("OAKD_POST_RECOVERY") == "1":
        return
    devs = dai.Device.getAllAvailableDevices()
    unbooted = [d for d in devs if d.state == dai.XLinkDeviceState.X_LINK_UNBOOTED]
    if not unbooted:
        return
    print("[stream] Device UNBOOTED → hub rebind...")
    _find_hub()
    devs2 = dai.Device.getAllAvailableDevices()
    candidates = [d for d in devs2 if d.state in (
        dai.XLinkDeviceState.X_LINK_UNBOOTED,
        dai.XLinkDeviceState.X_LINK_BOOTLOADER)]
    if candidates:
        _boot_memory(candidates[0])
    env = os.environ.copy()
    env["OAKD_POST_RECOVERY"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    global _jpg

    p = argparse.ArgumentParser()
    p.add_argument("--port",      type=int,   default=5601)
    p.add_argument("--crop-top",  type=float, default=0.35)
    p.add_argument("--show-mask", action="store_true",
                   help="Overlay masque blanc en vert")
    args = p.parse_args()

    _recovery()
    start_server(args.port)

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    # FOV ~81° : 1080P → ISP downscale 1/3 → 640×360 plein capteur → crop → 640×320
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(1, 3)
    cam.setPreviewSize(CAM_W, CAM_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(CAM_FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("preview")
    cam.preview.link(xout.input)

    crop_px = int(CAM_H * args.crop_top)
    t0 = time.time()
    fc = 0

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("preview", maxSize=4, blocking=False)
        print("[stream] OK — FOV~81° crop={:.0%} mask={}".format(
            args.crop_top, args.show_mask))

        while True:
            pkt = q.get()
            bgr = pkt.getCvFrame()
            bgr[:crop_px, :] = 0

            vis = bgr.copy()

            if args.show_mask:
                hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
                mask[:crop_px, :] = 0
                overlay = np.zeros_like(bgr)
                overlay[mask > 0] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            fc += 1
            fps = fc / max(time.time() - t0, 0.001)
            cv2.putText(vis, "{:.0f}fps  FOV~81deg".format(fps),
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.line(vis, (0, crop_px), (CAM_W, crop_px), (0, 100, 255), 1)

            _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _lock:
                _jpg = bytes(jpg)


if __name__ == "__main__":
    main()

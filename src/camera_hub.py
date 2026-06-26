"""
camera_hub.py — Diffuse les frames couleur de l'OAK-D à plusieurs process.

L'OAK-D parle en XLink (mono-client) : un seul process peut ouvrir le device.
Le hub l'ouvre UNE fois et rediffuse les frames couleur sur une socket TCP locale ;
les consommateurs (mask_stream, inference_realcar) deviennent de simples clients au
lieu d'ouvrir la caméra eux-mêmes. Résout le conflit d'exclusivité caméra et permet
preview + conduite autonome simultanées.

── Serveur (Jetson) ──
  OPENBLAS_CORETYPE=ARMV8 python3 camera_hub.py [--port 8077] [--width 512] [--height 256]

── Consommateur ──
  from camera_hub import FrameClient
  c = FrameClient(port=8077); c.connect()
  bgr = c.getCvFrame()        # derniere frame BGR (HxWx3 uint8)

Protocole (binaire, localhost) : header [magic 'RC' | H u16 | W u16 | seq u32] + raw BGR.
Frames brutes (non compressées) = sans perte, idéal pour la perception ; négligeable sur localhost.
"""

import argparse
import socket
import struct
import threading
import time

import numpy as np

MAGIC  = b"RC"
HEADER = struct.Struct(">2sHHI")  # magic, height, width, seq


class FrameClient:
    """Client du hub : lit les dernières frames couleur. API getCvFrame() compatible."""

    def __init__(self, host="127.0.0.1", port=8077, timeout=5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None

    def connect(self):
        s = socket.create_connection(self.addr, self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("hub fermé")
            buf += chunk
        return buf

    def get(self):
        """Retourne (seq, frame BGR HxWx3 uint8)."""
        if self.sock is None:
            self.connect()
        magic, h, w, seq = HEADER.unpack(self._recv_exact(HEADER.size))
        if magic != MAGIC:
            raise ConnectionError("header invalide")
        data = self._recv_exact(h * w * 3)
        return seq, np.frombuffer(data, np.uint8).reshape(h, w, 3)

    def getCvFrame(self):
        return self.get()[1]

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


class _Hub:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True

    def set(self, frame):
        with self.lock:
            self.frame = frame
            self.seq = (self.seq + 1) & 0xFFFFFFFF

    def snapshot(self):
        with self.lock:
            return self.seq, self.frame


def _capture(hub):
    import depthai as dai
    attempt = 0
    while hub.running:
        try:
            attempt += 1
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            # Full FOV : ISP downscale 1080P -> 640x360 plein capteur (memes params que controller)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setIspScale(1, 3)
            cam.setPreviewSize(hub.W, hub.H)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(15)
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.preview.link(xout.input)
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue("rgb", maxSize=1, blocking=False)
                print("[hub] OAK-D OK (attempt {}) — diffuse les frames".format(attempt))
                attempt = 0
                while hub.running:
                    hub.set(q.get().getCvFrame())
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[hub] OAK-D crash ({}) — reconnexion dans 5s...".format(e))
            time.sleep(5)
    print("[hub] capture terminee")


def _serve_client(conn, hub):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    last = -1
    try:
        while hub.running:
            seq, frame = hub.snapshot()
            if frame is not None and seq != last:
                last = seq
                conn.sendall(HEADER.pack(MAGIC, frame.shape[0], frame.shape[1], seq)
                             + frame.tobytes())
            else:
                time.sleep(0.003)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass  # consommateur déconnecté
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int, default=8077)
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    hub = _Hub(args.width, args.height)
    threading.Thread(target=_capture, args=(hub,), daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(8)
    print(f"[hub] sert sur :{args.port} ({args.width}x{args.height}) — Ctrl+C pour arrêter")
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_serve_client, args=(conn, hub), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[hub] arrêt")
    finally:
        hub.running = False
        srv.close()


if __name__ == "__main__":
    main()

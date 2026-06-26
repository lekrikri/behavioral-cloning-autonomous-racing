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
import json
import socket
import struct
import threading
import time

import numpy as np

MAGIC  = b"RC"
DEPTH_MAGIC = b"RD"
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


class IMUClient:
    """Client IMU du hub : flux push d'échantillons gyro/accel bruts (canal séparé).

    drain() rend les échantillons reçus depuis le dernier appel — le débiaisage et
    le choix de l'axe yaw restent à la charge de l'appelant (le hub passe le brut)."""

    def __init__(self, host="127.0.0.1", port=8078, timeout=5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None
        self.buf = b""

    def connect(self):
        s = socket.create_connection(self.addr, self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def drain(self):
        """list de (t, gx, gy, gz, ax, ay, az) reçus depuis le dernier appel."""
        if self.sock is None:
            self.connect()
        self.sock.setblocking(False)
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self.buf += chunk
        except (BlockingIOError, OSError):
            pass
        finally:
            self.sock.setblocking(True)
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                m = json.loads(line)
                g, a = m["g"], m["a"]
                out.append((m["t"], g[0], g[1], g[2], a[0], a[1], a[2]))
            except (ValueError, KeyError):
                pass
        return out

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


class DepthClient:
    """Reads aligned depth frames (uint16 mm, same size as color) from the hub."""

    def __init__(self, host="127.0.0.1", port=8079, timeout=5.0):
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
        if self.sock is None:
            self.connect()
        magic, h, w, seq = HEADER.unpack(self._recv_exact(HEADER.size))
        if magic != DEPTH_MAGIC:
            raise ConnectionError("header depth invalide")
        data = self._recv_exact(h * w * 2)
        return seq, np.frombuffer(data, np.uint16).reshape(h, w)

    def getFrame(self):
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
        self.depth = None
        self.depth_seq = 0
        self.running = True
        self.imu_clients = []
        self.imu_lock = threading.Lock()

    def set(self, frame):
        with self.lock:
            self.frame = frame
            self.seq = (self.seq + 1) & 0xFFFFFFFF

    def snapshot(self):
        with self.lock:
            return self.seq, self.frame

    def set_depth(self, frame):
        with self.lock:
            self.depth = frame
            self.depth_seq = (self.depth_seq + 1) & 0xFFFFFFFF

    def depth_snapshot(self):
        with self.lock:
            return self.depth_seq, self.depth

    def add_imu_client(self, conn):
        with self.imu_lock:
            self.imu_clients.append(conn)

    def broadcast_imu(self, t, gyro, accel):
        line = (json.dumps({"t": t, "g": list(gyro), "a": list(accel)}) + "\n").encode()
        with self.imu_lock:
            dead = [c for c in self.imu_clients if not _try_send(c, line)]
            for c in dead:
                self.imu_clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass


def _try_send(conn, data):
    try:
        conn.sendall(data)
        return True
    except OSError:
        return False


def _capture(hub):
    import depthai as dai
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(hub.W, hub.H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 200)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(20)
    ximu = pipeline.create(dai.node.XLinkOut)
    ximu.setStreamName("imu")
    imu.out.link(ximu.input)

    # Stereo depth aligned to the color camera (CAM_A) so depth pixels match the
    # color/mask pixels — used downstream to reject off-ground detections.
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(30)
    mono_r.setFps(30)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    xdepth = pipeline.create(dai.node.XLinkOut)
    xdepth.setStreamName("depth")
    stereo.depth.link(xdepth.input)

    with dai.Device(pipeline) as device:
        import cv2
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qi = device.getOutputQueue("imu", maxSize=50, blocking=False)
        qd = device.getOutputQueue("depth", maxSize=4, blocking=False)
        print("[hub] OAK-D OK — diffuse frames + IMU + depth")
        while hub.running:
            hub.set(q.get().getCvFrame())
            d = qi.tryGet()
            while d is not None:
                for pkt in d.packets:
                    g, a = pkt.gyroscope, pkt.acceleroMeter
                    hub.broadcast_imu(g.getTimestampDevice().total_seconds(),
                                      (g.x, g.y, g.z), (a.x, a.y, a.z))
                d = qi.tryGet()
            dd = qd.tryGet()
            if dd is not None:
                depth = dd.getFrame()
                if depth.shape != (hub.H, hub.W):
                    depth = cv2.resize(depth, (hub.W, hub.H), interpolation=cv2.INTER_NEAREST)
                hub.set_depth(depth)
    print("[hub] capture terminée")


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


def _serve_depth_client(conn, hub):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    last = -1
    try:
        while hub.running:
            seq, depth = hub.depth_snapshot()
            if depth is not None and seq != last:
                last = seq
                conn.sendall(HEADER.pack(DEPTH_MAGIC, depth.shape[0], depth.shape[1], seq)
                             + depth.tobytes())
            else:
                time.sleep(0.003)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=8077)
    parser.add_argument("--imu-port",   type=int, default=8078)
    parser.add_argument("--depth-port", type=int, default=8079)
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    hub = _Hub(args.width, args.height)
    threading.Thread(target=_capture, args=(hub,), daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(8)

    imu_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    imu_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    imu_srv.bind(("0.0.0.0", args.imu_port))
    imu_srv.listen(8)
    threading.Thread(target=_accept_imu, args=(imu_srv, hub), daemon=True).start()

    depth_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    depth_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    depth_srv.bind(("0.0.0.0", args.depth_port))
    depth_srv.listen(8)
    threading.Thread(target=_accept_depth, args=(depth_srv, hub), daemon=True).start()

    print(f"[hub] frames :{args.port} | IMU :{args.imu_port} | depth :{args.depth_port}"
          f" ({args.width}x{args.height}) — Ctrl+C")
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_serve_client, args=(conn, hub), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[hub] arrêt")
    finally:
        hub.running = False
        srv.close()
        imu_srv.close()
        depth_srv.close()


def _accept_imu(imu_srv, hub):
    while hub.running:
        try:
            conn, _ = imu_srv.accept()
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hub.add_imu_client(conn)
        print("[hub] client IMU connecté")


def _accept_depth(depth_srv, hub):
    while hub.running:
        try:
            conn, _ = depth_srv.accept()
        except OSError:
            break
        threading.Thread(target=_serve_depth_client, args=(conn, hub), daemon=True).start()
        print("[hub] client depth connecté")


if __name__ == "__main__":
    main()

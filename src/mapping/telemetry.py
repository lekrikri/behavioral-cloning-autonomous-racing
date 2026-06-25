"""Lightweight telemetry server: streams newline-delimited JSON to laptop clients.

Design constraints:
- The Jetson LISTENS, the laptop connects (the Tailscale share is asymmetric:
  Jetson->laptop is blocked, so the laptop pulls via `ssh -L`). See project memory.
- publish() must NEVER block the control loop. It stores the latest payload only
  (latest-wins); a slow or absent client just misses frames — drop, don't buffer.
- stdlib only (Python 3.6 on the Jetson).

Video frames are sent on a separate optional field (base64 JPEG) added later; the
JSON channel carries pose / rays / grid / scalars now.
"""

import json
import socket
import threading
import time


class TelemetryServer:
    def __init__(self, port, host="0.0.0.0", send_hz=30.0):
        self.addr = (host, port)
        self.period = 1.0 / send_hz
        self._latest = None
        self._lock = threading.Lock()
        self._clients = []
        self._running = False
        self._sock = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.addr)
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()
        print("[telemetry] listening on %s:%d" % self.addr)
        return self

    def publish(self, payload):
        """Store the latest payload (dict). Non-blocking, latest-wins."""
        with self._lock:
            self._latest = payload

    def _accept_loop(self):
        while self._running:
            try:
                conn, peer = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._clients.append(conn)
            print("[telemetry] client connected: %s" % (peer,))

    def _send_loop(self):
        while self._running:
            t0 = time.time()
            with self._lock:
                payload = self._latest
            if payload is not None and self._clients:
                line = (json.dumps(payload) + "\n").encode("utf-8")
                dead = []
                for c in self._clients:
                    try:
                        c.sendall(line)
                    except OSError:
                        dead.append(c)
                for c in dead:
                    self._drop(c)
            dt = self.period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def _drop(self, conn):
        try:
            self._clients.remove(conn)
        except ValueError:
            pass
        try:
            conn.close()
        except OSError:
            pass
        print("[telemetry] client dropped")

    def close(self):
        self._running = False
        for c in list(self._clients):
            self._drop(c)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass


def _selftest():
    """Loopback test: server publishes counters, a client reads a few lines."""
    srv = TelemetryServer(0)  # ephemeral port
    srv._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv._sock.bind(("127.0.0.1", 0))
    port = srv._sock.getsockname()[1]
    srv._sock.listen(4)
    srv._sock.settimeout(0.5)
    srv._running = True
    threading.Thread(target=srv._accept_loop, daemon=True).start()
    threading.Thread(target=srv._send_loop, daemon=True).start()
    print("[telemetry] selftest on port %d" % port)

    cli = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    time.sleep(0.2)
    for i in range(5):
        srv.publish({"i": i, "pose": [float(i), 0.0, 0.0]})
        time.sleep(0.1)
    data = cli.recv(4096).decode("utf-8")
    lines = [l for l in data.split("\n") if l]
    print("received %d JSON lines, last = %s" % (len(lines), lines[-1]))
    assert len(lines) >= 1
    assert json.loads(lines[-1])["pose"][0] >= 0.0
    cli.close()
    srv.close()
    print("telemetry selftest OK")


if __name__ == "__main__":
    _selftest()

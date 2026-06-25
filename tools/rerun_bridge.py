"""rerun_bridge.py — laptop-side live viewer for the car's mapping telemetry.

Connects to the Jetson telemetry server (newline-delimited JSON), rebuilds the
occupancy grid with the SAME OccupancyGrid the car uses (so the live map matches
the on-board one), and logs trajectory + live scan + occupancy image + scalar
plots into the Rerun viewer.

Networking: connect directly to the Jetson over Tailscale — laptop->Jetson is the
allowed direction (only Jetson->laptop is blocked), so no tunnel is needed:
    .venv/bin/python tools/rerun_bridge.py --host 100.112.10.119
Fallback if a Tailscale ACL ever blocks the port:
    ssh -L 5602:127.0.0.1:5602 robocar       # then run with default --host 127.0.0.1

Runs on the laptop in the project venv (rerun-sdk). Pure numpy occupancy import
works fine under Python 3.12.
"""

import argparse
import base64
import json
import os
import socket
import sys
import time

import numpy as np
import rerun as rr

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mapping.occupancy import OccupancyGrid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="telemetry host (localhost via ssh -L)")
    p.add_argument("--port", type=int, default=5602)
    p.add_argument("--grid-res", type=float, default=0.05)
    p.add_argument("--grid-size", type=float, default=20.0)
    p.add_argument("--map-every", type=int, default=5, help="log the occupancy image every N messages")
    p.add_argument("--no-spawn", action="store_true", help="don't auto-launch the viewer")
    return p.parse_args()


class Bridge:
    def __init__(self, grid_res, grid_size, map_every):
        self.grid = OccupancyGrid(res_m=grid_res, size_m=grid_size)
        self.map_every = map_every
        self.traj = []      # display-space points
        self.t0 = None
        self.n_msg = 0

    def to_disp(self, x, y):
        """World metres -> display cell coords (col, row), north-up (matches flipud)."""
        g = self.grid
        j = x / g.res + g.half
        i = y / g.res + g.half
        return [j, (g.n - 1) - i]

    def handle(self, msg):
        t = msg.get("t")
        if self.t0 is None:
            self.t0 = t
        rr.set_time("telemetry", duration=float(t - self.t0) if t is not None else 0.0)

        pose = msg.get("pose")
        if not pose:
            return
        x, y, theta = pose
        rays = msg.get("rays")
        angles = msg.get("ray_angles")
        ray_max_m = msg.get("ray_max_m", 3.0)

        if rays is not None and angles is not None:
            dists_m = np.asarray(rays, dtype=float) * ray_max_m
            self.grid.integrate_scan((x, y, theta), dists_m, angles)

        self.traj.append(self.to_disp(x, y))
        rr.log("map/trajectory", rr.LineStrips2D([self.traj], colors=[[80, 160, 255]]))
        rr.log("map/car", rr.Points2D([self.to_disp(x, y)], radii=[3.0], colors=[[255, 80, 80]]))

        if rays is not None and angles is not None:
            strips = []
            for d_norm, a in zip(rays, angles):
                d = d_norm * ray_max_m
                ex, ey = x + d * np.cos(theta + a), y + d * np.sin(theta + a)
                strips.append([self.to_disp(x, y), self.to_disp(ex, ey)])
            rr.log("map/scan", rr.LineStrips2D(strips, colors=[[0, 220, 180]]))

        if self.n_msg % self.map_every == 0:
            img = ((1.0 - self.grid.probability()) * 255).astype(np.uint8)
            rr.log("map/occupancy", rr.Image(np.flipud(img)))

        for key, entity in (("color_jpeg", "camera/color"),
                            ("depth_jpeg", "camera/depth"),
                            ("mask_jpeg", "camera/mask")):
            b64 = msg.get(key)
            if b64:
                rr.log(entity, rr.EncodedImage(contents=base64.b64decode(b64), media_type="image/jpeg"))

        if msg.get("speed") is not None:
            rr.log("plots/speed_mps", rr.Scalars(float(msg["speed"])))
        if msg.get("erpm") is not None:
            rr.log("plots/erpm", rr.Scalars(float(msg["erpm"])))
        rr.log("plots/heading_deg", rr.Scalars(float(np.degrees(theta))))

        self.n_msg += 1


def connect(host, port):
    while True:
        try:
            print("[bridge] connecting to %s:%d ..." % (host, port))
            return socket.create_connection((host, port), timeout=5.0)
        except OSError as e:
            print("[bridge] %s — retry in 1s (Jetson running + ssh -L up?)" % e)
            time.sleep(1.0)


def main():
    args = parse_args()
    # Make the bundled Rerun viewer findable even when the venv isn't activated
    # (running `.venv/bin/python` doesn't put .venv/bin on PATH).
    _bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    rr.init("robocar/live_map", spawn=not args.no_spawn)
    bridge = Bridge(args.grid_res, args.grid_size, args.map_every)

    sock = connect(args.host, args.port)
    buf = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            chunk = b""
        if not chunk:
            print("[bridge] disconnected — reconnecting")
            sock.close()
            time.sleep(0.5)   # avoid a tight loop when ssh -L accepts then EOFs (Jetson side down)
            sock = connect(args.host, args.port)
            buf = b""
            continue

        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            bridge.handle(msg)


if __name__ == "__main__":
    main()

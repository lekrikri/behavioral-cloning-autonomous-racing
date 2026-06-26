"""drive_and_map.py — teleop the car while estimating pose from polar raycasts.

Hub-client edition: the OAK is owned by camera_hub (run it first); this process pulls
COLOR frames (FrameClient) + IMU (IMUClient) from the hub, so no device-exclusivity
conflict. Perception = TRUE POLAR raycasts via IPM of the white-line mask (PolarRays),
the geometry that matches the sim lidar. Pose = gyro heading + VESC-eRPM odometry.
Streams pose + metric polar rays + camera/mask to the laptop (rerun_bridge).

All car parameters come from car.env (see car_config.py); CLI flags override a few.

Prereqs on the Jetson:
    OPENBLAS_CORETYPE=ARMV8 python3 src/camera_hub.py        # in one shell
    OPENBLAS_CORETYPE=ARMV8 python3 src/mapping/drive_and_map.py [--no-motor]

Controls (F710): right stick = steer, R2 = forward, L2 = reverse, START = quit.
"""

import argparse
import base64
import os
import sys
import threading
import time

import cv2
import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vesc_interface import VESCInterface
from visual_rays import white_line_mask
from camera_hub import FrameClient, IMUClient, DepthClient
from teleop_gamepad import (
    Gamepad, deadzone, calibrate_trigger_rest, trigger_fraction,
    AXIS_STEER, AXIS_ACCEL, AXIS_BRAKE, BTN_QUIT,
)

from mapping.car_config import load as load_car
from mapping.polar_rays import PolarRays
from mapping.pose import DeadReckoning, close_loop
from mapping.occupancy import OccupancyGrid
from mapping.telemetry import TelemetryServer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-motor", action="store_true", help="perception+pose+telemetry only, never command the motor")
    p.add_argument("--map-out", default=None, help="save the map+trajectory (.npz) at exit")
    p.add_argument("--hub-host", default=None, help="override car.env HUB_HOST")
    p.add_argument("--deadzone", type=float, default=0.08)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--no-depth", action="store_true", help="don't use the depth ground filter")
    p.add_argument("--no-images", action="store_true", help="don't stream color/mask/depth (saves bandwidth)")
    p.add_argument("--img-every", type=int, default=8, help="encode+send images every N loops")
    p.add_argument("--img-quality", type=int, default=50)
    return p.parse_args()


class _Reader(threading.Thread):
    """Background reader so the loop always has the latest hub frame, no lag.

    getter is the client method name ('getCvFrame' for color, 'getFrame' for depth)."""

    def __init__(self, client, getter="getCvFrame"):
        super(_Reader, self).__init__()
        self.daemon = True
        self.client = client
        self.getter = getter
        self.lock = threading.Lock()
        self.frame = None
        self.t_recv = 0.0
        self.running = True

    def run(self):
        while self.running:
            try:
                f = getattr(self.client, self.getter)()
                with self.lock:
                    self.frame = f
                    self.t_recv = time.time()
            except Exception:
                time.sleep(0.2)
                try:
                    self.client.connect()
                except Exception:
                    pass

    def latest(self):
        with self.lock:
            return self.frame, self.t_recv


def _jpeg_b64(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


def _depth_jpeg_b64(depth_mm, max_mm, quality):
    d8 = np.clip(depth_mm.astype(np.float32) / max_mm * 255.0, 0, 255).astype(np.uint8)
    return _jpeg_b64(cv2.applyColorMap(d8, cv2.COLORMAP_JET), quality)


def calibrate_gyro_bias(imu, seconds):
    """Average gyro at rest -> per-axis bias. Keep the car still."""
    t0 = time.time()
    samples = []
    while time.time() - t0 < seconds:
        samples += imu.drain()
        time.sleep(0.02)
    if not samples:
        raise RuntimeError("no IMU samples from hub during bias calibration")
    n = float(len(samples))
    return (sum(s[1] for s in samples) / n,
            sum(s[2] for s in samples) / n,
            sum(s[3] for s in samples) / n)


def main():
    args = parse_args()
    cfg = load_car()
    host = args.hub_host or cfg.hub_host

    # --- connect to the hub (color + IMU) ---
    frames = FrameClient(host=host, port=cfg.hub_port)
    imu = IMUClient(host=host, port=cfg.hub_imu_port)
    try:
        frames.connect()
        imu.connect()
    except OSError as e:
        print("[map] cannot reach the hub at %s (%s)." % (host, e))
        print("      Start it first:  OPENBLAS_CORETYPE=ARMV8 python3 src/camera_hub.py")
        return
    reader = _Reader(frames, "getCvFrame")
    reader.start()

    depth_cli = None
    depth_reader = None
    if not args.no_depth:
        depth_cli = DepthClient(host=host, port=cfg.hub_depth_port)
        try:
            depth_cli.connect()
            depth_reader = _Reader(depth_cli, "getFrame")
            depth_reader.start()
        except OSError as e:
            print("[map] no depth channel (%s) -> ground filter disabled" % e)
            depth_cli = None

    try:
        pad = Gamepad("/dev/input/js0")
    except OSError as e:
        print("[map] cannot open gamepad: %s (F710 on, switch on 'X'?)" % e)
        return

    vesc = VESCInterface(
        port=cfg.vesc_port,
        servo_center=cfg.vesc_servo_center,
        servo_range=cfg.vesc_servo_range,
        current_max=cfg.vesc_max_current,
        invert_motor=cfg.vesc_invert_motor,
        throttle_mode=cfg.vesc_throttle_mode,
    )

    polar = PolarRays(
        img_width=cfg.cam_width, img_height=cfg.cam_height_px,
        hfov_deg=cfg.cam_hfov_deg, height_m=cfg.cam_height_m, pitch_deg=cfg.cam_pitch_deg,
        n_rays=cfg.n_rays, max_range_m=cfg.ray_max_m,
        row_band=(cfg.row_band_lo, cfg.row_band_hi),
    )
    angles = polar.angles

    print("=" * 60)
    print("  DRIVE & MAP (polar/IPM, hub client)  %s" % ("[NO-MOTOR]" if args.no_motor else ""))
    print("  cam h=%.2fm pitch=%.1f deg fov=%.1f | k=%.3e | servo_c=%.3f" %
          (cfg.cam_height_m, cfg.cam_pitch_deg, cfg.cam_hfov_deg, cfg.vesc_k_erpm_to_ms, cfg.vesc_servo_center))
    print("  Calibrating triggers — don't touch R2/L2...")
    rest = calibrate_trigger_rest(pad, [AXIS_ACCEL, AXIS_BRAKE])
    print("  Keep the car STILL — calibrating gyro bias (%.1fs)..." % cfg.gyro_bias_seconds)
    bias = calibrate_gyro_bias(imu, cfg.gyro_bias_seconds)
    print("  gyro bias (rad/s): x=%+.5f y=%+.5f z=%+.5f" % bias)
    print("  right stick=steer R2=fwd L2=rev START=quit")
    print("=" * 60)

    dr = DeadReckoning(k_erpm_to_ms=cfg.vesc_k_erpm_to_ms)
    grid = OccupancyGrid(res_m=cfg.grid_res_m, size_m=cfg.grid_size_m, max_range_m=cfg.ray_max_m)
    tel = TelemetryServer(cfg.telemetry_port).start()

    yaw_axis, yaw_sign = cfg.gyro_yaw_axis, cfg.gyro_yaw_sign
    bx = bias[yaw_axis]
    dt_loop = 1.0 / args.hz
    prev_t = None
    step = 0
    consec_err = 0
    loop_hz = 0.0
    last_loop = None

    try:
        while True:
            t_loop = time.time()
            if last_loop is not None:
                inst = 1.0 / max(1e-3, t_loop - last_loop)
                loop_hz = 0.9 * loop_hz + 0.1 * inst if loop_hz else inst
            last_loop = t_loop
            try:
                pad.poll()
                if pad.button(BTN_QUIT):
                    print("\n[map] START -> quit")
                    break

                steer = deadzone(pad.axis(AXIS_STEER), args.deadzone)
                rt = trigger_fraction(pad, AXIS_ACCEL, rest[AXIS_ACCEL])
                lt = trigger_fraction(pad, AXIS_BRAKE, rest[AXIS_BRAKE])
                if not args.no_motor:
                    vesc.drive(steer, rt - lt)

                color, t_recv = reader.latest()
                frame_age_ms = (t_loop - t_recv) * 1000.0 if t_recv else -1.0
                depth = depth_reader.latest()[0] if depth_reader is not None else None
                dists_m = None
                mask_raw = None
                mask_f = None
                if color is not None:
                    mask_raw = white_line_mask(color)
                    if depth is not None and depth.shape == mask_raw.shape:
                        mask_f = polar.filter_ground(mask_raw, depth, cfg.depth_filter_tol)
                    else:
                        mask_f = mask_raw
                    dists_m, _ = polar(mask_f)

                erpm = vesc.get_rpm()
                speed = cfg.vesc_k_erpm_to_ms * erpm * cfg.vesc_erpm_sign

                for s in imu.drain():
                    if prev_t is not None:
                        dt = s[0] - prev_t
                        if 0.0 < dt < 0.1:
                            dr.update(yaw_sign * (s[1 + yaw_axis] - bx), speed, dt)
                    prev_t = s[0]

                if dists_m is not None:
                    grid.integrate_scan(dr.pose, dists_m, angles)

                payload = {
                    "t": t_loop,
                    "pose": [dr.x, dr.y, dr.theta],
                    "rays_m": dists_m.tolist() if dists_m is not None else None,
                    "ray_angles": angles.tolist(),
                    "ray_max_m": cfg.ray_max_m,
                    "speed": speed,
                    "erpm": erpm,
                    "loop_hz": loop_hz,
                    "frame_age_ms": frame_age_ms,
                }
                if not args.no_images and step % args.img_every == 0 and color is not None:
                    payload["color_jpeg"] = _jpeg_b64(color, args.img_quality)
                    if mask_f is not None:
                        payload["mask_jpeg"] = _jpeg_b64(mask_f, args.img_quality)
                    if mask_raw is not None:
                        payload["mask_raw_jpeg"] = _jpeg_b64(mask_raw, args.img_quality)
                    if depth is not None:
                        payload["depth_jpeg"] = _depth_jpeg_b64(depth, cfg.ray_max_m * 1000.0, args.img_quality)
                tel.publish(payload)

                if step % 30 == 0:
                    print("\r x=%+.2f y=%+.2f th=%+.1f deg  v=%+.2f m/s  %4.1f Hz  frame %3.0f ms   " %
                          (dr.x, dr.y, np.degrees(dr.theta), speed, loop_hz, frame_age_ms),
                          end="", flush=True)
                step += 1
                consec_err = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                consec_err += 1
                print("\n[map] loop error (%d): %s" % (consec_err, e))
                if not args.no_motor:
                    try:
                        vesc.stop()
                    except Exception:
                        pass
                if consec_err >= 50:
                    print("[map] too many consecutive errors -> abort")
                    break

            sleep = dt_loop - (time.time() - t_loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[map] Ctrl-C -> stop")
    finally:
        reader.running = False
        if depth_reader is not None:
            depth_reader.running = False
        vesc.stop()
        vesc.close()
        pad.close()
        tel.close()
        frames.close()
        imu.close()
        if depth_cli is not None:
            depth_cli.close()
        _save_map(args, dr, grid)


def _save_map(args, dr, grid):
    if not args.map_out:
        return
    traj = np.asarray([(h[0], h[1]) for h in dr.history], dtype=float)
    closed = close_loop(dr.history) if len(traj) > 2 else traj
    np.savez(args.map_out,
             prob=grid.probability().astype(np.float32),
             res=grid.res, half=grid.half, traj=traj, traj_closed=closed)
    print("[map] saved %s (%d poses)" % (args.map_out, len(traj)))


if __name__ == "__main__":
    main()

"""drive_and_map.py — teleop the car one lap while estimating pose and mapping.

The Jetson-side main: opens the OAK ONCE (depth + IMU), reads the F710 gamepad,
drives the VESC, turns depth into rays, dead-reckons a 2D pose (gyro heading +
VESC-eRPM speed), feeds an occupancy grid, and streams pose+rays to the laptop
(rerun_bridge rebuilds the map there). Reuses the existing clean modules; touches
no autonomous-driving code.

Run on the Jetson with the numpy fix:
    OPENBLAS_CORETYPE=ARMV8 python3 src/mapping/drive_and_map.py
Validate without moving the car first:
    OPENBLAS_CORETYPE=ARMV8 python3 src/mapping/drive_and_map.py --no-motor

Controls (F710): left stick = steer, R2 = forward, L2 = reverse, START = quit.
SAFETY: keep wheels clear / on a stand until the sign flags are confirmed.
"""

import argparse
import base64
import os
import sys
import time

import cv2
import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import depthai as dai

from vesc_interface import VESCInterface
from depth_to_rays import DepthToRays, create_depthai_pipeline
from visual_rays import white_line_mask
from teleop_gamepad import (
    Gamepad, deadzone, calibrate_trigger_rest, trigger_fraction,
    AXIS_STEER, AXIS_ACCEL, AXIS_BRAKE, BTN_QUIT,
)

from mapping.config import MappingConfig
from mapping.imu import enable_imu, IMUReader, IMU_STREAM
from mapping.pose import DeadReckoning, close_loop, ray_angles
from mapping.occupancy import OccupancyGrid
from mapping.telemetry import TelemetryServer


def parse_args():
    cfg = MappingConfig()
    p = argparse.ArgumentParser(description=__doc__)
    # hardware
    p.add_argument("--port", default="/dev/ttyACM0", help="VESC serial port")
    p.add_argument("--js", default="/dev/input/js0", help="gamepad device")
    p.add_argument("--max-current", type=float, default=15.0, help="A — |current| cap (lower than teleop for mapping)")
    p.add_argument("--servo-center", type=float, default=cfg.servo_center)
    p.add_argument("--servo-range", type=float, default=0.40)
    p.add_argument("--deadzone", type=float, default=0.08)
    p.add_argument("--hz", type=float, default=30.0)
    # pose sign conventions (confirm by driving)
    p.add_argument("--k", type=float, default=cfg.k_erpm_to_ms, help="eRPM->m/s scale")
    p.add_argument("--erpm-sign", type=float, default=-1.0,
                   help="multiplies eRPM so forward gives +speed (forward read as negative eRPM on this car)")
    p.add_argument("--gyro-yaw-axis", type=int, default=cfg.gyro_yaw_axis, choices=[0, 1, 2])
    p.add_argument("--gyro-yaw-sign", type=float, default=cfg.gyro_yaw_sign)
    p.add_argument("--flip-rays", action="store_true", help="reverse ray left/right order if the map is mirrored")
    p.add_argument("--bias-seconds", type=float, default=2.0, help="at-rest gyro bias calibration window")
    # map / telemetry
    p.add_argument("--grid-res", type=float, default=cfg.grid_res_m)
    p.add_argument("--grid-size", type=float, default=cfg.grid_size_m)
    p.add_argument("--telemetry-port", type=int, default=cfg.telemetry_port)
    p.add_argument("--map-out", default=None, help="path to save the map+trajectory (.npz) at exit")
    # debug images (color video + depth + line mask in rerun)
    p.add_argument("--no-images", action="store_true", help="disable the camera/depth/mask streams (saves bandwidth)")
    p.add_argument("--img-width", type=int, default=256, help="color preview width (height keeps 4:3)")
    p.add_argument("--img-fps", type=float, default=10.0, help="color camera fps")
    p.add_argument("--img-every", type=int, default=10, help="encode+send images every N control loops")
    p.add_argument("--img-quality", type=int, default=50, help="JPEG quality 1-100")
    # safety / debug
    p.add_argument("--no-motor", action="store_true", help="perception+pose+telemetry only, never command the motor")
    args = p.parse_args()
    return args


def add_color(pipeline, width, height, fps):
    """Add the RGB camera (CAM_A) to the pipeline for the debug video + line mask."""
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewSize(width, height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(fps)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("color")
    cam.preview.link(xout.input)


def _jpeg_b64(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


def _depth_jpeg_b64(depth_mm, max_mm, quality):
    d8 = np.clip(depth_mm.astype(np.float32) / max_mm * 255.0, 0, 255).astype(np.uint8)
    return _jpeg_b64(cv2.applyColorMap(d8, cv2.COLORMAP_JET), quality)


def read_fov(device, fallback=72.9):
    """Real horizontal FOV from the device calibration — the single source of truth."""
    try:
        calib = device.readCalibration()
        fov = float(calib.getFov(dai.CameraBoardSocket.CAM_B))
        if fov > 1.0:
            return fov
    except Exception as e:
        print("[map] getFov failed (%s) -> fallback %.1f deg" % (e, fallback))
    return fallback


def main():
    args = parse_args()

    pipeline = create_depthai_pipeline()   # stereo depth ('depth' stream)
    enable_imu(pipeline, rate_hz=200)      # adds 'imu' stream to the same pipeline
    if not args.no_images:
        add_color(pipeline, args.img_width, (args.img_width * 3) // 4, args.img_fps)

    try:
        pad = Gamepad(args.js)
    except OSError as e:
        print("[map] cannot open %s: %s (is the F710 on, switch on 'X'?)" % (args.js, e))
        return

    vesc = VESCInterface(
        port=args.port,
        servo_center=args.servo_center,
        servo_range=args.servo_range,
        current_max=args.max_current,
        invert_motor=False,          # this car drives forward with invert OFF (see teleop_gamepad)
        throttle_mode="current",
    )

    print("=" * 60)
    print("  DRIVE & MAP  %s" % ("[NO-MOTOR]" if args.no_motor else ""))
    print("  Calibrating triggers — don't touch R2/L2...")
    rest = calibrate_trigger_rest(pad, [AXIS_ACCEL, AXIS_BRAKE])

    with dai.Device(pipeline) as device:
        fov = read_fov(device)
        bridge = DepthToRays(fov_deg=fov)
        ray_max_m = bridge.max_dist_mm / 1000.0
        angles = ray_angles(bridge.n_rays, fov)
        if args.flip_rays:
            angles = angles[::-1].copy()

        imu = IMUReader(device, yaw_axis=args.gyro_yaw_axis, yaw_sign=args.gyro_yaw_sign)
        print("  Keep the car STILL — calibrating gyro bias (%.1fs)..." % args.bias_seconds)
        bias = imu.calibrate_bias(args.bias_seconds)
        print("  gyro bias (rad/s): x=%+.5f y=%+.5f z=%+.5f" % bias)
        print("  FOV=%.1f deg  ray_max=%.2f m  k=%.3e  | left stick=steer R2=fwd L2=rev START=quit" %
              (fov, ray_max_m, args.k))
        print("=" * 60)

        dr = DeadReckoning(k_erpm_to_ms=args.k)
        grid = OccupancyGrid(res_m=args.grid_res, size_m=args.grid_size, max_range_m=ray_max_m)
        tel = TelemetryServer(args.telemetry_port).start()

        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_color = None if args.no_images else device.getOutputQueue("color", maxSize=2, blocking=False)
        latest_rays = None
        latest_depth = None
        latest_color = None
        prev_t = None
        dt_loop = 1.0 / args.hz
        step = 0
        consec_err = 0

        try:
            while True:
                t_loop = time.time()
                try:
                    pad.poll()
                    if pad.button(BTN_QUIT):
                        print("\n[map] START -> quit")
                        break

                    steer = deadzone(pad.axis(AXIS_STEER), args.deadzone)
                    rt = trigger_fraction(pad, AXIS_ACCEL, rest[AXIS_ACCEL])
                    lt = trigger_fraction(pad, AXIS_BRAKE, rest[AXIS_BRAKE])
                    throttle = rt - lt
                    if not args.no_motor:
                        vesc.drive(steer, throttle)

                    depth = q_depth.tryGet()
                    if depth is not None:
                        latest_depth = depth.getFrame()
                        latest_rays = bridge(latest_depth)
                    if q_color is not None:
                        cframe = q_color.tryGet()
                        if cframe is not None:
                            latest_color = cframe.getCvFrame()

                    # one blocking serial read per loop, reused for telemetry
                    erpm = vesc.get_rpm()
                    speed = args.k * erpm * args.erpm_sign

                    for s in imu.drain():
                        if prev_t is not None:
                            dt = s[0] - prev_t
                            if 0.0 < dt < 0.1:
                                dr.update(imu.yaw_rate(s), speed, dt)
                        prev_t = s[0]

                    if latest_rays is not None:
                        grid.integrate_scan(dr.pose, latest_rays * ray_max_m, angles)

                    payload = {
                        "t": t_loop,
                        "pose": [dr.x, dr.y, dr.theta],
                        "rays": latest_rays.tolist() if latest_rays is not None else None,
                        "ray_angles": angles.tolist(),
                        "ray_max_m": ray_max_m,
                        "speed": speed,
                        "erpm": erpm,
                    }
                    if not args.no_images and step % args.img_every == 0:
                        if latest_color is not None:
                            payload["color_jpeg"] = _jpeg_b64(latest_color, args.img_quality)
                            try:
                                payload["mask_jpeg"] = _jpeg_b64(white_line_mask(latest_color), args.img_quality)
                            except Exception:
                                pass
                        if latest_depth is not None:
                            payload["depth_jpeg"] = _depth_jpeg_b64(latest_depth, bridge.max_dist_mm, args.img_quality)
                    tel.publish(payload)

                    if step % 30 == 0:
                        print("\r x=%+.2f y=%+.2f th=%+.1f deg  v=%+.2f m/s   " %
                              (dr.x, dr.y, np.degrees(dr.theta), speed), end="", flush=True)
                    step += 1
                    consec_err = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # Transient OAK/serial hiccup: drop this iteration, keep mapping.
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
            vesc.stop()
            vesc.close()
            pad.close()
            tel.close()
            _save_map(args, dr, grid)


def _save_map(args, dr, grid):
    if not args.map_out:
        return
    traj = np.asarray([(h[0], h[1]) for h in dr.history], dtype=float)
    closed = close_loop(dr.history) if len(traj) > 2 else traj
    np.savez(args.map_out,
             prob=grid.probability().astype(np.float32),
             res=grid.res, half=grid.half,
             traj=traj, traj_closed=closed)
    print("[map] saved %s (%d poses)" % (args.map_out, len(traj)))


if __name__ == "__main__":
    main()

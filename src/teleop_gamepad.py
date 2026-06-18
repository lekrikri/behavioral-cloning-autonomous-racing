"""
teleop_gamepad.py — Manual teleop of the car with a Logitech F710 (XInput mode).

Custom, dependency-free joystick reader: it parses the Linux joystick API
(/dev/input/js0) directly, no pygame/SDL. Each js event is 8 bytes:
    uint32 time_ms | int16 value | uint8 type | uint8 number
type: 0x01 = button, 0x02 = axis, bit 0x80 = synthetic "init" event (ignored).

Controls (F710, XInput / mode switch X):
    Left stick X   -> steering
    Left stick Y   -> throttle (up = forward)   [held RB = deadman, see below]
    RB (hold)      -> DEADMAN: throttle only applies while held; release = coast to 0
    START          -> quit (also Ctrl-C)

SAFETY (90 km/h car):
    - Throttle is gated by a deadman button: release it and the motor goes to 0.
    - Throttle magnitude capped by --max-current and --max-throttle.
    - On exit / signal / gamepad unplug -> emergency stop.
    - First runs: keep the car on a stand or in a clear space, low --max-current.

Usage:
    OPENBLAS_CORETYPE=ARMV8 .venv/bin/python src/teleop_gamepad.py
    .venv/bin/python src/teleop_gamepad.py --max-current 4 --max-throttle 0.3
    .venv/bin/python src/teleop_gamepad.py --debug      # print raw js events to map your pad
"""

import argparse
import os
import struct
import sys
import time

sys.path.insert(0, "src")
from vesc_interface import VESCInterface

_EVENT_FMT = "IhBB"
_EVENT_SIZE = struct.calcsize(_EVENT_FMT)   # 8 bytes
_TYPE_BUTTON = 0x01
_TYPE_AXIS = 0x02
_TYPE_INIT = 0x80

# F710 (XInput) default mapping — override on the CLI if your pad differs (use --debug).
AXIS_STEER = 0      # left stick X
AXIS_THROTTLE = 1   # left stick Y (up = negative raw -> forward after negation)
BTN_DEADMAN = 5     # RB
BTN_QUIT = 7        # START


class Gamepad:
    """Non-blocking reader for /dev/input/jsN (Linux joystick API)."""

    def __init__(self, path="/dev/input/js0"):
        # O_NONBLOCK so read() never stalls the control loop.
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = {}      # number -> float [-1, 1]
        self.buttons = {}   # number -> 0/1

    def poll(self, debug=False):
        """Drain all pending events; update self.axes / self.buttons."""
        while True:
            try:
                data = os.read(self.fd, _EVENT_SIZE)
            except BlockingIOError:
                break
            if not data or len(data) < _EVENT_SIZE:
                break
            _, value, etype, number = struct.unpack(_EVENT_FMT, data)
            etype &= ~_TYPE_INIT  # ignore the init flag
            if etype == _TYPE_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
                if debug:
                    print("axis %d = %+.3f" % (number, self.axes[number]), flush=True)
            elif etype == _TYPE_BUTTON:
                self.buttons[number] = value
                if debug:
                    print("button %d = %d" % (number, value), flush=True)

    def axis(self, number):
        return self.axes.get(number, 0.0)

    def button(self, number):
        return self.buttons.get(number, 0)

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


def deadzone(x, dz=0.08):
    """Remove stick jitter near centre, rescale the rest to keep full range."""
    if abs(x) < dz:
        return 0.0
    return (x - dz * (1 if x > 0 else -1)) / (1.0 - dz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--js", default="/dev/input/js0")
    p.add_argument("--max-current", type=float, default=5.0, help="A — |current| cap")
    p.add_argument("--max-throttle", type=float, default=0.35, help="scale on stick throttle [0..1]")
    p.add_argument("--servo-center", type=float, default=0.5)
    p.add_argument("--servo-range", type=float, default=0.40)
    p.add_argument("--invert-steer", action="store_true")
    p.add_argument("--no-invert-motor", action="store_true")
    p.add_argument("--no-deadman", action="store_true", help="DANGER: throttle without holding RB")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--debug", action="store_true", help="print raw js events and exit-less mapping aid")
    p.add_argument("--axis-steer", type=int, default=AXIS_STEER)
    p.add_argument("--axis-throttle", type=int, default=AXIS_THROTTLE)
    p.add_argument("--btn-deadman", type=int, default=BTN_DEADMAN)
    p.add_argument("--btn-quit", type=int, default=BTN_QUIT)
    args = p.parse_args()

    try:
        pad = Gamepad(args.js)
    except OSError as e:
        print("[teleop] cannot open %s: %s" % (args.js, e))
        print("         checklist:")
        print("           1. F710 dongle plugged in (lsusb | grep 046d) and pad ON")
        print("           2. switch on the back set to 'X' (XInput), not 'D'")
        print("           3. if still no js0:  sudo modprobe joydev  (then replug)")
        return

    print("═" * 56)
    print("  TELEOP F710 — G-CAR-000")
    print("  max_current=%.1f A | max_throttle=%.0f%% | deadman=%s"
          % (args.max_current, args.max_throttle * 100,
             "OFF (!)" if args.no_deadman else "hold RB (btn %d)" % args.btn_deadman))
    print("  Left stick: X=direction, Y=gaz | START=quitter | Ctrl-C=stop")
    print("═" * 56)

    if args.debug:
        print("[debug] move sticks / press buttons to see their numbers. Ctrl-C to stop.")
        try:
            while True:
                pad.poll(debug=True)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            pad.close()
        return

    vesc = VESCInterface(
        port=args.port,
        servo_center=args.servo_center,
        servo_range=args.servo_range,
        current_max=args.max_current,
        invert_steer=args.invert_steer,
        invert_motor=not args.no_invert_motor,
    )

    dt = 1.0 / args.hz
    try:
        while True:
            pad.poll()
            if pad.button(args.btn_quit):
                print("\n[teleop] START pressed -> quit")
                break

            steer = deadzone(pad.axis(args.axis_steer))
            # stick up = negative raw on Y -> negate so up = forward throttle
            throttle = -deadzone(pad.axis(args.axis_throttle)) * args.max_throttle

            armed = args.no_deadman or pad.button(args.btn_deadman)
            if not armed:
                throttle = 0.0

            vesc.drive(steer, throttle)
            print("\r steer=%+.2f thr=%+.2f %s   "
                  % (steer, throttle, "ARMED" if armed else "idle "),
                  end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[teleop] Ctrl-C -> stop")
    finally:
        vesc.close()
        pad.close()


if __name__ == "__main__":
    main()

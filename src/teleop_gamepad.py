"""
teleop_gamepad.py — Manual teleop of the car with a Logitech F710 (XInput mode).

Custom, dependency-free joystick reader: it parses the Linux joystick API
(/dev/input/js0) directly, no pygame/SDL. Each js event is 8 bytes:
    uint32 time_ms | int16 value | uint8 type | uint8 number
type: 0x01 = button, 0x02 = axis, bit 0x80 = synthetic "init" event.

Controls (F710, XInput / mode switch on 'X'):
    Left stick X    -> steering
    R2 (RT)         -> forward throttle   (analog, fine speed control)
    L2 (LT)         -> reverse throttle    (analog)
    START           -> quit (also Ctrl-C)

The analog triggers double as a natural dead-man: they spring back to rest the
instant you let go, so the motor returns to 0 on release — no separate button.

SAFETY (90 km/h car):
    - Forward/reverse magnitude capped by --max-current x --max-throttle / --max-reverse.
      Defaults (25 A, full range) are tuned for this car and within the VESC's limits.
    - Trigger rest value is auto-calibrated at startup so an untouched trigger
      reads 0, never a phantom mid-throttle.
    - Emergency stop on quit / Ctrl-C / exit / gamepad unplug.
    - First runs: keep the car on a stand or clear space until you trust the mapping.

Usage:
    .venv/bin/python src/teleop_gamepad.py            # tuned defaults, no flags needed
    .venv/bin/python src/teleop_gamepad.py --debug    # live axis/button map to verify your pad
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

# F710 (XInput) mapping under the xpad js driver, confirmed on this car. This file is
# the source of truth for the js0 teleop; src/input_manager.py is a separate pygame path
# (used by data_collector) with its own mapping — don't assume the two match.
AXIS_STEER = 6      # left stick X (confirmed analog on this F710)
AXIS_ACCEL = 5      # R2 / right trigger
AXIS_BRAKE = 2      # L2 / left trigger
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


def deadzone(x, dz):
    """Remove stick jitter near centre, rescale the rest to keep full range."""
    if abs(x) < dz:
        return 0.0
    return (x - dz * (1.0 if x > 0 else -1.0)) / (1.0 - dz)


def calibrate_trigger_rest(pad, axes, window_s=0.4):
    """Sample untouched triggers to learn their rest value (xpad rests at -1).

    The press fraction (raw - rest) / (1 - rest) is then 0 while released,
    so a never-touched trigger can't command a phantom mid-throttle at startup."""
    t_end = time.time() + window_s
    while time.time() < t_end:
        pad.poll()
        time.sleep(0.02)
    return {a: pad.axis(a) for a in axes}


def trigger_fraction(pad, axis, rest):
    """Map a trigger axis to [0,1]: 0 = released, 1 = fully pressed."""
    raw = pad.axis(axis)
    frac = (raw - rest) / (1.0 - rest) if rest < 1.0 else 0.0
    return max(0.0, min(1.0, frac))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--js", default="/dev/input/js0")
    p.add_argument("--max-current", type=float, default=25.0, help="A — |current| cap")
    p.add_argument("--max-throttle", type=float, default=1.0, help="forward scale on RT [0..1]")
    p.add_argument("--max-reverse", type=float, default=1.0, help="reverse scale on LT [0..1]")
    p.add_argument("--servo-center", type=float, default=0.5)
    p.add_argument("--servo-range", type=float, default=0.40)
    p.add_argument("--invert-steer", action="store_true")
    p.add_argument("--invert-motor", action="store_true",
                   help="flip motor direction (default off: R2=forward on this car)")
    p.add_argument("--throttle-mode", choices=["current", "duty", "erpm"], default="current")
    p.add_argument("--deadzone", type=float, default=0.08)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--debug", action="store_true", help="print raw js events to map your pad")
    p.add_argument("--axis-steer", type=int, default=AXIS_STEER)
    p.add_argument("--axis-accel", type=int, default=AXIS_ACCEL)
    p.add_argument("--axis-brake", type=int, default=AXIS_BRAKE)
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

    print("═" * 60)
    print("  TELEOP F710 — G-CAR-000")

    if args.debug:
        print("  [debug] move sticks / squeeze triggers to read indices. Ctrl-C to stop.")
        print("═" * 60)
        try:
            while True:
                pad.poll(debug=True)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            pad.close()
        return

    print("  Left stick=steer | R2=forward | L2=reverse | START=quit")
    print("  max_current=%.1f A | fwd=%.0f%% | rev=%.0f%%"
          % (args.max_current, args.max_throttle * 100, args.max_reverse * 100))
    print("  Calibrating triggers — don't touch R2/L2...")
    rest = calibrate_trigger_rest(pad, [args.axis_accel, args.axis_brake])
    print("  rest: R2=%.2f L2=%.2f" % (rest[args.axis_accel], rest[args.axis_brake]))
    print("═" * 60)

    vesc = VESCInterface(
        port=args.port,
        servo_center=args.servo_center,
        servo_range=args.servo_range,
        current_max=args.max_current,
        invert_steer=args.invert_steer,
        # this car drives forward with invert OFF — overrides VESCInterface's default (True)
        invert_motor=args.invert_motor,
        throttle_mode=args.throttle_mode,
    )

    dt = 1.0 / args.hz
    try:
        while True:
            pad.poll()
            if pad.button(args.btn_quit):
                print("\n[teleop] START -> quit")
                break

            steer = deadzone(pad.axis(args.axis_steer), args.deadzone)
            rt = trigger_fraction(pad, args.axis_accel, rest[args.axis_accel])
            lt = trigger_fraction(pad, args.axis_brake, rest[args.axis_brake])
            throttle = rt * args.max_throttle - lt * args.max_reverse

            vesc.drive(steer, throttle)

            if rt > lt:
                tag = "FWD"
            elif lt > rt:
                tag = "REV"
            else:
                tag = "idle"
            print("\r steer=%+.2f thr=%+.2f [R2=%.2f L2=%.2f] %s   "
                  % (steer, throttle, rt, lt, tag), end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[teleop] Ctrl-C -> stop")
    finally:
        vesc.stop()
        vesc.close()
        pad.close()


if __name__ == "__main__":
    main()

"""
bench_erpm.py — Map the usable ERPM band in closed-loop speed control.

The 12-point sweep showed erpm tracks ~1000-2000 but collapsed at 4000 (a single
point). This maps the limit properly and tests a key hypothesis: RAMPING the eRPM
setpoint (so the closed loop stays locked while accelerating) may reach much higher
speeds than STEPPING straight to a high target (which forces it through the
low-speed dead zone all at once).

Two tests:
  1. RAMP   : setpoint lo -> hi over --ramp-secs, logs target vs measured continuously.
  2. STEPS  : jump straight to each --steps target, hold --hold s, report mean.

!!! 90 km/h car — WHEELS OFF THE GROUND. Ctrl-C = emergency stop. !!!

Usage:
  OPENBLAS_CORETYPE=ARMV8 .venv/bin/python src/bench_erpm.py
  .venv/bin/python src/bench_erpm.py --hi 12000 --steps 4000,6000,8000,10000 --yes
"""

import argparse
import sys
import time

sys.path.insert(0, "src")
from vesc_interface import VESCInterface

FW = -1.0   # forward direction for this car


def read_rpm(v):
    return v.get_rpm()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--lo", type=float, default=800.0, help="ramp start eRPM")
    p.add_argument("--hi", type=float, default=10000.0, help="ramp end eRPM")
    p.add_argument("--ramp-secs", type=float, default=7.0)
    p.add_argument("--steps", default="3000,5000,7000,9000", help="comma-separated eRPM targets")
    p.add_argument("--hold", type=float, default=2.0, help="seconds per step")
    p.add_argument("--max-erpm", type=float, default=12000.0, help="interface clamp")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("═" * 60)
    print("  VESC ERPM BAND MAP — closed-loop high-range")
    print("  ⚠️  90 km/h — ROUES EN L'AIR. Ctrl-C = stop.")
    print("═" * 60)
    if not args.yes:
        try:
            if input("  Roues en l'air ? [tape OUI] : ").strip().upper() != "OUI":
                print("  annulé."); return
        except (EOFError, KeyboardInterrupt):
            return

    vesc = VESCInterface(port=args.port, max_erpm=args.max_erpm)
    if vesc._sim_mode:
        print("  VESC non connecté — abandon."); return
    vesc._alive_running = False  # we resend continuously; avoid read/write interleave
    time.sleep(0.05)

    def tracking(meas, tgt):
        return abs(abs(meas) - tgt) < 0.30 * tgt

    try:
        # ── TEST 1 : continuous ramp ──────────────────────────────────────────
        print("\n=== TEST 1 — RAMPE continue %d -> %d eRPM (%.0fs) ==="
              % (args.lo, args.hi, args.ramp_secs), flush=True)
        t0 = time.time(); nxt = 0.0
        best_locked = 0
        while time.time() - t0 < args.ramp_secs:
            a = (time.time() - t0) / args.ramp_secs
            tgt = args.lo + (args.hi - args.lo) * a
            vesc.set_rpm(FW * tgt)
            time.sleep(0.03)
            if time.time() - t0 >= nxt:
                m = read_rpm(vesc) or 0
                ok = tracking(m, tgt)
                if ok:
                    best_locked = max(best_locked, int(tgt))
                print("   tgt=%6d  meas=%+7d  %s" % (int(tgt), m, "OK" if ok else "DÉRIVE"), flush=True)
                nxt += 0.5
        vesc.set_current(0.0); time.sleep(1.2)
        print("   -> verrouillé jusqu'à ~%d eRPM en rampe" % best_locked, flush=True)

        # ── TEST 2 : direct steps ─────────────────────────────────────────────
        print("\n=== TEST 2 — PALIERS directs (saut depuis l'arrêt) ===", flush=True)
        for tgt in [float(s) for s in args.steps.split(",")]:
            t0 = time.time(); ms = []
            while time.time() - t0 < args.hold:
                vesc.set_rpm(FW * tgt)
                time.sleep(0.03)
                r = read_rpm(vesc)
                if r is not None:
                    ms.append(r)
            vesc.set_current(0.0); time.sleep(0.6)
            mn = sum(ms) / len(ms) if ms else 0
            print("   palier %5d -> meas=%+.0f  %s"
                  % (int(tgt), mn, "TRACK" if tracking(mn, tgt) else "COLLAPSE"), flush=True)
    except KeyboardInterrupt:
        print("\n⛔ Ctrl-C")
    finally:
        vesc.set_current(0.0); time.sleep(0.05); vesc.set_current(0.0)
        vesc.close()
    print("\n=== fini ===")


if __name__ == "__main__":
    main()

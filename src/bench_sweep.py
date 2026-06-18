"""
bench_sweep.py — Empirical sweep to find the least-bad low-speed operating point.

For each (mode, value) test point it commands the motor continuously for HOLD_S,
samples the eRPM, and reports mean / std (a coherence metric: low std = smooth).
Compares the three control modes:
  - duty    : open-loop voltage (set_duty)
  - current : open-loop torque  (set_current)
  - erpm    : CLOSED-LOOP speed (set_rpm) — the previous team's intended approach

IMPORTANT — what this can and cannot do:
  The low-speed smoothness of a SENSORLESS motor is set by the VESC FOC/BLDC config
  (R/L/flux-linkage, openloop ERPM, sensorless ERPM) — tuned in VESC Tool, NOT here.
  This sweep finds the best operating point GIVEN the current config, and shows whether
  closed-loop (erpm) is smoother than open-loop. It does not re-tune the VESC.

!!! 90 km/h car — WHEELS OFF THE GROUND. Ctrl-C = emergency stop. !!!

Usage:
  OPENBLAS_CORETYPE=ARMV8 .venv/bin/python src/bench_sweep.py
  .venv/bin/python src/bench_sweep.py --hold 2.5 --yes
"""

import argparse
import sys
import time

sys.path.insert(0, "src")
from vesc_interface import VESCInterface

# Forward direction for this car (positive current spins backward -> forward is negative).
FW = -1.0

# Test points: (mode, value). Values are absolute (duty fraction, amps, eRPM).
TEST_POINTS = [
    ("duty",    0.05), ("duty",    0.10), ("duty", 0.20), ("duty", 0.35), ("duty", 0.50),
    ("current", 1.0),  ("current", 2.0),  ("current", 3.0),
    ("erpm",    500),  ("erpm",    1000), ("erpm", 2000), ("erpm", 4000),
]


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var ** 0.5


def apply(vesc, mode, value):
    if mode == "duty":
        vesc.set_duty(FW * value)
    elif mode == "current":
        vesc.set_current(FW * value)
    elif mode == "erpm":
        vesc.set_rpm(FW * value)


def measure(vesc, mode, value, secs, hz=25):
    rpms = []
    t0 = time.time()
    while time.time() - t0 < secs:
        apply(vesc, mode, value)
        time.sleep(1.0 / hz)
        r = vesc.get_rpm()
        if r is not None:
            rpms.append(r)
    return rpms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--hold", type=float, default=3.0, help="seconds per test point")
    p.add_argument("--pause", type=float, default=1.0, help="seconds at 0 between points")
    p.add_argument("--max-current", type=float, default=8.0)
    p.add_argument("--max-erpm", type=float, default=6000.0)
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("═" * 64)
    print("  VESC SWEEP — least-bad low-speed operating point")
    print("  ⚠️  90 km/h — ROUES EN L'AIR. Ctrl-C = stop.")
    print("═" * 64)
    if not args.yes:
        try:
            if input("  Roues en l'air ? [tape OUI] : ").strip().upper() != "OUI":
                print("  annulé."); return
        except (EOFError, KeyboardInterrupt):
            return

    vesc = VESCInterface(port=args.port, current_max=args.max_current, max_erpm=args.max_erpm)
    if vesc._sim_mode:
        print("  VESC non connecté — abandon."); return

    rows = []
    try:
        for mode, value in TEST_POINTS:
            label = "%-8s %6.2f" % (mode, value)
            print("\n  >>> %s  (%.1fs)" % (label, args.hold), flush=True)
            rpms = measure(vesc, mode, value, args.hold)
            apply(vesc, mode, 0); vesc.set_current(0.0)
            m, s = mean_std(rpms)
            spun = abs(m) > 150
            rows.append((mode, value, m, s, len(rpms), spun))
            print("      rpm mean=%+8.1f  std=%7.1f  n=%d  %s"
                  % (m, s, len(rpms), "SPUN" if spun else "(barely moved)"), flush=True)
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\n⛔ Ctrl-C")
    finally:
        vesc.set_current(0.0); time.sleep(0.05); vesc.set_current(0.0)
        vesc.close()

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  RÉSULTATS  (std bas = régulier ; coh = std/|mean|, bas = lisse)")
    print("═" * 64)
    print("  %-8s %7s %9s %9s %7s  %s" % ("mode", "value", "mean_rpm", "std", "coh", "verdict"))
    for mode, value, m, s, n, spun in rows:
        coh = s / (abs(m) + 1e-6)
        verdict = "—"
        if spun:
            verdict = "LISSE" if coh < 0.15 else ("ok" if coh < 0.4 else "saccadé")
        print("  %-8s %7.2f %9.1f %9.1f %7.2f  %s" % (mode, value, m, s, coh, verdict))
    # Best smooth spinner
    spinners = [(s / (abs(m) + 1e-6), mode, value, m) for mode, value, m, s, n, sp in rows if sp]
    if spinners:
        spinners.sort()
        coh, mode, value, m = spinners[0]
        print("\n  → Plus lisse : mode=%s value=%.2f (mean=%.0f erpm, coh=%.2f)" % (mode, value, m, coh))
    else:
        print("\n  → Aucun point n'a vraiment tourné proprement — config VESC à revoir (VESC Tool).")
    print("═" * 64)


if __name__ == "__main__":
    main()

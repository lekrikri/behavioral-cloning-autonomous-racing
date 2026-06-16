"""
bench_vesc.py — Bench test for the VESC control layer (steering + throttle).

Named bench_* (not test_*) on purpose: it has real hardware side effects and
must NOT be auto-collected/run by pytest. Run it manually, wheels off the ground.

Runs 6 scripted sequences:
  1. steer left, steer right, back to centre
  2. steer full-left -> progressively full-right (sweep)
  3. accelerate then stop
  4. reverse then stop
  5. accelerate -> progressively reverse
  6. steering + throttle at the same time

!!! THIS CAR CAN REACH ~90 km/h !!!
  - Always run with the WHEELS OFF THE GROUND (car on a stand).
  - The motor phases (3-6) are gated behind a confirmation prompt.
  - DEFAULT mode is "duty" (voltage/speed): the wheel spins FAST and smooth, up to
    --max-duty (50%). On 4S that is a high free-running rpm — keep phases short.
  - Mode "current" (torque) is gentler/jerky at low speed: --mode current.
  - Ctrl-C at any time triggers an emergency stop.

Usage:
  OPENBLAS_CORETYPE=ARMV8 .venv/bin/python src/bench_vesc.py           # duty 50%, full test
  .venv/bin/python src/bench_vesc.py --steer-only                     # no motor
  .venv/bin/python src/bench_vesc.py --max-duty 0.3                   # gentler top speed (30%)
  .venv/bin/python src/bench_vesc.py --mode current --max-current 3   # torque mode
  .venv/bin/python src/bench_vesc.py --yes                            # skip prompts
"""

import argparse
import sys
import time

sys.path.insert(0, "src")
from vesc_interface import VESCInterface


def ramp(vesc, s0, s1, t0, t1, duration, hz=50):
    """Linearly interpolate steering s0->s1 and throttle t0->t1 over `duration` s.

    Commands are resent at `hz` so the VESC watchdog keeps the setpoint alive."""
    n = max(1, int(duration * hz))
    dt = 1.0 / hz
    for i in range(n + 1):
        a = i / n
        vesc.drive(s0 + (s1 - s0) * a, t0 + (t1 - t0) * a)
        time.sleep(dt)


def hold(vesc, steer, throttle, duration, hz=50):
    ramp(vesc, steer, steer, throttle, throttle, duration, hz)


def phase(n, title):
    print("\n" + "─" * 56)
    print("  [%d/6] %s" % (n, title))
    print("─" * 56)


def confirm(msg, auto_yes):
    if auto_yes:
        print("  (--yes) " + msg + " -> OK")
        return True
    try:
        return input("  >>> " + msg + " [tape OUI] : ").strip().upper() == "OUI"
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--mode", choices=["duty", "current"], default="duty",
                   help="duty = voltage/speed (smooth, default); current = torque")
    p.add_argument("--max-current", type=float, default=3.0, help="A — |current| cap (current mode)")
    p.add_argument("--max-duty", type=float, default=0.5, help="duty cap (duty mode); 0.5 = 50%%")
    p.add_argument("--level", type=float, default=1.0, help="peak throttle fraction [0..1] of the mode cap")
    p.add_argument("--servo-center", type=float, default=0.5)
    p.add_argument("--servo-range", type=float, default=0.40)
    p.add_argument("--invert-steer", action="store_true")
    p.add_argument("--no-invert-motor", action="store_true", help="disable motor direction inversion (default: inverted)")
    p.add_argument("--steer-only", action="store_true", help="skip motor phases 3-6")
    p.add_argument("--yes", action="store_true", help="skip confirmation prompts")
    args = p.parse_args()

    lvl = max(0.0, min(1.0, args.level))
    # pk_pct = effective peak shown in prints: duty% in duty mode, throttle-fraction% in current mode
    pk_pct = lvl * (args.max_duty if args.mode == "duty" else 1.0) * 100

    print("═" * 56)
    print("  VESC BENCH TEST — G-CAR-000")
    print("  ⚠️  VMax ~90 km/h — ROUES EN L'AIR OBLIGATOIRE")
    if args.mode == "duty":
        print("  mode=DUTY (vitesse) | peak = %.0f%% duty (cap %.0f%%) — tourne VITE"
              % (pk_pct, args.max_duty * 100))
    else:
        print("  mode=CURRENT (couple) | peak = %.1f A (cap %.1f A)" % (lvl * args.max_current, args.max_current))
    print("═" * 56)

    vesc = VESCInterface(
        port=args.port,
        servo_center=args.servo_center,
        servo_range=args.servo_range,
        current_max=args.max_current,
        invert_steer=args.invert_steer,
        invert_motor=not args.no_invert_motor,
        throttle_mode=args.mode,
        max_duty=args.max_duty,
    )
    if vesc._sim_mode:
        print("  ⚠️  VESC non connecté — mode SIMULATION (aucune commande réelle).")

    try:
        # ── Phase 1 : gauche / droite / centre ────────────────────────────────
        phase(1, "Direction : GAUCHE → DROITE → CENTRE")
        print("  gauche (-1.0)…");  hold(vesc, -1.0, 0.0, 1.2)
        print("  droite (+1.0)…");  hold(vesc, +1.0, 0.0, 1.2)
        print("  centre (0.0)…");   hold(vesc,  0.0, 0.0, 0.8)

        # ── Phase 2 : sweep gauche complète → droite complète ─────────────────
        phase(2, "Direction : SWEEP gauche complète → droite complète")
        hold(vesc, -1.0, 0.0, 0.6)
        print("  sweep -1 → +1…");  ramp(vesc, -1.0, +1.0, 0.0, 0.0, 3.0)
        print("  retour centre…");  hold(vesc, 0.0, 0.0, 0.5)

        if args.steer_only:
            print("\n  --steer-only : phases moteur ignorées.")
            return

        # ── Garde-fou avant les phases moteur ─────────────────────────────────
        print("\n" + "!" * 56)
        print("  PHASES MOTEUR — la roue va tourner (jusqu'à %.0f%% throttle)." % (pk_pct))
        print("  Vérifie que les ROUES NE TOUCHENT PAS LE SOL.")
        print("!" * 56)
        if not confirm("Roues en l'air, prêt à lancer le moteur ?", args.yes):
            print("  Annulé. (phases moteur non exécutées)")
            return

        # ── Phase 3 : accélère puis stop ──────────────────────────────────────
        phase(3, "Moteur : ACCÉLÈRE puis STOP")
        print("  ramp 0 → +%.0f%%…" % (pk_pct)); ramp(vesc, 0, 0, 0.0, +lvl, 0.8)
        print("  maintien…");                        hold(vesc, 0, +lvl, 0.7)
        print("  ramp → 0…");                        ramp(vesc, 0, 0, +lvl, 0.0, 0.5)
        vesc.set_throttle(0.0); time.sleep(0.6)

        # ── Phase 4 : marche arrière puis stop ────────────────────────────────
        phase(4, "Moteur : MARCHE ARRIÈRE puis STOP")
        print("  ramp 0 → -%.0f%%…" % (pk_pct)); ramp(vesc, 0, 0, 0.0, -lvl, 0.8)
        print("  maintien…");                        hold(vesc, 0, -lvl, 0.7)
        print("  ramp → 0…");                        ramp(vesc, 0, 0, -lvl, 0.0, 0.5)
        vesc.set_throttle(0.0); time.sleep(0.6)

        # ── Phase 5 : accélère → marche arrière progressivement ───────────────
        phase(5, "Moteur : ACCÉLÈRE → MARCHE ARRIÈRE progressivement")
        print("  ramp +%.0f%% → -%.0f%%…" % (pk_pct, pk_pct))
        ramp(vesc, 0, 0, +lvl, -lvl, 3.0)
        vesc.set_throttle(0.0); time.sleep(0.6)

        # ── Phase 6 : direction + gaz simultanés ──────────────────────────────
        phase(6, "Direction + Gaz SIMULTANÉS")
        print("  steer -1→+1 pendant un pulse moteur…")
        # throttle: 0 -> +lvl -> 0 while steering sweeps -1 -> +1
        ramp(vesc, -1.0, 0.0, 0.0, +lvl, 1.5)
        ramp(vesc,  0.0, +1.0, +lvl, 0.0, 1.5)
        print("  retour neutre…"); hold(vesc, 0.0, 0.0, 0.3)

        print("\n✅ Test terminé.")

    except KeyboardInterrupt:
        print("\n⛔ Ctrl-C — arrêt d'urgence.")
    finally:
        vesc.close()


if __name__ == "__main__":
    main()

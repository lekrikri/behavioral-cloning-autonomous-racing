"""
calibrate_servo.py — Calibration servo AVANT premier roulage.

Procédure :
  1. Voiture surélevée (roues en l'air)
  2. Moteur DÉSACTIVÉ (duty=0 forcé)
  3. Suivre les instructions à l'écran

Usage :
  python3.8 src/calibrate_servo.py [--port /dev/ttyACM0]
"""

import time
import argparse
import sys
import numpy as np

sys.path.insert(0, "src")
from vesc_interface import VESCInterface

def send_servo_only(vesc, servo_pos: float):
    """Envoie uniquement la position servo, moteur à 0."""
    import pyvesc
    servo_pos = float(np.clip(servo_pos, 0.10, 0.90))
    try:
        vesc.ser.write(pyvesc.encode(pyvesc.SetDutyCycle(0)))
        vesc.ser.write(pyvesc.encode(pyvesc.SetServoPos(servo_pos)))
    except Exception as e:
        print(f"  [!] Erreur: {e}")

def wait(msg="Appuie sur Entrée pour continuer..."):
    input(f"\n  >>> {msg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--center", type=float, default=0.50)
    parser.add_argument("--range",  type=float, default=0.35)
    args = parser.parse_args()

    print("═" * 52)
    print("  CALIBRATION SERVO — G-CAR-000")
    print("  ⚠️  Voiture surélevée obligatoire !")
    print("═" * 52)

    vesc = VESCInterface(port=args.port, duty_max=0.0)
    if vesc._sim_mode:
        print("  ❌ VESC non connecté — vérifier câble USB")
        return

    center = args.center
    srv_range = args.range

    # ── Étape 1 : Centre ────────────────────────────────────────────────────
    print("\n[1/4] TEST CENTRE (steer=0.0)")
    print(f"  servo_pos = {center:.3f}")
    send_servo_only(vesc, center)
    print("  → Les roues doivent être PARFAITEMENT droites.")
    print("  → Si elles dévient, note l'offset à corriger.")
    offset = input("  Entrer offset de correction (ex: 0.02 ou -0.03, 0 si OK) : ")
    try:
        center += float(offset)
        print(f"  ✅ Nouveau center = {center:.3f}")
        send_servo_only(vesc, center)
        time.sleep(1.0)
    except ValueError:
        pass

    # ── Étape 2 : Sens de braquage ───────────────────────────────────────────
    print("\n[2/4] TEST INVERSION (steer=+0.3)")
    send_servo_only(vesc, center + srv_range * 0.3)
    time.sleep(1.5)
    print("  steer=+0.3 envoyé. Les roues doivent tourner vers la DROITE.")
    rep = input("  Les roues tournent vers la droite ? (o/n) : ").strip().lower()
    invert = rep == "n"
    if invert:
        print("  → invert_steer=True sera appliqué")
    else:
        print("  ✅ Sens correct, pas d'inversion nécessaire")
    send_servo_only(vesc, center)
    time.sleep(0.5)

    # ── Étape 3 : Amplitude ─────────────────────────────────────────────────
    print("\n[3/4] TEST AMPLITUDE — écoute le servo")
    print(f"  range actuel = ±{srv_range:.2f}")

    for label, steer in [("MAX GAUCHE (-1.0)", -1.0), ("MAX DROITE (+1.0)", 1.0)]:
        s = -steer if invert else steer
        pos = center + srv_range * s
        print(f"  → {label} : servo_pos={pos:.3f}")
        send_servo_only(vesc, pos)
        time.sleep(2.0)
        rep = input("  Le servo grogne/force en butée ? (o/n) : ").strip().lower()
        if rep == "o":
            srv_range -= 0.03
            srv_range = round(srv_range, 3)
            print(f"  ⚠️  Range réduit à ±{srv_range:.2f}")

    send_servo_only(vesc, center)
    time.sleep(0.5)

    # ── Étape 4 : Sweep fluide ───────────────────────────────────────────────
    print("\n[4/4] SWEEP FLUIDE (steer -1 → +1)")
    wait("Appuie sur Entrée pour lancer le sweep...")
    for steer in np.linspace(-1.0, 1.0, 30):
        s = -steer if invert else steer
        send_servo_only(vesc, center + srv_range * s)
        time.sleep(0.08)
    send_servo_only(vesc, center)
    print("  ✅ Sweep terminé — mouvement fluide et symétrique ?")

    # ── Résumé ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  RÉSULTATS DE CALIBRATION")
    print("═" * 52)
    print(f"  servo_center  = {center:.3f}")
    print(f"  servo_range   = {srv_range:.3f}")
    print(f"  invert_steer  = {invert}")
    print()
    print("  → Copie ces valeurs dans inference_realcar.py :")
    print(f"     --servo-center {center:.3f}")
    print(f"     --servo-range  {srv_range:.3f}")
    if invert:
        print("     --invert-steer")
    print("═" * 52)

    vesc.stop()
    vesc.close()

if __name__ == "__main__":
    main()

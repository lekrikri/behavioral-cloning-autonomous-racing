"""
collect_real.py — Collecte de données sur la piste physique.
Lit la manette, conduit le VESC, extrait les rayons via le Hub et sauvegarde en CSV.

Usage (Sur la Jetson) :
  OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.collect_real --output data/run_real_01.csv
"""

import argparse
import csv
import time
import sys
from pathlib import Path
import cv2

from src.cam.hub import FrameClient, ensure_hub_or_prompt, SHM_COLOR
from src.control.input_manager import create_input_manager
from src.control.vesc_interface import VESCInterface
from src.mask.visual_rays import VisualRays

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/run_real_01.csv", help="Fichier CSV de sortie")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Port du VESC")
    parser.add_argument("--duty-max", type=float, default=0.20, help="Vitesse max pour la collecte")
    args = parser.parse_args()

    # 1. Vérifier que le Hub tourne
    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    # 2. Initialisations
    client = FrameClient(stream=SHM_COLOR)
    vesc = VESCInterface(port=args.port, throttle_mode="duty", max_duty=args.duty_max)
    manager = create_input_manager()
    vr = VisualRays(mode="hsv") # Utilise ta config HSV qui marche bien
    
    manager.start()

    # 3. Préparation du CSV (Même format que ton simulateur)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rays = 20
    header = ["episode_id", "step", "timestamp"] + [f"ray_{i}" for i in range(n_rays)] + ["speed", "steering", "acceleration"]
    
    episode_id = int(time.time())
    step = 0
    
    print(f"\n[Collecte] Prêt. Fichier : {out_path}")
    print("[Collecte] Appuie sur START (ou Echap) pour quitter et sauvegarder.\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        try:
            while not manager.should_quit():
                # Lire la manette
                steering, acceleration = manager.get_actions()
                
                # Envoyer au moteur
                vesc.drive(steering, acceleration)
                
                # Lire la caméra (Zéro-copie via le Hub)
                try:
                    bgr = client.getCvFrame()
                except ConnectionError:
                    print("Erreur de connexion au Hub vidéo.")
                    break
                
                # Calculer les rayons
                rays = vr(bgr)
                
                # Sauvegarder dans le CSV (seulement si la voiture avance un minimum)
                # On triche un peu sur "speed" en mettant l'accélération pour rester compatible avec dataset.py
                if abs(acceleration) > 0.05:
                    row = [episode_id, step, time.time()] + rays.tolist() + [acceleration, steering, acceleration]
                    writer.writerow(row)
                    step += 1
                
                # Affichage console pour voir que ça tourne
                if step % 10 == 0:
                    print(f"\rFrames sauvées: {step} | Steer: {steering:+.2f} | Accel: {acceleration:+.2f}", end="")
                    
                time.sleep(0.03) # ~30 FPS

        except KeyboardInterrupt:
            print("\nArrêt manuel.")
            
    # Fermeture propre
    vesc.stop()
    vesc.close()
    manager.stop()
    client.close()
    print(f"\n[Collecte] Terminé. {step} frames enregistrées dans {out_path}.")

if __name__ == "__main__":
    main()

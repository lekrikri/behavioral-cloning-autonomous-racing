"""
collect_real.py — Collecte de données MAXIMALE sur la piste physique.
Lit la manette, conduit le VESC, extrait les rayons (RGB + Depth) et l'IMU via le Hub.

Usage (Sur la Jetson) :
  OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.collect_real --output data/run_real_01.csv
"""

import argparse
import csv
import time
import sys
from pathlib import Path
import numpy as np

from src.cam.hub import FrameClient, ensure_hub_or_prompt, SHM_COLOR, SHM_DEPTH, SHM_IMU
from src.control.input_manager import create_input_manager
from src.control.vesc_interface import VESCInterface
from src.mask.visual_rays import VisualRays
from src.mask.depth_rays import DepthToRays

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/run_real_01.csv", help="Fichier CSV de sortie")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Port du VESC")
    parser.add_argument("--duty-max", type=float, default=0.20, help="Vitesse max pour la collecte")
    args = parser.parse_args()

    # 1. Vérifier que le Hub tourne sur le flux principal
    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    # 2. Initialisation des clients pour les 3 flux
    client_color = FrameClient(stream=SHM_COLOR)
    client_depth = FrameClient(stream=SHM_DEPTH)
    client_imu   = FrameClient(stream=SHM_IMU)
    
    vesc = VESCInterface(port=args.port, throttle_mode="duty", max_duty=args.duty_max)
    manager = create_input_manager()
    
    vr = VisualRays(mode="hsv") # Rayons via lignes blanches
    dr = DepthToRays()          # Rayons via caméra stéréoscopique 3D
    
    manager.start()

    # 3. Préparation du CSV : 20 rayons RGB + 20 rayons Depth + 3 axes IMU
    n_rays = 20
    header = ["episode_id", "step", "timestamp"] 
    header += [f"ray_vis_{i}" for i in range(n_rays)]
    header += [f"ray_depth_{i}" for i in range(n_rays)]
    header += ["gyro_x", "gyro_y", "gyro_z", "speed", "steering", "acceleration"]
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episode_id = int(time.time())
    step = 0
    
    print(f"\n[Collecte] Prêt. Fichier : {out_path}")
    print("[Collecte] N'oublie pas de lancer le hub avec le flag --depth !")
    print("[Collecte] Appuie sur START (ou Echap) pour quitter et sauvegarder.\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        try:
            while not manager.should_quit():
                # Commandes manette → Moteur
                steering, acceleration = manager.get_actions()
                vesc.drive(steering, acceleration)
                
                # Lecture bloquante de la caméra (donne le tempo à 30 FPS)
                try:
                    bgr = client_color.getCvFrame() 
                except ConnectionError:
                    print("Erreur de connexion au Hub vidéo RGB.")
                    break
                    
                # Lecture asynchrone (latest) pour ne pas ralentir la boucle
                depth = client_depth.latest()
                imu_data = client_imu.latest()
                
                # --- EXTRACTION DES FEATURES ---
                rays_vis = vr(bgr)
                
                # Profondeur (Si absente, on met tout à 1.0 = dégagé)
                if depth is not None and depth.max() >= 200:
                    rays_depth = dr(depth)
                else:
                    rays_depth = np.ones(n_rays, dtype=np.float32)
                    
                # IMU (Si absent, on met à 0.0)
                if imu_data is not None:
                    gx, gy, gz = float(imu_data[0, 0]), float(imu_data[0, 1]), float(imu_data[0, 2])
                else:
                    gx, gy, gz = 0.0, 0.0, 0.0
                
                # Sauvegarde si la voiture est en mouvement (accel significative)
                if abs(acceleration) > 0.05:
                    row = [episode_id, step, time.time()] 
                    row += rays_vis.tolist() 
                    row += rays_depth.tolist() 
                    row += [gx, gy, gz, acceleration, steering, acceleration]
                    writer.writerow(row)
                    step += 1
                
                # Feedback console
                if step % 10 == 0:
                    print(f"\rFrames: {step} | Steer: {steering:+.2f} | Accel: {acceleration:+.2f} | Depth dispo: {depth is not None} | IMU dispo: {imu_data is not None}   ", end="")
                    
        except KeyboardInterrupt:
            print("\nArrêt manuel via console.")
            
    # Fermeture propre
    vesc.stop()
    vesc.close()
    manager.stop()
    client_color.close()
    client_depth.close()
    client_imu.close()
    print(f"\n[Collecte] Terminé. {step} frames enregistrées dans {out_path}.")

if __name__ == "__main__":
    main()
    
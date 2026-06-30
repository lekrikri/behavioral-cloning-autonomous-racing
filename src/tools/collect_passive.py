"""
collect_passive.py — Collecte de données en "Écoute Passive".
Laisse le 'core' (teleop_gamepad.py) piloter le VESC. Ce script se contente 
de lire la RAM pour associer les images aux commandes du pilote. ZÉRO ACCOUP.
"""

import argparse
import csv
import time
import sys
from pathlib import Path
import numpy as np
import os

from src.cam.hub import FrameClient, ensure_hub_or_prompt, SHM_COLOR, SHM_DEPTH, SHM_IMU
from src.mask.visual_rays import VisualRays
from src.mask.depth_rays import DepthToRays

def get_telemetry():
    """Lit les actions actuelles du pilote depuis la RAM."""
    try:
        with open("/dev/shm/robocar_telemetry.txt", "r") as f:
            data = f.read().split(',')
            if len(data) == 2:
                return float(data[0]), float(data[1])
    except Exception:
        pass
    return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dataset_piste_1.csv", help="Fichier CSV de sortie")
    args = parser.parse_args()

    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    client_color = FrameClient(stream=SHM_COLOR)
    client_depth = FrameClient(stream=SHM_DEPTH)
    client_imu   = FrameClient(stream=SHM_IMU)
    
    vr = VisualRays(mode="hsv")
    dr = DepthToRays()

    n_rays = 20
    header = ["episode_id", "step", "timestamp"] 
    header += [f"ray_vis_{i}" for i in range(n_rays)]
    header += [f"ray_depth_{i}" for i in range(n_rays)]
    header += ["gyro_x", "gyro_y", "gyro_z", "speed", "steering", "acceleration"]
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episode_id = int(time.time())
    step = 0
    
    last_active_time = time.time()
    COASTING_TIMEOUT = 1.5
    
    print(f"\n[Collecte Passive] Prêt. Fichier : {out_path}")
    print("[Collecte Passive] En attente des actions du pilote (Allumez la manette)...")
    print("[Collecte Passive] Appuyez sur Ctrl+C ici pour arrêter et sauvegarder.\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        try:
            # La boucle est rythmée par la caméra (30 Hz)
            while True:
                try:
                    # Attente bloquante de la prochaine image
                    bgr = client_color.getCvFrame() 
                except ConnectionError:
                    print("Erreur de connexion au Hub vidéo RGB.")
                    break
                    
                depth = client_depth.latest()
                imu_data = client_imu.latest()
                
                # Lecture passive des actions du pilote (dictées par le Core)
                steer, accel = get_telemetry()
                
                rays_vis = vr(bgr)
                if depth is not None and depth.max() >= 200:
                    rays_depth = dr(depth)
                else:
                    rays_depth = np.ones(n_rays, dtype=np.float32)
                    
                if imu_data is not None:
                    gx, gy, gz = float(imu_data[0, 0]), float(imu_data[0, 1]), float(imu_data[0, 2])
                else:
                    gx, gy, gz = 0.0, 0.0, 0.0
                
                # Sauvegarde intelligente (avec Coasting)
                if abs(accel) > 0.05 or abs(steer) > 0.05:
                    last_active_time = time.time()
                
                is_recording = (time.time() - last_active_time) < COASTING_TIMEOUT
                
                if is_recording:
                    row = [episode_id, step, time.time()] 
                    row += rays_vis.tolist() 
                    row += rays_depth.tolist() 
                    row += [gx, gy, gz, accel, steer, accel]
                    writer.writerow(row)
                    step += 1
                
                if step % 10 == 0:
                    etat = "REC" if is_recording else "PAUSE"
                    print(f"\rFrames: {step} [{etat}] | Steer: {steer:+.2f} | Accel: {accel:+.2f}   ", end="")
                    
        except KeyboardInterrupt:
            print("\nArrêt manuel de la collecte.")
            
    client_color.close()
    client_depth.close()
    client_imu.close()
    
    # Nettoyage du fichier de télémétrie pour le prochain run
    if os.path.exists("/dev/shm/robocar_telemetry.txt"):
        os.remove("/dev/shm/robocar_telemetry.txt")
        
    print(f"\n[Collecte Passive] Terminé. {step} frames enregistrées dans {out_path}.")

if __name__ == "__main__":
    main()

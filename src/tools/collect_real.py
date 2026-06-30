"""
collect_real.py — Collecte de données MAXIMALE (Boucle Asynchrone Non-Bloquante).
Priorité absolue au moteur à 50Hz. Évite le problème du GIL de Python.
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

    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    # 1. Initialisation
    client_color = FrameClient(stream=SHM_COLOR)
    client_depth = FrameClient(stream=SHM_DEPTH)
    client_imu   = FrameClient(stream=SHM_IMU)
    
    # On force la connexion au hub pour pouvoir lire les numéros de séquence (frames)
    client_color.connect()
    
    vesc = VESCInterface(port=args.port, throttle_mode="duty", max_duty=args.duty_max)
    manager = create_input_manager()
    vr = VisualRays(mode="hsv")
    dr = DepthToRays()
    
    manager.start()

    # 2. Préparation du CSV
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
    
    # On mémorise la dernière image vue par le script
    last_seq = client_color.reader.latest_seq() if client_color.reader else 0
    
    print(f"\n[Collecte] Prêt. Fichier : {out_path}")
    print("[Collecte] Mode Non-Bloquant 50Hz : Moteur prioritaire.")
    print("[Collecte] Appuie sur START (ou Echap) pour quitter.\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        try:
            while not manager.should_quit():
                t_start = time.time()
                
                # ==========================================
                # ÉTAPE A : PRIORITÉ ABSOLUE (Le Moteur)
                # ==========================================
                steer, accel = manager.get_actions()
                vesc.drive(steer, accel)
                
                # ==========================================
                # ÉTAPE B : LA CAMÉRA (Seulement si prête)
                # ==========================================
                # On regarde quel est le numéro de la dernière image sur le Hub
                current_seq = client_color.reader.latest_seq() if client_color.reader else 0
                
                # S'il est différent, c'est qu'une NOUVELLE image est arrivée
                if current_seq != last_seq:
                    seq, bgr = client_color.get() # On la récupère instantanément, sans bloquer
                    last_seq = seq
                    
                    # On tire les infos des autres flux (instantané)
                    depth = client_depth.latest()
                    imu_data = client_imu.latest()
                    
                    # Traitement des rayons
                    rays_vis = vr(bgr)
                    
                    if depth is not None and depth.max() >= 200:
                        rays_depth = dr(depth)
                    else:
                        rays_depth = np.ones(n_rays, dtype=np.float32)
                        
                    if imu_data is not None:
                        gx, gy, gz = float(imu_data[0, 0]), float(imu_data[0, 1]), float(imu_data[0, 2])
                    else:
                        gx, gy, gz = 0.0, 0.0, 0.0
                    
                    # Sauvegarde
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
                        
                # ==========================================
                # ÉTAPE C : CADENCE STRICTE
                # ==========================================
                # On force la boucle à durer exactement 20 millisecondes (50 Hz)
                elapsed = time.time() - t_start
                sleep_time = max(0.0, 0.02 - elapsed)
                time.sleep(sleep_time)
                    
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
    
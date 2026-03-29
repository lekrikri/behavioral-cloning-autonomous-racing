"""
data_collector.py — Enregistre les trajectoires (observation, action) dans un CSV.

Usage:
    python src/data_collector.py --output data/run_01.csv --fov 180 --rays 10

Le simulateur doit tourner avant de lancer ce script.
Conduire avec Z/S/Q/D ou flèches. ESC pour arrêter et sauvegarder.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import RobocarEnv, Observation, SIMULATOR_PATH
from src.input_manager import create_input_manager


def build_csv_header(n_rays: int) -> list[str]:
    """Construit l'en-tête CSV dynamiquement selon le nombre de rayons."""
    ray_cols = [f"ray_{i}" for i in range(n_rays)]
    return ["episode_id", "step", "timestamp", *ray_cols, "speed", "steering", "acceleration"]


def collect(
    output_path: str,
    config_path: str = "config.json",
    port: int = 5005,
    max_frames: int = 0,  # 0 = illimité
    min_speed_threshold: float = -2.0,  # ignorer les frames à l'arrêt (speed est en [-1,1])
    verbose: bool = True,
    launch: bool = False,
):
    """
    Boucle principale de collecte.

    Paramètres
    ----------
    output_path : str
        Fichier CSV de sortie.
    config_path : str
        Config des agents.
    port : int
        Port gRPC.
    max_frames : int
        Nombre maximum de frames à enregistrer (0 = infini).
    min_speed_threshold : float
        Ne pas enregistrer les frames où speed < seuil (filtre arrêt).
    verbose : bool
        Afficher les stats en temps réel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = json.load(f)
    n_rays = cfg["agents"][0]["nbRay"]
    header = build_csv_header(n_rays)

    manager = create_input_manager()
    manager.start()

    frame_count = 0
    skipped = 0
    start_time = time.time()
    episode_id = int(start_time)  # identifiant unique par session de collecte
    step = 0                      # compteur de frames dans l'épisode

    print(f"[DataCollector] Démarrage — sortie: {output_path}")
    print(f"[DataCollector] Rayons: {n_rays} | Max frames: {max_frames or 'illimité'}")
    print("[DataCollector] Conduisez avec Z/S/Q/D. ESC = sauvegarder et quitter.\n")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        sim_path = SIMULATOR_PATH if launch else None
        with RobocarEnv(config_path=config_path, port=port,
                        simulator_path=sim_path, no_graphics=False) as env:
            observations = env.reset()
            print("[DataCollector] Connecté! Conduisez maintenant. ESC = quitter.\n", flush=True)

            last_print = time.time()
            while not manager.should_quit():
                if max_frames > 0 and frame_count >= max_frames:
                    print(f"\n[DataCollector] Limite {max_frames} frames atteinte.")
                    break

                # Récupérer les actions du joueur
                steering, acceleration = manager.get_actions()

                # Envoyer les actions au simulateur
                env.send_actions(steering=steering, acceleration=acceleration)
                try:
                    observations = env.step()
                except Exception:
                    print("\n[DataCollector] Simulateur déconnecté — sauvegarde en cours...")
                    break

                if not observations:
                    continue

                obs = observations[0]

                # Filtrer les frames à l'arrêt (optionnel)
                if obs.speed < min_speed_threshold:
                    skipped += 1
                else:
                    # Enregistrer la frame
                    timestamp = time.time()
                    row = [episode_id, step, timestamp, *obs.rays.tolist(), obs.speed, steering, acceleration]
                    writer.writerow(row)
                    frame_count += 1
                    step += 1

                if verbose and time.time() - last_print >= 1.0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  Frames: {frame_count:>6} | Ignorées: {skipped:>4} | "
                        f"FPS: {fps:.1f} | Steer: {steering:+.2f} | Accel: {acceleration:+.2f}    ",
                        end="", flush=True
                    )
                    last_print = time.time()

    manager.stop()
    elapsed = time.time() - start_time
    print(f"\n\n[DataCollector] Terminé!")
    print(f"  Frames enregistrées : {frame_count}")
    print(f"  Frames ignorées     : {skipped}")
    print(f"  Durée totale        : {elapsed:.1f}s")
    print(f"  FPS moyen           : {frame_count/elapsed:.1f}" if elapsed > 0 else "")
    print(f"  Fichier             : {output_path} ({output_path.stat().st_size // 1024} KB)")
    return frame_count


def merge_datasets(input_dir: str, output_path: str):
    """
    Fusionne plusieurs fichiers CSV de trajectoires en un seul dataset.
    Utile pour combiner plusieurs sessions de collecte.
    """
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"[ERROR] Aucun CSV trouvé dans {input_dir}")
        return

    import pandas as pd
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)
        print(f"  {f.name}: {len(df)} frames")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset fusionné: {len(merged)} frames totales → {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Collecte de données Robocar")
    subparsers = parser.add_subparsers(dest="command")

    # Commande collect
    p_collect = subparsers.add_parser("collect", help="Collecter des trajectoires")
    p_collect.add_argument("--output", required=True, help="Fichier CSV de sortie")
    p_collect.add_argument("--config", default="config.json")
    p_collect.add_argument("--port", type=int, default=5005)
    p_collect.add_argument("--launch", action="store_true", help="Lance le simulateur automatiquement")
    p_collect.add_argument("--max-frames", type=int, default=0)
    p_collect.add_argument("--min-speed", type=float, default=0.0)

    # Commande merge
    p_merge = subparsers.add_parser("merge", help="Fusionner plusieurs CSV")
    p_merge.add_argument("--input-dir", required=True)
    p_merge.add_argument("--output", required=True)

    # Compatibilité: python data_collector.py --output ...
    parser.add_argument("--output", help="(raccourci collect)")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--max-frames", type=int, default=0)

    args = parser.parse_args()

    if args.command == "merge":
        merge_datasets(args.input_dir, args.output)
    elif args.command == "collect" or (hasattr(args, "output") and args.output):
        output = args.output
        if not output:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"data/run_{ts}.csv"
        collect(
            output_path=output,
            config_path=args.config,
            port=args.port,
            max_frames=args.max_frames,
            launch=getattr(args, "launch", False),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

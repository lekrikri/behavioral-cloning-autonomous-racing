"""Entrypoint du superviseur.

    python3 -m core [--profile P1] [--vehicle ...] [--profiles ...]
ou  python3 core/__main__.py
"""
import argparse
import os
import signal
import sys

if __package__ in (None, ""):  # exécution directe du fichier
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.supervisor import Supervisor
else:
    from .supervisor import Supervisor


def main():
    p = argparse.ArgumentParser(description="Robocar core — superviseur")
    p.add_argument("--vehicle", default=None, help="chemin configs/vehicle.json")
    p.add_argument("--profiles", default=None, help="chemin configs/profiles.json")
    p.add_argument("--profile", default=None, help="profil de départ (défaut: config)")
    args = p.parse_args()

    # Bloquer SIGINT/SIGTERM AVANT de lancer les threads (ils héritent du masque) ; on les
    # attend dans le thread principal via sigwait → attente passive, zéro réveil, robuste.
    sigs = {signal.SIGINT, signal.SIGTERM}
    signal.pthread_sigmask(signal.SIG_BLOCK, sigs)

    # cwd des workers = racine repo (pour que 'src/...' résolve)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sup = Supervisor(args.vehicle, args.profiles, args.profile, cwd=repo_root)
    sup.start()
    try:
        signal.sigwait(sigs)
    finally:
        sup.shutdown()
        print("[core] arrêt propre")


if __name__ == "__main__":
    main()

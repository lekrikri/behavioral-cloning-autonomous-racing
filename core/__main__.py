"""Entrypoint du superviseur.

    python3 -m core [--profile P1] [--vehicle ...] [--profiles ...]
ou  python3 core/__main__.py
"""
import argparse
import os
import signal
import sys
import time

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

    # cwd des workers = racine repo (pour que 'src/...' résolve)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sup = Supervisor(args.vehicle, args.profiles, args.profile, cwd=repo_root)
    sup.start()

    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        while not stop["flag"]:
            time.sleep(0.5)
    finally:
        sup.shutdown()
        print("[core] arrêt propre")


if __name__ == "__main__":
    main()

"""Chargement des deux couches de config : statique véhicule + profils (JSON)."""
import json
from pathlib import Path

# core/config.py -> repo/configs
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_json(path):
    with open(str(path)) as f:
        return json.load(f)


def load_vehicle(path=None):
    """Params figés : caméra, contrôle, vesc, ui. Voir configs/vehicle.json."""
    return _load_json(Path(path) if path else CONFIG_DIR / "vehicle.json")


def load_profiles(path=None):
    """Profils + définition des workers. Voir configs/profiles.json."""
    return _load_json(Path(path) if path else CONFIG_DIR / "profiles.json")

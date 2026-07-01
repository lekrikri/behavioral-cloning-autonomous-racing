"""Canal de réglage à chaud entre le serveur debug (écrivain) et le sous-process de
conduite (lecteur). Un seul fichier JSON ; le lecteur est mtime-gated → la boucle de
contrôle ne paie qu'un `os.stat` par poll et ne re-parse que si le fichier a changé.

Le sous-process d'inférence détient le VESC ; le serveur ne peut pas muter son état en
mémoire → ce fichier est le seul canal. Écriture atomique (tmp + os.replace) pour qu'un
poll ne lise jamais un JSON tronqué.
"""

import json
import os
from pathlib import Path


def _live_path() -> Path:
    root = Path(__file__).resolve()
    while not (root / "src" / "__init__.py").exists() and root != root.parent:
        root = root.parent
    return root / "configs" / "control_live.json"


def write_live(params: dict) -> None:
    p = _live_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(params, f)
    os.replace(tmp, p)   # atomique : le lecteur voit l'ancien ou le nouveau, jamais un mix


class LiveParamsReader:
    """Poll mtime-gated. `poll()` retourne le dict seulement si le fichier a changé."""

    def __init__(self):
        self.path = _live_path()
        self._mtime = None

    def poll(self):
        try:
            m = self.path.stat().st_mtime_ns   # ns : deux réglages rapprochés ne collisionnent pas
        except OSError:
            return None
        if m == self._mtime:
            return None
        self._mtime = m
        try:
            with open(self.path) as f:
                return json.load(f)
        except (ValueError, OSError):
            return None

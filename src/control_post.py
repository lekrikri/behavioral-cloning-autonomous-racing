"""Post-traitement du contrôle — SOURCE UNIQUE (simulateur + voiture réelle).

Regroupe ce qui était dupliqué entre `inference` (sim) et `inference_realcar` (réel) :
- `SmoothingFilter` : lissage adaptatif + deadzone steering.
- `apply_steer_offset` : correction du biais droite sur virages doux.
- `accel_from_geometry` : accélération anticipée par géométrie + rayon frontal.

`accel_from_geometry` est paramétré (`corner_damp`, `accel_floor`) car sim et réel
divergeaient : le réel applique un amortissement de virage et un plancher plus haut.
La divergence est PRÉSERVÉE telle quelle ici (pas de changement de comportement),
mais rendue explicite — à trancher en équipe (voir le commentaire ci-dessous).
"""

from typing import Optional

import numpy as np


class SmoothingFilter:
    """Filtre adaptatif : alpha élevé en virage (réactif), bas en ligne droite (stable).
    Deadzone sur le steering pour supprimer le micro-zigzag.
    """

    def __init__(self, alpha: float = 0.57, alpha_max: float = 0.92, deadzone: float = 0.06):
        self.alpha_base = alpha
        self.alpha_max = alpha_max
        self.deadzone = deadzone
        self._smoothed: Optional[np.ndarray] = None

    def reset(self):
        self._smoothed = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)

        if self._smoothed is None:
            self._smoothed = raw.copy()
        else:
            delta = abs(raw[0] - self._smoothed[0])
            alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * min(delta, 1.0)
            self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed

        result = self._smoothed.copy()
        if abs(result[0]) < self.deadzone:
            result[0] = 0.0
        return result

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.update(raw)


def apply_steer_offset(steer: float) -> float:
    """Offset -0.02 sur les virages doux (0.05 < |steer| < 0.35) : corrige le biais droite.
    Ne clippe pas : l'appelant clippe à [-1, 1]."""
    if 0.05 < abs(steer) < 0.35:
        steer = steer - 0.02 * np.sign(steer)
    return steer


def accel_from_geometry(
    steering: float,
    front_raw: float,
    *,
    corner_damp: bool = False,
    accel_floor: float = 0.35,
) -> float:
    """Accélération anticipée : géométrie (|steering|) + rayon frontal brut [0,1].

    front_raw >= 0.65 (ligne droite / mur loin) -> pas de cap frontal.
    front_raw <  0.65 (virage approche)         -> cap progressif [0.45, 0.91].

    Divergence sim/réel préservée via paramètres :
    - sim  (`inference`)         : corner_damp=False, accel_floor=0.35
    - réel (`inference_realcar`) : corner_damp=True,  accel_floor=0.50
    """
    geo_base = max(0.35, 1.0 - 1.2 * abs(steering))
    front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
    accel = min(geo_base, front_cap)
    if corner_damp:
        accel = accel * (1.0 - 0.5 * abs(steering))
    return float(np.clip(accel, accel_floor, 0.95))

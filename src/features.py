"""Features dérivées — SOURCE UNIQUE du contrat d'entrée du modèle.

`derive_features` calcule [asymmetry, front_ray, min_ray] à partir des rayons.
C'est la définition de l'entrée du réseau : toute divergence entre dataset (train),
simulateur et voiture réelle casse SILENCIEUSEMENT la parité d'entraînement.
Centralisé ici pour qu'il n'existe qu'une seule vérité. Voir aussi `control_post`.
"""

import numpy as np


def derive_features(rays):
    """Dérive [asymmetry, front_ray, min_ray] depuis un vecteur de rayons.

    Args:
        rays: array (..., n). 1D (un pas d'inférence) ou 2D batché (N, n).
              Z-scoré à l'inférence ; selon le dataset à l'entraînement.

    Returns:
        array (..., 3) float32 : asymmetry, front_ray, min_ray.
        - asymmetry = (somme_droite - somme_gauche) / (somme_totale + 1e-8)
        - front_ray = moyenne des 2 rayons centraux
        - min_ray   = rayon minimal
    """
    rays = np.asarray(rays)
    n = rays.shape[-1]
    half = n // 2
    left = rays[..., :half].sum(axis=-1, keepdims=True)
    right = rays[..., half:].sum(axis=-1, keepdims=True)
    asymmetry = (right - left) / (left + right + 1e-8)
    front_ray = rays[..., half - 1 : half + 1].mean(axis=-1, keepdims=True)
    min_ray = rays.min(axis=-1, keepdims=True)
    return np.concatenate([asymmetry, front_ray, min_ray], axis=-1).astype(np.float32)

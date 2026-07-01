"""
reactive_controller.py — Contrôleur réactif polaire (rays → [acceleration, steering])

Cerveau « algorithme classique » interchangeable avec le modèle IA : consomme le
MÊME vecteur de raycasts que le réseau (rays normalisés [0,1], 1=libre — cf.
src/mask/polar_rays.py `PolarRays.normalized`) et sort le même vecteur d'action
[acceleration, steering]. Aucune I/O, aucune dépendance matérielle → testable hors-ligne.

Principe : les rays libres tirent le cap vers l'espace ouvert (steer-toward-gap).
Toute la trigonométrie est précalculée à l'init ; le hot path n'est que des
produits scalaires vectorisés sur n_rays floats (contrainte Jetson Nano).
"""

import time
from dataclasses import dataclass

import numpy as np


def _clamp(x, lo, hi):
    """Clamp scalaire sans numpy — évite le dispatch ufunc de np.clip dans le hot path.
    (On n'importe pas le _clamp de vesc_interface : ça tirerait pyserial et casserait
    la propriété numpy-only/testable-hors-ligne de ce module.)"""
    return lo if x < lo else hi if x > hi else x


@dataclass
class ReactiveConfig:
    n_rays: int = 20                # doit suivre PolarRays.n_rays du profil
    forward_sigma: float = 6.0      # largeur (indices) de la pondération avant : rays centraux priorisés
    central_frac: float = 0.25      # fraction centrale des rays servant à la clairance/vitesse

    kp: float = 0.9
    kd: float = 0.15
    steer_max: float = 0.85
    steer_deadzone: float = 0.03
    steer_rate_max: float = 0.35    # variation steering max par frame (anti-claquement servo)
    heading_ema: float = 0.4        # lissage temporel du cap ; court pour ne pas retarder (6 fps)

    # accel = fraction de throttle [0,1] passée à vesc.send (mise à l'échelle par le
    # duty_max du profil sur le VESC) — PAS un duty absolu. 1.0 = throttle max autorisé.
    v_max: float = 1.0
    v_min: float = 0.5              # plancher (comme accel_floor du modèle) pour vaincre la friction
    turn_slowdown: float = 0.7      # part de vitesse retirée à |cap|=1

    nominal_dt: float = 1.0 / 6.0   # période frame par défaut si dt non fourni


class ReactiveController:
    # Params dont un changement impose de reconstruire les tables précalculées.
    _STRUCTURAL = ("n_rays", "forward_sigma", "central_frac")

    def __init__(self, cfg: ReactiveConfig = None):
        self.cfg = cfg or ReactiveConfig()
        self._build()
        self.reset()

    def _build(self):
        """Précalculs constants (angles fixes → jamais recalculés dans le hot path)."""
        n = self.cfg.n_rays
        self._off = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
        self._off_norm = self._off / max(self._off.max(), 1.0)   # cap ramené dans [-1, 1]
        self._fwd_w = np.exp(-0.5 * (self._off / self.cfg.forward_sigma) ** 2).astype(np.float32)
        half = max(1, int(round(n * self.cfg.central_frac / 2)))
        c = n // 2
        self._central = slice(max(0, c - half), min(n, c + half))

    def set_params(self, params: dict):
        """Applique des params réglés à chaud. Ne reconstruit les tables que si un
        paramètre structurel change (rare, mouvement de slider), jamais à chaque frame."""
        rebuild = False
        for k, v in params.items():
            if not hasattr(self.cfg, k):
                continue
            old = getattr(self.cfg, k)
            try:
                new = bool(v) if isinstance(old, bool) else type(old)(v)
            except (TypeError, ValueError):
                continue                        # valeur live corrompue : on l'ignore, pas de crash
            if new != old:
                setattr(self.cfg, k, new)
                if k in self._STRUCTURAL:
                    rebuild = True
        if rebuild:
            self._build()

    def reset(self):
        self._heading_s = 0.0
        self._prev_heading_s = 0.0
        self._prev_steer = 0.0
        self._last_t = None

    def __call__(self, rays, dt: float = None):
        c = self.cfg
        rays = np.asarray(rays, dtype=np.float32)

        # Cap = moyenne des offsets d'indice pondérée par (avant × clairance).
        # Un ray libre (d→1) proche du centre pèse le plus → on vise l'espace ouvert.
        wd = self._fwd_w * rays
        denom = float(wd.sum())
        heading = float(wd @ self._off_norm) / denom if denom > 1e-6 else 0.0

        # Lissage temporel court + dérivée pour amortir (repère : boucle fermée par l'environnement)
        if dt is None:
            now = time.monotonic()
            dt = (now - self._last_t) if self._last_t is not None else c.nominal_dt
            self._last_t = now
        dt = max(dt, 1e-3)

        self._prev_heading_s = self._heading_s
        self._heading_s = c.heading_ema * self._heading_s + (1.0 - c.heading_ema) * heading
        d_term = (self._heading_s - self._prev_heading_s) / dt

        steer = c.kp * self._heading_s + c.kd * d_term
        steer = _clamp(steer, -c.steer_max, c.steer_max)
        steer = _clamp(steer, self._prev_steer - c.steer_rate_max,
                              self._prev_steer + c.steer_rate_max)
        if abs(steer) < c.steer_deadzone:
            steer = 0.0
        self._prev_steer = steer

        clearance = float(rays[self._central].mean())
        accel = c.v_min + (c.v_max - c.v_min) * clearance * (1.0 - c.turn_slowdown * abs(self._heading_s))
        accel = _clamp(accel, 0.0, c.v_max)

        return accel, steer


# ── Test hors-ligne : motifs de rays synthétiques (aucun matériel requis) ──────
if __name__ == "__main__":
    ctrl = ReactiveController()
    n = ctrl.cfg.n_rays

    def show(name, rays):
        ctrl.reset()
        a, s = ctrl(np.asarray(rays, dtype=np.float32), dt=1 / 6)
        print(f"{name:24s} accel={a:+.3f}  steer={s:+.3f}")

    straight   = np.ones(n)                                   # tout libre
    wall_right = np.concatenate([np.ones(n // 2), np.linspace(0.6, 0.0, n - n // 2)])
    wall_left  = wall_right[::-1].copy()
    corridor   = np.concatenate([[0.2, 0.4], np.ones(n - 4), [0.4, 0.2]])
    blocked    = np.zeros(n)                                  # obstacle partout

    print(f"ReactiveController | {n} rays")
    show("straight (libre)", straight)
    show("mur a droite", wall_right)
    show("mur a gauche", wall_left)
    show("couloir centre", corridor)
    show("bloque partout", blocked)

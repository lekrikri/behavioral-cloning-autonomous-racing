"""
client.py — Wrapper UnityEnvironment pour le simulateur Robocar.

Usage:
    from src.client import RobocarEnv
    env = RobocarEnv("config.json")
    env.reset()
    obs = env.get_observations()
    env.send_actions(steering=0.0, acceleration=1.0)
    env.step()
    env.close()
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    MLAGENTS_AVAILABLE = True
except ImportError:
    MLAGENTS_AVAILABLE = False
    print("[WARNING] mlagents-envs non installé. Installer avec: pip install mlagents-envs==0.28.0")


SIMULATOR_PATH = str(Path(__file__).parent.parent / "BuildLinux" / "RacingSimulator.x86_64")
DEFAULT_PORT = 5005  # 5004 est bloqué par Windows (port RTP) — utiliser 5005


@dataclass
class Observation:
    """Observation retournée par le simulateur à chaque step."""
    rays: np.ndarray       # distances normalisées [0,1], shape (n_rays,)
    speed: float           # vitesse normalisée [0,1]
    agent_id: int = 0

    @property
    def as_vector(self) -> np.ndarray:
        """Concatène rays + speed en un seul vecteur 1D."""
        return np.concatenate([self.rays, [self.speed]], dtype=np.float32)

    def __repr__(self):
        rays_str = np.array2string(self.rays, precision=3, suppress_small=True)
        return f"Observation(rays={rays_str}, speed={self.speed:.3f})"


class RobocarEnv:
    """
    Wrapper haut niveau autour de UnityEnvironment.

    Paramètres
    ----------
    config_path : str
        Chemin vers le JSON de configuration des agents.
    simulator_path : str, optional
        Chemin vers le binaire Unity. Si None, le simulateur doit déjà tourner.
    port : int
        Port gRPC (défaut: 5004).
    no_graphics : bool
        Lance le simulateur sans rendu graphique (plus rapide pour l'entraînement).
    """

    def __init__(
        self,
        config_path: str = "config.json",
        simulator_path: Optional[str] = None,
        port: int = DEFAULT_PORT,
        no_graphics: bool = False,
    ):
        if not MLAGENTS_AVAILABLE:
            raise ImportError("mlagents-envs requis: pip install mlagents-envs==0.28.0")

        self.config_path = str(Path(config_path).resolve())
        self.port = port

        # Charger la config pour connaître nbRay
        with open(self.config_path) as f:
            cfg = json.load(f)
        self._n_rays = cfg["agents"][0]["nbRay"]
        self._n_agents = len(cfg["agents"])
        # Distance max des rayons (en unités Unity). Dérivée à la première obs.
        self._ray_max_dist: float = cfg["agents"][0].get("rayMaxDistance", 300.0)

        additional_args = ["--config-path", self.config_path]

        self._env = UnityEnvironment(
            file_name=simulator_path,
            base_port=port,
            additional_args=additional_args,
            no_graphics=no_graphics,
            seed=42,
        )

        self._behavior_name: Optional[str] = None
        self._ready = False
        self._last_decision_steps = None

    def reset(self) -> list:
        """Réinitialise l'environnement et retourne les observations initiales."""
        self._env.reset()
        self._behavior_name = list(self._env.behavior_specs.keys())[0]
        self._ready = True
        self._last_decision_steps, _ = self._env.get_steps(self._behavior_name)
        return self._parse_observations(self._last_decision_steps)

    def step(self) -> list:
        """Avance d'un step et retourne les nouvelles observations."""
        self._env.step()
        self._last_decision_steps, _ = self._env.get_steps(self._behavior_name)
        return self._parse_observations(self._last_decision_steps)

    def send_actions(self, steering: float, acceleration: float, agent_idx: int = 0):
        """
        Envoie les actions continues à l'agent.

        steering : float in [-1, 1]  (gauche/droite)
        acceleration : float in [-1, 1]  (gaz/frein)
        """
        steering = float(np.clip(steering, -1.0, 1.0))
        acceleration = float(np.clip(acceleration, -1.0, 1.0))

        decision_steps = self._last_decision_steps
        n = len(decision_steps) if decision_steps is not None else 0
        if n == 0:
            return

        actions = np.tile([[acceleration, steering]], (n, 1)).astype(np.float32)
        self._env.set_actions(self._behavior_name, ActionTuple(continuous=actions))

    def get_observations(self) -> list:
        """Retourne les observations actuelles sans avancer la simulation."""
        if self._last_decision_steps is None:
            return []
        return self._parse_observations(self._last_decision_steps)

    def close(self):
        """Ferme proprement la connexion avec Unity."""
        if self._env:
            self._env.close()
            self._ready = False

    def _parse_observations(self, decision_steps) -> list:
        """Parse les observations brutes ML-Agents en objets Observation."""
        if not self._ready or decision_steps is None:
            return []

        observations = []

        for i, agent_id in enumerate(decision_steps.agent_id):
            raw_obs = decision_steps.obs[0][i]
            n = self._n_rays
            # obs layout: [dist_0..dist_n-1, tag_flags...] — distances en unités Unity
            # Normalise les distances en [0, 1]
            rays = (raw_obs[:n] / self._ray_max_dist).clip(0.0, 1.0).astype(np.float32)
            # La vitesse n'est pas dans le vecteur d'obs de ce simulateur
            speed = 0.0
            observations.append(Observation(rays=rays, speed=speed, agent_id=int(agent_id)))

        return observations

    @property
    def n_rays(self) -> int:
        return self._n_rays

    @property
    def obs_size(self) -> int:
        return self._n_rays + 1  # rays + speed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"RobocarEnv(n_rays={self._n_rays}, port={self.port}, ready={self._ready})"


def main():
    parser = argparse.ArgumentParser(description="Test connexion simulateur Robocar")
    parser.add_argument("--config", default="config.json", help="Chemin config agents")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--test-only", action="store_true", help="Smoke test sans simulateur")
    parser.add_argument("--launch", action="store_true", help="Lance le simulateur automatiquement")
    parser.add_argument("--no-graphics", action="store_true", help="Mode headless (batchmode)")
    args = parser.parse_args()

    if args.test_only:
        print("[TEST] Smoke test du module client...")
        if not MLAGENTS_AVAILABLE:
            print("[FAIL] mlagents-envs non installé")
            sys.exit(1)
        print("[OK] mlagents-envs importé")
        print("[OK] Smoke test réussi. Utiliser --launch pour lancer le simulateur automatiquement.")
        return

    sim_path = SIMULATOR_PATH if args.launch else None
    no_graphics = args.no_graphics or args.launch  # headless si on lance nous-mêmes

    if args.launch:
        print(f"[INFO] Lancement du simulateur: {SIMULATOR_PATH}")
        print(f"[INFO] Mode: {'headless' if no_graphics else 'graphique'}")
    else:
        print(f"[INFO] Connexion au simulateur sur port {args.port}...")
        print("[INFO] Démarrez le simulateur RacingSimulator.x86_64 d'abord!")

    with RobocarEnv(config_path=args.config, port=args.port,
                    simulator_path=sim_path, no_graphics=no_graphics) as env:
        print(f"[OK] Connecté: {env}")
        observations = env.reset()
        print(f"[OK] Reset OK — {len(observations)} agent(s)")
        if observations:
            print(f"[OK] Observation: {observations[0]}")

        for step in range(5):
            env.send_actions(steering=0.0, acceleration=0.5)
            obs = env.step()
            if obs:
                print(f"  Step {step+1}: speed={obs[0].speed:.3f}")

    print("[OK] Environnement fermé.")


if __name__ == "__main__":
    main()

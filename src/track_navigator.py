# -*- coding: utf-8 -*-
"""
track_navigator.py — Phase 2 : pilotage prédictif par carte IMU.

Principe (conseillé par les IA) :
- PAS de remplacement du PD vision — seulement un BIAS additif
- Carte en frames (pas secondes) → robuste à la vitesse variable
- Resync par signature gyro des N derniers événements
- confidence score → dégradation propre vers vision pure

steer_final = steer_pd + steering_bias   (jamais steer_final = steering_bias seul)
"""
import json
import math
import time

# Fenêtre de resync : compare les N derniers événements gyro réels vs carte
RESYNC_WINDOW = 3
CONFIDENCE_INC  = 0.08   # gain si signature gyro correspond
CONFIDENCE_DEC  = 0.12   # perte si divergence
CONFIDENCE_MIN  = 0.35   # en-dessous → fallback vision pure
CONFIDENCE_MAX  = 1.0

# Seuil pour détecter un virage dans le flux gyro temps réel (gyro brut, pas filtré)
# Calibré sur petite piste duty 8% : pics bruts 0.30-0.50 rad/s pendant virages
GYRO_TURN_THRESH = 0.25  # rad/s

# Anticipation : combien de frames avant l'événement on commence à agir
ANTICIPATION_FRAMES_BRAKE = 8    # commencer à ralentir
ANTICIPATION_FRAMES_STEER = 4    # commencer à braquer


class TrackNavigator(object):

    def __init__(self):
        self.events = []
        self.metadata = {}
        self.loaded = False
        self.confidence = 0.80   # démarre confiant
        self.fallback = False

        # Position courante dans la carte
        self._idx = 0            # index événement courant
        self._frame_in_seg = 0   # frames parcourues dans ce segment

        # Historique gyro réel (pour resync par signature)
        self._gyro_history = []  # liste de "L"/"R"/"S" sur les derniers virages

        # Anti-spam log
        self._last_log_idx = -1

    # ------------------------------------------------------------------
    def load(self, path="track_map.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.events = data.get("events", [])
            self.metadata = data.get("metadata", {})
            self.loaded = bool(self.events)
            if self.loaded:
                print("[NAV] Carte chargée : {0} événements — {1}".format(
                    len(self.events), path))
                self._print_events()
            else:
                print("[NAV] Carte vide ou invalide : {0}".format(path))
        except Exception as e:
            print("[NAV] Impossible de charger la carte : {0}".format(e))
            self.loaded = False
        return self.loaded

    def _print_events(self):
        for e in self.events:
            print("  #{id:02d} {type:8s} {dir} {frames:3d}fr peak={gyro_peak:+.2f} yaw={yaw_deg:+.1f}°".format(**e))

    # ------------------------------------------------------------------
    def update(self, gyro_z, n_blobs, err):
        """
        Appelé à chaque frame vision (~6 Hz).
        gyro_z  : rad/s courant (depuis IMU)
        n_blobs : 0/1/2 (nombre de lignes vues par la caméra)
        err     : erreur latérale PD (px), None si perdue

        Retourne un dict :
          steering_bias  : float à AJOUTER au steer_pd ([-0.50, +0.50])
          throttle_scale : float à MULTIPLIER au throttle ([0.60, 1.0])
          action         : str description pour les logs
          confidence     : float score [0,1]
          frames_to_next : int frames estimées avant prochain événement
        """
        if not self.loaded or self.fallback:
            return self._neutral("FALLBACK")

        self._frame_in_seg += 1

        # --- RESYNC PAR SIGNATURE GYRO ---
        self._update_gyro_history(gyro_z)
        self._try_resync(gyro_z)

        # Si confiance trop basse → fallback
        if self.confidence < CONFIDENCE_MIN:
            self.fallback = True
            print("[NAV] Confiance trop faible ({:.2f}) → FALLBACK vision pure".format(self.confidence))
            return self._neutral("LOW_CONFIDENCE")

        # --- AVANCE DANS LA CARTE ---
        curr = self.events[self._idx]
        frames_left = curr["frames"] - self._frame_in_seg

        if frames_left <= 0:
            self._advance()
            curr = self.events[self._idx]
            frames_left = curr["frames"]

        # Prochain événement
        next_idx = (self._idx + 1) % len(self.events)
        next_evt = self.events[next_idx]
        frames_to_next = frames_left  # frames restantes dans le segment courant

        # --- GÉNÉRATION DES ACTIONS ---
        if curr["type"] == "straight":
            if next_evt["type"] == "turn" and frames_to_next <= ANTICIPATION_FRAMES_BRAKE:
                # Pré-freinage avant virage
                bias = 0.0
                if frames_to_next <= ANTICIPATION_FRAMES_STEER:
                    # Pré-braquage léger dans la bonne direction
                    bias = 0.12 if next_evt["dir"] == "R" else -0.12
                self._log_once("PRE_TURN_{0} dans {1}fr".format(next_evt["dir"], frames_to_next))
                return {
                    "steering_bias": bias,
                    "throttle_scale": 0.75,
                    "action": "pre_turn_" + next_evt["dir"],
                    "confidence": self.confidence,
                    "frames_to_next": frames_to_next,
                }
            return self._neutral("STRAIGHT")

        elif curr["type"] == "turn":
            dir_sign = 1.0 if curr["dir"] == "R" else -1.0

            if n_blobs == 0:
                # Caméra aveugle → la carte prend plus de poids
                bias = dir_sign * 0.55
                scale = 0.60
                self._log_once("TURN_{0} AVEUGLE bias={1:.2f}".format(curr["dir"], bias))
            elif n_blobs == 1:
                # Une ligne → aide légère
                bias = dir_sign * 0.30
                scale = 0.65
            else:
                # Deux lignes visibles → biais léger d'aide
                bias = dir_sign * 0.18
                scale = 0.70

            return {
                "steering_bias": bias,
                "throttle_scale": scale,
                "action": "turn_" + curr["dir"],
                "confidence": self.confidence,
                "frames_to_next": frames_to_next,
            }

        return self._neutral("UNKNOWN")

    # ------------------------------------------------------------------
    def _advance(self):
        old = self._idx
        self._idx = (self._idx + 1) % len(self.events)
        self._frame_in_seg = 0
        evt = self.events[self._idx]
        print("[NAV] -> #{0} {1}({2}) {3}fr".format(self._idx, evt["type"], evt["dir"], evt["frames"]))

    # ------------------------------------------------------------------
    def _update_gyro_history(self, gyro_z):
        """Ajoute L/R/S à l'historique des virages détectés en réel."""
        if abs(gyro_z) > GYRO_TURN_THRESH:
            token = "L" if gyro_z > 0 else "R"
            if not self._gyro_history or self._gyro_history[-1] != token:
                self._gyro_history.append(token)
                if len(self._gyro_history) > 10:
                    self._gyro_history.pop(0)

    def _try_resync(self, gyro_z):
        """
        Compare les derniers virages détectés en réel avec la carte.
        Si la signature correspond → confidence++, sinon confidence--.
        Si pic gyro fort arrive AVANT l'événement attendu → recaler l'index.
        """
        if len(self.events) == 0:
            return

        curr = self.events[self._idx]

        # Si on est sur une ligne droite et qu'un virage fort arrive trop tôt
        if curr["type"] == "straight" and abs(gyro_z) > GYRO_TURN_THRESH:
            # Cherche si le prochain événement carte est un virage dans ce sens
            next_idx = (self._idx + 1) % len(self.events)
            next_evt = self.events[next_idx]
            expected_dir = "L" if gyro_z > 0 else "R"
            if next_evt["type"] == "turn" and next_evt["dir"] == expected_dir:
                # Le virage arrive — on avance si on a déjà fait > 60% du segment
                if self._frame_in_seg > int(curr["frames"] * 0.60):
                    self._advance()
                    self.confidence = min(CONFIDENCE_MAX, self.confidence + CONFIDENCE_INC)
                    return

        # Vérification de cohérence : on est sur un turn, est-ce que gyro confirme ?
        if curr["type"] == "turn":
            expected_sign = 1.0 if curr["dir"] == "L" else -1.0
            gyro_matches = (gyro_z * expected_sign) > 0.20
            if gyro_matches:
                self.confidence = min(CONFIDENCE_MAX, self.confidence + CONFIDENCE_INC * 0.5)
            elif abs(gyro_z) < 0.10 and self._frame_in_seg < 3:
                # On est censé être en virage mais gyro est plat → légère pénalité
                self.confidence = max(0.0, self.confidence - CONFIDENCE_DEC * 0.3)

    # ------------------------------------------------------------------
    def notify_vision_stable(self, n_blobs, err):
        """
        Appelé quand la vision est stable (b=2, err≈0 pendant N frames).
        Réduit le drift accumulé en resettant le compteur de segment.
        """
        if n_blobs == 2 and err is not None and abs(err) < 8:
            curr = self.events[self._idx] if self.events else None
            if curr and curr["type"] == "straight":
                self.confidence = min(CONFIDENCE_MAX, self.confidence + CONFIDENCE_INC)
                # Reset drift : on recale le compteur sur la médiane du segment
                if self._frame_in_seg > int(curr["frames"] * 1.10):
                    self._frame_in_seg = int(curr["frames"] * 0.80)

    def recover_from_fallback(self):
        """Tente de sortir du fallback si la vision est bonne."""
        if self.fallback and self.confidence > CONFIDENCE_MIN + 0.1:
            self.fallback = False
            print("[NAV] Sortie du fallback — reprise carte")

    # ------------------------------------------------------------------
    def _neutral(self, reason=""):
        return {
            "steering_bias": 0.0,
            "throttle_scale": 1.0,
            "action": reason,
            "confidence": self.confidence,
            "frames_to_next": 999,
        }

    def _log_once(self, msg):
        if self._idx != self._last_log_idx:
            print("[NAV] " + msg)
            self._last_log_idx = self._idx

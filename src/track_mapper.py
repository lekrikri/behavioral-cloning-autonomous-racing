# -*- coding: utf-8 -*-
"""
track_mapper.py — Phase 1 : enregistrement d'un tour de reconnaissance.

Principe : la piste est représentée comme une séquence d'événements invariants
(straight / turn) identifiés par leur signature gyro_z.
PAS de position XY absolue — trop de drift sans encodeurs.

Carte exportée : track_map.json (séquence de frames + gyro_peak + yaw_deg)
"""
import json
import time
import math

# Seuils hystérésis pour détecter les virages (rad/s)
GYRO_ENTER = 0.50   # au-dessus = virage détecté
GYRO_EXIT  = 0.25   # en-dessous = retour ligne droite
MIN_SEGMENT_FRAMES = 4   # ignore les micro-pics < 4 frames


class TrackMapper(object):

    def __init__(self):
        self.events = []
        self.is_mapping = False
        self.finish_requested = False

        # État courant du segment
        self._state = "straight"    # "straight" | "turn"
        self._seg_dir = "N"
        self._seg_frames = 0
        self._seg_gyro_peak = 0.0
        self._seg_yaw_accum = 0.0   # rad accumulés sur le segment

        # Accumulation globale
        self._total_frames = 0
        self._global_yaw = 0.0      # yaw total depuis le départ (rad)

        # Timestamp de début
        self._start_time = None

    # ------------------------------------------------------------------
    def start(self):
        self.events = []
        self.is_mapping = True
        self.finish_requested = False
        self._state = "straight"
        self._seg_dir = "N"
        self._seg_frames = 0
        self._seg_gyro_peak = 0.0
        self._seg_yaw_accum = 0.0
        self._total_frames = 0
        self._global_yaw = 0.0
        self._start_time = time.time()
        print("[MAPPER] Cartographie démarrée")

    # ------------------------------------------------------------------
    def process_frame(self, gyro_z, dt=None):
        """
        Appelé à chaque frame vision (~6 Hz).
        gyro_z  : rad/s (lacet — rotation autour de Z)
        dt      : durée de la frame en secondes (si None → 1/6)
        Retourne le type du segment courant ("straight" | "turn_L" | "turn_R")
        """
        if not self.is_mapping:
            return "idle"

        if dt is None:
            dt = 1.0 / 6.0

        self._total_frames += 1
        self._seg_frames += 1

        # Accumulation yaw
        dyaw = gyro_z * dt
        self._global_yaw += dyaw
        self._seg_yaw_accum += dyaw

        # Pic gyro sur ce segment
        if abs(gyro_z) > abs(self._seg_gyro_peak):
            self._seg_gyro_peak = gyro_z

        # --- MACHINE À ÉTATS ---
        abs_gz = abs(gyro_z)

        if self._state == "straight":
            if abs_gz > GYRO_ENTER:
                self._close_segment()
                self._state = "turn"
                self._seg_dir = "L" if gyro_z > 0 else "R"

        elif self._state == "turn":
            if abs_gz < GYRO_EXIT:
                self._close_segment()
                self._state = "straight"
                self._seg_dir = "N"

        # Fin demandée via HTTP /finish_map
        if self.finish_requested:
            self._close_segment()
            self.is_mapping = False
            self.finish_requested = False

        return self._state if self._state == "straight" else "turn_" + self._seg_dir

    # ------------------------------------------------------------------
    def _close_segment(self):
        if self._seg_frames < MIN_SEGMENT_FRAMES:
            # Trop court → on reset sans enregistrer
            self._reset_seg()
            return

        event = {
            "id": len(self.events),
            "type": self._state,
            "dir": self._seg_dir,
            "frames": self._seg_frames,
            "gyro_peak": round(self._seg_gyro_peak, 3),
            "yaw_deg": round(math.degrees(self._seg_yaw_accum), 1),
            "global_frame_start": self._total_frames - self._seg_frames,
        }
        self.events.append(event)
        print("[MAPPER] Segment #{id} {type}({dir}) {frames}fr peak={gyro_peak:.2f} yaw={yaw_deg:.1f}°".format(**event))
        self._reset_seg()

    def _reset_seg(self):
        self._seg_frames = 0
        self._seg_gyro_peak = 0.0
        self._seg_yaw_accum = 0.0

    # ------------------------------------------------------------------
    def save_map(self, path="track_map.json"):
        """Ferme le segment courant et exporte la carte."""
        if self._seg_frames >= MIN_SEGMENT_FRAMES:
            self._close_segment()

        total_yaw_deg = round(math.degrees(self._global_yaw), 1)
        loop_closed = abs(abs(total_yaw_deg) - 360.0) < 30.0  # tolérance ±30°

        data = {
            "version": 1,
            "metadata": {
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
                "total_frames": self._total_frames,
                "total_events": len(self.events),
                "total_yaw_deg": total_yaw_deg,
                "loop_closed": loop_closed,
                "elapsed_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
            },
            "events": self.events,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print("[MAPPER] Carte sauvegardée -> {0}  ({1} événements, yaw_total={2}°, loop={3})".format(
            path, len(self.events), total_yaw_deg, loop_closed))
        return data

    # ------------------------------------------------------------------
    def summary(self):
        """Affiche un résumé lisible de la carte enregistrée."""
        turns = [e for e in self.events if e["type"] == "turn"]
        straights = [e for e in self.events if e["type"] == "straight"]
        print("[MAPPER] {0} segments : {1} lignes droites, {2} virages".format(
            len(self.events), len(straights), len(turns)))
        for e in self.events:
            print("  #{id:02d} {type:8s} {dir} {frames:3d}fr peak={gyro_peak:+.2f} yaw={yaw_deg:+6.1f}°".format(**e))

# -*- coding: utf-8 -*-
"""
track_mapper.py — Cartographie de piste par dead-reckoning IMU + throttle.

Principe :
  - Heading : intégration gyro_z (rad/s) → angle absolu depuis départ
  - Vitesse  : throttle_duty × V_PER_DUTY (m/s) — calibration empirique
  - Position : intégration dx = v·cos(θ)·dt, dy = v·sin(θ)·dt
  - Résultat : nuage de waypoints XY + segments topologiques (straight/turn)

Limitations :
  - Pas d'encodeurs → drift vitesse (~5-10% sur un tour)
  - Gyro BMI270 RAW → drift heading (~2-5° sur un tour)
  - Suffisant pour cartographie qualitative d'un circuit ~10-20m

Export :
  - track_map.json : waypoints + métadonnées + segments
  - track_map.svg  : rendu vectoriel de la piste (auto-généré)
"""
import json
import math
import time

# ── Calibration ──────────────────────────────────────────────────────────────
V_PER_DUTY     = 3.0    # m/s pour duty=1.0 (à ajuster par mesure)
DT_DEFAULT     = 1.0 / 13.0  # fallback si dt non fourni (13fps)

# ── Détection virages (topologie) ─────────────────────────────────────────────
GYRO_ENTER     = 0.20   # rad/s → entrée virage
GYRO_EXIT      = 0.09   # rad/s → sortie virage
MIN_SEG_FRAMES = 10     # frames minimum pour enregistrer un segment

# ── SVG ───────────────────────────────────────────────────────────────────────
SVG_W          = 800
SVG_H          = 600
SVG_PAD        = 60     # marge en pixels


class TrackMapper:

    def __init__(self, v_per_duty=V_PER_DUTY):
        self.v_per_duty = v_per_duty
        self._reset()

    def _reset(self):
        self.waypoints   = []   # [{x, y, heading_deg, t, duty, gyro_z}]
        self.segments    = []   # [{type, dir, start_idx, end_idx, yaw_deg}]

        self.x           = 0.0
        self.y           = 0.0
        self.heading     = 0.0  # rad depuis départ (0 = axe X)

        self._state      = "straight"
        self._seg_start  = 0
        self._seg_yaw    = 0.0
        self._seg_peak   = 0.0
        self._seg_dir    = "N"

        self._start_time = None
        self.is_mapping  = False
        self.finish_requested = False

    # ── API publique ────────────────────────────────────────────────────────

    def start(self):
        self._reset()
        self._start_time = time.time()
        self.is_mapping  = True
        print("[MAPPER] Cartographie démarrée — V_PER_DUTY={:.1f} m/s".format(self.v_per_duty))

    def process_frame(self, gyro_z, throttle_duty=0.0, dt=None):
        """
        Appelé à chaque frame.
        gyro_z        : rad/s (lacet, positif = gauche)
        throttle_duty : 0.0-1.0 (duty VESC)
        dt            : durée frame en secondes
        """
        if not self.is_mapping:
            return "idle"

        if dt is None:
            dt = DT_DEFAULT

        # ── Intégration heading + position ────────────────────────────────
        self.heading += gyro_z * dt
        v = max(0.0, throttle_duty) * self.v_per_duty
        self.x += v * math.cos(self.heading) * dt
        self.y -= v * math.sin(self.heading) * dt  # Y inversé (image coords)

        t_rel = time.time() - self._start_time

        self.waypoints.append({
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "h": round(math.degrees(self.heading) % 360, 1),
            "t": round(t_rel, 2),
            "d": round(throttle_duty, 3),
            "gz": round(gyro_z, 3),
        })

        # ── Machine à états topologique ────────────────────────────────────
        abs_gz = abs(gyro_z)
        dyaw   = gyro_z * dt
        self._seg_yaw  += dyaw
        self._seg_peak  = max(self._seg_peak, abs_gz) if gyro_z >= 0 else min(self._seg_peak, gyro_z)

        if self._state == "straight":
            if abs_gz > GYRO_ENTER:
                self._close_segment(len(self.waypoints) - 1)
                self._state   = "turn"
                self._seg_dir = "L" if gyro_z > 0 else "R"
        else:
            if abs_gz < GYRO_EXIT:
                self._close_segment(len(self.waypoints) - 1)
                self._state   = "straight"
                self._seg_dir = "N"

        if self.finish_requested:
            self._close_segment(len(self.waypoints) - 1)
            self.is_mapping       = False
            self.finish_requested = False

        return self._state if self._state == "straight" else "turn_" + self._seg_dir

    def save(self, json_path="track_map.json", svg_path="track_map.svg"):
        """Ferme le dernier segment, sauvegarde JSON + SVG."""
        self._close_segment(len(self.waypoints))
        self.is_mapping = False

        data = {
            "version": 2,
            "meta": {
                "created_at":   time.strftime("%Y-%m-%d %H:%M"),
                "v_per_duty":   self.v_per_duty,
                "total_frames": len(self.waypoints),
                "elapsed_s":    round(time.time() - self._start_time, 1) if self._start_time else 0,
                "total_yaw_deg": round(math.degrees(self.heading), 1),
                "n_segments":   len(self.segments),
                "n_turns":      sum(1 for s in self.segments if s["type"] == "turn"),
            },
            "waypoints": self.waypoints,
            "segments":  self.segments,
        }

        with open(json_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        svg = self.render_svg()
        with open(svg_path, "w") as f:
            f.write(svg)

        print("[MAPPER] Sauvegardé → {0} ({1} wpts, {2} segs)  SVG → {3}".format(
            json_path, len(self.waypoints), len(self.segments), svg_path))
        return data

    # ── Legacy : compatibilité avec l'ancien code ────────────────────────
    def save_map(self, path="track_map.json"):
        svg_path = path.replace(".json", ".svg")
        return self.save(path, svg_path)

    def summary(self):
        turns = [s for s in self.segments if s["type"] == "turn"]
        straights = [s for s in self.segments if s["type"] == "straight"]
        print("[MAPPER] {0} segments : {1} droites, {2} virages".format(
            len(self.segments), len(straights), len(turns)))
        for s in self.segments:
            print("  #{id:02d} {type:8s} {dir} {frames:3d}fr yaw={yaw_deg:+6.1f}°".format(**s))

    # ── Rendu SVG ────────────────────────────────────────────────────────

    def render_svg(self):
        """Génère un SVG représentant le tracé XY de la piste."""
        if not self.waypoints:
            return "<svg/>"

        xs = [w["x"] for w in self.waypoints]
        ys = [w["y"] for w in self.waypoints]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dx = x_max - x_min or 1.0
        dy = y_max - y_min or 1.0

        usable_w = SVG_W - 2 * SVG_PAD
        usable_h = SVG_H - 2 * SVG_PAD
        scale    = min(usable_w / dx, usable_h / dy)

        def px(x, y):
            sx = SVG_PAD + (x - x_min) * scale
            sy = SVG_PAD + (y - y_min) * scale
            return sx, sy

        # Polyline principale
        pts = " ".join("{:.1f},{:.1f}".format(*px(w["x"], w["y"])) for w in self.waypoints)

        # Marqueurs virages
        turn_marks = []
        for s in self.segments:
            if s["type"] != "turn":
                continue
            mid = (s["start_idx"] + s["end_idx"]) // 2
            if mid < len(self.waypoints):
                w = self.waypoints[mid]
                cx, cy = px(w["x"], w["y"])
                color = "#FF4444" if s["dir"] == "R" else "#4444FF"
                lbl   = "{}  {:.0f}°".format(s["dir"], s["yaw_deg"])
                turn_marks.append(
                    '<circle cx="{:.1f}" cy="{:.1f}" r="8" fill="{}" opacity="0.8"/>'.format(cx, cy, color) +
                    '<text x="{:.1f}" y="{:.1f}" font-size="11" fill="{}" font-family="monospace">{}</text>'.format(
                        cx + 10, cy + 4, color, lbl)
                )

        # Flèche départ
        if len(self.waypoints) >= 2:
            w0 = self.waypoints[0]
            sx0, sy0 = px(w0["x"], w0["y"])
            start_mark = '<circle cx="{:.1f}" cy="{:.1f}" r="10" fill="#00CC44" opacity="0.9"/>'.format(sx0, sy0)
            start_mark += '<text x="{:.1f}" y="{:.1f}" font-size="12" fill="#00CC44" font-family="monospace">START</text>'.format(sx0 + 12, sy0 + 4)
        else:
            start_mark = ""

        # Flèche fin
        if self.waypoints:
            wn = self.waypoints[-1]
            ex, ey = px(wn["x"], wn["y"])
            end_mark = '<circle cx="{:.1f}" cy="{:.1f}" r="8" fill="#FF8800" opacity="0.9"/>'.format(ex, ey)
        else:
            end_mark = ""

        # Légende
        meta_lines = [
            "Tour : {:.1f}s".format(self.waypoints[-1]["t"] if self.waypoints else 0),
            "Virages : {}".format(sum(1 for s in self.segments if s["type"] == "turn")),
            "Yaw total : {:.0f}°".format(math.degrees(self.heading)),
        ]
        legend = "".join(
            '<text x="10" y="{}" font-size="11" fill="#AAAAAA" font-family="monospace">{}</text>'.format(
                SVG_H - 40 + i * 14, line)
            for i, line in enumerate(meta_lines)
        )

        return """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#111">
  <polyline points="{pts}" stroke="#00FF88" stroke-width="2" fill="none" opacity="0.85"/>
  {turns}
  {start}
  {end}
  {legend}
  <text x="10" y="20" font-size="13" fill="#FFFFFF" font-family="monospace">TRACK MAP — {date}</text>
</svg>""".format(
            w=SVG_W, h=SVG_H, pts=pts,
            turns="\n  ".join(turn_marks),
            start=start_mark, end=end_mark, legend=legend,
            date=time.strftime("%Y-%m-%d %H:%M"),
        )

    # ── Interne ──────────────────────────────────────────────────────────

    def _close_segment(self, end_idx):
        frames = end_idx - self._seg_start
        if frames < MIN_SEG_FRAMES:
            self._seg_start = end_idx
            self._seg_yaw   = 0.0
            self._seg_peak  = 0.0
            return
        seg = {
            "id":        len(self.segments),
            "type":      self._state,
            "dir":       self._seg_dir,
            "start_idx": self._seg_start,
            "end_idx":   end_idx,
            "frames":    frames,
            "yaw_deg":   round(math.degrees(self._seg_yaw), 1),
            "gyro_peak": round(self._seg_peak, 3),
        }
        self.segments.append(seg)
        print("[MAPPER] Seg #{id} {type}({dir}) {frames}fr yaw={yaw_deg:+.1f}°".format(**seg))
        self._seg_start = end_idx
        self._seg_yaw   = 0.0
        self._seg_peak  = 0.0

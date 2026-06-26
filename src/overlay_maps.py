#!/usr/bin/env python3
"""
overlay_maps.py — Superpose la piste réelle (photo SVG) + carto dead-reckoning.

Usage:
  python3 src/overlay_maps.py track_map.json track_reference.svg [output.svg]

Résultat :
  - Bleu/violet : contours de la vraie piste extraits de la photo
  - Vert         : tracé de la carto dead-reckoning (avec virages colorés)
  - Orange/rouge : marqueurs R/L détectés par le gyro

Les deux maps sont normalisées pour remplir le même canvas 900×600,
ce qui donne une comparaison visuelle de la précision du dead-reckoning.
"""

import sys, json, math, re, os

SVG_W = 900
SVG_H = 600
PAD   = 45   # marge en pixels


def load_carto(json_path):
    with open(json_path) as f:
        return json.load(f)


def carto_polyline(wpts, w=SVG_W, h=SVG_H, pad=PAD):
    """Waypoints JSON → polyline string scalé pour remplir le canvas."""
    if not wpts:
        return ""
    xs = [p["x"] for p in wpts]
    ys = [p["y"] for p in wpts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    dx = x_max - x_min or 1.0
    dy = y_max - y_min or 1.0
    scale = min((w - 2 * pad) / dx, (h - 2 * pad) / dy)

    def px(x, y):
        return pad + (x - x_min) * scale, pad + (y - y_min) * scale

    pts = " ".join("{:.1f},{:.1f}".format(*px(p["x"], p["y"])) for p in wpts)
    return pts, px, x_min, y_min, scale


def carto_turn_marks(segments, wpts, px_fn):
    """Marqueurs de virages pour la carto."""
    marks = []
    for s in segments:
        if s.get("type") != "turn":
            continue
        mid = (s["start_idx"] + s["end_idx"]) // 2
        if mid >= len(wpts):
            continue
        w = wpts[mid]
        cx, cy = px_fn(w["x"], w["y"])
        color = "#FF5555" if s["dir"] == "R" else "#5588FF"
        lbl   = "{} {:.0f}°".format(s["dir"], s["yaw_deg"])
        marks.append(
            '<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="{}" opacity="0.7"/>'.format(cx, cy, color) +
            '<text x="{:.1f}" y="{:.1f}" font-size="10" fill="{}" font-family="monospace">{}</text>'.format(
                cx + 7, cy + 3, color, lbl)
        )
    return "\n  ".join(marks)


def extract_ref_paths(ref_svg_path):
    """Extrait les <path> du SVG de référence (photo) et les colorie en bleu."""
    if not os.path.exists(ref_svg_path):
        return ""
    with open(ref_svg_path) as f:
        content = f.read()
    paths = re.findall(r'<path[^>]*/>', content)
    # Reteindre en bleu translucide
    out = []
    for p in paths:
        p = re.sub(r'fill="[^"]*"', 'fill="#6677FF"', p)
        p = re.sub(r'stroke="[^"]*"', 'stroke="#6677FF"', p)
        # Ajouter opacité si absent
        if 'opacity' not in p:
            p = p.replace('/>', ' opacity="0.40"/>')
        out.append(p)
    return "\n  ".join(out)


def main():
    carto_json = sys.argv[1] if len(sys.argv) > 1 else "track_map.json"
    ref_svg    = sys.argv[2] if len(sys.argv) > 2 else "track_reference.svg"
    out_svg    = sys.argv[3] if len(sys.argv) > 3 else "track_overlay.svg"

    if not os.path.exists(carto_json):
        print("ERREUR: {} introuvable".format(carto_json))
        sys.exit(1)

    data     = load_carto(carto_json)
    wpts     = data.get("waypoints", [])
    segments = data.get("segments", [])
    meta     = data.get("meta", {})

    if not wpts:
        print("ERREUR: aucun waypoint dans la carto")
        sys.exit(1)

    result = carto_polyline(wpts)
    pts_str, px_fn, _, _, _ = result

    # START/END markers
    x0, y0 = px_fn(wpts[0]["x"], wpts[0]["y"])
    xn, yn = px_fn(wpts[-1]["x"], wpts[-1]["y"])

    turn_marks  = carto_turn_marks(segments, wpts, px_fn)
    ref_layer   = extract_ref_paths(ref_svg)

    has_ref = bool(ref_layer.strip())
    legend_ref = "● Référence photo (bleu)" if has_ref else "● Référence photo : introuvable"

    legend = "Tour: {:.0f}s | {:.0f}m | {} virages | Yaw: {:.0f}°".format(
        meta.get("elapsed_s", 0),
        sum(
            math.hypot(wpts[i]["x"] - wpts[i-1]["x"], wpts[i]["y"] - wpts[i-1]["y"])
            for i in range(1, len(wpts))
        ),
        meta.get("n_turns", 0),
        meta.get("total_yaw_deg", 0),
    )

    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background:#111">
  <!-- Référence réelle (photo) — bleu -->
  {ref}
  <!-- Carto dead-reckoning — vert -->
  <polyline points="{pts}" stroke="#00FF88" stroke-width="2.5" fill="none" opacity="0.85"/>
  <!-- START -->
  <circle cx="{x0:.1f}" cy="{y0:.1f}" r="9" fill="#00CC44" opacity="0.9"/>
  <text x="{sx:.0f}" y="{sy:.0f}" font-size="11" fill="#00CC44" font-family="monospace">START</text>
  <!-- END -->
  <circle cx="{xn:.1f}" cy="{yn:.1f}" r="7" fill="none" stroke="#FF8800" stroke-width="2"/>
  <!-- Virages -->
  {turns}
  <!-- Légende -->
  <text x="10" y="18" font-size="11" fill="#00CC44" font-family="monospace">■ Carto DR</text>
  <text x="110" y="18" font-size="11" fill="#6677FF" font-family="monospace">■ {ref_legend}</text>
  <text x="10" y="{ty}" font-size="10" fill="#555" font-family="monospace">{legend}</text>
</svg>""".format(
        w=SVG_W, h=SVG_H,
        ref=ref_layer,
        pts=pts_str,
        x0=x0, y0=y0,
        sx=x0 + 11, sy=y0 + 4,
        xn=xn, yn=yn,
        turns=turn_marks,
        ref_legend=legend_ref,
        ty=SVG_H - 6,
        legend=legend,
    )

    with open(out_svg, "w") as f:
        f.write(svg)
    print("[OK] Overlay → {}".format(out_svg))
    print("     Carto    : {} waypoints".format(len(wpts)))
    print("     Référence: {}".format(ref_svg if has_ref else "ABSENTE"))


if __name__ == "__main__":
    main()

"""Mask page capture loop: hub frames -> white_line_mask + PolarRays -> overlay -> JPEG.

The heavy work (mask + rays + JPEG encode) runs here in one thread; the HTTP handler only
serves the latest JPEG. Tuning params live in a thread-safe MaskState the sliders mutate.
"""

import threading
import time

import cv2
import numpy as np

from src.mask.white_line import white_line_mask
from src.mask.perception_config import PerceptionProfile, list_profiles, load_profile


# Filter params exposed as sliders, with (min, max, step) for the UI.
SLIDERS = [
    ("v_min_floor",    0,   255, 5),
    ("s_max",          0,   120, 5),
    ("tophat_k",       0,   41,  2),
    ("tophat_thresh",  0,   60,  2),
    ("morph_k",        0,   11,  1),
    ("min_area",       0,   400, 10),
    ("min_elongation", 1.0, 12.0, 0.5),
    ("depth_tol",      0.0, 0.9, 0.05),
    ("clahe_clip",     0.0, 6.0, 0.5),
    ("bottom_ignore_frac", 0.0, 0.5, 0.02),
]

# Filtres activables/désactivables : (clé, libellé, param neutralisé, valeur "off").
# `depth` est spécial (on coupe l'entrée depth), d'où off_value None.
FILTER_TOGGLES = [
    ("clahe",   "CLAHE",              "clahe_clip",     0.0),
    ("tophat",  "top-hat",           "tophat_k",       0),
    ("morpho",  "morpho",            "morph_k",        0),
    ("depth",   "depth (surf. vert.)", None,           None),
    ("noise",   "bruit (aire)",      "min_area",       0),
    ("rectilin", "rectilinéarité",   "min_elongation", 1.0),
]


def _derive_enabled(params):
    """État des toggles déduit des valeurs : un filtre est ON sauf si son param est à la
    valeur 'off' (permet de refléter un profil où un filtre est déjà neutralisé)."""
    en = {}
    for key, _, param, off in FILTER_TOGGLES:
        en[key] = True if param is None else (params.get(param) != off)
    return en


class MaskState:
    """Thread-safe tuning params + latest rendered JPEG."""

    def __init__(self, profile: PerceptionProfile):
        self.lock = threading.Lock()
        self.profile = profile
        self.params = dict(profile.mask_kwargs())
        self.enabled = _derive_enabled(self.params)
        self.show_mask = True
        self.show_rays = True
        self.jpeg = None
        self.fps = 0.0
        self.frames = 0
        self.running = True

    def load_profile_into(self, name):
        """Charge un profil à chaud : reset des valeurs + toggles + géométrie (via self.profile,
        relue par la boucle de capture qui reconstruit géom/faisceau)."""
        try:
            prof = load_profile(name)
        except (FileNotFoundError, OSError):
            return "profil introuvable: %s" % name
        with self.lock:
            self.profile = prof
            self.params = dict(prof.mask_kwargs())
            self.enabled = _derive_enabled(self.params)
        return "profil chargé: %s" % name

    def save_to_profile(self):
        """Écrit les réglages EFFECTIFS (toggles appliqués) dans le JSON du profil courant.
        Un filtre décoché est sauvé à sa valeur neutre → rechargé, il réapparaît décoché."""
        from src.mask.perception_config import profiles_dir
        eff, _, _, _ = self.snapshot()
        with self.lock:
            prof = self.profile
            for k, v in eff.items():
                if hasattr(prof, k):
                    setattr(prof, k, v)
            name = prof.name
        prof.save(profiles_dir() / (name + ".json"))
        return "profil '%s' sauvegardé" % name

    def snapshot(self):
        """Params EFFECTIFS (toggles appliqués) pour la boucle de capture.
        Retourne (params, use_depth, show_mask, show_rays)."""
        with self.lock:
            p = dict(self.params)
            en = dict(self.enabled)
            sm, sr = self.show_mask, self.show_rays
        for key, _, param, off in FILTER_TOGGLES:
            if param is not None and not en[key]:
                p[param] = off
        return p, en["depth"], sm, sr

    def ui_state(self):
        """État brut pour l'UI : valeurs sliders conservées + flags on/off + toggles vue."""
        with self.lock:
            d = dict(self.params)
            d["show_mask"] = int(self.show_mask)
            d["show_rays"] = int(self.show_rays)
            for key, en in self.enabled.items():
                d["en_" + key] = int(en)
            d["_profile"] = self.profile.name
        d["_profiles"] = list_profiles() or [self.profile.name]
        return d

    def set_param(self, name, value):
        with self.lock:
            if name in ("show_mask", "show_rays"):
                setattr(self, name, bool(int(float(value))))
                return "%s=%d" % (name, getattr(self, name))
            if name.startswith("en_"):
                key = name[3:]
                if key not in self.enabled:
                    return "unknown:%s" % name
                self.enabled[key] = bool(int(float(value)))
                return "%s=%d" % (name, int(self.enabled[key]))
            if name not in self.params:
                return "unknown:%s" % name
            cur = self.params[name]
            self.params[name] = type(cur)(float(value)) if not isinstance(cur, bool) else bool(float(value))
            return "%s=%s" % (name, self.params[name])


def render(bgr, mask, dist, angles, geom, max_range, snap, fps=0.0, frames=0):
    """Overlay: mask (green) + polar fan projected from the car centre. Returns BGR image."""
    _, show_mask, show_rays = snap
    vis = bgr.copy()
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.4, 0)

    if show_rays and dist is not None:
        origin = geom.ground_to_pixel(0.0, 0.0)           # car centre on the ground
        o = (int(origin[0]), min(int(origin[1]), bgr.shape[0] - 1)) if origin else \
            (bgr.shape[1] // 2, bgr.shape[0] - 1)
        for d, a in zip(dist, angles):
            X, Y = d * np.cos(a), d * np.sin(a)
            end = geom.ground_to_pixel(float(X), float(Y))
            if end is None:
                continue
            free = float(np.clip(d / max_range, 0.0, 1.0))
            col = (0, int(255 * free), int(255 * (1.0 - free)))   # green=free, red=near
            cv2.line(vis, o, (int(end[0]), int(end[1])), col, 1)

    white_px = int(mask.sum() / 255)
    txt = "%dx%d | blanc=%dpx | %.0f fps | %d frames" % (
        vis.shape[1], vis.shape[0], white_px, fps, frames)
    cv2.putText(vis, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if show_mask:
        vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    return vis


def _frames():
    """Yield (bgr, depth_or_None) from the hub. depth is best-effort (latest)."""
    from src.cam.hub import FrameClient, SHM_COLOR, SHM_DEPTH
    c_color = FrameClient(stream=SHM_COLOR)
    c_depth = FrameClient(stream=SHM_DEPTH)
    while True:
        try:
            bgr = c_color.getCvFrame()
        except (ConnectionError, OSError):
            time.sleep(0.5)
            c_color = FrameClient(stream=SHM_COLOR)
            c_depth = FrameClient(stream=SHM_DEPTH)
            continue
        depth = None
        try:
            depth = c_depth.latest()
        except (ConnectionError, OSError):
            depth = None
        yield bgr, depth


def capture_loop(state: MaskState):
    geom = None
    polar = None
    cur_prof = None
    t_prev = time.time()
    fps = 0.0
    frames = 0
    for bgr, depth in _frames():
        if not state.running:
            break
        frames += 1
        prof = state.profile   # profil courant (peut changer à chaud via l'UI)
        if (geom is None or cur_prof is not prof
                or geom.H != bgr.shape[0] or geom.W != bgr.shape[1]):
            cur_prof = prof
            geom = _geom_for(prof, bgr.shape)      # géométrie à la taille réelle de la frame
            polar = prof.polar_rays(geom=geom)

        params, use_depth, show_mask, show_rays = state.snapshot()
        mask = white_line_mask(bgr, depth_mm=(depth if use_depth else None),
                               geom=geom, **params)
        dist = angles = None
        if show_rays:
            dist, angles = polar(mask)

        now = time.time()
        dt = now - t_prev; t_prev = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        vis = render(bgr, mask, dist, angles, geom, cur_prof.max_range_m,
                     (params, show_mask, show_rays), fps=fps, frames=frames)
        ok, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state.lock:
                state.jpeg = enc.tobytes()
                state.fps = fps
                state.frames = frames


def _geom_for(profile, shape):
    """CameraGround at the real frame size (hub may publish a size != the profile's)."""
    from src.mask.camera_ground import CameraGround
    h, w = shape[0], shape[1]
    return CameraGround(w, h, profile.fov_deg, profile.cam_height_m, profile.cam_pitch_deg)


def _selftest():
    prof = PerceptionProfile()
    st = MaskState(prof)
    assert st.set_param("v_min_floor", 190).endswith("190")
    assert isinstance(st.params["tophat_k"], int)

    # Toggles : désactiver un filtre neutralise son param dans snapshot mais garde le slider.
    st.set_param("tophat_k", 21)
    st.set_param("en_tophat", 0)
    eff, use_depth, _, _ = st.snapshot()
    assert eff["tophat_k"] == 0 and st.params["tophat_k"] == 21, "toggle off must not lose the slider"
    st.set_param("en_depth", 0)
    _, use_depth, _, _ = st.snapshot()
    assert use_depth is False
    assert st.ui_state()["en_tophat"] == 0 and st.ui_state()["en_depth"] == 0
    st.set_param("en_tophat", 1)
    assert st.snapshot()[0]["tophat_k"] == 21, "toggle on must restore the slider value"
    geom = prof.camera_ground()
    polar = prof.polar_rays(geom=geom)
    bgr = np.zeros((prof.cam_height, prof.cam_width, 3), np.uint8)
    cv2.line(bgr, (200, 130), (210, 250), (255, 255, 255), 14)
    mask = white_line_mask(bgr, **st.params)
    dist, ang = polar(mask)
    vis = render(bgr, mask, dist, ang, geom, prof.max_range_m, (st.params, True, True))
    assert vis.shape[0] == prof.cam_height and vis.shape[1] == 2 * prof.cam_width
    print("capture self-test OK (state + render, out shape %s)" % (vis.shape,))


if __name__ == "__main__":
    _selftest()

"""Perception profiles — a swappable preset for the whole perception stack.

A profile bundles everything needed to turn camera frames into polar rays for the model:
camera mounting geometry, the polar fan, the white-line mask filters, and the drive
command. The debug interface loads a profile on its home page; Play runs the drive
pipeline with it. Two profiles are planned: "classic" (this algorithmic mask, to be
calibrated) and "ai" (a trained segmentation model), selected by `kind`.

Two orthogonal axes: `kind` (classic|ai) selects the PERCEPTION stack (how frames become
rays); `brain` (model|reactive) selects the DECISION module (how rays become commands).

Flat dataclass -> trivial JSON round trip (same style as RobocarConfig). Profiles live
as JSON files under configs/profiles/.

Geometry note: cam_height_m / cam_pitch_deg are PHYSICAL mounting values — they must be
measured/calibrated on the car; the defaults are placeholders, not ground truth.
"""

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class PerceptionProfile:
    name: str = "classic"
    kind: str = "classic"              # "classic" | "ai"

    # --- Camera mounting geometry (CALIBRATE on the car) ---
    cam_width: int = 512
    cam_height: int = 256
    fov_deg: float = 68.79             # OAK-D Lite colour CAM_A (measured)
    cam_height_m: float = 0.15
    cam_pitch_deg: float = 20.0        # downward tilt

    # --- Polar fan ---
    n_rays: int = 20                   # must match the model (v18 = 20)
    fan_deg: float = 0.0               # 0 => use the camera horizontal FOV
    max_range_m: float = 4.0
    row_start_frac: float = 0.0        # ROI vertical band (fraction of height)
    row_end_frac: float = 1.0

    # --- White-line mask filters (cf. white_line_mask) ---
    clahe_clip: float = 2.0
    s_max: int = 60
    v_min_floor: int = 180
    otsu_cap: int = 220
    tophat_k: int = 15
    tophat_thresh: int = 12
    morph_k: int = 3
    min_area: int = 40
    min_elongation: float = 4.0
    depth_tol: float = 0.35
    bottom_ignore_frac: float = 0.0   # bande basse ignorée (carrosserie avant)

    # --- Drive ---
    model: str = "models/v18/best_jetson.onnx"
    duty_max: float = 0.20

    # --- Brain (module de décision, swappable) ---
    brain: str = "model"               # "model" (ONNX) | "reactive" (algo polaire)
    # Params du contrôleur réactif (préfixe ctrl_ ; miroir de ReactiveConfig, réglables live).
    ctrl_kp: float = 0.9
    ctrl_kd: float = 0.15
    ctrl_forward_sigma: float = 6.0
    ctrl_v_max: float = 1.0
    ctrl_v_min: float = 0.5
    ctrl_turn_slowdown: float = 0.7
    ctrl_steer_max: float = 0.85
    ctrl_steer_deadzone: float = 0.03
    ctrl_steer_rate_max: float = 0.35
    ctrl_heading_ema: float = 0.4

    # ── Builders ────────────────────────────────────────────────────────────────
    def camera_ground(self):
        from src.mask.camera_ground import CameraGround
        return CameraGround(self.cam_width, self.cam_height, self.fov_deg,
                            self.cam_height_m, self.cam_pitch_deg)

    def polar_rays(self, geom=None):
        from src.mask.polar_rays import PolarRays
        geom = geom or self.camera_ground()
        return PolarRays(geom, n_rays=self.n_rays,
                         fan_deg=(self.fan_deg or None), max_range_m=self.max_range_m,
                         row_band=(self.row_start_frac, self.row_end_frac))

    def mask_kwargs(self) -> dict:
        """kwargs for white_line_mask (excludes bgr/depth/geom)."""
        return dict(clahe_clip=self.clahe_clip, s_max=self.s_max,
                    v_min_floor=self.v_min_floor, otsu_cap=self.otsu_cap,
                    tophat_k=self.tophat_k, tophat_thresh=self.tophat_thresh,
                    morph_k=self.morph_k, min_area=self.min_area,
                    min_elongation=self.min_elongation, depth_tol=self.depth_tol,
                    bottom_ignore_frac=self.bottom_ignore_frac)

    def control_kwargs(self) -> dict:
        """kwargs pour ReactiveConfig (retire le préfixe ctrl_, ajoute n_rays)."""
        out = {"n_rays": self.n_rays}
        for f in fields(self):
            if f.name.startswith("ctrl_"):
                out[f.name[len("ctrl_"):]] = getattr(self, f.name)
        return out

    # ── Persistence ─────────────────────────────────────────────────────────────
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path) -> "PerceptionProfile":
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def profiles_dir() -> Path:
    root = Path(__file__).resolve()
    while not (root / "src" / "__init__.py").exists() and root != root.parent:
        root = root.parent
    return root / "configs" / "profiles"


def list_profiles() -> list:
    d = profiles_dir()
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def load_profile(name: str) -> PerceptionProfile:
    return PerceptionProfile.load(profiles_dir() / f"{name}.json")


def _active_file() -> Path:
    """Fichier device-local mémorisant le dernier profil choisi (configs/active_profile.txt)."""
    return profiles_dir().parent / "active_profile.txt"


def read_active_profile():
    """Nom du profil actif persisté, ou None si absent."""
    try:
        name = _active_file().read_text().strip()
        return name or None
    except (FileNotFoundError, OSError):
        return None


def write_active_profile(name: str) -> None:
    f = _active_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(name).strip())


def resolve_profile(explicit=None):
    """Profil à charger : --profile explicite > profil actif persisté > 'classic'.
    Retourne (PerceptionProfile, name). Tombe sur les défauts si le fichier manque."""
    name = explicit or read_active_profile() or "classic"
    try:
        return load_profile(name), name
    except (FileNotFoundError, OSError):
        return PerceptionProfile(), name


def _selftest():
    import tempfile
    p = PerceptionProfile(name="t", tophat_k=21, cam_pitch_deg=18.0)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp = f.name
    p.save(tmp)
    q = PerceptionProfile.load(tmp)
    assert q.tophat_k == 21 and abs(q.cam_pitch_deg - 18.0) < 1e-9
    assert q.mask_kwargs()["tophat_k"] == 21
    assert set(q.mask_kwargs()) and "bgr" not in q.mask_kwargs()
    print("PerceptionProfile self-test OK (round trip + mask kwargs)")


if __name__ == "__main__":
    _selftest()

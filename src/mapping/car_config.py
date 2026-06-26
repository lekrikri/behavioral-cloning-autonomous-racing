"""Central car configuration, loaded from car.env (KEY=VALUE at the repo root).

One place to set up the whole car: camera mounting, VESC control, perception, hub,
telemetry. Dependency-free parser (the Jetson runs Python 3.6, no python-dotenv).
Unknown/missing keys fall back to the defaults below, so the code runs without the
file. Access values as attributes in lower case, e.g. cfg.cam_height_m, cfg.vesc_max_current.
"""

import os

_DEFAULTS = {
    # camera mounting
    "CAM_HEIGHT_M": 0.31, "CAM_PITCH_DEG": 13.5, "CAM_HFOV_DEG": 68.8,
    "CAM_WIDTH": 512, "CAM_HEIGHT_PX": 256,
    # vesc
    "VESC_PORT": "/dev/ttyACM0", "VESC_MAX_CURRENT": 15.0, "VESC_MAX_DUTY": 0.20,
    "VESC_SERVO_CENTER": 0.53, "VESC_SERVO_RANGE": 0.40, "VESC_INVERT_MOTOR": False,
    "VESC_THROTTLE_MODE": "current", "VESC_K_ERPM_TO_MS": 0.000219, "VESC_ERPM_SIGN": -1.0,
    # imu / pose
    "GYRO_YAW_AXIS": 2, "GYRO_YAW_SIGN": 1.0, "GYRO_BIAS_SECONDS": 2.0,
    # perception
    "N_RAYS": 20, "RAY_MAX_M": 4.0, "ROW_BAND_LO": 0.0, "ROW_BAND_HI": 1.0,
    # hub / telemetry
    "HUB_HOST": "127.0.0.1", "HUB_PORT": 8077, "HUB_IMU_PORT": 8078, "TELEMETRY_PORT": 5602,
    # occupancy grid
    "GRID_RES_M": 0.05, "GRID_SIZE_M": 20.0,
}


def _find_env():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        p = os.path.join(d, "car.env")
        if os.path.exists(p):
            return p
        d = os.path.dirname(d)
    return None


def _coerce(default, raw):
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


class CarConfig(object):
    def __init__(self, values):
        self._values = values
        for k, v in values.items():
            setattr(self, k.lower(), v)

    def __repr__(self):
        return "CarConfig(%s)" % ", ".join("%s=%r" % kv for kv in sorted(self._values.items()))


def load(path=None):
    values = dict(_DEFAULTS)
    path = path or _find_env()
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw = line.split("=", 1)
                key = key.strip()
                raw = raw.split("#", 1)[0]      # allow trailing inline comments
                if key in _DEFAULTS:
                    values[key] = _coerce(_DEFAULTS[key], raw)
    return CarConfig(values)


if __name__ == "__main__":
    print("car.env:", _find_env())
    print(load())

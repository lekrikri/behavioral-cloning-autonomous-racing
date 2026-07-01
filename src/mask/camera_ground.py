"""Camera <-> ground-plane geometry (inverse perspective mapping).

Flat-track assumption: the camera sits at a fixed height, pitched down by a fixed
angle; every track pixel below the horizon lies on the ground plane. This maps an
image pixel to a metric ground position (X forward, Y left) and back.

Single source of the projection, shared by:
  - white_line_mask()  -> rejects pixels whose depth is closer than the ground
                          (off-ground = vertical surfaces: walls, chair legs)
  - PolarRays          -> projects mask pixels to the ground for the angular fan

World frame: X forward, Y left, Z up. Camera at height h, pitched down by phi.
Pure numpy; unit-testable via the pixel<->ground round trip in __main__.
"""

import math

import numpy as np


class CameraGround:
    def __init__(self, img_width: int, img_height: int, hfov_deg: float,
                 height_m: float, pitch_deg: float):
        self.W = int(img_width)
        self.H = int(img_height)
        self.fx = self.W / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        self.fy = self.fx                       # square pixels
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.h = float(height_m)
        self.phi = math.radians(pitch_deg)
        self._c = math.cos(self.phi)
        self._s = math.sin(self.phi)
        self._ground_z_mm = self._precompute_ground_depth()

    def _precompute_ground_depth(self) -> np.ndarray:
        """Optical-axis depth (Z_cam, mm) of the ground plane for each image row.

        For a flat ground and no roll this depends only on the row, so it is computed
        once. Rows at/above the horizon get +inf (never treated as off-ground).
        """
        rows = np.arange(self.H, dtype=np.float64)
        yc = (rows - self.cy) / self.fy
        denom = yc * self._c + self._s          # ray points to ground iff > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.h / denom
            X = t * (self._c - yc * self._s)    # forward distance on the ground
            z_mm = (self._c * X + self._s * self.h) * 1000.0
        z_mm[denom <= 1e-6] = np.inf
        return z_mm.astype(np.float32)

    @property
    def ground_depth_mm(self) -> np.ndarray:
        """(H,) expected ground depth per row, for the vertical-surface filter."""
        return self._ground_z_mm

    def pixels_to_ground(self, us, vs):
        """Vectorized pixel (u, v) -> ground (X forward, Y left) in metres.

        Pixels at/above the horizon return NaN in both X and Y.
        """
        xc = (np.asarray(us, dtype=np.float64) - self.cx) / self.fx
        yc = (np.asarray(vs, dtype=np.float64) - self.cy) / self.fy
        denom = yc * self._c + self._s
        valid = denom > 1e-6
        safe = np.where(valid, denom, 1.0)
        t = np.where(valid, self.h / safe, np.nan)
        X = t * (self._c - yc * self._s)
        Y = t * (-xc)
        return X, Y

    def pixel_to_ground(self, u: float, v: float):
        """Scalar pixel -> (X, Y) metres, or None if at/above the horizon."""
        X, Y = self.pixels_to_ground(np.array([u]), np.array([v]))
        return None if np.isnan(X[0]) else (float(X[0]), float(Y[0]))

    def ground_to_pixel(self, X: float, Y: float):
        """Ground (X forward, Y left) -> pixel (u, v), or None if behind the camera.

        Inverse of pixel_to_ground; used by the debug overlay and the self-test.
        """
        Z = self._c * X + self._s * self.h              # optical-axis depth
        if Z <= 1e-6:
            return None
        u = self.cx + self.fx * (-Y) / Z
        v = self.cy + self.fy * (-self._s * X + self._c * self.h) / Z
        return u, v


def _selftest():
    g = CameraGround(512, 256, hfov_deg=68.79, height_m=0.15, pitch_deg=20.0)
    for X, Y in [(0.5, 0.0), (1.0, 0.3), (0.8, -0.25)]:
        u, v = g.ground_to_pixel(X, Y)
        gx, gy = g.pixel_to_ground(u, v)
        assert abs(gx - X) < 1e-3 and abs(gy - Y) < 1e-3, \
            "round trip failed: (%.3f,%.3f) -> (%.3f,%.3f)" % (X, Y, gx, gy)
    # Ground depth is positive and grows toward the top of the frame (farther rows).
    # Rows above the horizon (if any are in frame) are +inf.
    zc = g.ground_depth_mm
    finite = zc[np.isfinite(zc)]
    assert finite.min() > 0, "ground depth must be positive"
    assert zc[20] > zc[240], "farther (upper) rows must have larger ground depth"
    print("CameraGround self-test OK (round trip + ground depth)")


if __name__ == "__main__":
    _selftest()

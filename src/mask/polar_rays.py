"""Polar raycasts from a binary white-line mask.

A regular angular fan emanating from a single point (the car centre), matching the
simulator's lidar model. Each white mask pixel is projected onto the flat ground plane
(CameraGround) and reduced, per angular sector, to the nearest line distance in metres.

Ray convention (matches the simulator / model):
  distance = max_range  -> free direction (no line seen)
  distance -> 0         -> line very close in that direction
  normalized()          -> [0, 1], 1 = free
"""

import math

import numpy as np

from src.mask.camera_ground import CameraGround


class PolarRays:
    def __init__(self, geom: CameraGround, n_rays: int = 20, fan_deg: float = None,
                 max_range_m: float = 4.0, row_band: tuple = (0.0, 1.0)):
        self.geom = geom
        self.n_rays = int(n_rays)
        self.max_range = float(max_range_m)
        # Default fan = the camera's own horizontal opening; can be narrowed to match
        # the simulator's lidar fan.
        self.fan = math.radians(fan_deg) if fan_deg else 2.0 * math.atan(
            (geom.W / 2.0) / geom.fx)
        # Sector centres: leftmost (+fan/2) to rightmost (-fan/2), same order as the sim.
        self.angles = np.linspace(self.fan / 2.0, -self.fan / 2.0, self.n_rays)
        self.row_start = int(geom.H * row_band[0])
        self.row_end = int(geom.H * row_band[1])

    def __call__(self, mask: np.ndarray):
        """mask: uint8 (H, W), 0/255. Returns (distances_m[n_rays], angles_rad[n_rays])."""
        sub = mask[self.row_start:self.row_end, :]
        vs, us = np.where(sub > 0)
        out = np.full(self.n_rays, self.max_range, dtype=np.float32)
        if len(us) == 0:
            return out, self.angles
        vs = vs + self.row_start

        X, Y = self.geom.pixels_to_ground(us, vs)
        finite = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[finite], Y[finite]
        if len(X) == 0:
            return out, self.angles

        rng = np.hypot(X, Y)
        az = np.arctan2(Y, X)
        keep = (rng <= self.max_range) & (np.abs(az) <= self.fan / 2.0 + 1e-9)
        rng, az = rng[keep], az[keep]
        if len(rng) == 0:
            return out, self.angles

        # Bin each point into its angular sector and keep the nearest distance.
        idx = np.clip(((self.fan / 2.0 - az) / (self.fan / self.n_rays)).astype(int),
                      0, self.n_rays - 1)
        np.minimum.at(out, idx, rng.astype(np.float32))
        return out, self.angles

    def normalized(self, distances: np.ndarray) -> np.ndarray:
        """distances_m -> [0, 1] (1 = free / max range), the sim/model convention."""
        return np.clip(distances / self.max_range, 0.0, 1.0).astype(np.float32)


def _selftest():
    g = CameraGround(512, 256, hfov_deg=68.79, height_m=0.15, pitch_deg=20.0)
    pr = PolarRays(g, n_rays=20, max_range_m=4.0)

    # Render a mask with line points at known ground positions, then recover them.
    mask = np.zeros((256, 512), np.uint8)
    truth = [(0.5, 0.0), (0.7, 0.2), (0.7, -0.2), (1.0, 0.0)]
    for X, Y in truth:
        u, v = g.ground_to_pixel(X, Y)
        mask[int(round(v)), int(round(u))] = 255

    dists, angles = pr(mask)
    hit = dists[dists < pr.max_range]
    assert abs(hit.min() - 0.5) < 0.05, "nearest ray %.3f (expected ~0.5)" % hit.min()
    # closest point is dead ahead -> the hit ray angle should be near 0
    centre_idx = int(np.argmin(dists))
    assert abs(angles[centre_idx]) < math.radians(6), "closest ray not centred"
    print("PolarRays self-test OK (%d/%d rays hit, min=%.2f m)"
          % (len(hit), pr.n_rays, hit.min()))


if __name__ == "__main__":
    _selftest()

"""Polar raycasts from a white-line mask via inverse-perspective mapping (IPM).

Unlike the column-scan VisualRays (a perspective proxy for the sim lidar), this
projects each white mask pixel onto the flat ground plane using the camera geometry
(height, pitch, FOV) and returns, for each of n_rays angular sectors, the nearest
line distance IN METRES. Every ray emanates from the car center — the true polar
fan, matching the simulation's lidar model.

Frame: world X forward, Y left, Z up; camera at height h, pitched down by phi.
Pure numpy — unit-testable on the laptop via the ground<->pixel round trip in __main__.
"""

import math

import numpy as np


class PolarRays:
    def __init__(self, img_width, img_height, hfov_deg, height_m, pitch_deg,
                 n_rays=20, fan_deg=None, max_range_m=4.0, row_band=(0.0, 1.0)):
        self.W = img_width
        self.H = img_height
        self.fx = img_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        self.fy = self.fx                      # square pixels
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        self.h = float(height_m)
        self.phi = math.radians(pitch_deg)
        self._cphi = math.cos(self.phi)
        self._sphi = math.sin(self.phi)
        self.n_rays = n_rays
        self.max_range = float(max_range_m)
        self.fan = math.radians(fan_deg) if fan_deg else math.radians(hfov_deg)
        self.row_start = int(img_height * row_band[0])
        self.row_end = int(img_height * row_band[1])
        # sector centre angles, left(+) to right(-) — same convention as pose.ray_angles
        self.angles = np.linspace(self.fan / 2.0, -self.fan / 2.0, n_rays)

        # Predicted ground depth (camera optical-axis Z, mm) per image ROW. For a flat
        # ground and no roll it depends only on the row, so precompute once. inf above
        # the horizon. Used by the optional depth filter to reject off-ground pixels.
        rows = np.arange(img_height, dtype=np.float64)
        yc = (rows - self.cy) / self.fy
        denom = yc * self._cphi + self._sphi
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.h / denom
            X = t * (self._cphi - yc * self._sphi)
            z_mm = (self._cphi * X + self._sphi * self.h) * 1000.0
        z_mm[denom <= 1e-6] = np.inf
        self._ground_z_mm = z_mm.astype(np.float32)

    def ground_xy(self, u, v):
        """Pixel (u,v) -> ground (forward X, lateral Y) in metres, or None if above horizon."""
        xc = (u - self.cx) / self.fx
        yc = (v - self.cy) / self.fy
        denom = yc * self._cphi + self._sphi   # ray points down iff > 0
        if denom <= 1e-6:
            return None
        t = self.h / denom
        return t * (self._cphi - yc * self._sphi), t * (-xc)

    def filter_ground(self, mask, depth_mm, tol=0.35):
        """Drop mask pixels whose VALID depth is well closer than the ground plane
        (off-ground objects: walls, chair legs). Invalid depth (textureless asphalt)
        is kept. depth_mm must be aligned to the color frame, same size as mask."""
        thresh = self._ground_z_mm * (1.0 - tol)        # (H,)
        reject = (depth_mm > 0) & (depth_mm.astype(np.float32) < thresh[:, None])
        out = mask.copy()
        out[reject] = 0
        return out

    def __call__(self, mask, depth_mm=None, tol=0.35):
        """mask: uint8 (H,W) 0/255. Optional depth_mm (same size, aligned) filters out
        off-ground pixels. Returns (distances_m[n_rays], angles_rad[n_rays])."""
        if depth_mm is not None:
            mask = self.filter_ground(mask, depth_mm, tol)
        sub = mask[self.row_start:self.row_end, :]
        ys, xs = np.where(sub > 0)
        out = np.full(self.n_rays, self.max_range, dtype=np.float32)
        if len(xs) == 0:
            return out, self.angles
        ys = ys + self.row_start

        xc = (xs - self.cx) / self.fx
        yc = (ys - self.cy) / self.fy
        denom = yc * self._cphi + self._sphi
        keep = denom > 1e-6
        xc, yc, denom = xc[keep], yc[keep], denom[keep]
        t = self.h / denom
        X = t * (self._cphi - yc * self._sphi)     # forward
        Y = t * (-xc)                              # lateral (left +)
        rng = np.sqrt(X * X + Y * Y)
        az = np.arctan2(Y, X)

        m = (rng <= self.max_range) & (np.abs(az) <= self.fan / 2.0 + 1e-9)
        rng, az = rng[m], az[m]
        if len(rng) == 0:
            return out, self.angles

        # sector 0 = leftmost (+fan/2); idx grows toward the right
        idx = np.clip(((self.fan / 2.0 - az) / (self.fan / self.n_rays)).astype(int),
                      0, self.n_rays - 1)
        np.minimum.at(out, idx, rng.astype(np.float32))
        return out, self.angles

    def normalized(self, distances):
        """distances_m -> [0,1] (1 = free/max range), the sim/model ray convention."""
        return np.clip(distances / self.max_range, 0.0, 1.0).astype(np.float32)


def _project(pr, X, Y):
    """Ground (X fwd, Y left) -> pixel (u,v). Inverse of ground_xy, for the self-test."""
    vec = np.array([X, Y, -pr.h])
    r_x = -Y                                          # x_cam . vec
    r_y = -pr._sphi * X + pr._cphi * pr.h             # y_cam . vec
    r_z = pr._cphi * X + pr._sphi * pr.h              # z_cam . vec
    return pr.cx + pr.fx * r_x / r_z, pr.cy + pr.fy * r_y / r_z


def _selftest():
    pr = PolarRays(512, 256, hfov_deg=68.8, height_m=0.15, pitch_deg=20.0,
                   n_rays=20, max_range_m=4.0)
    # place line points at known ground positions, render a mask, recover them
    mask = np.zeros((256, 512), np.uint8)
    truth = [(0.5, 0.0), (0.7, 0.2), (0.7, -0.2), (1.0, 0.0)]
    for X, Y in truth:
        u, v = _project(pr, X, Y)
        ui, vi = int(round(u)), int(round(v))
        assert 0 <= ui < 512 and 0 <= vi < 256, "test point off-image: %.0f,%.0f" % (u, v)
        mask[vi, ui] = 255

    # round-trip ground_xy on the closest point
    u, v = _project(pr, 0.5, 0.0)
    gx, gy = pr.ground_xy(u, v)
    print("round-trip (0.5,0.0) -> (%.3f, %.3f) m" % (gx, gy))
    assert abs(gx - 0.5) < 0.02 and abs(gy) < 0.02

    dists, angles = pr(mask)
    hit = dists[dists < pr.max_range]
    print("rays hit: %d / %d | min=%.2f max=%.2f m" % (len(hit), pr.n_rays, hit.min(), hit.max()))
    # nearest detected distance should match the closest truth point (0.5 m)
    assert abs(hit.min() - 0.5) < 0.05, "closest ray = %.3f (expected ~0.5)" % hit.min()

    # depth filter: a ground pixel (depth ~= ground Z) is kept; an off-ground pixel
    # (depth much closer than the ground) is dropped.
    u, v = _project(pr, 0.5, 0.0)
    vi, ui = int(round(v)), int(round(u))
    depth = np.zeros((256, 512), np.uint16)
    depth[vi, ui] = int(pr._ground_z_mm[vi])           # exactly ground depth -> keep
    kept = pr.filter_ground(mask, depth)
    assert kept[vi, ui] == 255, "ground pixel wrongly dropped"
    depth[vi, ui] = int(pr._ground_z_mm[vi] * 0.4)     # much closer -> off-ground
    dropped = pr.filter_ground(mask, depth)
    assert dropped[vi, ui] == 0, "off-ground pixel not filtered"
    print("depth ground-filter OK (keeps ground, drops off-ground)")
    print("PolarRays self-test OK")


if __name__ == "__main__":
    _selftest()

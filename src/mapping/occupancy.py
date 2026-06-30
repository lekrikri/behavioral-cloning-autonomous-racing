"""Log-odds occupancy grid built from raycasts projected at the car's pose.

Each scan: for every ray we march FREE cells from the car up to the measured
distance, and mark the endpoint OCCUPIED when the ray actually hit something
(distance < max range). Log-odds accumulation makes repeated observations sharpen
the map and tolerates occasional bad rays.

Pure numpy — unit-testable without hardware.
"""

import math

import numpy as np


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class OccupancyGrid:
    """Square grid centred on the world origin (the lap start pose).

    Cell value is log-odds: >0 likely occupied, <0 likely free, 0 unknown.
    Row index maps to world y, column index to world x.
    """

    def __init__(self, res_m=0.05, size_m=20.0, max_range_m=3.0,
                 l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0):
        self.res = float(res_m)
        self.n = int(round(size_m / res_m))
        self.half = self.n // 2
        self.max_range = float(max_range_m)
        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max
        self.grid = np.zeros((self.n, self.n), dtype=np.float32)

    def world_to_cell(self, x, y):
        i = int(round(y / self.res)) + self.half  # row <- y
        j = int(round(x / self.res)) + self.half  # col <- x
        return i, j

    def _in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def _bump(self, i, j, delta):
        if self._in_bounds(i, j):
            self.grid[i, j] = _clamp(self.grid[i, j] + delta, self.l_min, self.l_max)

    def integrate_scan(self, pose, ray_dists_m, ray_angles_rad):
        """Update the grid from one scan.

        pose: (x, y, theta). ray_dists_m: metric distances (a value >= max_range is
        treated as "no hit", only free space is carved). ray_angles_rad: per-ray
        angle relative to the heading (same order as ray_dists_m).
        """
        x, y, theta = pose[0], pose[1], pose[2]
        for d, a in zip(ray_dists_m, ray_angles_rad):
            ang = theta + a
            ca, sa = math.cos(ang), math.sin(ang)
            hit = d < self.max_range * 0.99
            reach = d if hit else self.max_range
            n_steps = int(reach / self.res)
            # carve free space up to (but not including) the endpoint
            for s in range(n_steps):
                r = s * self.res
                self._bump(*self.world_to_cell(x + ca * r, y + sa * r), delta=self.l_free)
            if hit:
                self._bump(*self.world_to_cell(x + ca * reach, y + sa * reach), delta=self.l_occ)

    def probability(self):
        """Occupancy probability in [0, 1] (0.5 = unknown)."""
        return 1.0 / (1.0 + np.exp(-self.grid))

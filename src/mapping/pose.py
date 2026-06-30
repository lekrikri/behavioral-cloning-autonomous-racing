"""2D pose estimation by dead reckoning.

Heading comes from integrating the gyro yaw rate (BMI270, reactive, slow linear
drift). Distance/speed comes from the VESC wheel eRPM (far more reliable than
double-integrating the accelerometer, which drifts quadratically). Residual drift
over a lap is removed afterwards by `close_loop`.

Pure numpy/math — no hardware dependency, so it is unit-testable on the laptop.
"""

import math

import numpy as np


def ray_angles(n_rays, fov_deg):
    """Angles (rad) of the rays relative to the heading, left(+) to right(-).

    Evenly spaced across `fov_deg` centred on the forward direction. Index 0 is the
    leftmost ray. The sign/order convention here MUST be reconciled with the real
    ray producer (visual_rays / depth_to_rays) when wiring the live pipeline.
    """
    half = math.radians(fov_deg) / 2.0
    if n_rays <= 1:
        return np.array([0.0])
    return np.linspace(half, -half, n_rays)


class DeadReckoning:
    """Integrates (yaw_rate, speed) into a planar pose (x, y, theta).

    Convention: theta is the heading in radians, 0 = +x axis, CCW positive.
    Feed an ALREADY-debiased yaw rate (bias removed by the at-rest IMU calibration);
    this class does not estimate the gyro bias itself.
    """

    def __init__(self, k_erpm_to_ms=2.19e-4, x0=0.0, y0=0.0, theta0=0.0, record=True):
        self.k = k_erpm_to_ms
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.record = record
        self.history = [(self.x, self.y, self.theta)] if record else []

    def speed_from_erpm(self, erpm):
        """Convert raw VESC eRPM to m/s using the calibrated scale."""
        return self.k * erpm

    def update(self, yaw_rate, speed, dt):
        """Advance the pose by one step. yaw_rate in rad/s, speed in m/s, dt in s.

        Heading is integrated first, then position uses the MIDPOINT heading over the
        step — this halves the discretisation error on curves so a perfect circle
        closes cleanly.
        """
        self.theta += yaw_rate * dt
        theta_mid = self.theta - 0.5 * yaw_rate * dt
        self.x += speed * math.cos(theta_mid) * dt
        self.y += speed * math.sin(theta_mid) * dt
        if self.record:
            self.history.append((self.x, self.y, self.theta))
        return (self.x, self.y, self.theta)

    def update_erpm(self, yaw_rate, erpm, dt):
        """Convenience: same as update() but takes raw eRPM instead of m/s."""
        return self.update(yaw_rate, self.speed_from_erpm(erpm), dt)

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


def close_loop(history, target_end=None):
    """Redistribute the loop-closing error linearly along the trajectory.

    When the car completes a lap, its estimated end pose should coincide with the
    start. The leftover gap is pure accumulated drift; we spread its correction over
    every point in proportion to cumulative path length (the standard v1 loop-closure
    for an open dead-reckoning chain). A full pose-graph optimiser could do better,
    but this removes the visible drift for a single lap.

    `history` is a sequence of (x, y, ...) tuples (extra fields ignored).
    Returns an (N, 2) array of corrected XY positions.
    """
    xy = np.asarray([(h[0], h[1]) for h in history], dtype=float)
    if len(xy) < 2:
        return xy
    start = xy[0] if target_end is None else np.asarray(target_end, dtype=float)
    error = xy[-1] - start  # gap to drive to zero
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return xy
    weights = (s / s[-1]).reshape(-1, 1)
    return xy - error * weights

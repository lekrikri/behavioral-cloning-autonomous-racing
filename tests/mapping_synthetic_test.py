"""Synthetic validation of the pure mapping layer — no hardware.

Run: python tests/mapping_synthetic_test.py
Validates pose integration accuracy, loop closure on simulated gyro drift, and
occupancy projection (orientation + free/occupied).

Named *_test.py (not test_*.py) on purpose: the repo .gitignore ignores test_*.py."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mapping.pose import DeadReckoning, close_loop, ray_angles
from src.mapping.occupancy import OccupancyGrid


def test_perfect_circle_closes():
    """A constant turn at constant speed must return to the start."""
    v, R = 1.0, 2.0
    w = v / R                      # yaw rate for radius R
    period = 2 * math.pi * R / v
    n = int(round(period / 0.02))
    dt = period / n                # exact integer number of steps per lap

    dr = DeadReckoning()
    for _ in range(n):
        dr.update(w, v, dt)

    gap = math.hypot(dr.x, dr.y)   # started at (0, 0)
    print("T1 perfect circle: closure gap = %.4f m (over %.2f m path)" % (gap, period))
    assert gap < 0.02, "circle did not close: %.4f m" % gap


def test_loop_closure_cancels_drift():
    """A biased gyro drifts the loop open; close_loop must pull the ends together."""
    v, R = 1.0, 2.0
    w = v / R
    period = 2 * math.pi * R / v
    n = int(round(period / 0.02))
    dt = period / n
    bias = 0.02                    # rad/s residual gyro bias

    dr = DeadReckoning()
    for _ in range(n):
        dr.update(w + bias, v, dt)

    gap_open = math.hypot(dr.x - dr.history[0][0], dr.y - dr.history[0][1])
    corrected = close_loop(dr.history)
    gap_closed = float(np.linalg.norm(corrected[-1] - corrected[0]))

    print("T2 loop closure: open gap = %.3f m -> closed gap = %.2e m" % (gap_open, gap_closed))
    assert gap_open > 0.2, "expected meaningful drift to correct, got %.3f" % gap_open
    assert gap_closed < 1e-6, "loop closure failed: %.3e" % gap_closed


def test_occupancy_forward_hit():
    """Facing +x, a 1 m forward ray marks (1,0) occupied and (0.5,0) free."""
    g = OccupancyGrid(res_m=0.05, size_m=10.0, max_range_m=5.0)
    g.integrate_scan((0.0, 0.0, 0.0), [1.0], [0.0])
    p = g.probability()

    i_hit, j_hit = g.world_to_cell(1.0, 0.0)
    i_free, j_free = g.world_to_cell(0.5, 0.0)
    print("T3 occupancy: P(hit@1m)=%.2f  P(free@0.5m)=%.2f" % (p[i_hit, j_hit], p[i_free, j_free]))
    assert p[i_hit, j_hit] > 0.5
    assert p[i_free, j_free] < 0.5


def test_occupancy_respects_heading():
    """Facing +y (theta=pi/2), a forward ray hit lands on the +y axis, not +x."""
    g = OccupancyGrid(res_m=0.05, size_m=10.0, max_range_m=5.0)
    g.integrate_scan((0.0, 0.0, math.pi / 2), [0.5], [0.0])
    p = g.probability()

    i_y, j_y = g.world_to_cell(0.0, 0.5)    # where it SHOULD be
    i_x, j_x = g.world_to_cell(0.5, 0.0)    # where it would be if heading were ignored
    print("T4 heading: P(@0,0.5)=%.2f  P(@0.5,0)=%.2f" % (p[i_y, j_y], p[i_x, j_x]))
    assert p[i_y, j_y] > 0.5
    assert p[i_x, j_x] == 0.5                # untouched -> unknown


def test_ray_angles_layout():
    a = ray_angles(20, 180.0)
    assert len(a) == 20
    assert abs(a[0] - math.pi / 2) < 1e-9    # leftmost = +90 deg
    assert abs(a[-1] + math.pi / 2) < 1e-9   # rightmost = -90 deg
    print("T5 ray angles: left=%+.1f deg  right=%+.1f deg" % (math.degrees(a[0]), math.degrees(a[-1])))


if __name__ == "__main__":
    tests = [
        test_perfect_circle_closes,
        test_loop_closure_cancels_drift,
        test_occupancy_forward_hit,
        test_occupancy_respects_heading,
        test_ray_angles_layout,
    ]
    for t in tests:
        t()
    print("\nAll %d synthetic mapping tests passed." % len(tests))

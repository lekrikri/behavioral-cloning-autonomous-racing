"""Live terrain-mapping subsystem for the RC car.

Additive package: reuses the existing clean modules (vesc_interface, input_manager,
visual_rays, depth_to_rays) without touching the autonomous-driving code.

Pure, hardware-free building blocks live in `pose` and `occupancy` so they can be
unit-tested on the laptop before any OAK/VESC wiring.
"""

from .config import MappingConfig
from .pose import DeadReckoning, close_loop, ray_angles
from .occupancy import OccupancyGrid

__all__ = [
    "MappingConfig",
    "DeadReckoning",
    "close_loop",
    "ray_angles",
    "OccupancyGrid",
]

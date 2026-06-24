"""Central configuration for the live-mapping subsystem.

Plain class (NOT a dataclass) on purpose: the Jetson runtime is Python 3.6.9 and
@dataclass is 3.7+. Defaults carry the values calibrated on 2026-06-24.
"""


class MappingConfig:
    def __init__(
        self,
        # --- odometry / pose ---
        k_erpm_to_ms=2.19e-4,   # v(m/s) = k * raw eRPM from VESCInterface.get_rpm()
        gyro_yaw_axis=2,        # IMU axis used as yaw (0=x,1=y,2=z) — confirm on the mounted cam
        gyro_yaw_sign=1.0,      # flip if positive yaw_rate turns the wrong way
        # --- steering (for teleop while mapping) ---
        servo_center=0.53,      # nominal straight; linkage has mechanical backlash (see plan)
        # --- ray geometry (must match visual_rays/depth_to_rays output) ---
        n_rays=20,
        fov_deg=180.0,
        ray_max_m=3.0,          # TODO reconcile with depth_to_rays clamp / config.json rayMaxDistance(=300)
        # --- occupancy grid ---
        grid_res_m=0.05,
        grid_size_m=20.0,
        # --- telemetry ---
        telemetry_port=5602,    # avoid 5600 (camera_stream*) and 5601 (controller_pd/masked)
    ):
        self.k_erpm_to_ms = k_erpm_to_ms
        self.gyro_yaw_axis = gyro_yaw_axis
        self.gyro_yaw_sign = gyro_yaw_sign
        self.servo_center = servo_center
        self.n_rays = n_rays
        self.fov_deg = fov_deg
        self.ray_max_m = ray_max_m
        self.grid_res_m = grid_res_m
        self.grid_size_m = grid_size_m
        self.telemetry_port = telemetry_port

    def __repr__(self):
        return ("MappingConfig(k_erpm_to_ms=%g, servo_center=%.3f, n_rays=%d, "
                "fov_deg=%.0f, ray_max_m=%.2f, grid_res_m=%.3f, grid_size_m=%.1f)"
                % (self.k_erpm_to_ms, self.servo_center, self.n_rays, self.fov_deg,
                   self.ray_max_m, self.grid_res_m, self.grid_size_m))

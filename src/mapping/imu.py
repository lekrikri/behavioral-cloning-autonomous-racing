"""OAK-D Lite IMU (BMI270) reader for pose estimation.

The OAK device is exclusive (one process), so this module does NOT open its own
connection in normal use: `enable_imu()` adds an IMU node to a shared pipeline and
`IMUReader` consumes that queue alongside the colour stream. A standalone __main__
is provided for bench testing (opens an IMU-only device).

Gyro bias is the dominant drift source (spike 1, 2026-06-24): subtracting an
at-rest constant bias drops residual yaw drift to ~0.01 deg/30s. So we calibrate
the bias for ~2s with the car still, then integrate.

Jetson-side module: requires depthai. Run with OPENBLAS_CORETYPE=ARMV8 (numpy fix).
"""

import time

import depthai as dai

IMU_STREAM = "imu"


def enable_imu(pipeline, rate_hz=200, batch=5):
    """Add an IMU node (raw gyro + accel) to an existing depthai pipeline.

    Returns the IMU node. The caller links nothing else; the node's output is sent
    to an XLinkOut named IMU_STREAM, read later via IMUReader.
    """
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, rate_hz)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(batch)
    imu.setMaxBatchReports(batch * 4)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(IMU_STREAM)
    imu.out.link(xout.input)
    return imu


class IMUReader:
    """Consumes the IMU queue, removes the gyro bias, exposes debiased samples.

    yaw_axis / yaw_sign select which gyro component is the vertical (yaw) axis and
    its sign — unknown until the camera is mounted, so they are configurable.
    """

    def __init__(self, device, yaw_axis=2, yaw_sign=1.0, stream=IMU_STREAM):
        self.q = device.getOutputQueue(stream, maxSize=50, blocking=False)
        self.yaw_axis = yaw_axis
        self.yaw_sign = yaw_sign
        self.bias = (0.0, 0.0, 0.0)

    def _raw_samples(self, timeout_s):
        """Collect (t_s, gx, gy, gz) device-timestamped gyro samples for timeout_s."""
        out = []
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            data = self.q.tryGet()
            if data is not None:
                for pkt in data.packets:
                    g = pkt.gyroscope
                    out.append((g.getTimestampDevice().total_seconds(), g.x, g.y, g.z))
            else:
                time.sleep(0.002)
        return out

    def calibrate_bias(self, seconds=2.0):
        """Average gyro at rest -> per-axis bias. Keep the car perfectly still."""
        s = self._raw_samples(seconds)
        if not s:
            raise RuntimeError("no IMU samples during bias calibration")
        n = float(len(s))
        self.bias = (sum(v[1] for v in s) / n,
                     sum(v[2] for v in s) / n,
                     sum(v[3] for v in s) / n)
        return self.bias

    def drain(self):
        """Return debiased gyro samples received since the last call.

        Each item is (t_s, gx, gy, gz) with the bias removed. The caller integrates
        yaw using the real per-sample dt from the device timestamps.
        """
        bx, by, bz = self.bias
        out = []
        data = self.q.tryGet()
        while data is not None:
            for pkt in data.packets:
                g = pkt.gyroscope
                out.append((g.getTimestampDevice().total_seconds(),
                            g.x - bx, g.y - by, g.z - bz))
            data = self.q.tryGet()
        return out

    def yaw_rate(self, sample):
        """Pick the configured yaw component (signed) from a (t,gx,gy,gz) sample."""
        return self.yaw_sign * sample[1 + self.yaw_axis]


def _standalone():
    """Bench test: open an IMU-only device, calibrate, then integrate all 3 axes.

    Rotate the car by hand about the vertical axis to see which axis is the yaw."""
    import math

    pipe = dai.Pipeline()
    enable_imu(pipe, rate_hz=200)
    print("Opening device (keep the car STILL for bias calibration)...")
    with dai.Device(pipe) as dev:
        reader = IMUReader(dev)
        bias = reader.calibrate_bias(2.0)
        print("Gyro bias (rad/s): x=%+.5f y=%+.5f z=%+.5f" % bias)
        print("Integrating for 15s — rotate the car to find the yaw axis. Ctrl-C to stop.")

        ang = [0.0, 0.0, 0.0]
        prev_t = None
        t0 = time.time()
        try:
            while time.time() - t0 < 15.0:
                for s in reader.drain():
                    if prev_t is not None:
                        dt = s[0] - prev_t
                        if 0 < dt < 0.1:
                            for i in range(3):
                                ang[i] += s[1 + i] * dt
                    prev_t = s[0]
                print("\rangle deg  X=%+7.2f  Y=%+7.2f  Z=%+7.2f" %
                      (math.degrees(ang[0]), math.degrees(ang[1]), math.degrees(ang[2])),
                      end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        print()


if __name__ == "__main__":
    _standalone()

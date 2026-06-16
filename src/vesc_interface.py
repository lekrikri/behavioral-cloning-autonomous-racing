"""
vesc_interface.py — Custom VESC codec for the Flipsky FSESC Mini V6.7 Pro.

Why custom (no pyvesc): the PyPI `pyvesc` wheel depends on the unmaintained
`PyCRC` (whose default CRC is NOT the VESC XMODEM CRC, hence the infamous patch),
and its message classes are not exposed consistently across versions. The VESC
short-packet protocol is small enough to implement directly with zero third-party
deps beyond `pyserial`. This keeps the control stack lightweight and portable
across Python 3.6 (Jetson) and any dev machine. See docs/CONTROL_STACK.md.

Command mapping (high level):
  steering in [-1, 1] -> servo position [0.0, 1.0]   (COMM_SET_SERVO_POS, id 12)
  throttle in [-1, 1] -> motor current [-imax, +imax] A  (COMM_SET_CURRENT, id 6)
                         (negative = reverse / brake)

SAFETY:
  - current_max defaults to 8.0 A; for bench tests use a much lower cap.
  - This car's motor can reach ~90 km/h: always test wheels off the ground.
  - A COMM_ALIVE heartbeat is sent every 300 ms; without it the VESC watchdog
    cuts the motor (set App -> General -> Timeout = 200 ms in VESC Tool).
  - servo position is hard-clamped to [0.10, 0.90] to protect the linkage.

Protocol (short packet, payload < 256 bytes):
  0x02 | len(1) | payload(len) | crc_hi | crc_lo | 0x03
  crc = CRC-16/XMODEM (poly 0x1021, init 0x0000) over payload only.
"""

import struct
import threading
import time


# ── VESC command IDs ────────────────────────────────────────────────────────
COMM_GET_VALUES    = 4
COMM_SET_DUTY      = 5
COMM_SET_CURRENT   = 6
COMM_SET_RPM       = 8
COMM_SET_SERVO_POS = 12   # NB: id 11 is COMM_SET_DETECT — using 11 silently no-ops the servo
COMM_ALIVE         = 30


# ── CRC-16/XMODEM (poly 0x1021, init 0x0000) ────────────────────────────────
def crc16_xmodem(data: bytes) -> int:
    """VESC frame CRC. Validated against COMM_ALIVE: crc([0x1e]) == 0xF3FF."""
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def frame(payload: bytes) -> bytes:
    """Wrap a payload in a VESC short packet (payload length must be < 256)."""
    if len(payload) >= 256:
        raise ValueError("payload too long for a short packet (use long frame)")
    crc = crc16_xmodem(payload)
    return bytes([0x02, len(payload)]) + payload + bytes([crc >> 8, crc & 0xFF, 0x03])


# ── Payload builders ────────────────────────────────────────────────────────
def _servo_payload(pos: float) -> bytes:
    return struct.pack(">Bh", COMM_SET_SERVO_POS, int(round(pos * 1000.0)))  # int16, x1000


def _current_payload(amps: float) -> bytes:
    return struct.pack(">Bi", COMM_SET_CURRENT, int(round(amps * 1000.0)))   # int32, mA


def _duty_payload(duty: float) -> bytes:
    return struct.pack(">Bi", COMM_SET_DUTY, int(round(duty * 100000.0)))    # int32, x1e5


_ALIVE_FRAME = frame(bytes([COMM_ALIVE]))
_GET_VALUES_FRAME = frame(bytes([COMM_GET_VALUES]))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class VESCInterface:
    """
    Lightweight VESC interface for the Flipsky FSESC Mini V6.7 Pro.

    Calibration params (from calibrate_servo.py):
      servo_center : neutral servo position (0.5 = centre)
      servo_range  : +/- amplitude around centre (0.35 default)
      current_max  : max |motor current| in A (low for bench tests)
      invert_steer : flip servo direction if wired/mounted reversed
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        servo_center: float = 0.5,
        servo_range: float = 0.40,   # calibrated 2026-06-16: 0.10/0.90 extremes reach lock, no strain
        current_max: float = 8.0,
        invert_steer: bool = False,
    ):
        self.servo_center = servo_center
        self.servo_range  = servo_range
        self.current_max  = current_max
        self.invert_steer = invert_steer
        self._sim_mode    = False
        self._alive_running = False
        self.ser = None

        try:
            import serial
        except ImportError:
            print("[VESC] pyserial missing -> simulation mode")
            self._sim_mode = True
            return

        try:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.05)
            print("[VESC] connected on %s" % port)
            self._alive_running = True
            self._alive_thread = threading.Thread(target=self._alive_loop, daemon=True)
            self._alive_thread.start()
        except serial.SerialException as e:
            print("[VESC] cannot open %s: %s -> simulation mode" % (port, e))
            self._sim_mode = True

    # ── heartbeat ─────────────────────────────────────────────────────────────
    def _alive_loop(self) -> None:
        """Send COMM_ALIVE every 300 ms to keep the VESC watchdog satisfied."""
        while self._alive_running:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(_ALIVE_FRAME)
            except Exception:
                pass
            time.sleep(0.3)

    # ── low-level writes ──────────────────────────────────────────────────────
    def _write(self, data: bytes) -> None:
        if self._sim_mode or not self.ser or not self.ser.is_open:
            return
        self.ser.write(data)

    def set_servo(self, pos: float) -> None:
        """Set raw servo position [0,1] (clamped for mechanical safety)."""
        self._write(frame(_servo_payload(_clamp(pos, 0.10, 0.90))))

    def set_current(self, amps: float) -> None:
        """Set motor current in A (signed; clamped to [-current_max, +current_max])."""
        self._write(frame(_current_payload(_clamp(amps, -self.current_max, self.current_max))))

    def set_duty(self, duty: float) -> None:
        """Set motor duty cycle (signed [-1,1]); prefer set_current for smoothness."""
        self._write(frame(_duty_payload(_clamp(duty, -1.0, 1.0))))

    # ── mapping helpers ───────────────────────────────────────────────────────
    def _steer_to_servo(self, steering: float) -> float:
        s = -steering if self.invert_steer else steering
        return self.servo_center + self.servo_range * _clamp(s, -1.0, 1.0)

    def set_throttle(self, throttle: float) -> None:
        """throttle in [-1,1] -> current in [-current_max, +current_max] (neg = reverse)."""
        self.set_current(_clamp(throttle, -1.0, 1.0) * self.current_max)

    def drive(self, steering: float, throttle: float) -> None:
        """Full command: steering in [-1,1], throttle in [-1,1] (signed)."""
        if self._sim_mode:
            print("\r[VESC SIM] steer=%+.2f -> servo=%.3f | thr=%+.2f -> %+.2fA   "
                  % (steering, _clamp(self._steer_to_servo(steering), 0.10, 0.90),
                     throttle, _clamp(throttle, -1, 1) * self.current_max),
                  end="", flush=True)
            return
        self.set_servo(self._steer_to_servo(steering))
        self.set_throttle(throttle)

    def send(self, steering: float, accel: float) -> None:
        """Autonomous-driving command: steering in [-1,1], accel in [0,1] (forward only)."""
        self.drive(steering, _clamp(accel, 0.0, 1.0))

    def stop(self) -> None:
        """Emergency stop: motor current 0, servo centred."""
        if not self._sim_mode and self.ser and self.ser.is_open:
            try:
                self._write(frame(_current_payload(0.0)))
                self._write(frame(_servo_payload(self.servo_center)))
            except Exception:
                pass
        print("\n[VESC] STOP (current=0, servo centred)")

    def get_rpm(self) -> float:
        """Best-effort read of motor eRPM via COMM_GET_VALUES (firmware-dependent)."""
        if self._sim_mode or not self.ser or not self.ser.is_open:
            return 0.0
        try:
            self.ser.reset_input_buffer()
            self.ser.write(_GET_VALUES_FRAME)
            time.sleep(0.01)
            payload = self._unframe(self.ser.read(80))
            if payload and payload[0] == COMM_GET_VALUES and len(payload) >= 27:
                # fw 3.x layout: id, temp_fet(2), temp_motor(2), motor_i(4),
                # input_i(4), id(4), iq(4), duty(2), rpm(4) -> rpm at offset 23
                return float(struct.unpack(">i", payload[23:27])[0])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _unframe(raw: bytes):
        """Extract payload from a short VESC packet, verifying CRC. None on failure."""
        if not raw or raw[0] != 0x02:
            return None
        length = raw[1]
        if len(raw) < length + 5:
            return None
        payload = raw[2:2 + length]
        crc = (raw[2 + length] << 8) | raw[3 + length]
        if crc != crc16_xmodem(payload):
            return None
        return payload

    def close(self) -> None:
        self._alive_running = False
        self.stop()
        if getattr(self, "ser", None) and self.ser.is_open:
            self.ser.close()
        print("[VESC] connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    # Hardware-free self-test of the codec (CRC + framing).
    assert crc16_xmodem(bytes([COMM_ALIVE])) == 0xF3FF, "XMODEM CRC mismatch"
    assert _ALIVE_FRAME == bytes([0x02, 0x01, 0x1e, 0xf3, 0xff, 0x03]), "alive frame mismatch"
    print("vesc_interface self-test OK: COMM_ALIVE frame =", _ALIVE_FRAME.hex())

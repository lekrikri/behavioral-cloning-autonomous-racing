"""
teleop_mapping.py — Conduite manuelle + cartographie assistée IMU + dataset.

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/teleop_mapping.py
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/teleop_mapping.py --max-current 20
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/teleop_mapping.py --stream-port 5602

Contrôles F710 (XInput) :
  Left stick X  → steering
  R2 (RT)       → gaz avant (analog)
  L2 (LT)       → marche arrière (analog)
  SELECT (btn6) → démarrer / redémarrer la cartographie
  START  (btn7) → terminer la carto + sauvegarder + quitter

Dataset sauvegardé dans : data/mapping_YYYYMMDD_HHMMSS/
  frames/          — images JPEG 640×320 (une par frame caméra)
  frames.csv       — timestamp, frame_idx, gyro_z, steer, throttle, segment
  track_map.json   — carte TrackMapper (segments droite/virage)
"""

import argparse
import csv
import fcntl
import glob
import math
import os
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from track_mapper import TrackMapper
from visual_rays import white_line_mask

try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

try:
    import depthai as dai
    _DAI_OK = True
except ImportError:
    _DAI_OK = False
    print("[teleop_map] ATTENTION: depthai non installe — mode dry-run")


# ─── Recovery OAK-D (identique à controller_pd.py) ───────────────────────────

def _find_oak_sysfs():
    for vendor_path in glob.glob('/sys/bus/usb/devices/*/idVendor'):
        try:
            with open(vendor_path) as f:
                if f.read().strip() != '03e7':
                    continue
        except Exception:
            continue
        dir_path = os.path.dirname(vendor_path)
        return dir_path, os.path.basename(dir_path)
    return None, None


def _hub_rebind():
    HUB_NAME = '1-2.1'
    hub_driver = '/sys/bus/usb/drivers/usb'
    try:
        with open(os.path.join(hub_driver, 'unbind'), 'w') as f:
            f.write(HUB_NAME)
        print("[map] USB hub {} unbind".format(HUB_NAME))
        time.sleep(3.0)
        with open(os.path.join(hub_driver, 'bind'), 'w') as f:
            f.write(HUB_NAME)
        print("[map] USB hub {} bind".format(HUB_NAME))
        time.sleep(5.0)
        return True
    except Exception as e:
        print("[map] hub rebind err: {}".format(e))
    return False


def _open_oak_pipeline(pipeline):
    """Ouvre le device OAK-D avec recovery automatique UNBOOTED → bootMemory → execve."""
    _all_devs = dai.Device.getAllConnectedDevices()
    _dev_info = _all_devs[0] if _all_devs else None

    if _dev_info is None or "UNBOOTED" in str(getattr(_dev_info, 'state', '')):
        print("[map] Device UNBOOTED/invisible — hub rebind...")
        _hub_rebind()
        _all_devs = dai.Device.getAllConnectedDevices()
        if not _all_devs:
            _all_devs = dai.DeviceBootloader.getAllAvailableDevices()
        _dev_info = _all_devs[0] if _all_devs else None

    _forced_unbooted = False
    if _dev_info is None:
        try:
            _dev_info = dai.DeviceInfo("1.2.1.4")
            _forced_unbooted = True
            print("[map] Fallback DeviceInfo hardcode 1.2.1.4")
        except Exception as e:
            raise RuntimeError("OAK-D introuvable: {}".format(e))

    _state_str     = str(getattr(_dev_info, 'state', ''))
    _recovery_count = int(os.environ.get('OAKD_POST_RECOVERY', '0'))
    _post_recovery  = _recovery_count >= 1
    _need_execve    = False
    _did_boot_memory = False

    if "BOOTLOADER" in _state_str and not _post_recovery:
        _need_execve = True

    if ("UNBOOTED" in _state_str or _forced_unbooted) and _recovery_count < 3:
        print("[map] UNBOOTED → bootMemory subprocess...")
        _bootmem_code = (
            "import depthai as dai, time, sys\n"
            "bls = []\n"
            "for _r in range(6):\n"
            "    bls = dai.DeviceBootloader.getAllAvailableDevices()\n"
            "    if not bls: bls = dai.Device.getAllConnectedDevices()\n"
            "    if bls: break\n"
            "    time.sleep(3)\n"
            "if not bls:\n"
            "    try:\n"
            "        bls = [dai.DeviceInfo('1.2.1.4')]\n"
            "    except: sys.exit(1)\n"
            "bl = dai.DeviceBootloader(bls[0], allowFlashingBootloader=True)\n"
            "fw = dai.DeviceBootloader.getEmbeddedBootloaderBinary("
            "dai.DeviceBootloader.Type.USB)\n"
            "bl.bootMemory(fw)\n"
            "del bl\n"
            "print('bootMemory_OK')\n"
            "time.sleep(2)\n"
        )
        _env_bm = dict(os.environ)
        _env_bm['OPENBLAS_CORETYPE'] = 'ARMV8'
        try:
            _proc = subprocess.Popen(
                ['python3', '-c', _bootmem_code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env_bm)
            _out, _ = _proc.communicate(timeout=90)
            _did_boot_memory = b"bootMemory_OK" in _out or _proc.returncode == 0
            print("[map] bootMem: {}".format(_out.decode().strip()))
            _need_execve = True
        except Exception as e:
            print("[map] bootMemory err: {}".format(e))

    _can_execve = _did_boot_memory or ("BOOTLOADER" in _state_str)
    if _need_execve and _can_execve:
        _env_exec = dict(os.environ)
        _env_exec['OAKD_POST_RECOVERY'] = str(_recovery_count + 1)
        _env_exec['OPENBLAS_CORETYPE']  = 'ARMV8'
        print("[map] Restart XLink-clean (OAKD_POST_RECOVERY={})...".format(
            _recovery_count + 1))
        os.execve(sys.executable, [sys.executable, '-u'] + sys.argv, _env_exec)

    print("[map] Device: {} state={}".format(_dev_info.getMxId(), _dev_info.state))
    return dai.Device(pipeline, _dev_info, True)

try:
    from vesc_interface import VESCInterface
    _VESC_OK = True
except ImportError:
    _VESC_OK = False

# ─── Constantes gamepad F710 XInput ──────────────────────────────────────────
_EVENT_FMT  = "IhBB"
_EVENT_SIZE = struct.calcsize(_EVENT_FMT)
_TYPE_BUTTON = 0x01
_TYPE_AXIS   = 0x02
_TYPE_INIT   = 0x80

AXIS_STEER  = 6   # left stick X — IDENTIQUE teleop_gamepad.py (Mode LED allumé requis)
AXIS_ACCEL  = 5   # R2 / right trigger
AXIS_BRAKE  = 2   # L2 / left trigger
BTN_SELECT  = 6   # BACK  → start mapping   (même numéro que AXIS_STEER — espace séparé)
BTN_START   = 7   # START → finish + save + quit

# ─── Caméra ──────────────────────────────────────────────────────────────────
CAM_W   = 640
CAM_H   = 320
CAM_FPS = 13

# Filtre EMA sur gyro_z (lisse le bruit de conduite manuelle basse vitesse)
# IMU 100 Hz → tau ≈ 200ms : bruit <100ms atténué à ~20%, vrais virages >500ms conservés
GYRO_EMA_ALPHA  = 0.05   # 0.0=figé | 1.0=pas de filtre | augmenter si trop lent à réagir
GYRO_DEADZONE_Z = 0.03   # rad/s — bruit de fond à l'arrêt complet


# ─────────────────────────────────────────────────────────────────────────────
# Gamepad (non-bloquant, identique à teleop_gamepad.py)
# ─────────────────────────────────────────────────────────────────────────────

class Gamepad:
    def __init__(self, path="/dev/input/js0"):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes    = {}
        self.buttons = {}
        self._just_pressed = {}

    def poll(self):
        while True:
            try:
                data = os.read(self.fd, _EVENT_SIZE)
            except BlockingIOError:
                break
            if not data or len(data) < _EVENT_SIZE:
                break
            _, value, etype, number = struct.unpack(_EVENT_FMT, data)
            etype &= ~_TYPE_INIT
            if etype == _TYPE_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
            elif etype == _TYPE_BUTTON:
                prev = self.buttons.get(number, 0)
                self.buttons[number] = value
                if value == 1 and prev == 0:
                    self._just_pressed[number] = True

    def axis(self, n):
        return self.axes.get(n, 0.0)

    def button(self, n):
        return self.buttons.get(n, 0)

    def just_pressed(self, n):
        return self._just_pressed.pop(n, False)

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


def deadzone(x, dz=0.08):
    if abs(x) < dz:
        return 0.0
    return (x - dz * (1.0 if x > 0 else -1.0)) / (1.0 - dz)


def calibrate_triggers(pad, axes, window_s=0.4):
    t_end = time.time() + window_s
    while time.time() < t_end:
        pad.poll()
        time.sleep(0.02)
    return {a: pad.axis(a) for a in axes}


def trigger_frac(pad, axis, rest):
    raw  = pad.axis(axis)
    frac = (raw - rest) / (1.0 - rest) if rest < 1.0 else 0.0
    return max(0.0, min(1.0, frac))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset recorder
# ─────────────────────────────────────────────────────────────────────────────

class DatasetRecorder:
    """Enregistre frames JPEG + CSV pendant la conduite manuelle."""

    def __init__(self, base_dir):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir    = os.path.join(base_dir, "mapping_" + ts)
        self.frames_dir = os.path.join(self.out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._csv_path = os.path.join(self.out_dir, "frames.csv")
        self._csv_f    = open(self._csv_path, "w", newline="")
        self._writer   = csv.writer(self._csv_f)
        self._writer.writerow(["frame_idx", "timestamp", "gyro_z",
                                "steer", "throttle", "segment"])
        self._idx    = 0
        self.active  = False
        print("[recorder] Dataset vers : {}".format(self.out_dir))

    def start(self):
        self.active = True
        print("[recorder] Enregistrement DÉMARRÉ")

    def record(self, bgr, gyro_z, steer, throttle, segment):
        if not self.active:
            return
        fname = "{:06d}.jpg".format(self._idx)
        cv2.imwrite(os.path.join(self.frames_dir, fname), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._writer.writerow([
            self._idx,
            round(time.time(), 4),
            round(gyro_z,   4),
            round(steer,    4),
            round(throttle, 4),
            segment,
        ])
        self._idx += 1

    def stop(self):
        self.active = False
        self._csv_f.flush()
        print("[recorder] {} frames enregistrées dans {}".format(
            self._idx, self.out_dir))

    def close(self):
        try:
            self._csv_f.close()
        except Exception:
            pass

    @property
    def out_dir_path(self):
        return self.out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stream MJPEG (optionnel, port 5602)
# ─────────────────────────────────────────────────────────────────────────────

_stream_frame = [None]
_stream_lock  = threading.Lock()


class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--jpgbnd")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    jpg = _stream_frame[0]
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--jpgbnd\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.07)
        except Exception:
            pass


def _push_stream_frame(bgr, gyro_z, steer, throttle, segment, mapping_on):
    """Encode l'image avec overlay et pousse dans le buffer stream."""
    vis = bgr.copy()
    mask = white_line_mask(vis, morph_k=5, blur_k=3, use_clahe=True, min_area=400)
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.4, 0)

    color_map = {"straight": (0, 200, 0), "turn_L": (0, 180, 255),
                 "turn_R": (255, 130, 0), "idle": (180, 180, 180)}
    seg_color = color_map.get(segment, (200, 200, 200))

    cv2.putText(vis, "steer={:+.2f} thr={:+.2f}".format(steer, throttle),
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(vis, "gyro_z={:+.2f} seg={}".format(gyro_z, segment),
                (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, seg_color, 1)
    rec_txt = "REC" if mapping_on else "STANDBY"
    rec_col = (0, 0, 255) if mapping_on else (120, 120, 120)
    cv2.putText(vis, rec_txt,
                (CAM_W - 70, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_col, 2)

    ok, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        with _stream_lock:
            _stream_frame[0] = enc.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Teleop + mapping assisté IMU")
    p.add_argument("--port",        default="/dev/ttyACM0")
    p.add_argument("--js",          default="/dev/input/js0")
    p.add_argument("--max-current",  type=float, default=3.0,
                   help="A cap pour mode current")
    p.add_argument("--max-duty",     type=float, default=0.08,
                   help="duty cycle max [0..1] — 0.08 = 8%% très lent pour mapping")
    p.add_argument("--max-throttle", type=float, default=1.0,
                   help="scale gaz avant [0..1]")
    p.add_argument("--max-reverse",  type=float, default=0.60,
                   help="scale marche arrière")
    p.add_argument("--servo-center", type=float, default=0.5)
    p.add_argument("--servo-range",  type=float, default=0.40)
    p.add_argument("--invert-steer", action="store_true")
    p.add_argument("--throttle-mode",choices=["current", "duty", "erpm"], default="current")
    p.add_argument("--deadzone",    type=float, default=0.08)
    p.add_argument("--hz",          type=float, default=50.0)
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--map-file",    default="track_map.json")
    p.add_argument("--stream-port", type=int,   default=5602,
                   help="Port HTTP stream MJPEG (0 = désactivé)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Pas de VESC — vision + mapping seuls")
    p.add_argument("--cam-crop-top",type=float, default=0.35,
                   help="Fraction du haut de l'image à supprimer (0 = aucun)")
    args = p.parse_args()

    # ── Gamepad ──────────────────────────────────────────────────────────────
    try:
        pad = Gamepad(args.js)
    except OSError as e:
        print("[teleop_map] Manette introuvable : {}".format(e))
        print("  Vérifier : manette allumée, dongle branché, mode XInput (switch X)")
        return

    print("═" * 65)
    print("  TELEOP MAPPING — G-CAR-000")
    print("  Left=steer | R2=gaz | L2=retro | SELECT=start | START=fin+save")
    print("  Calibration triggers — ne pas toucher R2/L2...")
    rest = calibrate_triggers(pad, [AXIS_ACCEL, AXIS_BRAKE])
    print("  rest R2={:.2f} L2={:.2f}".format(rest[AXIS_ACCEL], rest[AXIS_BRAKE]))
    pad._just_pressed.clear()  # vider les events init pour éviter REC automatique
    print("═" * 65)

    # ── VESC ─────────────────────────────────────────────────────────────────
    vesc = None
    if not args.dry_run and _VESC_OK:
        try:
            vesc = VESCInterface(
                port=args.port,
                servo_center=args.servo_center,
                servo_range=args.servo_range,
                current_max=args.max_current,
                invert_steer=args.invert_steer,
                invert_motor=False,
                throttle_mode=args.throttle_mode,
                max_duty=args.max_duty,
            )
            print("[teleop_map] VESC connecté sur {}".format(args.port))
        except Exception as e:
            print("[teleop_map] VESC KO : {} — dry-run forcé".format(e))
            vesc = None
    else:
        print("[teleop_map] Dry-run : VESC non commandé")

    # ── Dataset recorder ─────────────────────────────────────────────────────
    recorder = DatasetRecorder(args.data_dir)

    # ── TrackMapper ──────────────────────────────────────────────────────────
    mapper = TrackMapper()

    # ── Stream MJPEG ─────────────────────────────────────────────────────────
    if args.stream_port > 0:
        srv = ThreadingHTTPServer(("0.0.0.0", args.stream_port), _StreamHandler)
        t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
        t_srv.start()
        print("[teleop_map] Stream MJPEG → http://{}:{}".format(
            _local_ip(), args.stream_port))

    # ── Boucle principale depthai ─────────────────────────────────────────────
    dt_ctrl = 1.0 / args.hz
    mapping_on = False
    segment    = "idle"

    if not _DAI_OK:
        print("[teleop_map] depthai absent — boucle gamepad seule (sans caméra)")
        _run_gamepad_only(pad, rest, vesc, recorder, mapper, args, dt_ctrl)
        return

    # Pipeline depthai (API v2 — même que controller_pd.py)
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(CAM_W, CAM_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(CAM_FPS)
    xout_cam = pipeline.create(dai.node.XLinkOut)
    xout_cam.setStreamName("preview")
    cam.preview.link(xout_cam.input)

    imu_node = pipeline.create(dai.node.IMU)
    imu_node.enableIMUSensor([dai.IMUSensor.GYROSCOPE_RAW], 100)
    imu_node.setBatchReportThreshold(1)
    imu_node.setMaxBatchReports(10)
    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu_node.out.link(xout_imu.input)

    print("[teleop_map] Ouverture OAK-D Lite...")

    try:
        with _open_oak_pipeline(pipeline) as device:
            q_cam = device.getOutputQueue("preview", maxSize=1, blocking=False)
            q_imu = device.getOutputQueue("imu",     maxSize=50, blocking=False)

            gyro_z_raw = 0.0   # lecture brute OAK-D BMI270
            gyro_z     = 0.0   # signal lissé EMA (utilisé par mapper + recorder)
            last_bgr   = None

            print("[teleop_map] OK — prêt. SELECT=démarrer, START=finir+sauver.")

            while True:
                # ── Lecture caméra ────────────────────────────────────────────
                pkt_cam = q_cam.tryGet()
                if pkt_cam is not None:
                    bgr = pkt_cam.getCvFrame()
                    if args.cam_crop_top > 0:
                        y0  = int(CAM_H * args.cam_crop_top)
                        bgr = cv2.resize(bgr[y0:, :], (CAM_W, CAM_H),
                                         interpolation=cv2.INTER_LINEAR)
                    last_bgr = bgr

                # ── Lecture IMU ───────────────────────────────────────────────
                pkt_imu = q_imu.tryGet()
                if pkt_imu is not None:
                    for pkt in pkt_imu.packets:
                        gyro_z_raw = pkt.gyroscope.z
                        gyro_z = GYRO_EMA_ALPHA * gyro_z_raw + (1.0 - GYRO_EMA_ALPHA) * gyro_z
                        if abs(gyro_z) < GYRO_DEADZONE_Z:
                            gyro_z = 0.0

                # ── Lecture gamepad ───────────────────────────────────────────
                pad.poll()

                steer    = deadzone(pad.axis(AXIS_STEER), args.deadzone)
                rt       = trigger_frac(pad, AXIS_ACCEL, rest[AXIS_ACCEL])
                lt       = trigger_frac(pad, AXIS_BRAKE, rest[AXIS_BRAKE])
                throttle = rt * args.max_throttle - lt * args.max_reverse

                # ── Boutons contrôle ─────────────────────────────────────────
                if pad.just_pressed(BTN_SELECT):
                    if not mapping_on:
                        mapping_on = True
                        mapper.start()
                        recorder.start()
                        print("[teleop_map] *** MAPPING DÉMARRÉ *** (START pour terminer)")
                    else:
                        print("[teleop_map] Mapping déjà actif — appuie START pour terminer")

                if pad.just_pressed(BTN_START):
                    if not mapping_on:
                        print("[teleop_map] START ignoré — appuie d'abord BACK pour démarrer")
                        continue
                    print("[teleop_map] *** FIN MAPPING *** sauvegarde...")
                    if mapping_on:
                        recorder.stop()
                        mapper_data = mapper.save_map(args.map_file)
                        mapper.summary()
                        map_dest = os.path.join(recorder.out_dir_path, "track_map.json")
                        mapper.save_map(map_dest)
                        print("[teleop_map] Carte aussi copiée dans {}".format(map_dest))
                        _print_dataset_summary(recorder, mapper_data)
                    break

                # ── Mise à jour mapper (chaque frame caméra disponible) ──────
                if last_bgr is not None:
                    dt_imu_frame = 1.0 / CAM_FPS
                    segment = mapper.process_frame(gyro_z, dt_imu_frame)
                    if not mapping_on:
                        segment = "idle"

                    # Enregistrement dataset
                    if mapping_on and last_bgr is not None:
                        recorder.record(last_bgr, gyro_z, steer, throttle, segment)

                    # Stream overlay
                    if args.stream_port > 0:
                        _push_stream_frame(last_bgr, gyro_z, steer, throttle,
                                           segment, mapping_on)

                # ── Envoi commandes VESC ─────────────────────────────────────
                if vesc is not None:
                    vesc.drive(steer, throttle)

                # ── Log terminal ─────────────────────────────────────────────
                rec_tag = "REC" if mapping_on else "---"
                print("\r [{}] steer={:+.2f} thr={:+.2f} raw={:+.3f} ema={:+.3f} seg={}   ".format(
                    rec_tag, steer, throttle, gyro_z_raw, gyro_z, segment), end="", flush=True)

                time.sleep(dt_ctrl)

    except KeyboardInterrupt:
        print("\n[teleop_map] Ctrl-C — sauvegarde d'urgence...")
        if mapping_on:
            recorder.stop()
            mapper.save_map(args.map_file)
            if recorder.out_dir_path:
                mapper.save_map(os.path.join(recorder.out_dir_path, "track_map.json"))
    finally:
        if vesc is not None:
            vesc.stop()
            vesc.close()
        recorder.close()
        pad.close()
        print("[teleop_map] Terminé.")


# ─────────────────────────────────────────────────────────────────────────────
# Boucle de secours sans caméra (depthai absent)
# ─────────────────────────────────────────────────────────────────────────────

def _run_gamepad_only(pad, rest, vesc, recorder, mapper, args, dt_ctrl):
    print("[teleop_map] Mode sans caméra — manette seule")
    mapping_on = False
    try:
        while True:
            pad.poll()
            steer    = deadzone(pad.axis(AXIS_STEER), args.deadzone)
            rt       = trigger_frac(pad, AXIS_ACCEL, rest[AXIS_ACCEL])
            lt       = trigger_frac(pad, AXIS_BRAKE, rest[AXIS_BRAKE])
            throttle = rt * args.max_throttle - lt * args.max_reverse

            if pad.just_pressed(BTN_SELECT) and not mapping_on:
                mapping_on = True
                mapper.start()
                recorder.start()
                print("\n[teleop_map] *** MAPPING DÉMARRÉ (sans caméra) ***")

            if pad.just_pressed(BTN_START):
                print("\n[teleop_map] *** FIN ***")
                if mapping_on:
                    recorder.stop()
                    mapper.save_map(args.map_file)
                    mapper.summary()
                break

            if vesc is not None:
                vesc.drive(steer, throttle)

            rec_tag = "REC" if mapping_on else "---"
            print("\r [{}] steer={:+.2f} thr={:+.2f}   ".format(
                rec_tag, steer, throttle), end="", flush=True)
            time.sleep(dt_ctrl)
    except KeyboardInterrupt:
        print("\n[teleop_map] Stop.")
    finally:
        if vesc is not None:
            vesc.stop()
            vesc.close()
        recorder.close()
        pad.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _print_dataset_summary(recorder, mapper_data):
    meta = mapper_data.get("metadata", {})
    evts = mapper_data.get("events", [])
    turns    = [e for e in evts if e["type"] == "turn"]
    straights = [e for e in evts if e["type"] == "straight"]
    print("=" * 55)
    print("  RÉSUMÉ DATASET")
    print("  Frames   : {}".format(recorder._idx))
    print("  Segments : {} droites, {} virages".format(len(straights), len(turns)))
    print("  Yaw total: {}°  Loop={} ".format(
        meta.get("total_yaw_deg", "?"), meta.get("loop_closed", "?")))
    print("  Durée    : {}s".format(meta.get("elapsed_s", "?")))
    print("  Dossier  : {}".format(recorder.out_dir_path))
    print("=" * 55)
    print("  Prochaine étape :")
    print("  python3 src/track_mapper_viz.py --data {}".format(
        recorder.out_dir_path))


if __name__ == "__main__":
    main()

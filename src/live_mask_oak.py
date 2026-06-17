"""
OAK-D Lite — Live preview + masque lignes blanches HSV
Compatible Windows (depthai 3.x) et Linux (depthai 2.x ou 3.x)

Usage:
  python src/live_mask_oak.py [--out ./data/raw_cam] [--width 512] [--height 256]

Touches:
  ESPACE → sauvegarde image brute + masque (256x128)
  M      → toggle affichage masque
  +/-    → ajuster seuil de blanc (V min HSV)
  Q/Esc  → quitter

Installation Linux:
  pip install depthai opencv-python numpy
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger

Installation Windows:
  - Driver WinUSB via Zadig (Options > List All Devices > Movidius MyriadX > WinUSB)
  - pip install depthai opencv-python numpy
"""

import sys, pathlib, argparse
import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    print("depthai non installe : pip install depthai")
    sys.exit(1)

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out",    default=str(pathlib.Path(__file__).parent.parent / "data" / "raw_cam"))
parser.add_argument("--width",  type=int, default=512)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--mode",   choices=["hsv", "canny"], default="hsv",
                    help="hsv=seuil couleur (eclairage homogene) | canny=bords (eclairage variable)")
parser.add_argument("--canny-low",  type=int, default=50,  help="Seuil bas Canny")
parser.add_argument("--canny-high", type=int, default=150, help="Seuil haut Canny")
args = parser.parse_args()

W, H = args.width, args.height
out  = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
MODE = args.mode

# ── Parametres masque blanc ────────────────────────────────────────────────────
HSV_LOW  = np.array([  0,   0, 180])
HSV_HIGH = np.array([180,  50, 255])
ROI_TOP  = H // 2
MORPH_K  = 3
CANNY_LOW  = args.canny_low
CANNY_HIGH = args.canny_high

DEPTHAI_VERSION = tuple(int(x) for x in dai.__version__.split(".")[:2])
IS_V3 = DEPTHAI_VERSION[0] >= 3

# ── Masque (HSV ou Canny selon --mode) ────────────────────────────────────────
def make_mask(bgr):
    k = np.ones((MORPH_K, MORPH_K), np.uint8)
    if MODE == "canny":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        mask = cv2.dilate(mask, k, iterations=2)  # epaissir les bords
    else:
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask[:ROI_TOP, :] = 0
    return mask

def find_line_center(mask):
    M = cv2.moments(mask)
    if M["m00"] > 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None

def overlay(bgr, mask):
    vis   = bgr.copy()
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis   = cv2.addWeighted(vis, 1.0, green, 0.4, 0)
    center = find_line_center(mask)
    if center:
        cv2.circle(vis, center, 6, (0, 0, 255), -1)
        cv2.line(vis, (W // 2, H - 1), center, (255, 0, 0), 1)
        err = center[0] - W // 2
        cv2.putText(vis, f"err={err:+d}px", (4, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    return vis, center

# ── Connexion device ───────────────────────────────────────────────────────────
def find_device():
    # Cherche UNBOOTED ou BOOTED
    for d in dai.Device.getAllConnectedDevices():
        print(f"Device : {d.name}  state={d.state}")
        return d
    return None

device_info = find_device()
if device_info is None:
    print("Aucun device OAK-D trouve.")
    if sys.platform == "linux":
        print("  -> Verifier les regles udev (voir commentaire en haut du fichier)")
    else:
        print("  -> Verifier driver WinUSB via Zadig")
    sys.exit(1)

# ── Pipeline depthai ───────────────────────────────────────────────────────────
def run(device):
    print("=" * 50)
    print(f"depthai {dai.__version__} | {'v3 API' if IS_V3 else 'v2 API'}")
    try:
        calib = device.readCalibration()
        intr  = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        print(f"Intrinsics {W}x{H} : fx={intr[0][0]:.1f}  fy={intr[1][1]:.1f}  "
              f"cx={intr[0][2]:.1f}  cy={intr[1][2]:.1f}")
        print(f"FOV horizontal : {calib.getFov(dai.CameraBoardSocket.CAM_A):.1f} deg")
    except Exception as e:
        print(f"Calibration : {e}")
    print(f"Capture -> {out}")
    print(f"Mode : {MODE.upper()}")
    print("SPACE=sauver | M=masque | +/-=seuil (HSV) | Q=quitter")
    print("=" * 50)

    if IS_V3:
        pipeline = dai.Pipeline(device)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q   = cam.requestOutput((W, H), dai.ImgFrame.Type.BGR888p).createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()
    else:
        # API depthai 2.x
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(W, H)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(30)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)
        device.startPipeline(pipeline)
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    show_mask = True
    counter   = 0
    print("En attente du premier frame...")

    while True:
        raw   = q.get()
        bgr   = raw.getCvFrame()
        seq   = raw.getSequenceNum()
        mask  = make_mask(bgr)
        vis, center = overlay(bgr, mask)

        info = f"seq={seq} | {W}x{H} | blanc={int(mask.sum()/255)}px"
        cv2.putText(vis, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        display = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]) if show_mask else vis
        cv2.imshow("OAK-D Lite | G-CAR-000", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('+'):
            HSV_LOW[2] = min(255, HSV_LOW[2] + 5)
            print(f"Seuil V min : {HSV_LOW[2]}")
        elif key == ord('-'):
            HSV_LOW[2] = max(0,   HSV_LOW[2] - 5)
            print(f"Seuil V min : {HSV_LOW[2]}")
        elif key == ord(' '):
            f_img  = out / f"{counter}_original_image.png"
            f_mask = out / f"{counter}_mask.png"
            cv2.imwrite(str(f_img),  cv2.resize(bgr,  (256, 128)))
            cv2.imwrite(str(f_mask), cv2.resize(mask, (256, 128)))
            print(f"  [{counter}] sauvegarde ({int(mask.sum()/255)}px blancs)")
            counter += 1

    cv2.destroyAllWindows()
    print(f"Termine — {counter} images dans {out}")

with dai.Device(device_info, usb2Mode=True) as device:
    run(device)

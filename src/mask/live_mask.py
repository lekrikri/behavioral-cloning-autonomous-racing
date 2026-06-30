"""
OAK-D Lite — Live preview + masque lignes blanches HSV
Compatible Windows (depthai 3.x) et Linux (depthai 2.x ou 3.x)

Usage:
  python -m src.mask.live_mask [--out ./data/raw_cam] [--width 512] [--height 256]

Touches:
  ESPACE → sauvegarde image brute + masque (256x128)
  M      → toggle affichage masque
  R      → toggle raycasts visuels
  +/-    → ajuster seuil de blanc (V min HSV)
  T      → toggle top-hat (rejette tapis/plinthe/reflet large)
  F      → toggle filtre de forme (rejette reflets ronds)
  C      → toggle cohérence temporelle (anti-scintillement spéculaire)
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

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))
from src.mask.visual_rays import white_line_mask, VisualRays  # source unique du masquage

try:
    import depthai as dai
except ImportError:
    print("depthai non installe : pip install depthai")
    sys.exit(1)

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out",    default=str(_ROOT / "data" / "raw_cam"))
parser.add_argument("--width",  type=int, default=512)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--mode",   choices=["hsv", "canny"], default="hsv",
                    help="hsv=seuil couleur (eclairage homogene) | canny=bords (eclairage variable)")
parser.add_argument("--canny-low",  type=int, default=50,  help="Seuil bas Canny")
parser.add_argument("--canny-high", type=int, default=150, help="Seuil haut Canny")
parser.add_argument("--source", choices=["device", "hub"], default="hub",
                    help="hub=lit le camera_hub (mémoire partagée, défaut) | device=ouvre l'OAK-D")
args = parser.parse_args()

W, H = args.width, args.height
out  = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
MODE = args.mode

# ── Parametres masque blanc ────────────────────────────────────────────────────
HSV_LOW  = np.array([  0,   0, 200])   # V min 200 (180 trop permissif en labo)
HSV_HIGH = np.array([180,  40, 255])   # S max 40 (moins de surfaces colorées claires)
ROI_TOP  = int(H * 0.6)               # ignorer 60% du haut (fond/plafond/décor)
MORPH_K  = 5                           # kernel plus grand → moins de bruit
CANNY_LOW  = args.canny_low
CANNY_HIGH = args.canny_high

# ── Couches anti-artefacts (toggles live, OFF par défaut) ──────────────────────
TOPHAT_K       = 0     # touche 't' → 25 : ne garde que les lignes fines (rejette tapis/plinthe/reflet large)
TOPHAT_THRESH  = 12    # seuil réponse top-hat
MAX_FILL_RATIO = 1.0   # touche 'f' → 0.45 : rejette les blobs trop « pleins » (reflets ronds)
TEMPORAL_WIN   = 1     # touche 'c' → 5 : médiane temporelle par rayon (anti-scintillement spéculaire)
TOPHAT_K_ON    = 25
MAX_FILL_ON    = 0.45
TEMPORAL_ON    = 5

DEPTHAI_VERSION = tuple(int(x) for x in dai.__version__.split(".")[:2])
IS_V3 = DEPTHAI_VERSION[0] >= 3

# ── Masque (HSV ou Canny selon --mode) ────────────────────────────────────────
def make_mask(bgr):
    mask = white_line_mask(
        bgr, mode=MODE, hsv_low=HSV_LOW, hsv_high=HSV_HIGH,
        canny_low=CANNY_LOW, canny_high=CANNY_HIGH, morph_k=MORPH_K,
        tophat_k=TOPHAT_K, tophat_thresh=TOPHAT_THRESH, max_fill_ratio=MAX_FILL_RATIO,
    )
    mask[:ROI_TOP, :] = 0  # l'outil de dev masque toute la moitié haute
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

# ── Source de frames ───────────────────────────────────────────────────────────
class _HubPkt:
    """Imite un paquet depthai (getCvFrame/getSequenceNum) au-dessus d'une frame SHM."""
    __slots__ = ("_seq", "_bgr")
    def __init__(self, seq, bgr):
        self._seq, self._bgr = seq, bgr
    def getCvFrame(self):    return self._bgr
    def getSequenceNum(self): return self._seq


class _HubQueue:
    """Adapte un FrameClient SHM à l'API queue depthai (.get().getCvFrame())."""
    def __init__(self):
        from src.cam.hub import FrameClient
        self._c = FrameClient()
    def get(self):
        try:
            seq, bgr = self._c.get()
        except (ConnectionError, OSError) as e:
            # la boucle de live_mask catch RuntimeError (perte de flux) → sortie propre
            raise RuntimeError("hub indisponible: {}".format(e))
        return _HubPkt(seq, bgr)


def find_device():
    # Cherche UNBOOTED ou BOOTED
    for d in dai.Device.getAllConnectedDevices():
        print(f"Device : {d.name}  state={d.state}")
        return d
    return None

device_info = find_device() if args.source == "device" else None
if args.source == "device" and device_info is None:
    print("Aucun device OAK-D trouve.")
    if sys.platform == "linux":
        print("  -> Verifier les regles udev (voir commentaire en haut du fichier)")
    else:
        print("  -> Verifier driver WinUSB via Zadig")
    sys.exit(1)

# ── Pipeline depthai ───────────────────────────────────────────────────────────
def run(device):
    global TOPHAT_K, MAX_FILL_RATIO, TEMPORAL_WIN
    print("=" * 50)
    print(f"depthai {dai.__version__} | {'v3 API' if IS_V3 else 'v2 API'}")
    if device is not None:
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
    print("SPACE=sauver | M=masque | R=raycasts | +/-=seuil V (HSV)")
    print("t=top-hat | f=filtre forme | c=coherence temporelle | Q=quitter")
    print("=" * 50)

    if device is None:
        q = _HubQueue()   # frames depuis le hub (SHM, zéro-copie)
        print("source = hub (SHM /dev/shm/robocar_cam_color)")
    elif IS_V3:
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

    vr = VisualRays(
        img_width=W, img_height=H,
        mode=MODE,
        hsv_low=tuple(HSV_LOW), hsv_high=tuple(HSV_HIGH),
        morph_k=MORPH_K,
        row_band=(ROI_TOP / H, 1.0),
        tophat_k=TOPHAT_K, tophat_thresh=TOPHAT_THRESH,
        max_fill_ratio=MAX_FILL_RATIO, temporal_window=TEMPORAL_WIN,
    )

    show_mask  = True
    show_rays  = True
    counter    = 0
    print("En attente du premier frame...")

    while True:
        try:
            raw = q.get()
        except RuntimeError as e:
            # X_LINK_ERROR : le lien USB a lâché (souvent alim/câble). Sortie propre
            # plutôt qu'un stacktrace. Fix robuste (reconnexion) à voir plus tard.
            print(f"\n[OAK-D] Lien USB perdu : {e}")
            print("  -> Cause probable : alimentation/câble. Essayer un câble USB3 data "
                  "+ alim externe (câble Y), ou forcer l'USB2.")
            break
        bgr   = raw.getCvFrame()
        seq   = raw.getSequenceNum()
        mask  = make_mask(bgr)
        vis, center = overlay(bgr, mask)

        # Raycasts visuels (vert=libre, rouge=bord proche)
        if show_rays:
            rays = vr(bgr)
            for col, ray in zip(vr.cols, rays):
                r = int(255 * (1.0 - ray)); g = int(255 * ray)
                y_top = int(H - ray * H * 0.4)
                cv2.line(vis, (col, H), (col, y_top), (0, g, r), 1)

        info = f"seq={seq} | {W}x{H} | blanc={int(mask.sum()/255)}px | V>={HSV_LOW[2]}"
        cv2.putText(vis, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        layers = (f"TH(t)={'ON' if TOPHAT_K > 1 else 'off'} "
                  f"FILL(f)={'ON' if MAX_FILL_RATIO < 1.0 else 'off'} "
                  f"TEMP(c)={TEMPORAL_WIN if TEMPORAL_WIN > 1 else 'off'}")
        cv2.putText(vis, layers, (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        display = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]) if show_mask else vis
        cv2.imshow("OAK-D Lite | G-CAR-000", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('r'):
            show_rays = not show_rays
            print(f"Raycasts : {'ON' if show_rays else 'OFF'}")
        elif key == ord('+'):
            HSV_LOW[2] = min(255, HSV_LOW[2] + 5)
            print(f"Seuil V min : {HSV_LOW[2]}")
        elif key == ord('-'):
            HSV_LOW[2] = max(0,   HSV_LOW[2] - 5)
            print(f"Seuil V min : {HSV_LOW[2]}")
        elif key == ord('t'):
            TOPHAT_K = 0 if TOPHAT_K > 1 else TOPHAT_K_ON
            vr.tophat_k = TOPHAT_K
            print(f"Top-hat : {'ON (k=%d)' % TOPHAT_K if TOPHAT_K > 1 else 'OFF'}")
        elif key == ord('f'):
            MAX_FILL_RATIO = 1.0 if MAX_FILL_RATIO < 1.0 else MAX_FILL_ON
            vr.max_fill_ratio = MAX_FILL_RATIO
            print(f"Filtre forme : {'ON (fill<=%.2f)' % MAX_FILL_RATIO if MAX_FILL_RATIO < 1.0 else 'OFF'}")
        elif key == ord('c'):
            TEMPORAL_WIN = 1 if TEMPORAL_WIN > 1 else TEMPORAL_ON
            vr.set_temporal(TEMPORAL_WIN)
            print(f"Coherence temporelle : {'ON (win=%d)' % TEMPORAL_WIN if TEMPORAL_WIN > 1 else 'OFF'}")
        elif key == ord(' '):
            f_img  = out / f"{counter}_original_image.png"
            f_mask = out / f"{counter}_mask.png"
            cv2.imwrite(str(f_img),  cv2.resize(bgr,  (256, 128)))
            cv2.imwrite(str(f_mask), cv2.resize(mask, (256, 128)))
            print(f"  [{counter}] sauvegarde ({int(mask.sum()/255)}px blancs)")
            counter += 1

    cv2.destroyAllWindows()
    print(f"Termine — {counter} images dans {out}")

if args.source == "hub":
    run(None)                       # frames depuis le hub (SHM), aucun device ouvert
else:
    with dai.Device(device_info) as device:
        run(device)

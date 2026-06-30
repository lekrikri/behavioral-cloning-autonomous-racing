"""
mask_stream.py — Traitement masque lignes sur la Jetson + stream MJPEG vers navigateur.

Affichage déporté FLUIDE (JPEG compressé) avec contrôle clavier depuis le PC, sans
aucune dépendance côté PC (juste un navigateur). Alternative à X11 (lent, non compressé).

Le traitement (white_line_mask + 3 couches + raycasts) tourne sur la Jetson ;
seules les frames rendues + compressées transitent vers le PC.

── Jetson ──
  OPENBLAS_CORETYPE=ARMV8 python3 mask_stream.py [--port 8088] [--width 512] [--height 256]
  # --source hub (défaut) : lit le hub caméra SHM (preview + conduite autonome partagent la caméra).
  #   Le hub doit tourner en service système robocar-cam-hub (docs/SERVICES.md) ; absent →
  #   on avertit et on indique comment le relancer.

── PC (LG Gram) ──
  ssh -L 8088:localhost:8088 robocar      # forward le port (sens laptop→Jetson, OK Tailscale)
  # puis ouvrir http://localhost:8088 dans Firefox

Contrôles (clavier dans le navigateur, ou boutons) :
  t  top-hat on/off        , / .  kernel top-hat -/+
  f  filtre forme on/off   ; / '  max_fill -/+
  c  coherence temp on/off n / b  fenetre -/+
  [ / ]  tophat_thresh -/+    o / p  crop ROI haut +/-
  + / -  seuil V (blanc)
  m  masque cote-a-cote    r  raycasts
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

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
    raise SystemExit(1)

# Valeurs "ON" des toggles (mêmes défauts que live_mask_oak.py)
ON_TOPHAT_K = 25
ON_FILL     = 0.45
ON_TEMPORAL = 5


class State:
    """État partagé thread-safe : params de réglage + dernière frame JPEG."""

    def __init__(self, width, height):
        self.lock = threading.Lock()
        self.W = width
        self.H = height
        self.hsv_v_min     = 200
        self.tophat_k      = 0
        self.tophat_thresh = 12
        self.max_fill      = 1.0
        self.temporal      = 1
        self.roi_frac      = 0.0   # fraction du haut ignorée (0 = plein cadre)
        self.show_mask     = True
        self.show_rays     = True
        self.jpeg          = None
        self.fps           = 0.0
        self.running       = True

    def snapshot(self):
        with self.lock:
            return dict(
                hsv_v_min=self.hsv_v_min, tophat_k=self.tophat_k,
                tophat_thresh=self.tophat_thresh, max_fill=self.max_fill,
                temporal=self.temporal, show_mask=self.show_mask,
                show_rays=self.show_rays, roi_frac=self.roi_frac,
            )

    def apply_key(self, k):
        """Applique une touche de contrôle. Retourne un libellé du nouvel état."""
        with self.lock:
            if   k == "t": self.tophat_k = 0 if self.tophat_k > 1 else ON_TOPHAT_K
            elif k == "f": self.max_fill = 1.0 if self.max_fill < 1.0 else ON_FILL
            elif k == "c": self.temporal = 1 if self.temporal > 1 else ON_TEMPORAL
            elif k == ".": self.tophat_k = min(99, max(3, self.tophat_k) + 2)
            elif k == ",": self.tophat_k = max(0, self.tophat_k - 2)
            elif k == "]": self.tophat_thresh = min(120, self.tophat_thresh + 2)
            elif k == "[": self.tophat_thresh = max(0, self.tophat_thresh - 2)
            elif k == "'": self.max_fill = round(min(1.0, self.max_fill + 0.05), 2)
            elif k == ";": self.max_fill = round(max(0.05, self.max_fill - 0.05), 2)
            elif k == "n": self.temporal = min(15, max(1, self.temporal) + 2)
            elif k == "b": self.temporal = max(1, self.temporal - 2)
            elif k == "o": self.roi_frac = round(min(0.9, self.roi_frac + 0.05), 2)
            elif k == "p": self.roi_frac = round(max(0.0, self.roi_frac - 0.05), 2)
            elif k in ("+", "="): self.hsv_v_min = min(255, self.hsv_v_min + 5)
            elif k == "-":        self.hsv_v_min = max(0,   self.hsv_v_min - 5)
            elif k == "m": self.show_mask = not self.show_mask
            elif k == "r": self.show_rays = not self.show_rays
            return (f"V>={self.hsv_v_min} TH(k={self.tophat_k},thr={self.tophat_thresh}) "
                    f"FILL={self.max_fill} TEMP={self.temporal} ROIcrop={int(self.roi_frac*100)}%")


class Driver:
    """Supervise le sous-process de conduite : none / manual / auto.

    Nécessite d'être lancé depuis le vrai dépôt (scripts src/, models/, vesc_interface).
    Pour que 'auto' coexiste avec la preview, déployer la topologie hub :
      hub (src/cam/hub.py, SHM)  +  mask_stream --source hub  +  ce sélecteur (lance inference --source hub).
    """

    def __init__(self, repo_root, vesc_port="/dev/ttyACM0"):
        self.repo_root = repo_root
        self.vesc_port = vesc_port
        self.lock = threading.Lock()
        self.proc = None
        self.mode = "none"

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _vesc_stop(self):
        """Envoie un stop moteur best-effort (le watchdog VESC coupe déjà si on tue un process)."""
        try:
            if self.repo_root not in sys.path:
                sys.path.insert(0, self.repo_root)
            from src.control.vesc_interface import VESCInterface
            v = VESCInterface(port=self.vesc_port)
            v.stop()
            v.close()
        except Exception as e:
            print(f"[drive] VESC stop best-effort échoué : {e}")

    def set_mode(self, m):
        with self.lock:
            self._kill()
            env = dict(os.environ)
            env["OPENBLAS_CORETYPE"] = "ARMV8"
            if m == "manual":
                cmd = [sys.executable, "-m", "src.control.teleop_gamepad",
                       "--port", self.vesc_port, "--js", "/dev/input/js0"]
            elif m == "auto":
                cmd = [sys.executable, "-m", "src.control.inference_realcar",
                       "--perception-mode", "visual", "--source", "hub",
                       "--model", "models/v18/best_jetson.onnx", "--duty-max", "0.20"]
            else:
                cmd = None

            if cmd is None:
                self._vesc_stop()
                m = "none"
            else:
                try:
                    self.proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env)
                    time.sleep(0.3)
                    if self.proc.poll() is not None:  # mort immédiate = échec lancement
                        print(f"[drive] '{m}' n'a pas démarré (code {self.proc.returncode}) — "
                              f"dépôt/périphériques ?")
                        self.proc = None
                        m = "none"
                except (OSError, FileNotFoundError) as e:
                    print(f"[drive] lancement '{m}' impossible : {e}")
                    m = "none"
            self.mode = m
            return self.status()

    def status(self):
        alive = self.proc is not None and self.proc.poll() is None
        return {"mode": self.mode, "alive": alive}

    def shutdown(self):
        with self.lock:
            self._kill()
            self._vesc_stop()


def render(bgr, mask, rays, vr, snap, fps):
    """Construit l'overlay (identique d'esprit à live_mask_oak.py)."""
    vis = bgr.copy()
    green = np.zeros_like(vis)
    green[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green, 0.4, 0)

    if snap["show_rays"]:
        H = vis.shape[0]
        for col, ray in zip(vr.cols, rays):
            r = int(255 * (1.0 - ray)); g = int(255 * ray)
            y_top = int(H - ray * H * 0.4)
            cv2.line(vis, (int(col), H), (int(col), y_top), (0, g, r), 1)

    white_px = int(mask.sum() / 255)
    info = (f"{vis.shape[1]}x{vis.shape[0]} | blanc={white_px}px | "
            f"V>={snap['hsv_v_min']} | {fps:.0f}fps")
    cv2.putText(vis, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
    layers = (f"TH(t)={'k%d/thr%d' % (snap['tophat_k'], snap['tophat_thresh']) if snap['tophat_k'] > 1 else 'off'} "
              f"FILL(f)={snap['max_fill'] if snap['max_fill'] < 1.0 else 'off'} "
              f"TEMP(c)={snap['temporal'] if snap['temporal'] > 1 else 'off'} "
              f"ROIcrop(o/p)={int(snap['roi_frac'] * 100)}%")
    cv2.putText(vis, layers, (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    if snap["show_mask"]:
        vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    return vis


def _frames_device(state):
    """Générateur de frames depuis l'OAK-D directement (mode standalone)."""
    W, H = state.W, state.H
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(W, H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        print(f"[stream] OAK-D OK (device) — {W}x{H} — sert le flux")
        while state.running:
            yield q.get().getCvFrame()


def _frames_hub(state):
    """Générateur de frames depuis le hub (mémoire partagée, zéro-copie)."""
    from src.cam.hub import FrameClient
    client = FrameClient()
    print("[stream] source = hub (SHM /dev/shm/robocar_cam_color)")
    while state.running:
        try:
            yield client.getCvFrame()   # get() auto-connecte (attend la région SHM)
        except (ConnectionError, OSError):
            print("[stream] hub indisponible — reconnexion…")
            client.close()
            time.sleep(1.0)
    client.close()


def capture_loop(state, source="device"):
    """Thread : frames (device|hub) → masque (params live) → raycasts → overlay → JPEG."""
    vr = VisualRays(img_width=state.W, img_height=state.H, mode="hsv",
                    row_band=(0.0, 1.0))
    prev_temporal = 1
    prev_roi = -1.0
    t_prev = time.time()
    fps = 0.0

    H = state.H
    frames = _frames_hub(state) if source == "hub" else _frames_device(state)
    for bgr in frames:
            snap = state.snapshot()

            hsv_low  = (0, 0, snap["hsv_v_min"])
            hsv_high = (180, 40, 255)
            mask = white_line_mask(
                bgr, mode="hsv", hsv_low=hsv_low, hsv_high=hsv_high, morph_k=5,
                tophat_k=snap["tophat_k"], tophat_thresh=snap["tophat_thresh"],
                max_fill_ratio=snap["max_fill"],
            )
            roi_top = int(H * snap["roi_frac"])
            if roi_top > 0:
                mask[:roi_top, :] = 0

            # Synchroniser les params de VisualRays (rays) avec l'état live
            vr.tophat_k       = snap["tophat_k"]
            vr.tophat_thresh  = snap["tophat_thresh"]
            vr.max_fill_ratio = snap["max_fill"]
            vr.hsv_low[2]     = snap["hsv_v_min"]
            if snap["temporal"] != prev_temporal:
                vr.set_temporal(snap["temporal"])
                prev_temporal = snap["temporal"]
            if snap["roi_frac"] != prev_roi:
                vr.set_row_band(snap["roi_frac"], 1.0)
                prev_roi = snap["roi_frac"]
            rays = vr(bgr)

            now = time.time()
            dt = now - t_prev; t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            vis = render(bgr, mask, rays, vr, snap, fps)
            ok, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with state.lock:
                    state.jpeg = enc.tobytes()
                    state.fps = fps
    print("[stream] capture terminee")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>mask_stream</title>
<style>body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:8px}
img{max-width:100%;border:1px solid #333}button{background:#222;color:#ddd;border:1px solid #444;
padding:6px 10px;margin:2px;cursor:pointer;font-family:monospace}button:hover{background:#333}
#k{color:#6cf}.row{margin:6px 0}</style></head><body>
<div class=row><img src="/stream.mjpg"></div>
<div class=row style="border:1px solid #533;padding:6px">conduite:
<button onclick="M('none')" style="background:#511">RIEN (stop)</button>
<button onclick="M('manual')">MANUEL</button>
<button onclick="M('auto')" style="background:#151">AUTONOME</button>
<span id=md>mode=none</span></div>
<div class=row>
<button onclick="K('t')">t top-hat</button>
<button onclick="K(',')">, k-</button><button onclick="K('.')">. k+</button>
<button onclick="K('[')">[ thr-</button><button onclick="K(']')">] thr+</button>
<button onclick="K('f')">f forme</button>
<button onclick="K(';')">; fill-</button><button onclick="K(&quot;'&quot;)">' fill+</button>
<button onclick="K('c')">c temporel</button>
<button onclick="K('b')">b win-</button><button onclick="K('n')">n win+</button>
<button onclick="K('-')">- V</button><button onclick="K('=')">+ V</button>
<button onclick="K('o')">o crop+</button><button onclick="K('p')">p crop-</button>
<button onclick="K('m')">m masque</button><button onclick="K('r')">r rays</button>
</div>
<div class=row>etat: <span id=k>—</span></div>
<script>
function K(k){fetch('/key?k='+encodeURIComponent(k)).then(r=>r.text()).then(t=>{document.getElementById('k').textContent=t});}
function M(m){fetch('/mode?m='+m).then(r=>r.json()).then(s=>{document.getElementById('md').textContent='mode='+s.mode+(s.alive?' (process actif)':'');});}
document.addEventListener('keydown',function(e){
  var k=e.key; if(k===' '||k.length===1){e.preventDefault();K(k);}
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    state = None   # injecté
    driver = None  # injecté

    def log_message(self, *a):
        pass  # silencieux

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/key":
            qs = parse_qs(urlparse(self.path).query)
            k = (qs.get("k", [""])[0])[:1]
            label = self.state.apply_key(k) if k else ""
            body = label.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/mode":
            qs = parse_qs(urlparse(self.path).query)
            m = qs.get("m", ["none"])[0]
            st = self.driver.set_mode(m) if self.driver else {"mode": "n/a", "alive": False}
            body = json.dumps(st).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while self.state.running:
                    with self.state.lock:
                        jpg = self.state.jpeg
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)  # ~25 fps max
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (navigateur) deconnecte
        else:
            self.send_error(404)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=8088)
    parser.add_argument("--width",    type=int, default=512)
    parser.add_argument("--height",   type=int, default=256)
    parser.add_argument("--source",   choices=["device", "hub"], default="hub",
                        help="hub=lit le hub caméra en mémoire partagée (défaut, partage l'OAK-D) | device=ouvre l'OAK-D")
    args = parser.parse_args()

    state = State(args.width, args.height)
    Handler.state = state
    repo_root = str(_ROOT)
    Handler.driver = Driver(repo_root)

    if args.source == "hub":
        from src.cam.hub import ensure_hub_or_prompt
        if not ensure_hub_or_prompt():
            sys.exit(1)

    t = threading.Thread(target=capture_loop,
                         args=(state, args.source), daemon=True)
    t.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[stream] HTTP sur :{args.port}  →  ssh -L {args.port}:localhost:{args.port} robocar")
    print(f"[stream] puis ouvrir http://localhost:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stream] arret")
    finally:
        state.running = False
        if Handler.driver is not None:
            Handler.driver.shutdown()  # coupe la conduite + stop VESC
        time.sleep(0.2)


if __name__ == "__main__":
    main()

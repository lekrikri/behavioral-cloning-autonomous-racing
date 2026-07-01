"""Debug interface HTTP server: routes the home + mask pages over the hub client.

  OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.debug.server [--port 8088] [--profile classic]
  # PC : ssh -L 8088:localhost:8088 robocar  puis  http://localhost:8088
"""

import argparse
import json
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import pathlib
_ROOT = pathlib.Path(__file__).resolve()
while not (_ROOT / "src" / "__init__.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from src.mask.perception_config import resolve_profile, write_active_profile
from src.tools.debug.capture import MaskState, capture_loop
from src.tools.debug.control_state import ControlState
from src.tools.debug.driver import Driver
from src.tools.debug import pages


class Handler(BaseHTTPRequestHandler):
    state = None      # injected: MaskState
    control = None    # injected: ControlState
    driver = None     # injected: Driver

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/plain; charset=utf-8", code=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path == "/":
            self._send(pages.home_page(), "text/html; charset=utf-8")
        elif path == "/mask":
            self._send(pages.mask_page(), "text/html; charset=utf-8")
        elif path == "/intelligence":
            self._send(pages.intelligence_page(), "text/html; charset=utf-8")
        elif path == "/control_params":
            self._send(json.dumps(self.control.ui_state()), "application/json")
        elif path == "/control_param":
            name = qs.get("name", [""])[0]
            value = qs.get("value", [""])[0]
            self._send(self.control.set_param(name, value) if name else "")
        elif path == "/brain":
            self._send(self.control.set_brain(qs.get("kind", [""])[0]))
        elif path == "/control_save":
            self._send(self.control.save_to_profile())
            write_active_profile(self.control.profile.name)
        elif path == "/params":
            self._send(json.dumps(self.state.ui_state()), "application/json")
        elif path == "/stats":
            with self.state.lock:
                st = {"fps": round(self.state.fps, 1), "frames": self.state.frames}
            self._send(json.dumps(st), "application/json")
        elif path == "/param":
            name = qs.get("name", [""])[0]
            value = qs.get("value", [""])[0]
            self._send(self.state.set_param(name, value) if name else "")
        elif path == "/mask_profile":
            name = qs.get("name", [""])[0]
            self.driver.set_profile(name)               # sync le profil de conduite (Play)
            msg = self.state.load_profile_into(name)
            self.control.load_profile_into(name)        # sync l'onglet Intelligence + canal live
            write_active_profile(name)                  # persiste -> prod + prochain démarrage
            self._send(msg)
        elif path == "/mask_save":
            msg = self.state.save_to_profile()
            write_active_profile(self.state.profile.name)
            self._send(msg)
        elif path == "/status":
            self._send(json.dumps(self.driver.status()), "application/json")
        elif path == "/profile":
            name = qs.get("name", ["classic"])[0]
            self.state.load_profile_into(name)          # sync la preview masque
            self.control.load_profile_into(name)        # sync l'onglet Intelligence + canal live
            st = self.driver.set_profile(name)
            write_active_profile(name)                  # persiste -> prod + prochain démarrage
            self._send(json.dumps(st), "application/json")
        elif path == "/drive":
            action = qs.get("action", ["stop"])[0]
            fn = {"play": self.driver.play, "pause": self.driver.pause}.get(action, self.driver.stop)
            self._send(json.dumps(fn()), "application/json")
        elif path == "/stream.mjpg":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while self.state.running:
                with self.state.lock:
                    jpg = self.state.jpeg
                if jpg is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                    self.wfile.write(jpg + b"\r\n")
                time.sleep(0.04)   # ~25 fps max
        except (BrokenPipeError, ConnectionResetError):
            pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--profile", default=None,
                        help="force un profil ; sinon reprend le dernier profil actif persisté")
    parser.add_argument("--vesc-port", default="/dev/ttyACM0")
    args = parser.parse_args()

    # Profil : --profile explicite > dernier profil actif (configs/active_profile.txt) > classic.
    profile, name = resolve_profile(args.profile)
    if args.profile:
        write_active_profile(name)          # un lancement explicite devient le profil actif
    print("[debug] profil actif = %s" % name)

    from src.cam.hub import ensure_hub_or_prompt, SHM_COLOR
    if not ensure_hub_or_prompt(SHM_COLOR):
        sys.exit(1)

    state = MaskState(profile)
    Handler.state = state
    Handler.control = ControlState(profile)
    Handler.driver = Driver(str(_ROOT), vesc_port=args.vesc_port, profile_name=name)

    t = threading.Thread(target=capture_loop, args=(state,), daemon=True)
    t.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print("[debug] http :%d  →  ssh -L %d:localhost:%d robocar  puis  http://localhost:%d"
          % (args.port, args.port, args.port, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[debug] arrêt")
    finally:
        state.running = False
        Handler.driver.shutdown()
        time.sleep(0.2)


if __name__ == "__main__":
    main()

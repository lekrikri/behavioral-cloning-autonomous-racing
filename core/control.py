"""Endpoint de contrôle léger (toujours-on, localhost) : status + commandes.

Routes :
  GET  /status          -> état du superviseur
  POST /ui/connect      -> spawn le stream UI (à la 1re connexion web)
  POST /ui/disconnect   -> kill le stream UI
  POST /takeover        -> prise de main manuelle (coupe l'auto)
  POST /release         -> rend la main (retour auto si profil lancé, sinon inerte)
  POST /profile {profile} -> lance un profil (worker auto)
  POST /stop            -> arrête le profil (retour inerte)
  POST /update          -> demande d'update (refusée en conduite)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_server(supervisor, port, host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:           # body malformé -> {} (réponse propre, pas de 500)
                return {}

        def do_GET(self):
            if self.path == "/status":
                self._json(200, supervisor.status())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            simple = {
                "/ui/connect": supervisor.ui_connect,
                "/ui/disconnect": supervisor.ui_disconnect,
                "/takeover": supervisor.takeover,
                "/release": supervisor.release,
                "/stop": supervisor.stop_profile,
                "/update": supervisor.request_update,
            }
            if self.path == "/profile":
                self._json(200, supervisor.launch_profile(self._body().get("profile")))
            elif self.path in simple:
                self._json(200, simple[self.path]())
            else:
                self._json(404, {"error": "not found"})

    srv = HTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

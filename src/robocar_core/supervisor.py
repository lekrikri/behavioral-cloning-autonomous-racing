"""Superviseur : charge la config, lance le minimum, gère manette / UI / update / profils.

Le core ne fait pas la perception ni le contrôle : il spawn/kill des workers. Voir
docs/CORE_DESIGN.md et docs/schemas/.
"""
import threading

from . import config
from .control import make_server
from .gamepad import GamepadWatcher
from .workers import WorkerManager


class Supervisor:
    def __init__(self, vehicle_path=None, profiles_path=None, profile=None, cwd=None):
        self.vehicle = config.load_vehicle(vehicle_path)
        self.profiles_cfg = config.load_profiles(profiles_path)
        self.profile = profile or self.profiles_cfg.get("default")
        self.workers = WorkerManager(self.profiles_cfg["workers"], cwd=cwd)
        self._lock = threading.RLock()
        self.mode = "auto"          # "auto" | "manual"
        self.manual_armed = False   # manette présente (manuel lancé mais passif)
        self.driving = False        # garde-fou update — TODO: vrai signal depuis le worker
        self._gamepad = None
        self._control = None

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        self._start_auto()
        self._gamepad = GamepadWatcher(self._on_gamepad_connect, self._on_gamepad_disconnect)
        self._gamepad.start()
        port = self.vehicle.get("ui", {}).get("control_port", 8090)
        self._control = make_server(self, port)
        print("[core] superviseur démarré — profil=%s, contrôle :%d" % (self.profile, port))

    def shutdown(self):
        if self._gamepad:
            self._gamepad.stop()
        self.workers.stop_all()

    # --- profils / auto ----------------------------------------------------
    def _auto_worker(self):
        return self.profiles_cfg["profiles"][self.profile]["auto_worker"]

    def _start_auto(self):
        with self._lock:
            if self.mode == "auto":
                self.workers.start(self._auto_worker())

    def select_profile(self, profile):
        with self._lock:
            if profile not in self.profiles_cfg["profiles"]:
                return {"ok": False, "error": "profil inconnu: %s" % profile}
            self.workers.stop(self._auto_worker())
            self.profile = profile
            if self.mode == "auto":
                self.workers.start(self._auto_worker())
            return {"ok": True, "profile": self.profile}

    # --- manette : armement + prise de main explicite ----------------------
    def _on_gamepad_connect(self):
        with self._lock:
            self.manual_armed = True
            self.workers.start("manual")  # PASSIF : lancé mais ne prend pas la main
            print("[core] manette détectée — manuel ARMÉ (passif)")

    def _on_gamepad_disconnect(self):
        with self._lock:
            self.manual_armed = False
            self.workers.stop("manual")
            if self.mode == "manual":
                self.mode = "auto"
                self._start_auto()
            print("[core] manette retirée — manuel désarmé")

    def takeover(self):
        with self._lock:
            if not self.manual_armed:
                return {"ok": False, "error": "pas de manette"}
            self.workers.stop(self._auto_worker())  # coupe l'auto
            self.mode = "manual"
            print("[core] PRISE DE MAIN manuelle — auto coupé")
            return {"ok": True, "mode": self.mode}

    def release(self):
        with self._lock:
            self.mode = "auto"
            self._start_auto()
            print("[core] main rendue — retour auto")
            return {"ok": True, "mode": self.mode}

    # --- UI à la demande ---------------------------------------------------
    def ui_connect(self):
        with self._lock:
            self.workers.start("stream_ui")
            return {"ok": True, "stream": "on"}

    def ui_disconnect(self):
        with self._lock:
            self.workers.stop("stream_ui")
            return {"ok": True, "stream": "off"}

    # --- update (squelette) ------------------------------------------------
    def request_update(self):
        with self._lock:
            if self.driving:
                return {"ok": False, "error": "refusé : conduite en cours"}
            # TODO: git pull (code) + deploy/sync-services.sh (units) + restart workers
            return {"ok": True, "todo": "git pull + sync-services + restart (à implémenter)"}

    # --- status ------------------------------------------------------------
    def status(self):
        with self._lock:
            return {
                "profile": self.profile,
                "mode": self.mode,
                "manual_armed": self.manual_armed,
                "driving": self.driving,
                "workers_running": self.workers.running(),
            }

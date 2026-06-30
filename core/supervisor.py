"""Superviseur : charge la config, reste INERTE au boot, gère manette / UI / update / profils.

Le core n'implémente pas la perception ni le contrôle (ce sont les workers) ; il choisit
lesquels tournent (profil + contexte) et les spawn/kill — il gouverne les comportements
sans les calculer. Au boot : aucun worker de conduite ; un worker ne démarre que sur trigger
explicite (manette allumée OU profil lancé). Voir docs/CORE_DESIGN.md et docs/schemas/.

TODO (pas encore là) : surveillance de liveness des workers. Si un worker meurt seul (ex.
auto en route), `mode` ne se resynchronise pas et rien ne le relance — il faudra une boucle
de santé.
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
        self.profile = profile or self.profiles_cfg.get("default")  # sélectionné, PAS lancé
        self.workers = WorkerManager(self.profiles_cfg["workers"], cwd=cwd,
                                     params=self._worker_params())
        self._lock = threading.RLock()
        self.mode = "idle"            # "idle" | "auto" | "manual"
        self.manual_armed = False     # manette présente
        self._profile_launched = False
        self._gamepad = None
        self._control = None

    def _worker_params(self):
        """Params véhicule interpolés dans les argv des workers — source unique vehicle.json."""
        v = self.vehicle
        return {
            "max_duty": v.get("control", {}).get("max_duty", 0.05),
            "teleop_max_duty": v.get("control", {}).get("teleop_max_duty", 0.12),
            "steering_max": v.get("control", {}).get("steering_max", 1.0),
            "stream_port": v.get("ui", {}).get("stream_port", 5601),
        }

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        # INERTE : aucun worker de conduite au boot (voir docs/CORE_DESIGN.md).
        idle_off = self.vehicle.get("gamepad", {}).get("idle_off_s", 30.0)
        self._gamepad = GamepadWatcher(self._on_gamepad_connect, self._on_gamepad_disconnect,
                                       idle_off=idle_off)
        self._gamepad.start()
        port = self.vehicle.get("ui", {}).get("control_port", 8090)
        self._control = make_server(self, port)
        print("[core] superviseur démarré — INERTE (profil sélectionné=%s), contrôle :%d"
              % (self.profile, port))

    def shutdown(self):
        if self._gamepad:
            self._gamepad.stop()
        self.workers.stop_all()

    # --- profils / auto (trigger explicite : terminal / UI) ----------------
    def _auto_worker(self):
        return self.profiles_cfg["profiles"][self.profile]["auto_worker"]

    def launch_profile(self, profile):
        """Lance le worker auto d'un profil."""
        with self._lock:
            if profile not in self.profiles_cfg["profiles"]:
                return {"ok": False, "error": "profil inconnu: %s" % profile}
            if self._profile_launched:
                self.workers.stop(self._auto_worker())
            self.profile = profile
            self._profile_launched = True
            self.mode = "auto"
            self.workers.start(self._auto_worker())
            print("[core] profil lancé : %s (auto)" % self.profile)
            return {"ok": True, "profile": self.profile, "mode": self.mode}

    def stop_profile(self):
        """Arrête le worker auto → retour inerte."""
        with self._lock:
            if self._profile_launched:
                self.workers.stop(self._auto_worker())
            self._profile_launched = False
            if self.mode == "auto":
                # invariant : si la manette est armée, on retombe en manuel, pas en idle
                self.mode = "manual" if self.manual_armed else "idle"
            print("[core] profil arrêté — mode=%s" % self.mode)
            return {"ok": True, "mode": self.mode}

    # --- manette : déclenche le manuel, prise de main explicite ------------
    def _on_gamepad_connect(self):
        with self._lock:
            self.manual_armed = True
            self.workers.start("manual")
            if self.mode == "idle":
                self.mode = "manual"          # rien d'autre ne tourne -> manuel actif
            # si mode == auto : manuel PASSIF, attend la prise de main explicite
            print("[core] manette détectée — worker manuel lancé (mode=%s)" % self.mode)

    def _on_gamepad_disconnect(self):
        with self._lock:
            self.manual_armed = False
            self.workers.stop("manual")
            if self.mode == "manual":
                self._resume_after_manual()
            print("[core] manette retirée — mode=%s" % self.mode)

    def takeover(self):
        with self._lock:
            if not self.manual_armed:
                return {"ok": False, "error": "pas de manette"}
            if self.mode == "auto":
                self.workers.stop(self._auto_worker())  # coupe l'auto (profil reste lancé)
            self.mode = "manual"
            print("[core] PRISE DE MAIN manuelle")
            return {"ok": True, "mode": self.mode}

    def release(self):
        with self._lock:
            self._resume_after_manual()
            print("[core] main rendue — mode=%s" % self.mode)
            return {"ok": True, "mode": self.mode}

    def _resume_after_manual(self):
        """Sortie du manuel : reprend l'auto si un profil est lancé ; sinon manuel si la manette
        est encore armée ; sinon inerte. Invariant : manual_armed ⇒ mode ∈ {auto, manual}."""
        if self._profile_launched:
            self.mode = "auto"
            self.workers.start(self._auto_worker())
        elif self.manual_armed:
            self.mode = "manual"
        else:
            self.mode = "idle"

    # --- UI à la demande (stream lourd → seulement quand connecté) ----------
    def ui_connect(self):
        with self._lock:
            self.workers.start("stream_ui")
            return {"ok": True, "stream": "on"}

    def ui_disconnect(self):
        with self._lock:
            self.workers.stop("stream_ui")
            return {"ok": True, "stream": "off"}

    # --- update (squelette : NON implémenté) -------------------------------
    def request_update(self):
        with self._lock:
            if self.mode != "idle":
                return {"ok": False, "error": "refusé : conduite en cours (mode=%s)" % self.mode}
            # TODO: git pull (code) + deploy/sync-services.sh (units) + restart workers.
            # ok=False tant que ce n'est pas réel — ne pas laisser une UI croire au succès.
            return {"ok": False, "error": "not_implemented",
                    "todo": "git pull + sync-services + restart"}

    # --- status ------------------------------------------------------------
    def status(self):
        with self._lock:
            return {
                "profile": self.profile,
                "profile_launched": self._profile_launched,
                "mode": self.mode,
                "manual_armed": self.manual_armed,
                "driving": self.mode != "idle",
                "workers_running": self.workers.running(),
            }

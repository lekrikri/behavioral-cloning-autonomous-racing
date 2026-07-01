"""Drive supervisor for the home page: Play / Pause / Stop a perception profile.

Play launches the autonomous drive pipeline (inference_realcar) with the selected profile,
reading the hub. Stop kills it and forces the motor safe (VESC stop). Pause is a safety hold:
it kills the drive process and stops the motor, keeping the selected profile so Play resumes.

The hub (camera) is a separate service and keeps running across Play/Pause/Stop.
"""

import os
import subprocess
import sys
import threading
import time

from src.mask.perception_config import list_profiles, load_profile, PerceptionProfile


class Driver:
    def __init__(self, repo_root, vesc_port="/dev/ttyACM0", profile_name="classic"):
        self.repo_root = repo_root
        self.vesc_port = vesc_port
        self.lock = threading.Lock()
        self.proc = None
        self.mode = "stop"                 # stop | play | pause
        self.profile_name = profile_name

    # ── process helpers ───────────────────────────────────────────────────────
    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _vesc_stop(self):
        """Best-effort motor cut (the VESC watchdog also cuts when a process dies)."""
        try:
            if self.repo_root not in sys.path:
                sys.path.insert(0, self.repo_root)
            from src.control.vesc_interface import VESCInterface
            v = VESCInterface(port=self.vesc_port)
            v.stop(); v.close()
        except Exception as e:
            print("[drive] VESC stop best-effort échoué : %s" % e)

    # ── public API ────────────────────────────────────────────────────────────
    def set_profile(self, name):
        with self.lock:
            if name in list_profiles() or name == "classic":
                self.profile_name = name
            return self.status_locked()

    def play(self):
        with self.lock:
            self._kill()
            try:
                prof = load_profile(self.profile_name)
            except (FileNotFoundError, OSError):
                prof = PerceptionProfile()
            env = dict(os.environ)
            env["OPENBLAS_CORETYPE"] = "ARMV8"        # sinon SIGILL numpy sur le Tegra
            cmd = [sys.executable, "-m", "src.control.inference_realcar",
                   "--profile", self.profile_name, "--source", "hub",
                   "--brain", prof.brain,
                   "--model", prof.model, "--duty-max", str(prof.duty_max)]
            try:
                self.proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env)
                time.sleep(0.3)
                if self.proc.poll() is not None:      # mort immédiate = échec lancement
                    print("[drive] play n'a pas démarré (code %s)" % self.proc.returncode)
                    self.proc = None
                    self.mode = "stop"
                else:
                    self.mode = "play"
            except (OSError, FileNotFoundError) as e:
                print("[drive] lancement play impossible : %s" % e)
                self.mode = "stop"
            return self.status_locked()

    def pause(self):
        with self.lock:
            self._kill()
            self._vesc_stop()
            self.mode = "pause"
            return self.status_locked()

    def stop(self):
        with self.lock:
            self._kill()
            self._vesc_stop()
            self.mode = "stop"
            return self.status_locked()

    def status(self):
        with self.lock:
            return self.status_locked()

    def status_locked(self):
        alive = self.proc is not None and self.proc.poll() is None
        return {"mode": self.mode, "alive": alive, "profile": self.profile_name,
                "profiles": list_profiles() or ["classic"]}

    def shutdown(self):
        with self.lock:
            self._kill()
            self._vesc_stop()

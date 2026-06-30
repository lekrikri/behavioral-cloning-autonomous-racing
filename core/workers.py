"""Cycle de vie des workers (subprocess) : spawn / kill, un worker par nom."""
import os
import subprocess
import time


class Worker:
    def __init__(self, name, argv, env=None, cwd=None):
        self.name = name
        self.argv = list(argv)
        self.env = dict(env or {})
        self.cwd = cwd
        self.proc = None

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.is_alive():
            return
        env = dict(os.environ)
        env.update(self.env)
        self.proc = subprocess.Popen(self.argv, env=env, cwd=self.cwd)

    def stop(self, timeout=5.0):
        if not self.is_alive():
            self.proc = None
            return
        self.proc.terminate()
        t0 = time.time()
        while self.proc.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc = None


class WorkerManager:
    """Lance/arrête des workers nommés depuis des specs {name: {argv, env}}."""

    def __init__(self, specs, cwd=None, params=None):
        self._specs = specs
        self._cwd = cwd
        self._params = params or {}  # params véhicule injectés dans les argv ({hub_port}, ...)
        self._running = {}  # name -> Worker

    def _resolve(self, argv):
        """Interpole les params véhicule dans l'argv → source unique (vehicle.json)."""
        return [a.format(**self._params) if "{" in a else a for a in argv]

    def start(self, name):
        cur = self._running.get(name)
        if cur and cur.is_alive():
            return cur
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError("worker inconnu: %s" % name)
        w = Worker(name, self._resolve(spec["argv"]), spec.get("env"), cwd=self._cwd)
        w.start()
        self._running[name] = w
        return w

    def stop(self, name):
        w = self._running.pop(name, None)
        if w:
            w.stop()

    def is_running(self, name):
        w = self._running.get(name)
        return bool(w and w.is_alive())

    def running(self):
        return [n for n, w in self._running.items() if w.is_alive()]

    def stop_all(self):
        for name in list(self._running):
            self.stop(name)

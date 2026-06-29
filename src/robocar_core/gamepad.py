"""Surveillance manette : présence d'un /dev/input/js* -> arme le manuel (ne bascule pas)."""
import glob
import threading
import time


class GamepadWatcher(threading.Thread):
    def __init__(self, on_connect, on_disconnect, poll=1.0):
        super().__init__(daemon=True)
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._poll = poll
        self._present = False
        self._stop = threading.Event()

    @staticmethod
    def _detect():
        return bool(glob.glob("/dev/input/js*"))

    def run(self):
        while not self._stop.is_set():
            present = self._detect()
            if present and not self._present:
                self._present = True
                self._on_connect()
            elif not present and self._present:
                self._present = False
                self._on_disconnect()
            time.sleep(self._poll)

    def stop(self):
        self._stop.set()

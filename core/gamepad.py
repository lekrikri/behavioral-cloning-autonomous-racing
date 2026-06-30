"""Surveillance manette par ACTIVITÉ (pas par présence du device).

Le dongle USB du F710 crée `/dev/input/js*` en permanence (manette ON ou OFF), et le driver
joystick envoie des events `JS_EVENT_INIT` synthétiques à l'ouverture. Donc la simple présence
du device ne dit rien. On considère la manette « allumée » uniquement quand elle émet des events
RÉELS (non-init = mouvement/bouton). Disarm si plus d'activité pendant `idle_off` s (ou si le
device disparaît).
"""
import glob
import os
import threading
import time

_JS_EVENT_SIZE = 8          # struct js_event: u32 time, i16 value, u8 type, u8 number
_JS_EVENT_INIT = 0x80       # bit positionné sur les events synthétiques d'init


def _is_real_event(buf):
    """True si l'event joystick n'est PAS un init synthétique."""
    return len(buf) >= _JS_EVENT_SIZE and not (buf[6] & _JS_EVENT_INIT)


class GamepadWatcher(threading.Thread):
    def __init__(self, on_connect, on_disconnect, poll=0.3, idle_off=30.0):
        super().__init__(daemon=True)
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._poll = poll
        self._idle_off = idle_off
        self._stop = threading.Event()

    @staticmethod
    def _device():
        devs = sorted(glob.glob("/dev/input/js*"))
        return devs[0] if devs else None

    def run(self):
        fd = None
        active = False
        last_event = 0.0
        while not self._stop.is_set():
            if fd is None:                                   # glob UNIQUEMENT device fermé
                dev = self._device()
                if dev is None:
                    active = self._set(active, False)
                    time.sleep(self._poll)
                    continue
                try:
                    fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
                except OSError:
                    time.sleep(self._poll)
                    continue
                self._drain(fd)          # jette le backlog (init + events périmés) à l'ouverture
                last_event = time.time()
                time.sleep(self._poll)
                continue                 # on n'arme que sur des events postérieurs à l'ouverture

            got_real = self._read_real(fd)
            if got_real is None:                             # device arraché (read OSError)
                fd = self._close(fd)
                active = self._set(active, False)
                time.sleep(self._poll)
                continue

            now = time.time()
            if got_real:
                last_event = now
                active = self._set(active, True)
            elif active and (now - last_event) > self._idle_off:
                active = self._set(active, False)
            time.sleep(self._poll)
        self._close(fd)

    @staticmethod
    def _read_real(fd):
        """Vide les events disponibles (buffers multi-events) ; True s'il y a eu un event RÉEL,
        False sinon, None si le device a disparu (OSError sur read)."""
        got = False
        try:
            while True:
                data = os.read(fd, _JS_EVENT_SIZE * 32)
                if not data:
                    break
                for off in range(0, len(data) - _JS_EVENT_SIZE + 1, _JS_EVENT_SIZE):
                    if not (data[off + 6] & _JS_EVENT_INIT):
                        got = True
                if len(data) < _JS_EVENT_SIZE * 32:
                    break                # buffer vidé
        except BlockingIOError:
            pass
        except OSError:
            return None
        return got

    def _set(self, active, new):
        """Applique la transition et déclenche le callback si l'état change."""
        if new and not active:
            self._on_connect()
        elif not new and active:
            self._on_disconnect()
        return new

    @staticmethod
    def _drain(fd):
        """Lit et jette tout ce qui est déjà en buffer (backlog, gros buffers) sans l'évaluer."""
        try:
            while len(os.read(fd, _JS_EVENT_SIZE * 32)) == _JS_EVENT_SIZE * 32:
                pass
        except (BlockingIOError, OSError):
            pass

    @staticmethod
    def _close(fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        return None

    def stop(self):
        self._stop.set()

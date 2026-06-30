"""
hub.py — Propriétaire unique de l'OAK-D : capture → mémoire partagée (zéro-copie).

L'OAK-D parle en XLink (mono-client) : un seul process peut l'ouvrir. Le hub l'ouvre
UNE fois et publie ses flux dans des rings /dev/shm ; les consommateurs LOCAUX lisent
en ZÉRO-COPIE (vue numpy directe sur la mmap — voir src/cam/shm.py). Pas de socket
réseau : tous les consommateurs sont locaux (le PC passe par le HTTP de mask_stream,
lui-même client local du hub). Résout l'exclusivité caméra → preview + conduite + IMU
simultanés sans copie par client (décisif sur le Nano memory-bandwidth-bound).

Canaux = régions SHM séparées (l'IMU latence-critique n'est jamais coincé derrière une
grosse frame) :
  /dev/shm/robocar_cam_color : BGR HxWx3 uint8
  /dev/shm/robocar_cam_depth : depth HxW uint16 (mm)             [si --depth]
  /dev/shm/robocar_cam_imu   : (1,3) float32 [gyro_x, gyro_y, gyro_z]   [si IMU]

── Serveur (Jetson, service robocar-cam-hub) ──
  OPENBLAS_CORETYPE=ARMV8 python3 -m src.cam.hub [--width 640] [--height 320] [--depth] [--no-imu]

── Consommateur ──
  from src.cam.hub import FrameClient
  c = FrameClient()                 # défaut : flux couleur
  bgr = c.getCvFrame()              # dernière frame BGR (vue zéro-copie, read-only)
"""

import argparse
import os
import time

import numpy as np

from src.cam.shm import ShmRingReader, ShmRingWriter

SHM_COLOR = "robocar_cam_color"
SHM_DEPTH = "robocar_cam_depth"
SHM_IMU   = "robocar_cam_imu"

STREAM_COLOR = 0
STREAM_DEPTH = 1

SERVICE_NAME = "robocar-cam-hub.service"


# ───────────────────────────── Côté consommateur ─────────────────────────────

def shm_is_up(name=SHM_COLOR, settle=0.3):
    """True si le hub publie ACTIVEMENT sur la région `name` (existe ET seq avance).

    Plus fort qu'un simple test de présence : un hub gelé (région présente mais seq figé)
    est détecté comme down. Le hub publie à la cadence caméra même sans consommateur.
    """
    if not os.path.exists(os.path.join("/dev/shm", name)):
        return False
    try:
        r = ShmRingReader(name)
    except Exception:
        return False
    try:
        s0 = r.latest_seq()
        time.sleep(settle)
        s1 = r.latest_seq()
    finally:
        r.close()
    return s1 > 0 and s1 != s0


def ensure_hub_or_prompt(stream=SHM_COLOR, service=SERVICE_NAME, **_legacy):
    """Vérifie que le hub publie ; sinon avertit (anormal) et indique comment le relancer.

    Le hub tourne en permanence comme service SYSTÈME ('robocar-cam-hub', Restart=always,
    lancé au boot — voir docs/SERVICES.md). Son absence est un défaut de setup ; comme c'est
    un service système, le relancer demande des privilèges → le client ne le démarre pas à la
    place de l'utilisateur, il affiche la commande exacte. `**_legacy` absorbe les anciens
    kwargs (host/port) du transport TCP, désormais ignorés (transport = mémoire partagée).
    Retourne True si le hub publie, False sinon (le client doit alors abandonner proprement).
    """
    if shm_is_up(stream):
        return True
    print(f"\n[hub] /!\\ le hub ne publie pas sur /dev/shm/{stream} — ce n'est PAS normal.")
    print(f"[hub]     Le hub doit tourner en permanence (service système '{service}', Restart=always).")
    print(f"[hub]     Relancer :   sudo systemctl restart {service}")
    print(f"[hub]     Diagnostic : systemctl status {service} ; journalctl -u {service} -e")
    print(f"[hub]     Setup :      voir docs/SERVICES.md")
    return False


class FrameClient:
    """Client SHM du hub : lit la dernière frame d'un flux en zéro-copie. API getCvFrame()
    compatible avec l'ancien client TCP (la vue est read-only, comme l'ancien np.frombuffer).
    """

    def __init__(self, stream=SHM_COLOR, timeout=5.0, **_legacy):
        # _legacy absorbe host=/port= de l'ancien client TCP (ignorés : transport SHM).
        self.name = stream
        self.timeout = timeout
        self.reader = None
        self._last = 0

    def connect(self):
        """Attend que la région SHM existe (le hub peut démarrer après le client)."""
        deadline = time.time() + self.timeout
        path = os.path.join("/dev/shm", self.name)
        while time.time() < deadline:
            if os.path.exists(path):
                self.reader = ShmRingReader(self.name)
                return
            time.sleep(0.05)
        raise ConnectionError(f"région SHM absente : {self.name}")

    def get(self, copy=False):
        """Bloque jusqu'à une frame FRAÎCHE (seq avancé) ; retourne (seq, ndarray).

        Par défaut zéro-copie (vue read-only) : à consommer promptement (le writer relape
        après slot_count frames). copy=True pour garder la frame longtemps.
        """
        if self.reader is None:
            self.connect()
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            out = self.reader.read(copy=copy)
            if out is not None and out[0]["seq"] != self._last:
                self._last = out[0]["seq"]
                return out[0]["seq"], out[1]
            time.sleep(0.001)
        raise ConnectionError(f"aucune frame du hub (timeout {self.timeout}s) — hub mort ?")

    def getCvFrame(self):
        return self.get()[1]

    def latest(self, copy=False):
        """Dernière frame dispo SANS attendre une nouvelle (None si rien encore publié).
        Pour les consommateurs qui ne veulent pas bloquer sur ce flux (ex. depth en fusion)."""
        if self.reader is None:
            self.connect()
        out = self.reader.read(copy=copy)
        return out[1] if out is not None else None

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None


# ───────────────────────────── Côté serveur (hub) ─────────────────────────────

class _Publisher:
    """Gère les ShmRingWriter par flux ; crée chaque ring paresseusement à la 1re frame
    (slot_bytes = taille réelle de la frame → pas besoin de connaître la résolution avant)."""

    def __init__(self, slot_count=4):
        self.slot_count = slot_count
        self.writers = {}

    def publish(self, name, arr, stream_id=0):
        w = self.writers.get(name)
        if w is None:
            w = ShmRingWriter(name, slot_bytes=arr.nbytes, slot_count=self.slot_count)
            self.writers[name] = w
            print(f"[hub] région /dev/shm/{name} ({arr.shape} {arr.dtype}, {arr.nbytes} o/slot ×{self.slot_count})")
        w.write(arr, stream_id=stream_id)

    def closeall(self):
        for w in self.writers.values():
            w.unlink()
        self.writers.clear()


def _build_pipeline(dai, width, height, fps, want_imu, want_depth):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    # Plein capteur : 1080P + ISP downscale 1/3 → ~640x360 (FOV complet, mêmes params
    # que le hub track-mapping validé). setPreviewSize recadre ensuite à width×height.
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(1, 3)
    cam.setPreviewSize(width, height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(fps)
    xrgb = pipeline.create(dai.node.XLinkOut)
    xrgb.setStreamName("rgb")
    cam.preview.link(xrgb.input)

    if want_imu:
        # BMI270 : GYROSCOPE_RAW uniquement sur OAK-D Lite (6 axes, pas de ROTATION_VECTOR).
        # L'IMU est sur SPI, indépendant du VPU → concurrent à l'inférence on-cam (doc Luxonis).
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.GYROSCOPE_RAW], 100)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        ximu = pipeline.create(dai.node.XLinkOut)
        ximu.setStreamName("imu")
        imu.out.link(ximu.input)

    if want_depth:
        # Config depth canonique partagée avec create_depthai_pipeline → DepthToRays calibré
        # reste valide quelle que soit la source (hub ou device direct).
        from src.mask.depth_rays import add_stereo_depth
        stereo = add_stereo_depth(pipeline, dai)
        xd = pipeline.create(dai.node.XLinkOut)
        xd.setStreamName("depth")
        stereo.depth.link(xd.input)

    return pipeline


def _capture(pub, width, height, fps, want_imu, want_depth, running):
    """Boucle de capture avec reconnexion (l'OAK peut tomber sur l'USB au boot/à chaud)."""
    import depthai as dai
    gyro = np.zeros((1, 3), dtype=np.float32)
    attempt = 0
    while running[0]:
        try:
            attempt += 1
            pipeline = _build_pipeline(dai, width, height, fps, want_imu, want_depth)
            with dai.Device(pipeline) as device:
                q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
                q_imu = device.getOutputQueue("imu", maxSize=50, blocking=False) if want_imu else None
                q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False) if want_depth else None
                print(f"[hub] OAK-D OK (essai {attempt}) — publie en SHM")
                attempt = 0
                while running[0]:
                    pub.publish(SHM_COLOR, q_rgb.get().getCvFrame(), STREAM_COLOR)
                    if q_imu is not None:
                        d = q_imu.tryGet()
                        if d:
                            for pkt in d.packets:
                                g = pkt.gyroscope
                                gyro[0, 0], gyro[0, 1], gyro[0, 2] = g.x, g.y, g.z
                            pub.publish(SHM_IMU, gyro)
                    if q_depth is not None:
                        dd = q_depth.tryGet()
                        if dd is not None:
                            pub.publish(SHM_DEPTH, dd.getFrame(), STREAM_DEPTH)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[hub] OAK-D crash ({exc}) — reconnexion dans 5s…")
            time.sleep(5)
    print("[hub] capture terminée")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--depth",   action="store_true", help="publier aussi le flux depth (uint16 mm)")
    parser.add_argument("--no-imu",  action="store_true", help="ne pas publier l'IMU")
    args = parser.parse_args()

    pub = _Publisher()
    running = [True]
    print(f"[hub] démarrage SHM-only — color {args.width}x{args.height}@{args.fps}"
          f"{' +depth' if args.depth else ''}{'' if args.no_imu else ' +imu'}")
    try:
        _capture(pub, args.width, args.height, args.fps,
                 want_imu=not args.no_imu, want_depth=args.depth, running=running)
    except KeyboardInterrupt:
        print("\n[hub] arrêt")
    finally:
        running[0] = False
        pub.closeall()  # libère /dev/shm/robocar_cam_*


if __name__ == "__main__":
    main()

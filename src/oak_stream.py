"""
oak_stream.py — Stream H.264 bas-latence de l'OAK-D Lite vers un PC (RTP/UDP).

Encodage matériel SUR la caméra (nœud VideoEncoder du VPU Myriad X) : la Jetson ne
fait que ré-empaqueter le flux déjà compressé en RTP via gst-launch. Headless —
aucun affichage requis (contrairement à live_mask_oak.py qui ouvre une fenêtre).

Usage (sur la Jetson) :
  python src/oak_stream.py --host <IP_DU_PC> [--port 5000] [--fps 30] [--bitrate 4000]

Réception (sur le PC) :
  gst-launch-1.0 -v udpsrc port=5000 \
    caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
    ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false

Mode TCP (--tcp) — quand le sens Jetson -> PC est bloqué (nœud Tailscale partagé) :
  PC  : ssh -L 5000:127.0.0.1:5000 <jetson>          # tunnel
  Jet : python src/oak_stream.py --tcp --host 127.0.0.1
  PC  : gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=5000 \
          ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
"""

import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")  # depthai importe numpy → garde anti-SIGILL

import argparse
import subprocess

import depthai as dai


def build_pipeline(fps: int, bitrate_kbps: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(fps)

    enc = pipeline.create(dai.node.VideoEncoder)
    enc.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    enc.setBitrateKbps(bitrate_kbps)
    cam.video.link(enc.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("h264")
    enc.bitstream.link(xout.input)

    return pipeline


def main():
    ap = argparse.ArgumentParser(description="Stream OAK-D Lite H.264 -> RTP/UDP ou TCP")
    ap.add_argument("--host", default=None,
                    help="UDP: IP du PC recepteur (requis). TCP: adresse de bind (defaut 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4000, help="kbps")
    ap.add_argument("--tcp", action="store_true",
                    help="Sert le H.264 brut en TCP (le PC se connecte au lieu de recevoir en push). "
                         "A combiner avec 'ssh -L' pour traverser un noeud Tailscale partage, "
                         "ou le sens Jetson -> PC est bloque.")
    args = ap.parse_args()

    if args.tcp:
        bind = args.host or "0.0.0.0"
        gst = [
            "gst-launch-1.0", "-q",
            "fdsrc", "!", "h264parse", "!",
            "tcpserversink", "host=%s" % bind, "port=%d" % args.port,
        ]
        target = "TCP serveur %s:%d (le PC se connecte)" % (bind, args.port)
    else:
        if not args.host:
            ap.error("--host est requis en mode UDP")
        # config-interval=1 : ré-émet SPS/PPS chaque seconde → un récepteur qui rejoint
        # en cours de route peut décoder (UDP ne retransmet pas les headers initiaux).
        gst = [
            "gst-launch-1.0", "-q",
            "fdsrc", "!", "h264parse", "!",
            "rtph264pay", "config-interval=1", "pt=96", "!",
            "udpsink", "host=%s" % args.host, "port=%d" % args.port, "sync=false",
        ]
        target = "RTP/UDP -> %s:%d" % (args.host, args.port)
    gst_proc = subprocess.Popen(gst, stdin=subprocess.PIPE)

    with dai.Device(build_pipeline(args.fps, args.bitrate)) as device:
        q = device.getOutputQueue("h264", maxSize=30, blocking=True)
        print("Stream H.264 %dfps %dkbps  [%s]  (Ctrl-C pour arreter)"
              % (args.fps, args.bitrate, target))
        try:
            while True:
                gst_proc.stdin.write(q.get().getData().tobytes())
        except KeyboardInterrupt:
            print("\nArret.")
        finally:
            try:
                gst_proc.stdin.close()
            except Exception:
                pass
            gst_proc.wait()


if __name__ == "__main__":
    main()

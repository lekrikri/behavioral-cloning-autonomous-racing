"""
camera_stream.py — Streaming MJPEG/H.264 depuis le Jetson Nano (depthai 2.x)

Usage recommandé (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 src/camera_stream.py --serve --dst-port 5600

Usage VLC (Windows/Linux) :
  ouvrir "tcp://192.168.0.100:5600" dans VLC (Média > Ouvrir un flux réseau)

Options :
  --serve        : mode serveur TCP (VLC se connecte à tcp://JETSON_IP:PORT)
  --dst-ip       : IP du PC récepteur en mode UDP (défaut: broadcast)
  --dst-port     : Port (défaut: 5600)
  --codec        : mjpeg (défaut, faible latence) | h264 (meilleure compression)
  --width/height : Résolution (défaut: 416x312)
  --fps          : FPS (défaut: 30)
  --bitrate      : Bitrate H.264 kbps (défaut: 8000)
  --record       : Enregistrement local (ex: /tmp/cam.h264)
  --preview      : Preview local (nécessite écran)

Notes :
  - USB2 forcé (maxUsbSpeed=HIGH) pour stabilité alimentation OAK-D sur Jetson Nano
  - Reconnexion auto exponentielle si X_LINK_ERROR (crash alimentation USB)
"""

import argparse
import subprocess
import sys
import threading
import time

try:
    import depthai as dai
except ImportError:
    print("[camera_stream] depthai non installe : pip install depthai")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dst-ip",   default="255.255.255.255",
                   help="IP du PC récepteur (ou broadcast) — ignoré en mode --serve")
    p.add_argument("--dst-port", type=int, default=5600,
                   help="Port UDP destination / port TCP serveur")
    p.add_argument("--serve",    action="store_true",
                   help="Mode serveur TCP : VLC se connecte à tcp://JETSON_IP:PORT (traverse les firewalls)")
    p.add_argument("--width",    type=int, default=416)
    p.add_argument("--height",   type=int, default=312)
    p.add_argument("--fps",      type=int, default=20,
                   help="FPS (défaut: 20 — réduit pour stabilité alimentation USB)")
    p.add_argument("--bitrate",  type=int, default=8000,
                   help="Bitrate kbps (H.264 ou MJPEG)")
    p.add_argument("--codec",    choices=["h264", "mjpeg"], default="mjpeg",
                   help="mjpeg = faible latence (recommandé) | h264 = meilleure compression")
    p.add_argument("--record",   default=None,
                   help="Chemin fichier enregistrement local (ex: /tmp/cam.h264)")
    p.add_argument("--preview",  action="store_true",
                   help="Affichage preview local (nécessite display)")
    return p.parse_args()


def build_pipeline(args):
    pipeline = dai.Pipeline()
    # Réduit la taille des chunks XLink → latence USB plus basse
    pipeline.setXLinkChunkSize(0)

    # Caméra couleur
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(args.width, args.height)
    cam.setVideoSize(args.width, args.height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(args.fps)

    # Encodeur (MJPEG par défaut = faible latence, ou H.264)
    encoder = pipeline.create(dai.node.VideoEncoder)
    if args.codec == "mjpeg":
        encoder.setDefaultProfilePreset(
            args.fps,
            dai.VideoEncoderProperties.Profile.MJPEG,
        )
        encoder.setQuality(80)  # 80 = bon compromis taille/qualité
    else:
        encoder.setDefaultProfilePreset(
            args.fps,
            dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        encoder.setBitrateKbps(args.bitrate)
        encoder.setKeyframeFrequency(args.fps)  # keyframe chaque seconde

    cam.video.link(encoder.input)

    # Output encodé — limite débit XLink pour soulager l'alimentation USB
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("encoded")
    xout.setFpsLimit(args.fps)
    encoder.bitstream.link(xout.input)

    # Preview optionnel (non encodé)
    xout_preview = None
    if args.preview:
        xout_preview = pipeline.create(dai.node.XLinkOut)
        xout_preview.setStreamName("preview")
        xout_preview.setFpsLimit(10)  # preview basse fréquence
        cam.preview.link(xout_preview.input)

    return pipeline, xout_preview is not None


def _gst_proc(cmd):
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None


def gst_sink(dst_ip, dst_port, codec):
    """Mode push UDP/RTP vers le PC récepteur (faible latence, pas de retransmission)."""
    # queue leaky : si le débit est trop lent on jette les vieilles frames
    leaky_queue = ["queue", "max-size-buffers=1", "leaky=downstream"]
    if codec == "mjpeg":
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "jpegparse",
               "!"] + leaky_queue + [
               "!", "rtpjpegpay", "pt=26",
               "!", "udpsink", "host=%s" % dst_ip, "port=%d" % dst_port,
               "sync=false", "async=false"]
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "h264parse",
               "!"] + leaky_queue + [
               "!", "rtph264pay", "config-interval=1", "pt=96",
               "!", "udpsink", "host=%s" % dst_ip, "port=%d" % dst_port,
               "sync=false", "async=false"]
    return _gst_proc(cmd)


def gst_server(port, codec):
    """Mode serveur TCP — VLC se connecte à tcp://JETSON_IP:PORT."""
    # queue leaky : on jette les vieilles frames si le client est lent (pas d'accumulation)
    leaky_queue = ["queue", "max-size-buffers=2", "leaky=downstream"]
    common_sink = [
        "tcpserversink", "host=0.0.0.0", "port=%d" % port,
        "sync=false", "async=false", "recover-policy=keyframe",
        "buffers-max=2", "buffers-soft-max=1",  # drop vieilles frames → latence réduite
    ]
    if codec == "mjpeg":
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "jpegparse",
               "!"] + leaky_queue + [
               "!", "mpegtsmux",
               "!"] + common_sink
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "h264parse",
               "!"] + leaky_queue + [
               "!", "mpegtsmux",
               "!"] + common_sink
    return _gst_proc(cmd)


def run(args):
    pipeline, has_preview = build_pipeline(args)

    codec_info = "MJPEG q=85" if args.codec == "mjpeg" else "H.264 %dkbps" % args.bitrate
    print("[camera_stream] %dx%d @ %dfps | %s" %
          (args.width, args.height, args.fps, codec_info))

    if args.serve:
        print("[camera_stream] Mode SERVEUR TCP port %d" % args.dst_port)
        print("[camera_stream] VLC : ouvrir tcp://192.168.0.100:%d" % args.dst_port)
        gst_proc = gst_server(args.dst_port, args.codec)
    else:
        print("[camera_stream] Destination RTP : udp://%s:%d" % (args.dst_ip, args.dst_port))
        gst_proc = gst_sink(args.dst_ip, args.dst_port, args.codec)

    if gst_proc is None:
        print("[camera_stream] ERREUR: gst-launch-1.0 introuvable.")
        sys.exit(1)

    record_file = open(args.record, "wb") if args.record else None
    if record_file:
        print("[camera_stream] Enregistrement local -> %s" % args.record)

    frame_count = 0
    t0 = time.time()

    try:
        import cv2
    except ImportError:
        cv2 = None

    try:
        with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
            q_enc  = device.getOutputQueue("encoded", maxSize=4, blocking=False)
            q_prev = device.getOutputQueue("preview", maxSize=4, blocking=False) if has_preview else None

            print("[camera_stream] Streaming... Ctrl+C pour arrêter")

            while True:
                pkt  = q_enc.get()
                data = pkt.getData()

                try:
                    gst_proc.stdin.write(data.tobytes())
                    gst_proc.stdin.flush()
                except BrokenPipeError:
                    print("\n[camera_stream] GStreamer pipe fermé — arrêt")
                    break

                if record_file:
                    record_file.write(data.tobytes())

                frame_count += 1
                if frame_count % (args.fps * 5) == 0:
                    elapsed = time.time() - t0
                    print("[camera_stream] %d frames | %.1f fps" % (
                        frame_count, frame_count / elapsed))

                if q_prev and cv2:
                    prev_msg = q_prev.tryGet()
                    if prev_msg is not None:
                        bgr = prev_msg.getCvFrame()
                        cv2.imshow("Camera Stream (local)", bgr)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            break
    finally:
        if gst_proc and gst_proc.poll() is None:
            gst_proc.stdin.close()
            gst_proc.terminate()
            gst_proc.wait()
        if record_file:
            record_file.close()
        if cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[camera_stream] Terminé — %d frames envoyées" % frame_count)


if __name__ == "__main__":
    args = parse_args()
    attempt = 0
    while True:
        try:
            attempt += 1
            if attempt > 1:
                print("[camera_stream] Tentative %d..." % attempt)
            run(args)
            break  # sortie propre (Ctrl+C)
        except KeyboardInterrupt:
            print("\n[camera_stream] Arrêt demandé.")
            break
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e) or "Device crashed" in str(e) or "Couldn't read" in str(e):
                delay = min(5 * attempt, 30)  # 5s, 10s, 15s... max 30s
                print("[camera_stream] OAK-D crash — reconnexion dans %ds..." % delay)
                time.sleep(delay)
            else:
                raise

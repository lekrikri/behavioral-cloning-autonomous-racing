"""
camera_stream.py — Streaming MJPEG/H.264 depuis le Jetson Nano (depthai 2.x)

Usage recommandé (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve --dst-port 5600

Usage VLC (Windows/Linux) :
  ouvrir "tcp://192.168.0.100:5600" dans VLC (Média > Ouvrir un flux réseau)

Options :
  --serve        : mode serveur TCP (VLC se connecte à tcp://JETSON_IP:PORT)
  --dst-ip       : IP du PC récepteur en mode UDP (défaut: broadcast)
  --dst-port     : Port (défaut: 5600)
  --codec        : h264 (défaut, compatible VLC) | mjpeg (faible latence)
  --width/height : Résolution (défaut: 640x360)
  --fps          : FPS (défaut: 15)
  --bitrate      : Bitrate H.264 kbps (défaut: 2000)
  --record       : Enregistrement local (ex: /tmp/cam.h264)
  --preview      : Preview local (nécessite écran)

Notes :
  - usb2Mode=True forcé (API depthai 2.x) — réduit consommation OAK-D sur Jetson Nano
  - Reconnexion auto exponentielle si X_LINK_ERROR (crash alimentation USB)
  - Fix définitif instabilité OAK-D : hub USB alimenté 5V 2A+
"""

import argparse
import subprocess
import sys
import time

try:
    import depthai as dai
except ImportError:
    print("[camera_stream] depthai non installe : pip install depthai")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dst-ip",   default="255.255.255.255",
                   help="IP du PC recepteur (ou broadcast) — ignore en mode --serve")
    p.add_argument("--dst-port", type=int, default=5600,
                   help="Port UDP destination / port TCP serveur")
    p.add_argument("--serve",    action="store_true",
                   help="Mode serveur TCP : VLC se connecte a tcp://JETSON_IP:PORT")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=360)
    p.add_argument("--fps",      type=int, default=15)
    p.add_argument("--bitrate",  type=int, default=2000,
                   help="Bitrate kbps H.264 (defaut: 2000)")
    p.add_argument("--codec",    choices=["h264", "mjpeg"], default="h264",
                   help="h264 = compatible VLC via TCP | mjpeg = faible latence UDP")
    p.add_argument("--record",   default=None,
                   help="Chemin fichier enregistrement local (ex: /tmp/cam.h264)")
    p.add_argument("--preview",  action="store_true",
                   help="Affichage preview local (necessite display)")
    return p.parse_args()


def build_pipeline(args):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(args.width, args.height)
    cam.setVideoSize(args.width, args.height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(args.fps)

    encoder = pipeline.create(dai.node.VideoEncoder)
    if args.codec == "mjpeg":
        encoder.setDefaultProfilePreset(
            args.fps,
            dai.VideoEncoderProperties.Profile.MJPEG,
        )
        encoder.setQuality(85)
    else:
        encoder.setDefaultProfilePreset(
            args.fps,
            dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        encoder.setBitrateKbps(args.bitrate)
        encoder.setKeyframeFrequency(5)  # keyframe toutes les 5 frames (~333ms) → VLC decode vite

    cam.video.link(encoder.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("encoded")
    encoder.bitstream.link(xout.input)

    xout_preview = None
    if args.preview:
        xout_preview = pipeline.create(dai.node.XLinkOut)
        xout_preview.setStreamName("preview")
        cam.preview.link(xout_preview.input)

    return pipeline, xout_preview is not None


def _gst_proc(cmd):
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None


def gst_sink(dst_ip, dst_port, codec):
    """Mode push UDP/RTP vers le PC recepteur."""
    if codec == "mjpeg":
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "jpegparse",
               "!", "rtpjpegpay", "pt=26",
               "!", "udpsink", "host=%s" % dst_ip, "port=%d" % dst_port, "sync=false"]
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "h264parse",
               "!", "rtph264pay", "config-interval=1", "pt=96",
               "!", "udpsink", "host=%s" % dst_ip, "port=%d" % dst_port, "sync=false"]
    return _gst_proc(cmd)


def gst_server(port, codec):
    """Mode serveur TCP — VLC se connecte a tcp://JETSON_IP:PORT."""
    # sync-method=2 : nouveau client recoit depuis le dernier keyframe → pas de noir
    # buffers-max=60 buffers-soft-max=30 : drop si VLC trop lent (evite backpressure)
    sink = ["tcpserversink", "host=0.0.0.0", "port=%d" % port,
            "sync=false", "recover-policy=keyframe", "sync-method=2",
            "buffers-max=60", "buffers-soft-max=30"]
    if codec == "mjpeg":
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "jpegparse",
               "!", "matroskamux", "!"] + sink
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc",
               "!", "h264parse",
               "!", "mpegtsmux", "!"] + sink
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
        # usb2Mode=True : API depthai 2.x — force USB2 (reduit pic courant OAK-D)
        with dai.Device(pipeline, True) as device:
            q_enc  = device.getOutputQueue("encoded", maxSize=8, blocking=False)
            q_prev = device.getOutputQueue("preview", maxSize=4, blocking=False) if has_preview else None

            print("[camera_stream] Streaming... Ctrl+C pour arreter")

            while True:
                pkt  = q_enc.get()
                data = pkt.getData()

                try:
                    gst_proc.stdin.write(data.tobytes())
                    gst_proc.stdin.flush()
                except BrokenPipeError:
                    print("\n[camera_stream] GStreamer pipe ferme — arret")
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
        print("[camera_stream] Termine — %d frames envoyees" % frame_count)


if __name__ == "__main__":
    args = parse_args()
    attempt = 0
    while True:
        try:
            attempt += 1
            if attempt > 1:
                print("[camera_stream] Tentative %d..." % attempt)
            run(args)
            break
        except KeyboardInterrupt:
            print("\n[camera_stream] Arret demande.")
            break
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e) or "Device crashed" in str(e) or "Couldn't read" in str(e):
                delay = min(5 * attempt, 30)
                print("[camera_stream] OAK-D crash — reconnexion dans %ds..." % delay)
                time.sleep(delay)
            else:
                raise

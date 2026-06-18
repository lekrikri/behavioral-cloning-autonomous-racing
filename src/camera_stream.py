"""
camera_stream.py — Streaming H.264 RTP/UDP depuis le Jetson Nano (depthai 2.x)

Usage (Jetson Nano) :
  python3.8 src/camera_stream.py --dst-ip 192.168.0.x --dst-port 5600

Usage (PC récepteur) :
  # Linux/Mac
  gst-launch-1.0 udpsrc port=5600 caps="application/x-rtp,media=video,encoding-name=H264" \
      ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

  # Windows (avec GStreamer installé)
  gst-launch-1.0 udpsrc port=5600 ! "application/x-rtp,media=video,encoding-name=H264" \
      ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

  # VLC (le plus simple)
  vlc rtp://@:5600

Options :
  --dst-ip       : IP du PC récepteur (défaut: broadcast 255.255.255.255)
  --dst-port     : Port UDP (défaut: 5600)
  --width        : Largeur (défaut: 1280)
  --height       : Hauteur (défaut: 720)
  --fps          : FPS (défaut: 30)
  --bitrate      : Bitrate H.264 kbps (défaut: 4000)
  --record       : Enregistrer aussi en local (fichier .h264)
  --preview      : Afficher preview local (nécessite affichage connecté)
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
    p.add_argument("--width",    type=int, default=1280)
    p.add_argument("--height",   type=int, default=720)
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--bitrate",  type=int, default=4000,
                   help="Bitrate H.264 en kbps")
    p.add_argument("--record",   default=None,
                   help="Chemin fichier enregistrement local (ex: /tmp/cam.h264)")
    p.add_argument("--preview",  action="store_true",
                   help="Affichage preview local (nécessite display)")
    return p.parse_args()


def build_pipeline(args):
    pipeline = dai.Pipeline()

    # Caméra couleur
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(args.width, args.height)
    cam.setVideoSize(args.width, args.height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(args.fps)

    # Encodeur H.264
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(
        args.fps,
        dai.VideoEncoderProperties.Profile.H264_MAIN,
    )
    encoder.setBitrateKbps(args.bitrate)
    encoder.setKeyframeFrequency(args.fps * 2)  # I-frame toutes les 2s

    cam.video.link(encoder.input)

    # Output encodé (bitstream H.264)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("h264")
    encoder.bitstream.link(xout.input)

    # Preview optionnel (non encodé)
    xout_preview = None
    if args.preview:
        xout_preview = pipeline.create(dai.node.XLinkOut)
        xout_preview.setStreamName("preview")
        cam.preview.link(xout_preview.input)

    return pipeline, xout_preview is not None


def gst_sink(dst_ip, dst_port):
    """Mode push UDP RTP vers le PC récepteur."""
    cmd = [
        "gst-launch-1.0", "-q",
        "fdsrc",
        "!", "h264parse",
        "!", "rtph264pay", "config-interval=1", "pt=96",
        "!", "udpsink",
        "host=%s" % dst_ip,
        "port=%d" % dst_port,
        "sync=false",
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None


def gst_server(port):
    """Mode serveur TCP — VLC se connecte à tcp://JETSON_IP:PORT.
    Traverse tous les firewalls (connexion sortante depuis le PC).
    """
    cmd = [
        "gst-launch-1.0", "-q",
        "fdsrc",
        "!", "h264parse",
        "!", "mpegtsmux",
        "!", "tcpserversink",
        "host=0.0.0.0",
        "port=%d" % port,
        "sync=false",
        "recover-policy=keyframe",
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None


def run(args):
    pipeline, has_preview = build_pipeline(args)

    print("[camera_stream] %dx%d @ %dfps | H.264 %dkbps" %
          (args.width, args.height, args.fps, args.bitrate))

    if args.serve:
        print("[camera_stream] Mode SERVEUR TCP port %d" % args.dst_port)
        print("[camera_stream] VLC : ouvrir tcp://192.168.0.100:%d (remplace IP Jetson)" % args.dst_port)
        gst_proc = gst_server(args.dst_port)
    else:
        print("[camera_stream] Destination RTP : udp://%s:%d" % (args.dst_ip, args.dst_port))
        print("[camera_stream] VLC récepteur   : vlc rtp://@:%d" % args.dst_port)
        gst_proc = gst_sink(args.dst_ip, args.dst_port)
    if gst_proc is None:
        print("[camera_stream] ERREUR: gst-launch-1.0 introuvable.")
        print("  -> Installer GStreamer : sudo apt install gstreamer1.0-tools "
              "gstreamer1.0-plugins-good gstreamer1.0-plugins-bad libgstreamer1.0-dev")
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

    with dai.Device(pipeline) as device:
        q_h264 = device.getOutputQueue("h264", maxSize=10, blocking=False)
        q_prev = device.getOutputQueue("preview", maxSize=4, blocking=False) if has_preview else None

        print("[camera_stream] Streaming... Ctrl+C pour arrêter")

        while True:
            # Flux H.264 → GStreamer → RTP/UDP
            pkt = q_h264.get()
            data = pkt.getData()

            # Envoyer au pipe GStreamer
            try:
                gst_proc.stdin.write(data.tobytes())
                gst_proc.stdin.flush()
            except BrokenPipeError:
                print("\n[camera_stream] GStreamer pipe fermé — arrêt")
                break

            # Enregistrement local optionnel
            if record_file:
                record_file.write(data.tobytes())

            frame_count += 1
            if frame_count % (args.fps * 5) == 0:
                elapsed = time.time() - t0
                print("[camera_stream] %d frames | %.1f fps" % (
                    frame_count, frame_count / elapsed))

            # Preview local optionnel
            if q_prev and cv2:
                prev_msg = q_prev.tryGet()
                if prev_msg is not None:
                    bgr = prev_msg.getCvFrame()
                    cv2.imshow("Camera Stream (local)", bgr)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break

    if gst_proc and gst_proc.poll() is None:
        gst_proc.stdin.close()
        gst_proc.terminate()
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
    while True:
        try:
            run(args)
            break  # sortie propre (Ctrl+C)
        except KeyboardInterrupt:
            print("\n[camera_stream] Arrêt demandé.")
            break
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e) or "Device crashed" in str(e):
                print("[camera_stream] OAK-D crash détecté — reconnexion dans 3s...")
                time.sleep(3)
            else:
                raise

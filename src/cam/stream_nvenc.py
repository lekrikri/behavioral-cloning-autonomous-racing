"""
camera_stream_nvenc.py — Streaming OAK-D Lite via NVENC Jetson Nano

Architecture :
  OAK-D Lite → NV12 brut (PAS de VideoEncoder Myriad X) → USB2
  → Jetson → nvv4l2h264enc (NVENC hardware) → TCP/VLC

Avantage vs camera_stream.py :
  - Le VPU Myriad X ne fait plus d'encodage H.264 → -30-50% conso OAK-D
  - Réduit fortement les X_LINK_ERROR (brownout USB)
  - NVENC Jetson encode sans impacter le CPU (hardware)

Usage (Jetson Nano) :
  OPENBLAS_CORETYPE=ARMV8 python3 -u -m src.cam.stream_nvenc --serve --dst-port 5600

VLC : tcp://192.168.0.100:5600

Test A/B recommandé :
  Comparer la stabilité (nb de crashs/heure) avec camera_stream.py (Myriad encoder)
  vs ce script (NVENC). Si crashs réduits → Myriad encoder était la cause.
"""

import argparse
import subprocess
import sys
import time

try:
    import depthai as dai
except ImportError:
    print("[nvenc] depthai non installe")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dst-ip",   default="255.255.255.255")
    p.add_argument("--dst-port", type=int, default=5600)
    p.add_argument("--serve",    action="store_true",
                   help="Mode serveur TCP : VLC se connecte a tcp://JETSON_IP:PORT")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=360)
    p.add_argument("--fps",      type=int, default=15)
    p.add_argument("--bitrate",  type=int, default=2000000,
                   help="Bitrate NVENC bps (defaut: 2000000 = 2Mbps)")
    return p.parse_args()


def build_pipeline(args):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setVideoSize(args.width, args.height)
    cam.setInterleaved(False)
    cam.setFps(args.fps)

    # Sortie NV12 brute — PAS de VideoEncoder (decharge le VPU Myriad X)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("raw")
    cam.video.link(xout.input)

    return pipeline


def build_gst_cmd(args):
    # Taille d'une frame NV12 : width * height * 3/2 bytes
    frame_bytes = args.width * args.height * 3 // 2

    if args.serve:
        sink = [
            "!", "mpegtsmux",
            "!", "tcpserversink", "host=0.0.0.0", "port=%d" % args.dst_port,
            "sync=false", "recover-policy=keyframe",
        ]
    else:
        sink = [
            "!", "rtph264pay", "config-interval=1", "pt=96",
            "!", "udpsink", "host=%s" % args.dst_ip, "port=%d" % args.dst_port,
            "sync=false", "async=false",
        ]

    cmd = [
        "gst-launch-1.0", "-q",
        # Lecture NV12 brut depuis stdin (blocksize = exactement 1 frame)
        "fdsrc", "blocksize=%d" % frame_bytes,
        "!",
        "video/x-raw,format=NV12,width=%d,height=%d,framerate=%d/1" % (
            args.width, args.height, args.fps),
        # Conversion vers mémoire GPU NVMM (zero-copy)
        "!", "nvvidconv",
        "!", "video/x-raw(memory:NVMM),format=NV12",
        # Encodage H.264 hardware NVENC (Jetson Nano)
        "!", "nvv4l2h264enc",
        "maxperf-enable=1",
        "bitrate=%d" % args.bitrate,
        "iframeinterval=%d" % args.fps,  # keyframe chaque seconde
        "control-rate=1",                 # CBR — evite les pics de bitrate
        "!", "h264parse",
    ] + sink

    return cmd


def run(args):
    pipeline = build_pipeline(args)
    gst_cmd = build_gst_cmd(args)

    print("[nvenc] %dx%d @ %dfps | NVENC H.264 %dkbps (OAK-D decharge)" % (
        args.width, args.height, args.fps, args.bitrate // 1000))

    if args.serve:
        print("[nvenc] Mode SERVEUR TCP port %d" % args.dst_port)
        print("[nvenc] VLC : tcp://192.168.0.100:%d" % args.dst_port)
    else:
        print("[nvenc] UDP vers %s:%d" % (args.dst_ip, args.dst_port))

    try:
        gst_proc = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[nvenc] ERREUR: gst-launch-1.0 introuvable")
        sys.exit(1)

    frame_count = 0
    t0 = time.time()
    frame_bytes = args.width * args.height * 3 // 2

    try:
        # usb2Mode=True : API depthai 2.x (reduit conso OAK-D)
        with dai.Device(pipeline, True) as device:
            q = device.getOutputQueue("raw", maxSize=4, blocking=False)
            print("[nvenc] Streaming... Ctrl+C pour arreter")

            while True:
                pkt = q.get()
                data = pkt.getData().tobytes()

                # Verifier taille frame (NV12 attendu)
                if len(data) != frame_bytes:
                    print("[nvenc] WARN: frame %d bytes (attendu %d) — format pas NV12 ?" % (
                        len(data), frame_bytes))
                    # Tentative : peut-etre BGR (width*height*3)
                    if len(data) == args.width * args.height * 3:
                        print("[nvenc] Detection BGR — recompiler avec cam.isp ou changer setColorOrder")
                    continue

                try:
                    gst_proc.stdin.write(data)
                    gst_proc.stdin.flush()
                except BrokenPipeError:
                    print("\n[nvenc] GStreamer pipe ferme — arret")
                    break

                frame_count += 1
                if frame_count % (args.fps * 5) == 0:
                    elapsed = time.time() - t0
                    print("[nvenc] %d frames | %.1f fps" % (frame_count, frame_count / elapsed))

    finally:
        if gst_proc and gst_proc.poll() is None:
            gst_proc.stdin.close()
            gst_proc.terminate()
            gst_proc.wait()
        print("[nvenc] Termine — %d frames" % frame_count)


if __name__ == "__main__":
    args = parse_args()
    attempt = 0
    while True:
        try:
            attempt += 1
            if attempt > 1:
                print("[nvenc] Tentative %d..." % attempt)
            run(args)
            break
        except KeyboardInterrupt:
            print("\n[nvenc] Arret.")
            break
        except RuntimeError as e:
            msg = str(e)
            if "X_LINK_ERROR" in msg or "Device crashed" in msg or "Couldn't read" in msg:
                delay = min(5 * attempt, 30)
                print("[nvenc] OAK-D crash — reconnexion dans %ds..." % delay)
                time.sleep(delay)
            else:
                raise

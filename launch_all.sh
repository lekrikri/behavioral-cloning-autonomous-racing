#!/usr/bin/env bash
# launch_all.sh — Lance camera_hub + mask_stream (port 8088) + controller dry-run (port 5601)
#
# Depuis la Jetson :
#   chmod +x launch_all.sh && ./launch_all.sh
#
# Depuis ton PC (SSH tunnel pour mask_stream) :
#   ssh -L 8088:localhost:8088 robocar
#   puis ouvrir http://localhost:8088      → UI calibration masque (anti-blob)
#        ouvrir http://10.41.58.41:5601   → stream controller + UI save config

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO/src"
HUB_PORT=8077
MASK_PORT=8088
STREAM_PORT=5601

export OPENBLAS_CORETYPE=ARMV8

echo "[launch] Arret des anciens process..."
pkill -f camera_hub.py    2>/dev/null || true
pkill -f mask_stream.py   2>/dev/null || true
pkill -f controller_pd.py 2>/dev/null || true
sleep 2

echo "[launch] Demarrage camera_hub sur :$HUB_PORT ..."
python3 -u "$SRC/camera_hub.py" --port "$HUB_PORT" --width 640 --height 320 \
    > /tmp/hub.log 2>&1 &
HUB_PID=$!
echo "[launch] camera_hub pid=$HUB_PID"

echo "[launch] Attente init OAK-D (5s)..."
sleep 5

# Verifier que le hub est pret
if ! python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$HUB_PORT),2); s.close()" 2>/dev/null; then
    echo "[launch] ERREUR: camera_hub ne repond pas sur :$HUB_PORT"
    echo "[launch] Logs hub:"
    tail -20 /tmp/hub.log
    exit 1
fi
echo "[launch] camera_hub OK"

echo "[launch] Demarrage mask_stream sur :$MASK_PORT (source=hub)..."
python3 -u "$SRC/mask_stream.py" --source hub --no-auto-hub \
    --port "$MASK_PORT" --width 640 --height 320 \
    > /tmp/mask_stream.log 2>&1 &
MASK_PID=$!
echo "[launch] mask_stream pid=$MASK_PID"

echo "[launch] Demarrage controller dry-run sur :$STREAM_PORT (source=hub)..."
echo "[launch] ---"
echo "[launch] UI controller  : http://10.41.58.41:$STREAM_PORT"
echo "[launch] UI calibration : http://localhost:$MASK_PORT  (apres: ssh -L $MASK_PORT:localhost:$MASK_PORT robocar)"
echo "[launch] ---"
echo "[launch] Ctrl+C pour tout arreter"

# Lance le controller en foreground (Ctrl+C le tue + cleanup)
trap "echo '[launch] Arret...'; kill $HUB_PID $MASK_PID 2>/dev/null; exit 0" INT TERM

python3 -u "$SRC/controller_pd.py" --dry-run \
    --source hub --hub-port "$HUB_PORT" \
    --stream-port "$STREAM_PORT"

# Cleanup si controller se termine seul
kill $HUB_PID $MASK_PID 2>/dev/null || true

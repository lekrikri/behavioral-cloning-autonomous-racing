#!/usr/bin/env bash
# supervisor_controller.sh — Garde le controller_pd.py toujours vivant.
# Relance automatiquement si crash ou STOP inattendu.
#
# Usage depuis Jetson (en SSH ou tmux) :
#   chmod +x supervisor_controller.sh
#   ./supervisor_controller.sh [args supplémentaires pour controller_pd.py]
#
# Stopper proprement : Ctrl+C (SIGINT propagé au controller)

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO/src"

export OPENBLAS_CORETYPE=ARMV8

CTRL_ARGS="--source hub --hub-port 8077 --stream-port 5601 --max-duty 0.05 --steering-max 1.0 --roi-far 0.20"

# Passer des args supplémentaires depuis la ligne de commande
if [ $# -gt 0 ]; then
    CTRL_ARGS="$*"
fi

RESTART_DELAY=3
MAX_RESTARTS=20
restarts=0

echo "[sup] Superviseur controller_pd.py démarré"
echo "[sup] Args : $CTRL_ARGS"
echo "[sup] Ctrl+C pour arrêter"

cleanup() {
    echo ""
    echo "[sup] Arrêt reçu — extinction..."
    kill "$CTRL_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

while true; do
    echo "[sup] Lancement controller (tentative $((restarts + 1)))..."
    python3 -u "$SRC/controller_pd.py" $CTRL_ARGS &
    CTRL_PID=$!
    echo "[sup] controller PID=$CTRL_PID"

    wait "$CTRL_PID"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[sup] controller terminé proprement (exit 0) — relance dans ${RESTART_DELAY}s..."
    else
        echo "[sup] controller crash (exit $EXIT_CODE) — relance dans ${RESTART_DELAY}s..."
    fi

    restarts=$((restarts + 1))
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "[sup] ERREUR : $MAX_RESTARTS crashs consécutifs — abandon."
        exit 1
    fi

    sleep "$RESTART_DELAY"
    RESTART_DELAY=3  # reset délai après chaque relance réussie
done

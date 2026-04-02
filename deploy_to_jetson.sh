#!/bin/bash
# deploy_to_jetson.sh — Transfert des fichiers essentiels vers le Jetson Nano
#
# Usage : bash deploy_to_jetson.sh <JETSON_IP>
# Exemple: bash deploy_to_jetson.sh 192.168.1.42
#
# Sur le Jetson, créer d'abord le repo :
#   git clone https://github.com/lekrikri/behavioral-cloning-autonomous-racing.git

JETSON_IP="${1:-}"
JETSON_USER="${2:-robocar}"
REMOTE_DIR="/home/${JETSON_USER}/behavioral-cloning-autonomous-racing"

if [ -z "$JETSON_IP" ]; then
    echo "Usage: bash deploy_to_jetson.sh <JETSON_IP> [user]"
    echo "Exemple: bash deploy_to_jetson.sh 192.168.1.42"
    echo ""
    echo "Pour trouver l'IP du Jetson:"
    echo "  Sur le Jetson: hostname -I"
    exit 1
fi

echo "════════════════════════════════════════════════"
echo "  Deploy → ${JETSON_USER}@${JETSON_IP}"
echo "  Destination : ${REMOTE_DIR}"
echo "════════════════════════════════════════════════"

# ── Vérifier connexion SSH ──────────────────────────────────────────────────
echo "[0] Test connexion SSH..."
ssh -o ConnectTimeout=5 "${JETSON_USER}@${JETSON_IP}" "echo '  Connexion OK'" || {
    echo "  ❌ Connexion SSH échouée"
    echo "  Vérifier: ssh ${JETSON_USER}@${JETSON_IP}"
    exit 1
}

# ── Créer les répertoires distants ─────────────────────────────────────────
echo "[1] Création des répertoires..."
ssh "${JETSON_USER}@${JETSON_IP}" "mkdir -p ${REMOTE_DIR}/models/v18 ${REMOTE_DIR}/src"

# ── Transférer le modèle ONNX ───────────────────────────────────────────────
echo "[2] Transfert modèle ONNX v18..."
scp models/v18/best.onnx "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/models/v18/"
echo "  ✅ best.onnx ($(du -h models/v18/best.onnx | cut -f1))"

# ── Transférer les stats Z-score ────────────────────────────────────────────
echo "[3] Transfert ray_stats.json..."
if [ -f "models/real_ray_stats.json" ]; then
    scp models/real_ray_stats.json "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/models/"
    echo "  ✅ real_ray_stats.json (données RÉELLES)"
elif [ -f "models/ray_stats.json" ]; then
    scp models/ray_stats.json "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/models/"
    echo "  ⚠️  ray_stats.json (stats simulation — recalibrer avec calibrate_ray_stats.py)"
else
    echo "  ⚠️  Aucun ray_stats.json trouvé — inférence utilisera fallback simulation"
fi

# ── Transférer le code source ───────────────────────────────────────────────
echo "[4] Transfert code source..."
scp src/depth_to_rays.py     "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
scp src/vesc_interface.py    "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
scp src/inference_realcar.py "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
scp src/calibrate_ray_stats.py "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
scp src/test_pipeline.py     "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
scp src/__init__.py          "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/src/"
echo "  ✅ src/ (5 fichiers)"

# ── Transférer les scripts ──────────────────────────────────────────────────
echo "[5] Transfert scripts..."
scp setup_jetson.sh          "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/"
scp requirements_jetson.txt  "${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/"
echo "  ✅ setup_jetson.sh + requirements_jetson.txt"

# ── Résumé ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ Transfert terminé !"
echo ""
echo "  Sur le Jetson, lancer dans l'ordre :"
echo ""
echo "  1. Setup (1 seule fois)"
echo "     bash ${REMOTE_DIR}/setup_jetson.sh"
echo ""
echo "  2. Activer mode perf max"
echo "     sudo nvpmodel -m 0 && sudo jetson_clocks"
echo ""
echo "  3. Calibrer Z-score (2-5 min sur piste)"
echo "     cd ${REMOTE_DIR}"
echo "     python3 src/calibrate_ray_stats.py"
echo ""
echo "  4. Test roues en l'air !"
echo "     python3 src/inference_realcar.py --duty-max 0.15"
echo "════════════════════════════════════════════════"

#!/bin/bash
# setup_jetson.sh — Installation complète G-CAR-000 sur Jetson Nano
#
# Usage : bash setup_jetson.sh
# Temps estimé : 10-20 min
#
# Hardware cible : Jetson Nano 4GB + OAK-D Lite + Flipsky FSESC Mini V6.7 Pro
# JetPack : 4.6.1 (R32.7.1) — Python 3.6 default, on installe Python 3.8

set -e

PY="python3.8"
PIP="python3.8 -m pip"

echo "════════════════════════════════════════════════"
echo "  G-CAR-000 — Setup Jetson Nano (JetPack 4.6.1)"
echo "════════════════════════════════════════════════"
echo ""

# ── 1. Dépendances système ──────────────────────────────────────────────────
echo "[1/5] Dépendances système + Python 3.8..."
sudo apt-get update -qq
sudo apt-get install -y python3.8 python3.8-dev python3-pip \
    cmake libusb-1.0-0-dev libglib2.0-dev udev git curl
# Bootstrap pip pour Python 3.8 si absent
$PY -m pip --version 2>/dev/null || {
    curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | sudo $PY
}
echo "  ✅ Python 3.8 + pip prêts"

# ── 2. numpy + pyserial + pyvesc ────────────────────────────────────────────
echo "[2/5] numpy + pyserial + pyvesc..."
$PIP install --upgrade pip
$PIP install "numpy>=1.21" pyserial>=3.5
$PIP install pyvesc || echo "  ⚠️  pyvesc: install manuel si besoin — pip install pyvesc"
echo "  ✅ numpy + pyserial OK"

# ── 3. ONNX Runtime ─────────────────────────────────────────────────────────
echo "[3/5] ONNX Runtime..."
# Chercher wheel GPU local d'abord (téléchargeable sur elinux.org/Jetson_Zoo)
ONNX_WHEEL=$(find /home -name "onnxruntime_gpu*cp38*.whl" 2>/dev/null | head -1)
if [ -n "$ONNX_WHEEL" ]; then
    echo "  Wheel GPU trouvé : $ONNX_WHEEL"
    $PIP install "$ONNX_WHEEL"
    echo "  ✅ ONNX Runtime GPU"
else
    echo "  Wheel GPU non trouvé — installation CPU (suffisant pour notre modèle ~30k params)"
    $PIP install onnxruntime
    echo "  ✅ ONNX Runtime CPU"
    echo ""
    echo "  Pour GPU (optionnel) :"
    echo "  wget https://elinux.org/Jetson_Zoo → onnxruntime 1.17 JetPack4.6 cp38"
    echo "  puis : python3.8 -m pip install onnxruntime_gpu*.whl"
fi

# ── 4. OAK-D Lite (depthai) ─────────────────────────────────────────────────
echo "[4/5] OAK-D Lite (depthai)..."
echo "  Installation udev rules Luxonis..."
sudo curl -fsSL https://docs.luxonis.com/install_dependencies.sh | bash 2>/dev/null || true
$PIP install depthai
echo "  ✅ depthai installé"

# ── 5. Droits USB (VESC + OAK-D) ────────────────────────────────────────────
echo "[5/5] Droits USB..."
if ! groups $USER | grep -q dialout; then
    sudo usermod -aG dialout $USER
    echo "  ✅ Ajouté au groupe dialout (reconnexion requise)"
else
    echo "  ✅ Groupe dialout OK"
fi

# ── Vérification finale ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Vérification avec python3.8..."
echo "════════════════════════════════════════════════"
$PY -c "import numpy; print(f'  numpy       {numpy.__version__} ✅')"
$PY -c "import onnxruntime as ort; print(f'  onnxruntime {ort.__version__} — {ort.get_available_providers()} ✅')"
$PY -c "import serial; print(f'  pyserial    {serial.__version__} ✅')"
$PY -c "import pyvesc; print('  pyvesc      ✅')" 2>/dev/null || echo "  pyvesc      ⚠️  (optionnel)"
$PY -c "import depthai; print(f'  depthai     {depthai.__version__} ✅')" || echo "  depthai     ⚠️  (brancher OAK-D + vérifier udev)"

echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ Setup terminé !"
echo ""
echo "  Lancer avec python3.8 :"
echo "  1. sudo nvpmodel -m 0 && sudo jetson_clocks"
echo "  2. python3.8 src/calibrate_ray_stats.py"
echo "  3. python3.8 src/inference_realcar.py --duty-max 0.15"
echo "════════════════════════════════════════════════"

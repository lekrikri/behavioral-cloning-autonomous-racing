#!/bin/bash
# setup_jetson.sh — Installation complète G-CAR-000 sur Jetson Nano
#
# Usage : bash setup_jetson.sh
# Temps estimé : 15-30 min (selon vitesse réseau + build cmake)
#
# Hardware cible : Jetson Nano 4GB + OAK-D Lite + Flipsky FSESC Mini V6.7 Pro

set -e  # arrêt si erreur

echo "════════════════════════════════════════════════"
echo "  G-CAR-000 — Setup Jetson Nano"
echo "════════════════════════════════════════════════"
echo ""

# ── 1. Dépendances système ──────────────────────────────────────────────────
echo "[1/5] Dépendances système..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-dev cmake libusb-1.0-0-dev \
    libglib2.0-dev udev git curl

# ── 2. udev rules pour OAK-D ───────────────────────────────────────────────
echo "[2/5] Installation OAK-D Lite (depthai)..."
echo "  Installation des udev rules Luxonis..."
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash

pip3 install depthai --no-deps || {
    echo "  ⚠️  Fallback: build depthai depuis source..."
    pip3 install depthai
}
echo "  ✅ depthai installé"

# ── 3. ONNX Runtime GPU ─────────────────────────────────────────────────────
echo "[3/5] ONNX Runtime GPU (Jetson wheel)..."
# Chercher wheel local d'abord
ONNX_WHEEL=$(find /home -name "onnxruntime_gpu*.whl" 2>/dev/null | head -1)
if [ -n "$ONNX_WHEEL" ]; then
    echo "  Wheel trouvé : $ONNX_WHEEL"
    pip3 install "$ONNX_WHEEL"
else
    echo "  ⚠️  Wheel GPU non trouvé — installation CPU fallback"
    echo "  Pour GPU: télécharger le wheel ARM64 sur https://elinux.org/Jetson_Zoo#ONNX_Runtime"
    echo "  puis : pip3 install onnxruntime_gpu-*.whl"
    pip3 install onnxruntime
fi

# ── 4. Autres dépendances ───────────────────────────────────────────────────
echo "[4/5] numpy + pyserial + pyvesc..."
pip3 install numpy>=1.21 pyserial>=3.5 pyvesc>=0.1.6

# ── 5. Vérification droits USB (VESC) ──────────────────────────────────────
echo "[5/5] Droits USB VESC (ttyACM)..."
if ! groups $USER | grep -q dialout; then
    sudo usermod -aG dialout $USER
    echo "  ✅ Ajouté au groupe dialout (reconnexion nécessaire)"
else
    echo "  ✅ Déjà dans le groupe dialout"
fi

# ── Vérification finale ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Vérification..."
echo "════════════════════════════════════════════════"
python3 -c "import numpy; print(f'  numpy     {numpy.__version__} ✅')"
python3 -c "import onnxruntime as ort; print(f'  onnxruntime {ort.__version__} — providers: {ort.get_available_providers()} ✅')"
python3 -c "import serial; print(f'  pyserial  {serial.__version__} ✅')"
python3 -c "import pyvesc; print('  pyvesc    ✅')" 2>/dev/null || echo "  pyvesc    ⚠️  (optionnel)"
python3 -c "import depthai; print(f'  depthai   {depthai.__version__} ✅')" || echo "  depthai   ⚠️  (vérifier udev rules)"

echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ Setup terminé !"
echo ""
echo "  Prochaines étapes :"
echo "  1. Vérifier VESC port : ls /dev/ttyACM* /dev/ttyUSB*"
echo "  2. Calibrer Z-score   : python3 src/calibrate_ray_stats.py"
echo "  3. Test roues en l'air: python3 src/inference_realcar.py --duty-max 0.15"
echo "════════════════════════════════════════════════"

# G-CAR-000 — État du déploiement (2026-06-12)

## Ce qui est FAIT ✅

### Simulation
| Élément | Détail |
|---------|--------|
| Modèle BC v18 entraîné | **Record 24s** sur circuit simulateur Unity |
| Export ONNX | `models/v18/best.onnx` — prêt pour Jetson |
| Architecture | 5 raycasts Z-scorés + heuristique front_raw → steering + accel |

### Jetson Nano (hardware)
| Élément | Détail |
|---------|--------|
| JetPack 4.6.1 installé | Python 3.8, CUDA 10.2 |
| depthai 2.21.2 installé | wheel aarch64 précompilé |
| onnxruntime CPU | 480 FPS d'inférence validé |
| pyvesc installé + patché | Fix CRC XModem (`init=0x0000`) obligatoire |
| SSH fonctionnel | `robocar@192.168.0.115` — projet `~/behavioral-cloning-autonomous-racing/` |

### VESC (ESC moteur)
| Élément | Détail |
|---------|--------|
| Mode BLDC Sensorless | FOC abandonné (overcurrent -126.2A) |
| Détection moteur OK | `param_detect 8.0 1000 0.08` |
| Enable Servo Output | True |
| Timeout | 1000ms (à passer à 200ms — voir ci-dessous) |
| CRC pyvesc fix | `CRCCCITT("XModem")` — patch sur Jetson ✅ |

### Calibrations
| Élément | Détail |
|---------|--------|
| Servo | center=0.500, range=±0.350, invert=False |
| Z-score réel | 5469 frames → `models/real_ray_stats.json` ✅ |

### Premier roulage autonome ✅
```bash
python3.8 -m src.control.inference_realcar --duty-max 0.20
```
La voiture avance de manière autonome, steer/accel calculés en temps réel (~25 Hz).

### OAK-D Lite sur PC Windows (session 2026-06-12)
- depthai **3.6.1** installé dans `C:\python311\`
- Driver **WinUSB** installé via Zadig (nécessaire sur Windows)
- Script `C:\Users\Admin\oak_info_mask.py` fonctionnel :
  - Preview live 512×256
  - Masque HSV temps réel (lignes blanches)
  - Calibration intrinsics : fx=402, FOV=68.8°
  - ESPACE = capturer image brute + masque → `C:\Users\Admin\raw_cam\`

---

## Ce qui RESTE à faire ⚠️

### Priorité 1 — Fix accoups moteur (BLOQUANT pour piste)
```
VESC Tool → App Settings → General → APP = No App
(actuellement APP=UART, le timeout UART override les commandes pyvesc)
```
Après fix → relancer `inference_realcar.py` → vérifier fluidité

### Priorité 2 — Watchdog sécurité VESC
```
VESC Tool → App Settings → General → Timeout = 200ms
(actuellement 1000ms — trop long en cas de crash Python)
```
Si Python plante → VESC coupe moteur après 200ms au lieu de 1000ms.

### Priorité 3 — Test sur piste réelle
1. Poser repères visuels sur piste (ruban adhésif blanc)
2. Quelqu'un avec kill-switch physique (déconnecter batterie)
3. Premier test : `--duty-max 0.15` (lent)
4. Montée progressive : 0.15 → 0.25 → 0.35 → 0.40

### Priorité 4 — Dataset piste physique (optionnel)
- Capturer images piste réelle avec OAK-D Lite (`oak_info_mask.py`)
- Comparer distribution des raycasts sim vs réel
- Fine-tuning possible si sim-to-real gap trop grand

---

## Commandes de référence

### Sur PC (Windows) — lancer la caméra
```powershell
C:\python311\python.exe C:\Users\Admin\oak_info_mask.py
```

### Sur Jetson Nano
```bash
ssh robocar@192.168.0.115
cd ~/behavioral-cloning-autonomous-racing

# Calibration Z-score (si besoin de recalibrer)
python3.8 -m src.tools.calibrate_rays

# Inférence (roues en l'air d'abord !)
python3.8 -m src.control.inference_realcar --duty-max 0.15

# Servo seulement (debug)
python3.8 -m src.tools.calibrate_servo
```

### Patch pyvesc (si réinstallé)
```bash
sudo sed -i "s/CRCCCITT().calculate/CRCCCITT(\"XModem\").calculate/g" \
  /usr/local/lib/python3.8/dist-packages/pyvesc/packet/structure.py \
  /usr/local/lib/python3.8/dist-packages/pyvesc/packet/codec.py
```

---

## Architecture finale

```
OAK-D Lite (depth map)
    ↓ depth_to_rays.py
Raycasts virtuels [20 angles]
    ↓ Z-score (real_ray_stats.json)
[23 features normalisées]
    ↓ models/v18/best.onnx (inchangé)
steering [-1,1] + accel [0,1]
    ↓ vesc_interface.py
FSESC VESC → servo + moteur brushless
```

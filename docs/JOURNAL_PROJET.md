# Journal de Projet — G-CAR-000 Robocar Autonome

## Objectif
Entraîner un modèle de Behavioral Cloning (imitation learning) dans un simulateur Unity,
puis le déployer sur une vraie voiture RC équipée d'un Jetson Nano et d'une caméra de profondeur OAK-D Lite.

---

## PHASE 1 — Simulateur Unity & Behavioral Cloning

### Environnement
- **Simulateur** : Unity ML-Agents, circuit fermé avec raycasts de profondeur
- **Algorithme** : Behavioral Cloning (BC) — apprentissage par imitation
- **Framework** : PyTorch, gRPC port 5005

### Architecture du modèle (RobocarSpatial v18)
- **Entrées** : 5 raycasts de profondeur normalisés (gauche → droite) + heuristique front_raw
- **Sorties** : steering [-1, 1] + acceleration [0, 1]
- **Normalisation** : Z-score par canal (mean/std par raycast)
- **Heuristique front_raw** : si front < 0.65 → réduction accel (corner_damp)
- **SmoothingFilter** : alpha_base=0.57, alpha_max=0.92, deadzone=0.06

### Résultat simulateur
- **Record : 24 secondes** (meilleur tour complet du circuit)
- Modèle sauvegardé : `models/v18/best.pth`
- Exporté en ONNX : `models/v18/best.onnx`

---

## PHASE 2 — Déploiement sur Jetson Nano

### Hardware
- **Cerveau** : Jetson Nano (JetPack 4.6.1, Ubuntu 18.04, CUDA 10.2)
- **Caméra** : OAK-D Lite (profondeur stéréo, depthai 2.21.2)
- **ESC** : Flipsky FSESC Mini V6.7 Pro (basé VESC)
- **Moteur** : Traxxas brushless 3300kV (sensorless)
- **Batterie** : LiPo 4S (16.74V)
- **Connexion ESC** : USB `/dev/ttyACM0` (STM32F407)

### Connexion SSH
```
IP : 192.168.0.115
User : robocar / robocar
Projet : ~/behavioral-cloning-autonomous-racing/
```

### Installation sur Jetson
- Python 3.8, onnxruntime (CPU), depthai 2.21.2.0 (wheel précompilé aarch64)
- pyvesc (communication VESC via serial)
- PyCRC (corrigée manuellement — mauvais paquet pip)
- Performance validée : **480 FPS** (inférence ONNX sur CPU Jetson)

---

## PHASE 3 — Configuration VESC (ESC)

### Problème initial : FOC Overcurrent
- La détection moteur en mode **FOC** (Field-Oriented Control) a généré un pic de -126.2A
- Flux linkage = 0.00 mWb → VESC ne pouvait pas piloter le moteur
- Fault : `FAULT_CODE_ABS_OVER_CURRENT` bloquant toutes les commandes

### Solution : Mode BLDC
1. `drv_reset_faults` dans terminal VESC Tool → efface le fault DRV
2. `param_detect 8.0 1000 0.08` → détection paramètres BLDC (motor tourne ✅)
   - Cycle integrator limit : 23.55
   - Coupling factor : 972.26
   - Sensorless (pas de capteurs hall)
3. Direction correcte confirmée (pas d'inversion)

### Configuration finale VESC
- **Mode** : BLDC Sensorless
- **Enable Servo Output** : True
- **Timeout** : 1000ms (limited mode TCP empêchait de changer à 200ms)
- **Connexion TCP debug** : socat `TCP-LISTEN:65102 ↔ /dev/ttyACM0`

---

## PHASE 4 — Debug Communication pyvesc

### Bug critique découvert : CRC incorrect
**Symptôme** : moteur fonctionne depuis VESC Tool mais pas depuis Python/pyvesc.

**Cause** : pyvesc utilisait `CRCCCITT(init=0xFFFF)` mais le VESC attend `CRC-CCITT XModem (init=0x0000)`.

| Source | CRC pour SetDutyCycle(15%) |
|--------|---------------------------|
| pyvesc (avant fix) | `da b4` ❌ |
| VESC Tool (correct) | `cb b8` ✅ |

**Fix** : patch des deux fichiers pyvesc sur le Jetson :
```bash
sudo sed -i "s/CRCCCITT().calculate/CRCCCITT(\"XModem\").calculate/g" \
  /usr/local/lib/python3.8/dist-packages/pyvesc/packet/structure.py \
  /usr/local/lib/python3.8/dist-packages/pyvesc/packet/codec.py
```

**Résultat** : les packets VESC sont maintenant identiques entre VESC Tool et pyvesc ✅

---

## PHASE 5 — Calibrations & Premier Roulage

### calibrate_servo.py
- `servo_center = 0.500`
- `servo_range = ±0.350`
- `invert_steer = False`

### calibrate_ray_stats.py (Z-score réel)
- 5469 frames collectées en 3 phases (droites / virages doux / virages serrés)
- Sauvegardé dans `models/real_ray_stats.json`
- 215 valeurs Z>5 → bruit depth réel, normal en sim-to-real

### Premier roulage autonome ✅
```bash
python3.8 -m src.control.inference_realcar --duty-max 0.20
```
- **La voiture avance de manière autonome !**
- steer / accel calculés en temps réel par le modèle BC
- ~25 Hz boucle de contrôle, 400 FPS inférence

### Problème en cours : accoups moteur
- Cause probable : APP = UART dans VESC → timeout UART App override les commandes COMM
- **Fix à appliquer** : App Settings → General → APP = `No App`

---

## État Actuel

| Composant | Statut |
|-----------|--------|
| Modèle BC v18 (simulateur) | ✅ 24s record |
| Déploiement Jetson Nano | ✅ 480 FPS |
| Communication VESC (CRC fix) | ✅ |
| Calibration servo | ✅ center=0.5, range=±0.35 |
| Calibration Z-score réel | ✅ 5469 frames |
| Premier roulage autonome | ✅ La voiture roule ! |
| Fluidité moteur | ⚠️ Accoups → fix APP=No App en cours |

---

## Prochaines Étapes

1. **Fix APP=No App** → relancer inference → vérifier fluidité
2. **Vrai test sur piste** avec repères visuels (ruban adhésif au sol)
3. **Ajustement hyperparamètres** si sim-to-real gap trop important
4. **Possible fine-tuning** avec données réelles collectées sur piste

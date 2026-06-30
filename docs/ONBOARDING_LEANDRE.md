# G-CAR-000 — Onboarding Technique pour Léandre
> Projet Epitech Robocar Racing · Behavioral Cloning · Mai 2026

---

## TL;DR — Ce que fait ce projet en 3 lignes

On entraîne un **réseau de neurones à conduire une voiture RC en autonome**, sans programmer de règles à la main.
Le modèle apprend en imitant Christophe qui conduit manuellement (= Behavioral Cloning).
**Phase 1 (sim) : terminée — record 24s/tour. Phase 2 (voiture physique) : en cours de déploiement.**

---

## 1. Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 : SIMULATION (✅ terminée)                              │
│                                                                  │
│  Unity Simulator ──gRPC──► Python client                        │
│       │                         │                               │
│   raycasts virtuels         inference.py                        │
│   (20 distances)            ONNX model v18                      │
│       │                         │                               │
│  ◄──── steering + accel ────────┘                               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 2 : VOITURE PHYSIQUE (🔧 en cours)                        │
│                                                                  │
│  OAK-D Lite (caméra depth) ──► depth_to_rays.py                 │
│       │                              │                           │
│   depth map uint16 (mm)         raycasts virtuels [0,1]         │
│                                      │                           │
│                                 inference_realcar.py             │
│                                 ONNX model v18                   │
│                                      │                           │
│                               vesc_interface.py                  │
│                                      │                           │
│                         FSESC Mini V6.7 Pro ──► moteur + servo  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Le pipeline d'entraînement

### 2.1 Collecte de données — `data_collector.py`

Christophe conduit manuellement via Unity. À chaque frame, on enregistre dans un CSV :
- `ray_0` ... `ray_19` : 20 distances aux murs (valeurs [0,1], normalisées)
- `steering` : [-1.0 (gauche) → +1.0 (droite)]
- `accel` / `speed` : accélération et vitesse

```
data/
  episode_001.csv
  episode_002.csv
  ...
```

### 2.2 Le vecteur d'observation — 23 features

```python
# Ce que le modèle reçoit à chaque frame :
obs = [
    ray_0, ray_1, ..., ray_19,   # 20 raycasts Z-scorés
    Δray_0, ..., Δray_19,         # 20 deltas (ray_t - ray_{t-1}) — contexte temporel
    front_raw,                    # rayon frontal brut
]
# → shape [23] (ou [43] avec use_delta=True)
```

**Pourquoi pas une image ?** Trop lourd pour du temps réel embarqué sur Jetson Nano. 23 floats = ~1µs d'inférence. Une image 640×480 = des ms de preprocessing.

### 2.3 Augmentation — `dataset.py`

Pour robustifier le modèle et éviter l'overfitting :

| Augmentation | Description |
|---|---|
| **Flip gauche-droite** (`flip_prob=0.5`) | Mirror horizontal — double le dataset gratuitement |
| **Gaussian noise adaptatif** (`noise_std=0.02`) | Bruit ∝ distance du rayon (rayons lointains = plus bruités en réalité) |
| **Ray cutout** | Masquer 1-2 rayons aléatoires → simule un capteur défaillant |
| **Speed jitter** (`±10%`) | Variation de vitesse → robustesse aux conditions variables |
| **WeightedRandomSampler** | Rééquilibre la distribution steering (évite le biais "tout droit") |

### 2.4 Le modèle — `model.py` — `RobocarSpatial` v18

Architecture **CNN 1D + GRU** :

```
Input: [B, 23]
       │
       ▼
CNN 1D (3 couches, kernel 3)
  → détecte les patterns spatiaux dans les raycasts
  → ex: "mur à gauche + dégagement à droite = virage à droite"
       │
       ▼
GRU (hidden=64, seq_len=10)
  → "mémoire" : garde les 10 dernières frames
  → sait qu'on EST en train de tourner, pas juste qu'on est tourné
       │
       ▼
MLP (2 têtes)
  ├── Tanh → steering ∈ [-1, 1]
  └── Sigmoid → accel ∈ [0, 1]
```

**29 862 paramètres** — ultra-léger, tourne en < 10ms/frame sur CPU.

### 2.5 La loss function

```python
# WeightedHuber : moins sensible aux outliers que MSE
loss = 0.85 * huber(pred_steer, true_steer) +
       0.15 * huber(pred_accel, true_accel)

# + Temporal Consistency Loss (anti-zigzag pour le GRU)
loss += 0.25 * mean(|steer_t - steer_{t-1}|²)

# + PairwiseSmoothingLoss (anti-zigzag pour frame-indépendant)
loss += 0.30 * mean(|steer_{t+1} - steer_t|²)
```

Le steering est pondéré 5.67x plus que l'accel — c'est lui qui fait crasher la voiture.

### 2.6 Training — `train.py`

```bash
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber
```

- Optimiseur : **Adam** avec `ReduceLROnPlateau`
- **Mixed precision** AMP (torch.amp) — accélération CUDA
- **Early stopping** sur val_loss
- **Temporal split** : les derniers 20% des épisodes → validation (pas de data leakage temporel)
- Export final en **ONNX** (opset 17) → portable sur Jetson Nano sans PyTorch

---

## 3. Le modèle en production — v18

```
models/v18/
  best.pth              ← weights PyTorch (training uniquement)
  best.onnx             ← modèle exporté (inference Jetson + Unity)
  training_history.json ← courbes loss/metrics
```

**Record : 24 secondes par tour** sur la piste Unity (track 1).

---

## 4. Sim-to-Real — Le pont OAK-D → raycasts

C'est le challenge principal de la Phase 2. En simulation, les raycasts sont parfaits (Unity les calcule). Sur la voiture physique, il faut les **reconstruire depuis une depth map**.

### `depth_to_rays.py` — comment ça marche

```
OAK-D Lite caméra stéréo
       │
       ▼
depth_frame: uint16 array (400×640), valeurs en mm
       │
       ▼
ROI = band 40%→62% de hauteur (anti-tangage)
  → si la voiture tangue, on ne prend pas le sol ni le plafond
       │
       ▼
Pour chaque rayon i (de -48.5° à +48.5°) :
  → calculer la colonne pixel correspondante (géométrie FOV 97°)
  → prendre le MINIMUM de la bande ROI (= premier obstacle = vrai raycast)
  → ignorer pixels < 100mm (réflexions/bruit sol)
  → normaliser [0, 1] : 0 = très proche, 1 = loin
       │
       ▼
rays: float32 array (20,) → identique aux raycasts Unity
```

**Pourquoi le minimum ?** Un raycast physique s'arrête au premier obstacle. Si on prend la moyenne, un petit obstacle peut être noyé dans les pixels "vide".

### Calibration requise avant premier test

```bash
# Étape 1 : calibrer le servo (trouver center, range, inversion)
python3.8 -m src.tools.calibrate_servo

# Étape 2 : collecter les stats Z-score réels de l'OAK-D (pas ceux de la sim)
# → 3 phases : ligne droite, virages gauche, virages droite
python3.8 -m src.tools.calibrate_rays
# → génère models/ray_stats.json (mean + std par rayon)
```

---

## 5. La voiture physique — hardware

| Composant | Modèle | Rôle |
|---|---|---|
| **Ordinateur embarqué** | Jetson Nano 4GB | Inference ONNX + contrôle |
| **Caméra depth** | Luxonis OAK-D Lite | Depth map stéréo → raycasts |
| **Contrôleur moteur** | Flipsky FSESC Mini V6.7 Pro | ESC VESC-compatible |
| **OS** | JetPack 4.6 (Ubuntu 18.04 + Python 3.8) | Contrainte CUDA Jetson |

### `vesc_interface.py` — pourquoi pas `SetDutyCycle` ?

```python
# ❌ SetDutyCycle(0.20) → 50A overcurrent → FAULT_CODE_ABS_OVER_CURRENT
# ✅ SetCurrent(5.0A)  → FOC gère le couple → pas de spike
```

Le protocole VESC est implémenté **from scratch** (pas de lib pyvesc) pour être compatible Python 3.6+ sur le Jetson.

---

## 6. `inference_realcar.py` — boucle temps réel

```
Thread 1 (Perception, 30 fps) :
  OAK-D Lite → depth frame → DepthToRays → rays (20,) → Queue

Thread 2 (Contrôle, 30 fps) :
  Queue.get() → Z-score → ONNX inference → SmoothingFilter → VESC

Watchdog :
  Si pas de frame depuis 250ms → SetCurrent(0) + SetServoPos(center)
  → arrêt d'urgence automatique
```

**SmoothingFilter** : lisse les outputs du modèle (exponentiel + corner damping) pour éviter les oscillations servo.

```bash
# Lancer l'inférence (roues en l'air d'abord !)
python3.8 -m src.control.inference_realcar --duty-max 0.15
```

---

## 7. Structure du repo

```
G-CAR-000/
├── src/
│   ├── model.py              ← architectures (MLP, CNN, RobocarSpatial)
│   ├── dataset.py            ← DrivingDataset + augmentations
│   ├── train.py              ← boucle training + export ONNX
│   ├── inference.py          ← inférence Unity (simulation)
│   ├── inference_realcar.py  ← inférence voiture physique (multi-thread)
│   ├── depth_to_rays.py      ← bridge OAK-D → raycasts virtuels
│   ├── vesc_interface.py     ← protocole VESC natif Python
│   ├── calibrate_servo.py    ← calibration servo (à lancer en 1er)
│   ├── calibrate_ray_stats.py← collecte Z-score réel OAK-D
│   ├── data_collector.py     ← collecte données depuis Unity
│   ├── evaluate.py           ← métriques (MAE steer, MAE accel, %)
│   ├── client.py             ← client gRPC Unity
│   └── config.py             ← hyperparamètres centralisés
├── models/
│   ├── v18/
│   │   ├── best.pth          ← record 24s — modèle actif
│   │   ├── best.onnx         ← export déployé
│   │   └── training_history.json
│   ├── ray_stats.json        ← stats Z-score (sim, à recalibrer pour real)
│   └── v6/ ... v17/          ← historique des itérations
├── data/                     ← CSVs de collecte (gitignorés)
├── deploy_to_jetson.sh       ← scp vers le Jetson Nano
└── ONBOARDING_LEANDRE.md    ← ce fichier
```

---

## 8. Commandes essentielles

```bash
# Entraîner un nouveau modèle
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber

# Lancer l'inférence en simulation (Unity doit tourner)
python src/inference.py --model models/v18/best.onnx

# Évaluer le modèle sur le val set
python src/evaluate.py --model models/v18/best.pth --data data/

# Déployer sur le Jetson Nano (SSH configuré)
bash deploy_to_jetson.sh

# Sur le Jetson — calibration servo (OBLIGATOIRE en 1er)
python3.8 -m src.tools.calibrate_servo

# Sur le Jetson — calibration raycasts OAK-D (OBLIGATOIRE en 2ème)
python3.8 -m src.tools.calibrate_rays

# Sur le Jetson — lancer l'inférence real car (ROUES EN L'AIR d'abord)
python3.8 -m src.control.inference_realcar --duty-max 0.15
```

---

## 9. Contexte projet & next steps

### Historique des versions

| Version | Score | Changement clé |
|---|---|---|
| v6 | ~45s | Baseline MLP |
| v12 | ~35s | CNN 1D + augmentation |
| v15 | ~28s | GRU + temporal consistency loss |
| v17 | ~26s | Z-score normalization + temporal split |
| **v18** | **24s** | PSL + front_raw heuristic + corner damp ✅ |

### Ce qui reste à faire (Phase 2)

1. **Test physique chez Boris** — la voiture est au hub, code prêt
2. **Calibration servo** sur la vraie voiture (valeurs sim ≠ réel)
3. **Calibration ray_stats** OAK-D (les Z-scores sim ≠ réel)
4. **Première run à petite vitesse** (`--duty-max 0.10`) pour vérifier la direction
5. **Ajustement sim-to-real gap** si nécessaire (bruit, FOV, latence)

### Idées futures (Phase 3)

- **DAgger** : améliorer le modèle in-situ en le faisant conduire + expert qui reprend la main sur les erreurs
- **Généralisation multi-pistes** : entraîner sur plusieurs tracks différentes
- **Réduction latence** : quantization INT8 du modèle ONNX pour le Jetson

---

## 10. Questions fréquentes

**Q : Pourquoi pas du reinforcement learning ?**
Le RL nécessite des milliers d'épisodes d'échec en simulation avant de converger. Le Behavioral Cloning converge en quelques heures avec ~10k frames humaines. Pour une course RC, c'est largement suffisant.

**Q : Pourquoi ONNX et pas PyTorch direct ?**
PyTorch n'est pas optimisé pour l'inférence CPU/embedded. ONNX Runtime utilise des optimisations spécifiques (fusion d'ops, quantization) et tourne sur Jetson sans installer CUDA PyTorch (~4GB de libs économisées).

**Q : Le modèle peut-il généraliser à une nouvelle piste ?**
En l'état, v18 est entraîné sur la piste 1 Unity. Il faut re-collecter des données sur la nouvelle piste et fine-tuner (quelques epochs suffisent, les features bas-niveau restent valables).

**Q : Pourquoi Python 3.8 sur le Jetson ?**
JetPack 4.6 (Ubuntu 18.04) est la dernière version supportée sur le Jetson Nano 4GB. Elle impose Python 3.8. Toutes les dépendances sont compatibles 3.8 (`onnxruntime`, `depthai 2.21.2`, `pyserial`).

---

*Document généré pour l'équipe G-CAR-000 · Epitech 2026*
*Auteur : Christophe · Contact : ganou.christophe@gmail.com*

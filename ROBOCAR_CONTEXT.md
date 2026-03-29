# G-CAR-000 — Robocar Racing Simulator
## Contexte Technique & Architecture

> Doc technique pour référence rapide.
> Pour la version pédagogique détaillée → **GUIDE_COMPLET.md**

---

## Stack

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Simulation | Unity + ML-Agents | - |
| Communication | gRPC (mlagents-envs) | 0.28.0 |
| Framework IA | PyTorch | 2.9 + CUDA |
| Manette | pygame | 2.6 |
| Clavier | pynput | 1.8 |
| Export | ONNX Runtime | 1.24 |
| Language | Python | 3.12 |

---

## Architecture du système

```
Unity (RacingSimulator.x86_64)
        │ gRPC port 5004
        ▼
src/client.py          → RobocarEnv (wrapper UnityEnvironment)
        │
    ┌───┴───────────────────────────────────────┐
    │                                           │
    ▼ Phase Collecte                            ▼ Phase Inférence
src/input_manager.py                     src/inference.py
  GamepadManager (pygame)                  load_model(.pth/.onnx)
  KeyboardManager (pynput)                 SmoothingFilter (α=0.7)
  auto-detect via create_input_manager()   JerkTracker
        │                                        │
        ▼                                        ▼
src/data_collector.py                     actions → Unity
  CSV: [timestamp, ray_0..N, speed,
        steering, acceleration]
        │
        ▼ Phase Entraînement
src/dataset.py
  DrivingDataset
    augmentations: flip, noise_adaptatif, speed_jitter, cutout
    WeightedRandomSampler (rééquilibre steering)
        │
src/model.py
  RobocarCNN  (1D-CNN + GlobalAvgPool + MLP) ← recommandé
  RobocarMLP  (baseline)
  WeightedHuberLoss (steer×0.7 + accel×0.3)
        │
src/train.py
  mixed precision (torch.amp)
  Adam + ReduceLROnPlateau + early stopping
  export ONNX opset 18
        │
src/evaluate.py
  MAE / RMSE / R² / within_threshold
  plots matplotlib
```

---

## Modèles disponibles

### RobocarCNN (recommandé — Gemini)

```
Input [N+1] → Rays branch (Conv1d×2 + GAP) + Speed →
Concat → Dense(64→32→2) → Tanh
Params: ~6 178 | Latence Jetson: ~1ms
```

### RobocarMLP (baseline)

```
Input [N+1] → Dense(128→64→32) → Linear(2) → Tanh
Params: ~12 322 | Latence Jetson: < 0.5ms
```

---

## Commandes

```bash
# Installation
cd /home/lekrikri/Projects/G-CAR-000
pip install -r requirements.txt  # déjà installé

# Tests pipeline (sans simulateur)
python src/model.py
python src/dataset.py
python src/train.py --data data/synthetic.csv --arch cnn --epochs 30

# Test connexion simulateur (simulateur lancé)
python src/client.py --test-only
python src/client.py

# Collecte données
python src/data_collector.py collect --output data/run_01.csv
python src/data_collector.py merge --input-dir data/ --output data/merged.csv

# EDA
jupyter notebook notebooks/eda.ipynb

# Entraînement
python src/train.py --data data/ --arch cnn --loss huber --epochs 100

# Évaluation
python src/evaluate.py --model models/best.pth --data data/

# Inférence (simulateur lancé)
python src/inference.py --model models/best.pth
python src/inference.py --model models/best.onnx  # ONNX Runtime
```

---

## Config agents (config.json)

```json
{ "agents": [{ "fov": 180, "nbRay": 10 }] }
```

`fov` : 1–180° | `nbRay` : 1–50 | Port : 5004

---

## Améliorations v2 (post Gemini/Grok)

| Fix | Module | Status |
|-----|--------|--------|
| 1D-CNN (motifs spatiaux) | model.py | ✅ |
| WeightedHuberLoss | model.py | ✅ |
| BatchNorm order: BN→ReLU→Drop | model.py | ✅ |
| WeightedRandomSampler | dataset.py | ✅ |
| Augmentation: noise_adaptatif + speed_jitter + cutout | dataset.py | ✅ |
| Flip: .flip(0) au lieu de [::-1] | dataset.py | ✅ |
| Mixed precision (torch.amp) | train.py | ✅ |
| ONNX export: eval+cpu+opset18 | model.py | ✅ |
| SmoothingFilter v2 (reset+NaN+1er step) | inference.py | ✅ |
| GamepadManager (pygame) + auto-detect | input_manager.py | ✅ |
| config.py centralisé (dataclass) | config.py | ✅ |

---

## Roadmap

- [x] Phase 1 : Infrastructure + code complet + tests pipeline
- [x] Phase 2 : Améliorations v2 (Gemini/Grok)
- [ ] Phase 3 : Connexion simulateur réel + test clavier/manette
- [ ] Phase 4 : Collecte données réelles (10k+ frames)
- [ ] Phase 5 : EDA notebook
- [ ] Phase 6 : Entraînement sur vraies données
- [ ] Phase 7 : Inférence temps réel
- [ ] Bonus  : DAgger / Quantization / RL (PPO)

---

## Structure des fichiers

```
G-CAR-000/
├── ROBOCAR_CONTEXT.md    ← cette doc (référence technique)
├── GUIDE_COMPLET.md      ← guide pédagogique étudiant (A→Z)
├── config.json           ← config agents Unity
├── requirements.txt      ← dépendances Python
├── BuildLinux/           ← simulateur Unity (binaire)
├── src/
│   ├── config.py         ← RobocarConfig (dataclass centralisée)
│   ├── client.py         ← RobocarEnv (wrapper ML-Agents)
│   ├── input_manager.py  ← GamepadManager + KeyboardManager
│   ├── data_collector.py ← collecte CSV + merge
│   ├── model.py          ← RobocarCNN / MLP + losses
│   ├── dataset.py        ← DrivingDataset + augmentations
│   ├── train.py          ← boucle entraînement
│   ├── evaluate.py       ← métriques + plots
│   └── inference.py      ← run modèle en simulation
├── notebooks/
│   └── eda.ipynb
├── data/                 ← CSV trajectoires (gitignored)
└── models/               ← checkpoints .pth + .onnx (gitignored)
```

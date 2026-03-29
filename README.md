# Behavioral Cloning — Autonomous Racing

> Imitation learning for autonomous racing in a Unity simulation.
> A Conv1D spatial model (PyTorch) learns to drive from human demonstrations — achieving a **24-second lap** with smooth, zigzag-free driving.

---

## Demo

```
Agent 0: 00:24   ← record lap time
Best Score: 00:24
```

The car perceives the track via **20 raycasts** (laser distances, [0,1]) and predicts steering + acceleration at each frame (~20 FPS via gRPC).

---

## Architecture

```
Input: 20 Z-scored raycasts + 3 derived features (asymmetry, front_ray, min_ray)
                         │
          ┌──────────────┴──────────────┐
          ▼                              ▼
   Rays [20] → Conv1D(2→12, k=3)    Derived [3] → FC(3→16)
               flatten [120]
          └──────────────┬──────────────┘
                   Concat [136]
                         ↓
                  FC(136→96) → BN → ReLU → Dropout
                         ↓
                   FC(96→48) → BN → ReLU
                         ↓
              ┌──────────┴──────────┐
         steer_head             accel_head
         Tanh → [-1,1]          Sigmoid → [0,1]
```

**~30k parameters** — fast enough for real-time inference on Jetson Nano via ONNX/TensorRT.

---

## Key Techniques

### PairwiseSmoothingLoss (PSL)
Penalizes abrupt steering changes between consecutive frames:
```
PSL = λ × mean( (steer[t+1] - steer[t])² )    λ = 0.30
```
Combined with **temporal split** (chronological train/val/test, no shuffle), this eliminated zigzag almost entirely.

### BimodalLoss
Treats acceleration as binary classification (go / brake):
```
Loss = 0.88 × HuberLoss(steering) + 0.12 × BCEWithLogitsLoss(accel > 0.25)
```

### Front-Ray Heuristic
Uses raw (non-normalized) front raycasts to anticipate corners:
```python
if front_raw >= 0.65:   # straight / wall far away
    front_cap = 1.0     # full throttle
else:
    front_cap = 0.45 + 0.70 * front_raw   # progressive braking
```

### Adaptive SmoothingFilter
Alpha adapts to steering magnitude — reactive in corners, stable on straights. Deadzone of 0.06 suppresses micro-oscillations.

---

## Results

| Version | Lap Time | Notes |
|---------|----------|-------|
| v10 | 1m06s | First complete lap, 20 rays |
| v18 | 40s | PSL + temporal split, zero zigzag |
| v18 + front_raw | 36s | Corner anticipation |
| **v18 + threshold 0.65** | **24s** | **Record** — full speed on straight |

---

## Project Structure

```
src/
├── client.py          # RobocarEnv — Unity ML-Agents gRPC wrapper (port 5005)
├── data_collector.py  # Human driving data collection → CSV
├── dataset.py         # DrivingDataset — Z-score, augmentation, temporal split
├── model.py           # RobocarSpatial (Conv1D) + RobocarMLP
├── train.py           # Training loop — BimodalLoss + PairwiseSmoothingLoss
├── inference.py       # Real-time inference — SmoothingFilter + front_raw heuristic
└── evaluate.py        # MAE / RMSE / R² metrics + plots

data/                  # Collected CSV datasets (gitignored)
models/                # Trained checkpoints .pth + .onnx (gitignored)
notebooks/             # EDA notebook
config.json            # Simulator config (nbRay=20, fov=180)
GUIDE_COMPLET.md       # In-depth technical guide (French)
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Launch the simulator (Unity build)
./BuildLinux/RacingSimulator.x86_64 --mlagents-port 5005 \
  --config-path config.json

# 2. Collect driving data (human)
python src/data_collector.py collect --output data/run_01.csv

# 3. Train
python src/train.py --data data/run_01.csv --arch spatial \
  --no-action-cond --no-delta --no-sampler \
  --temporal-split \
  --epochs 150 --steer-weight 0.88 --accel-weight 0.12 \
  --output models/v_new

# 4. Run inference
python src/inference.py --model models/v_new/best.pth
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | PyTorch — Conv1D + MLP |
| Simulation | Unity ML-Agents (gRPC) |
| Export | ONNX Runtime (Jetson Nano ready) |
| Data | ~35k frames, 20 raycasts @ 49 FPS |
| Training | BimodalLoss + PSL, temporal split, mixed precision |

---

## Next Steps

- [ ] Modify Unity C# scripts to transmit real speed → enable speed/TTC features
- [ ] Train v20 with speed input → replace geometric heuristic with learned acceleration
- [ ] LightGRU model (seq_len=10) for temporal context → better corner anticipation
- [ ] Deploy on Jetson Nano via ONNX → TensorRT FP16

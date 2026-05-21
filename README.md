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
behavioral-cloning-autonomous-racing/
├── src/                    # Core ML code
│   ├── model.py            # RobocarSpatial — Conv1D + GRU
│   ├── dataset.py          # DrivingDataset + augmentations
│   ├── train.py            # Training loop + ONNX export
│   ├── inference.py        # Real-time inference (simulation)
│   ├── inference_realcar.py# Real-time inference (physical car)
│   ├── depth_to_rays.py    # OAK-D Lite depth map → virtual raycasts
│   ├── vesc_interface.py   # Native VESC protocol (Python 3.6+)
│   ├── data_collector.py   # Human driving data collection
│   ├── calibrate_servo.py  # Servo calibration (run first on Jetson)
│   ├── calibrate_ray_stats.py # Z-score calibration for OAK-D
│   └── evaluate.py         # MAE / RMSE / R² metrics
├── tests/                  # Hardware & integration tests
├── docs/                   # Technical documentation
│   ├── ONBOARDING_LEANDRE.md   # Team onboarding guide
│   ├── GUIDE_COMPLET.md        # In-depth technical guide
│   └── HARDWARE_REALCAR.md     # Physical car hardware reference
├── configs/                # Configuration files
│   └── config.json         # Simulator config (nbRay=20, fov=180)
├── data/                   # Collected CSV datasets (gitignored)
├── models/                 # Trained models
│   ├── v18/best.onnx       # Active model — 24s lap record
│   └── ray_stats.json      # Z-score normalization stats
├── notebooks/              # EDA and exploration
├── requirements.txt        # Python deps (simulation)
├── requirements_jetson.txt # Python deps (Jetson Nano)
├── deploy_to_jetson.sh     # Deploy to Jetson via SSH
└── setup_jetson.sh         # Jetson initial setup
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Launch the simulator (Unity build)
./BuildLinux/RacingSimulator.x86_64 --mlagents-port 5005 \
  --config-path configs/config.json

# 2. Collect driving data (human)
python src/data_collector.py collect --output data/run_01.csv

# 3. Train
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber

# 4. Run inference
python src/inference.py --model models/v18/best.onnx
```

> New to the project? Read [docs/ONBOARDING_LEANDRE.md](docs/ONBOARDING_LEANDRE.md) first.
> Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md).

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

# Behavioral Cloning — Autonomous Racing

> Imitation learning for autonomous racing — Unity simulation + physical RC car.
> A Conv1D+GRU model (PyTorch) learns to drive from human demonstrations — achieving a **24-second lap** with smooth, zigzag-free driving.
> Phase 2 : real car deployment on Jetson Nano + OAK-D Lite (in progress).

---

## Team

| Name | Role |
|------|------|
| **Christophe** | ML / Software — simulation + physical car |
| **Léandre** | Mask Generator — dataset annotation |
| **Mathieu** | Mask Generator — dataset annotation |

---

## Demo

```
Agent 0: 00:24   ← record lap time (simulation)
Best Score: 00:24
```

The car perceives the track via **20 raycasts** (laser distances, [0,1]) and predicts steering + acceleration at each frame via gRPC.

---

## Project Phases

### Phase 1 — Simulation ✅

Behavioral cloning in Unity. Model v18 achieves 24s/lap on track 1.

### Phase 2 — Physical Car 🔧

Deploying v18 on a real RC car: OAK-D Lite depth camera → virtual raycasts → ONNX inference → FSESC VESC motor controller.

### Mask Generator 🆕

Generating binary track masks from the car's camera (256×128 RGB) to give the AI vision of its environment.
See [docs/MASK_GENERATOR.md](docs/MASK_GENERATOR.md).

---

## Architecture — Navigation Model (v18)

```
Input: [23 features]
  ray_0…ray_19  (20 Z-scored raycasts)
  Δray_0…Δray_19 (temporal deltas)
  front_raw      (raw front raycast)
         │
   CNN 1D (3 layers, kernel=3)
   → spatial patterns in raycasts
         │
   GRU (hidden=64, seq_len=10)
   → temporal memory: "we ARE turning"
         │
   MLP  ──┬── Tanh  → steering ∈ [-1, 1]
          └── Sigmoid → accel   ∈ [0, 1]
```

**29 862 parameters** — < 10ms inference on CPU.

---

## Architecture — Mask Generator

```
Input: RGB image 256×128 (OAK-D Lite camera)
         │
   CLAHE (normalize lighting, reduce reflections)
         │
   HSV → detect white lines (S<45, V>195)
         │
   Threshold → isolate dark track surface
         │
   Flood fill from image bottom (track always touches bottom)
         │
   Morphological closing → fill reflection holes
         │
   Largest connected component
         │
Output: binary mask 256×128  (255=track, 0=off-track)

→ Phase 2: fine-tuned with Micro U-Net (~90k params, IoU > 0.85)
```

---

## Key Techniques

### PairwiseSmoothingLoss (PSL)
Penalizes abrupt steering changes between frames:
```
PSL = 0.30 × mean( (steer[t+1] - steer[t])² )
```

### WeightedHuber Loss
```
loss = 0.85 × huber(steering) + 0.15 × huber(accel)
```
Steering weighted 5.67× higher — it's what crashes the car.

### Front-Ray Heuristic
```python
front_cap = 1.0 if front_raw >= 0.65 else 0.45 + 0.70 * front_raw
```

### Sim-to-Real Bridge
OAK-D Lite depth map → minimum-per-column sampling across 97° FOV → normalized raycasts [0,1] identical to Unity format.

---

## Results

| Version | Lap Time | Key change |
|---------|----------|------------|
| v6  | ~45s | MLP baseline |
| v12 | ~35s | CNN 1D + augmentation |
| v15 | ~28s | GRU + temporal consistency loss |
| v17 | ~26s | Z-score + temporal split |
| **v18** | **24s** | PSL + front_raw heuristic ✅ |

---

## Project Structure

```
behavioral-cloning-autonomous-racing/
├── src/
│   ├── model.py               # RobocarSpatial — Conv1D + GRU (v18)
│   ├── dataset.py             # DrivingDataset + augmentations
│   ├── train.py               # Training loop + ONNX export
│   ├── inference.py           # Real-time inference (simulation)
│   ├── inference_realcar.py   # Real-time inference (physical car)
│   ├── depth_to_rays.py       # OAK-D depth map → virtual raycasts
│   ├── vesc_interface.py      # Native VESC protocol (Python 3.6+)
│   ├── mask_generator.py      # Track mask generation (OpenCV)  🆕
│   ├── unet_model.py          # Micro U-Net segmentation model   🆕
│   ├── dataset_masks.py       # Mask dataset + augmentations     🆕
│   ├── train_mask.py          # U-Net training + ONNX export     🆕
│   ├── data_collector.py      # Human driving data collection
│   ├── calibrate_servo.py     # Servo calibration (Jetson first step)
│   ├── calibrate_ray_stats.py # Z-score calibration for OAK-D
│   └── evaluate.py            # MAE / RMSE / R² metrics
├── tests/                     # Hardware & integration tests
├── docs/
│   ├── MASK_GENERATOR.md      # Mask Generator — full doc 🆕
│   ├── ONBOARDING_LEANDRE.md  # Team onboarding guide
│   ├── GUIDE_COMPLET.md       # In-depth technical guide (FR)
│   └── HARDWARE_REALCAR.md    # Physical car hardware reference
├── configs/
│   └── config.json            # Simulator config (nbRay=20, fov=180)
├── data/                      # Datasets (gitignored)
├── models/
│   ├── v18/best.onnx          # Active navigation model — 24s record
│   └── ray_stats.json         # Z-score normalization stats
├── notebooks/                 # EDA
├── requirements.txt           # Python deps (simulation)
├── requirements_jetson.txt    # Python deps (Jetson Nano / Python 3.8)
├── deploy_to_jetson.sh        # SSH deploy script
└── setup_jetson.sh            # Jetson initial setup
```

---

## Quickstart — Simulation

```bash
pip install -r requirements.txt

# 1. Launch simulator
./BuildLinux/RacingSimulator.x86_64 --mlagents-port 5005 --config-path configs/config.json

# 2. Collect data (human driving)
python src/data_collector.py collect --output data/run_01.csv

# 3. Train
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber

# 4. Run inference
python src/inference.py --model models/v18/best.onnx
```

## Quickstart — Mask Generator

```bash
pip install opencv-python-headless

# Generate masks automatically for all images
python src/mask_generator.py \
    --input  data/256_128/256_128 \
    --output data/masks_auto \
    --preview data/masks_preview

# Train U-Net on corrected masks
python src/train_mask.py --masks data/masks_auto --output models/mask_v1
```

> Full documentation: [docs/MASK_GENERATOR.md](docs/MASK_GENERATOR.md)

## Quickstart — Physical Car (Jetson Nano)

```bash
# Step 1: servo calibration (mandatory)
python3.8 src/calibrate_servo.py

# Step 2: OAK-D raycasts calibration (mandatory)
python3.8 src/calibrate_ray_stats.py

# Step 3: run inference (wheels off ground first!)
python3.8 src/inference_realcar.py --duty-max 0.15
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Navigation model | PyTorch — Conv1D + GRU, ONNX export |
| Mask Generator | OpenCV (CLAHE + flood fill) + Micro U-Net |
| Simulation | Unity ML-Agents (gRPC port 5005) |
| Physical camera | Luxonis OAK-D Lite (stereo depth) |
| Motor controller | Flipsky FSESC Mini V6.7 Pro (VESC) |
| Embedded computer | Jetson Nano 4GB (JetPack 4.6, Python 3.8) |

---

## Next Steps

### Phase 2 — Physical car
- [ ] Physical test at Boris's — car is ready, code ready
- [ ] Servo calibration on real car (sim values ≠ real)
- [ ] OAK-D ray_stats calibration (sim Z-scores ≠ real)
- [ ] First slow run (`--duty-max 0.10`) to verify steering
- [ ] Sim-to-real gap adjustment if needed

### Mask Generator
- [ ] Generate auto masks for all 2793 images
- [ ] Manual correction of ~500 images (Léandre + Mathieu)
- [ ] Train Micro U-Net → target IoU > 0.85
- [ ] Integrate mask output into `inference_realcar.py`

### Phase 3 — Future
- [ ] DAgger: online expert corrections during driving
- [ ] Multi-track generalization
- [ ] INT8 quantization for Jetson Nano TensorRT

---

> New to the project? Read [docs/ONBOARDING_LEANDRE.md](docs/ONBOARDING_LEANDRE.md) first.
> Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md).

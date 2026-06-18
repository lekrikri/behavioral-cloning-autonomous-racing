# Commandes essentielles — référence unique

> **Source de vérité unique** pour toutes les commandes du projet.
> Les autres docs (README, ONBOARDING, CONTRIBUTING…) doivent pointer ici plutôt
> que recopier ces commandes — c'est ce qui évite qu'elles divergent.
>
> Convention : sur le **PC de dev** → `python`. Sur le **Jetson Nano** → `python3.8`.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

---

## 2. Simulation (PC de dev, Unity lancé)

```bash
# Lancer le simulateur Unity (port gRPC 5005 — 5004 est bloqué par Windows/RTP)
./BuildLinux/RacingSimulator.x86_64 --mlagents-port 5005 --config-path configs/config.json

# Collecte de données (conduite humaine → CSV)
python src/data_collector.py collect --output data/run_01.csv
python src/data_collector.py merge --input-dir data/ --output data/merged.csv

# Entraînement
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber

# Évaluation (MAE / RMSE / R²)
python src/evaluate.py --model models/v18/best.pth --data data/

# Inférence en simulation
python src/inference.py --model models/v18/best.onnx
```

---

## 3. Mask Generator (perception)

```bash
pip install opencv-python-headless

# Générer les masques automatiques pour tout le dataset
python src/mask_generator.py \
    --input   data/256_128/256_128 \
    --output  data/masks_auto \
    --preview data/masks_preview

# Entraîner le U-Net sur les masques corrigés
python src/train_mask.py --masks data/masks_auto --output models/mask_v1
```

> Détails : [`MASK_GENERATOR.md`](MASK_GENERATOR.md)

---

## 4. Voiture réelle (Jetson Nano)

> ⚠️ **Toujours roues en l'air pour le premier test.** L'ordre des calibrations
> est obligatoire : servo puis raycasts, sinon l'inférence part sur de mauvaises stats.

```bash
# Déploiement depuis le PC de dev (SSH configuré)
bash deploy_to_jetson.sh

# Étape 1 — calibration servo (OBLIGATOIRE en 1er)
python3.8 src/calibrate_servo.py

# Étape 2 — calibration raycasts OAK-D / Z-score réel (OBLIGATOIRE en 2ème)
python3.8 src/calibrate_ray_stats.py

# Étape 3 — inférence real-car (ROUES EN L'AIR d'abord !)
python3.8 src/inference_realcar.py --duty-max 0.15
python3.8 src/inference_realcar.py --duty-max 0.15 --servo-center 0.52 --invert-steer
```

> Pile de contrôle bas niveau (VESC, codec, sécurité) : [`CONTROL_STACK.md`](CONTROL_STACK.md)

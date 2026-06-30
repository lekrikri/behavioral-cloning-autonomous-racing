# Mask Generator — Documentation Équipe G-CAR-000

> Projet Epitech Robocar Racing · Christophe, Léandre, Mathieu · 2026

---

## Pourquoi ce projet existe

Boris a lancé le **Mask Generator** comme prérequis technique pour la Phase 2 du projet Robocar.

L'IA de la voiture physique a besoin de **voir où est la piste** pour naviguer.  
Sans ça, elle ne peut pas distinguer le sol roulable des murs et des zones interdites.

> "Ce projet est le socle technique indispensable pour que l'IA soit en mesure d'analyser et d'appréhender son environnement." — Boris, 29/04/2026

---

## Le problème en une image

La caméra frontale de la voiture prend des images **256×128 pixels** comme celle-ci :

```
┌──────────────────────────────────┐
│  [mur clair / décor en arrière]  │  ← hors-piste
│  ────────────────────────────    │  ← ligne blanche
│  [sol anthracite — piste]        │  ← zone navigable ✓
│  [sol anthracite — piste]        │  ← zone navigable ✓
└──────────────────────────────────┘
         caméra frontale voiture
```

Ce qu'on veut produire automatiquement :

```
┌──────────────────────────────────┐
│  [noir  — hors-piste]            │  0
│  [noir  — ligne blanche exclue]  │  0
│  [blanc — piste navigable]       │  255
│  [blanc — piste navigable]       │  255
└──────────────────────────────────┘
              masque binaire
```

---

## Ce que Boris nous a envoyé

- **2793 images** PNG 256×128 RGB dans `data/256_128/256_128/`
- Images brutes de la vraie voiture, **sans masques** — c'est notre boulot de les générer

---

## Architecture du dataset

```
data/
├── 256_128/256_128/        ← images brutes de Boris
│   ├── 0_original_image.png
│   ├── 154_original_image.png
│   └── ...  (2793 images)
└── masks_auto/             ← masques générés automatiquement (à créer)
    ├── 0_original_image_mask.png
    └── ...
```

---

## Stratégie choisie (validée par Gemini + ChatGPT + Grok)

Les 3 IA consultées recommandent la **même approche hybride** :

### Phase 1 — Vision classique OpenCV (baseline rapide)

Générer automatiquement ~80-85% des masques exploitables.  
Pas d'entraînement, pas de GPU, très rapide.

**Pipeline :**
```
Image RGB 256×128
    │
    ▼
CLAHE (normalise éclairage, atténue reflets)
    │
    ▼
Espace HSV — détection lignes blanches (S<45, V>195)
    │
    ▼
Threshold inverse — isole le sol sombre (<155)
    │
    ▼
Flood Fill multi-seed depuis le bas de l'image
  (la piste touche TOUJOURS le bas en caméra frontale)
    │
    ▼
Morphological Close (kernel 13×9) — comble les trous/reflets
    │
    ▼
Plus grande composante connexe — supprime les artefacts
    │
    ▼
Masque binaire 256×128
```

**Pourquoi le flood fill depuis le bas ?**  
La voiture est toujours SUR la piste. Donc le bas de l'image = sol de la piste, garanti.  
On diffuse depuis ce point en ne traversant pas les lignes blanches.

**Pourquoi CLAHE en premier ?**  
Les reflets lumineux sur le sol anthracite créent des pixels trop clairs qui casseraient le threshold.  
CLAHE "aplatit" localement ces pics de luminosité avant d'analyser.

### Phase 2 — Micro U-Net (robustesse production)

Un réseau de neurones minuscule (~90k paramètres) entraîné sur les masques corrigés.  
Robuste aux variations de lumière, ombres, reflets que l'OpenCV ne gère pas.

```
Input RGB (3, 128, 256)
    │
    ▼  Conv 3→8  → MaxPool
    ▼  Conv 8→16 → MaxPool
    ▼  Conv 16→32  (bottleneck)
    ▼  UpConv 32→16 + Skip
    ▼  UpConv 16→8  + Skip
    ▼  Conv 8→1 + Sigmoid
    │
    ▼
Masque probabilités (1, 128, 256)  → seuil 0.5 → binaire
```

**Performances cibles :**
- < 5ms sur CPU, < 2ms sur Jetson Nano GPU
- IoU > 0.85 sur val set avec 300-500 images annotées

---

## Fichiers créés

| Fichier | Rôle |
|---|---|
| [src/mask/training/mask_generator.py](../src/mask/training/mask_generator.py) | Génération auto masques (OpenCV, Phase 1) |
| [src/mask/training/unet_model.py](../src/mask/training/unet_model.py) | Architecture Micro U-Net + DiceBCELoss |
| [src/mask/training/dataset_masks.py](../src/mask/training/dataset_masks.py) | Dataset PyTorch + augmentations |
| [src/mask/training/train_mask.py](../src/mask/training/train_mask.py) | Entraînement U-Net + export ONNX |

---

## Comment démarrer — Étape par étape

### Prérequis

```bash
pip install opencv-python-headless pillow numpy
pip install torch torchvision   # pour la Phase 2 uniquement
```

### Étape 1 — Générer les masques automatiques

```bash
# Génère les masques pour tout le dataset (quelques minutes)
python -m src.mask.training.mask_generator \
    --input  data/256_128/256_128 \
    --output data/masks_auto \
    --preview data/masks_preview   # optionnel : overlay vert pour vérifier

# Tester sur une seule image
python -m src.mask.training.mask_generator --single data/256_128/256_128/154_original_image.png
# → génère 154_original_image_mask.png et 154_original_image_preview.png
```

### Étape 2 — Vérifier visuellement les masques

Ouvrir quelques images dans `data/masks_preview/` :
- Overlay **vert** = zone détectée comme piste navigable
- Si un masque est mauvais → noter le nom pour correction manuelle

### Étape 3 — Correction manuelle (300-500 images)

Outils recommandés :
- **CVAT** (web, gratuit) : https://cvat.ai
- **Label Studio** (local) : `pip install label-studio`
- Simple script à la main avec `cv2.imshow` + clicks

Format de sortie attendu : PNG binaire, même nom + `_mask.png`, 255=piste 0=hors.

### Étape 4 — Entraîner le U-Net

```bash
python -m src.mask.training.train_mask \
    --images data/256_128/256_128 \
    --masks  data/masks_auto \
    --output models/mask_v1 \
    --epochs 50 \
    --batch  32
```

Sorties dans `models/mask_v1/` :
- `best.pth` — poids PyTorch
- `best.onnx` — prêt pour Jetson Nano

### Étape 5 — Inférence sur la voiture

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("models/mask_v1/best.onnx")

def predict(image_rgb):  # np.ndarray (128, 256, 3) uint8
    x = image_rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[np.newaxis]    # (1, 3, 128, 256)
    out = session.run(None, {"image": x})[0]  # (1, 1, 128, 256)
    return (out[0, 0] > 0.5).astype(np.uint8) * 255
```

---

## Métriques à suivre

| Métrique | Cible | Commande |
|---|---|---|
| IoU (Intersection over Union) | > 0.85 | affiché pendant `train_mask.py` |
| % masques auto exploitables | > 80% | vérification visuelle |
| Temps inférence ONNX | < 10ms CPU | `onnxruntime` benchmark |

---

## Ce qu'on N'a PAS fait (et pourquoi)

| Approche | Pourquoi écartée |
|---|---|
| **SAM (Segment Anything)** | Trop lourd pour Jetson Nano, overkill pour scène simple |
| **DeepLab / SegFormer** | Trop lourd, > 10ms sur CPU |
| **Annotation manuelle 2793 images** | Perte de temps — les masques auto couvrent 80%+ |
| **Threshold simple sans CLAHE** | Casse sur les reflets et variations d'éclairage |

---

## Répartition suggérée

| Tâche | Qui |
|---|---|
| Générer masques auto + vérification | Christophe |
| Correction manuelle 300 images | Léandre + Mathieu |
| Entraînement U-Net + validation IoU | Christophe |
| Intégration dans inference_realcar.py | Tous |

---

## Liens utiles

- Dataset : `data/256_128/256_128/` (2793 images de Boris)
- Code : `src/mask/training/mask_generator.py`, `src/mask/training/unet_model.py`
- Modèle actif : `models/mask_v1/best.onnx` (après entraînement)
- Contact Boris pour questions sur le dataset : via Discord projet

---

*Dernière mise à jour : 2026-05-21*

# Hardware — Voiture Réelle (Jetson Nano)

> Document de référence pour le passage simulation → hardware réel.
> Boris s'occupe de la partie mécanique/montage. Le software/ML est à charge de lekrikri.

---

## Architecture hardware complète

```
Batterie LiPo
      ↓
[Carte puissance centrale + grand radiateur]
      ├──→ [Matek Systems UBEC Duo]
      │         └── 5V → Jetson Nano (alimentation)
      │              + monitoring batterie
      └──→ [Flipsky FSESC Mini V6.7 Pro]  (VESC — ESC open-source)
                ├── Moteur brushless Traxxas BLSS 3300
                ├── Servo direction
                └── USB → Jetson Nano (commandes + télémétrie)

Jetson Nano — ports USB:
      ├── Luxonis OAK-D Lite autofocus (p/n: a00483)   USB3 — caméra RGB + stéréo depth
      ├── Flipsky FSESC Mini V6.7 Pro                  USB  — contrôle moteur/direction
      ├── TP-Link dongle                               USB  — WiFi
      └── Logitech dongle                              USB  — clavier sans fil
```

---

## Composants clés

### Luxonis OAK-D Lite (caméra)
- RGB + stéréo depth (disparity map)
- Autofocus
- Connectée en USB3 au Jetson Nano
- Bibliothèque Python : `depthai`
- **Rôle IA** : fournir une depth map → extraire des raycasts virtuels

### Flipsky FSESC Mini V6.7 Pro (contrôleur moteur)
- VESC (open-source ESC) — protocole VESC standard
- Connecté USB au Jetson Nano
- Bibliothèque Python : `pyvesc`
- **Entrées** : duty cycle (throttle), angle servo (steering)
- **Sorties lues** : RPM moteur → vitesse réelle (résout le bug speed=0.0 du simulateur!)

### Matek Systems UBEC Duo
- Régulateur 5V pour alimenter le Jetson Nano
- Monitoring batterie intégré

### Jetson Nano
- GPU Maxwell 128 cores
- 4GB RAM
- Cible d'inférence : ONNX Runtime GPU ou TensorRT FP16

---

## Problème central — Sim-to-Real Gap

| Simulation (Unity) | Voiture réelle |
|--------------------|---------------|
| 20 raycasts parfaits [0,1] | Depth map OAK-D (image 2D) |
| speed = 0.0 (hardcodé) | Vitesse réelle via VESC (RPM) |
| gRPC port 5005 | depthai + pyvesc |
| ~20 FPS gRPC | ~30 FPS OAK-D |

---

## Solution : Bridge Depth Map → Raycasts Virtuels

L'OAK-D Lite produit une **depth map** (distance en mm pour chaque pixel).
On peut simuler les 20 raycasts du modèle en échantillonnant la depth map à 20 angles :

```
fov=180°, nbRay=20 → angles de -90° à +90° par pas de ~9.47°

Pour chaque angle i:
    col = int(width/2 + tan(angle_i) × focal_length)
    distance = depth_map[row_center, col]       # en mm
    ray_i = min(distance / MAX_DISTANCE, 1.0)   # normaliser [0,1]
```

**MAX_DISTANCE** : à calibrer selon la piste réelle (ex: 3000mm = 3m).

---

## Pipeline logiciel cible

```python
# Pseudo-code inference_realcar.py

import depthai as dai
import pyvesc
import onnxruntime as ort
import numpy as np

# 1. Init caméra OAK-D
pipeline = dai.Pipeline()
# ... setup depth stream

# 2. Init VESC
vesc = pyvesc.VESC('/dev/ttyUSB0')  # ou ttyACM0

# 3. Charger modèle ONNX
sess = ort.InferenceSession('models/v18/best.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 4. Charger Z-score stats
ray_stats = json.load(open('models/ray_stats.json'))
ray_mu = np.array(ray_stats['mean'])
ray_sigma = np.array(ray_stats['std'])

# Boucle principale
while True:
    # Lire depth map
    depth_frame = get_depth_frame()

    # Bridge → raycasts virtuels
    rays = depth_to_rays(depth_frame, n_rays=20, fov=180)   # [0,1]

    # Z-score (même normalisation que l'entraînement!)
    rays_norm = (rays - ray_mu) / ray_sigma

    # Features dérivées
    n = len(rays_norm); half = n // 2
    asymmetry = (rays_norm[half:].sum() - rays_norm[:half].sum()) / (rays_norm.sum() + 1e-8)
    front_ray  = float(rays_norm[half-1:half+1].mean())
    min_ray    = rays_norm.min()
    x = np.concatenate([rays_norm, [asymmetry, front_ray, min_ray]]).astype(np.float32)

    # Inférence ONNX
    pred = sess.run(None, {'rays': x.reshape(1, -1)})[0][0]

    # Smoothing + heuristique accel (même qu'inference.py)
    steering     = float(np.clip(pred[0], -1.0, 1.0))
    acceleration = compute_accel_heuristic(pred, rays)  # front_raw

    # Envoyer commandes VESC
    vesc.set_servo(steering_to_servo(steering))
    vesc.set_duty_cycle(accel_to_duty(acceleration))
```

---

## Étapes de développement

### Phase 1 — Environnement Jetson Nano
- [ ] Installer `onnxruntime-gpu` sur Jetson Nano
- [ ] Installer `depthai` (OAK-D SDK)
- [ ] Installer `pyvesc`
- [ ] Tester connexion OAK-D (`python3 -c "import depthai"`)
- [ ] Tester connexion VESC (`pyvesc` → lire RPM)
- [ ] Identifier le port USB du FSESC (`/dev/ttyUSB0` ou `ttyACM0`)

### Phase 2 — Bridge Depth → Raycasts
- [ ] Capturer une depth frame OAK-D
- [ ] Implémenter `depth_to_rays()` (20 angles, fov=180°)
- [ ] Calibrer `MAX_DISTANCE` sur la piste réelle
- [ ] Vérifier que les valeurs [0,1] sont cohérentes avec la simulation

### Phase 3 — Inférence ONNX sur Jetson
- [ ] Exporter `models/v18/best.pth` → `best.onnx` (si pas déjà fait)
- [ ] Copier `best.onnx` + `ray_stats.json` sur le Jetson
- [ ] Tester inférence ONNX seule (bench latence)
- [ ] Viser < 5ms par frame (largement faisable)

### Phase 4 — Intégration complète
- [ ] Écrire `src/inference_realcar.py`
- [ ] Tester en statique (voiture soulevée)
- [ ] Calibrer steering_to_servo() et accel_to_duty()
- [ ] Premier test sur piste réelle

### Phase 5 — Optimisation TensorRT
- [ ] `trtexec --onnx=best.onnx --fp16 --saveEngine=best.plan`
- [ ] Switcher vers TensorRT dans inference_realcar.py
- [ ] Mesurer gain latence FP16

---

## Points critiques à ne pas oublier

1. **ray_stats.json obligatoire** : le Z-score doit être identique entraînement ↔ inférence
2. **MAX_DISTANCE à calibrer** : dépend de la piste réelle (3m ? 5m ?)
3. **Fréquence OAK-D** : ~30 FPS → le modèle peut tourner plus vite qu'en simulation
4. **Port USB FSESC** : vérifier `/dev/ttyUSB*` ou `/dev/ttyACM*` avec `ls /dev/tty*`
5. **La vitesse réelle est maintenant disponible** via VESC RPM → peut être utilisée comme feature pour un futur v20

---

## Export ONNX (depuis le PC de dev)

```bash
cd /home/lekrikri/Projects/G-CAR-000
python3 -c "
import torch, json
from src.model import load_model

model = load_model('models/v18/best.pth')
model.eval()
dummy = torch.zeros(1, 23)  # 20 rays + 3 derived features
torch.onnx.export(
    model, dummy, 'models/v18/best.onnx',
    input_names=['rays'],
    output_names=['actions'],
    opset_version=11
)
print('Export OK → models/v18/best.onnx')
"
```

Transférer ensuite sur le Jetson :
```bash
scp models/v18/best.onnx models/ray_stats.json user@jetson-ip:~/behavioral-cloning/
```

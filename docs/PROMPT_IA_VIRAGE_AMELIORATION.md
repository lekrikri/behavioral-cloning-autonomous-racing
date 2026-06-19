# Prompt IA — Amélioration navigation en virage, voiture autonome lignes blanches

> Copier-coller dans Gemini / ChatGPT / Claude

---

## Contexte : Voiture RC autonome sur piste lignes blanches

Je travaille sur une voiture RC autonome (Jetson Nano + OAK-D Lite) qui suit une piste délimitée par des **lignes blanches peintes au sol**, en extérieur et en intérieur.

**Stack vision** :
- Pipeline : GaussianBlur → CLAHE → HSV inRange (V≥130 + Otsu adaptatif, S≤55) → Morph → blobs
- Résolution : 512×256 px @ 12fps, caméra avec crop top 35% (simule inclinaison vers le bas)
- Détection : `get_blobs()` → garde uniquement le blob le **plus à gauche** et le **plus à droite**
- Erreur latérale : centroïde entre blob gauche et blob droit → PD controller (KP=0.012, KD=0.003)
- Vitesse : 10% duty cycle (très lente, ~0.3 m/s)

**État machine à états** :
- `FIXED` : n_blobs=2, err normal, throttle=0.10
- `BLIND` : n_blobs=0 → throttle=0 (arrêt complet)
- `CORNER [L]` : blob compact détecté (marqueur de virage) → steer=±0.85 pendant 15 frames

---

## Problème principal : virages difficiles

En virage à 90°, la caméra voit :
- Des **lignes en tirets** (marquages de carrefour) au lieu de lignes continues
- Des **blobs fragmentés** à gauche et à droite
- err ≈ -4 → steer ≈ -0.06 : la voiture reste quasi droite alors qu'elle devrait beaucoup tourner
- Le mode CORNER ne se déclenche pas toujours (condition sur la compacité du blob L)

**Ce que voit la caméra en virage** :
```
[vue caméra]
┌─────────────────────────────┐
│  _ _ fond de salle _ _ _ _ │  ← faux positifs filtrés par ROI
│                             │
│  ╔══╗          ╔══╗        │  ← tirets de la ligne extérieure (virage)
│  ╚══╝          ╚══╝        │
│  ║                  ║      │  ← lignes droites (avant le virage)
│  ║                  ║      │
└─────────────────────────────┘
```

Le blob "le plus à gauche" peut être un tiret loin dans le virage, et le "plus à droite" aussi → le centre calculé est quasi-nul alors qu'en réalité la voiture doit fortement tourner.

---

## Ce qui fonctionne déjà

- **Ligne droite** : err ≈ 0, steer ≈ 0, b=2 → parfait
- **Virage détecté** : mode CORNER injecte steer=±0.85 pendant 15 frames si blob compact vu
- **Adaptation éclairage** : Otsu recalcule le seuil V à chaque frame (intérieur/extérieur)
- **Filtrage flèches sol** : blobs compacts area>3000 et 0.5<aspect<2.0 ignorés

---

## Questions

### 1. Meilleure stratégie d'erreur en virage

Avec des lignes en tirets en virage, comment calculer une erreur latérale fiable ?

Options envisagées :
- **A** : Utiliser l'asymétrie des blobs (différence de surface blob_gauche vs blob_droit) comme signal de virage
- **B** : Détecter le "flow" optique des lignes entre frames (si les lignes se déplacent vers la droite → tourner à droite)
- **C** : Machine à états avec zone de lookahead différente en virage (ROI décalée vers la ligne extérieure)
- **D** : Estimer la courbure de la piste via les positions des blobs sur plusieurs frames passées

Laquelle est réaliste sur Jetson Nano CPU (Python 3.6.9, ~80ms/frame budget) ?

### 2. Détection précoce des virages

Comment détecter un virage à venir **avant** que la voiture soit dedans (anticipation) ?
- Les raycasts visuels (20 valeurs [0,1] = espace libre) peuvent-ils servir à ça ?
- Ou faut-il une zone de lookahead plus haute dans l'image ?

### 3. Vitesse adaptative en virage

La voiture roule à 10% duty constant. En virage, elle devrait ralentir.
Comment estimer le radius du virage depuis les blobs pour adapter la vitesse ?

---

## Contraintes techniques

- Python 3.6.9 (Jetson Nano) — pas de walrus `:=`, pas de match/case
- Pas de GPU pour vision (cuDNN utilisé par depthai)
- Budget CPU : ~80ms/frame
- Packages disponibles : numpy, opencv, scipy
- **Pas d'apt/pip upgrade autorisé**
- depthai 2.x (API différente de 3.x)

---

## Code actuel (extraits clés)

```python
# Constantes PD
KP = 0.012
KD = 0.003
CORNER_DURATION = 15   # frames virage forcé (~1.25s @ 12fps)
TRACK_WIDTH_EST_PX = 385

# Erreur depuis 2 blobs (gauche + droite)
def err_from_two_lines(blobs):
    CLEAR_LEFT, CLEAR_RIGHT = 180, 332
    left_blobs  = [b for b in blobs if b["cx"] < CLEAR_LEFT]
    right_blobs = [b for b in blobs if b["cx"] > CLEAR_RIGHT]
    if left_blobs and right_blobs:
        cx_l = max(left_blobs,  key=lambda b: b["area"])["cx"]
        cx_r = min(right_blobs, key=lambda b: b["area"])["cx"]
        return (cx_l + cx_r) // 2 - CAM_W // 2, "LR"
    if left_blobs:
        cx_l = max(left_blobs, key=lambda b: b["area"])["cx"]
        return cx_l + TRACK_WIDTH_EST_PX // 2 - CAM_W // 2, "L"
    if right_blobs:
        cx_r = min(right_blobs, key=lambda b: b["area"])["cx"]
        return cx_r - TRACK_WIDTH_EST_PX // 2 - CAM_W // 2, "R"
    return None, "?"

# Détection coin L (marqueur de virage)
def detect_corner_blob(mask):
    # blob compact (area>=1500, aspect<1.8, cy>=35%) dans mask_wide (ROI 45%)
    ...

# Machine à états CORNER
if corner_blob is not None and n_blobs <= 2:
    self.corner_dir   = 1.0 if corner_blob["cx"] > CAM_W // 2 else -1.0
    self.corner_mode  = True
    self.corner_count = CORNER_DURATION  # 15 frames
```

---

## Demande

Propose des **modifications concrètes en Python 3.6** pour améliorer la navigation en virage sur une piste à lignes blanches avec marquages au sol variables (continus, tirets, carrefour). Code directement applicable, pas de pseudo-code.

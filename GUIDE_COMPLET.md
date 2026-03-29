# Guide Complet — Robocar Racing Simulator
## Tout comprendre, de A à Z

> Ce document est ton guide personnel d'apprentissage.
> Il explique **pourquoi** chaque composant existe, **comment** il fonctionne,
> et **ce que tu apprends** en le construisant.

---

## Table des matières

1. [Le problème qu'on résout](#1-le-problème-quon-résout)
2. [L'architecture globale du système](#2-larchitecture-globale)
3. [Behavioral Cloning — le principe IA](#3-behavioral-cloning--le-principe-ia)
4. [Le simulateur Unity (ML-Agents)](#4-le-simulateur-unity--ml-agents)
5. [Les données — comment la voiture "voit"](#5-les-données--comment-la-voiture-voit)
6. [Le modèle IA — architecture détaillée](#6-le-modèle-ia--architecture-détaillée)
7. [Pourquoi le 1D-CNN est meilleur que le MLP](#7-pourquoi-le-1d-cnn-est-meilleur)
8. [La loss function — comment on mesure l'erreur](#8-la-loss-function)
9. [L'entraînement — comment le modèle apprend](#9-lentraînement)
10. [Les augmentations de données](#10-les-augmentations-de-données)
11. [Le WeightedRandomSampler — équilibrer les données](#11-le-weightedrandomsampler)
12. [L'inférence — brancher le modèle sur la voiture](#12-linférence)
13. [L'export ONNX — préparer le Jetson Nano](#13-lonnx--pour-le-jetson-nano)
14. [La manette vs le clavier](#14-manette-vs-clavier)
15. [Roadmap détaillée et progressive](#15-roadmap-détaillée)
16. [Glossaire des termes techniques](#16-glossaire)

---

## 1. Le problème qu'on résout

### Le but du projet

Tu dois entraîner une voiture autonome à **rester sur une piste** dans un simulateur.
La voiture n'a pas de caméra — elle perçoit le monde via des **rayons laser** (raycasts).
Chaque rayon mesure la distance jusqu'au bord de la piste dans une direction.

```
                     🏁 Piste
          ┌──────────────────────────────────┐
          │      /  |  |  \                  │
          │    /    |  |    \  ← rayons      │
          │   🚗                             │
          │  (voiture)                       │
          └──────────────────────────────────┘
```

### Ce que doit faire l'IA

À chaque instant (frame), la voiture reçoit :
- Les **N distances** mesurées par les rayons (ex: 10 rayons)
- Sa **vitesse** actuelle

Et doit décider :
- De quel côté tourner (**steering**) : valeur entre -1 (gauche) et +1 (droite)
- Si elle accélère ou freine (**acceleration**) : valeur entre -1 (frein) et +1 (gaz)

### Pourquoi c'est difficile

Sans IA, il faudrait écrire des règles à la main :
```
SI rayon_gauche < 0.3 ALORS tourner_droite
SI rayon_droite < 0.3 ALORS tourner_gauche
...
```
C'est fragile. L'IA va **apprendre ces règles automatiquement** en observant un humain conduire.

---

## 2. L'architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATEUR UNITY                              │
│   ┌──────────┐   Rayons (distances)   ┌─────────────────────┐   │
│   │  Piste   │ ──────────────────────▶│   Agent (voiture)   │   │
│   │          │ ◀────────────────────── │                     │   │
│   └──────────┘  Actions (steer+accel) └─────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ gRPC (protocole réseau, port 5005)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON (ton code)                             │
│                                                                  │
│  Phase COLLECTE:                                                 │
│   Toi (manette) ──▶ input_manager.py ──▶ data_collector.py      │
│                                              │                   │
│                                         data/run_01.csv          │
│                                                                  │
│  Phase ENTRAÎNEMENT:                                             │
│   data/*.csv ──▶ dataset.py ──▶ model.py ──▶ train.py           │
│                                              │                   │
│                                         models/best.pth          │
│                                                                  │
│  Phase INFÉRENCE:                                                │
│   models/best.pth ──▶ inference.py ──▶ actions vers Unity       │
└─────────────────────────────────────────────────────────────────┘
```

### Les 3 phases du projet

| Phase | Ce que tu fais | Ce que le programme fait |
|-------|---------------|--------------------------|
| **Collecte** | Tu conduis avec la manette | Enregistre (observations, actions) dans CSV |
| **Entraînement** | Tu lances `train.py` | Le modèle apprend à imiter ta conduite |
| **Inférence** | Tu lances `inference.py` | Le modèle conduit à ta place |

---

## 3. Behavioral Cloning — le principe IA

### C'est quoi exactement ?

Le **Behavioral Cloning (BC)** = "apprendre par imitation".

**Idée simple :** Si tu montres à un enfant comment faire du vélo 1000 fois,
il finit par associer *"situation → action correcte"*.
Le BC fait la même chose avec un réseau de neurones.

```
Données d'entraînement:
┌────────────────────┬──────────────────┐
│ Observation (X)    │ Action (Y)        │
├────────────────────┼──────────────────┤
│ [0.8, 0.3, 0.9...] │ steering = +0.4  │ → virer à droite (obstacle à gauche)
│ [0.9, 0.9, 0.2...] │ steering = -0.6  │ → virer à gauche (obstacle à droite)
│ [0.8, 0.9, 0.8...] │ steering = +0.0  │ → tout droit (piste libre)
└────────────────────┴──────────────────┘

Le modèle apprend: f(X) → Y
```

### Pourquoi pas du Reinforcement Learning (RL) ?

Le sujet dit **pas de RL** pour commencer. Voici pourquoi c'est sage :

| Aspect | Behavioral Cloning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Données** | Ta conduite (supervisé) | Expérimentation (essai/erreur) |
| **Vitesse** | Rapide (quelques minutes) | Lent (des heures/jours) |
| **Difficulté** | Accessible | Complexe (reward design) |
| **Qualité max** | Limitée par l'expert | Peut dépasser l'humain |
| **Quand utiliser** | Phase 1 ← **ici** | Phase bonus |

### Le problème du Covariate Shift

C'est le talon d'Achille du BC :

```
Entraînement: voiture toujours au centre → modèle n'a jamais vu "je suis sur le bord"
Inférence:    petite erreur → voiture dévie → situation inconnue → erreur plus grande → ...
```

**Solution** : Collecter des données de récupération (conduire depuis les bords vers le centre).
Et plus tard : **DAgger** (Dataset Aggregation) — laisser le modèle conduire et annoter ses erreurs.

---

## 4. Le simulateur Unity (ML-Agents)

### C'est quoi ML-Agents ?

**Unity ML-Agents** est un toolkit qui permet à Unity (moteur de jeu) de communiquer avec Python.
Il crée un "pont" entre la simulation 3D et ton code Python.

```
Unity (C#) ←──── gRPC ────▶ Python (mlagents-envs)
  - physique voiture           - décide des actions
  - raycasts                   - entraîne le modèle
  - rendu graphique            - collecte données
```

### Comment se connecter

```python
from mlagents_envs.environment import UnityEnvironment

# Lance (ou se connecte à) le simulateur
env = UnityEnvironment(
    file_name=None,  # None = simulateur déjà lancé manuellement
    base_port=5005,  # ⚠️ port 5004 bloqué par Windows RTP → utiliser 5005
    additional_args=["--config-path", "config.json"]
)
env.reset()
```

### La config des agents

```json
{
  "agents": [
    {
      "fov": 180,   ← champ de vision en degrés (1-180)
      "nbRay": 20   ← nombre de rayons — 20 est optimal (+11.7 pts R² vs 10)
    }
  ]
}
```

**fov = 180** : les rayons couvrent un demi-cercle devant la voiture.
**nbRay = 20** : 20 rayons espacés uniformément dans ce demi-cercle (optimal, validé v18).

```
fov=180, nbRay=20:
Ray 0: -90°  Ray 1: -80°  ...  Ray 19: +90°

     ← gauche              devant              droite →
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  ↖  ↖  ↖  ↖  ↗  ↗  ↗  ↗  ↑  ↑  ↑  ↑  ↗  ↗  ↗  ↗  ↘  ↘  ↘  ↘
```

---

## 5. Les données — comment la voiture "voit"

### Format des observations

Chaque "frame" (instant) est un vecteur de nombres :

```
Observation = [ray_0, ray_1, ..., ray_19]  + features dérivées (n_derived=3)

Rayons (20 valeurs brutes [0,1]):
[0.95, 0.88, 0.20, ..., 0.94]
  ↑     ↑     ↑            ↑
 loin  loin  MÛRE!        loin
       (obstacle proche à gauche → tourner à droite)

Features dérivées calculées automatiquement:
  asymmetry = (somme_droite - somme_gauche) / total  → virage prédit
  front_ray  = moyenne 2 rayons centraux             → obstacle frontal
  min_ray    = rayon minimal                         → danger immédiat
```

⚠️ **speed = 0.0 hardcodé** : Unity ne transmet pas la vitesse réelle (client.py ligne 162).
La vitesse ne peut PAS être utilisée comme feature ML sans modifier les scripts C# Unity.

**Valeur proche de 1.0** = rayon long = rien dans cette direction = dégagé
**Valeur proche de 0.0** = rayon court = obstacle proche = DANGER

### Format du CSV de collecte

```csv
episode_id,  step, timestamp, ray_0, ..., ray_19, speed, steering, acceleration
1711000000,  0,    1711.23,   0.95,  ..., 0.94,   0.0,   +0.15,    +0.80
1711000000,  1,    1711.28,   0.93,  ..., 0.91,   0.0,   +0.25,    +0.75
1711000000,  2,    1711.33,   0.88,  ..., 0.88,   0.0,   +0.45,    +0.60
```

- **episode_id** : timestamp de début de session — unique par collecte
- **step** : compteur de frame dans l'épisode → nécessaire pour le split temporel

### Pourquoi normaliser tout dans [0, 1] ou [-1, 1] ?

Les réseaux de neurones apprennent beaucoup mieux quand les valeurs sont dans des plages similaires.
Si ray_0 = 500m et speed = 0.65, le réseau aurait du mal à les traiter ensemble.
Le simulateur normalise directement : distances → [0,1], vitesse → [0,1].

---

## 6. Le modèle IA — architecture détaillée

### Le réseau de neurones : une boîte de décision

Un réseau de neurones est une fonction mathématique apprise :
```
f([ray_0, ..., ray_9, speed]) → [steering, acceleration]
```

### Architecture RobocarSpatial (modèle actuel — v18) ✅

```
Input [23] = [ray_0..ray_19 (Z-scorés), asymmetry, front_ray, min_ray]
                         ↓
              Séparer en 2 branches:
                         │
          ┌──────────────┴──────────────┐
          ▼                              ▼
      Rays [20]                    Derived [3]
      reshape (2, 10)              FC(3→16) → ReLU
      ↓
      Conv1d(2→12, kernel=3, padding=1)
      BatchNorm + ReLU
      ↓
      flatten [120]
          │
          └────────────┬──────────────┘
                       ▼
                 Concat [136]
                       ↓
                  Linear(136→96) → BN → ReLU → Dropout(0.2)
                       ↓
                  Linear(96→48) → BN → ReLU
                       ↓
              ┌────────┴────────┐
              ▼                  ▼
         steer_head           accel_head
         Linear(48→1)         Linear(48→1)
         Tanh → [-1,1]        Sigmoid → [0,1] (logit si bimodal_accel=True)
              └────────┬────────┘
                       ▼
              Output [2]: [steering, acceleration]
```

### Architecture MLP (baseline)

```
Input [11]
  ↓
Linear(11→128) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(128→64) → BatchNorm → ReLU → Dropout(0.1)
  ↓
Linear(64→32) → ReLU
  ↓
Linear(32→2) → Tanh
  ↓
Output [2]: [steering, acceleration] dans [-1, 1]
```

**Chaque composant a un rôle :**

```
Linear    : combinaison linéaire (comme y = ax + b, mais avec beaucoup de x et b)
BatchNorm : normalise les valeurs entre chaque couche → stabilise l'entraînement
ReLU      : activation non-linéaire (max(0, x)) → permet d'apprendre des patterns complexes
Dropout   : éteint aléatoirement des neurones pendant l'entraînement → évite l'overfitting
Tanh      : borne la sortie dans [-1, 1] → parfait pour steering et accel
```

### Architecture CNN (recommandée — Gemini)

```
Input [11] = [ray_0...ray_9, speed]
                  ↓
           Séparer en 2 branches:
                  │
     ┌────────────┴──────────────┐
     ▼                           ▼
  Rays [10]                   Speed [1]
  reshape (1,10)              │
  ↓                           │
  Conv1d(1→16, k=3)           │
  BatchNorm + ReLU            │
  ↓                           │
  Conv1d(16→32, k=3)          │
  BatchNorm + ReLU            │
  ↓                           │
  GlobalAvgPool → [32]        │
  ↓                           │
  └────────────┬──────────────┘
               ▼
            Concat [33] (32 features CNN + 1 speed)
               ↓
            Linear(33→64) → BN → ReLU
               ↓
            Linear(64→32) → ReLU
               ↓
            Linear(32→2) → Tanh
               ↓
            [steering, acceleration]
```

**Pourquoi deux branches ?**
La vitesse est une information globale (scalaire), pas spatiale.
On la traite séparément et on la fusionne **après** l'extraction de features spatiales.

---

## 7. Pourquoi le 1D-CNN est meilleur

### Le problème du MLP avec des raycasts

Le MLP traite chaque rayon comme **indépendant**. Il ne "voit" pas la structure spatiale.

```
Situation: obstacle à gauche (rayons 0,1,2 faibles)
MLP voit:  ray_0=0.1, ray_1=0.15, ray_2=0.2, ray_3=0.8, ...
           → traite chaque nombre séparément, pas leur relation

CNN voit:  [0.1, 0.15, 0.2, 0.8, ...]
           Conv kernel size=3 → "je détecte un mur sur les 3 premiers rayons"
           → comprend que c'est un MOTIF spatial continu
```

### Analogie visuelle

Imagine des rayons comme un "mini signal audio 1D" :

```
          ← gauche  |  droite →
1.0  ┤.....                .....
0.8  ┤   ..              ..
0.6  ┤     .           ..
0.4  ┤      .        ..
0.2  ┤       .......
0.0  ┼─────────────────────────
           Obstacle à gauche!

Un Conv1d kernel=3 va détecter ce "creux" comme une feature = OBSTACLE À GAUCHE
```

### Les filtres CNN apprennent automatiquement

Après entraînement, les kernels CNN apprendront probablement :
- Kernel 1 : "obstacle frontal" (rayons centraux faibles)
- Kernel 2 : "mur à gauche" (rayons gauches faibles)
- Kernel 3 : "ouverture large" (tous les rayons élevés)
- etc.

### Comparaison avec notre implementation

| | MLP | CNN |
|--|-----|-----|
| **Params** | 12 322 | 6 178 |
| **Qualité spatiale** | Faible | Haute |
| **Latence Jetson** | < 0.5ms | ~1ms |
| **Utiliser pour** | baseline rapide | production |

---

## 8. La Loss Function

### C'est quoi une loss function ?

C'est la mesure de l'erreur du modèle. Plus la loss est basse, meilleure est la prédiction.

```python
pred   = modele([0.9, 0.2, 0.8, ..., 0.7])  # → [+0.3, +0.8]  (prédit)
target = [+0.5, +0.7]                         # (ce que l'humain a fait = vrai)

erreur = mesure(pred, target)  # → loss
```

### MSE vs Huber : pourquoi on change

**MSE (Mean Squared Error)** :
```
erreur_grande = (0.3 - 0.5)² = 0.04  → OK
erreur_outlier = (0.1 - 0.9)² = 0.64 → ÉNORME influence sur l'apprentissage
```
Problème : un seul coup de volant brusque dans le dataset peut déstabiliser tout l'entraînement.

**Huber Loss** (le meilleur des deux mondes) :
```
Si |erreur| < δ :  L = ½ × erreur²    (comme MSE → précis pour les petites erreurs)
Si |erreur| ≥ δ :  L = δ × |erreur| - ½δ²  (comme L1 → robuste aux outliers)
```

```
         MSE      │      Huber (δ=1)
         (quadra- │      (quadratique puis linéaire)
         tique)   │
         ↗        │      ↗
        ↗         │     ↗
       ↗          │    /
      ↗           │   /
─────────────     │ ─────────────
-1  0  +1  erreur │ -1  0  +1  erreur
```

### BimodalLoss (v18 — actuelle)

L'accélération est traitée comme un problème de **classification binaire** (rapide ou lente) :

```python
BimodalLoss = 0.88 × HuberLoss(steering) + 0.12 × BCEWithLogitsLoss(accel > 0.25)
```

**Pourquoi 88/12 et pas 70/30 ?**
- Le steering est le levier principal de survie — on le pondère encore plus
- L'accel bimodale (go/no-go) est plus simple à apprendre qu'une régression continue
- La tête accel retourne un logit → BCEWithLogitsLoss (AMP-safe, pas de double sigmoid!)

### PairwiseSmoothingLoss (PSL) — clé anti-zigzag

```python
# Penalise les changements brusques de steering entre frames consécutives
PSL = λ × mean( (steer[t+1] - steer[t])² )    # λ = 0.30

loss_totale = BimodalLoss + PSL
```

**Condition obligatoire :** nécessite `temporal_split=True` — sinon les frames consécutives
dans le batch proviennent d'épisodes différents et la pénalité n'a aucun sens!

**Résultat mesuré :** Élimine presque totalement le zigzag résiduel (jerk moyen divisé par ~3).

---

## 9. L'entraînement

### Boucle d'entraînement (comment le modèle apprend)

```
Pour chaque époque:
  Pour chaque batch de données:
    1. Forward pass:   pred = modele(observations)
    2. Calcul loss:    loss = loss_fn(pred, actions_vraies)
    3. Backward pass:  calculer les gradients (dérivées de la loss)
    4. Update:         ajuster les poids dans la bonne direction

  Évaluer sur validation set (données que le modèle n'a pas vues)
  Sauvegarder si meilleur score
```

### L'optimizer Adam

**Adam (Adaptive Moment Estimation)** est l'algorithme qui ajuste les poids.

```
Idée: chaque poids a son propre taux d'apprentissage
Si un poids oscille beaucoup → réduire son lr
Si un poids ne bouge pas → augmenter son lr
```

C'est beaucoup plus intelligent que le simple gradient descent.

### Learning Rate Scheduler (ReduceLROnPlateau)

```
Si la val_loss ne s'améliore pas depuis 5 époques:
    lr = lr × 0.5   (diviser le LR par 2)
```

Cela permet une convergence fine : au début on fait de grands pas, puis des petits.

### Mixed Precision (float16)

Normalement les calculs sont en float32 (32 bits). Avec mixed precision:
- Calculs GPU en **float16** (2× plus rapide, 2× moins de mémoire)
- Mise à jour des poids en **float32** (précision conservée)

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    pred = model(obs)   # calcul rapide en fp16
    loss = loss_fn(pred, action)
scaler.scale(loss).backward()  # gradient précis en fp32
```

### Early Stopping

```
Si val_loss ne s'améliore pas depuis patience=25 époques:
    Arrêter l'entraînement → évite l'overfitting
```

**Overfitting** = le modèle mémorise les données d'entraînement mais généralise mal.
```
Epoch 20: train_loss=0.01, val_loss=0.01 → OK
Epoch 50: train_loss=0.001, val_loss=0.05 → OVERFITTING (trop spécialisé)
```

### Temporal Split (CRITIQUE pour PSL)

Au lieu de mélanger les frames aléatoirement, on respecte l'ordre chronologique :

```
Dataset (N frames, triées par épisode+step):
├── Train  : frames 0    → 75%N   (75%)  ← no shuffle!
├── Val    : frames 75%N → 90%N   (15%)
└── Test   : frames 90%N → N      (10%)

Split ALÉATOIRE (classique):  train/val/test = tirage au sort → frames mélangées
Split TEMPOREL  (v18):        train/val/test = tranches contiguës → ordre préservé
```

**Pourquoi temporal split + PSL sont inséparables :**
- PSL pénalise `|steer[t+1] - steer[t]|²` dans le batch
- Si frames mélangées → frames "consécutives" dans le batch viennent d'épisodes différents
- PSL pénaliserait des sauts de steering qui n'ont AUCUN sens physique
- Avec temporal split → frames consécutives dans le batch = vraiment consécutives en temps ✅

---

## 10. Les augmentations de données

### Pourquoi augmenter les données ?

Plus le dataset est grand et varié, meilleur sera le modèle.
L'augmentation crée artificiellement de nouveaux samples en modifiant les existants.

### 1. Flip Horizontal (le plus important)

```python
# Original: obstacle à droite → tourner gauche
rays = [0.9, 0.8, 0.7, 0.6, 0.9, 0.9, 0.4, 0.2, 0.1, 0.8]
steering = -0.5  # tourner gauche

# Flipped: obstacle à gauche → tourner droite
rays_flip = [0.8, 0.1, 0.2, 0.4, 0.9, 0.9, 0.6, 0.7, 0.8, 0.9]  # ordre inversé
steering_flip = +0.5  # tourner droite
```

**Effet :** Double le dataset ET équilibre les virages gauche/droite.

### 2. Bruit gaussien adaptatif

```python
# Les rayons lointains (valeur proche de 1.0) sont naturellement plus bruités
# (capteur réel = moins précis à longue distance)
noise_std = 0.01 × (1 + ray_value)
ray_bruité = ray + N(0, noise_std).clamp(0, 1)
```

**Effet :** Le modèle devient robuste aux petites imprécisions du capteur.

### 3. Speed Jitter

```python
speed = speed × (1 + uniform(-0.10, +0.10))  # ±10%
```

**Effet :** Le modèle apprend que la même trajectoire peut être prise à des vitesses légèrement différentes.

### 4. Ray Cutout

```python
# Masquer 1-2 rayons aléatoirement (30% de probabilité)
idx = random.choice(range(n_rays))
rays[idx] = 0.0  # simuler un capteur défaillant
```

**Effet :** Le modèle devient robuste si un capteur tombe en panne (important pour le Jetson réel).

---

## 11. Le WeightedRandomSampler

### Le problème du déséquilibre de classes

En conduisant normalement, la majorité du temps tu vas tout droit :
```
Distribution steering:
  -0.9 à -0.3 : ██ (5%)   ← peu de vrais virages
  -0.3 à -0.1 : ████ (10%)
  -0.1 à +0.1 : ████████████████ (60%)  ← ligne droite dominante!
  +0.1 à +0.3 : ████ (10%)
  +0.3 à +0.9 : ██ (15%)
```

**Conséquence :** Le modèle apprend à toujours aller tout droit → parfait sur ligne droite,
mais catastrophique dans les virages.

### La solution : WeightedRandomSampler

```python
# Calculer le poids de chaque sample = inverse de la fréquence de son steering
poids_steering_0.0 = 1 / (60% samples)  = 1.6  → peu de chance d'être tiré
poids_steering_0.8 = 1 / (5% samples)   = 20.0 → grande chance d'être tiré
```

**Résultat :** Le modèle voit autant de virages que de lignes droites pendant l'entraînement.

---

## 12. L'inférence

### Mode collecte vs mode IA

```
Mode collecte (data_collector.py):
  Toi (manette) → actions → simulateur + sauvegarde CSV

Mode inférence (inference.py):
  Modèle IA → actions → simulateur (sans toi!)
```

### Le SmoothingFilter (adaptatif — v18)

**Problème :** Sans lissage, le modèle peut osciller rapidement (zigzag).

```python
# SmoothingFilter adaptatif (v18):
# alpha_base=0.57, alpha_max=0.92, deadzone=0.06

delta = |pred_steer - smoothed_steer|   # changement brusque?
alpha = 0.57 + (0.92 - 0.57) × min(delta, 1.0)

# En virage (grand delta) → alpha proche de 0.92 → réactif
# En ligne droite (petit delta) → alpha proche de 0.57 → stable

# Deadzone sur le steering:
if |smoothed[0]| < 0.06:
    smoothed[0] = 0.0   # supprime le micro-zigzag en ligne droite
```

**Pourquoi deadzone=0.06 et pas 0.03 ?**
- deadzone=0.03 → micro-zigzag réapparu (oscillations rapides)
- deadzone=0.06 → optimal : supprime le bruit sans émoussér les virages réels
- deadzone=0.12 → trop engourdi, mauvaise réactivité en virage

### La heuristique d'accélération (front_raw — v18)

Le modèle ML ne prédit pas l'accélération de manière fiable (pas de vitesse en input).
On utilise une heuristique géométrique basée sur `front_raw` — le rayon frontal **brut** [0,1] :

```python
# ⚠️ Utiliser obs.rays (AVANT Z-score), pas les rays normalisés!
raw_rays = obs.rays          # valeurs originales [0,1]
front_raw = mean(raw_rays[9:11])  # 2 rayons centraux (rayons 9 et 10)

# front_raw proche de 1.0 → ligne droite, mur loin
# front_raw proche de 0.0 → mur proche, virage serré

geo_base  = max(0.35, 1.0 - 1.2 × |steering|)  # ralentit proportionnellement au virage

if front_raw >= 0.65:
    front_cap = 1.0                              # libre → pleine vitesse!
else:
    front_cap = 0.45 + 0.70 × front_raw         # 0→0.45 | 0.65→0.91

acceleration = clip(min(geo_base, front_cap), 0.35, 0.95)
```

**Pourquoi threshold 0.65 et pas 0.5 ?**
- Avant (0.5) → voiture bridée dès la mi-distance → freinait inutilement sur la ligne droite
- Après (0.65) → voiture en pleine vitesse sur les 2/3 de la distance → **record 24s** ✅

### Le Jerk Tracker (métrique de fluidité)

```python
jerk = |steering[t] - steering[t-1]|  # variation brusque = mauvaise conduite

# Un bon modèle a un jerk faible (conduite fluide)
# Un mauvais modèle a un jerk élevé (zigzag)
# v18 : jerk ~3× plus faible que v10 grâce à PSL
```

---

## 13. L'ONNX — pour le Jetson Nano

### Pourquoi exporter en ONNX ?

Le Jetson Nano tourne sous Linux ARM64. PyTorch peut tourner dessus mais c'est lourd.
**ONNX (Open Neural Network Exchange)** est un format universel pour les modèles IA :

```
PyTorch (.pth)  ──▶  ONNX (.onnx)  ──▶  TensorRT (.plan)
   (format         (format universel)    (optimisé Jetson,
  Python/PyTorch)                         le plus rapide)
```

### Workflow Jetson Nano (plus tard)

```bash
# Sur le PC de développement:
python src/train.py --data data/ --arch cnn  # → models/best.pth + best.onnx

# Sur le Jetson Nano:
trtexec --onnx=best.onnx --fp16 --saveEngine=best.plan
# → moteur TensorRT optimisé pour le hardware Jetson, <1.5ms d'inférence
```

### FP16 sur Jetson Nano

Le Jetson Nano a un GPU Maxwell. En **FP16 (demi-précision)** :
- 2× plus rapide que FP32
- 2× moins de mémoire GPU
- Précision quasi-identique pour l'inférence

---

## 14. Manette vs Clavier

### Pourquoi la manette est bien meilleure pour la collecte

```
Clavier:
  Appuyer Q → steering passe de 0.0 à -1.0 instantanément
  → données très "carrées", peu naturelles

Manette (joystick analogique):
  Pousser doucement → steering = -0.15
  Pousser à fond →   steering = -1.0
  → données fluides, naturelles, comme un vrai conducteur
```

**Impact sur le modèle :**
- Données clavier → modèle qui oscille (apprend des mouvements brusques)
- Données manette → modèle fluide (apprend des mouvements continus)

### Configuration GamepadManager

```python
# Layout Xbox (par défaut):
GamepadManager(
    steer_axis=0,   # Joystick gauche X → steering [-1, +1]
    accel_axis=2,   # Trigger droit RT → accélérer [0, 1] normalisé
    brake_axis=5,   # Trigger gauche LT → freiner [0, 1] normalisé
    use_triggers=True,  # RT-LT = accél/frein (recommandé)
    quit_button=7,  # Start/Options → quitter
)
```

### Auto-détection

```python
# Détecte automatiquement si une manette est branchée
manager = create_input_manager()
# → GamepadManager si manette USB/Bluetooth détectée
# → KeyboardManager sinon
```

---

## 15. Roadmap Détaillée

### Phase 1 — Installation et test (aujourd'hui)

```bash
# 1. Installer les dépendances
cd /home/lekrikri/Projects/G-CAR-000
pip install -r requirements.txt
# ✅ Déjà fait !

# 2. Vérifier le pipeline (données synthétiques)
python src/model.py          # test architecture
python src/dataset.py        # test dataset + augmentations
python src/train.py --data data/synthetic.csv --arch cnn --epochs 30
# ✅ Déjà fait — CNN entraîné, ONNX exporté!
```

**Ce que tu apprends :** PyTorch, forward pass, backprop, architectures CNN/MLP

---

### Phase 2 — Connexion au simulateur

```bash
# 1. Lancer le simulateur
./BuildLinux/RacingSimulator.x86_64

# 2. Test connexion Python
python src/client.py --test-only  # smoke test
python src/client.py              # connexion réelle

# 3. Test input
python src/input_manager.py --gamepad  # test manette
python src/input_manager.py            # auto-détection
```

**Ce que tu apprends :** gRPC, ML-Agents API, communication client/serveur

---

### Phase 3 — Collecte de données

```bash
# Collecte session 1 (circuit 1, ~10 min de conduite)
python src/data_collector.py collect \
  --output data/run_01.csv

# Collecte session 2 (circuit différent ou style différent)
python src/data_collector.py collect \
  --output data/run_02.csv

# Fusionner toutes les sessions
python src/data_collector.py merge \
  --input-dir data/ \
  --output data/dataset_complet.csv
```

**Objectif :** Minimum 10 000 frames, idéalement 30 000+
**Conseils de conduite :**
- Varier les vitesses (ne pas toujours aller vite)
- Inclure des récupérations (partir des bords vers le centre)
- Faire plusieurs fois chaque circuit dans les deux sens
- Si possible, conduire depuis des positions inhabituelles

**Ce que tu apprends :** Collecte de données, qualité des données, biais

---

### Phase 4 — EDA (Exploratory Data Analysis)

```bash
jupyter notebook notebooks/eda.ipynb
```

**Questions à répondre :**
- Quelle est la distribution du steering ? (déséquilibre gauche/droite ?)
- Quelle est la corrélation entre rayons et steering ?
- Y a-t-il des outliers (valeurs aberrantes) ?
- Le dataset est-il suffisamment varié ?

**Ce que tu apprends :** Pandas, Matplotlib, Seaborn, analyse statistique

---

### Phase 5 — Entraînement réel

```bash
# Entraîner avec tes vraies données
python src/train.py \
  --data data/ \
  --arch cnn \
  --loss huber \
  --epochs 100 \
  --batch-size 256 \
  --workers 4

# Voir les courbes d'apprentissage
python src/evaluate.py \
  --model models/best.pth \
  --data data/
```

**Métriques à surveiller :**
| Métrique | Excellent | Bon | À améliorer |
|----------|-----------|-----|-------------|
| MAE Steering | < 0.05 | < 0.10 | > 0.20 |
| MAE Accel | < 0.08 | < 0.15 | > 0.25 |
| R² Steering | > 0.90 | > 0.80 | < 0.70 |

**Ce que tu apprends :** Métriques de régression, overfitting, hyperparamètres

---

### Phase 6 — Inférence en simulation

```bash
# Lancer le simulateur, puis:
python src/inference.py \
  --model models/best.pth \
  --smoothing 0.7

# Ou avec ONNX (plus rapide):
python src/inference.py \
  --model models/best.onnx
```

**Ce que tu observes :**
- La voiture reste-t-elle sur la piste ?
- Conduit-elle de manière fluide ou zigzague ?
- Que se passe-t-il dans les virages serrés ?
- Que se passe-t-il si tu changes de circuit ?

**Ce que tu apprends :** Covariate shift, généralisation, distribution shift

---

### Phase 7 — Itération et amélioration

```
Problème observé → Diagnostic → Solution

Voiture zigzague        → jerk élevé    → PSL λ=0.30 + temporal_split (v18 ✅)
Voiture sort en virage  → peu de données → recollecte focalisée virages
Voiture ralentit trop   → front_cap trop bas → augmenter threshold (0.5→0.65 ✅)
Biais droite            → offset -0.02  → steer -= 0.02 × sign(steer) si virage doux
Crash fin de tour       → plein gaz avant mur → front_raw heuristique (v18 ✅)
Modèle ne généralise    → overfitting   → Dropout, plus de données
```

**Ce que tu apprends :** Debugging IA, diagnostic, hyperparameter tuning

---

### Phase 8 — Résultats atteints (2026-03-29) ✅

| Version | Temps | Améliorations clés |
|---------|-------|-------------------|
| v10 | 1m06 | Premier tour complet, 20 rays |
| v18 | 40s | PSL + temporal split, zéro zigzag |
| v18 + heuristic | 36s | front_raw anticipation virages |
| **v18 + threshold 0.65** | **24s** | **RECORD** — pleine vitesse sur ligne droite |

**Commande record :**
```bash
./BuildLinux/RacingSimulator.x86_64 --mlagents-port 5005 \
  --config-path /home/lekrikri/Projects/G-CAR-000/config.json &
python3 src/inference.py --model models/v18/best.pth
```

---

### Phase Bonus — Améliorations expertes

#### A. DAgger (Dataset Aggregation)

```
1. Entraîner modèle v1 avec tes données humaines
2. Laisser v1 conduire en simulation
3. Toi (expert) tu annotes les bonnes actions sur les états visités par v1
4. Mixer tes données + données DAgger → réentraîner v2
5. Répéter
```

**Pourquoi :** Corrige le covariate shift. Le modèle voit les situations qu'il a générées.

#### B. Quantization pour Jetson Nano

```python
# Quantization INT8 dynamique (PyTorch)
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
)
# Modèle ~4× plus petit, ~2× plus rapide sur CPU

# Export ONNX → TensorRT FP16 (sur Jetson)
# trtexec --onnx=best.onnx --fp16 --saveEngine=best.plan
```

#### C. Reinforcement Learning (PPO)

```
1. Partir du modèle BC comme initialisation de l'acteur PPO
2. Reward = vitesse × cos(angle_piste) - λ × CTE - γ × jerk
3. Laisser le PPO améliorer la conduite en simulant des milliers d'épisodes
4. Le modèle peut dépasser les performances humaines!
```

---

## 16. Glossaire

| Terme | Explication simple |
|-------|-------------------|
| **Behavioral Cloning** | Apprendre par imitation d'un expert |
| **Raycast** | Rayon laser virtuel qui mesure la distance à un obstacle |
| **Forward pass** | Calculer la sortie du réseau (entrée → sortie) |
| **Backward pass** | Calculer les gradients pour ajuster les poids |
| **Gradient** | Direction dans laquelle les poids doivent bouger pour réduire l'erreur |
| **Epoch** | Une passe complète sur tout le dataset d'entraînement |
| **Batch** | Sous-ensemble du dataset traité en une fois (ex: 256 samples) |
| **Overfitting** | Le modèle mémorise le train set mais généralise mal |
| **Loss function** | Mesure de l'erreur entre prédiction et vérité |
| **Optimizer** | Algorithme qui met à jour les poids (Adam, SGD...) |
| **Learning rate** | "Taille du pas" lors de la mise à jour des poids |
| **BatchNorm** | Normalise les valeurs entre les couches → entraînement stable |
| **Dropout** | Éteint des neurones aléatoirement → régularisation |
| **ReLU** | Activation non-linéaire : max(0, x) |
| **Tanh** | Activation qui borne dans [-1, 1] |
| **Conv1d** | Convolution 1D : détecte des motifs locaux dans un signal |
| **Covariate Shift** | Écart entre les données d'entraînement et d'inférence |
| **DAgger** | Technique pour corriger le covariate shift (collecte itérative) |
| **MAE** | Mean Absolute Error : erreur moyenne absolue |
| **RMSE** | Root Mean Square Error : penalise plus les grandes erreurs |
| **R²** | Coefficient de détermination : 1.0 = parfait, 0 = nul |
| **ONNX** | Format universel pour exporter des modèles IA |
| **TensorRT** | Compilateur Nvidia pour optimiser l'inférence |
| **Jerk** | Dérivée de l'accélération : mesure la brutalité des changements |
| **CTE** | Cross-Track Error : distance par rapport au centre de la piste |
| **gRPC** | Protocole réseau utilisé par ML-Agents pour communiquer |
| **Mixed Precision** | Utiliser fp16 pour les calculs GPU → 2× plus rapide |
| **BimodalLoss** | Loss combinée : Huber(steer) + BCE(accel>0.25) → steer régression, accel classification |
| **PSL** | PairwiseSmoothingLoss : pénalise les sauts de steering entre frames consécutives → anti-zigzag |
| **Temporal Split** | Split chronologique (pas aléatoire) : respecte l'ordre des frames → requis pour PSL |
| **front_raw** | Rayon frontal AVANT Z-score [0,1] : utilisé pour l'heuristique d'accélération |
| **Deadzone** | Zone morte sur le steering : valeurs < seuil → forcées à 0 → supprime micro-zigzag |
| **Covariate Shift** | Écart entre les données d'entraînement et d'inférence |
| **DAgger** | Technique pour corriger le covariate shift (collecte itérative) |

---

## Schéma récapitulatif de l'apprentissage

```
                        TON APPRENTISSAGE

  Niveau 1          Niveau 2           Niveau 3          Niveau 4
  ─────────         ─────────          ─────────         ─────────
  Python basics     PyTorch            Architecture IA   Système complet

  NumPy             Tensors            MLP / CNN         Simulation
  Pandas            Autograd           BatchNorm         ML-Agents
  Matplotlib        DataLoader         Dropout           gRPC
  CSV               Training loop      Loss functions    ONNX/TensorRT
                    Adam optimizer     Metrics           Jetson Nano
                    Early stopping     Augmentation
                                       Sampler

  ↑ ─────── ─────── ─────── ─────── ─────── ─────── ─────── ─────── ↑
  Tu connais déjà   ←─────────────── Tu maîtrises maintenant ────────
```

---

> **État actuel (2026-03-29) :** Le modèle v18 tourne le circuit en **24 secondes** — record absolu.
> Conduite fluide (zéro zigzag), trajectoires correctes, vitesse optimisée.
>
> **Prochaine étape :** Modifier les scripts Unity C# pour transmettre la vitesse réelle →
> entraîner v20 avec speed + TTC → remplacer l'heuristique par un ML pur.
> Objectif : battre les 20s avec accélération intelligente.

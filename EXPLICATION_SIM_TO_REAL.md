# Comprendre le passage Simulation → Voiture Réelle

> Ce document explique simplement ce qu'on fait, pourquoi c'est difficile,
> et pourquoi les recommandations des LLMs sont critiques.

---

## 1. Ce qu'on a accompli en simulation

Le modèle v18 conduit parfaitement dans le simulateur Unity.
Il fait le tour en **24 secondes**, conduite fluide, zéro zigzag.

Mais il n'a **jamais vu** la vraie voiture. Il a appris dans un monde virtuel parfait.

```
SIMULATION (ce qu'il connaît)          RÉALITÉ (ce qu'il va rencontrer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raycasts Unity parfaits [0,1]    →     Depth map stéréo bruitée
Piste virtuelle propre           →     Sol réel avec textures, reflets
Voiture stable, pas de tangage   →     Voiture qui tangue en accélérant
Speed = 0.0 (hardcodé)           →     Vitesse réelle via VESC RPM
gRPC port 5005                   →     OAK-D Lite + VESC USB
```

Le modèle ne change pas. Ce qui change, c'est tout ce qui l'entoure.

---

## 2. Le problème central : le Sim-to-Real Gap

### C'est quoi exactement ?

En simulation, le simulateur Unity génère des raycasts **parfaits** :
```
Ray 9  (frontal) = 0.87   → obstacle à 87% de la distance max, précis à 0.001 près
```

Dans la réalité, l'OAK-D Lite calcule des distances par **stéréo vision** (deux caméras) :
```
Ray 9  (frontal) = 0.82 ± 0.05  → même obstacle, mais bruité, parfois NaN, parfois 0
```

**Le modèle a appris sur des données parfaites. Il reçoit des données imparfaites.**
C'est le *Sim-to-Real Gap* — l'écart entre la simulation et la réalité.

### Pourquoi c'est dangereux ?

Si on branche le modèle directement sur l'OAK-D sans adaptation :
- Les valeurs Z-scorées seront fausses (distribution différente)
- Le modèle prédit un steering "normal" mais basé sur des données "anormales"
- La voiture part dans le mur dès le premier virage

---

## 3. Ce qu'on fait pour combler ce gap

### Étape 1 — Convertir la depth map en raycasts virtuels

L'OAK-D Lite ne produit pas de raycasts. Elle produit une **depth map** :
une image où chaque pixel = distance en millimètres.

```
Depth map (image 640×400 pixels, chaque pixel = distance en mm) :
┌─────────────────────────────────────┐
│  1200  1100  1050  980  1200  1500  │ ← pixel (u, v) = distance en mm
│  1180  1090  1020  940  1170  1480  │
│   850   800   600  500   820   900  │ ← ROI (zone d'intérêt)
│   840   790   580  490   810   880  │
│  1300  1200  1100  950  1250  1600  │
└─────────────────────────────────────┘
```

Notre code `depth_to_rays.py` échantillonne cette image à 20 angles
pour créer 20 raycasts virtuels, comme si c'était du Unity :

```
angle -90° → colonne 0  → distance 1200mm → ray_0 = 1200/3500 = 0.34
angle -80° → colonne 34 → distance 1100mm → ray_1 = 1100/3500 = 0.31
...
angle   0° → colonne 320 → distance 500mm → ray_9 = 500/3500  = 0.14
...
angle +90° → colonne 640 → distance 1500mm → ray_19 = 1500/3500 = 0.43
```

### Étape 2 — Recalibrer le Z-score

Le modèle attend des rays **Z-scorés** avec les statistiques de simulation.
Les statistiques de simulation sont fausses pour les données réelles.

```
Simulation :  ray_9 moyen = 0.72, écart-type = 0.18
Réalité    :  ray_9 moyen = 0.55, écart-type = 0.24  ← différent !

Z-score simulation appliqué à la réalité :
  z = (0.55 - 0.72) / 0.18 = -0.94  ← wrong !

Z-score réel :
  z = (0.55 - 0.55) / 0.24 = 0.00   ← correct
```

`calibrate_ray_stats.py` collecte 2-5 minutes de données réelles
et recalcule mean/std pour que le Z-score soit juste.

---

## 4. Pourquoi les recommandations des LLMs sont critiques

### Problème 1 — Le tangage (découvert par Gemini) ← BLOQUANT

Notre approche initiale prenait UNE seule ligne de pixels au centre de l'image.

```
Caméra stable (simulation) :
┌──────────────────────────┐
│                          │
│  ←←← ROI ligne ←←←←←   │  ← ligne unique, ça marche
│                          │
└──────────────────────────┘

Caméra qui tangue (accélération réelle) :
┌──────────────────────────┐
│                          │
│                          │
│  ←←← ROI ligne ←←←←←   │  ← la ligne pointe vers le CIEL !
└──────────────────────────┘
     (le nez de la voiture monte)
```

**Conséquence** : dès que la voiture accélère, le nez monte légèrement.
La ligne de pixels pointe vers le ciel → distance = infini → rayons = 1.0 partout
→ le modèle croit que la piste est libre → pleine vitesse → crash.

**Solution LLM** : utiliser une **bande de 40% à 62%** de la hauteur
et prendre la distance **minimum** dans cette bande.
Même si la caméra tangue de 10°, la bande capture toujours le sol devant.

---

### Problème 2 — Les NaN et zéros (Grok + Gemini)

La stéréo vision OAK-D ne peut pas calculer la distance de certains pixels :
- Surfaces lisses (pas de texture → pas de matching stéréo)
- Zones trop proches ou trop loin
- Zones occludées

Ces pixels valent **0** dans la depth map.

```
Notre code naïf : ray = depth[row, col] / MAX_DIST
Si depth[row, col] = 0 : ray = 0.0 / 3500 = 0.0

Le modèle interprète 0.0 comme "obstacle à 0mm" → freine à mort ou tourne brusquement
```

**Solution LLM** : remplacer les 0 par MAX_DIST (pas d'obstacle détecté = loin = libre).

---

### Problème 3 — Le watchdog hardware VESC (Gemini) ← VITAL SÉCURITÉ

Si le Jetson Nano plante (crash Python, surcharge CPU, perte alimentation)...
**sans watchdog** : le VESC maintient la dernière commande → voiture fonce jusqu'au mur.
**avec watchdog** : le VESC coupe automatiquement le moteur après 200ms sans commande.

```
À configurer dans VESC Tool (à faire UNE FOIS) :
  App Settings → General → Timeout = 200ms
```

C'est une ligne de config dans un logiciel. Si on oublie ça,
un simple freeze Python = accident physique.

---

### Problème 4 — Le Z-score simulation ≠ réel (consensus unanime)

Les 3 LLMs ont dit la même chose : **les stats simulation sont inutilisables**.
C'est la cause n°1 d'échec du sim-to-real en Behavioral Cloning raycast.

Sans recalibrage → le modèle reçoit des valeurs hors de sa distribution d'entraînement
→ prédictions aléatoires → comportement imprévisible dès le premier mètre.

---

## 5. Le plan complet étape par étape

```
┌─────────────────────────────────────────────────────────────┐
│  CE QU'ON A FAIT                                            │
│                                                             │
│  1. depth_to_rays.py      ← convertit depth map → raycasts │
│     • ROI band anti-tangage                                 │
│     • NaN/0 gérés                                           │
│     • projection correcte (focal length réel)               │
│                                                             │
│  2. vesc_interface.py     ← commande le moteur + servo      │
│     • steering [-1,1] → servo [0,1]                         │
│     • accel [0,1] → duty cycle (max 15% au début)           │
│     • arrêt d'urgence intégré                               │
│                                                             │
│  3. inference_realcar.py  ← boucle principale               │
│     • Thread 1 : OAK-D → depth → raycasts                  │
│     • Thread 2 : raycasts → ONNX → smoother → VESC          │
│     • Watchdog logiciel (500ms)                             │
│     • Preprocessing IDENTIQUE à la simulation               │
│                                                             │
│  4. calibrate_ray_stats.py ← recalibrage Z-score           │
│     • 2-5 min de roulage manuel sur piste réelle            │
│     • génère real_ray_stats.json                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  CE QU'IL RESTE À FAIRE (sur le Jetson)                     │
│                                                             │
│  A. Config matérielle (UNE FOIS)                            │
│     • VESC Tool → Timeout = 200ms                           │
│     • Vérifier port USB FSESC : ls /dev/ttyACM*             │
│     • Vérifier sens servo (invert_steer si besoin)          │
│                                                             │
│  B. Installer les dépendances                               │
│     pip install onnxruntime-gpu depthai pyvesc pyserial     │
│                                                             │
│  C. Calibrer le Z-score                                     │
│     python3 src/calibrate_ray_stats.py                      │
│     (pousser la voiture 2-5 min sur la piste)               │
│                                                             │
│  D. Test 1 — Roues en l'air (OBLIGATOIRE)                   │
│     python3 src/inference_realcar.py --duty-max 0.15        │
│     → Vérifier que le servo tourne dans le bon sens         │
│     → Vérifier que le moteur répond                         │
│                                                             │
│  E. Test 2 — Premier tour piste (lent)                      │
│     python3 src/inference_realcar.py --duty-max 0.15        │
│     → Quelqu'un avec kill-switch physique                   │
│     → Observer la trajectoire                               │
│                                                             │
│  F. Montée en puissance progressive                         │
│     --duty-max 0.15 → 0.25 → 0.35 → 0.40                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Ce qui peut encore mal se passer (et comment réagir)

| Problème | Cause probable | Solution |
|----------|---------------|----------|
| Voiture tourne tout le temps dans un sens | Servo mal calibré | Ajuster `servo_center` dans vesc_interface.py |
| Voiture freine tout le temps | Z-score simulation non recalibré | Lancer calibrate_ray_stats.py d'abord |
| Modèle prédit bien mais servo inversé | Montage physique inversé | `invert_steer=True` dans vesc_interface.py |
| Crash Python → voiture fonce | Watchdog VESC non configuré | VESC Tool → Timeout = 200ms |
| Depth map toute noire (0) | OAK-D mal initialisée | Redémarrer le script |
| Conduite erratique au premier virage | Distribution rays trop différente | Recalibrer Z-score + ajuster MAX_DISTANCE_MM |

---

## 7. Ce que le modèle NE CHANGERA PAS

Le modèle v18 lui-même n'est pas modifié.
Il a appris à transformer `[23 features Z-scorées]` → `[steering, accel_logit]`.
Cette logique reste exactement la même.

Ce qu'on adapte c'est uniquement **la couche de perception** :
comment obtenir ces 23 features depuis la caméra réelle au lieu du simulateur.

```
SIMULATION :   Unity gRPC  →  raycasts parfaits  →  Z-score sim  →  [23]  →  Modèle v18
RÉEL       :   OAK-D depth →  raycasts virtuels  →  Z-score réel →  [23]  →  Modèle v18
                                                                              (inchangé)
```

**Le modèle ne sait pas qu'il est sur une vraie voiture.
Il croit toujours qu'il est dans le simulateur.**
C'est exactement l'objectif.

# Prompt IA — Contrôleur autonome course chrono

> Copier-coller dans Gemini / Grok / ChatGPT

---

## Contexte : Concours de voiture autonome — meilleur chrono sur piste

Je participe à un concours de voiture autonome sur piste délimitée par des lignes
blanches peintes au sol. L'objectif unique est de réaliser le meilleur temps au tour.

---

## Matériel disponible

- Voiture RC modifiée, moteur brushless BLDC commandé par VESC (ESC open-source FSESC Mini V6.7)
- Caméra OAK-D Lite (RGB + depth) fixée sur la voiture, branchée à une Jetson Nano 4GB
- VESC commandé en UART depuis la Jetson via protocole custom (CRC-16 XMODEM)
  - `steering in [-1, 1]` → servo direction via COMM_SET_SERVO_POS
  - `throttle in [-1, 1]` → courant moteur via COMM_SET_CURRENT (max 8A)
  - Heartbeat COMM_ALIVE toutes les 300ms (watchdog VESC)
- Piste intérieure, sol foncé, lignes blanches ~5cm de largeur
- Largeur piste ~60-80cm, virages larges et serrés mixés, 1 tour ~30-60s estimé

---

## Pipeline vision opérationnel (Python 3.6.9, Jetson Nano, depthai 2.x)

```
OAK-D Lite → frames BGR 512x256 @ 12fps
    ↓
Gaussian blur (3x3) + CLAHE (channel V, clip=1.5)
    ↓
HSV mask : V>=195, S<=40 → blanc pur uniquement
    ↓
Morpho OPEN+CLOSE+DILATE (kernel 5) + filtre blobs (min 400px)
    ↓
ROI : ignorer haut 35%, garder 35%-100% de l'image
    ↓
Données extraites :
  err_now    = centroïde zone basse (55%-100% H) - W/2   [-256, +256]
  err_ahead  = centroïde zone haute (35%-55% H) - W/2    [-256, +256]
  rays[20]   = raycasts visuels [0.0=bord proche, 1.0=libre]
  blobs[]    = composantes connexes (cx, cy, area)
  n_blobs    = nombre de lignes détectées (0, 1 ou 2)
  curvature  = std(rays)  — 0=ligne droite, >0.3=virage serré
```

Latence estimée frame→commande : ~80-120ms (pipeline CPU only, pas de GPU)

---

## Contrôleur actuel implémenté (4 niveaux)

```python
# Niveau 1 — PD basique
steering = KP * err_now + KD * (err_now - prev_err)

# Niveau 2 — Lookahead
err_combined = 0.35 * err_now + 0.65 * err_ahead
steering = KP * err_combined + KD * (err_combined - prev_err)

# Niveau 3 — Centre réel via 2 lignes séparées
err = (cx_gauche + cx_droite) / 2 - W/2
steering = PD(err)

# Niveau 4 — Machine à états vitesse
STRAIGHT: throttle = V_MAX (0.30)
TURN:     throttle = V_TURN (0.18)
RECOVER:  throttle = V_SLOW (0.12)  # 1 seule ligne visible
STOP:     throttle = 0.0             # 0 ligne visible
```

Paramètres actuels : KP=0.004, KD=0.002, V_MAX=0.30, STEERING_MAX=0.8

---

## Questions pour améliorer le chrono

### 1. Tuning PD optimal

- Comment calculer KP et KD optimaux sans tachymètre ni encodeur roue ?
- Méthode Ziegler-Nichols applicable sur une voiture RC ? Autre méthode empirique ?
- Faut-il un PID (terme I) ou le PD suffit pour suivre des lignes ?
- Comment gérer la non-linéarité du servo (réponse mécanique non-linéaire) ?

### 2. Lookahead dynamique selon la vitesse

- À haute vitesse, doit-on augmenter le poids W_AHEAD (regarder plus loin) ?
- Formule recommandée pour adapter W_AHEAD = f(speed) ?
- Faut-il 3 bandes horizontales (loin/moyen/proche) au lieu de 2 ?
- Comment estimer la vitesse réelle sans encodeur (tachymètre VESC ERPM disponible) ?

### 3. Séparation lignes gauche/droite robuste

- Comment séparer 2 blobs qui se fondent en virage serré (perspective) ?
- Que faire si une seule ligne est visible (bord de piste ou virage extrême) ?
- RANSAC pour fitter une droite sur chaque ligne : justifié en 12fps/80ms budget ?
- Peut-on utiliser le Kalman pour tracker chaque ligne entre les frames ?

### 4. Vitesse maximale et freinage

- Comment détecter une ligne droite longue pour accélérer au-delà de V_MAX ?
- Freinage régénératif via VESC (courant négatif) : comment doser pour ne pas patiner ?
- Peut-on prédire la fin d'une ligne droite à partir des raycasts avant d'arriver au virage ?
- Quelle est la formule de la distance de freinage en fonction de la vitesse actuelle ?

### 5. Trajectoire optimale (racing line)

- Peut-on implémenter une approximation de la trajectoire apex avec seulement la vision ?
- Comment détecter l'apex d'un virage depuis le masque HSV ?
- Couper les virages légèrement (rester proche de la ligne intérieure) est-il faisable ?
- Faut-il un modèle mémorisé de la piste (carte) ou tout peut être réactif ?

### 6. Compensation de la latence

- Avec 80-120ms de latence, comment éviter les oscillations à haute vitesse ?
- Dead-reckoning : peut-on prédire la position future depuis la vitesse VESC ERPM ?
- Filtre de Kalman pour fusionner vision + VESC telemetry : justifié sur Jetson Nano CPU ?
- Smith Predictor pour PID avec retard : trop complexe pour ce cas ?

### 7. Petit réseau de neurones vs PD

- À partir de quand un MLP ou CNN léger surpasse un PD bien tuné sur ce type de piste ?
- Si réseau : architecture minimale pour <10ms sur Jetson Nano CPU (pas de GPU dispo) ?
- Entrées recommandées : rays[20] uniquement ? rays + err + vitesse_erpm ?
- Sortie : delta_steering ou steering absolu ? Inclure throttle dans la sortie ?
- Comment collecter le dataset de training (imitation learning depuis téléop gamepad) ?
- Stratégie d'augmentation de données pour améliorer la robustesse ?

### 8. Stratégie globale course

- Vaut-il mieux maximiser la vitesse de pointe ou minimiser les corrections de direction ?
- Comment gérer l'asphalte différent au démarrage à froid (moins d'adhérence) ?
- Un warm-up automatique du premier tour (vitesse réduite, calibration) est-il recommandé ?
- Multi-voitures sur piste : détecter et dépasser un adversaire avec seulement le masque HSV ?

---

## Contraintes techniques

- Python 3.6.9 (pas de walrus :=, pas de match/case, f-strings OK)
- RAM : 1.2GB libre, **pas de GPU pour l'inférence** (cuDNN occupé par depthai)
- Packages disponibles : numpy, opencv, scipy, sklearn, torch CPU, pyserial
- depthai 2.x API (pas de 3.x), usb2Mode=True (limite ~900mA)
- Pas de mise à jour pip/apt autorisée
- VESC : commandes UART série, heartbeat obligatoire toutes les 300ms

---

## Fichiers pertinents du projet

- `src/control/controller_pd.py`   — contrôleur PD 4 niveaux (fichier principal)
- `src/mask/visual_rays.py`     — pipeline vision HSV + raycasts
- `src/control/vesc_interface.py`  — interface UART VESC (duty, current, servo, ERPM)
- `src/control/inference_realcar.py` — ancien contrôleur Behavioral Cloning v18 (référence)

Le modèle BC v18 tourne le circuit en ~24s. L'objectif est de faire mieux avec le PD.

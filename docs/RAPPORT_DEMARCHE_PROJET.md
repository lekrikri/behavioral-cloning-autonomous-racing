# G-CAR-000 — Rapport de démarche projet
## Voiture autonome par Behavioral Cloning : du simulateur au hardware réel

---

## 1. Objectif du projet

L'objectif est de développer une voiture RC capable de naviguer de manière autonome en utilisant uniquement une caméra de profondeur, sans GPS ni lidar. La technique choisie est le **Behavioral Cloning** : un réseau de neurones apprend à imiter les décisions d'un conducteur expert en associant des observations (distances aux obstacles) à des actions (direction, accélération).

**Stack hardware cible :**
- Ordinateur embarqué : Jetson Nano
- Caméra : Luxonis OAK-D Lite (depth stereo)
- Contrôleur moteur : Flipsky FSESC Mini V6.7 Pro (firmware VESC 6.0)
- Moteur : Traxxas Velineon 3300kV (brushless sensorless)

---

## 2. Démarche initiale prévue

La démarche originale était linéaire et séquentielle :

```
1. Entraînement dans le simulateur Unity (ML-Agents)
       ↓
2. Export du modèle → ONNX
       ↓
3. Déploiement direct sur Jetson Nano
       ↓
4. Tests sur circuit réel
```

L'hypothèse de départ était que le passage simulateur → hardware serait principalement un problème de **domain adaptation** (les raycasts simulés ≠ la profondeur réelle OAK-D), et que le contrôleur moteur se configurerait rapidement via VESC Tool.

---

## 3. Ce qui a réellement posé problème

### 3.1 Le simulateur : itérations plus longues que prévu

L'entraînement en simulation a nécessité **19 versions successives** du modèle avant d'atteindre des performances satisfaisantes. Les principaux obstacles :

- Choix de l'architecture réseau (MLP vs LSTM, taille des couches)
- Définition des features d'entrée (raycasts bruts vs Z-scorés + features dérivées)
- Gestion du déséquilibre des données (le conducteur expert tourne rarement à fond)
- Stratégie de split train/validation (temporal split obligatoire pour éviter le data leakage)

Le **modèle v18** a finalement atteint un temps de tour record de **24 secondes** sur le circuit simulé, avec une architecture : 20 raycasts Z-scorés + 3 features dérivées (asymétrie, front_ray, min_ray) → MLP → [steer, accel].

### 3.2 Le hardware : un chemin d'intégration non anticipé

C'est ici que la démarche a le plus dévié du plan initial. Trois problèmes indépendants se sont enchaînés, chacun bloquant le suivant.

---

#### Problème A — Bug firmware Flipsky (configuration FOC corrompue)

Le VESC Tool ne pouvait pas lire la configuration de l'ESC :
> *"Warning: Could not set mcconf due to wrong signature"*

L'EEPROM du contrôleur était corrompue (probablement lors d'une mise à jour firmware partielle). Il était impossible de configurer le moteur via l'interface graphique standard.

**Adaptation :** Développement d'un outil CLI en Python (`vesc_cli.py`) qui contourne l'EEPROM en :
1. Lisant les paramètres par défaut firmware via `COMM_GET_MCCONF_DEFAULT`
2. Patchant directement le blob binaire en mémoire
3. Réécrivant la config en RAM via `COMM_SET_MCCONF`

Ce travail a nécessité de comprendre et implémenter le protocole série VESC 6.0 (CRC16-CCITT, encodage big-endian, structure des paquets) sans documentation officielle complète, en s'appuyant sur le code source firmware.

---

#### Problème B — Bug dans les paramètres de démarrage FOC sensorless

Même avec la config appliquée, le moteur produisait des **saccades violentes** et ne tournait pas. L'analyse des données de télémétrie a révélé un bug dans les valeurs par défaut Flipsky :

| Paramètre | Valeur défaut Flipsky | Valeur correcte |
|-----------|----------------------|-----------------|
| `foc_sl_openloop_time` | **0.05 s** | 0.60 s |
| `foc_sl_openloop_time_ramp` | 0.10 s | 0.25 s |
| `foc_sl_openloop_boost_q` | **0 A** | 7 A |

`openloop_time` (0.05s) était inférieur à `openloop_time_ramp` (0.10s) : le VESC tentait de basculer en closed-loop avant même d'avoir terminé sa rampe de démarrage, créant une boucle infinie de micro-tentatives de démarrage. De plus, le courant de démarrage `boost_q = 0A` ne donnait aucun couple au moteur.

Ces offsets n'étaient pas documentés. Ils ont été trouvés empiriquement via un scanner de blob binaire (`vesc_find_offsets.py`) qui recherchait des valeurs float32/int16 connues dans le blob de configuration.

---

#### Problème C — Phase moteur déconnectée

Après correction des paramètres logiciels, le moteur restait bloqué à 0 RPM avec une consommation anormalement élevée (~50A à 20% de duty cycle). 

Le diagnostic a été posé par calcul de la résistance effective :
- **Mesurée :** V/I = (16V × 0.20) / 50A ≈ **66 mΩ**
- **Attendue** (Velineon 3 phases) : ~**11–15 mΩ**

Un facteur 4-5× trop élevé indique qu'une phase était déconnectée : sans les 3 phases, le champ magnétique tournant ne peut pas se former et le rotor reste statique.

**Fix :** Reconnexion physique des 3 connecteurs bullet entre l'ESC et le moteur. Suivi d'une détection FOC automatique (`COMM_DETECT_APPLY_ALL_FOC`) pour recalibrer les paramètres R, L et λ du moteur avec les 3 phases correctement connectées.

---

## 4. État actuel du pipeline

Le pipeline complet fonctionne en temps réel sur le hardware :

```
OAK-D Lite (depth frame)
       ↓  DepthToRays
20 raycasts virtuels (mm)
       ↓  Z-score normalisation
Features [23] normalisées
       ↓  ONNX Runtime (CPU, ~430 Hz)
[steer ∈ [-1,1], accel ∈ [0,1]]
       ↓  SmoothingFilter + heuristique front_cap
Commandes lissées
       ↓  VESCInterface (protocole natif SET_CURRENT + SET_SERVO_POS)
Moteur + servo
```

**Performances mesurées sur Jetson Nano :**
- Boucle d'inférence : **~430 Hz** (largement au-dessus du besoin 30 Hz)
- Latence totale perception→action : < 10 ms
- Courant moteur : réglable (3A pour tests, 8A normal)
- Servo calibré : center=0.500, range=±0.290, inversion activée

**Configuration moteur persistante** via service systemd `vesc-config` qui réapplique les patches à chaque démarrage du Jetson (contournement EEPROM).

---

## 5. Écarts par rapport à la démarche initiale

| Aspect | Prévu | Réel |
|--------|-------|------|
| Configuration ESC | 30 min via VESC Tool | Développement d'un CLI Python complet |
| Protocole VESC | Bibliothèque pyvesc | Implémentation protocole natif (CRC16, big-endian) |
| Problèmes hardware | Aucun anticipé | 3 problèmes bloquants enchaînés |
| Temps intégration hardware | ~1 semaine | Plusieurs semaines |
| Pipeline inference | pyvesc + SetDutyCycle | Protocole natif + SetCurrent (FOC) |

---

## 6. Ce que le projet a apporté au-delà du cahier des charges

- **Maîtrise du protocole VESC bas niveau** : lecture/écriture de blob binaire, CRC16, sérialisation big-endian, commandes ID
- **Diagnostic hardware embarqué** : méthode de mesure R_eff pour détecter une phase déconnectée sans oscilloscope
- **Architecture multi-thread temps réel** : perception (OAK-D) et contrôle (VESC) dans des threads séparés avec watchdog
- **Outils de développement réutilisables** : `vesc_cli.py`, `vesc_find_offsets.py`, `vesc_detect_rl.py` — utilisables sur tout VESC avec EEPROM corrompue

---

## 7. Prochaines étapes

1. **Tests au sol** — monter progressivement le courant (3A → 6A → 8A) sur circuit délimité
2. **Recalibration Z-score** — collecter des données de profondeur réelles pour remplacer les stats simulateur (`calibrate_ray_stats.py`)
3. **Fine-tuning du smoothing** — ajuster les paramètres du filtre selon le comportement réel observé
4. **Évaluation quantitative** — mesurer le taux de succès sur circuit (tours sans sortie de piste)

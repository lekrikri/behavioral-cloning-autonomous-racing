# Guide Compétition — Robocar Controller PD

> Guide pratique pour le jour de compétition.
> Mis à jour : 2026-06-19

---

## Vue d'ensemble du système

La voiture suit une piste délimitée par des **lignes blanches peintes au sol** via un contrôleur PD couplé à un masquage HSV de la caméra OAK-D Lite.

```
OAK-D Lite (caméra couleur 512×256)
      │
      ▼
  Masque HSV + Otsu adaptatif      ← visual_rays.py
  (filtre les pixels blancs)
      │
      ▼
  Détection blobs                  ← controller_pd.py
  (blob gauche + blob droit)
      │
      ▼
  Erreur latérale en pixels
  (centroïde lignes vs centre image)
      │
      ▼
  Contrôleur PD
  steering = KP*err + KD*(err - err_prev)
      │
      ▼
  VESC (/dev/ttyACM0)
  (commande moteur + servo)
```

---

## Hardware

| Composant | Détail |
|-----------|--------|
| Calculateur | Jetson Nano 4GB |
| Caméra | OAK-D Lite (USB) |
| ESC / Moteur | FSESC VESC sur `/dev/ttyACM0` |
| Réseau | WiFi → Tailscale VPN |

### Accès SSH

```bash
# IP Tailscale Jetson
sshpass -p 'robocar' ssh robocar@100.112.10.119

# Depuis Windows via WSL
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'COMMANDE'"
```

---

## Procédure de démarrage (dans l'ordre)

### 1. Ouvrir le tunnel SSH pour le stream (terminal dédié, garder ouvert)

```bash
pkill -f 'ssh.*5601'
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=999 -L 5601:localhost:5601 robocar@100.112.10.119"
```

Ouvrir dans le navigateur : **http://localhost:5601**

### 2. Valider la vision (dry-run, VESC non commandé)

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S bash -c \"cd /home/robocar/behavioral-cloning-autonomous-racing && OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py --fixed-speed 0.10 --dry-run --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &\"'"
```

Dans le stream, vérifier que **b=2** et que les zones vertes collent aux lignes blanches.

### 3. Lancer en mode réel (VESC commandé)

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3 ; sleep 2 ; echo robocar | sudo -S bash -c \"cd /home/robocar/behavioral-cloning-autonomous-racing && OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py --fixed-speed 0.10 --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &\"'"
```

---

## Contrôle en cours de run

### Stopper la voiture SANS couper le stream

```bash
curl http://localhost:5601/stop    # arrêt voiture
curl http://localhost:5601/go      # reprendre
curl http://localhost:5601/status  # état actuel
```

### Arrêter le script complètement

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3'"
```

### Lire les logs

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'tail -f /tmp/ctrl.log'"
```

---

## Lecture du stream

```
err=+12  steer=+0.14  thr=0.10  FIXED [LR]  b=2  ray=-0.03
  │          │            │        │     │    │       └─ asymétrie raycasts gauche/droite
  │          │            │        │     │    └─ blobs détectés (0-2)
  │          │            │        │     └─ lignes vues : L=gauche, R=droite, LR=les deux
  │          │            │        └─ état : FIXED / BLIND / CORNER
  │          │            └─ throttle duty cycle VESC
  │          └─ steering (-1.0 plein gauche → +1.0 plein droite)
  └─ erreur latérale en pixels (négatif = trop à gauche)
```

---

## États machine

| État | Condition | Comportement |
|------|-----------|--------------|
| `FIXED [LR]` | 2 blobs détectés | Suivi normal PD |
| `FIXED [L]` | 1 blob gauche seulement | Estime position droite (+385px) |
| `FIXED [R]` | 1 blob droit seulement | Estime position gauche (-385px) |
| `CORNER [L]` | Virage détecté à gauche | Steering +0.85, throttle ×0.60 |
| `CORNER [R]` | Virage détecté à droite | Steering -0.85, throttle ×0.60 |
| `BLIND` | b=0, aucune ligne | **Arrêt complet** |

---

## Paramètres et ajustements rapides

### Paramètres vision

| Paramètre | Valeur par défaut | Rôle |
|-----------|-------------------|------|
| `HSV_LOW[V]` | 165 | Luminosité min — monte si trop de faux positifs |
| `HSV_HIGH[S]` | 40 | Saturation max — définit le "blanc pur" |
| `MIN_BLOB_AREA` | 800 | Surface min blob valide (px²) |
| `--cam-crop-top` | 0.35 | Crop haut image (35%) |

### Paramètres contrôleur

| Paramètre | Valeur par défaut | Rôle |
|-----------|-------------------|------|
| `KP` | 0.012 | Gain proportionnel — monte = tourne plus fort |
| `KD` | 0.003 | Amortissement — monte = moins d'oscillations |
| `--fixed-speed` | 0.10 | Vitesse (duty cycle, ~0.3 m/s) |
| `CORNER_DURATION` | 15 | Durée virage forcé en frames (~1.9s) |

### Recettes d'ajustement

| Symptôme | Action |
|----------|--------|
| Voiture oscille en ligne droite | Augmenter `KD` : 0.003 → 0.005 |
| Virage trop timide | Augmenter `KP` : 0.012 → 0.016 |
| Trop de faux positifs (meubles, tapis) | Augmenter `HSV_LOW[V]` : 165 → 180 |
| Lignes non détectées (éclairage faible) | Baisser `HSV_LOW[V]` : 165 → 150 |
| Voiture trop rapide en virage | Baisser `--fixed-speed` : 0.10 → 0.08 |
| b=0 trop souvent (BLIND fréquent) | Baisser `MIN_BLOB_AREA` : 800 → 500 |

---

## Filtres blob actifs (get_blobs)

Les blobs sont filtrés pour ne garder que les vraies lignes de piste :

| Filtre | Valeur | Élimine |
|--------|--------|---------|
| `cy_min` | 40% hauteur | Fond de salle, plafond |
| `aspect >= 0.8` | rapport w/h | Pieds de chaises (aspect 0.05-0.3), poteaux |
| `w_min = 18px` | largeur min | Objets trop fins |
| `area < 800px²` | surface min | Petits reflets, bruit |
| `area>3000 et 0.5<asp<2.0` | compact+grand | Flèches sol, logos |

---

## Problèmes courants et solutions

### La voiture reste immobile (BLIND, b=0)

1. Vérifier le stream — les lignes sont-elles visibles en vert ?
2. Si oui mais b=0 → baisser `MIN_BLOB_AREA` ou `cy_min`
3. Si non → ajuster les seuils HSV (`HSV_LOW[V]` plus bas)

### Faux positifs (chaises, murs, tapis détectés en vert)

Monter `HSV_LOW[V]` à 175-185 et/ou baisser `HSV_HIGH[S]` à 30.

### Caméra OAK-D en boucle de crash (X_LINK_UNBOOTED)

Le script tente un reset USB automatique toutes les 10s (watchdog).
Si ça dure >60s : **débrancher/rebrancher physiquement le câble USB de l'OAK-D**.

Fix définitif non encore fait : hub USB alimenté ~15-20€.

### Stream coupé

```bash
pkill -f 'ssh.*5601'
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh -f -N -L 5601:localhost:5601 robocar@100.112.10.119"
```

### Jetson instable / USB qui coupe souvent

Passer en mode 10W (puissance max) :

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S nvpmodel -m 0'"
```

---

## SECURITE CRITIQUE

**L'anti-spark a été retiré du VESC.**

1. Ne **jamais** débrancher la batterie pendant que le script tourne
2. Arrêter d'abord : `curl http://localhost:5601/stop` puis `pkill -9 python3`
3. Attendre 2-3 secondes avant de toucher à la batterie

---

## Modifier un paramètre et redéployer

```bash
# 1. Modifier src/controller_pd.py en local (sur le PC de dev)

# 2. Commit + push
cd /home/lekrikri/Projects/G-CAR-000
git add src/controller_pd.py
git commit -m "fix: ..."
git push origin feat/controller-pd

# 3. Pull sur le Jetson
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'cd ~/behavioral-cloning-autonomous-racing && git fetch origin && git reset --hard origin/feat/controller-pd'"

# 4. Relancer (voir procédure démarrage étape 3)
```

Ne jamais modifier directement sur le Jetson — les changements sont écrasés au prochain pull.

---

## Fichiers clés

| Fichier | Rôle |
|---------|------|
| `src/controller_pd.py` | Contrôleur principal — vision, blobs, PD, états, HTTP, USB reset |
| `src/visual_rays.py` | Masquage HSV + Otsu adaptatif + raycasts visuels |
| `src/vesc_interface.py` | Communication série VESC |
| `docs/QUICKSTART.md` | Fiche commandes rapides (2 pages) |

Repo : `git@github.com:lekrikri/behavioral-cloning-autonomous-racing.git`
Branche : **`feat/controller-pd`**

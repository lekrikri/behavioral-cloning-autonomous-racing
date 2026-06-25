# Commandes — G-CAR-000 Robocar

> Référence complète de toutes les commandes disponibles.
> Jetson Nano IP Tailscale : `100.112.10.119` | user : `robocar` | pass : `robocar`

---

## SSH — Accès Jetson

```bash
# Connexion SSH
sshpass -p 'robocar' ssh robocar@100.112.10.119

# Voir les logs en temps réel
sshpass -p 'robocar' ssh robocar@100.112.10.119 'tail -f /tmp/ctrl.log'

# Vérifier si le script tourne
sshpass -p 'robocar' ssh robocar@100.112.10.119 'ps aux | grep controller | grep -v grep'

# Tuer le script
sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3'

# Ressources système
sshpass -p 'robocar' ssh robocar@100.112.10.119 'free -m && df -h / | tail -1'
```

---

## Tunnel SSH — Stream vidéo

```bash
# Ouvrir le tunnel (à faire une fois par session)
pkill -f 'ssh.*5601'
sshpass -p 'robocar' ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=999 \
  -L 5601:localhost:5601 robocar@100.112.10.119

# Stream accessible sur le PC à :
# http://localhost:5601
```

---

## Git — Mise à jour du code sur la Jetson

```bash
# Après un git push depuis le PC → pull sur la Jetson
sshpass -p 'robocar' ssh robocar@100.112.10.119 \
  'cd ~/behavioral-cloning-autonomous-racing && git fetch origin && git reset --hard origin/main'

# Changer de branche (ex: feat/track-mapping)
sshpass -p 'robocar' ssh robocar@100.112.10.119 \
  'cd ~/behavioral-cloning-autonomous-racing && git fetch origin && git reset --hard origin/feat/track-mapping'
```

---

## Lancement du script controller_pd.py

> ⚠️ Toujours lancer avec `sudo` (nécessaire pour le power cycle USB OAK-D).
> ⚠️ Toujours ajouter `OPENBLAS_CORETYPE=ARMV8` (sinon crash SIGILL sur Cortex-A57).

### Mode test visuel (dry-run — VESC non commandé)

```bash
# Standard : vision seule, stream activé
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.10 --dry-run --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &"
```

### Mode conduite réelle (VESC commandé)

```bash
# Vitesse 10% — calibration / test lent
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.10 --cam-crop-top 0.35 --camera-offset-px -27 > /tmp/ctrl.log 2>&1 &"

# Vitesse 15% — course normale
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.15 --cam-crop-top 0.35 --camera-offset-px -27 > /tmp/ctrl.log 2>&1 &"
```

### Mode cartographie IMU (Phase 1 — 1 tour de reconnaissance)

```bash
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.08 --dry-run --mapping --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &"

# Puis terminer le tour et sauvegarder la carte :
curl http://localhost:5601/finish_map
```

### Mode course chrono (Phase 2 — anticipation IMU)

```bash
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.15 --racing --cam-crop-top 0.35 --camera-offset-px -27 > /tmp/ctrl.log 2>&1 &"
```

### Mode enregistrement trajectoire (CSV)

```bash
# Enregistre err/steering/throttle dans un CSV pour analyse
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.10 --cam-crop-top 0.35 --record /tmp/track.csv > /tmp/ctrl.log 2>&1 &"
```

### Mode replay trajectoire (feedforward CSV)

```bash
# Rejoue une trajectoire enregistrée (70% feedforward + 30% PD vision)
echo robocar | sudo -S bash -c "cd ~/behavioral-cloning-autonomous-racing && \
  OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py \
  --fixed-speed 0.10 --cam-crop-top 0.35 --replay /tmp/track.csv > /tmp/ctrl.log 2>&1 &"

# Ajuster le poids du feedforward (défaut 0.70)
  --replay /tmp/track.csv --replay-weight 0.50
```

---

## Arguments CLI — Référence complète

| Argument | Défaut | Description |
|---|---|---|
| `--port` | `/dev/ttyACM0` | Port série VESC |
| `--baud` | `115200` | Baudrate VESC |
| `--level` | `3` | Niveau algo 1-4 |
| `--dry-run` | off | Vision seule, VESC non commandé |
| `--fixed-speed` | None | Vitesse constante [0.0-1.0] — bypass machine à états |
| `--stream-port` | `5601` | Port stream MJPEG (0 = désactivé) |
| `--cam-crop-top` | `0.0` | Crop haut image [0.0-0.6] (ex: 0.35 = enlever 35%) |
| `--camera-offset-px` | `0` | Biais latéral caméra en pixels (actuel : **-27**) |
| `--roi-far` | auto | Override ROI_FAR [0.0-0.9] |
| `--steering-max` | `0.85` | Steering maximum [0.3-1.0] |
| `--no-corner` | off | Désactive détection virage (mode ligne droite) |
| `--record FILE` | None | Enregistre la session dans un CSV |
| `--replay FILE` | None | Rejoue un CSV comme feedforward |
| `--replay-weight` | `0.70` | Poids feedforward replay [0.0-1.0] |
| `--mapping` | off | Phase 1 : enregistre la piste → track_map.json |
| `--racing` | off | Phase 2 : charge la carte et anticipe les virages |
| `--map-file` | `track_map.json` | Chemin du fichier carte IMU |

---

## Commandes HTTP (curl) — Contrôle en temps réel

> Stream accessible sur `http://localhost:5601` (après tunnel SSH).

```bash
# Arrêter la voiture (VESC stop, script continue)
curl http://localhost:5601/stop

# Reprendre la conduite
curl http://localhost:5601/go

# Statut actuel
curl http://localhost:5601/status

# Calibrer l'offset caméra (voiture au centre piste)
curl http://localhost:5601/calibrate

# Terminer le mapping et sauvegarder track_map.json
curl http://localhost:5601/finish_map
```

---

## Commandes USB — Diagnostic

```bash
# Liste des périphériques USB connectés
sshpass -p 'robocar' ssh robocar@100.112.10.119 'lsusb'

# Arbre USB (topologie hub)
sshpass -p 'robocar' ssh robocar@100.112.10.119 'lsusb -t'
```

Périphériques attendus :
| ID USB | Périphérique |
|---|---|
| `03e7:2485` | OAK-D Lite (Myriad X) |
| `0483:5740` | VESC (STM32 CDC ACM) |
| `0bda:b812` | WiFi TP-Link |
| `046d:c21f` | Gamepad Logitech F710 |
| `05e3:0620` | Hub USB 3.0 alimenté (Genesys Logic) |

---

## Workflow complet — Du démarrage à la course

```bash
# 1. Vérifier que la Jetson est connectée
sshpass -p 'robocar' ssh robocar@100.112.10.119 'uptime'

# 2. Ouvrir le tunnel stream
sshpass -p 'robocar' ssh -f -N -L 5601:localhost:5601 robocar@100.112.10.119

# 3. Mettre à jour le code si nécessaire
sshpass -p 'robocar' ssh robocar@100.112.10.119 \
  'cd ~/behavioral-cloning-autonomous-racing && git pull'

# 4. Lancer le script (sur la Jetson en SSH)
# → voir section "Lancement" ci-dessus

# 5. Ouvrir http://localhost:5601 dans le navigateur

# 6. Placer la voiture au centre piste → calibrer
curl http://localhost:5601/calibrate

# 7. Lancer la course : curl http://localhost:5601/go

# 8. Arrêter proprement AVANT de débrancher la batterie
curl http://localhost:5601/stop
# Attendre 2s puis :
sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3'
# Attendre 2-3s → débrancher batterie
```

# Quickstart — Robocar Controller PD

> Fiche rapide. Guide complet : `docs/GUIDE_COMPETITION.md`

---

## Connexion Jetson

```bash
sshpass -p 'robocar' ssh robocar@100.112.10.119
# ou depuis Windows :
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'COMMANDE'"
```

---

## Stream vidéo (ouvrir en premier, garder ouvert)

```bash
pkill -f 'ssh.*5601'
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=999 -L 5601:localhost:5601 robocar@100.112.10.119"
```

**http://localhost:5601**

---

## Lancer

```bash
# Dry-run (vision seule, voiture immobile)
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S bash -c \"cd /home/robocar/behavioral-cloning-autonomous-racing && OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py --fixed-speed 0.10 --dry-run --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &\"'"

# Mode réel (VESC commandé — la voiture roule)
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3 ; sleep 2 ; echo robocar | sudo -S bash -c \"cd /home/robocar/behavioral-cloning-autonomous-racing && OPENBLAS_CORETYPE=ARMV8 nohup python3 -u src/controller_pd.py --fixed-speed 0.10 --cam-crop-top 0.35 > /tmp/ctrl.log 2>&1 &\"'"
```

---

## Contrôle

```bash
curl http://localhost:5601/stop    # stopper la voiture (stream reste actif)
curl http://localhost:5601/go      # reprendre
curl http://localhost:5601/status  # état

# Tuer le script
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3'"

# Logs
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'tail -f /tmp/ctrl.log'"
```

---

## Après une modif du code

```bash
# Sur le PC de dev
git add src/controller_pd.py && git commit -m "fix: ..." && git push origin feat/controller-pd

# Pull sur le Jetson
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'cd ~/behavioral-cloning-autonomous-racing && git fetch origin && git reset --hard origin/feat/controller-pd'"
```

---

## Si la caméra plante (X_LINK_UNBOOTED)

Le script tente un reset automatique (watchdog 10s). Si ça dure >60s : **débrancher/rebrancher le câble USB OAK-D**.

---

## Stream coupé → recréer le tunnel

```bash
pkill -f 'ssh.*5601'
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh -f -N -L 5601:localhost:5601 robocar@100.112.10.119"
```

---

## Jetson mode 10W (si USB instable)

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S nvpmodel -m 0'"
```

---

## Éteindre le Jetson

```bash
wsl -d Ubuntu -e bash -c "sshpass -p 'robocar' ssh robocar@100.112.10.119 'echo robocar | sudo -S pkill -9 python3 ; sleep 1 ; echo robocar | sudo -S shutdown -h now'"
```

---

## SECURITE

**Anti-spark retiré du VESC** — ne jamais débrancher la batterie pendant que le script tourne.
Toujours : `curl http://localhost:5601/stop` → attendre → `pkill -9 python3` → puis batterie.

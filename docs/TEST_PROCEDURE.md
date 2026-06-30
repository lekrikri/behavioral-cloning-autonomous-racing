# Procédure de test sur la voiture — PR #14 (services + superviseur core)

> Valide les specs de la PR sur la Jetson **sans casser le setup de Christophe** (on garde
> `robocar-hub` pour la caméra, on arrête juste `robocar-ctrl` le temps du test, et on le
> restaure). Test depuis un **worktree jetable**, pas le clone de prod.

## ⚠️ Sécurité & limites

- **ROUES EN L'AIR** : tout worker de conduite (manuel ou auto) commande le VESC/moteur.
- **Exclusivité VESC** : on arrête `robocar-ctrl` pendant le test (il commande aussi le VESC),
  on le redémarre à la fin.
- **Limite connue** : sur `main`, `controller_pd` n'a pas `--source hub` → le worker **auto**
  sort en erreur. On valide donc l'**orchestration** (transitions d'état, spawn/kill, garde-fous)
  et le worker **manuel** réel ; l'algo de conduite **auto** est hors scope (gated sur l'intégration
  du code worker de `feat/track-mapping` → `main`, côté Christophe).

## 0. Récupérer la PR (worktree jetable)

```bash
ssh robocar
cd ~/behavioral-cloning-autonomous-racing
git fetch origin feat/camera-hub-service
git worktree add /tmp/pr14 origin/feat/camera-hub-service
cd /tmp/pr14
```

*(Option « observer le spawn/kill auto » : comme `controller_pd` sort en erreur sur main,
remplace temporairement le worker `auto_pd` par un `sleep` dans `/tmp/pr14/configs/profiles.json* :*
`"argv": ["sleep", "600"]` *— ça rend le mode auto observable via `ps`/`/status`.)*

## 1. Libérer le VESC (on garde la caméra)

```bash
sudo systemctl stop robocar-ctrl     # libère le VESC ; robocar-cam-hub (caméra, SHM) reste up
```

## 2. Lancer le superviseur

```bash
OPENBLAS_CORETYPE=ARMV8 python3 -m core
# attendu : [core] superviseur démarré — INERTE (profil sélectionné=P1), contrôle :8090
```

Ouvre un **2e terminal** (`ssh robocar`) pour les `curl` ci-dessous (l'endpoint est sur
`127.0.0.1:8090`).

## 3. Spec « config 2 couches » + « boot inerte »

```bash
curl -s localhost:8090/status
# ✅ {"profile":"P1","profile_launched":false,"mode":"idle","manual_armed":false,"driving":false,"workers_running":[]}
```
→ la config a chargé (sinon le superviseur aurait planté au démarrage), et **aucun worker de
conduite** au boot.

## 4. Spec « update : terminal/UI, refusé en conduite »

```bash
curl -s -XPOST localhost:8090/update            # ✅ {"ok":true,...}  (idle)
curl -s -XPOST localhost:8090/profile -d '{"profile":"P1"}'
curl -s localhost:8090/status                   # ✅ mode=auto, profile_launched=true
curl -s -XPOST localhost:8090/update            # ✅ {"ok":false,"error":"refusé : conduite..."}
curl -s -XPOST localhost:8090/stop              # retour inerte (mode=idle)
```
(En option-`sleep` : `ps -ef | grep sleep` montre le worker auto pendant le profil lancé.)

## 5. Spec « manette = armement + prise de main explicite »

1. **Allumer la F710**, puis :
   ```bash
   curl -s localhost:8090/status
   # ✅ manual_armed=true ; mode reste "idle" -> "manual" si rien d'autre, ou "auto" passif si profil lancé
   ps -ef | grep teleop_gamepad | grep -v grep    # ✅ worker manuel lancé
   ```
2. **Roues en l'air** : bouger le **stick droit** → la direction doit braquer (teleop actif).
3. Avec un profil lancé (refais l'étape 4 `profile`), tester la **prise de main** :
   ```bash
   curl -s -XPOST localhost:8090/takeover    # ✅ mode=manual (auto coupé)
   curl -s -XPOST localhost:8090/release     # ✅ mode=auto (profil repris)
   ```

## 6. Spec « manette depuis idle = manuel actif »

```bash
curl -s -XPOST localhost:8090/stop          # inerte
# éteindre puis rallumer la manette
curl -s localhost:8090/status               # ✅ mode=manual (rien d'autre ne tournait)
```

## 7. Spec « UI / stream à la demande »

```bash
curl -s -XPOST localhost:8090/ui/connect    # ✅ spawn le worker stream
ps -ef | grep mask_stream | grep -v grep
curl -s -XPOST localhost:8090/ui/disconnect # ✅ kill le worker stream
```

## 8. Arrêt propre + restauration

```bash
# Ctrl+C sur le superviseur -> "[core] arrêt propre"
sudo systemctl start robocar-ctrl                                   # restaure Christophe
git -C ~/behavioral-cloning-autonomous-racing worktree remove --force /tmp/pr14
```

---

## (Optionnel, INTRUSIF) Test du déploiement des services

⚠️ Remplace les services live de Christophe — à **coordonner avec lui**, et restaurer après.

```bash
cd /tmp/pr14/deploy && ./sync-services.sh         # installe cam-hub + core, WorkingDirectory=/tmp/pr14
grep WorkingDirectory /etc/systemd/system/robocar-core.service   # ✅ = /tmp/pr14 (réécrit)
systemctl status robocar-cam-hub robocar-core --no-pager
journalctl --user -u robocar-core -f              # (ou sans --user) -> superviseur INERTE
# reboot -> vérifier l'autostart inerte
```
**Restaurer** : `sudo systemctl disable --now robocar-cam-hub robocar-core` puis
`sudo systemctl enable --now robocar-hub robocar-ctrl`.

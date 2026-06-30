# Services systemd Robocar

> Deux services système démarrent la voiture **au boot**. Versionnés ici (`deploy/systemd/`)
> et synchronisés sur la Jetson via `deploy/sync-services.sh`.

## Les services

| Service | Rôle | Dépend de |
|---|---|---|
| **`robocar-cam-hub.service`** | Hub caméra : possède l'OAK-D, rediffuse les frames sur `:8077`. | — |
| **`robocar-core.service`** | Noyau du soft : **superviseur** (`python3 -m core`) qui orchestre les workers selon le profil. | `robocar-cam-hub` (`Requires=`) |

Un seul process possède l'OAK-D → tout passe par le hub. `robocar-core` ne démarre qu'avec le
hub. Les deux : `Restart=always`, `OPENBLAS_CORETYPE=ARMV8`. Au boot, le superviseur démarre mais
reste **inerte** : **aucun worker de conduite** tant qu'il n'y a pas de trigger (manette allumée
ou profil lancé). Voir [CORE_DESIGN.md](CORE_DESIGN.md) § Démarrage.

> ⚠️ `robocar-core` lance **le superviseur** (`core/`, dans cette branche), pas `controller_pd.py`.
> Le contrôleur PD est un **worker** (un god-file à déconstruire) spawné par le superviseur —
> voir `configs/profiles.json` (`auto_pd`). Design du core : [CORE_DESIGN.md](CORE_DESIGN.md).

## Prod vs dev (WorkingDirectory)

Le `WorkingDirectory` installé pointe sur **le repo d'où on lance `sync-services.sh`** :

```bash
# PROD (boot sur main) — depuis le clone de référence :
cd ~/robocar-Paris-PGE_MSC/deploy && ./sync-services.sh

# DEV (tester le code d'un clone de dev) — depuis ce clone :
cd <mon-clone-dev>/deploy && ./sync-services.sh
# … puis re-sync depuis ~/robocar-Paris-PGE_MSC pour revenir en prod.
```

Le script copie les units dans `/etc/systemd/system/` (en réécrivant `WorkingDirectory`),
`daemon-reload`, `enable`, `restart`. **Il remplace l'ancien `robocar-update.sh`** (git pull au
boot — supprimé). Le déploiement de **code** reste **explicite** (`git pull` manuel), pas au boot.

## Exploitation

```bash
systemctl status robocar-cam-hub robocar-core
journalctl -u robocar-core -f
sudo systemctl restart robocar-core
```


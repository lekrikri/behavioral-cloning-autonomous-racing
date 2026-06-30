# Robocar Core — design (brouillon)

> **Statut : cadrage figé.** Schémas : [`docs/schemas/`](schemas/).

## Rôle

Le **core** est le soft minimal lancé au boot (`robocar-core.service`). C'est un **superviseur** :
il **n'implémente pas** la perception ni le contrôle (ce sont les workers), mais c'est lui qui
**choisit lesquels tournent** (profil + contexte) et les spawn/kill — il **gouverne** les
comportements de perception/contrôle sans les calculer. Objectif : ne lancer que le strict
nécessaire.

Voir [`schemas/01-architecture.md`](schemas/01-architecture.md).

## Responsabilités

1. **Lancer le minimum selon le contexte** (pas tout en permanence) — voir gamepad ci-dessous
   et [`schemas/04-boot-lifecycle.md`](schemas/04-boot-lifecycle.md).
2. **Cycle de vie** : déclencher les **updates** (git pull explicite + `sync-services.sh` +
   restart) — [`schemas/05-update-flow.md`](schemas/05-update-flow.md).
3. **Déployer des modèles sur l'OAK-D** (inférence on-camera) à la demande.
4. **Profils d'usage** : combinaison *lieu de perception × intelligence* —
   [`schemas/02-profiles.md`](schemas/02-profiles.md).

## Démarrage — inerte par défaut

Au boot, `robocar-core.service` lance **le superviseur + le hub** (`robocar-cam-hub`), mais
**aucun worker de conduite** : la voiture est *prête* mais *inerte*. Un worker de conduite ne
démarre que sur **trigger explicite** :

- **manette allumée** (`/dev/input/js*`) → worker de **commande manuelle** ;
- **profil lancé** (terminal ou UI web) → worker **auto** du profil.

Le profil par défaut est *sélectionné* au boot (config) mais **pas lancé**.

## Configuration — deux couches distinctes

On sépare deux concerns qui n'ont rien à voir :

| Couche | Contenu | Change | Reco |
|---|---|---|---|
| **Statique véhicule** | offset/position caméra, vitesse max, duty caps, servo center/range, ports, FOV fixe | quasi jamais (justifié) | **JSON** sous `configs/` (ex. `configs/vehicle.json`) |
| **Profils** | P1..P4 → workers à lancer + leurs params runtime | à la sélection | **JSON** (ex. `configs/profiles.json`) |

**Pourquoi JSON** (et pas YAML/TOML) : zéro dépendance sur la Jetson (Python **3.6**, `json` est
stdlib ; `tomllib` n'existe qu'en 3.11, YAML demande PyYAML) et cohérent avec `configs/config.json`
déjà présent. Optionnel : un petit loader typé (dataclass, cf. `src/config.py`) au-dessus du JSON.
Voir [`schemas/06-config-layers.md`](schemas/06-config-layers.md).

## Contrôle manuel (manette) — prise de main *explicite*

Détecter l'**allumage de la manette** ne doit PAS interrompre ce qui tourne. Séquence voulue
([`schemas/03-gamepad-takeover.md`](schemas/03-gamepad-takeover.md)) :

1. **Manette OFF** → seul le mode courant tourne (auto / idle).
2. **Activité manette détectée** (events *réels*, pas la simple présence du device) → le core
   **lance le worker de commande manuelle**, mais **passif** : le mode courant (ex. auto) continue.
3. **L'user clique « prendre la main »** → le core **coupe le contrôle auto** (s'il tournait) et
   passe en **manuel actif**.
4. **« rendre la main »** → retour au mode courant.

> **Détection / désarmement (limite F710)** : le dongle USB garde `/dev/input/js*` même manette
> éteinte, et le driver émet des events `INIT` à l'ouverture → la *présence* ne signale rien. On
> arme donc sur **activité réelle** (events non-init, backlog vidé à l'ouverture) et on désarme par
> **inactivité** (`gamepad.idle_off_s`, défaut **30 s**), faute de signal OFF fiable. Inoffensif :
> manette éteinte = commandes neutres, la voiture ne bouge pas.

## Modèle de process

Le core **spawn/kill des workers** ; `cam-hub` reste un **service séparé** (dépendance), pas un
enfant du core. Signaux surveillés : présence manette, connexion UI, profil courant.

## Décisions (cadrage figé)

- **Config** = 2 couches JSON (statique véhicule / profils). ✓
- **Manette** = prise de main explicite (détection ≠ bascule). ✓
- **Démarrage** = inerte au boot ; worker de conduite seulement sur **manette** OU **profil lancé**. ✓
- **UI / stream** = le stream est **lourd** → écoute ultra-légère toujours-on (endpoint de contrôle),
  spawn du worker de stream à la **1re connexion** web, kill quand plus de client. ✓
- **Update** = déclenchable **terminal OU UI web** ; garde-fou « pas d'update en conduite ». ✓
- **Modèle → caméra** = poussable **ligne de commande OU UI web** (`.blob` depthai). ✓
- **Core ↔ workers** = **le plus léger** : lifecycle subprocess ; petit canal seulement si besoin. ✓
- Le core vit avec les services qui le lancent (PR #14, `feat/camera-hub-service`). ✓

## Squelette (livré)

Package `core/` (Python 3.6, sans `dataclass`) :
- `config.py` — charge `configs/vehicle.json` + `configs/profiles.json`.
- `workers.py` — `WorkerManager` : spawn/kill de workers nommés (subprocess).
- `gamepad.py` — `GamepadWatcher` : poll `/dev/input/js*` → arme le manuel.
- `control.py` — endpoint HTTP localhost (`/status`, `/takeover`, `/release`, `/ui/connect`, `/profile`, `/update`).
- `supervisor.py` — câble : **inerte au boot** ; profil lancé → auto worker, manette → manuel, prise de main explicite.
- `__main__.py` — `python3 -m core`.

**Stubs/TODO** : update (git pull + `sync-services.sh`), modèle→cam (P3/P4), détection « UI
connectée » réelle, et **décomposition perception/policy** (aujourd'hui l'auto worker = le
monolithe `controller_pd` / `inference_realcar`).


# Robocar Core — design (brouillon)

> **Statut : cadrage en cours.** Vision validée + structure proposée + questions ouvertes.
> Schémas : [`docs/schemas/`](schemas/). On fige le design avant d'implémenter.

## Rôle

Le **core** est le soft minimal lancé au boot (`robocar-core.service`). Il ne *fait* ni la
perception ni le contrôle : c'est un **superviseur** qui orchestre des workers en subprocess
selon le **profil** et le **contexte runtime**. Objectif : ne lancer que le strict nécessaire.

Voir [`schemas/01-architecture.md`](schemas/01-architecture.md).

## Responsabilités

1. **Lancer le minimum selon le contexte** (pas tout en permanence) — voir gamepad ci-dessous
   et [`schemas/04-boot-lifecycle.md`](schemas/04-boot-lifecycle.md).
2. **Cycle de vie** : déclencher les **updates** (git pull explicite + `sync-services.sh` +
   restart) — [`schemas/05-update-flow.md`](schemas/05-update-flow.md).
3. **Déployer des modèles sur l'OAK-D** (inférence on-camera) à la demande.
4. **Profils d'usage** : combinaison *lieu de perception × intelligence* —
   [`schemas/02-profiles.md`](schemas/02-profiles.md).

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
2. **Manette détectée** (`/dev/input/js*`) → le core **lance le worker de commande manuelle**,
   mais **passif** : le mode courant (ex. auto) **continue de tourner**.
3. **L'user clique « prendre la main »** → le core **coupe le contrôle auto** (s'il tournait) et
   passe en **manuel actif**.
4. **« rendre la main »** → retour au mode courant.

## Modèle de process

Le core **spawn/kill des workers** ; `cam-hub` reste un **service séparé** (dépendance), pas un
enfant du core. Signaux surveillés : présence manette, connexion UI, profil courant.

## Questions ouvertes (restantes)

1. **Détection « UI connectée »** : le core démarre le stream à la 1re connexion (écoute un port
   léger / endpoint) ? ou bouton « activer preview » ?
2. **Update** : déclenché d'où (UI / commande) ; garde-fou « pas d'update en conduite ».
3. **Modèle → caméra** : format `.blob` (depthai `NeuralNetwork`), qui convertit/pousse.
4. **Communication core ↔ workers** : lifecycle subprocess seul, ou un canal (signaux / IPC) ?
5. **Prérequis** : `main` n'a pas le stack worker hub-capable (vit sur `feat/track-mapping`) → à
   intégrer pour tester de bout en bout.

## Résolu

- Config = 2 couches JSON (statique véhicule / profils). ✓
- Manette = prise de main explicite (détection ≠ bascule). ✓
- Base = `feat/robocar-core` depuis `main`, branche dédiée. ✓

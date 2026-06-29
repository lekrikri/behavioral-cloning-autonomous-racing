# Robocar Core — design (brouillon)

> **Statut : brouillon de cadrage.** Capture la vision validée + une structure proposée +
> les questions ouvertes. À raffiner avant d'implémenter.

## Rôle

Le **core** est le soft minimal lancé au boot (`robocar-core.service`). Il ne *fait* ni la
perception ni le contrôle : c'est un **superviseur** qui orchestre les workers selon le
**profil** et le **contexte runtime**. Objectif : ne lancer que le strict nécessaire.

## Responsabilités

1. **Lancer le minimum selon le contexte** (à la demande, pas tout en permanence) :
   - pas de **streaming/UI** tant qu'aucune interface web n'est connectée ;
   - en mode **manuel** (manette détectée) → pas d'IA/algo de conduite autonome ;
   - en mode **autonome** → pas de pile de contrôle manuel.
2. **Cycle de vie du soft** : déclencher les **updates** (git pull explicite + `sync-services.sh`
   + redémarrage des workers concernés).
3. **Déployer des modèles sur l'OAK-D** (inférence on-camera) à la demande.
4. **Profils d'usage** : sélectionner une combinaison *lieu de perception × intelligence*.

## Profils (proposition)

| Profil | Perception | Intelligence | Idée |
|---|---|---|---|
| P1 | masque sur Jetson | algo (PD) | baseline actuelle |
| P2 | masque sur Jetson | IA (NN Jetson) | modèle BC sur Jetson |
| P3 | inférence **dans la caméra** | algo (PD) | décharge la Jetson |
| P4 | inférence **dans la caméra** | IA | tout déporté |

## Modèle de process (proposition)

```
robocar-core.service  ──>  core (superviseur, ce composant)
                              │ spawn/kill selon profil + contexte
        ┌─────────────────────┼───────────────────────┐
   cam-hub (service          perception           policy/intelligence
   séparé, dépendance)    (mask worker /        (controller_pd PD /
                           on-cam inference)     inference_realcar NN)
                                                       │
                                                  UI web (à la demande)
```

- Le core **spawn des workers en subprocess** et gère leur cycle de vie (pas de god-file).
- `cam-hub` reste un **service séparé** (dépendance), pas un enfant du core.
- Signaux runtime surveillés : présence manette (`/dev/input/js*`), connexion UI, profil courant.

## Questions ouvertes (à trancher avant de coder)

1. **Détection « UI connectée »** : le core écoute un port léger / un endpoint, et démarre le
   stream à la première connexion ? Ou un bouton « activer preview » ?
2. **Détection manette** : poll `/dev/input/js*` (hotplug via udev ?) → bascule manuel/auto.
3. **Update** : déclenché d'où (UI ? commande ?) et quoi exactement (code via git pull, services
   via `sync-services.sh`). Garde-fou : pas d'update pendant la conduite.
4. **Modèle → caméra** : format `.blob` (depthai `NeuralNetwork`), pipeline on-device, et qui
   convertit/pousse le modèle.
5. **Config des profils** : un fichier de config (cf. règle harness « params via config, pas en
   dur ») décrivant chaque profil et ses workers.
6. **Communication core ↔ workers** : juste lifecycle subprocess, ou un canal (signaux/IPC) ?
7. **Prérequis** : `main` n'a pas encore le stack worker hub-capable (vit sur `feat/track-mapping`)
   → à intégrer pour tester le core de bout en bout.

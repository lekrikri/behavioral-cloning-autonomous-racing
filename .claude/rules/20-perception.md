# Perception

## Raycasts polaires (parité simulateur)

- La représentation **principale** des distances est **polaire** : rayons angulaires depuis
  un centre, comme le simulateur (`configs/config.json` : `nbRay`, `fov`, `rayMaxDistance`).
  Le modèle a été entraîné sur ce format → le réel doit le reproduire.
- D'autres représentations sont tolérées en complément, mais **la polaire doit toujours
  rester disponible** et faire référence.

## FOV caméra : FIXE

- Le FOV de l'OAK-D est **fixe et connu** — ne pas le re-deviner ni le hardcoder à l'aveugle.
- Valeurs réelles (`getFov()`) : depth (CAM_B/C) ≈ **72.9°**, color (CAM_A) ≈ **68.79°**.
  (Le 97° historique dans `depth_to_rays.py` était FAUX → rayons mal projetés.)

## Masque de lignes

- Anti-artefacts (reflets, plinthes blanches) = couches structurelles empilées, togglables,
  **OFF par défaut**. Le **depth gating** est le différenciateur. Ne pas réintroduire un
  seuil couleur seul (il ne sépare pas le blanc achromatique des lignes).

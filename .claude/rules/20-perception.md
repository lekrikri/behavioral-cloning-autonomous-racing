# Perception

## Raycasts polaires (parité simulateur)

- La **seule** représentation de sortie est **polaire** : un faisceau régulier depuis le
  centre voiture, comme le simulateur (`configs/config.json` : `nbRay`, `fov`, `rayMaxDistance`).
  Le modèle a été entraîné sur ce format → le réel doit le reproduire.
- Implémentation : `src/mask/polar_rays.py` (`PolarRays`) via IPM sur le plan sol
  (`src/mask/camera_ground.py`, géométrie partagée). Les anciens raycasts scan-colonne
  (VisualRays) et depth (DepthToRays) ont été **supprimés** — ne pas les réintroduire.
- La depth n'est **plus** une source de rayons : elle sert de **filtre** du masque
  (rejet des surfaces verticales, voir ci-dessous).

## FOV caméra : FIXE

- Le FOV de l'OAK-D est **fixe et connu** — ne pas le re-deviner ni le hardcoder à l'aveugle.
- Valeurs réelles (`getFov()`) : depth (CAM_B/C) ≈ **72.9°**, color (CAM_A) ≈ **68.79°**.
  (Le 97° historique dans `depth_to_rays.py` était FAUX → rayons mal projetés.)

## Masque de lignes

- Source unique : `src/mask/white_line.py` (`white_line_mask`). Entrée = frame BGR
  (+ depth alignée), sortie = image binaire (lignes blanches, reste noir).
- Anti-artefacts = couches empilées **paramétrables** et désormais **ON par défaut** (le
  redesign vise la robustesse multi-situations) : CLAHE, gate achromatique-brillant
  (rejet sombres + couleurs), white top-hat (rejet grandes plages brillantes), morpho,
  **filtre depth** (rejet surfaces verticales vs plan sol) et filtre composantes
  (aire mini + rectilinéarité PCA). Réglables via le profil de perception (`configs/profiles/`).
- Ne pas réintroduire un seuil couleur seul (il ne sépare pas le blanc achromatique des
  lignes) : le **filtre depth** reste le différenciateur clé.

# Luxonis OAK-D Lite (autofocus)

> Caméra RGB + depth stéréo. Source des raycasts virtuels. Partie du [montage](../MONTAGE.md).

## Identité
- **Modèle** : Luxonis OAK-D Lite autofocus
- **P/N** : a00483
- **MxId** : `14442C1071DC48D400`
- **Connexion** : USB3 → Jetson Nano
- **SDK** : `depthai` **2.26** (PC de dev) — Boris utilise 2.21.2.0 sur la cible ⚠️ écart de version à surveiller

## Rôle IA
- Produit une **depth map** (distance en mm par pixel).
- On échantillonne cette depth map à **20 angles** (fov 180°, pas de ~9,47°) pour reconstituer les **20 raycasts** attendus par le modèle (bridge sim-to-real).
  ```
  col = int(width/2 + tan(angle_i) × focal_length)
  ray_i = min(depth_map[row_center, col] / MAX_DISTANCE, 1.0)   # [0,1]
  ```

## Capteurs disponibles
Cette caméra n'est pas un simple capteur RGB — elle embarque **trois sources image + une centrale inertielle** :
- **1× RGB couleur** `CAM_A` (IMX214, rolling shutter) → utilisée pour le **masque de lignes** (`visual_rays.py`).
- **2× mono** `CAM_B` / `CAM_C` (OV7251, **global shutter**, 480p, niveaux de gris natifs) → paire stéréo pour la **depth**.
- **depth map** dérivée de la paire mono (`depth_to_rays.py`).
- **IMU BMI270** (6-DoF, sans magnéto) — *cette unité l'a bel et bien* (confirmé par lecture réelle), exploitable pour détection de crash/dérapage.

## Pourquoi pas du monochrome pour la détection de lignes
Question récurrente : « les lignes sont blanches, autant passer en monochrome (on a déjà 2 capteurs mono global-shutter) ». **Non — ce serait contre-productif.** Raisonnement :

- Le masque actuel (`white_line_mask`, mode HSV) sépare le blanc par **`S ≤ 40` (désaturé) ET `V` élevé (brillant)**. Le critère `R≈G≈B` (faible saturation) est *le* discriminant couleur.
- Passer en niveaux de gris **jette ce critère** : on ne garde que la luminance. On **perdrait** la capacité de rejeter les objets **colorés** brillants — que le seuil `S ≤ 40` élimine déjà gratuitement.
- ⚠️ Le monochrome **n'aide pas** sur le vrai problème (faux positifs : plinthes blanches, tapis clair, reflets). Ces artefacts sont **achromatiques par nature** (blanc diffus) → signature `V` élevé / `S` faible **identique** aux lignes. Aucun seuil couleur — HSV *ou* mono — ne peut les distinguer.
  - Nuance (confirmée par recherche, *Springer AI Review 2025*) : la saturation HSV rejette les reflets **spéculaires** (lumière vive = faible saturation, cue exploitable) mais **pas** les surfaces **diffuses** blanches. Le mono perdrait même le premier sans rien gagner sur le second.
- **Conclusion** : le rejet des artefacts doit être **structurel/géométrique**, pas chromatique. Leviers retenus (cf. `white_line_mask` couches 1-3 + depth gating) :
  1. **top-hat** morphologique → fin (ligne) vs large (tapis/plinthe/gros reflet) ;
  2. **filtre de forme** (étirement par composante) → ligne allongée vs reflet patatoïde ;
  3. **cohérence temporelle** (médiane par rayon) → tue les reflets spéculaires qui scintillent ;
  4. **depth gating** (mode `fusion`) → rejette le vertical hors-sol (plinthes) — *l'atout unique de cette caméra, qu'aucun projet caméra-seule n'exploite*.

Les capteurs mono/global-shutter gardent un intérêt **ailleurs** (depth, robustesse au flou de mouvement à vitesse), pas pour discriminer le blanc.

## Quirks / pièges connus
- ⚠️ **`MAX_DISTANCE` à calibrer** sur la piste réelle (3 m ? 5 m ?).
- ⚠️ **`ray_stats.json` obligatoire** : le Z-score doit être identique entraînement ↔ inférence.
- ~30 FPS → le modèle peut tourner plus vite qu'en simulation.

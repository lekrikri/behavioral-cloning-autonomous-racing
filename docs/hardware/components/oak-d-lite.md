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

## Quirks / pièges connus
- ⚠️ **`MAX_DISTANCE` à calibrer** sur la piste réelle (3 m ? 5 m ?).
- ⚠️ **`ray_stats.json` obligatoire** : le Z-score doit être identique entraînement ↔ inférence.
- ~30 FPS → le modèle peut tourner plus vite qu'en simulation.

# Procédure de calibration — perception polaire

À exécuter **sur la Jetson** après le passage au faisceau polaire, **avant tout roulage**.
Trois étapes : géométrie caméra → recalibration `ray_stats` → vérification roues en l'air.

> Contexte : le faisceau polaire (`src/mask/polar_rays.py`) projette les pixels du masque
> sur le plan sol via l'IPM (`src/mask/camera_ground.py`). Cette projection dépend de la
> **géométrie de montage** (hauteur + pitch), et le modèle attend une **distribution de
> rayons** cohérente (`ray_stats`). Les deux doivent être recalés une fois.

## Prérequis

- Hub caméra actif : `sudo systemctl status robocar-cam-hub` (sinon `restart`).
- Depuis le PC : `ssh -L 8088:localhost:8088 robocar` puis `http://localhost:8088`.
- Toutes les commandes Python préfixées par `OPENBLAS_CORETYPE=ARMV8` (SIGILL numpy sur le Tegra).

---

## 1. Géométrie caméra (`cam_height_m`, `cam_pitch_deg`)

Valeurs dans `configs/profiles/classic.json` — aujourd'hui des **placeholders**.

### 1a. Hauteur (mesure directe)
Mètre ruban : du **sol** au **centre optique** de la caméra couleur (CAM_A). Reporter en
mètres dans `cam_height_m`.

### 1b. Pitch (calage sur repère au sol)
1. Poser une bande adhésive **en travers de la piste, bien centrée**, à distance connue
   **D** devant l'axe des roues avant (ex. **1.00 m**).
2. Lancer l'outil de lecture live en testant une valeur de pitch :
   ```bash
   OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.calibrate_geometry --pitch 18
   ```
   Lire `rayon proche = X.XX m @ ~0°` (le repère centré tape un rayon proche de l'axe).
3. Ajuster `--pitch` (± quelques degrés) jusqu'à ce que **X.XX ≈ D**. `--height` peut aussi
   être testé sans éditer le fichier (`--height 0.16`).
4. Vérifier avec une **2e bande** à une autre distance (ex. 1.5 m) : la lecture doit suivre.
5. Reporter `cam_pitch_deg` (et `cam_height_m` si ajusté) dans `configs/profiles/classic.json`.

> Astuce : un pitch trop faible surestime les distances lointaines (rayons « fuient » vers
> l'horizon) ; trop fort les écrase. Le repère à distance connue tranche.

---

## 2. Recalibration `ray_stats` (Z-score réel)

Le passage scan-colonne → vrai polaire change la **distribution** des rayons : les anciennes
stats réelles sont **invalides**.

1. Supprimer l'ancien fichier réel sur la Jetson (il n'existe pas dans le repo) :
   ```bash
   rm -f models/real_ray_stats.json
   ```
   (En son absence, l'inférence retombe sur `models/ray_stats.json` — le baseline **sim**,
   désormais cohérent puisque le polaire reproduit la simu — avec un avertissement.)
2. Lancer la collecte 3 phases (pousser la voiture à la main, moteur coupé) :
   ```bash
   OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.calibrate_rays
   ```
   - Phase 1 : lignes droites · Phase 2 : virages doux · Phase 3 : virages serrés.
   - Produit `models/real_ray_stats.json` (rechargé automatiquement par l'inférence).

---

## 3. Vérification — ⚠️ roues en l'air d'abord

1. **Masque/faisceau** : `http://localhost:8088` → page **Masque**. Vérifier que les lignes
   ressortent en blanc, le reste noir, et que le **faisceau polaire** épouse les bords. Régler
   les sliders au besoin, reporter dans le profil (cf. [`MASK_TUNING.md`](MASK_TUNING.md)).
2. **Inférence, roues en l'air** : page **Accueil** → profil `classic` → **PLAY** avec un
   `duty-max` bas. Bouger la voiture devant une piste : la direction doit réagir de façon
   cohérente (ligne à gauche → braque à droite, etc.).
3. **Au sol, prudemment** : seulement une fois 1 et 2 validés. **STOP** = coupe tout.

---

## Récapitulatif fichiers

| Quoi | Où |
|---|---|
| Géométrie + filtres + polaire | `configs/profiles/classic.json` |
| Lecture live géométrie | `src/tools/calibrate_geometry.py` |
| Collecte `ray_stats` réel | `src/tools/calibrate_rays.py` → `models/real_ray_stats.json` |
| Baseline sim (fallback, conservé) | `models/ray_stats.json` |

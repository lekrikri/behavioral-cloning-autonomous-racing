# Réglage du masque de lignes — couches anti-artefacts

> Procédure pour régler les couches qui éliminent les **faux positifs** du masque de
> lignes blanches (plinthes, reflets spéculaires, tapis clair). Outil : [`src/mask_stream.py`](../src/mask_stream.py).
> Voir aussi [pourquoi pas le monochrome](hardware/components/oak-d-lite.md#pourquoi-pas-du-monochrome-pour-la-détection-de-lignes).

## Le principe (à comprendre avant de régler)

Les artefacts sont **blancs et désaturés comme les lignes** → aucun seuil de couleur (HSV
ou monochrome) ne peut les distinguer. On les rejette donc par leur **structure
géométrique**, via 4 couches empilées et **complémentaires** (chacune attrape ce que les
autres laissent passer). Elles sont *count-agnostic* : indépendantes du nombre de lignes
visibles et de l'écartement (qui varient sur notre circuit).

| Couche | Cible | Idée |
|--------|-------|------|
| seuil V (baseline) | base du masque | garder le blanc brillant |
| top-hat | tapis, plinthe, gros reflet | ne garder que les structures **fines** |
| filtre de forme | reflets ronds/pleins | ne garder que les blobs **étirés** |
| cohérence temporelle | reflets qui **scintillent** | médiane par rayon dans le temps |

> ⚠️ Un faux **négatif** (ligne effacée) est plus grave qu'un faux **positif** (artefact
> laissé) : la policy BC a été entraînée à *voir* la ligne. Réglez toujours pour ne
> jamais perdre la ligne, même si quelques artefacts passent.

## L'outil : `mask_stream.py`

Traitement sur la Jetson, affichage déporté fluide (MJPEG) dans le navigateur du PC.

```bash
# Sur la Jetson
~/mask_test/run-stream.sh
#   ou : cd ~/mask_test && OPENBLAS_CORETYPE=ARMV8 python3 mask_stream.py

# Sur le PC
ssh -L 8088:localhost:8088 robocar      # tunnel
# puis ouvrir http://localhost:8088 dans un navigateur
```

`OPENBLAS_CORETYPE=ARMV8` est **obligatoire** (sinon `numpy`/`cv2` crashent en SIGILL sur ce Jetson).

### Contrôles (clavier dans le navigateur, ou boutons)

| Touche | Effet | | Touche | Effet |
|--------|-------|-|--------|-------|
| `t` | top-hat on/off | | `c` | cohérence temporelle on/off |
| `,` / `.` | kernel top-hat − / + | | `b` / `n` | fenêtre temporelle − / + |
| `[` / `]` | seuil top-hat − / + | | `+` / `-` | seuil V (blanc) + / − |
| `f` | filtre de forme on/off | | `m` | masque côte-à-côte |
| `;` / `'` | `max_fill` − / + | | `r` | raycasts |

L'état courant (valeurs actives + fps) s'affiche en surimpression sur l'image.

## Les paramètres, concrètement

| Param (touches) | Ce que ça fait | Trop **bas** → | Trop **haut** → |
|---|---|---|---|
| **seuil V** (`+`/`-`) | luminance min pour être « blanc » | bruit, surfaces grises captées | la ligne se troue / disparaît |
| **tophat_k** (`,`/`.`) | taille du noyau : sépare *fin* de *large* | l'artefact large survit | la ligne (trop proche de la taille du noyau) s'efface |
| **tophat_thresh** (`[`/`]`) | force min de la réponse top-hat | plages faiblement brillantes passent | la ligne se troue |
| **max_fill** (`;`/`'`) | « plénitude » max d'un blob (aire/bbox) | un segment de ligne court/de face tombe | les reflets ronds passent |
| **fenêtre temp.** (`b`/`n`) | nb de frames de la médiane | scintillement non filtré | latence + traînée sur les bords qui apparaissent |

## Procédure de réglage (ordre impératif)

Réglez **une couche à la fois, dans cet ordre**, sur une scène qui contient **à la fois une
vraie ligne ET l'artefact à tuer**. À chaque étape, surveillez les deux échecs opposés
(l'artefact survit / la ligne s'efface) et cherchez la **marge la plus large** entre eux.

0. **Conditions réelles.** Réglez sous l'éclairage et sur le sol de la **compétition** (les
   reflets sont spécifiques à la lumière), et à la **résolution/ROI d'inférence** (les
   largeurs en pixels changent avec la résolution).

1. **Baseline — tout OFF.** Ajustez le **seuil V** (`+`/`-`) jusqu'à ce que la ligne soit
   **pleine et continue** avec un minimum de bruit. Les couches suivantes raffinent une
   bonne base, elles ne réparent pas une mauvaise. Notez la valeur.

2. **Top-hat (`t`).** Le noyau doit être **plus grand que la largeur de ligne, plus petit
   que l'artefact**. Départ : `k ≈ 2-3×` la largeur de ligne en pixels.
   - Ligne qui disparaît → `k` trop petit ou `thresh` trop haut → `.` (k+) / `[` (thr−).
   - Artefact large qui survit → `k` trop grand → `,` (k−).
   - Validez sur le **pire** artefact : il doit disparaître, la ligne rester continue.

3. **Filtre de forme (`f`).** Baissez `max_fill` (`;`) jusqu'à ce que le reflet rond tombe.
   Si la ligne commence à tomber (segment court vu de face = fill plus élevé) → remontez (`'`).

4. **Cohérence temporelle (`c`).** **Caméra en mouvement** (un test statique masque la
   latence), trouvez la **plus petite fenêtre** qui tue le scintillement spéculaire.
   3-5 est le sweet spot à 30 fps. Plus grand = plus de latence.

5. **Tout ensemble + test adverse.** Activez les 3 couches et essayez de **casser** :
   tous les types d'artefacts, la zone la plus mal éclairée, un virage serré (ligne
   diagonale/courte). Top-hat et filtre de forme sont les plus susceptibles d'effacer une
   ligne légitime dans ces cas limites — vérifiez qu'elle survit partout où elle doit.

> 💡 Jugez sur les **raycasts** (touche `r`), pas seulement sur le masque : c'est le vecteur
> de 20 rayons que la policy consomme. Un beau masque avec de mauvais rayons est inutile.

## Figer les valeurs trouvées

Une fois les bonnes valeurs identifiées, reportez-les comme **défauts** dans le code (elles
sont aujourd'hui OFF par défaut pour préserver le comportement historique) :

- [`src/visual_rays.py`](../src/visual_rays.py) — défauts de `white_line_mask()` et `VisualRays`
  (`tophat_k`, `tophat_thresh`, `max_fill_ratio`, `temporal_window`).
- [`src/live_mask_oak.py`](../src/live_mask_oak.py) — constantes `TOPHAT_K_ON` / `MAX_FILL_ON` / `TEMPORAL_ON`.

Le mode `visual`/`fusion` de [`src/inference_realcar.py`](../src/inference_realcar.py) lit
ces défauts via `VisualRays` → les bonnes valeurs s'appliquent automatiquement à l'inférence.

# Visual Teach & Repeat par signatures de rays

> Comment se localiser **uniquement à la caméra**, sans LiDAR, sans IMU, sans pose
> globale métrique — en réutilisant notre pipeline `U-Net → masque → rays`.
> C'est la brique « localisation » de la navigation algorithmique.
> Vue d'ensemble et arbitrage des options : [`NAVIGATION_ALGORITHMIQUE.md`](NAVIGATION_ALGORITHMIQUE.md).

---

## 1. Le principe : localisation *relative*, pas *globale*

Visual Teach & Repeat (VT&R) découpe le problème en deux passes :

```
TEACH  (tour manuel)                 REPEAT (tours auto)
────────────────────                 ───────────────────
On enregistre une SÉQUENCE de        À chaque frame :
keyframes le long du parcours :        1. trouver le keyframe courant (matching)
  - signature de rays                  2. mesurer l'écart latéral/cap vs ce keyframe
  - commande conduite                  3. corriger pour s'y aligner
  - distance cumulée (odométrie)       4. avancer vers le keyframe suivant
```

> **Pourquoi ça ne dérive pas** : on ne maintient jamais un `(x, y, θ)` global qui
> accumulerait l'erreur. À chaque keyframe atteint, la position est *recalée* sur
> la séquence enregistrée. L'erreur est bornée par l'espacement des keyframes, pas
> par la longueur parcourue. C'est ce qui rend VT&R robuste là où le SLAM métrique
> dérive.

On ne demande donc jamais « où suis-je sur la planète » (fragile à la caméra),
mais « par rapport au bout de piste enregistré ici, je suis décalé de combien ? »
(local, robuste).

---

## 2. La signature : pourquoi les rays plutôt qu'ORB

Plutôt que des features visuelles ORB (sensibles au flou et à la lumière), on
décrit chaque keyframe par **le vecteur de rays** déjà produit par l'U-Net.

```
signature = [d_0, d_1, ..., d_19]   # 20 distances aux bords de piste, en mètres
```

Avantages :

- **Compact** : 20 floats par keyframe. Un tour complet ≈ quelques centaines de
  keyframes → quelques Ko. Matcher = comparer des 20-vecteurs → trivial sur le Nano.
- **Robuste à la lumière** : le masque U-Net normalise déjà l'apparence ; la *forme*
  du couloir (le profil de distances) change peu avec l'éclairage.
- **Zéro coût supplémentaire** : c'est déjà la sortie de `visual_rays.py`. On
  réutilise la perception existante au lieu d'empiler ORB-SLAM.

---

## 3. Phase TEACH (tour à la manette)

On enregistre, à intervalle régulier (distance ou temps), un keyframe :

```python
# Keyframe recorded during the manual teach lap.
Keyframe = {
    "idx":       int,          # position in the sequence
    "rays":      list[float],  # ray signature [d_0..d_19], metres
    "s":         float,        # cumulative arc-length since start (odometry), metres
    "steering":  float,        # driver command at this point (feed-forward prior)
    "accel":     float,
}
```

Notes :

- `s` (distance cumulée) vient de l'**odométrie** (tours roue / ERPM VESC). Elle a
  le droit de dériver : on ne s'en sert pas pour une pose absolue, juste pour
  ordonner les keyframes et estimer la progression locale.
- La séquence boucle : le dernier keyframe se referme près du premier (ligne de
  départ) → on connaît la longueur totale du tour.
- Espacement typique à régler : un keyframe tous les ~10–30 cm (compromis
  précision de recalage ↔ taille mémoire).

---

## 4. Phase REPEAT (tours autonomes)

### 4.1 Matching — quel keyframe suis-je en train de vivre ?

Naïvement : prendre le keyframe dont la signature est la plus proche.

```python
def match(current_rays, keyframes):
    # L2 distance between ray signatures
    return min(keyframes, key=lambda kf: l2(current_rays, kf["rays"]))
```

**MAIS** — piège du FOV limité (cf. §6) : sur une ligne droite, plein de keyframes
ont presque la même signature. Solution : **matching séquentiel avec contrainte de
monotonie**. On ne cherche pas dans tout le tour, mais dans une *fenêtre* autour du
keyframe précédent, et on n'autorise que la progression avant :

```python
def match_sequential(current_rays, keyframes, last_idx, window=15):
    # Search only forward, in a small window around the last matched keyframe.
    candidates = keyframes[last_idx : last_idx + window]
    best = min(candidates, key=lambda kf: l2(current_rays, kf["rays"]))
    return best
```

→ supprime les faux matchs lointains (autre virage qui « ressemble »), et lève
l'ambiguïté des lignes droites en s'appuyant sur la continuité du parcours.

### 4.2 Correction — de combien suis-je décalé ?

Une fois le keyframe courant connu, on compare la signature *vécue* à la signature
*enregistrée* pour en déduire l'écart latéral et de cap :

```
Si les rays de GAUCHE sont plus courts que prévu et ceux de DROITE plus longs
  → la voiture a dérivé vers la GAUCHE → corriger à droite.
```

Concrètement, l'asymétrie gauche/droite du résidu `current_rays - kf["rays"]`
donne le signe et l'amplitude de l'écart latéral `e_lat`. Un terme de cap `e_psi`
se déduit du gradient avant/arrière des rays.

### 4.3 Commande — pure pursuit vers le keyframe suivant

```python
# Feed-forward (driver prior) + feedback (lateral/heading correction).
steering = kf_next["steering"] + Kp_lat * e_lat + Kp_psi * e_psi
accel    = kf_next["accel"]              # or a separate speed profile
```

Le `steering` enregistré du conducteur sert de **terme d'anticipation** (on sait
déjà grossièrement comment tourne ce virage) ; la correction PID ne fait que
rattraper l'écart. C'est ce qui rend le suivi doux même avec une localisation
approximative.

---

## 5. Boucle complète (pseudo-code)

```python
last_idx = 0
while driving:
    rays = unet_rays(camera_frame())          # perception (visual_rays.py)
    kf   = match_sequential(rays, KEYFRAMES, last_idx)
    last_idx = kf["idx"]

    e_lat, e_psi = lateral_heading_error(rays, kf["rays"])
    kf_next = KEYFRAMES[(kf["idx"] + LOOKAHEAD) % len(KEYFRAMES)]

    steering = kf_next["steering"] + KP_LAT * e_lat + KP_PSI * e_psi
    accel    = kf_next["accel"]
    send_to_vesc(steering, accel)              # cf. CONTROL_STACK.md
```

---

## 6. Limites assumées

- **FOV limité (~120°, vers l'avant)** : sur une longue ligne droite, le profil de
  rays est quasi constant → on ne sait pas *où* on est le long de la droite. Le
  matching séquentiel (§4.1) contourne le problème pour le *suivi* (rester centré
  suffit), mais une localisation métrique pure casserait ici. Raison de plus de
  rester en VT&R plutôt qu'en SLAM global.
- **Robustesse perception = robustesse localisation** : si l'U-Net rate le masque
  (lumière extrême, obstacle inattendu), la signature est fausse. Prévoir un
  garde-fou : si `l2(rays, kf["rays"])` dépasse un seuil, basculer en repli réactif
  (Follow-the-Gap) plutôt que suivre un keyframe douteux.
- **Conditions du teach** : la luminosité/le décor du tour manuel doivent
  ressembler à ceux des tours auto. Re-teacher si l'environnement change beaucoup.

---

## 7. Vers la trajectoire optimale (extension niveau 2)

VT&R de base rejoue **la ligne manuelle conduite**, pas la raceline optimale. Pour
viser l'optimal sans repasser au SLAM global :

1. Pendant le teach, reconstruire les bords de piste à partir des rays + `s`
   (les keyframes forment déjà un graphe du circuit).
2. Calculer offline la raceline min-curvature dans ce repère « keyframes ».
3. À chaque keyframe, stocker l'**offset latéral** entre la ligne conduite et la
   ligne optimale → en repeat, on vise `e_lat_cible = offset` au lieu de `0`.

On garde ainsi le recalage local robuste de VT&R, tout en suivant une ligne
meilleure que celle du conducteur. À ne tenter qu'**après** un niveau 1 qui marche.

---

## 8. Plan de prototypage

1. **Sim d'abord** : remplacer `unet_rays()` par les raycasts du simulateur
   (propres) → valider le teach, le matching séquentiel, la correction latérale.
2. Régler `window`, `LOOKAHEAD`, `KP_LAT`, `KP_PSI`, l'espacement des keyframes.
3. **Portage réel** : ne changer que la source des rays (U-Net). Le reste de la
   logique est identique.
4. Ajouter le garde-fou Follow-the-Gap (§6) avant les essais sur la vraie voiture.

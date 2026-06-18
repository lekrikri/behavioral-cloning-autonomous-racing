# Navigation algorithmique — SLAM, raceline & suivi de ligne

> Piste *non-apprentissage* pour la « partie intelligence » de la voiture :
> au lieu du Behavioral Cloning / RL, faire de l'algo classique de course autonome.
> Idée de départ : **un tour manuel pour cartographier → calcul de la trajectoire
> optimale → suivi de cette ligne**.
> Cette doc pose le pipeline, identifie le *vrai* point dur, et arbitre entre les
> options compte tenu de notre matériel **figé** (OAK-D Lite, pas de LiDAR, pas d'IMU).
>
> Deep-dive de la technique retenue : voir [`VISUAL_TEACH_REPEAT.md`](VISUAL_TEACH_REPEAT.md).

---

## 1. L'idée en trois étapes

```
Tour 1 (manette)            Offline                    Tours suivants (auto)
─────────────────           ───────                    ─────────────────────
"pseudo-SLAM"        →      raceline optimale     →    suivi de ligne
cartographier le            (min-curvature /            (pure pursuit sur
circuit                     min-time)                   la ligne virtuelle)
```

C'est **l'architecture standard** de la course autonome (stack F1TENTH, optimiseur
de trajectoire de la TUM). L'intuition est bonne, ce n'est pas une lubie. Mais le
découpage en 3 étapes masque une 4ᵉ brique invisible qui fait ou défait le projet.

---

## 2. Le vrai point dur : la localisation, pas la ligne optimale

Contre-intuitif, mais central :

| Étape                        | Difficulté réelle | Pourquoi |
|------------------------------|-------------------|----------|
| Calcul de la raceline        | **Facile**        | Offline, problème résolu, code open-source (TUM `global_racetrajectory_optimization`). On a tout le temps. |
| Suivi de ligne (pure pursuit)| **Facile**        | ~30 lignes : viser un point cible devant soi, braquage géométrique. |
| **Localisation en ligne**    | **DUR**           | La raceline est stockée dans un repère fixe → à chaque frame il faut estimer la pose `(x, y, θ)` de la voiture dans CE repère. |

> **Le piège** : le « bête suivi de ligne » n'est bête que si on connaît en
> permanence la position de la voiture *par rapport à la ligne*. L'odométrie seule
> dérive (les erreurs s'accumulent) → au bout d'un demi-tour, on suit une ligne
> décalée d'un mètre. La vraie stack de course rajoute donc un **localisateur**
> (filtre particulaire AMCL, scan-matching) qui recale la pose en continu.

C'est cette brique-là qu'il faut dimensionner le projet autour, pas l'optimiseur.

---

## 3. Contrainte matérielle : caméra uniquement

Le matériel est **figé** : OAK-D Lite + Jetson Nano. Conséquences directes :

- **Pas de LiDAR 2D** → on se prive de la voie royale (Cartographer/Hector SLAM
  pour le mapping, AMCL pour la localisation, tout prêt en ROS).
- **OAK-D Lite sans IMU** → pas de fusion visuo-inertielle (VIO). L'IMU aurait
  surtout aidé sur les rotations rapides en virage.
- **MAIS c'est une caméra stéréo** → la profondeur métrique est donnée par
  triangulation. Le problème nº1 de la vision *monoculaire* (ambiguïté d'échelle :
  petit-et-proche vs gros-et-loin) est donc **déjà résolu**. On part moins handicapé
  qu'il n'y paraît.

---

## 4. Pourquoi le SLAM visuel métrique « global » est le chemin fragile

Tentation : ORB-SLAM3 (stéréo) ou RTAB-Map pour obtenir une pose globale continue.
Un **circuit de course est presque le pire cas possible** pour ces méthodes :

- **Flou de mouvement** à vitesse → les features visuelles s'étalent → le tracking
  décroche. La course est l'ennemi nº1 de l'odométrie visuelle.
- **Murs lisses / répétitifs** → peu de texture → peu de features → perte de
  localisation.
- **Virages serrés** → grande rotation inter-frame → décrochage (et c'est là que
  l'IMU absent ferait le plus mal).
- **Compute Nano** → ORB-SLAM3 stéréo y rame → framerate bas → plus grand
  déplacement entre frames → cercle vicieux.

→ Possible en théorie, **non recommandé** comme pari principal.

---

## 5. Le bon pari : abandonner la pose globale métrique

La sortie élégante, c'est de **changer de question** :

- ❌ « Où suis-je dans un repère global `(x, y)` ? » → dérive, fragile.
- ✅ « Par rapport au bout de circuit que j'ai enregistré ici, je suis décalé de
  combien latéralement ? » → problème *local*, sans dérive, robuste.

C'est le paradigme **Visual Teach & Repeat (VT&R)**, qui est *exactement* l'idée
« un tour manuel puis on suit », mais formalisée pour être camera-native et
drift-free. Détails et pseudo-code : [`VISUAL_TEACH_REPEAT.md`](VISUAL_TEACH_REPEAT.md).

**Atout maison** : notre pipeline `U-Net → masque piste → ~20 rays` transforme déjà
la caméra en pseudo-LiDAR. On peut construire la *signature* de chaque keyframe à
partir du vecteur de rays (compact, robuste à la lumière, déjà calculé) au lieu
d'empiler un SLAM visuel lourd.

---

## 6. Recommandation : une montée en escalier

| Niveau | Technique | Carte ? | Localisation ? | Chrono | Risque |
|--------|-----------|---------|----------------|--------|--------|
| **0 — Baseline** | Follow-the-Gap / centerline sur les rays | non | **aucune** | moyen | très faible |
| **1 — VT&R** | Teach-and-Repeat par signatures de rays | séquence de keyframes | locale (relative au keyframe) | bon | faible |
| **2 — Optimal** | raceline min-curvature recalée sur les keyframes | oui | relative + offset raceline | optimal | moyen |

**Ordre conseillé :**

1. **Niveau 0 d'abord** — faire *rouler* la voiture sans aucune localisation. Filet
   de sécurité, valide toute la chaîne capteurs → commande (cf. [`CONTROL_STACK.md`](CONTROL_STACK.md)).
2. **Niveau 1 (VT&R)** — incarnation robuste de l'idée d'origine, réutilise l'U-Net.
   C'est le cœur du projet algo.
3. **Niveau 2** — seulement si le chrono du niveau 1 ne suffit pas.

> ⚠️ **Tension niveau 1 → 2** : VT&R rejoue *la ligne manuelle conduite*, pas la
> raceline optimale. Pour viser l'optimal il faut exprimer la ligne optimale
> *relativement aux keyframes* et la suivre via eux — ce qui réintroduit une part
> de la complexité « pose relative à la carte ». À ne tenter qu'après le niveau 1.

---

## 7. Où prototyper

Tout se valide **d'abord dans le simulateur** : les raycasts y sont propres et
l'odométrie y est fiable, ce qui permet d'écrire la logique VT&R / raceline sans
se battre en même temps contre le bruit de perception réel. Le portage sur la
Jetson ne change que la *source* des rays (U-Net au lieu des raycasts sim).

---

## Références

- Stack de course autonome : **F1TENTH** (Follow-the-Gap, pure pursuit, AMCL).
- Optimiseur de trajectoire : **TUM `global_racetrajectory_optimization`**
  (minimum-curvature / minimum-time raceline).
- Paradigme caméra : **Visual Teach & Repeat** (Furgale & Barfoot) — voir doc dédiée.

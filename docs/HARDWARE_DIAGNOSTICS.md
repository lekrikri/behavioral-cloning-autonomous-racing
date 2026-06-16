# Diagnostics Hardware — Journal

> Journal vivant des problèmes matériels rencontrés sur la voiture réelle (Jetson Nano)
> et de leur résolution. On complète ce document au fur et à mesure de l'avancement.
> Voir aussi : [`HARDWARE_REALCAR.md`](HARDWARE_REALCAR.md) (référence matérielle),
> [`VESC_DIAGNOSTIC_RESUME.md`](VESC_DIAGNOSTIC_RESUME.md).

## Chaîne d'alimentation (rappel)

```
Batterie LiPo
      ↓
[Flipsky Anti-Spark Switch Smart 13S]   ← interrupteur principal (anti-étincelle + MOSFET piloté)
      ↓
[Carte puissance centrale + radiateur]
      ├──→ [Matek UBEC Duo] ── 5V ──→ Jetson Nano (barrel jack, J48 jumper)
      └──→ [Flipsky FSESC Mini V6.7 Pro] ──→ moteur + servo
```

---

## 2026-06-15 — La voiture se coupe et reboote toute seule (~20 min, à l'arrêt) ✅ RÉSOLU

### Symptôme
La voiture s'éteint et redémarre seule, à intervalle régulier (~15–20 min), **uniquement à
l'arrêt / sur l'établi**, jamais observé en roulant. Sessions SSH coupées net.

### Cause racine
Le **Flipsky Anti-Spark Switch Smart (13S / 150 A)**, câblé entre la batterie et l'UBEC,
possède un **minuteur d'auto-extinction de 20 minutes** non désactivable. Citation de la
documentation Flipsky :

> *« Will turn off automatically after 20 minutes. The output voltage must vary by at least
> 500mV within any three second interval to reset the turn-off timer. »*

Le minuteur ne se réarme que si la tension de sortie du switch **varie d'au moins 500 mV en
3 s**. Or :

- **À l'arrêt** : la Jetson tire **~0,38 A constant** (`POM_5V_IN ≈ 1876 mW`) à travers un UBEC
  régulé → tension de sortie parfaitement stable → minuteur **jamais réarmé** → coupure à 20 min.
- **En roulant** : les moteurs créent de gros appels de courant → la tension oscille bien
  au-delà de 500 mV en permanence → minuteur réarmé en continu → **ne coupe jamais**.

C'est pour cette raison que le problème ne se manifeste qu'au banc / en développement.

### Preuves (relevées sur la Jetson)
| Vérification | Résultat | Interprétation |
|---|---|---|
| `last` | sessions en `crash`, jamais `shutdown` | coupure brutale, pas d'arrêt logiciel |
| `dmesg` | `last reset is due to power on reset`, `PMC reset status reg: 0x0` | reset à froid = **perte d'alimentation** (ni watchdog, ni panic, ni soft-reboot) |
| `thermal_zone*` | 28–50 °C | **pas** une coupure thermique |
| `tegrastats` | `POM_5V_IN ≈ 1876 mW` (~0,38 A) | conso ridicule → **pas** une saturation courant de l'UBEC |
| `nvpmodel -q` | MAXN | conso pic max (aggravant, mais pas la cause) |

Hypothèses **écartées par les données** : extinction logicielle, watchdog/kernel panic,
surchauffe, UBEC sous-dimensionné en courant.

### Correctif
L'auto-off de 20 min est une sécurité firmware **figée** (pas de réglage pour la désactiver).
Solution = **ne pas faire passer l'alimentation de la Jetson par le Flipsky** :

1. **★ Recommandé** — alimenter l'**UBEC directement sur la batterie** (avec un **XT90-AS
   passif** pour conserver l'anti-étincelle au branchement). Réserver le Flipsky à la **ligne
   moteur/ESC**, où le courant varie naturellement → son minuteur ne se déclenche jamais.
2. Palliatif de validation : maintenir une charge qui perturbe le rail ≥ 500 mV / 3 s (sale).
3. Appui sur le bouton du switch = réarme le minuteur (inutilisable en autonome).

### Test de confirmation visuel (sans outil)
Laisser tourner à vide et observer la **LED du Flipsky** au moment du reset :
- LED qui **s'éteint** → c'est bien le switch qui coupe (confirmé).
- LED qui **reste allumée** → chercher en aval (UBEC, barrel/J48, UVLO batterie).

### Statut
- [x] Cause identifiée et documentée
- [ ] Recâblage UBEC hors antispark (à faire — partie matériel)
- [ ] Validation : > 20 min à vide au banc sans reboot

### Références
- Flipsky Anti-Spark Switch Smart — <https://flipsky.net/collections/anti-spark-switch>
- Comportement auto-off (fiche distributeur) —
  <https://dronespark.myshopify.com/products/flipsky-antispark-switch-smart-enhanced-200a-for-electric-skateboard-ebike-scooter-robots>

---

<!--
Modèle pour une nouvelle entrée :

## AAAA-MM-JJ — <titre court du symptôme> [✅ RÉSOLU | 🔬 EN COURS | ❌ BLOQUÉ]

### Symptôme
### Cause racine
### Preuves
### Correctif
### Statut
### Références
-->

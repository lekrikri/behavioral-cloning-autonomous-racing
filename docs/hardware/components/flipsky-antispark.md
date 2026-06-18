# Flipsky Anti-Spark Switch Smart 13S

> Interrupteur principal de puissance + anti-étincelle. Partie du [montage](../MONTAGE.md).

## Identité
- **Modèle** : Flipsky Anti-Spark Switch Smart
- **Tension max** : **13S (60 V)** — plafond, pas la tension d'usage (la batterie de la voiture est plus basse).
- **Courant** : **60 A continu / 800 A crête**.
- **Liens** :
  - <https://flipsky.net/collections/anti-spark-switch>
  - <https://dronespark.myshopify.com/products/flipsky-antispark-switch-smart-enhanced-200a-for-electric-skateboard-ebike-scooter-robots>

## Rôle
- Bouton ON/OFF principal de la voiture.
- **Anti-étincelle** : précharge les condensateurs du VESC via une résistance avant de fermer le MOSFET → supprime l'arc électrique au branchement de la batterie.
- Câblé en **XT60**. **Actuel** : sa sortie alimente **à la fois** le VESC et l'UBEC (→ il coupe tout à 20 min). **Cible** : le réserver à la **ligne moteur/VESC** et déporter l'UBEC sur la batterie.

## Quirks / pièges connus
- ⚠️ **Auto-extinction 20 min non désactivable** : se coupe seul si la tension de sortie ne varie pas d'au moins **500 mV en 3 s**. Citation Flipsky :
  > *« Will turn off automatically after 20 minutes. The output voltage must vary by at least 500mV within any three second interval to reset the turn-off timer. »*
- Conséquence : **ne jamais y faire passer l'alimentation de la Jetson** (courant constant → coupure au banc). Cf. [`../../HARDWARE_DIAGNOSTICS.md`](../../HARDWARE_DIAGNOSTICS.md#2026-06-15).
- Sur la ligne moteur, le courant varie naturellement → minuteur réarmé en continu → pas de coupure en roulant.
- Appui sur le bouton = réarme le minuteur (inutilisable en autonome).

## Test rapide
LED du switch au moment d'un reboot : **s'éteint** = c'est bien le switch qui coupe ; **reste allumée** = chercher en aval (UBEC, barrel/J48, UVLO batterie).

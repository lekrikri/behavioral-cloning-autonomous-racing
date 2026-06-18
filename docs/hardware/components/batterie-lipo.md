# Batterie LiPo

> Source d'énergie de la voiture. Tête de la chaîne d'alimentation ([montage](../MONTAGE.md)).

## Identité
- **Type** : LiPo
- ⚠️ **Nombre de cellules (S), capacité (mAh), C-rating : à renseigner** (relever sur la batterie).
- Le Flipsky antispark accepte jusqu'à 13S, mais la batterie réelle de la voiture est bien plus basse — à confirmer.
- **Connecteur** : XT60.

## Distribution
Deux dérivations depuis la batterie (cf. [montage](../MONTAGE.md)) :
1. **Ligne moteur** → Flipsky Anti-Spark → carte puissance → VESC.
2. **Ligne Jetson** → UBEC direct (hors antispark) → Jetson.

## À surveiller
- ⚠️ Tension de coupure (UVLO) : à vérifier comme cause possible si reboot avec LED antispark restée allumée.

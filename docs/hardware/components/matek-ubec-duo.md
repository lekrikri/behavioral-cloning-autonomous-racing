# Matek Systems UBEC Duo

> Régulateur 5 V qui alimente la Jetson Nano. Partie du [montage](../MONTAGE.md).

## Identité
- **Modèle** : Matek Systems UBEC Duo
- **Sortie** : 5 V → Jetson Nano (barrel jack, jumper **J48**)
- **Fonction bonus** : monitoring batterie intégré

## Rôle
- Convertit la tension batterie en 5 V régulé pour la Jetson.
- **Doit être branché directement sur la batterie** (XT60), **hors antispark** — sinon l'auto-off 20 min du Flipsky coupe la Jetson au banc. Cf. [`../../HARDWARE_DIAGNOSTICS.md`](../../HARDWARE_DIAGNOSTICS.md#2026-06-15).

## Specs mesurées
- Conso Jetson à l'arrêt via l'UBEC : `POM_5V_IN ≈ 1876 mW` (~**0,38 A**). Conso ridicule → l'UBEC n'est **pas** sous-dimensionné en courant.
- Inrush propre négligeable (faible capacité d'entrée) → **pas besoin d'antispark** sur cette ligne.

## À confirmer
- ⚠️ Calibre exact (5 V / combien d'ampères ?) et version précise du module.

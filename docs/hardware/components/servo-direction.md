# Servo de direction

> Actionne la direction (steering). Partie du [montage](../MONTAGE.md).

## Identité
- ⚠️ **Modèle à renseigner** (relever la réf sur le servo).
- Alimenté et piloté via le [Flipsky FSESC Mini V6.7 Pro](flipsky-fsesc.md) (sortie servo PPM/PWM du VESC).

## Pilotage
- Le modèle ML sort un `steering` ∈ [-1, 1] → converti en consigne servo via `steering_to_servo()`.

## À calibrer
- ⚠️ **Calibration sur voiture réelle obligatoire** : les valeurs de la simulation ≠ réel (butées, centre, sens).

# Moteur brushless Traxxas BLSS 3300

> Moteur de traction. Piloté par le VESC. Partie du [montage](../MONTAGE.md).

## Identité
- **Modèle** : Traxxas BLSS 3300 (brushless sensorless)
- ⚠️ **3300** = KV (tr/min par volt) — à confirmer sur le moteur.

## Pilotage
- Commandé par le [Flipsky FSESC Mini V6.7 Pro](flipsky-fsesc.md) en **courant (couple)**, pas en duty cycle, pour éviter les `FAULT_CODE_ABS_OVER_CURRENT`.
- **Sensorless** : démarrage FOC en boucle ouverte → sensible aux paramètres `openloop_*` du firmware (cf. fiche VESC).

## Précautions
- ⚠️ **Jamais de plein gaz soutenu** au banc : limiter le duty max, surveiller la chaleur du moteur à la main. Tests à `--duty-max 0.10` / `0.15`.

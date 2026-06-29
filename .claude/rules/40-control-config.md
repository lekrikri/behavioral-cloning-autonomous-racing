# Contrôle & configuration

## Paramètres via la config — jamais en dur

- Tout paramètre réglable passe par la **config** (`src/config.py` / `configs/`), pas codé
  en dur dans la logique : vitesse max, duty cycle, mode de perception, ports, seuils…
- Les paramètres de **contrôle voiture** (vitesse max, duty-max) doivent rejoindre la config
  au même titre que les hyperparamètres — pas dispersés dans `inference_realcar.py` / teleop.
- **Justifier tout changement de paramètre** (commit/PR) : pourquoi, effet attendu, mesuré sur quoi.

## Sécurité contrôle

- **Roues en l'air d'abord.** Démarrer bas (`--duty-max` faible) et monter prudemment.
- Vecteur d'action = **`[acceleration, steering]`** (pas l'inverse).

## Contrôle web

- Pilotage (Rien / Manuel / Autonome) depuis l'UI web du hub.
- État par défaut sûr = **moteur coupé + direction recentrée**.

# Procédure complète — preview masque + conduite (hub caméra)

> Comment lancer le système de bout en bout : streaming du masque dans le navigateur,
> réglage des couches anti-artefacts, et conduite **manuelle / autonome** depuis l'interface.
> Réglage fin des seuils : voir [`MASK_TUNING.md`](MASK_TUNING.md).

## Architecture

```
OAK-D ──> src/cam/hub.py ──(/dev/shm, zéro-copie)──┬──> mask/stream.py   (preview + UI navigateur)
          (possède la caméra)                      └──> control/inference_realcar.py --source hub  (autonome)
```
Un seul process ouvre l'OAK-D (le **hub**) ; preview et inférence en sont de simples clients →
preview et conduite autonome **coexistent**. Le hub tourne en permanence comme service système
**`robocar-cam-hub`** (démarré au boot, voir [`SERVICES.md`](SERVICES.md)) ; les clients s'y
connectent par défaut (`--source hub`).

## Prérequis (Jetson)

- `OPENBLAS_CORETYPE=ARMV8` devant toute commande Python (sinon **SIGILL** numpy). Les lanceurs `run-*.sh` le posent déjà.
- Conduite **manuelle** : manette F710 sur `/dev/input/js0`, VESC sur `/dev/ttyACM0`.
- Conduite **autonome** : modèle **`models/v18/best_jetson.onnx`** présent.
  - ⚠️ Ce modèle n'est **pas sur `main`** — il vit sur la branche `fix/inference-fusion`. Le restaurer si absent :
    ```bash
    git checkout origin/fix/inference-fusion -- models/v18/best_jetson.onnx
    ```

## Procédure

### 1. Lancer l'interface de debug
```bash
# Jetson — le hub tourne déjà en service (robocar-cam-hub). UN seul terminal :
cd ~/robocar-Paris-lecrabe
OPENBLAS_CORETYPE=ARMV8 python3 -m src.tools.debug.server   # lit le hub (SHM)
```
Attendre `[debug] http :8088 ...`. Si le hub ne publie pas, l'interface avertit
et indique comment le relancer (`sudo systemctl restart robocar-cam-hub`).

### 2. Ouvrir l'interface depuis le PC
```bash
ssh -L 8088:localhost:8088 robocar      # tunnel (le PC initie → OK même via Tailscale)
# puis http://localhost:8088
```

### 3. Régler le masque
Page **Masque** : sliders des filtres (seuil V, S max, top-hat, morpho, aire mini,
rectilinéarité, depth_tol, CLAHE) + cases masque/rayons. Le faisceau **polaire** est dessiné
depuis le centre voiture ; jugez sur les rayons, pas que le masque. Détails : [`MASK_TUNING.md`](MASK_TUNING.md).

### 4. Conduite — ⚠️ roues en l'air d'abord
Page **Accueil** : choisir un **profil** puis piloter le pipeline conduite :
- **PLAY** → lance l'inférence autonome avec le profil (lit le hub). La direction suit le masque.
- **PAUSE** → coupe le moteur (maintien sûr), le profil reste chargé.
- **STOP** → arrêt moteur + arrêt du pipeline (état par défaut, sûr).

Si l'état ne passe pas `(process actif)`, le lancement a échoué (modèle absent, VESC débranché, etc.) →
l'interface retombe sur `stop` et le terminal logge `play n'a pas démarré (code …)`.

## Variantes

- **Hub à la main** (debug, service arrêté) : `sudo systemctl stop robocar-cam-hub`, lancer
  `python3 -m src.cam.hub` dans un terminal, puis `python3 -m src.tools.debug.server`.

## Dépannage

| Symptôme | Cause / fix |
|---|---|
| `Illegal instruction (core dumped)` | `OPENBLAS_CORETYPE=ARMV8` manquant |
| Preview noire en `--source hub` | le hub n'a pas pris la caméra → un autre process la tenait (streamer device, inférence) |
| `Modèle introuvable : …best_jetson.onnx` | restaurer le modèle (voir Prérequis) |
| `AUTONOME` retombe sur `none` | modèle absent, ou hub non joignable |
| `Connection refused` dans le terminal SSH | normal si le streamer n'est pas encore lancé côté Jetson |

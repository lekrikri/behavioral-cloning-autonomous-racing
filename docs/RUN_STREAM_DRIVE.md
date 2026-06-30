# Procédure complète — preview masque + conduite (hub caméra)

> Comment lancer le système de bout en bout : streaming du masque dans le navigateur,
> réglage des couches anti-artefacts, et conduite **manuelle / autonome** depuis l'interface.
> Réglage fin des seuils : voir [`MASK_TUNING.md`](MASK_TUNING.md).

## Architecture

```
OAK-D ──> camera_hub.py ──(socket locale)──┬──> mask_stream.py   (preview + UI navigateur)
          (possède la caméra)              └──> inference_realcar.py --source hub  (autonome)
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

### 1. Lancer le streamer
```bash
# Jetson — le hub tourne déjà en service (robocar-cam-hub). UN seul terminal :
cd ~/robocar-Paris-lecrabe
OPENBLAS_CORETYPE=ARMV8 python3 -m src.mask.stream   # --source hub est le défaut
```
Attendre `[stream] source = camera_hub :8077`. Si le hub ne répond pas, le streamer avertit
et indique comment le relancer (`sudo systemctl restart robocar-cam-hub`).

### 2. Ouvrir l'interface depuis le PC
```bash
ssh -L 8088:localhost:8088 robocar      # tunnel (le PC initie → OK même via Tailscale)
# puis http://localhost:8088
```

### 3. Régler le masque
Toggles `t` (top-hat), `f` (forme), `c` (temporel), `o`/`p` (ROI), `+`/`-` (seuil V), etc.
Procédure détaillée dans [`MASK_TUNING.md`](MASK_TUNING.md). Jugez sur les **raycasts** (`r`), pas que le masque.

### 4. Conduite — ⚠️ roues en l'air d'abord
Barre **conduite** dans l'interface :
- **MANUEL** → lance le teleop (stick droit = direction, R2 = avance). 
- **AUTONOME** → lance l'inférence (lit le hub, `duty-max=0.20`). La direction suit le masque.
- **RIEN** → arrêt moteur + direction recentrée (état par défaut, sûr).

Si un mode n'apparaît pas `(process actif)`, il a échoué au lancement (manette débranchée, modèle absent, etc.) →
l'interface retombe sur `none` et le terminal logge `'manual'/'auto' n'a pas démarré (code …)`.

## Variantes

- **Réglage seul, caméra en direct** (sans passer par le hub) : `python3 -m src.mask.stream --source device`.
- **Hub à la main** (debug, service arrêté) : `sudo systemctl stop robocar-cam-hub`, lancer
  `python3 -m src.cam.hub` dans un terminal, puis `python3 -m src.mask.stream` (défaut `--source hub`).

## Dépannage

| Symptôme | Cause / fix |
|---|---|
| `Illegal instruction (core dumped)` | `OPENBLAS_CORETYPE=ARMV8` manquant |
| Preview noire en `--source hub` | le hub n'a pas pris la caméra → un autre process la tenait (streamer device, inférence) |
| `Modèle introuvable : …best_jetson.onnx` | restaurer le modèle (voir Prérequis) |
| `AUTONOME` retombe sur `none` | modèle absent, ou hub non joignable |
| `Connection refused` dans le terminal SSH | normal si le streamer n'est pas encore lancé côté Jetson |

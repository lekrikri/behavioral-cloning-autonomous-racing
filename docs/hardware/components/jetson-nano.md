# Jetson Nano 4GB

> Ordinateur embarqué — inférence + contrôle. Partie du [montage](../MONTAGE.md).

## Identité
- **Carte** : NVIDIA Jetson Nano 4GB
- **Module** : P3448-0002
- **L4T** : R32.7.6 — **JetPack 4.6** (Ubuntu 18.04, Python 3.8)
- **GPU** : Maxwell 128 cores
- **Alimentation** : 5 V via barrel jack, jumper **J48** (depuis l'UBEC)

## Accès
- Login : `robocar` / `robocar` (reset 2026-06-12)
- Clé SSH GitHub configurée. Détails : [`../../SSH_ACCESS.md`](../../SSH_ACCESS.md).

## Quirks / pièges connus
- ⚠️ **NE JAMAIS faire `apt upgrade`** (casse le L4T). Backup système dans `data/`.
- ⚠️ **numpy SIGILL** au démarrage → fix : `export OPENBLAS_CORETYPE=ARMV8`.
- venv propre via `virtualenv` (`ensurepip` cassé sur cette image).
- Cible d'inférence : ONNX Runtime GPU ou TensorRT FP16.
- `nvpmodel` en MAXN = conso pic max.

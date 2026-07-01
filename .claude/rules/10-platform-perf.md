# Plateforme & performance

## Cible : Jetson Nano (L4T)

- Stack réel (vérifié 2026-06-29 sur la Jetson) : **L4T R32.7.6 / JetPack 4.6 / Ubuntu 18.04
  / Python 3.6.9 / aarch64 (arm64)**. Toute dépendance doit s'installer **sur ce stack précis**.
- `aarch64` est **nécessaire mais pas suffisant** : un wheel peut être aarch64 et exiger
  Python ≥ 3.8, une glibc récente ou CUDA 11+ → il échoue sur la Nano. **Vérifier l'install
  réellement sur la Jetson**, pas seulement l'archi.
- `OPENBLAS_CORETYPE=ARMV8` devant tout Python (sinon SIGILL numpy sur le Tegra).

## Temps réel (contrainte centrale)

- **Maximiser les frames traitées, ne jamais bloquer** la boucle.
- **Paralléliser** ce qui peut l'être dans les limites de la Jetson : **vidéo+mask /
  contrôle / IA** sur des chemins concurrents.
- **Déporter sur la caméra (OAK-D)** un maximum de charge (encodage, depth, pré-traitement)
  pour soulager la Jetson — en équilibrant : ne pas saturer le VPU au détriment du débit.
- **Sobriété** : préférer le plus léger. Toute allocation/copie/thread dans le hot path
  doit se justifier.

## Réflexe avant d'ajouter du calcul

Est-ce déportable sur la caméra ? parallélisable ? bloquant ?

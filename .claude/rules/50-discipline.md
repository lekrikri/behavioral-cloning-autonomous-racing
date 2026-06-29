# Discipline d'ingénierie

## Qualité (ne pas tâtonner)

- Avant de déclarer « fait » : lancer **`/check`** (relecture du diff) et, pour le réel,
  **vérifier sur la Jetson/matériel**. Pas d'affirmation non vérifiée.

## Documentation factorisée & atomique

- Une info = **un seul endroit**. Documenter une modif en l'ajoutant au doc atomique du
  domaine, **sans dupliquer**. Préférer un lien à un copier-coller.

## Propriété des fichiers (repo partagé)

- Avant d'éditer un module, **vérifier l'auteur** (`git log -- <fichier>`). Ne pas réécrire
  le code/doc d'un coéquipier sans coordination.
- Repère actuel (toujours confirmer via `git log`) :
  - **Mathieu** : perception/masque (`mask_*`, `visual_rays.py`, `camera_hub.py`), docs
    `RUN_STREAM_DRIVE` / `MASK_TUNING` / `CONTROL_STACK` / `HARDWARE_DIAGNOSTICS`.
  - **Christophe** : modèle/entraînement (`model.py`, `train*.py`), `inference_realcar.py`,
    `camera_stream*`, `README`, `CONTRIBUTING`.
  - **Léandre** : annotation dataset / masques.

## Ces règles sont protégées

- `CLAUDE.md` et `.claude/**` sont **à édition explicitement confirmée** : le hook PreToolUse
  renvoie `ask` → jamais de modif silencieuse. Ils sont aussi **gelés localement**
  (`git update-index --skip-worktree`).
- Les modifier est un **acte délibéré** : passer par une **PR justifiée**.
- Pré-autoriser une série d'éditions délibérées (sauter la confirmation) : `ALLOW_HARNESS_EDIT=1`.
  Pour committer un changement gelé : `git update-index --no-skip-worktree <fichier>`.
- **Pas de modif locale de ces règles hors PR justifiée.**

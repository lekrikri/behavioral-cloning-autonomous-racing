---
description: Contrôle qualité du diff courant (Jetson / temps-réel aware) avant de livrer
---
Effectue un contrôle qualité avant de déclarer « fait ». Rapporte, ne corrige pas en masse sans accord.

1. **Relecture du `git diff`** sous l'angle de ce projet :
   - **Temps réel** : rien de bloquant dans la boucle ? allocations/copies inutiles dans le hot path ?
   - **Jetson/L4T** : dépendance compatible aarch64 ? charge déportable sur la caméra ?
   - **Config** : paramètres via la config (pas en dur) ? changement justifié ?
   - **Perception** : la voie raycasts polaires reste-t-elle disponible et intacte ?
   - **Propriété** : édite-t-on le module d'un coéquipier sans raison (`git log`) ?
2. **Verdict** : liste courte — bloquants vs nits — et si c'est prêt à committer.

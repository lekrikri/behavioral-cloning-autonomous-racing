# Robocar — Règles projet (harness Claude Code)

> Repo partagé (Christophe, Léandre, Mathieu). Ce fichier est chargé automatiquement
> par Claude Code (terminal **et** extension IDE). Il fixe le cap et renvoie vers des
> règles atomiques — **lis la règle du domaine concerné avant d'agir.**

## Non-négociables (détail dans `.claude/rules/`)

- **Cible = Jetson Nano / L4T.** Temps réel, non-bloquant, sobre. → [10-platform-perf](.claude/rules/10-platform-perf.md)
- **Raycasts polaires, parité simulateur** (représentation principale). → [20-perception](.claude/rules/20-perception.md)
- **Flux vidéo via le hub partagé** (service au boot), accès `ssh -L`. → [30-streaming](.claude/rules/30-streaming.md)
- **Paramètres via la config**, jamais en dur ; tout changement justifié. → [40-control-config](.claude/rules/40-control-config.md)
- **Intelligence swappable**, modules paramétrables, contrôle depuis l'UI web. → [00-direction](.claude/rules/00-direction.md)
- **Docs factorisées/atomiques, qualité vérifiée, propriété des fichiers respectée.** → [50-discipline](.claude/rules/50-discipline.md)

## Avant de livrer

Lance `/check` (relecture du diff). Ne déclare pas « fait » sans avoir vérifié — sur la Jetson/matériel quand c'est du réel.

## Ces règles sont protégées

Toute édition de `CLAUDE.md` ou `.claude/**` **exige une confirmation explicite** (hook → `ask`) ; les fichiers sont aussi gelés localement. Les modifier = acte délibéré via PR justifiée → [50-discipline](.claude/rules/50-discipline.md).

> Onboarding humain : [CONTRIBUTING.md](CONTRIBUTING.md).

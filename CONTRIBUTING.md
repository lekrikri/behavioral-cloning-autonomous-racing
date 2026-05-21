# Contributing — G-CAR-000 Robocar Racing

Bienvenue sur le projet ! Ce repo est partagé entre **Christophe**, **Léandre** et **Mathieu**.

## Premiers pas

1. Lire l'onboarding complet : [`docs/ONBOARDING_LEANDRE.md`](docs/ONBOARDING_LEANDRE.md)
2. Installer les dépendances : `pip install -r requirements.txt`
3. Configuration simulateur : [`configs/config.json`](configs/config.json)

## Structure du repo

```
behavioral-cloning-autonomous-racing/
├── src/                    # Code principal (model, training, inference)
├── tests/                  # Scripts de test hardware/VESC
├── docs/                   # Documentation technique
├── configs/                # Fichiers de configuration
├── data/                   # Données collectées (gitignorées — générer en local)
├── models/                 # Modèles entraînés
│   ├── v18/best.onnx       # Modèle actif — record 24s/tour
│   └── ray_stats.json      # Stats Z-score pour normalisation
├── notebooks/              # EDA et exploration
├── requirements.txt        # Dépendances Python (simulation)
├── requirements_jetson.txt # Dépendances Python (Jetson Nano)
├── deploy_to_jetson.sh     # Script de déploiement SSH → Jetson
└── setup_jetson.sh         # Setup initial du Jetson Nano
```

## Workflow git

```bash
# Créer une branche pour ta feature
git checkout -b feat/mon-amelioration

# Commiter
git add src/mon_fichier.py
git commit -m "feat: description courte"

# Push + PR
git push origin feat/mon-amelioration
```

## Conventions de commit

| Préfixe | Usage |
|---------|-------|
| `feat:` | Nouvelle fonctionnalité |
| `fix:` | Correction de bug |
| `refactor:` | Refactoring sans changement de comportement |
| `docs:` | Documentation |
| `chore:` | Scripts, config, dépendances |

## Lancer une session d'entraînement

```bash
# Collecter des données (Unity doit tourner)
python src/data_collector.py collect --output data/run_new.csv

# Entraîner
python src/train.py --data data/ --arch cnn --epochs 100 --loss huber

# Évaluer
python src/evaluate.py --model models/v18/best.pth --data data/
```

## Questions

Contacter Christophe : ganou.christophe@gmail.com

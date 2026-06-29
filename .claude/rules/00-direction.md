# Direction projet

Behavioral cloning + sim-to-real : la voiture perçoit la piste en **raycasts** et prédit
direction + accélération. Objectif : conduite autonome fluide sur la vraie voiture
(Jetson Nano + OAK-D Lite), en parité avec le simulateur.

## Principes d'architecture

- **Intelligence swappable** : le module de décision (modèle IA *ou* algorithme classique)
  est interchangeable derrière une interface commune. On remplace le « cerveau » sans
  toucher au reste (perception, contrôle, flux).
- **Modules paramétrables** : chaque module expose ses paramètres (vitesse max, mode de
  perception, seuils…) — pas de constante magique enfouie. Voir [40-control-config](40-control-config.md).
- **Contrôle depuis l'UI web** : manuel et autonome pilotables depuis l'interface navigateur
  servie par le hub. Voir [30-streaming](30-streaming.md).

## Ce qui ne doit pas dériver

- La représentation **raycasts polaires** reste la voie principale. Voir [20-perception](20-perception.md).
- La cible reste la **Jetson Nano** sous contrainte temps réel. Voir [10-platform-perf](10-platform-perf.md).

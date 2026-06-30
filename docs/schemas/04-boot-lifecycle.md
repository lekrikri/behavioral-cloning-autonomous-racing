# Boot & cycle de vie

Au boot, systemd lance le hub puis le core. Le core charge la config et reste **inerte** :
**aucun worker de conduite** tant qu'il n'y a pas de trigger (manette ou profil lancé).

```mermaid
sequenceDiagram
    participant SYS as systemd
    participant HUB as cam-hub
    participant CORE as Core
    participant W as Workers

    SYS->>HUB: start robocar-cam-hub.service
    SYS->>CORE: start robocar-core.service (Requires hub)
    CORE->>CORE: charge config + sélectionne le profil par défaut (sans le lancer)
    Note over CORE,W: INERTE — aucun worker de conduite
    Note over CORE,W: trigger requis : manette allumée OU profil lancé (terminal / UI)
    Note over CORE,W: stream UI seulement à la 1re connexion web
```

# États & prise de main

Au boot : **Inerte** (aucun worker de conduite). On en sort par un trigger explicite (profil
lancé → Auto, ou manette → Manuel). Si l'auto tourne, allumer la manette **arme** le manuel
(passif) ; seul un clic explicite **prend la main** et coupe l'auto.

```mermaid
stateDiagram-v2
    [*] --> Inerte: boot (aucun worker de conduite)

    Inerte --> Auto: profil lancé (terminal / UI)
    Inerte --> ManuelActif: manette allumée (rien d'autre ne tourne)

    Auto --> ManuelArme: manette allumée (manuel PASSIF, auto continue)
    ManuelArme --> Auto: manette éteinte
    ManuelArme --> ManuelActif: clic « prendre la main » (coupe l'auto)
    ManuelActif --> Auto: clic « rendre la main » (reprend le profil)

    Auto --> Inerte: profil arrêté
    ManuelActif --> Inerte: manette éteinte (si aucun profil lancé)

    note right of Inerte
        prête mais ne conduit pas
    end note
    note right of ManuelArme
        worker manuel lancé mais passif
    end note
```

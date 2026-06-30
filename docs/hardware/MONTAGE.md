# Montage complet — Voiture réelle (Jetson Nano)

> Vue d'ensemble du câblage et de la topologie hardware de la voiture.
> Fiche détaillée par composant dans [`components/`](components/).
>
> Docs liées :
> - [`../HARDWARE_REALCAR.md`](../HARDWARE_REALCAR.md) — référence software/ML (auteur : Christophe), **ne pas dupliquer ici**
> - [`../HARDWARE_DIAGNOSTICS.md`](../HARDWARE_DIAGNOSTICS.md) — journal des pannes matérielles
> - [`../CONTROL_STACK.md`](../CONTROL_STACK.md) — communication Jetson ↔ VESC

---

## Chaîne d'alimentation — câblage ACTUEL (= le problème)

```
Batterie LiPo
   │
   └──[XT60]──► [Flipsky Anti-Spark Switch Smart] (13S/60V, 60A, 800A crête)
                     │   interrupteur principal + anti-étincelle
                     ▼
                la sortie se sépare vers DEUX charges :
                     ├──► [Carte puissance + radiateur] ──► [Flipsky FSESC Mini V6.7 Pro] (VESC)
                     │                                          ├──► Moteur Traxxas BLSS 3300
                     │                                          └──► Servo de direction
                     └──► [Matek UBEC Duo] ──5V──► Jetson Nano (barrel jack, jumper J48)
```

**L'antispark alimente le VESC ET l'UBEC.** Quand son minuteur 20 min se déclenche (à l'arrêt),
il coupe sa sortie → **VESC + Jetson tombent ensemble**. L'UBEC ne s'éteint pas de lui-même, il
perd simplement son alimentation d'entrée. Analyse :
[`../HARDWARE_DIAGNOSTICS.md`](../HARDWARE_DIAGNOSTICS.md#2026-06-15).

## Chaîne d'alimentation — câblage CIBLE (= le correctif)

```
Batterie LiPo
   │
   ├──[XT60]──► [Flipsky Anti-Spark Switch Smart] ──► carte puissance ──► VESC ──► moteur + servo
   │                 (réservé à la ligne moteur)
   │
   └──[XT60]──► [Matek UBEC Duo] ──5V──► Jetson Nano
                     branché DIRECTEMENT sur la batterie, en amont de l'antispark
```

### Pourquoi l'UBEC ne passe PAS par l'antispark

Le Flipsky Anti-Spark s'éteint seul au bout de **20 min** si sa tension de sortie ne varie
pas d'au moins **500 mV / 3 s**. À l'arrêt, le VESC ne tire quasiment rien (moteur immobile) ;
la seule charge est la Jetson, qui consomme un courant **constant** à travers un UBEC régulé
→ sortie antispark parfaitement stable → minuteur jamais réarmé → coupure à 20 min. En roulant,
les appels de courant moteur font osciller la tension → minuteur réarmé en continu → jamais de
coupure. C'est pour ça que le reboot n'arrivait **qu'au banc**. Analyse complète :
[`../HARDWARE_DIAGNOSTICS.md`](../HARDWARE_DIAGNOSTICS.md#2026-06-15).

**Correctif :** alimenter l'UBEC en parallèle, directement sur la batterie. Le gros appel de
courant (inrush) à l'allumage vient des condensateurs du **VESC**, pas de l'UBEC (convertisseur
5 V à faible capacité d'entrée). Le Flipsky, resté sur la ligne moteur, couvre donc déjà cet
inrush. **La ligne UBEC n'a pas besoin de son propre antispark : du XT60 standard suffit**
(il n'existe de toute façon pas de connecteur antispark passif en XT60, contrairement au XT90-S).

> **Statut au 2026-06-18 :** recâblage UBEC-hors-antispark **à faire** (voir checklist du journal).

---

## Topologie USB (Jetson Nano)

| Port | Périphérique | Usage |
|------|--------------|-------|
| USB3 | Luxonis OAK-D Lite (p/n a00483) | Caméra RGB + depth stéréo — via **hub USB à injection alimenté par l'UBEC** (5V), pas directement sur le bus Jetson. Corrige le brownout `X_LINK_ERROR`, cf. [`../HARDWARE_DIAGNOSTICS.md`](../HARDWARE_DIAGNOSTICS.md) (2026-06-18 / résolu 2026-06-24) |
| USB  | Flipsky FSESC Mini V6.7 Pro | Contrôle moteur/servo + télémétrie (`/dev/ttyACM0`) |
| USB  | Dongle TP-Link | WiFi |
| USB  | Dongle Logitech (manette F710) | Pilotage manuel / collecte de données |

---

## Connecteurs

- **Connecteurs de puissance : XT60** (batterie, antispark, UBEC, VESC).
- Pas de connecteur antispark passif côté UBEC (cf. section ci-dessus).

---

## Inventaire des composants

| Composant | Fiche |
|-----------|-------|
| Batterie LiPo | [batterie-lipo.md](components/batterie-lipo.md) |
| Flipsky Anti-Spark Switch Smart 13S | [flipsky-antispark.md](components/flipsky-antispark.md) |
| Carte puissance centrale + radiateur | [carte-puissance.md](components/carte-puissance.md) |
| Flipsky FSESC Mini V6.7 Pro (VESC) | [flipsky-fsesc.md](components/flipsky-fsesc.md) |
| Matek UBEC Duo | [matek-ubec-duo.md](components/matek-ubec-duo.md) |
| Jetson Nano 4GB | [jetson-nano.md](components/jetson-nano.md) |
| Luxonis OAK-D Lite | [oak-d-lite.md](components/oak-d-lite.md) |
| Moteur Traxxas BLSS 3300 | [moteur-traxxas.md](components/moteur-traxxas.md) |
| Servo de direction | [servo-direction.md](components/servo-direction.md) |
| Manette Logitech F710 | [manette-f710.md](components/manette-f710.md) |

# Diagnostics Hardware — Journal

> Journal vivant des problèmes matériels rencontrés sur la voiture réelle (Jetson Nano)
> et de leur résolution. On complète ce document au fur et à mesure de l'avancement.
> Voir aussi : [`hardware/MONTAGE.md`](hardware/MONTAGE.md) (montage + fiches par composant),
> [`HARDWARE_REALCAR.md`](HARDWARE_REALCAR.md) (référence matérielle),
> [`VESC_DIAGNOSTIC_RESUME.md`](VESC_DIAGNOSTIC_RESUME.md).

## Chaîne d'alimentation (rappel)

```
Batterie LiPo
      ↓
[Flipsky Anti-Spark Switch Smart 13S]   ← interrupteur principal (anti-étincelle + MOSFET piloté)
      ↓
[Carte puissance centrale + radiateur]
      ├──→ [Matek UBEC Duo] ── 5V ──→ Jetson Nano (barrel jack, J48 jumper)
      └──→ [Flipsky FSESC Mini V6.7 Pro] ──→ moteur + servo
```

---

## 2026-06-15 — La voiture se coupe et reboote toute seule (~20 min, à l'arrêt) ✅ RÉSOLU

### Symptôme
La voiture s'éteint et redémarre seule, à intervalle régulier (~15–20 min), **uniquement à
l'arrêt / sur l'établi**, jamais observé en roulant. Sessions SSH coupées net.

### Cause racine
Le **Flipsky Anti-Spark Switch Smart (13S / 60 V, 60 A continu, 800 A crête)**, câblé entre la batterie et l'UBEC,
possède un **minuteur d'auto-extinction de 20 minutes** non désactivable. Citation de la
documentation Flipsky :

> *« Will turn off automatically after 20 minutes. The output voltage must vary by at least
> 500mV within any three second interval to reset the turn-off timer. »*

Le minuteur ne se réarme que si la tension de sortie du switch **varie d'au moins 500 mV en
3 s**. Or :

- **À l'arrêt** : la Jetson tire **~0,38 A constant** (`POM_5V_IN ≈ 1876 mW`) à travers un UBEC
  régulé → tension de sortie parfaitement stable → minuteur **jamais réarmé** → coupure à 20 min.
- **En roulant** : les moteurs créent de gros appels de courant → la tension oscille bien
  au-delà de 500 mV en permanence → minuteur réarmé en continu → **ne coupe jamais**.

C'est pour cette raison que le problème ne se manifeste qu'au banc / en développement.

### Preuves (relevées sur la Jetson)
| Vérification | Résultat | Interprétation |
|---|---|---|
| `last` | sessions en `crash`, jamais `shutdown` | coupure brutale, pas d'arrêt logiciel |
| `dmesg` | `last reset is due to power on reset`, `PMC reset status reg: 0x0` | reset à froid = **perte d'alimentation** (ni watchdog, ni panic, ni soft-reboot) |
| `thermal_zone*` | 28–50 °C | **pas** une coupure thermique |
| `tegrastats` | `POM_5V_IN ≈ 1876 mW` (~0,38 A) | conso ridicule → **pas** une saturation courant de l'UBEC |
| `nvpmodel -q` | MAXN | conso pic max (aggravant, mais pas la cause) |

Hypothèses **écartées par les données** : extinction logicielle, watchdog/kernel panic,
surchauffe, UBEC sous-dimensionné en courant.

### Correctif
L'auto-off de 20 min est une sécurité firmware **figée** (pas de réglage pour la désactiver).
Solution = **ne pas faire passer l'alimentation de la Jetson par le Flipsky** :

1. **★ Recommandé** — alimenter l'**UBEC directement sur la batterie**, en **XT60 standard**.
   Réserver le Flipsky à la **ligne moteur/ESC**, où le courant varie naturellement → son
   minuteur ne se déclenche jamais. Pas besoin d'antispark sur la ligne UBEC : l'inrush vient
   des condensateurs du VESC (déjà couverts par le Flipsky), pas de l'UBEC dont la capacité
   d'entrée est minime. Un connecteur antispark passif n'existe de toute façon pas en XT60
   (le XT90-S/AS oui, mais la voiture est câblée en XT60).
2. Palliatif de validation : maintenir une charge qui perturbe le rail ≥ 500 mV / 3 s (sale).
3. Appui sur le bouton du switch = réarme le minuteur (inutilisable en autonome).

### Test de confirmation visuel (sans outil)
Laisser tourner à vide et observer la **LED du Flipsky** au moment du reset :
- LED qui **s'éteint** → c'est bien le switch qui coupe (confirmé).
- LED qui **reste allumée** → chercher en aval (UBEC, barrel/J48, UVLO batterie).

### Statut
- [x] Cause identifiée et documentée
- [ ] Recâblage UBEC hors antispark (à faire — partie matériel)
- [ ] Validation : > 20 min à vide au banc sans reboot

### Références
- Flipsky Anti-Spark Switch Smart — <https://flipsky.net/collections/anti-spark-switch>
- Comportement auto-off (fiche distributeur) —
  <https://dronespark.myshopify.com/products/flipsky-antispark-switch-smart-enhanced-200a-for-electric-skateboard-ebike-scooter-robots>

---

## 2026-06-18 — OAK-D Lite crashe en streaming (X_LINK_ERROR) 🔬 EN COURS

### Symptôme
Le stream caméra s'interrompt toutes les 10-60 secondes avec :
```
[host] [warning] Device crashed, but no crash dump could be extracted.
RuntimeError: X_LINK_ERROR — Couldn't read data from stream: 'encoded'
```
La reconnexion automatique reprend le stream, mais VLC voit la connexion TCP couper.

### Cause racine
**Brownout USB** : le Jetson Nano fournit 900mA max sur son hub USB interne,
partagé entre 4 périphériques :

```
OAK-D Lite    → ~500-900mA (encodage VPU Myriad X)
FSESC VESC    → ~100mA
TP-Link WiFi  → ~400mA
Clavier Logi  → ~100mA  ← débranché 2026-06-18
```

Pic de consommation OAK-D au démarrage de l'encodage > 900mA disponibles
→ tension bus USB chute → brownout → crash OAK-D.

### Preuves
- Crashs systématiques quelques secondes après début d'encodage
- Débrancher le clavier réduit légèrement la fréquence
- `Device crashed, but no crash dump` = coupure brutale (pas arrêt logiciel)
- Jamais de crash quand OAK-D est en mode preview seul (moins de charge VPU)

### Correctifs

**Fix définitif (non encore appliqué)** — Hub USB alimenté :
- Brancher l'OAK-D sur un hub USB alimenté (5V ≥ 2A) → alimentation indépendante du Jetson
- Coût : ~15-20€ (Anker, UGREEN, etc.)

**Fix software appliqué** — Reconnexion auto + usb2Mode :
- `usb2Mode=True` dans le constructeur `dai.Device(pipeline, True)` → réduit pic courant
- Reconnexion automatique avec backoff exponentiel (5s, 10s... max 30s)
- Config validée stable : 640x360 @ 15fps H.264 2000kbps

**À éviter (aggrave les crashs)** :
- `pipeline.setXLinkChunkSize(0)` → plus de petits transferts USB = plus de ripple
- `xout.setFpsLimit(N)` → interruptions USB plus fréquentes
- MJPEG codec → bande passante USB x10-50 vs H.264

### Statut
- [x] Cause identifiée et documentée
- [x] Fix software : reconnexion auto + usb2Mode=True
- [x] Débranché le clavier (libère ~100mA)
- [ ] **Hub USB alimenté à acheter et câbler** (fix définitif)
- [ ] Valider stabilité > 5 min après ajout hub USB

### Références
- [`docs/OAK_CAMERA_STREAM.md`](OAK_CAMERA_STREAM.md) — doc complète streaming
- [Luxonis — USB Deployment Guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide)
- [depthai 2.x usb2Mode API](https://oak-web.readthedocs.io/en/stable/components/device/)

---

<!--
Modèle pour une nouvelle entrée :

## AAAA-MM-JJ — <titre court du symptôme> [✅ RÉSOLU | 🔬 EN COURS | ❌ BLOQUÉ]

### Symptôme
### Cause racine
### Preuves
### Correctif
### Statut
### Références
-->

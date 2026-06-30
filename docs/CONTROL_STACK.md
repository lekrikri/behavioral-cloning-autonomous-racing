# Control Stack — Jetson Nano (notre carte)

> Comment on communique avec le contrôleur moteur (Flipsky FSESC) depuis la Jetson.
> Couvre l'environnement Python (venv) et le protocole VESC bas niveau.
> Carte cible : **Jetson Nano, L4T R32.7.6, Python 3.6.9** (≠ la carte de l'équipe
> précédente décrite dans `STATUS_DEPLOIEMENT.md`, qui était en JetPack 4.6.1 / Python 3.8).

---

## 1. Les couches de communication

```
NOTRE code applicatif        "tourne à -0.3, accélère à 0.5"
      │
src/control/vesc_interface.py   ◄── le plus bas niveau QU'ON écrit
   • framing  0x02 | len | payload | crc | 0x03
   • CRC-16/XMODEM · IDs de commande · scaling (Ampères, position servo)
      │  octets bruts
pyserial (serial.Serial)     ← lib générique, on l'utilise
driver noyau cdc_acm → /dev/ttyACM0   ← port série virtuel
bus USB
firmware STM32 sur le FSESC  ← l'autre bout (code Flipsky/VESC)
```

`vesc_interface.py` est la **frontière de traduction** : au-dessus on raisonne en
`steering`/`accel` (sens physique), en dessous tout est en octets. C'est le code le plus
bas niveau spécifique au projet — en dessous il n'y a plus que du générique.

Le FSESC est vu sur **`/dev/ttyACM0`** (STM32F407, USB id `0483:5740`).

---

## 2. Environnement Python — venv isolé

L'équipe précédente (Montpellier) avait tout installé en `~/.local` (Python système 3.6).
On repart **propre** avec un venv dédié dans le worktree (`.venv/`, déjà gitignoré).

### Création (sans sudo)
`python3 -m venv` est inutilisable ici : **`ensurepip` est cassé** sur cette image Ubuntu 18.04
→ le venv se créerait sans pip. Parade : `virtualenv` (qui embarque pip) installé en `--user`.

```bash
python3 -m pip install --user virtualenv
cd ~/robocar-Paris-lecrabe
python3 -m virtualenv .venv
.venv/bin/python -m pip install numpy==1.19.5 pyserial==3.5   # pas de pyvesc — voir §4
```

### ⚠️ Piège numpy : `Illegal instruction (core dumped)` (SIGILL)
Sur le Nano (Cortex-A57), `import numpy` (wheel pip 1.19.5) **crashe** : son OpenBLAS
auto-détecte un coretype ARM trop récent. **Fix** :

```bash
export OPENBLAS_CORETYPE=ARMV8
```

Cette variable est ajoutée à la fin de `.venv/bin/activate` → automatique quand on active
le venv. À reposer dans tout script de lancement (`systemd`, cron, lanceur).

| Dépendance | Version | Note |
|---|---|---|
| `pyserial` | 3.5 | communication série |
| `numpy` | 1.19.5 | dernier wheel cp36 aarch64 — nécessite `OPENBLAS_CORETYPE=ARMV8` |
| ~~`pyvesc`~~ | — | **abandonné**, on a notre propre codec (§4) |

---

## 3. Protocole VESC — trame courte (short packet)

Le VESC communique en trames binaires. Pour un payload < 256 octets :

```
0x02 | len(1 octet) | payload(len octets) | crc_hi | crc_lo | 0x03
```

- `len` = longueur du payload.
- `crc` = **CRC-16/XMODEM** (poly `0x1021`, init `0x0000`) calculé sur le **payload seul**.
- `payload[0]` = ID de commande, suivi des arguments en **big-endian**.

### IDs de commande utilisés
| Commande | ID | Argument | Scaling | Type |
|---|---|---|---|---|
| `COMM_SET_DUTY` | 5 | duty cycle | ×100000 | int32 |
| `COMM_SET_CURRENT` | 6 | courant (A) | ×1000 (mA) | int32 |
| `COMM_SET_RPM` | 8 | eRPM | ×1 | int32 |
| `COMM_SET_SERVO_POS` | 12 | position [0..1] | ×1000 | int16 |
| `COMM_GET_VALUES` | 4 | (requête) | — | — |
| `COMM_ALIVE` | 30 | heartbeat | — | — |

> ⚠️ **Piège vécu** : l'ID du servo est **12**, pas 11. L'ID **11 = `COMM_SET_DETECT`** (un
> no-op silencieux pour le servo). Le code de l'équipe précédente utilisait 11 → sur notre
> firmware **6.05** le servo ne bougeait jamais. Les IDs ont évolué entre versions : toujours
> les vérifier contre le `datatypes.h` de la version de firmware réellement flashée.

### Heartbeat obligatoire
Le VESC a un **watchdog** : sans trame récente il coupe le moteur. On envoie `COMM_ALIVE`
toutes les **300 ms** (thread daemon dans `VESCInterface`). Régler le timeout côté firmware :
`VESC Tool → App → General → Timeout = 200 ms`.

---

## 4. Pourquoi un codec maison (pas pyvesc)

`★ Le piège pyvesc/CRC :`
- Le `pyvesc` de **PyPI** (1.0.5) dépend de `PyCRC` **sans le déclarer** (bug de packaging) →
  `ModuleNotFoundError: No module named 'PyCRC'` à l'import dans un venv propre.
- `PyCRC.CRCCCITT()` calcule par défaut un CRC en **init=0xFFFF** (variante « CCITT-FALSE »),
  **pas** le XMODEM (init=0x0000) attendu par le VESC → toutes les trames rejetées. D'où le
  patch `CRCCCITT("XModem")` qu'on voit dans `STATUS_DEPLOIEMENT.md`.
- Le `pyvesc` qui traînait en `~/.local` (Montpellier) était une **build différente** (basée
  sur `crccheck`), avec une API encore différente (classes de messages non exposées au
  top-level). Bref : « pyvesc 1.0.5 » = plusieurs codebases incompatibles selon la source.

→ Le protocole étant trivial, on l'implémente nous-mêmes dans `src/control/vesc_interface.py` :
**zéro dépendance VESC**, juste `pyserial`. Portable Python 3.6 ↔ 3.8, pas de patch.

### Validation du CRC (test sans matériel)
La trame `COMM_ALIVE` de référence est connue : `02 01 1e f3 ff 03`. Le `f3 ff` au milieu
est `crc16_xmodem([0x1e]) = 0xF3FF`. Si notre fonction reproduit `0xF3FF`, le CRC est correct
**par construction** :

```python
from vesc_interface import crc16_xmodem
assert crc16_xmodem(bytes([0x1e])) == 0xF3FF   # COMM_ALIVE
```

---

## 5. CRC-16/XMODEM — c'est quoi

Un **CRC** (Cyclic Redundancy Check) est un détecteur d'erreurs (pas du chiffrement) : on fait
passer les octets dans une division polynomiale binaire, le reste 16 bits sert d'empreinte. Le
récepteur recalcule ; si ça diffère → trame corrompue, jetée.

« XMODEM » désigne une combinaison précise de paramètres (`poly=0x1021, init=0x0000`, pas de
réflexion), héritée du vieux protocole de transfert de fichiers du même nom (1977). La variante
voisine « CCITT-FALSE » utilise le **même polynôme mais init=0xFFFF** — un seul paramètre qui
change, et tous les CRC deviennent faux. C'est exactement ce qui piégeait l'ancien pyvesc.

---

## 6. Moteur — Traxxas BL-2s 3300

Le moteur de traction est un **Traxxas BL-2s 3300** (réf. TRA3384) :

| Caractéristique | Valeur |
|---|---|
| Type | brushless **sensorless** |
| kV | **3300** (rpm par volt) |
| Tension prévue | **2S uniquement** (≈ 7,4–8,4 V) selon Traxxas |
| Bobinage / can | 16 AWG / 37 mm, ventilateur intégré |

### « Faut-il vraiment monter autant en ampères ? » → non, 5 A c'est très peu
Ce moteur encaisse des **dizaines d'ampères** en continu. Le comportement « tourne très
lentement, par à-coups, même à 5 A » observé au banc n'est **pas** un problème d'ampérage. Causes :

1. **`set_current` pilote le COUPLE, pas la vitesse.** Sur une roue libre (sans charge), un
   couple modéré accélère lentement ; nos impulsions courtes ne laissent pas monter le régime.
   Pour faire tourner *vite* au banc → piloter en **duty cycle** (tension), pas en courant.
2. **Sensorless à basse vitesse = saccadé** : sans capteur de position, la commutation est faite
   à l'aveugle sous un certain régime (zone « openloop ») → à-coups. **Sous charge** (au sol),
   l'amorçage est généralement plus net.

### Ce que dit (et ne dit pas) la fiche Traxxas
La page produit officielle donne **kV=3300** et « sensorless » mais **aucune valeur d'ampérage**
ni de tension/cellules (c'est dans le PDF d'install). Le « 2S » vient du **système BL-2s**
(moteur + ESC Traxxas 2S). Il n'y a donc **pas de "courant max" officiel** pour le moteur nu :
la **limite de courant est fixée par NOTRE config VESC** (l'équipe précédente détectait ~8 A).

### ⚠️ Survolt : moteur 2S alimenté en 4S
La batterie lit **16,0 V (4S)** alors que le moteur est un **système 2S**. Les enroulements
craignent le **courant** (chauffe), pas directement la tension — mais 4S double la **vitesse** à
duty donné. À plein duty : `3300 kV × 16 V ≈ 52 800 rpm` à vide → explique la VMax ~90 km/h,
**mais** ~2× le régime nominal → risques de **survitesse** (roulements, aimants) et surtout
**thermiques** (pertes fer ↑ → chauffe → **démagnétisation** permanente).
⚠️ **Aggravant** : `temp_motor` lit du bruit → **pas de sonde température moteur** → le VESC ne
protège **que ses FET**, pas le moteur. **Aucun filet thermique côté moteur.** Règles : limiter le
duty max, **jamais de plein gaz soutenu**, surveiller la chaleur du moteur à la main. À nos
niveaux de test (duty 0.08, ≤ 3 A) → aucun risque.

### Duty (tension/vitesse) vs Courant (couple) — mesuré au banc (roue libre)
| Mode | Commande | rpm relevé | Comportement |
|---|---|---|---|
| **DUTY** | `set_duty(0.08)` | ~680 erpm **stable** | régime régulier, lisse |
| **COURANT** | `set_current(3 A)` | 9→424→39→301 **erratique** | à-coups, observateur perdu |

Leçon : à basse vitesse, un sensorless est lisse en **duty** (on impose la tension) mais saccadé
en **courant** (couple faible + observateur qui ne s'accroche pas). Le courant ne devient propre
qu'une fois lancé ou **sous charge**. Pour l'inférence/teleop on pilotera quand même en courant
(c'est le couple/traction qui compte), en acceptant cette rugosité de démarrage.

### ⚠️ Sens de rotation inversé
Sous courant **positif** (sens « avant » par convention), les roues tournent **en arrière**
(confirmé visuellement + `rpm` négatifs lus sous courant positif). Il faut **inverser le sens
moteur** : flag `invert_motor` côté logiciel (négation du courant), ou échange de 2 phases côté
câblage. Sans ça, `accel > 0` ferait **reculer** la voiture.

### Calibration validée (2026-06-16)
| Paramètre | Valeur | Note |
|---|---|---|
| `servo_center` | 0.50 | roues droites au centre |
| `servo_range` | 0.40 | extrêmes 0.10/0.90 atteignent la butée sans forcer |
| `invert_motor` | **True** | courant positif = avant (corrige le sens inversé) |
| seuil démarrage | ~1 A | en dessous (0.6 A) le moteur ne bouge pas |

---

## Références
- Traxxas BL-2s 3300 (TRA3384) — <https://traxxas.com/products/parts/3384>
- VESC firmware (datatypes / `commands.c`) — <https://github.com/vedderb/bldc>
- CRC-16/XMODEM — poly 0x1021, init 0x0000 (RevEng catalogue)

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
src/vesc_interface.py   ◄── le plus bas niveau QU'ON écrit
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
| `COMM_SET_SERVO_POS` | 11 | position [0..1] | ×1000 | int16 |
| `COMM_GET_VALUES` | 4 | (requête) | — | — |
| `COMM_ALIVE` | 30 | heartbeat | — | — |

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

→ Le protocole étant trivial, on l'implémente nous-mêmes dans `src/vesc_interface.py` :
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

## Références
- VESC firmware (datatypes / `commands.c`) — <https://github.com/vedderb/bldc>
- CRC-16/XMODEM — poly 0x1021, init 0x0000 (RevEng catalogue)

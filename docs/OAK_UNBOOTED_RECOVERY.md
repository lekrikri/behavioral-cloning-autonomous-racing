# OAK-D Lite — Récupération X_LINK_UNBOOTED (depthai 2.26, Jetson Nano)

## Symptôme

Après un crash ou un mauvais paramètre (`usb2Mode=False` sur un bus USB 2.0), l'OAK-D reste bloqué :

```
[depthai] [warning] skipping X_LINK_UNBOOTED device having name "1.2.1.4"
RuntimeError: Failed to boot device!
```

- `lsusb` montre `03e7:2485` (mode DFU) → device physiquement présent
- `getAllConnectedDevices()` → vide ou UNBOOTED
- `DeviceBootloader.getAllAvailableDevices()` → vide (depthai skippe les UNBOOTED)
- `dai.Device(pipeline, dev_info, True)` → RuntimeError immédiat

## Cause racine

Le MyriadX démarre toujours en mode DFU (`03e7:2485`). depthai doit uploader le firmware en RAM via USB. Si cette tentative échoue (crash en cours de DFU, mauvaise config USB 3.0 sur bus 2.0, perte d'alimentation), le cache XLink marque le device comme "skip UNBOOTED". Les méthodes de reset sysfs (`authorized=0→1`, `unbind/rebind` du device) ne suffisent pas car XLink a son propre cache interne.

## Séquence de récupération (validée sur Jetson Nano, depthai 2.26.0.0)

### Étape 1 — Forcer une ré-énumération via le hub parent

Le hub intermédiaire `1-2.1` (au-dessus de l'OAK-D, ne porte ni WiFi ni VESC) :

```bash
# En tant que root sur la Jetson
echo '1-2.1' > /sys/bus/usb/drivers/usb/unbind
sleep 3
echo '1-2.1' > /sys/bus/usb/drivers/usb/bind
sleep 5
```

Vérification : `lsusb | grep 03e7` — le numéro de device doit changer (ex: Dev 064 → Dev 066).

> **Note topologie :** `1-2` = hub Genesys Logic principal | `1-2.1` = hub intermédiaire 214b:7250 | `1-2.1.4` = OAK-D

### Étape 2 — bootMemory : charger le firmware en RAM

```python
import depthai as dai
import time

bls = dai.DeviceBootloader.getAllAvailableDevices()
# Après le rebind, getAllAvailableDevices() retourne le device UNBOOTED

bl = dai.DeviceBootloader(bls[0], allowFlashingBootloader=True)
print("BL version:", bl.getVersion())  # → 0.0.28

fw = dai.DeviceBootloader.getEmbeddedBootloaderBinary(dai.DeviceBootloader.Type.USB)
print("FW size:", len(fw))  # → ~788 400 bytes

bl.bootMemory(fw)  # charge le bootloader en RAM — pas de callback nécessaire !
del bl
time.sleep(12.0)  # attendre reboot MyriadX (~10-15s)
```

Le device passe de `X_LINK_UNBOOTED` à `X_LINK_BOOTLOADER`.

### Étape 3 — Ouvrir le pipeline normalement

```python
connected = dai.Device.getAllConnectedDevices()
# → [('1.2.1.4', XLinkDeviceState.X_LINK_BOOTLOADER)]

dev_info = connected[0]
with dai.Device(pipeline, dev_info, True) as device:  # True = USB 2.0
    q = device.getOutputQueue("preview", maxSize=1, blocking=False)
    pkt = q.get()
    bgr = pkt.getCvFrame()
    print(bgr.shape)  # → (320, 640, 3) ✅
```

## API depthai 2.26 — Points clés

| Méthode | Signature | Note |
|---------|-----------|------|
| `DeviceBootloader.bootMemory(fw)` | `fw: List[int]` | **Pas de callback !** Contrairement à `flashBootloader` |
| `DeviceBootloader.bootUsbRomBootloader()` | aucun arg | Alternative si bootMemory ne suffit pas |
| `DeviceBootloader.getEmbeddedBootloaderBinary(type)` | `type: DeviceBootloader.Type` | Retourne `List[int]` |
| `DeviceBootloader.getAllAvailableDevices()` | - | Retourne les UNBOOTED **seulement après** le rebind hub |
| `DeviceBootloader.getAllConnectedDevices()` | - | Idem |

## Intégration dans controller_pd.py

La recovery est désormais automatique dans la boucle de reconnexion :

1. `_usb_reset_method4_parent_hub()` — unbind/rebind `1-2.1`
2. Si device UNBOOTED/BOOTLOADER → `bootMemory(getEmbeddedBootloaderBinary(USB))`
3. Attendre 12s → `dai.Device(pipeline, dev_info, True)`

La méthode 4 est intégrée dans la rotation `_usb_reset_oak()` (1 → 2 → 4 → 3).

## Prévention

- **Ne jamais passer `usb2Mode=False`** sur Jetson Nano — le bus USB est physiquement 480M, pas 5Gbps.
- En production, toujours `dai.Device(pipeline, dev_info, True)` avec `True` = USB 2.0 forcé.
- Si le device reste UNBOOTED après 3 cycles de recovery automatique → débrancher/rebrancher le câble USB physiquement.

## Topologie USB Jetson Nano (référence)

```
Bus 01 (USB 2.0, 480M)
└── Port 2: 05e3:0610 Hub Genesys Logic 4-port        [1-2]
    ├── Port 1: 214b:7250 Hub intermédiaire            [1-2.1]  ← unbind/rebind ICI
    │   ├── Port 2: 214b:7250 Hub                      [1-2.1.2]
    │   └── Port 4: 03e7:2485 OAK-D Lite (DFU/UNBOOTED) [1-2.1.4]
    ├── Port 2: 0bda:b812 WiFi TP-Link                 [1-2.2]
    └── Port 4: 0483:5740 VESC STM32                   [1-2.4]
Bus 02 (USB 3.0, 5000M)
└── Port 1: 05e3:0620 Hub Genesys Logic 4-port SS     [2-1]
    (aucun device actif côté SS — OAK-D sur bus 01 uniquement)
```

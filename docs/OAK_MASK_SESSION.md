# OAK-D Lite + Masque Piste — Session 2026-06-05 / 2026-06-12

## Problème de départ

L'OAK-D Lite **ne fonctionne pas sous WSL2**. Raison technique :
- Le chip Movidius MyriadX démarre en état `X_LINK_UNBOOTED`
- depthai envoie le firmware → le device se **déconnecte USB** pendant le boot
- WSL2 perd la connexion usbipd à ce moment → impossible de récupérer le device
- 15 tentatives de reconnexion dans la boucle : toutes échouent

**Conclusion** : OAK-D Lite = **Windows natif uniquement** (pas de WSL).

---

## Solution : Python Windows natif

### Installation
- Python 3.11 embeddable téléchargé dans `C:\python311\`
- Activation de `pip` : éditer `python311._pth` → décommenter `import site`
- Télécharger `get-pip.py` manuellement puis installer
- Packages installés :
  ```
  C:\python311\python.exe -m pip install depthai opencv-python numpy
  ```
- Version installée : **depthai 3.6.1**

### Driver USB (obligatoire)
- Zadig 2.9 → **Options → List All Devices** → sélectionner `Movidius MyriadX`
- Installer driver **WinUSB** (remplace le driver par défaut)
- Sans ça, Windows ne peut pas communiquer avec l'OAK-D

### Détacher de WSL si besoin
```powershell
usbipd detach --busid 1-2   # ou le busid correspondant
usbipd list                 # vérifier STATE = "Not shared"
```

---

## Différences API depthai 2.x vs 3.x

En depthai **3.x**, l'API a changé par rapport à 2.x :

| depthai 2.x | depthai 3.x |
|-------------|-------------|
| `dai.Pipeline()` → configurer → `dai.Device(pipeline)` | `dai.Device()` d'abord, puis `dai.Pipeline(device)` |
| `dai.node.XLinkOut` | **Supprimé** → queues directement sur les outputs |
| `dai.node.ColorCamera` | Deprecated → utiliser `dai.node.Camera` |
| `cam.setFps(30)` | **Supprimé** sur Camera node |
| `device.getOutputQueue(name)` | `output.createOutputQueue(maxSize, blocking)` |

### Code correct depthai 3.x
```python
with dai.Device() as device:
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q = cam.requestOutput((W, H), dai.ImgFrame.Type.BGR888p).createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    while True:
        frame = q.get()          # bloquant — attend le frame
        bgr = frame.getCvFrame() # numpy array BGR
```

### Connexion aux devices BOOTED
Si le device est déjà en état `X_LINK_BOOTED` (script précédent non terminé) :
```python
# getAllConnectedDevices() inclut tous les états
for d in dai.Device.getAllConnectedDevices():
    device_info = d
    break
with dai.Device(device_info) as device:
    ...
```
Si bloqué : `Get-Process python | Stop-Process -Force` + débrancher/rebrancher USB.

---

## Infos calibration OAK-D Lite (mesurées)

Résolution de travail : **512 × 256 pixels**

```
fx = 402.1 px    (focale horizontale)
fy = 402.1 px    (focale verticale — caméra symétrique)
cx = 262.4 px    (centre optique x)
cy = 127.3 px    (centre optique y)
FOV horizontal = 68.8°
Distortion = [-4.41, 12.05, -0.0002, 0.0002, -5.99, -4.48, 12.29, -6.27, ...]
```

Ces valeurs sont stockées dans la calibration interne de la caméra
et lues via `device.readCalibration()`.

---

## Pipeline de masquage (lignes blanches)

### Principe
Méthode **100% OpenCV**, sans ML, temps réel même sur Jetson Nano.

```
Image BGR 512×256
    ↓ cv2.cvtColor → HSV
    ↓ cv2.inRange(HSV_LOW=[0,0,180], HSV_HIGH=[180,50,255])
Masque binaire (blanc = pixel blanc détecté)
    ↓ ROI : annuler la moitié haute (ciel / plafond)
    ↓ cv2.morphologyEx OPEN  → enlève bruit sel-et-poivre
    ↓ cv2.morphologyEx CLOSE → referme les trous dans la ligne
Masque propre
    ↓ cv2.moments → centroïde
Centre de ligne (cx, cy)
    ↓ erreur = cx - W/2   (négatif = ligne à gauche, positif = droite)
```

### Paramètres HSV ajustables
```python
HSV_LOW  = [  0,   0, 180]   # H min, S min, V min
HSV_HIGH = [180,  50, 255]   # H max, S max, V max
ROI_TOP  = H // 2            # ignorer tout ce qui est au-dessus
MORPH_K  = 3                 # taille kernel nettoyage
```
- Touche `+` : augmente le seuil V (blanc plus strict)
- Touche `-` : baisse le seuil V (blanc plus permissif)

### Sortie visuelle
- **Gauche** : image live avec overlay vert (masque), point rouge (centre ligne), trait bleu (erreur vs centre)
- **Droite** : masque binaire pur

---

## Script principal

**Fichier** : `C:\Users\Admin\oak_info_mask.py`

**Touches** :
- `ESPACE` → sauvegarde `N_original_image.png` + `N_mask.png` (256×128) dans `C:\Users\Admin\raw_cam\`
- `M` → toggle affichage masque
- `+` / `-` → ajuster seuil de blanc
- `Q` ou `Échap` → quitter

**Images sauvegardées dans** : `C:\Users\Admin\raw_cam\`
**Accessibles depuis WSL** : `/mnt/c/Users/Admin/raw_cam/`

### Lancer
```powershell
# Vérifier que l'OAK-D n'est pas attachée à WSL
usbipd list   # STATE doit être "Not shared"

C:\python311\python.exe C:\Users\Admin\oak_info_mask.py
```

---

## Prochaines étapes dataset

1. Monter l'OAK-D Lite sur la voiture à la **même hauteur/angle** que prévu en déploiement
2. Faire rouler la voiture sur la piste (manuellement)
3. Capturer avec `ESPACE` : viser **200-500 images** couvrant :
   - Lignes droites (piste centrée)
   - Virages gauche / droite
   - Bords de piste (recovery)
4. Copier depuis `C:\Users\Admin\raw_cam\` vers WSL :
   ```bash
   cp /mnt/c/Users/Admin/raw_cam/*.png /home/lekrikri/Projects/G-CAR-000/data/raw_cam/
   ```
5. Lancer `src/mask/training/mask_generator.py` sur les images brutes pour générer les masques finaux

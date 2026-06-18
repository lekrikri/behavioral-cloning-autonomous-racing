# OAK-D Lite — Streaming Vidéo sur Jetson Nano

> Documentation complète du pipeline de streaming caméra (Issue #2).
> État : ✅ FONCTIONNEL avec limitations hardware (voir section Instabilité USB).
> Voir aussi : [`HARDWARE_DIAGNOSTICS.md`](HARDWARE_DIAGNOSTICS.md), [`HARDWARE_REALCAR.md`](HARDWARE_REALCAR.md)

---

## Résumé rapide

```bash
# Sur le Jetson Nano :
cd ~/behavioral-cloning-autonomous-racing
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve --dst-port 5600

# Sur le PC (VLC) :
# Média > Ouvrir un flux réseau → tcp://192.168.0.100:5600
```

---

## Architecture du pipeline

```
OAK-D Lite (caméra RGB)
      │
      │  USB2 (forcé via usb2Mode=True)
      ▼
Jetson Nano — depthai 2.26 (Python 3.6.9)
  ┌─ ColorCamera node (640x360 @ 15fps)
  ├─ VideoEncoder node (H.264 2000kbps)
  └─ XLinkOut → Python host
      │
      │  stdin pipe
      ▼
gst-launch-1.0
  fdsrc ! h264parse ! mpegtsmux ! tcpserversink port=5600
      │
      │  TCP (Jetson écoute, VLC se connecte)
      ▼
VLC sur Windows/Linux
  tcp://192.168.0.100:5600
```

---

## Utilisation

### Lancer le stream (Jetson Nano)

```bash
# Connexion SSH
ssh robocar@192.168.0.100

# Commande standard (H.264, 640x360, 15fps)
cd ~/behavioral-cloning-autonomous-racing
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve --dst-port 5600

# Avec enregistrement local en même temps
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve --dst-port 5600 --record /tmp/session.h264
```

### Recevoir sur le PC

**VLC (recommandé) :**
1. Média → Ouvrir un flux réseau
2. URL : `tcp://192.168.0.100:5600`
3. *(Optionnel)* Outils → Préférences → Tout → Codec → Mise en cache réseau → `50 ms`

**GStreamer (Linux/Mac) :**
```bash
gst-launch-1.0 tcpclientsrc host=192.168.0.100 port=5600 \
  ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

---

## Options disponibles

| Option | Défaut | Description |
|---|---|---|
| `--serve` | — | Mode serveur TCP (recommandé — traverse les firewalls) |
| `--dst-ip IP` | broadcast | IP destination en mode UDP |
| `--dst-port PORT` | 5600 | Port TCP/UDP |
| `--codec h264\|mjpeg` | h264 | H.264 = stable via TCP / MJPEG = faible latence UDP |
| `--width W` | 640 | Largeur caméra |
| `--height H` | 360 | Hauteur caméra |
| `--fps N` | 15 | Fréquence (réduire si crashs OAK-D) |
| `--bitrate N` | 2000 | Bitrate H.264 kbps |
| `--record PATH` | — | Enregistrer en local |
| `--preview` | — | Fenêtre OpenCV locale (nécessite display) |

---

## Pourquoi `OPENBLAS_CORETYPE=ARMV8` est obligatoire

Le Jetson Nano contient un CPU Cortex-A57 (ARM v8). La version de numpy installée utilise OpenBLAS, qui tente par défaut d'auto-détecter le cœur CPU. Sur le Cortex-A57, cette détection échoue et provoque un `SIGILL` (Illegal Instruction) immédiat.

```
Illegal instruction (core dumped)
```

**Fix** : forcer le type de cœur avant tout import numpy/depthai :
```bash
export OPENBLAS_CORETYPE=ARMV8
```

---

## Pourquoi TCP server mode (`--serve`) et pas UDP

Le réseau 4G-CPE-ADC1 classe la connexion Windows en réseau **Public**. Le pare-feu Windows bloque tous les paquets UDP entrants. Deux approches possibles :

| Approche | Avantage | Inconvénient |
|---|---|---|
| **TCP server** (`--serve`) ✅ | Traverse le firewall (connexion sortante VLC) | +500ms à 2s de latence (TCP bufferisation) |
| UDP + règle firewall | Latence minimale (~100ms) | Nécessite créer une règle firewall Windows |

**TCP server mode** : la Jetson écoute sur le port 5600, VLC se connecte en sortant → aucune règle firewall nécessaire.

**UDP si firewall autorisé** :
```bash
# Jetson
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --dst-ip 192.168.0.104 --dst-port 5600 --codec h264

# VLC : Média > Ouvrir un flux réseau → rtp://@:5600  (avec fichier SDP)
```

---

## Instabilité OAK-D — X_LINK_ERROR

### Symptôme

```
[host] [warning] Device crashed, but no crash dump could be extracted.
RuntimeError: Communication exception — X_LINK_ERROR
```

Le stream s'interrompt après quelques secondes à quelques minutes. La reconnexion automatique reprend (`Tentative 2...`, `Tentative 3...`...).

### Cause racine

**Manque d'alimentation USB** sur le Jetson Nano.

Le Jetson Nano fournit **900mA max** sur son hub USB interne, partagés entre tous les ports. Les 4 ports sont tous occupés :

```
Port USB1 → OAK-D Lite    ← ~500-900mA en streaming (encodage VPU Myriad X)
Port USB2 → FSESC (VESC)  ← ~100mA
Port USB3 → TP-Link WiFi  ← ~400mA
Port USB4 → Clavier       ← ~100mA (débranché depuis)
```

**Total maximal théorique : ~1500mA. Disponible : 900mA. Déficit : ~600mA.**

Lorsque l'OAK-D démarre l'encodage H.264, la VPU Myriad X consomme un pic de courant. La tension sur le bus USB chute → brownout → crash OAK-D → X_LINK_ERROR.

### Preuves

- Les crashs surviennent systématiquement quelques secondes après le démarrage de l'encodage
- Débrancher le clavier (~100mA) réduit légèrement la fréquence des crashs
- Le phénomène ne se produit pas à l'arrêt (quand le transfert USB est minimal)
- Le message `Device crashed, but no crash dump could be extracted` = crash brutal (pas d'arrêt logiciel)

### Facteurs aggravants identifiés (à éviter)

| Configuration | Impact |
|---|---|
| `pipeline.setXLinkChunkSize(0)` | ❌ Plus de petits transferts = plus de ripple courant |
| `xout.setFpsLimit(N)` | ❌ Génère des interruptions USB plus fréquentes |
| `maxSize=4` dans `getOutputQueue` | ❌ Trop petit = blocage Python = pics de latence |
| Queue GStreamer `leaky=downstream` | ⚠️ Overhead CPU supplémentaire |
| `async=false` sur tcpserversink | ⚠️ Peut créer du backpressure |

### Solutions

#### Fix définitif (hardware) — Recommandé ✅

**Hub USB alimenté** branché sur l'OAK-D Lite.

```
Batterie ou secteur → Hub USB alimenté (5V, ≥2A)
                            └── OAK-D Lite (alimentation indépendante du Jetson)
                            └── (autres périphériques optionnel)

Jetson Nano USB
  ├── FSESC (VESC)
  └── TP-Link WiFi
```

Le hub alimenté fournit sa propre alimentation 5V à l'OAK-D → plus de dépendance au bus USB du Jetson.

**Hub recommandé** : n'importe quel hub USB 3.0 alimenté avec adaptateur 5V 2A+. Exemples :
- Anker USB 3.0 Hub 4 ports alimenté (~20€)
- UGREEN Hub USB alimenté (~15€)

#### Fix software (paliatif) — Appliqué ✅

Le script `camera_stream.py` gère la reconnexion automatique :

```python
# Backoff exponentiel : 5s, 10s, 15s... max 30s entre les tentatives
delay = min(5 * attempt, 30)
```

Quand l'OAK-D crashe :
1. GStreamer est proprement fermé (`try/finally`)
2. Attente du délai de backoff
3. Réinitialisation de l'OAK-D et du pipeline depthai
4. VLC voit la connexion TCP se couper → relancer la lecture manuellement

#### Réduction de la charge USB — Paliatif partiel

```bash
# Réduire FPS et résolution pour diminuer la charge VPU
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve \
  --fps 10 --width 416 --height 312 --bitrate 1000
```

---

## Choix techniques — Pourquoi pas MJPEG ?

### Codec

| Codec | Latence | Bande passante | Compatible TCP+VLC | Charge VPU |
|---|---|---|---|---|
| **H.264** ✅ | ~500ms-2s | Faible (~2Mbps) | ✅ via mpegtsmux | Moyenne |
| MJPEG | ~100-200ms | Très élevée (~50-200Mbps) | ⚠️ via matroskamux | Faible |

Le MJPEG est idéal en latence mais :
1. **`mpegtsmux` ne supporte pas `image/jpeg`** — il faut `matroskamux`
2. La bande passante très élevée du MJPEG sature le bus USB → aggrave les crashs
3. VLC ne démarre pas toujours proprement avec un stream matroskamux live

**Conclusion** : H.264 via mpegtsmux est le seul codec fiable en mode TCP server avec VLC sur cette configuration.

### Conteneur

- `mpegtsmux` : supporte H.264, H.265, MPEG — mais pas MJPEG
- `matroskamux` : supporte MJPEG — mais mauvaise compatibilité live streaming VLC

### API depthai 2.x vs 3.x

| API | depthai 2.x (Jetson, Python 3.6) | depthai 3.x (Windows, Python 3.11) |
|---|---|---|
| Force USB2 | `dai.Device(pipeline, True)` | `dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)` |
| Créer Device | `with dai.Device(pipeline) as device:` | idem |
| XLinkOut | `device.getOutputQueue(name, maxSize, blocking)` | idem |

⚠️ **Piège** : utiliser `maxUsbSpeed=dai.UsbSpeed.HIGH` sur depthai 2.26 est silencieusement ignoré — USB3 reste actif.

---

## Debugging

### Vérifier que le stream tourne sur la Jetson

```bash
ssh robocar@192.168.0.100
cat /tmp/cam.log                          # logs du stream
ps aux | grep -E "camera_stream|gst-launch"  # process actifs
```

### Tester la pipeline GStreamer seule

```bash
# Test MJPEG (sur la Jetson)
gst-launch-1.0 fdsrc ! jpegparse ! matroskamux ! tcpserversink \
  host=0.0.0.0 port=5601 sync=false < /dev/zero

# Test H.264 (sur la Jetson)
gst-launch-1.0 fdsrc ! h264parse ! mpegtsmux ! tcpserversink \
  host=0.0.0.0 port=5601 sync=false < /dev/zero
```

### Vérifier l'alimentation USB

```bash
# Sur la Jetson (lecture conso globale)
sudo cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio\:device0/in_power0_input  # 5V rail mW
# Ou via jtop (si installé) : sudo jtop
```

### VLC ne démarre pas / image noire

1. Vérifier que le process tourne : `ps aux | grep camera_stream`
2. Vérifier les logs : `cat /tmp/cam.log`
3. Dans VLC : Outils → Messages → Erreurs (chercher codec/demux errors)
4. Essayer de relancer VLC — si crash OAK-D entre-temps, le stream a redémarré
5. Si la connexion TCP est refusée : attendre la fin du délai de reconnexion

---

---

## Architecture alternative : NVENC Jetson (recommandée pour stabilité)

> Script : `src/camera_stream_nvenc.py`

Au lieu de laisser le VPU Myriad X de l'OAK-D encoder le H.264 (grosse conso),
on sort du NV12 brut et on encode sur le GPU Jetson (NVENC).

```
OAK-D Lite (cam.video → NV12 brut)
      │ USB2 (~350-450mA vs ~750-900mA avant)
      ▼
Jetson — nvvidconv (CPU→NVMM zero-copy)
      ▼
nvv4l2h264enc (NVENC hardware, quasi 0% CPU)
      ▼
mpegtsmux ! tcpserversink
      ▼
VLC tcp://192.168.0.100:5600
```

**Gain estimé** : -30 à 50% de consommation OAK-D (plus d'encodeur Myriad X actif).

### Lancer le mode NVENC

```bash
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream_nvenc.py --serve --dst-port 5600
```

### Test A/B pour confirmer la cause des crashs

```bash
# Session 1 : encoder Myriad (camera_stream.py) — compter crashs sur 10 min
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream.py --serve

# Session 2 : NVENC Jetson (camera_stream_nvenc.py) — compter crashs sur 10 min
OPENBLAS_CORETYPE=ARMV8 python3 -u src/camera_stream_nvenc.py --serve
```

Si session 2 a nettement moins de crashs → le VPU encoder Myriad X était la cause principale.

### Intégration depth map + NVENC (inférence temps réel)

Pour le pipeline complet (streaming + inférence BC simultanés) :

```python
# Pipeline depthai recommandé (faible conso)
cam.video.link(xout_rgb.input)      # NV12 brut → NVENC → stream VLC
# Depth separée, basse fréquence :
stereo.depth.link(xout_depth.input) # setFpsLimit(5) sur xout_depth

# Au lieu d'envoyer la depth map entière (500KB/frame)
# Calculer les 20 raycasts sur Jetson (80 bytes/frame)
# → division trafic USB par ~6250
```

---

## Roadmap caméra

- [x] Streaming H.264 TCP fonctionnel (640x360 @ 15fps)
- [x] Reconnexion automatique après crash OAK-D
- [x] Mode UDP/RTP disponible (`--codec mjpeg` sans `--serve`)
- [x] `usb2Mode=True` (réduit pic courant OAK-D, API depthai 2.x correcte)
- [x] `src/camera_stream_nvenc.py` — NVENC Jetson, décharge VPU Myriad X
- [x] **Test A/B validé** (2026-06-18) : NVENC ne réduit PAS les crashs — NV12 brut
  (345KB/frame) génère PLUS de trafic USB que H.264 Myriad (~16KB/frame).
  L'encodeur Myriad X compense son coût VPU par la réduction massive de trafic USB.
  → Garder `camera_stream.py` avec VideoEncoder OAK-D comme solution principale.
- [ ] **Hub USB alimenté** → fix définitif crashs (hardware ~15-20€)
- [ ] Intégration dans `inference_realcar.py` (stream + depth + inférence simultanés)
- [ ] Réduire latence TCP (RTSP avec `gst-rtsp-server` → ~200ms vs ~1-2s TCP)
- [ ] Calcul raycasts sur Jetson (envoyer 80 bytes/frame au lieu de 500KB depth map)
- [ ] Fix VESC APP=No App (accoups moteur) — voir `VESC_DIAGNOSTIC_RESUME.md`

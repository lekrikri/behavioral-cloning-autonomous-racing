# Implémentation Voiture Réelle — Plan consolidé (LLMs consensus)

> Synthèse des recommandations Gemini + Grok + ChatGPT.
> Points critiques identifiés, code prêt à déployer.

---

## ⚠️ Points critiques identifiés par les LLMs

### 1. Problème de tangage (BLOQUANT — Gemini)
Notre approche `row_center` unique échoue dès que la voiture accélère ou freine :
le nez monte/descend → les rayons pointent vers le ciel (∞) ou le sol (0).

**Solution : ROI band** (bande horizontale 40%→60% de la hauteur) + distance min par colonne.

### 2. tan(angle) incomplet (Grok)
Notre projection était correcte conceptuellement mais manquait les intrinsics caméra réels.
Utiliser `focal_length = w / (2 * tan(fov/2))` pour la projection.

### 3. Z-score simulation ≠ Z-score réel (BLOQUANT — consensus)
Les `ray_stats.json` de Unity ne fonctionneront pas sur la depth map réelle.
**Recalcul obligatoire** sur 200–500 frames réelles (2-5 min de roulage manuel).

### 4. Watchdog hardware VESC (VITAL — Gemini)
Dans VESC Tool → App Settings → General → **Timeout = 200ms**.
Si le Jetson plante → VESC coupe les moteurs en 0.2s automatiquement.

### 5. Premier test = roues en l'air (SÉCURITÉ — consensus)
**Jamais tester au sol d'abord.** Châssis surélevé, roues dans le vide.
Moteur Traxxas BLSS 3300kV à 100% duty → crash immédiat.

---

## Paramètres recommandés (consensus)

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| MAX_DISTANCE_MM | 3000–4000 | Piste RC indoor ~3-4m, au-delà = bruit stéréo |
| row_band | 40%→60% hauteur | Robustesse tangage |
| duty_max (tests) | 0.15 (15%) | Sécurité — Traxxas BLSS 3300 très puissant |
| duty_max (stable) | 0.35–0.40 | Après calibration confirmée |
| fréquence contrôle | 30–50 Hz | OAK-D ~30 FPS, ONNX <2ms |
| VESC timeout | 200ms | Watchdog hardware |

---

## depth_to_rays.py — Version finale consolidée

```python
import numpy as np
import cv2

class DepthToRays:
    """
    Bridge robuste depth map OAK-D Lite → raycasts virtuels [0,1].
    Gère : tangage (ROI band), NaN/0, outliers, fov réel caméra.
    """

    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 400,
        fov_deg: float = 100.0,        # FOV horizontal OAK-D Lite (~100°)
        n_rays: int = 20,
        max_dist_mm: float = 3500.0,
        row_band: tuple = (0.40, 0.60), # 40%→60% hauteur (robustesse tangage)
    ):
        self.W = img_width
        self.H = img_height
        self.max_dist_mm = max_dist_mm
        self.n_rays = n_rays
        self.row_start = int(img_height * row_band[0])
        self.row_end   = int(img_height * row_band[1])

        # Focal length depuis FOV réel (intrinsics approximatifs)
        self.focal = img_width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))

        # Angles des rayons (-fov/2 → +fov/2)
        self.angles_deg = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_rays)

        # Colonnes pixel pré-calculées (optimisation)
        self.cols = np.clip(
            (self.W / 2.0 + np.tan(np.deg2rad(self.angles_deg)) * self.focal).astype(int),
            0, self.W - 1
        )

    def __call__(self, depth_frame: np.ndarray) -> np.ndarray:
        """
        depth_frame : uint16, valeurs en mm (OAK-D depth stream)
        retourne    : np.ndarray [n_rays] float32, valeurs dans [0, 1]
        """
        # 1. Extraire la ROI (bande horizontale anti-tangage)
        roi = depth_frame[self.row_start:self.row_end, :].astype(np.float32)

        # 2. Masquer les valeurs invalides (0 = pas de disparité, trop loin)
        roi[roi == 0] = np.nan
        roi[roi > self.max_dist_mm] = np.nan

        # 3. Bruit sol / objets parasites : ignorer distances < 100mm
        roi[roi < 100] = np.nan

        # 4. Distance minimum par colonne dans la ROI
        #    → simule le comportement d'un raycast physique (premier obstacle)
        with np.errstate(all='ignore'):
            col_distances = np.nanmin(roi, axis=0)  # shape (W,)

        # 5. Remplir les colonnes sans donnée valide par max_dist (pas d'obstacle)
        col_distances = np.where(np.isnan(col_distances), self.max_dist_mm, col_distances)

        # 6. Échantillonner aux angles des raycasts
        sampled = col_distances[self.cols]  # shape (n_rays,)

        # 7. Normaliser [0, 1]
        rays = np.clip(sampled / self.max_dist_mm, 0.0, 1.0).astype(np.float32)

        return rays  # shape (20,)
```

---

## vesc_interface.py — Version finale

```python
import serial
import pyvesc
import numpy as np
import time


class VESCInterface:
    """
    Interface VESC pour Flipsky FSESC Mini V6.7 Pro.
    steering ∈ [-1,1] → servo | accel ∈ [0,1] → duty cycle
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        servo_center: float = 0.5,
        servo_range: float = 0.35,   # amplitude ±0.35 (ajuster selon servo physique)
        duty_max: float = 0.15,      # 15% pour les premiers tests (SÉCURITÉ)
    ):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.05)
        self.servo_center = servo_center
        self.servo_range  = servo_range
        self.duty_max     = duty_max
        self.last_cmd_time = time.time()

    def send(self, steering: float, accel: float) -> None:
        """Envoyer commandes de direction et accélération."""
        # Servo : [-1,1] → [center-range, center+range]
        servo_pos = self.servo_center + self.servo_range * float(steering)
        servo_pos = np.clip(servo_pos, 0.15, 0.85)  # sécurité mécanique

        # Duty cycle : [0,1] → [0, duty_max]
        duty = float(accel) * self.duty_max
        duty = np.clip(duty, 0.0, self.duty_max)

        self.ser.write(pyvesc.encode(pyvesc.SetServoPos(servo_pos)))
        self.ser.write(pyvesc.encode(pyvesc.SetDutyCycle(duty)))
        self.last_cmd_time = time.time()

    def stop(self) -> None:
        """Arrêt d'urgence."""
        self.ser.write(pyvesc.encode(pyvesc.SetDutyCycle(0.0)))
        self.ser.write(pyvesc.encode(pyvesc.SetServoPos(self.servo_center)))
        print("[VESC] STOP")

    def get_rpm(self) -> float:
        """Lire le RPM moteur (vitesse réelle)."""
        try:
            self.ser.write(pyvesc.encode_request(pyvesc.GetValues))
            time.sleep(0.005)
            raw = self.ser.read(100)
            msg, _ = pyvesc.decode(raw)
            if isinstance(msg, pyvesc.GetValues):
                return float(msg.rpm)
        except Exception:
            pass
        return 0.0

    def close(self) -> None:
        self.stop()
        self.ser.close()
```

---

## inference_realcar.py — Architecture multi-thread

```python
"""
inference_realcar.py — Inférence temps réel voiture physique.
Architecture : Perception thread → Inference thread → Control thread
"""

import threading
import time
import json
import numpy as np
import depthai as dai
import onnxruntime as ort

from depth_to_rays import DepthToRays
from vesc_interface import VESCInterface


# ─── Config ────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/v18/best.onnx"
STATS_PATH    = "models/real_ray_stats.json"   # ← recalculé sur données réelles !
VESC_PORT     = "/dev/ttyACM0"
DUTY_MAX      = 0.15    # 15% pour les premiers tests → augmenter progressivement
WATCHDOG_S    = 0.5     # arrêt si pas de frame depuis 500ms
# ───────────────────────────────────────────────────────────────────────────


class AdaptiveSmoothingFilter:
    def __init__(self, alpha=0.57, alpha_max=0.92, deadzone=0.06):
        self.alpha_base = alpha
        self.alpha_max  = alpha_max
        self.deadzone   = deadzone
        self._s = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw)
        if self._s is None:
            self._s = raw.copy()
        else:
            delta = abs(raw[0] - self._s[0])
            alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * min(delta, 1.0)
            self._s = alpha * raw + (1 - alpha) * self._s
        result = self._s.copy()
        if abs(result[0]) < self.deadzone:
            result[0] = 0.0
        return result


class RealCarInference:

    def __init__(self):
        # Caméra
        self.depth_bridge = DepthToRays()
        self._setup_depthai()

        # Modèle ONNX
        self.sess = ort.InferenceSession(
            MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

        # Z-score RÉEL (recalculé sur données réelles)
        with open(STATS_PATH) as f:
            stats = json.load(f)
        self.ray_mu    = np.array(stats["mean"], dtype=np.float32)
        self.ray_sigma = np.array(stats["std"],  dtype=np.float32)

        # VESC
        self.vesc    = VESCInterface(port=VESC_PORT, duty_max=DUTY_MAX)
        self.smoother = AdaptiveSmoothingFilter()

        # État partagé (thread-safe via verrou)
        self._lock       = threading.Lock()
        self._latest_rays = None
        self._last_frame  = time.time()
        self._running     = True

    def _setup_depthai(self):
        pipeline = dai.Pipeline()

        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        stereo  = pipeline.create(dai.node.StereoDepth)
        xout    = pipeline.create(dai.node.XLinkOut)

        mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)      # réduit le bruit stéréo
        stereo.setSubpixel(False)
        stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setConfidenceThreshold(230)  # filtre agressif des pixels incertains

        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)
        stereo.depth.link(xout.input)
        xout.setStreamName("depth")

        self._device = dai.Device(pipeline)
        self._depth_queue = self._device.getOutputQueue("depth", maxSize=1, blocking=False)

    # ── Thread 1 : Perception ──────────────────────────────────────────────
    def _perception_thread(self):
        while self._running:
            msg = self._depth_queue.tryGet()
            if msg is not None:
                depth_frame = msg.getFrame()   # uint16, mm
                rays = self.depth_bridge(depth_frame)
                with self._lock:
                    self._latest_rays = rays
                    self._last_frame  = time.time()
            time.sleep(0.005)

    # ── Thread 2 : Inférence + Contrôle ───────────────────────────────────
    def _control_thread(self):
        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                rays      = self._latest_rays
                last_frame = self._last_frame

            # Watchdog : pas de frame depuis WATCHDOG_S → arrêt
            if time.time() - last_frame > WATCHDOG_S:
                self.vesc.stop()
                print(f"[WATCHDOG] Pas de frame depuis {WATCHDOG_S}s — arrêt.")
                time.sleep(0.1)
                continue

            if rays is None:
                time.sleep(0.005)
                continue

            # Z-score réel
            rays_z = (rays - self.ray_mu[:20]) / (self.ray_sigma[:20] + 1e-8)

            # Features dérivées (identiques à inference.py simulation)
            half = len(rays_z) // 2
            asymmetry = (rays_z[half:].sum() - rays_z[:half].sum()) / (rays_z.sum() + 1e-8)
            front_ray = float(rays_z[half-1:half+1].mean())
            min_ray   = float(rays_z.min())
            derived   = np.array([asymmetry, front_ray, min_ray], dtype=np.float32)

            features = np.concatenate([rays_z, derived]).reshape(1, -1)

            # Inférence ONNX
            pred = self.sess.run(None, {self.input_name: features})[0][0]

            # Smoothing adaptatif
            pred_smooth = self.smoother(pred)

            # Steering : offset biais droite (hérité simulation)
            steer_raw = pred_smooth[0]
            if 0.05 < abs(steer_raw) < 0.35:
                steer_raw -= 0.02 * np.sign(steer_raw)
            steering = float(np.clip(steer_raw, -1.0, 1.0))

            # Accel : heuristique front_raw (rayon frontal brut non Z-scoré)
            front_raw = float(rays[half-1:half+1].mean())  # non Z-scoré
            geo_base  = max(0.35, 1.0 - 1.2 * abs(steering))
            front_cap = 1.0 if front_raw >= 0.65 else (0.45 + 0.70 * front_raw)
            accel = float(np.clip(min(geo_base, front_cap), 0.35, 0.95))

            # Envoi VESC
            self.vesc.send(steering, accel)

            # ~30 Hz
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.033 - elapsed))

    # ── Run ────────────────────────────────────────────────────────────────
    def run(self):
        print("[RealCar] Démarrage — Ctrl+C pour arrêter.")
        t_perc = threading.Thread(target=self._perception_thread, daemon=True)
        t_ctrl = threading.Thread(target=self._control_thread, daemon=True)
        t_perc.start()
        t_ctrl.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[RealCar] Arrêt demandé.")
        finally:
            self._running = False
            self.vesc.stop()
            self._device.close()


if __name__ == "__main__":
    RealCarInference().run()
```

---

## calibrate_ray_stats.py — Recalibrer le Z-score sur données réelles

```python
"""
Roule manuellement 2-5 minutes sur la piste → collecte les raycasts réels.
Lance ce script pour recalculer ray_stats.json.
"""
import depthai as dai
import numpy as np
import json
import time
from depth_to_rays import DepthToRays

bridge = DepthToRays()
all_rays = []

print("Collecte Z-score — pousse la voiture manuellement 2-5 min. Ctrl+C pour terminer.")

pipeline = dai.Pipeline()
# ... (même setup stereo que inference_realcar.py)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("depth", maxSize=1, blocking=True)
    try:
        while True:
            frame = q.get().getFrame()
            rays  = bridge(frame)
            all_rays.append(rays)
            if len(all_rays) % 100 == 0:
                print(f"  {len(all_rays)} frames collectées...")
    except KeyboardInterrupt:
        pass

rays_arr = np.stack(all_rays)  # (N, 20)
mean = rays_arr.mean(axis=0).tolist()
std  = np.maximum(rays_arr.std(axis=0), 1e-6).tolist()

# Ajouter les 3 derived features (mean=0, std=1 → pas de normalisation)
mean += [0.0, 0.0, 0.0]
std  += [1.0, 1.0, 1.0]

stats = {"mean": mean, "std": std}
with open("models/real_ray_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"[OK] real_ray_stats.json sauvegardé ({len(all_rays)} frames)")
```

---

## Checklist déploiement

### Avant de poser la voiture au sol

- [ ] **VESC Tool** : App Settings → General → Timeout = **200ms** (watchdog hardware)
- [ ] **VESC Tool** : Motor Settings → Current Limits → max ~25A
- [ ] Tester roues en l'air : passer la main devant la caméra gauche → vérifier que le servo tourne dans le bon sens
- [ ] Vérifier le port USB du FSESC : `ls /dev/ttyACM*` ou `/dev/ttyUSB*`
- [ ] `inference_realcar.py` : `DUTY_MAX = 0.15` (15%)

### Phase 1 — Roues en l'air (statique)
- [ ] Démarrer `inference_realcar.py`
- [ ] Vérifier que le servo répond correctement (sens, amplitude)
- [ ] Vérifier que le moteur répond (bruit, sens de rotation)
- [ ] Aucun emballement moteur

### Phase 2 — Premier tour piste (très lent)
- [ ] `DUTY_MAX = 0.15`
- [ ] Quelqu'un avec un kill-switch physique (débrancher batterie)
- [ ] Piste fermée, zone dégagée
- [ ] Observer : la voiture suit-elle la piste ? Dépasse-t-elle les bords ?

### Phase 3 — Recalibration Z-score
- [ ] Lancer `calibrate_ray_stats.py` (2-5 min de roulage manuel)
- [ ] Remplacer `ray_stats.json` → `real_ray_stats.json`
- [ ] Retester

### Phase 4 — Montée en vitesse progressive
| Phase | DUTY_MAX | Condition |
|-------|----------|-----------|
| Test 1 | 0.15 | Toujours OK au sol |
| Test 2 | 0.25 | Trajectoire confirmée |
| Stable | 0.35–0.40 | Après calibration complète |

---

## Prochaines améliorations identifiées

1. **Intégrer la vitesse VESC** : RPM → feature supplémentaire en entrée du modèle → accel apprise (pas heuristique)
2. **Fine-tuning sur données réelles** : freeze Conv1D, réentraîner MLP uniquement, LR=1e-5, mix 80% sim + 20% réel
3. **TensorRT FP16** : `trtexec --onnx=best.onnx --fp16 --saveEngine=best.plan` → latence <1ms
4. **Extrinsics caméra** : calibration précise de l'angle et hauteur de montage de l'OAK-D pour `row_band` optimal

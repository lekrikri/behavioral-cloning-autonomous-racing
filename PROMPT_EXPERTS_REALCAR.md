# Prompt Expert — Déploiement Modèle BC sur Voiture Autonome Réelle

> Copiez-collez ce prompt complet dans Gemini / ChatGPT / Grok pour obtenir leurs meilleures recommandations.

---

## CONTEXTE DU PROJET

Je travaille sur un projet de **voiture autonome par Behavioral Cloning (BC)** — imitation learning.
Le modèle a été entraîné en simulation Unity et doit maintenant tourner sur la **voiture physique réelle**.

### Ce qui a été accompli (simulation)

- **Modèle** : `RobocarSpatial` — Conv1D(2ch→12, k=3) + MLP(136→96→48→2), ~30k paramètres, PyTorch
- **Input** : 20 raycasts Z-scorés [simulateur] + 3 features dérivées (asymmetry, front_ray, min_ray) = vecteur [23]
- **Output** : [steering ∈ [-1,1], accel ∈ [0,1]]
- **Loss** : BimodalLoss = 0.88×HuberLoss(steer) + 0.12×BCEWithLogitsLoss(accel>0.25)
- **Anti-zigzag** : PairwiseSmoothingLoss λ=0.30 sur frames consécutives + temporal split (pas de shuffle)
- **Résultats simulation** : tour de piste en **24 secondes**, conduite fluide, zéro zigzag
- **Export** : modèle disponible en `.pth` (PyTorch) et `.onnx` (ONNX opset 11)

### Hardware de la voiture réelle

```
Batterie LiPo
      ↓
[Carte puissance centrale + grand radiateur]
      ├──→ Matek Systems UBEC Duo → 5V → Jetson Nano
      └──→ Flipsky FSESC Mini V6.7 Pro (VESC open-source)
                ├── Moteur brushless Traxxas BLSS 3300
                ├── Servo direction
                └── USB → Jetson Nano

Jetson Nano — ports USB:
      ├── Luxonis OAK-D Lite autofocus (p/n: a00483)  [USB3 — RGB + stéréo depth]
      ├── Flipsky FSESC Mini V6.7 Pro                 [USB — pyvesc]
      ├── TP-Link WiFi dongle
      └── Logitech dongle (clavier)
```

**OS Jetson** : JetPack (Ubuntu 18.04 / 20.04)
**GPU Jetson** : Maxwell 128 CUDA cores, 4GB RAM

### Le problème central — Sim-to-Real Gap

Le modèle a appris à conduire depuis des **raycasts Unity parfaits** [0,1].
La voiture réelle a une **caméra OAK-D Lite** avec depth map stéréo.

**Solution envisagée** : bridge depth map → raycasts virtuels :
```python
# Pour chaque angle i dans [-90°, +90°] avec 20 rayons uniformes :
col = int(width/2 + tan(angle_i) * focal_length)
distance_mm = depth_frame[row_center, col]
ray_i = min(distance_mm / MAX_DISTANCE_MM, 1.0)  # normaliser [0,1]
```

**Problème** : MAX_DISTANCE_MM inconnu, occlusions, bruit depth map stéréo, alignement caméra.

---

## QUESTIONS SPÉCIFIQUES

### Question 1 — Bridge Depth → Raycasts

Notre bridge `depth_to_rays()` est-il correct ?
Quelles corrections/améliorations proposez-vous pour :
- Gérer le bruit de la depth map stéréo OAK-D Lite (zones sans disparité = NaN/0)
- Choisir `MAX_DISTANCE_MM` (la piste fait ~5m de large, rayon max simulateur = 1.0)
- Choisir `row_center` (hauteur optimale dans l'image pour lire les distances de piste)
- Filtrer les outliers depth (valeurs aberrantes du stéréo)
- Gérer les objets au-dessus de la piste (jambes, etc.)

### Question 2 — Interface VESC (pyvesc)

Comment mapper proprement les sorties du modèle vers le FSESC ?
- `steering ∈ [-1,1]` → angle servo (commande PWM ou position)
- `accel ∈ [0,1]` → duty cycle moteur brushless
- Comment lire le RPM réel pour calculer la vitesse ?
- Quelle fréquence d'envoi des commandes est recommandée ?
- Y a-t-il des risques de saturation / emballement moteur à gérer ?

### Question 3 — Latence et fréquence

La boucle doit tourner le plus vite possible sur Jetson Nano.
- OAK-D Lite : ~30 FPS en depth
- ONNX Runtime GPU : < 2ms par inférence estimé
- Quelle architecture de boucle recommandez-vous ? (threading, asyncio, pipeline ?)
- Comment éviter que la lecture depth bloque l'envoi des commandes VESC ?

### Question 4 — Calibration caméra → piste réelle

Le modèle a été entraîné avec `ray_stats.json` (mean/std des 20 raycasts en simulation).
Ces stats ne correspondent PAS à la distribution des raycasts réels (depth map).
- Faut-il recalculer `ray_stats.json` sur des données depth réelles ? Comment ?
- Ou y a-t-il une meilleure stratégie de normalisation pour le domaine réel ?
- La distribution des raycasts virtuels depuis une depth map sera-t-elle trop différente de la simulation ?

### Question 5 — Améliorations du modèle pour le réel

Le modèle v18 utilise une heuristique pour l'accélération (pas apprise).
La vitesse réelle est maintenant disponible via VESC RPM.
- Comment intégrer la vitesse réelle comme feature d'entrée ?
- Faut-il recollecte sur piste réelle ou adapter le modèle existant ?
- La technique **domain randomization** en simulation aurait-elle aidé ?
- Y a-t-il un moyen de fine-tuner le modèle existant sur quelques passes réelles sans perdre les acquis simulation ?

### Question 6 — Robustesse et sécurité

La voiture est physique, une erreur = crash réel.
- Comment implémenter un **watchdog** (arrêt d'urgence si latence > seuil) ?
- Comment gérer la perte de signal depth (OAK-D déconnectée / zone sans disparité) ?
- Quelle stratégie de fallback si le modèle prédit des valeurs aberrantes ?
- Faut-il limiter la vitesse max pendant les tests initiaux ?

---

## CODE ACTUEL — RÉFÉRENCES

### inference.py (simulation) — parties clés

```python
# SmoothingFilter adaptatif
class SmoothingFilter:
    def __init__(self, alpha=0.57, alpha_max=0.92, deadzone=0.06):
        ...
    def update(self, raw):
        delta = abs(raw[0] - self._smoothed[0])
        alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * min(delta, 1.0)
        self._smoothed = alpha * raw + (1 - alpha) * self._smoothed
        if abs(self._smoothed[0]) < self.deadzone:
            self._smoothed[0] = 0.0
        return self._smoothed

# Heuristique accélération
raw_rays = obs.rays   # non Z-scorés [0,1]
front_raw = float(raw_rays[9:11].mean())
geo_base = max(0.35, 1.0 - 1.2 * abs(steering))
if front_raw >= 0.65:
    front_cap = 1.0
else:
    front_cap = 0.45 + 0.70 * front_raw
acceleration = float(np.clip(min(geo_base, front_cap), 0.35, 0.95))

# Offset biais droite
if 0.05 < abs(steer_raw) < 0.35:
    steer_raw = steer_raw - 0.02 * np.sign(steer_raw)
```

### Modèle RobocarSpatial

```python
class RobocarSpatial(nn.Module):
    def __init__(self, n_rays=20, n_derived=3, bimodal_accel=True):
        # Branch 1: Conv1D sur les rays
        # reshape rays → (B, 2, n_rays//2) : 2 canaux de 10 rays chacun
        self.conv = nn.Sequential(
            nn.Conv1d(2, 12, kernel_size=3, padding=1),
            nn.BatchNorm1d(12), nn.ReLU()
        )
        # Branch 2: FC sur derived features
        self.derived_fc = nn.Sequential(nn.Linear(n_derived, 16), nn.ReLU())
        # Fusion
        self.mlp = nn.Sequential(
            nn.Linear(120 + 16, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 48), nn.BatchNorm1d(48), nn.ReLU()
        )
        self.steer_head = nn.Linear(48, 1)
        self.accel_head = nn.Linear(48, 1)

    def forward(self, x):
        rays = x[:, :self.n_rays]          # [B, 20]
        derived = x[:, self.n_rays:]        # [B, 3]
        r = rays.reshape(B, 2, 10)          # 2 canaux spatiaux
        feat_r = self.conv(r).flatten(1)    # [B, 120]
        feat_d = self.derived_fc(derived)   # [B, 16]
        feat = self.mlp(torch.cat([feat_r, feat_d], dim=1))
        steer = torch.tanh(self.steer_head(feat))
        accel = torch.sigmoid(self.accel_head(feat))  # ou logit si bimodal
        return torch.cat([steer, accel], dim=-1)
```

### ray_stats.json (Z-score simulation)

```json
{
  "mean": [0.72, 0.68, 0.71, ...],   // 20 valeurs — moyennes raycasts simulation
  "std":  [0.18, 0.21, 0.19, ...]    // 20 valeurs — écarts-types raycasts simulation
}
```

---

## CE QU'ON ATTEND DE VOUS

1. **Critique et correction** de notre approche bridge depth → raycasts virtuels
2. **Code Python complet** pour `depth_to_rays()` robuste (gestion NaN, bruit, outliers)
3. **Code Python complet** pour `inference_realcar.py` avec depthai + pyvesc + ONNX
4. **Recommandations architecture** boucle temps réel (threading ? pipeline ?)
5. **Stratégie de recalibration** ray_stats.json pour le domaine réel
6. **Plan de fine-tuning** si le modèle ne transfère pas bien
7. **Checklist de sécurité** pour les premiers tests sur piste réelle

Soyez précis, proposez du code fonctionnel, et signalez les pièges courants
du sim-to-real transfer en robotique autonome.

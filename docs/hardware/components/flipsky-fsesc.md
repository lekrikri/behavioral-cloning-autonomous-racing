# Flipsky FSESC Mini V6.7 Pro (VESC)

> Contrôleur moteur open-source (ESC VESC-compatible). Partie du [montage](../MONTAGE.md).

## Identité
- **Modèle** : Flipsky FSESC Mini V6.7 Pro
- **Firmware** : VESC 6.0 (MCU STM32 embarqué)
- **Port Jetson** : USB → **`/dev/ttyACM0`**

## Rôle
- Pilote le **moteur brushless** (throttle) et le **servo de direction** (steering).
- Fournit la **télémétrie** : RPM moteur → vitesse réelle (résout le `speed = 0.0` hardcodé du simulateur).

## Communication
- Protocole VESC implémenté **from scratch** (pas de `pyvesc` / `PyCRC`), pour compatibilité Python 3.6+ sur le Jetson. CRC-XMODEM validé. Détails : [`../../CONTROL_STACK.md`](../../CONTROL_STACK.md).
- **Commande couple, pas duty** :
  ```python
  # ❌ SetDutyCycle(0.20) → ~50A overcurrent → FAULT_CODE_ABS_OVER_CURRENT
  # ✅ SetCurrent(5.0A)   → le FOC gère le couple → pas de spike
  ```

## Quirks / pièges connus
- ⚠️ **À-coups (« saccades ») à basse vitesse = APP réglée sur UART** dans la config VESC (et non un problème de latence). Vérifier le mode APP.
- Bug firmware Flipsky d'usine : valeurs FOC sensorless de démarrage incohérentes (`openloop_time < openloop_time_ramp`) → saccades violentes au boot. Cf. [`../../VESC_DIAGNOSTIC_RESUME.md`](../../VESC_DIAGNOSTIC_RESUME.md).
- Config de référence : `mcconf_HFI_ok.xml` (racine du repo).

## Tech
- Embarqué : Jetson Nano, ONNX Runtime GPU / TensorRT FP16.
- Banc de test mouvement : `bench_vesc.py`.

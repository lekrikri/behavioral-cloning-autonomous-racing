# Diagnostic VESC — Résumé des problèmes et solutions

## Contexte

Moteur : Traxxas Velineon 3300kV (brushless sensorless)
ESC    : Flipsky FSESC Mini V6.7 Pro (firmware VESC 6.0)
Problème initial : **saccades violentes au démarrage**, roues qui bloquent par à-coups.

---

## Problème 1 — Bug firmware Flipsky : openloop_time < openloop_time_ramp

Le firmware par défaut du Flipsky contient une **incohérence dans les paramètres de démarrage FOC sensorless** :

| Paramètre | Valeur par défaut | Rôle |
|-----------|-------------------|------|
| `foc_sl_openloop_time_ramp` | 0.10 s | durée de la montée en courant |
| `foc_sl_openloop_time` | **0.05 s** | durée totale de la phase open-loop |

`foc_sl_openloop_time` (0.05 s) était **inférieur** à `foc_sl_openloop_time_ramp` (0.10 s).  
Résultat : le VESC tentait de basculer en closed-loop sensorless **avant même d'avoir fini sa rampe de démarrage**. Il recommençait en boucle → saccades répétées.

De plus, `foc_sl_openloop_boost_q = 0 A` : aucun courant de couple au démarrage → le moteur n'avait aucune force pour vaincre l'inertie.

**Fix appliqué (via patch blob COMM_GET_MCCONF_DEFAULT) :**
- `foc_sl_openloop_time` : 0.05 s → **0.60 s**
- `foc_sl_openloop_time_ramp` : 0.10 s → **0.25 s**
- `foc_sl_openloop_boost_q` : 0 A → **7 A**
- `foc_openloop_rpm` : 900 → **2000 ERPM**

---

## Problème 2 — EEPROM corrompue : VESC Tool inutilisable

La commande `COMM_GET_MCCONF` (lire la config active) retournait :  
> *"Warning: Could not set mcconf due to wrong signature"*

La signature EEPROM était invalide — probablement corrompue lors d'une mise à jour ou d'une coupure brutale. VESC Tool ne pouvait ni lire ni écrire la config persistante.

**Contournement :** utiliser `COMM_GET_MCCONF_DEFAULT` (ID=14) qui retourne les valeurs hardcodées du firmware, patcher le blob en mémoire, puis réécrire via `COMM_SET_MCCONF` (ID=15). La config est appliquée en RAM immédiatement, sans passer par l'EEPROM.

`COMM_STORE_CONFIG` (ID=25) ne répond pas → la config ne persiste pas au redémarrage. **Solution de contournement :** service systemd `vesc-config` sur le Jetson Nano qui réapplique les patches à chaque démarrage.

---

## Problème 3 — Phase moteur déconnectée

Malgré les patches logiciels corrects, le moteur produisait 0 RPM avec des à-coups et consommait ~50 A à duty=20%.

**Diagnostic :**
- Résistance effective mesurée : **~66 mΩ**
- Résistance attendue (3 phases Velineon) : **~11-15 mΩ**
- Rapport : 4-5× trop élevé → **une phase était déconnectée**

Avec seulement 2 phases connectées, le champ magnétique tournant ne peut pas se former → le rotor reçoit des impulsions asymétriques → à-coups sans rotation nette.

**Fix :** rebrancher fermement les 3 connecteurs bullet entre le VESC et le moteur.

---

## Problème 4 — Détection FOC nécessaire après reconnexion

Après reconnexion des phases, les paramètres R, L et λ (flux linkage) en RAM étaient ceux mesurés avec une phase manquante → valeurs aberrantes → courant axe-d (id) saturait à 40 A au lieu de rester à 0.

**Fix :** envoyer `COMM_DETECT_APPLY_ALL_FOC` (ID=58) pour que le VESC mesure automatiquement R, L et λ du vrai moteur correctement câblé. Après détection : id ≈ 0, iq = courant de couple → rotation fluide.

---

## Résumé chronologique

```
Saccades
  └─► Bug openloop_time < openloop_time_ramp  →  patché via blob
        └─► rpm=0 malgré patches corrects
              └─► Phase déconnectée  →  reconnexion physique
                    └─► id=40A (paramètres FOC invalides)
                          └─► DETECT_APPLY_ALL_FOC  →  rotation fluide ✅
```

---

## Commandes utiles (Jetson Nano)

```bash
# Appliquer la config VESC manuellement
python3 ~/vesc_cli.py --apply

# Test rotation moteur (SET_CURRENT)
python3 ~/test_fluide.py

# Relancer la détection FOC automatique
python3 ~/vesc_detect_rl.py

# Vérifier le service autoconfig
sudo systemctl status vesc-config
```

# Tutoriel — Cartographie de la piste (mode CARTO)

## Principe

Le mode CARTO permet d'enregistrer le tracé du circuit pendant que tu le conduis manuellement avec la manette.  
Il utilise le gyroscope de l'OAK-D (BMI270) + la commande moteur pour reconstruire la position XY de la voiture frame par frame (*dead-reckoning*).

À la fin, tu obtiens :
- un fichier **`track_map.json`** avec tous les waypoints
- un fichier **`track_map.svg`** avec une carte vectorielle du circuit (virages colorés, start/finish)

---

## Matériel requis

- La voiture allumée (batterie OK)
- La manette Logitech F710 branchée en USB sur la Jetson
- Un téléphone ou PC connecté au **même réseau WiFi** que la voiture

---

## Étapes

### 1. Ouvrir l'interface

Sur ton téléphone ou PC, ouvre un navigateur et va à :

```
http://100.112.10.119:5601
```

> Si tu es sur le réseau Tailscale, utilise `100.112.10.119`. Sur le réseau local de la salle, demande l'IP locale à Christophe.

Tu dois voir le flux caméra en direct et les boutons de contrôle.

---

### 2. Passer en mode MANUEL (manette)

Clique le bouton **MANUEL (manette)** dans l'interface.

Le bouton passe en jaune = mode téléopération actif.

> La manette doit être connectée. Si la ligne `manette: —` ne change pas, vérifie que la manette est bien en mode **X** (pas D) et que le câble USB est branché sur la Jetson.

---

### 3. Placer la voiture au départ

Pose la voiture sur la piste, au niveau du repère **START** (si vous en avez un).

La voiture est à l'arrêt — elle ne démarre pas toute seule.

---

### 4. Démarrer l'enregistrement CARTO

Clique **▶ DÉMARRER CARTO** dans l'interface.

Le bouton passe en rouge (**■ ARRÊTER CARTO**) et le compteur de waypoints s'affiche :
```
rec: 0wpts → rec: 45wpts → rec: 150wpts ...
```

---

### 5. Conduire le circuit

Lance la voiture avec la manette et fais **un tour complet** à vitesse normale :

- **Stick droit axe X** → direction (gauche/droite)
- **Gâchette droite (RT)** → accélération
- **Gâchette gauche (LT)** → frein

**Conseils pour une bonne carte :**
- Roule à vitesse constante (pas de stop-and-go brutal)
- Reste bien centré sur la piste
- Reviens à peu près à l'endroit de départ

---

### 6. Arrêter l'enregistrement

Une fois le tour fini, clique **■ ARRÊTER CARTO**.

Le bouton repasse en vert et affiche :
```
idle: 650wpts
```

La carte est automatiquement sauvegardée dans `data/track_map.json` et `data/track_map.svg`.

---

### 7. Visualiser la carte

Clique sur **🗺 Voir carte** dans l'interface (ou va directement sur `http://100.112.10.119:5601/map.svg`).

La carte SVG s'ouvre dans le navigateur. Elle montre :

| Élément | Signification |
|---|---|
| Trait vert | Trajectoire complète |
| Cercle vert **START** | Point de départ |
| Cercle orange **END** | Point d'arrivée |
| Cercle rouge **R** | Virage à droite |
| Cercle bleu **L** | Virage à gauche |
| Légende en bas | Durée du tour, nb de virages, yaw total |

---

## Repasser en mode AUTONOME

Après avoir cartographié, clique **AUTONOME** pour remettre la voiture en pilotage automatique, puis **GO** pour démarrer.

---

## FAQ

**La voiture ne répond pas à la manette**  
→ Vérifie que le mode **MANUEL** est actif (bouton jaune). Vérifie que la manette est bien en mode X (switch X/D en haut à gauche de la manette).

**Le compteur de waypoints reste à 0**  
→ Le mapper est prêt mais la voiture ne bouge pas — assure-toi d'avoir appuyé sur GO ou d'être en mode MANUEL et de pousser la gâchette.

**La carte SVG est bizarre (forme étrange)**  
→ Normal si tu as fait des arrêts brutaux ou des demi-tours. La précision est d'environ ±5-10% sur un tour de ~10m (pas de GPS, calcul par intégration).

**Le fichier JSON est où ?**  
→ Sur la Jetson : `/home/robocar/behavioral-cloning-autonomous-racing/track_map.json`  
→ Pour récupérer : `scp jetson:/home/robocar/behavioral-cloning-autonomous-racing/track_map.json .`

**Changer la vitesse de référence pour la carte**  
→ Le paramètre `V_PER_DUTY` dans `src/track_mapper.py` (défaut = 3.0 m/s pour duty=1.0). Si la carte est trop grande ou trop petite par rapport à la réalité, ajuste cette valeur.

---

## Résumé en une ligne

> Ouvre l'UI → MANUEL → positionne la voiture → **▶ DÉMARRER CARTO** → fais un tour → **■ ARRÊTER CARTO** → clique **Voir carte**

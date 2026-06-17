# Accès SSH à la robocar (`ssh robocar`)

Comment configurer un accès SSH simple et sans mot de passe à la Jetson de la
voiture depuis **n'importe quel nouveau PC**. Objectif : ne plus jamais chercher
l'IP du robot sur le réseau.

---

## 1. Fiche d'identité du robot

| Paramètre        | Valeur                          |
|------------------|---------------------------------|
| Hostname         | `jetson-pge-msc`                |
| Utilisateur SSH  | `robocar`                       |
| Mot de passe     | `robocar` *(1ʳᵉ connexion seulement)* |
| Carte            | Jetson Nano (JetPack / Ubuntu 18.04, arm64) |
| Interface réseau | `wlan0`                         |
| IP sur la box 4G | `192.168.0.104` (passerelle `192.168.0.1`) |

---

## 2. Contraintes réseau découvertes (à connaître avant tout)

Ces faits expliquent les choix de setup ci-dessous.

- **Le mDNS (`*.local`) ne fonctionne PAS sur la box 4G.** La box filtre le
  trafic multicast (`224.0.0.251`). Donc `jetson-pge-msc.local` ne se résout
  jamais sur ce réseau, quelle que soit la config du PC. → On résout le nom
  autrement (voir étape 4).
- **L'unicast fonctionne** : ping vers l'IP, SSH, `scp` passent normalement.
- **Latence Wi-Fi élevée et variable** (~300–800 ms, pertes ponctuelles). Un
  premier `ping` peut échouer puis remarcher : ne pas conclure trop vite à une
  panne. Relancer avec plusieurs paquets (`ping -c5`).

> Le hostname `jetson-pge-msc` + le démon `avahi` sont bien configurés côté
> Jetson : le `.local` redeviendra utilisable automatiquement sur un réseau qui
> ne filtre pas le multicast (box domestique classique).

---

## 3. Clé SSH dédiée (sur le nouveau PC)

Une clé par robot, sans passphrase (accès robot de labo, réseau local) :

```bash
ssh-keygen -t ed25519 -f ~/.ssh/jetson_ed25519 -N "" -C "robocar-jetson"
```

Puis déposer la clé publique sur la Jetson (mot de passe `robocar`, **une seule
fois**). Lancer la commande **depuis le nouveau PC**, pas depuis la Jetson :

```bash
ssh-copy-id -i ~/.ssh/jetson_ed25519.pub robocar@192.168.0.104
```

> `ssh-copy-id` part toujours du **client** (le PC) vers le **serveur** (la
> Jetson) : c'est lui qui détient la clé à installer.

---

## 4. Résoudre le nom `jetson-pge-msc`

Le `.local` étant inutilisable sur la box 4G, on choisit selon l'usage.

### Cas A — usage fixe sur la box 4G (le plus simple)

Mapper le nom à l'IP dans `/etc/hosts` du nouveau PC :

```bash
echo "192.168.0.104  jetson-pge-msc" | sudo tee -a /etc/hosts
```

Pour que l'IP ne change jamais, **réserver le bail DHCP** sur la box : interface
d'admin (`http://192.168.0.1`) → section *DHCP / bail statique* → lier l'adresse
MAC du `wlan0` de la Jetson à `192.168.0.104`. Récupérer la MAC sur la Jetson :

```bash
ip link show wlan0
```

> Limite : `/etc/hosts` + IP fixe ne vaut que sur **ce** réseau. Sur un autre
> Wi-Fi, l'IP changera et il faudra mettre à jour la ligne (ou utiliser le cas B).

### Cas B — robot mobile, multi-réseaux (optionnel)

Installer Tailscale sur la Jetson : le nom suit la voiture sur **tout** réseau
ayant Internet, et le filtrage multicast n'a plus d'effet (tunnel WireGuard
unicast). Coût réel sur la Nano : ~30–50 Mo de RAM, ~0 % CPU au repos.

```bash
# sur la Jetson
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up   # ouvrir l'URL affichée et s'authentifier
```

Joindre ensuite la Jetson par son nom MagicDNS (`jetson-pge-msc`) ou son IP
`100.x`. Pas besoin de `/etc/hosts` dans ce cas.

> L'install passe par `apt` : l'erreur `nvidia-l4t-bootloader ... FAILED` en fin
> de transaction est **préexistante et sans rapport** (paquets NVIDIA à moitié
> configurés sur l'image). Tailscale s'installe quand même — vérifier avec
> `tailscale version`.

---

## 5. Alias SSH (sur le nouveau PC)

Ajouter dans `~/.ssh/config` :

```sshconfig
Host robocar jetson
    HostName jetson-pge-msc
    User robocar
    IdentityFile ~/.ssh/jetson_ed25519
    IdentitiesOnly yes
```

- `HostName jetson-pge-msc` : résolu via `/etc/hosts` (cas A) ou MagicDNS (cas B).
  Sur un réseau sans filtrage multicast, on peut mettre `jetson-pge-msc.local`.
- `IdentitiesOnly yes` : ne présente que cette clé (évite l'échec
  *« too many authentication failures »* si le PC a beaucoup de clés).

---

## 6. Vérification

```bash
ssh robocar
```

Doit ouvrir la session **sans mot de passe ni recherche d'IP**. Depuis ce point,
`ssh robocar`, `scp ... robocar:`, etc. fonctionnent partout où le nom se résout.

---

## 7. Dépannage

| Symptôme | Cause probable | Action |
|----------|----------------|--------|
| `ssh-copy-id` → `Connection refused` | Serveur SSH inactif sur la Jetson | Sur la Jetson : `sudo systemctl enable --now ssh`, puis refaire le `ssh-copy-id` |
| `ping jetson-pge-msc.local` → `Name or service not known` | Multicast filtré (box 4G) | Normal ici. Utiliser `/etc/hosts` (cas A) ou Tailscale (cas B) |
| `ssh robocar` redemande un mot de passe | Clé non copiée | Refaire l'étape 3 (`ssh-copy-id`) |
| Premier `ping` échoue puis remarche | Latence/pertes du Wi-Fi 4G | Ignorer, relancer `ping -c5` |
| Nom non résolu malgré `/etc/hosts` | Ligne `hosts:` de `nsswitch.conf` n'inclut pas `files` | Vérifier `grep '^hosts' /etc/nsswitch.conf` (doit contenir `files`) |
| L'IP a changé | Pas de réservation DHCP | Réserver le bail (étape 4, cas A) ou passer en Tailscale |
```

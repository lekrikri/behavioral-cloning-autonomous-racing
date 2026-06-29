# Flux vidéo & accès distant

## Hub partagé (service au boot — À CRÉER)

- Un **seul** process possède l'OAK-D : le **hub**. Preview navigateur et inférence sont
  des **clients** → réglage et conduite coexistent.
- Cible : le hub tourne en **service systemd au démarrage** de la Jetson, sur un **port
  défini**, prêt en permanence. (Aujourd'hui lancé à la main via `mask_stream.py --source hub` ;
  le passage en service reste à faire.)

## Accès depuis le PC

- **Tailscale** fournit l'accès réseau-indépendant (`ssh robocar`).
- Le partage Tailscale est **asymétrique** : Jetson→PC bloqué. **C'est toujours le PC qui initie.**
- On accède donc au flux par **`ssh -L <port>:localhost:<port> robocar`**, puis navigateur
  sur `localhost:<port>`. **Aucun push depuis la Jetson.**

## Référence

Procédure complète : `docs/RUN_STREAM_DRIVE.md`. Ne pas réintroduire les anciens streamers
push (RTP/UDP, `oak_stream.py --tcp`) — ils luttent contre l'asymétrie Tailscale.

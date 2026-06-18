# Kyber sur la Jetson — audit de faisabilité (flux caméra)

> Faut-il utiliser **Kyber** (streaming vidéo+contrôle ultra-faible latence de
> J-B Kempf / VLC) pour le retour vidéo du robocar, à la place de NoMachine ?
> **Non, pas sur notre Jetson Nano.** Ce document dit pourquoi et pointe l'alternative.
>
> Contexte : issue #2 « Flux caméra ». Cible : **Jetson Nano, L4T R32.7.6** (JetPack 4,
> Ubuntu 18.04, Cortex-A57 ×4 @ 1,43 GHz, GPU Maxwell). Audit du 2026-06-17.

---

## 1. Ce qu'est Kyber

`https://gitlab.com/kyber/kyber` — pas la crypto post-quantique CRYSTALS-Kyber (faux-ami),
mais un **SDK de streaming vidéo + contrôle bidirectionnel sur QUIC**, bâti sur
**libvlc + FFmpeg**. Annoncé mi-2025 par J-B Kempf (lead VLC). Latence annoncée ~8 ms,
cas d'usage cités : cloud gaming, robotique, drones, véhicules télécommandés.

| Aspect | Constat |
|---|---|
| Stack | Rust 1.89.0 + libvlc + FFmpeg ; multi-process ; multi-dépôts `gitlab.com/kyber.stream` |
| Serveur Linux | X11/Xorg, testé Debian, build figé à `rootfs-x86_64-linux-gnu` |
| ARM documenté | macOS uniquement (`aarch64-apple-darwin`) ; aucune cible aarch64-Linux |
| Encodage matériel | NVENC / QSV / AMF — tous *desktop* ; sinon fallback CPU |
| Build deps | cmake, meson, ninja, clang, Python 3, Lua 5.4, Vulkan, libpulse |
| Maturité | Très jeune, AGPLv3, aucun portage Jetson/aarch64-Linux connu |

---

## 2. Verdict : non recommandé

Chaque ligne ❌ est un point distinct ; les 🔴/⚠️ s'ajoutent au coût.

| Critère | Note | Pourquoi |
|---|---|---|
| Buildable tel quel | ❌ | build x86_64 only ; sur Jetson c'est un **portage** (cible Rust aarch64-linux-gnu + libvlc/FFmpeg/Vulkan recompilés), jamais fait publiquement |
| Encodage matériel | ❌ | NVENC (`libnvidia-encode.so`) **absent de L4T** et non prévu par NVIDIA ; le HW encoder Nano n'est joignable que par V4L2/Tegra (`nvv4l2h264enc`), que Kyber n'utilise pas |
| Latence réelle | ❌ | sans NVENC → encodage **soft x264** sur Cortex-A57 → CPU saturé, latence qui explose : on perd l'unique raison de prendre Kyber |
| Adéquation au besoin | ⚠️ | serveur orienté capture **bureau X11**, pas une caméra ; on veut le flux **OAK-D**. Capture de source arbitraire non documentée → sinon = ce que NoMachine fait déjà |
| Friction OS | 🔴 | Ubuntu 18.04 (gcc 7, VLC 3.x) à rebours des deps récentes ; Vulkan ancien ; règle projet **jamais d'`apt upgrade`** → compilation hors apt |
| Coût total | 🔴 | R&D de plusieurs semaines, sans support amont |

**À reconsidérer si** un jour on passe à une carte plus récente **et** que Kyber publie
une cible aarch64-Linux **et** un backend d'encodage V4L2/Tegra. Pas avant.

---

## 3. La piste qui marche : GStreamer

Pour le retour caméra bas-latence vers le PC, GStreamer prend ce que Kyber rate :
encodeur **matériel** réel de la Nano (`nvv4l2h264enc`), éprouvé sur Jetson, sans toucher
au système. Deux variantes à départager ensuite :

- **(a) `nvv4l2h264enc` → RTP/UDP** — latence minimale.
- **(b) WebRTC** — visible navigateur, meilleur sur NAT/Wi-Fi.

NoMachine reste utile **seulement** pour déboguer le bureau (terminal, scripts), pas pour
le flux caméra temps réel.

---

## 4. Sources

- Kyber : https://gitlab.com/kyber/kyber · build : https://gitlab.com/kyber.stream/apps/kyber-desktop
- Présentation GIGAZINE : https://gigazine.net/gsc_news/en/20250804-vlc-kyber/
- Interview J-B Kempf : https://streaminglearningcenter.com/codecs/an-interview-with-jean-baptiste-kempf-of-kyber.html
- NVENC absent sur Tegra (NVIDIA forums) : https://forums.developer.nvidia.com/t/does-jetson-tx2-can-not-use-ffmpeg-h264-nvenc-encoder-to-encode-video/192411
- Jetson Software Encode (fallback CPU) : https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/Multimedia/SoftwareEncodeInOrinNano.html

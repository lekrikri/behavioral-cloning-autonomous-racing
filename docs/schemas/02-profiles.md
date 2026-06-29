# Profils d'usage

Combinaison *lieu de perception × intelligence*. Le profil détermine quels workers le core lance.

```mermaid
flowchart TD
    SEL["Profil sélectionné"] --> P1 & P2 & P3 & P4
    P1["P1 · masque Jetson + algo PD"] --> W1["hub + mask-worker + pd-policy"]
    P2["P2 · masque Jetson + IA NN"] --> W2["hub + mask-worker + nn-policy"]
    P3["P3 · inférence caméra + algo PD"] --> W3["hub on-cam-nn + pd-policy"]
    P4["P4 · inférence caméra + IA NN"] --> W4["hub on-cam-nn + nn-policy"]
```

| Profil | Perception | Intelligence | Charge Jetson |
|---|---|---|---|
| P1 | masque sur Jetson | algo (PD) | moyenne |
| P2 | masque sur Jetson | IA (NN Jetson) | élevée |
| P3 | inférence dans la caméra | algo (PD) | faible |
| P4 | inférence dans la caméra | IA | faible |

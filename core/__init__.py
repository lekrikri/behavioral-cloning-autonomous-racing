"""Robocar core — superviseur minimal lancé au boot.

Orchestre les workers (perception / policy / UI / manuel) selon le profil et le contexte.
Il n'implémente pas la perception ni le contrôle (ce sont les workers) ; il sélectionne et
gouverne lesquels tournent via le profil. Voir docs/CORE_DESIGN.md.
"""

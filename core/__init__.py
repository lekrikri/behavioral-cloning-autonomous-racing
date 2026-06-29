"""Robocar core — superviseur minimal lancé au boot.

Orchestre les workers (perception / policy / UI / manuel) selon le profil et le contexte.
Ne fait ni la perception ni le contrôle lui-même. Voir docs/CORE_DESIGN.md.
"""

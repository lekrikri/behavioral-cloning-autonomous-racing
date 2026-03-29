"""
input_manager.py — Capture clavier ET manette de jeu (gamepad).

La manette est FORTEMENT recommandée pour la collecte de données:
- Axes analogiques → steering/accel continus et précis
- Pas d'effet on/off du clavier → données bien meilleures
- Résultat: modèle plus fluide et plus précis

Usage manette:
    manager = GamepadManager()
    manager.start()
    steering, accel = manager.get_actions()

Usage clavier (fallback):
    manager = KeyboardManager()
    manager.start()

Usage auto (manette si dispo, clavier sinon):
    manager = create_input_manager()
    manager.start()

Contrôles clavier:
    Z / Flèche haut    → Accélérer
    S / Flèche bas     → Freiner / Reculer
    Q / Flèche gauche  → Tourner à gauche
    D / Flèche droite  → Tourner à droite
    ESPACE             → Frein d'urgence
    ESC                → Quitter

Contrôles manette (Xbox/PS layout):
    Joystick gauche X  → Steering
    Trigger droit (RT) → Accélérer
    Trigger gauche (LT)→ Freiner/Reculer
    OU
    Joystick gauche Y  → Accel/Frein
    Bouton Start/Menu  → Quitter
"""

import threading
import time
from typing import Optional, Protocol

# ─────────────────────────────────────────────
# Protocol commun
# ─────────────────────────────────────────────

class InputManagerProtocol(Protocol):
    def start(self): ...
    def stop(self): ...
    def get_actions(self) -> tuple[float, float]: ...
    def should_quit(self) -> bool: ...


# ─────────────────────────────────────────────
# Manager Manette (pygame)
# ─────────────────────────────────────────────

class GamepadManager:
    """
    Gestionnaire manette via pygame.

    Supporte Xbox, PS4/PS5, manettes génériques USB.
    La manette offre des entrées analogiques continues → bien meilleure
    qualité de données que le clavier pour le Behavioral Cloning.

    Layout par défaut (Xbox):
      axis 0 = stick gauche X  → steering
      axis 2 = trigger droit   → accélération
      axis 5 = trigger gauche  → frein
      axis 1 = stick gauche Y  → accel alternatif (mode alt)
    """

    def __init__(
        self,
        deadzone: float = 0.08,
        steer_axis: int = 0,
        accel_axis: int = 2,       # trigger droit (RT)
        brake_axis: int = 5,       # trigger gauche (LT)
        alt_accel_axis: int = 1,   # stick gauche Y (alternatif)
        use_triggers: bool = True,  # True=RT/LT | False=stick gauche Y
        quit_button: int = 7,      # Start/Options button
        invert_steer: bool = False,
    ):
        self.deadzone = deadzone
        self.steer_axis = steer_axis
        self.accel_axis = accel_axis
        self.brake_axis = brake_axis
        self.alt_accel_axis = alt_accel_axis
        self.use_triggers = use_triggers
        self.quit_button = quit_button
        self.invert_steer = invert_steer

        self._steering = 0.0
        self._acceleration = 0.0
        self._quit = False
        self._running = False
        self._joystick = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
        except ImportError:
            raise ImportError("pygame requis: pip install pygame")

        import pygame
        n = pygame.joystick.get_count()
        if n == 0:
            raise RuntimeError(
                "Aucune manette détectée! Vérifier la connexion USB/Bluetooth.\n"
                "Utiliser KeyboardManager comme fallback."
            )

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        name = self._joystick.get_name()
        n_axes = self._joystick.get_numaxes()
        n_buttons = self._joystick.get_numbuttons()
        print(f"[Gamepad] Connectée: '{name}' ({n_axes} axes, {n_buttons} boutons)")
        print(f"[Gamepad] Contrôles: Stick gauche=steering | RT=accel | LT=frein | Start=quitter")

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            import pygame
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        print("[Gamepad] Arrêtée.")

    def get_actions(self) -> tuple[float, float]:
        with self._lock:
            return round(self._steering, 4), round(self._acceleration, 4)

    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def _apply_deadzone(self, value: float) -> float:
        """Supprime le bruit autour du centre (deadzone)."""
        if abs(value) < self.deadzone:
            return 0.0
        # Normaliser pour que la deadzone soit transparente
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _update_loop(self):
        try:
            import pygame
        except ImportError:
            return

        while self._running:
            pygame.event.pump()  # traiter les events internes

            j = self._joystick

            # Steering (stick gauche X)
            steer_raw = j.get_axis(self.steer_axis) if j.get_numaxes() > self.steer_axis else 0.0
            steering = self._apply_deadzone(steer_raw)
            if self.invert_steer:
                steering = -steering

            # Accélération
            if self.use_triggers and j.get_numaxes() > max(self.accel_axis, self.brake_axis):
                # Les triggers Xbox retournent [-1, 1] au repos à -1
                # Normaliser: [-1, 1] → [0, 1]
                rt = (j.get_axis(self.accel_axis) + 1.0) / 2.0    # [0, 1]
                lt = (j.get_axis(self.brake_axis) + 1.0) / 2.0    # [0, 1]
                accel = rt - lt  # [-1, 1]: négatif=frein, positif=gaz
            else:
                # Stick gauche Y (inversé: haut = positif en pygame)
                y_raw = j.get_axis(self.alt_accel_axis) if j.get_numaxes() > self.alt_accel_axis else 0.0
                accel = self._apply_deadzone(-y_raw)  # inverser: haut = avancer

            # Bouton quitter
            quit_pressed = (
                j.get_button(self.quit_button)
                if j.get_numbuttons() > self.quit_button
                else False
            )

            with self._lock:
                self._steering = float(max(-1.0, min(1.0, steering)))
                self._acceleration = float(max(-1.0, min(1.0, accel)))
                if quit_pressed:
                    self._quit = True

            time.sleep(1.0 / 60.0)


# ─────────────────────────────────────────────
# Manager Clavier (fallback)
# ─────────────────────────────────────────────

class KeyboardManager:
    """
    Gestionnaire d'entrées clavier (ZSQD + flèches).
    Moins précis que la manette pour la collecte de données.
    """

    STEER_INCREMENT = 0.12
    STEER_DECAY = 0.08
    ACCEL_INCREMENT = 0.18
    ACCEL_DECAY = 0.12

    def __init__(self, steer_sensitivity: float = 1.0):
        try:
            from pynput import keyboard as kb
            self._kb = kb
        except ImportError:
            raise ImportError("pynput requis: pip install pynput")

        self.steer_sensitivity = steer_sensitivity
        self._left = self._right = self._fwd = self._back = self._brake = False
        self._quit = False
        self._steering = 0.0
        self._acceleration = 0.0
        self._lock = threading.Lock()
        self._listener = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._listener = self._kb.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Démarré — Z/S/Q/D ou flèches | ESPACE=frein | ESC=quitter")

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()

    def get_actions(self) -> tuple[float, float]:
        with self._lock:
            return round(self._steering, 4), round(self._acceleration, 4)

    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def _update_loop(self):
        dt = 1.0 / 60.0
        while self._running:
            with self._lock:
                left, right, fwd, back, brake = (
                    self._left, self._right, self._fwd, self._back, self._brake
                )

            if left and not right:
                steer = self._steering - self.STEER_INCREMENT * self.steer_sensitivity
            elif right and not left:
                steer = self._steering + self.STEER_INCREMENT * self.steer_sensitivity
            else:
                decay = self.STEER_DECAY
                steer = 0.0 if abs(self._steering) < decay else \
                    self._steering - decay * (1 if self._steering > 0 else -1)

            if brake:
                accel = -1.0
            elif fwd and not back:
                accel = self._acceleration + self.ACCEL_INCREMENT
            elif back and not fwd:
                accel = self._acceleration - self.ACCEL_INCREMENT
            else:
                decay = self.ACCEL_DECAY
                accel = 0.0 if abs(self._acceleration) < decay else \
                    self._acceleration - decay * (1 if self._acceleration > 0 else -1)

            with self._lock:
                self._steering = float(max(-1.0, min(1.0, steer)))
                self._acceleration = float(max(-1.0, min(1.0, accel)))

            time.sleep(dt)

    def _on_press(self, key):
        with self._lock:
            self._left = self._left or self._match(key, "q", self._kb.Key.left)
            self._right = self._right or self._match(key, "d", self._kb.Key.right)
            self._fwd = self._fwd or self._match(key, "z", self._kb.Key.up)
            self._back = self._back or self._match(key, "s", self._kb.Key.down)
            self._brake = self._brake or (key == self._kb.Key.space)
            if key == self._kb.Key.esc:
                self._quit = True

    def _on_release(self, key):
        with self._lock:
            if self._match(key, "q", self._kb.Key.left): self._left = False
            if self._match(key, "d", self._kb.Key.right): self._right = False
            if self._match(key, "z", self._kb.Key.up): self._fwd = False
            if self._match(key, "s", self._kb.Key.down): self._back = False
            if key == self._kb.Key.space: self._brake = False

    def _match(self, key, char, special) -> bool:
        try:
            return key.char == char
        except AttributeError:
            return key == special


# ─────────────────────────────────────────────
# Manager Clavier Global (Xlib — pas besoin de focus)
# ─────────────────────────────────────────────

class XlibKeyboardManager:
    """
    Capture clavier globale via Xlib XQueryKeymap.
    Lit l'état du clavier directement depuis le serveur X11 —
    aucune fenêtre ne doit avoir le focus. Fonctionne même quand
    le simulateur Unity est au premier plan.
    """

    STEER_DECAY = 0.08

    def __init__(self):
        from Xlib import display as _xdisplay, XK as _XK
        self._d = _xdisplay.Display()
        self._kc = {
            'z':     self._d.keysym_to_keycode(_XK.XK_z),
            's':     self._d.keysym_to_keycode(_XK.XK_s),
            'q':     self._d.keysym_to_keycode(_XK.XK_q),
            'd':     self._d.keysym_to_keycode(_XK.XK_d),
            'up':    self._d.keysym_to_keycode(_XK.XK_Up),
            'down':  self._d.keysym_to_keycode(_XK.XK_Down),
            'left':  self._d.keysym_to_keycode(_XK.XK_Left),
            'right': self._d.keysym_to_keycode(_XK.XK_Right),
            'esc':   self._d.keysym_to_keycode(_XK.XK_Escape),
            'space': self._d.keysym_to_keycode(_XK.XK_space),
        }
        self._steering = 0.0
        self._acceleration = 0.0
        self._quit = False

    def _read_all(self) -> dict:
        """Lit l'état de TOUTES les touches en un seul appel XQueryKeymap."""
        km = self._d.query_keymap()
        return {k: bool(km[kc // 8] & (1 << (kc % 8))) for k, kc in self._kc.items()}

    def start(self):
        print("[Keyboard] Capture GLOBALE Xlib — pas besoin de focus sur une fenêtre!")
        print("[Keyboard] Z/↑=gaz | S/↓=frein | Q/←=gauche | D/→=droite | ESC=quitter")

    def stop(self):
        self._d.close()

    def should_quit(self) -> bool:
        return self._quit

    def get_actions(self) -> tuple:
        keys = self._read_all()
        fwd   = keys['z'] or keys['up']
        back  = keys['s'] or keys['down']
        left  = keys['q'] or keys['left']
        right = keys['d'] or keys['right']
        brake = keys['space']

        if keys['esc']:
            self._quit = True

        if left and not right:
            self._steering = -1.0
        elif right and not left:
            self._steering = 1.0
        else:
            if abs(self._steering) < self.STEER_DECAY:
                self._steering = 0.0
            elif self._steering > 0:
                self._steering -= self.STEER_DECAY
            else:
                self._steering += self.STEER_DECAY

        if fwd and not back:
            self._acceleration = 1.0
        elif back and not fwd:
            self._acceleration = -0.5
        elif brake:
            self._acceleration = 0.0
        else:
            if self._acceleration > 0.05:
                self._acceleration -= 0.05
            elif self._acceleration < -0.05:
                self._acceleration += 0.05
            else:
                self._acceleration = 0.0

        return round(self._steering, 4), round(self._acceleration, 4)


# ─────────────────────────────────────────────
# Manager Clavier Pygame (fenêtre de contrôle)
# ─────────────────────────────────────────────

class TerminalKeyboardManager:
    """
    Capture clavier via pygame (thread principal).
    get_actions() pompe les events pygame à chaque step — appelé depuis la boucle principale.
    Une petite fenêtre "Contrôles" s'ouvre : cliquer dessus puis Z/S/Q/D.
    """

    STEER_DECAY = 0.08

    def __init__(self):
        import pygame as _pg
        self._pg = _pg
        self._quit = False
        self._steering = 0.0
        self._acceleration = 0.0
        self._screen = None
        self._font = None

    def start(self):
        """Appelé depuis le thread principal — crée la fenêtre pygame."""
        pg = self._pg
        if not pg.get_init():
            pg.init()
        pg.display.set_caption("Contrôles — Z/S/Q/D | ESC quitter")
        self._screen = pg.display.set_mode((300, 80))
        self._font = pg.font.SysFont("monospace", 14)
        print("[Keyboard] Fenêtre Contrôles ouverte — cliquer dessus puis Z/S/Q/D | ESC=quitter")

    def stop(self):
        pass

    def get_actions(self) -> tuple:
        """Pompe les events pygame et retourne (steering, acceleration). Thread principal."""
        pg = self._pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._quit = True
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self._quit = True

        keys = pg.key.get_pressed()
        fwd   = keys[pg.K_z] or keys[pg.K_UP]
        back  = keys[pg.K_s] or keys[pg.K_DOWN]
        left  = keys[pg.K_q] or keys[pg.K_LEFT]
        right = keys[pg.K_d] or keys[pg.K_RIGHT]
        brake = keys[pg.K_SPACE]

        if left and not right:
            self._steering = -0.8
        elif right and not left:
            self._steering = 0.8
        else:
            if abs(self._steering) < self.STEER_DECAY:
                self._steering = 0.0
            elif self._steering > 0:
                self._steering -= self.STEER_DECAY
            else:
                self._steering += self.STEER_DECAY

        if fwd and not back:
            self._acceleration = 1.0
        elif back and not fwd:
            self._acceleration = -0.5
        elif brake:
            self._acceleration = 0.0
        else:
            if self._acceleration > 0.05:
                self._acceleration -= 0.05
            elif self._acceleration < -0.05:
                self._acceleration += 0.05
            else:
                self._acceleration = 0.0

        # Affichage
        if self._screen:
            screen = self._screen
            screen.fill((30, 30, 30))
            txt  = self._font.render(f"Steer:{self._steering:+.2f}  Accel:{self._acceleration:+.2f}", True, (200, 255, 200))
            hint = self._font.render("Z/S=gaz/frein  Q/D=virage  ESC=fin", True, (150, 150, 150))
            screen.blit(txt,  (10, 10))
            screen.blit(hint, (10, 40))
            pg.display.flip()

        return round(self._steering, 4), round(self._acceleration, 4)

    def should_quit(self) -> bool:
        return self._quit


# ─────────────────────────────────────────────
# Factory auto-détection
# ─────────────────────────────────────────────

def create_input_manager(prefer_gamepad: bool = True, **kwargs):
    """
    Crée automatiquement le meilleur gestionnaire disponible.

    1. Essaie la manette (pygame) si prefer_gamepad=True
    2. Fallback sur le clavier (pynput)

    La manette est fortement recommandée pour la collecte de données!
    """
    if prefer_gamepad:
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                mgr = GamepadManager(**kwargs)
                print("[Input] Manette détectée → GamepadManager sélectionné")
                return mgr
            else:
                print("[Input] Aucune manette → fallback clavier")
        except ImportError:
            print("[Input] pygame non installé → fallback clavier")

    # Capture globale Xlib (pas besoin de focus)
    try:
        mgr = XlibKeyboardManager()
        print("[Input] Capture globale Xlib → clique nulle part, appuie juste sur Z!")
        return mgr
    except Exception as e:
        print(f"[Input] Xlib indisponible ({e}) → fallback pygame")

    # Fallback: clavier via pygame (fenêtre dédiée)
    print("[Input] Clavier pygame → fenêtre Contrôles (cliquer dessus pour le focus)")
    return TerminalKeyboardManager()


# ─────────────────────────────────────────────
# Mock pour tests sans hardware
# ─────────────────────────────────────────────

class MockInputManager:
    """Gestionnaire simulé pour les tests automatisés."""

    def __init__(self, pattern: str = "sine"):
        self._pattern = pattern
        self._t = 0

    def start(self): pass
    def stop(self): pass
    def should_quit(self) -> bool: return False

    def get_actions(self) -> tuple[float, float]:
        self._t += 1
        if self._pattern == "straight":
            return 0.0, 0.8
        elif self._pattern == "sine":
            import math
            return math.sin(self._t * 0.04) * 0.4, 0.7
        elif self._pattern == "left_turn":
            return -0.5, 0.6
        return 0.0, 0.0


# ─────────────────────────────────────────────
# Test standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamepad", action="store_true", help="Forcer manette")
    parser.add_argument("--keyboard", action="store_true", help="Forcer clavier")
    args = parser.parse_args()

    if args.gamepad:
        manager = GamepadManager()
    elif args.keyboard:
        manager = KeyboardManager()
    else:
        manager = create_input_manager()

    manager.start()
    print("Test input (Ctrl+C pour arrêter)\n")

    try:
        while not manager.should_quit():
            steering, accel = manager.get_actions()
            bar_s = "#" * int(abs(steering) * 20)
            bar_a = "#" * int(max(0, accel) * 20)
            side = "L" if steering < -0.02 else ("R" if steering > 0.02 else "=")
            print(
                f"\r  Steer [{side}] {bar_s:<20}  |  Accel {bar_a:<20}  "
                f"({steering:+.3f}, {accel:+.3f})    ",
                end="", flush=True
            )
            time.sleep(0.033)  # ~30Hz affichage
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
    print()

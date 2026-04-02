"""
vesc_interface.py — Interface Python pour Flipsky FSESC Mini V6.7 Pro (VESC).

Commandes :
  steering ∈ [-1, 1] → position servo [0.0, 1.0]
  accel    ∈ [0, 1]  → duty cycle moteur [0.0, duty_max]

⚠️ SÉCURITÉ :
  - duty_max = 0.15 pour les premiers tests (15% de puissance)
  - Configurer le watchdog hardware dans VESC Tool :
      App Settings → General → Timeout = 200ms
  - Vérifier le sens du servo (peut être inversé selon montage)
"""

import time
import numpy as np

try:
    import pyvesc
    import serial
    _PYVESC_AVAILABLE = True

    # SetServoPos absent de certaines versions de pyvesc — on l'ajoute (VESC cmd ID=11)
    if not hasattr(pyvesc, "SetServoPos"):
        from pyvesc.messages.base import VESCMessage
        class SetServoPos(metaclass=VESCMessage):
            """Set servo position (0.0 to 1.0)."""
            id = 11
            fields = [("servo_pos", "H", 1000)]
        pyvesc.SetServoPos = SetServoPos

except ImportError:
    _PYVESC_AVAILABLE = False
    print("[VESC] pyvesc non installé — mode simulation activé")


class VESCInterface:
    """
    Interface VESC pour Flipsky FSESC Mini V6.7 Pro.

    Paramètres à calibrer :
      servo_center  : position neutre du servo (0.5 = centre, ajuster si déviation)
      servo_range   : amplitude ±X autour du centre (0.35 par défaut, ajuster mécaniquement)
      duty_max      : puissance max moteur (0.15 = 15% pour premiers tests)
      invert_steer  : inverser le sens du servo si nécessaire
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        servo_center: float = 0.5,
        servo_range: float = 0.35,   # ±0.35 autour du centre → à ajuster
        duty_max: float = 0.15,      # 15% pour tests initiaux — monter progressivement
        invert_steer: bool = False,  # inverser si le servo tourne dans le mauvais sens
    ):
        self.servo_center = servo_center
        self.servo_range  = servo_range
        self.duty_max     = duty_max
        self.invert_steer = invert_steer
        self._sim_mode    = not _PYVESC_AVAILABLE

        if not self._sim_mode:
            try:
                self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.05)
                print(f"[VESC] Connecté sur {port}")
            except serial.SerialException as e:
                print(f"[VESC] ⚠️  Impossible d'ouvrir {port} : {e}")
                print("[VESC] Mode simulation activé")
                self._sim_mode = True
        else:
            self.ser = None

    def send(self, steering: float, accel: float) -> None:
        """
        Envoyer commandes de direction et accélération.
        steering ∈ [-1, 1] | accel ∈ [0, 1]
        """
        steering = float(np.clip(steering, -1.0, 1.0))
        accel    = float(np.clip(accel,    0.0,  1.0))

        if self.invert_steer:
            steering = -steering

        # Mapping steering → servo [0, 1]
        servo_pos = self.servo_center + self.servo_range * steering
        servo_pos = float(np.clip(servo_pos, 0.10, 0.90))  # sécurité mécanique

        # Mapping accel → duty cycle
        duty = accel * self.duty_max
        duty = float(np.clip(duty, 0.0, self.duty_max))

        if self._sim_mode:
            print(f"\r[VESC SIM] steer={steering:+.3f} → servo={servo_pos:.3f} | "
                  f"accel={accel:.3f} → duty={duty:.3f}   ", end="", flush=True)
            return

        try:
            self.ser.write(pyvesc.encode(pyvesc.SetServoPos(servo_pos)))       # scale=1000 → float OK
            self.ser.write(pyvesc.encode(pyvesc.SetDutyCycle(int(duty * 1e5)))) # no scale → int requis
        except Exception as e:
            print(f"[VESC] Erreur envoi commande : {e}")
            self.stop()

    def stop(self) -> None:
        """Arrêt d'urgence — moteur à 0, servo centré."""
        if not self._sim_mode and self.ser and self.ser.is_open:
            try:
                self.ser.write(pyvesc.encode(pyvesc.SetDutyCycle(0)))
                self.ser.write(pyvesc.encode(pyvesc.SetServoPos(self.servo_center)))
            except Exception:
                pass
        print("\n[VESC] ⛔ ARRÊT D'URGENCE")

    def get_rpm(self) -> float:
        """Lire le RPM moteur (vitesse réelle via encodeur)."""
        if self._sim_mode or not self.ser or not self.ser.is_open:
            return 0.0
        try:
            self.ser.write(pyvesc.encode_request(pyvesc.GetValues))
            time.sleep(0.005)
            raw = self.ser.read(100)
            if raw:
                msg, _ = pyvesc.decode(raw)
                if isinstance(msg, pyvesc.GetValues):
                    return float(getattr(msg, "rpm", 0.0))
        except Exception:
            pass
        return 0.0

    def close(self) -> None:
        """Fermer proprement la connexion."""
        self.stop()
        if getattr(self, "ser", None) and self.ser.is_open:
            self.ser.close()
        print("[VESC] Connexion fermée.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

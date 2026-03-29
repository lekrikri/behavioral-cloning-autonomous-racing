"""Test TCP brut : Unity peut-il se connecter à localhost:5004 ?"""
import socket
import threading
import subprocess
import sys
import time
from pathlib import Path

PORT = 5004
SIM = str(Path(__file__).parent / "BuildLinux" / "RacingSimulator.x86_64")
CONFIG = str(Path(__file__).parent / "config.json")

def tcp_server():
    for af, addr in [(socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")]:
        try:
            s = socket.socket(af, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, PORT))
            s.listen(5)
            s.settimeout(60)
            print(f"[TCP] Écoute sur {addr}:{PORT} (AF={'INET' if af==socket.AF_INET else 'INET6'})")
            try:
                conn, addr_from = s.accept()
                print(f"[TCP] CONNEXION REÇUE de {addr_from}!")
                data = conn.recv(1024)
                print(f"[TCP] {len(data)} octets reçus: {data[:50]!r}")
                conn.close()
            except socket.timeout:
                print(f"[TCP] Timeout — aucune connexion sur {addr}:{PORT}")
            finally:
                s.close()
            break
        except OSError as e:
            print(f"[TCP] Impossible de binder {addr}:{PORT}: {e}")

print("[INFO] Démarrage serveur TCP...")
t = threading.Thread(target=tcp_server, daemon=True)
t.start()

time.sleep(1)
print(f"[INFO] Lancement simulateur avec --mlagents-port {PORT}...")
proc = subprocess.Popen(
    [SIM, f"--mlagents-port", str(PORT), f"--config-path", CONFIG, "-nographics", "-batchmode"],
    start_new_session=True,
)

t.join(timeout=90)
proc.terminate()
print("[INFO] Test terminé.")

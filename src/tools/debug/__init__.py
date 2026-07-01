"""Debug web interface — operator + tuning UI served by the hub client.

Pages:
  /        home: load a perception profile + Play / Pause / Stop the drive pipeline
  /mask    mask calibration: live colour + mask overlay + polar fan + filter sliders

Stdlib only (http.server + MJPEG). Reads the camera hub over SHM as a client; never
opens the OAK-D. Access from the PC via `ssh -L <port>:localhost:<port> robocar`.
"""

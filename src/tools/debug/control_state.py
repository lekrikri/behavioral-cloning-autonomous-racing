"""État de réglage du cerveau (onglet Intelligence) : sélection du cerveau + params du
contrôleur réactif. Analogue à MaskState, mais le cerveau tourne dans un sous-process
séparé (il détient le VESC) : chaque changement de slider est donc poussé dans le canal
live (`src.control.live_params`) que le sous-process d'inférence relit à chaud.

Le choix du cerveau (model/reactive), lui, est structurel (ONNX vs algo chargés à l'init)
→ il prend effet au prochain Play, pas à chaud.
"""

import threading

from src.control.live_params import write_live
from src.mask.perception_config import (
    PerceptionProfile, list_profiles, load_profile, profiles_dir)


# Params réactifs exposés en sliders : (nom de champ profil, min, max, step).
CONTROL_SLIDERS = [
    ("ctrl_kp",             0.0,  3.0,  0.05),
    ("ctrl_kd",             0.0,  1.0,  0.01),
    ("ctrl_forward_sigma",  1.0,  12.0, 0.5),
    ("ctrl_v_max",          0.0,  1.0,  0.05),
    ("ctrl_v_min",          0.0,  1.0,  0.05),
    ("ctrl_turn_slowdown",  0.0,  1.0,  0.05),
    ("ctrl_steer_max",      0.1,  1.0,  0.05),
    ("ctrl_steer_deadzone", 0.0,  0.2,  0.01),
    ("ctrl_steer_rate_max", 0.05, 1.0,  0.05),
    ("ctrl_heading_ema",    0.0,  0.9,  0.05),
]
_NAMES = [s[0] for s in CONTROL_SLIDERS]
_PREFIX = "ctrl_"


class ControlState:
    def __init__(self, profile: PerceptionProfile):
        self.lock = threading.Lock()
        self.profile = profile
        self.brain = profile.brain
        self.params = {n: getattr(profile, n) for n in _NAMES}
        self._push_live()

    def _payload(self):
        """Params au format ReactiveConfig (préfixe ctrl_ retiré)."""
        return {n[len(_PREFIX):]: v for n, v in self.params.items()}

    def _push_live(self):
        write_live(self._payload())

    def set_param(self, name, value):
        with self.lock:
            if name not in self.params:
                return "unknown:%s" % name
            cur = self.params[name]
            self.params[name] = type(cur)(float(value))
            self._push_live()          # réglage à chaud : le sous-process relira au prochain poll
            return "%s=%s" % (name, self.params[name])

    def set_brain(self, kind):
        if kind not in ("model", "reactive"):
            return "unknown brain:%s" % kind
        with self.lock:
            self.brain = kind
            self.profile.brain = kind
            # Persiste tout de suite : Driver.play() relit le cerveau depuis le JSON du profil,
            # pas depuis cet état mémoire. Sans ça, le choix serait ignoré silencieusement au Play.
            self.profile.save(profiles_dir() / (self.profile.name + ".json"))
        return "cerveau=%s persisté — prend effet au prochain Play" % kind

    def load_profile_into(self, name):
        try:
            prof = load_profile(name)
        except (FileNotFoundError, OSError):
            return "profil introuvable: %s" % name
        with self.lock:
            self.profile = prof
            self.brain = prof.brain
            self.params = {n: getattr(prof, n) for n in _NAMES}
            self._push_live()
        return "profil chargé: %s" % name

    def save_to_profile(self):
        with self.lock:
            prof = self.profile
            for n, v in self.params.items():
                setattr(prof, n, v)
            prof.brain = self.brain
            name = prof.name
        prof.save(profiles_dir() / (name + ".json"))
        return "profil '%s' sauvegardé (cerveau=%s)" % (name, self.brain)

    def ui_state(self):
        with self.lock:
            d = dict(self.params)
            d["_brain"] = self.brain
            d["_profile"] = self.profile.name
        d["_profiles"] = list_profiles() or [self.profile.name]
        return d


def _selftest():
    """Garde-fou : la liste des params est miroir dans 3 fichiers (ReactiveConfig, champs
    ctrl_ du profil, CONTROL_SLIDERS). On échoue ICI (au test) si un miroir dérive, plutôt
    que silencieusement au volant (slider mort / param non persistable)."""
    from dataclasses import fields
    from src.control.reactive_controller import ReactiveConfig
    prof_fields = {f.name for f in fields(PerceptionProfile)}
    rc_fields = {f.name for f in fields(ReactiveConfig)}
    for name in _NAMES:
        assert name in prof_fields, "slider %s sans champ profil" % name
        assert name[len(_PREFIX):] in rc_fields, "slider %s absent de ReactiveConfig" % name
    for f in prof_fields:
        if f.startswith(_PREFIX):
            assert f in _NAMES, "champ profil %s sans slider (param muet)" % f
    assert (set(PerceptionProfile().control_kwargs()) - {"n_rays"}) <= rc_fields
    print("control_state self-test OK (%d sliders <-> profil <-> ReactiveConfig)" % len(_NAMES))


if __name__ == "__main__":
    _selftest()

"""The twelve scenes of the reel.  Each scene is built lazily (painting its assets once, cached on disk) and
renders a float RGB frame for any absolute time inside its window."""
import importlib

from timeline import SCENES

_MODULES = {
    'hook': 'k01_hook', 'record': 'k02_record', 'oracle': 'k03_oracle', 'objects': 'k04_objects', 'sun': 'k05_sun',
    'swirl': 'k06_swirl', 'nature': 'k07_nature', 'compound': 'k08_compound', 'map': 'k10_map',
    'scholar': 'k11_scholar', 'message': 'k12_message', 'endcard': 'k13_endcard',
}
_built = {}


def get(name):
    if name not in _built:
        t0, t1 = next((a, b) for n, a, b in SCENES if n == name)
        mod = importlib.import_module('scenes.' + _MODULES[name])
        _built[name] = mod.Scene(t0, t1)
    return _built[name]

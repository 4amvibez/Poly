"""The eight scenes of the reel.  Each scene is built lazily (painting its assets once, cached on disk) and renders
a float RGB frame for any absolute time inside its window."""
import importlib

from timeline import SCENES

_MODULES = {
    'hook': 's01_hook', 'scroll': 's02_scroll', 'confusion': 's03_confusion', 'alike': 's04_alike', 'niwa': 's05_niwa',
    'answer': 's06_answer', 'obstacle': 's07_obstacle', 'endcard': 's08_endcard',
}
_built = {}


def get(name):
    if name not in _built:
        t0, t1 = next((a, b) for n, a, b in SCENES if n == name)
        mod = importlib.import_module('scenes.' + _MODULES[name])
        _built[name] = mod.Scene(t0, t1)
    return _built[name]

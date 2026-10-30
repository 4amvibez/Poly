"""The seven scenes of the reel.  Each scene is built lazily (painting its assets once, cached on disk) and renders
a float RGB frame for any absolute time inside its window."""
import importlib

from timeline import SCENES

_MODULES = {
    'hook': 's01_hook', 'secret': 's02_secret', 'work': 's03_work', 'pass': 's04_pass', 'field': 's05_field',
    'more': 's06_more', 'final': 's08_final',
}
_built = {}


def get(name):
    if name not in _built:
        t0, t1 = next((a, b) for n, a, b in SCENES if n == name)
        mod = importlib.import_module('scenes.' + _MODULES[name])
        _built[name] = mod.Scene(t0, t1)
    return _built[name]

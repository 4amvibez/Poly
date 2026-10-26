"""The eight scenes of the reel.  Each scene is built lazily (painting its assets once, cached on disk) and renders
a float RGB frame for any absolute time inside its window."""
import importlib

from timeline import SCENES

_MODULES = {
    'hook': 'r01_hook', 'spaces': 'r02_spaces', 'kanji': 'r03_kanji', 'hiragana': 'r04_hiragana',
    'katakana': 'r05_katakana', 'secret': 'r06_secret', 'endless': 'r07_endless', 'endcard': 'r08_endcard',
}
_built = {}


def get(name):
    if name not in _built:
        t0, t1 = next((a, b) for n, a, b in SCENES if n == name)
        mod = importlib.import_module('scenes.' + _MODULES[name])
        _built[name] = mod.Scene(t0, t1)
    return _built[name]

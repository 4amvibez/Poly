"""The ten scenes of the reel.  Each scene is built lazily (painting its assets once, cached on disk) and
renders a float RGB frame for any absolute time inside its window."""
import importlib

from timeline import SCENES

_MODULES = {
    'hook': 's01_hook', 'before_writing': 's02_before', 'map': 's03_map', 'kanji': 's04_kanji',
    'hiragana': 's05_hiragana', 'katakana': 's06_katakana', 'triptych': 's07_triptych',
    'sentence': 's08_sentence', 'street': 's09_street', 'modern': 's10_modern', 'endcard': 's11_endcard',
}
_built = {}


def get(name):
    if name not in _built:
        t0, t1 = next((a, b) for n, a, b in SCENES if n == name)
        mod = importlib.import_module('scenes.' + _MODULES[name])
        _built[name] = mod.Scene(t0, t1)
    return _built[name]

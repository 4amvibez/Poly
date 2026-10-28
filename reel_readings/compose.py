"""compose — one finished frame at any time: active scene(s), transition, subtitles, grade."""
import numpy as np

import scenes
from engine import Frame
from timeline import SCENES

_exit_maps = {}


def frame_at(t, only=None, subtitles=True):
    active = [(n, a, b) for n, a, b in SCENES if a <= t < b or (n == SCENES[-1][0] and t >= a)]
    if only:
        active = [x for x in active if x[0] == only] or [next(x for x in SCENES if x[0] == only)]
    if len(active) == 1:
        img = scenes.get(active[0][0]).render(t)
    else:
        (pn, pa, pb), (cn, ca, cb) = active[0], active[1]
        prev, cur = scenes.get(pn), scenes.get(cn)
        f = (t - ca) / (pb - ca)
        img = cur.enter(prev.render(t), cur.render(t), f, prev)
    fr = Frame(img)
    if subtitles:
        import subtitles as subs
        subs.draw(fr, t)
    return fr.finish(t)

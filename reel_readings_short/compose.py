"""compose — one finished frame at any time: active scene(s), transition, subtitles, grade."""
import numpy as np

import scenes
from engine import Frame
from timeline import SCENES

_exit_maps = {}


def drift(img, t):
    """a slow hand-held breath over every frame (a gentle push and sway), so the page never sits still"""
    import cv2
    from engine import H, W
    z = 1.018 + 0.008 * np.sin(2 * np.pi * t / 6.5)
    dx, dy = 7 * np.sin(2 * np.pi * t / 9.0), 6 * np.cos(2 * np.pi * t / 11.0)
    a = 0.002 * np.sin(2 * np.pi * t / 13.0)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), np.degrees(a), z)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


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
    fr = Frame(drift(img, t))
    if subtitles:
        import subtitles as subs
        subs.draw(fr, t)
    return fr.finish(t)

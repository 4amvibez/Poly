"""Characters written with the poly_ink bristle brush, stroke by stroke, in the correct stroke order.

Stroke centrelines and order come from KanjiVG (assets/kanjivg, CC BY-SA 3.0, Ulrich Apel and contributors,
https://kanjivg.tagaini.net). Each stroke's KanjiVG type (㇐ horizontal, ㇒ left sweep, ㇏ right sweep, ㇚ hook…)
picks the brush pressure profile, so sweeps lift into a sharp tip and hooks flick, the way they are written.
"""
import os
import re

import numpy as np

import poly_ink  # noqa: F401  (puts the claude-drawing lib on sys.path)
from poly_ink.brush import HOOK, PRESS_DOWN, brush_stroke

HERE = os.path.dirname(os.path.abspath(__file__))
KVG = os.path.join(HERE, 'assets', 'kanjivg')

HORIZ = ((0, 0.8), (0.07, 1.0), (0.8, 0.9), (0.93, 1.06), (1, 0.72))     # press in, run, press to stop (止め)
VERT = ((0, 0.85), (0.06, 1.0), (0.88, 0.94), (1, 0.7))
NA = ((0, 0.4), (0.2, 0.75), (0.82, 1.18), (0.92, 0.9), (1, 0.12))      # 捺: thin entry, heavy foot, out
DOT = ((0, 0.35), (0.35, 1.0), (0.8, 1.1), (1, 0.6))
RISE = ((0, 0.95), (0.15, 1.0), (1, 0.2))
KANA = ((0, 0.55), (0.12, 1.0), (0.72, 0.86), (1, 0.32))                 # kana: soft entry, tapering exit
PROFILES = {'㇐': HORIZ, '㇑': VERT, '㇒': PRESS_DOWN, '㇓': PRESS_DOWN, '㇏': NA, '㇔': DOT, '㇀': RISE,
            '㇚': HOOK, '㇆': HOOK, '㇟': HOOK, '㇂': HOOK, '㇁': HOOK, '㇉': HOOK, '㇈': HOOK, '㇌': HOOK,
            '㇋': HOOK, '㇊': HOOK, '㇃': HOOK, '㇇': PRESS_DOWN, '㇛': PRESS_DOWN, '㇜': PRESS_DOWN}


def _path_points(d, per=10):
    """SVG path data -> list of polylines (M, L, H, V, C, S, Z; absolute and relative)"""
    toks = re.findall(r'[MmLlHhVvCcSsZz]|-?\d*\.?\d+(?:e-?\d+)?', d)
    i, cmd, cur, start, prev_c2 = 0, None, np.zeros(2), np.zeros(2), None
    pts = []

    def num():
        nonlocal i
        i += 1
        return float(toks[i - 1])

    while i < len(toks):
        if re.match(r'[A-Za-z]', toks[i]):
            cmd = toks[i]
            i += 1
        rel = cmd.islower()
        c = cmd.upper()
        base = cur if rel else np.zeros(2)
        if c == 'M':
            cur = base + (num(), num())
            start = cur.copy()
            pts.append(cur.copy())
            cmd = 'l' if rel else 'L'
            prev_c2 = None
        elif c == 'L':
            cur = base + (num(), num())
            pts.append(cur.copy())
            prev_c2 = None
        elif c == 'H':
            cur = np.array([(cur[0] if rel else 0) + num(), cur[1]])
            pts.append(cur.copy())
        elif c == 'V':
            cur = np.array([cur[0], (cur[1] if rel else 0) + num()])
            pts.append(cur.copy())
        elif c in 'CS':
            if c == 'C':
                c1 = base + (num(), num())
            else:
                c1 = 2 * cur - prev_c2 if prev_c2 is not None else cur.copy()
            c2 = base + (num(), num())
            end = base + (num(), num())
            t = np.linspace(0, 1, per + 1)[1:, None]
            pts.extend(list((1 - t) ** 3 * cur + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * end))
            prev_c2, cur = c2, end
        elif c == 'Z':
            cur = start.copy()
            pts.append(cur.copy())
    return np.array(pts, np.float32)


def resample(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(3, int(s[-1] / step) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])], 1)


_cache = {}


def strokes_of(ch):
    """[(points in KanjiVG's 109x109 box, stroke type)] in stroke order"""
    if ch not in _cache:
        path = os.path.join(KVG, f'{ord(ch):05x}.svg')
        svg = open(path, encoding='utf-8').read()
        out = []
        for m in re.finditer(r'<path[^>]*?(?:kvg:type="([^"]*)")?[^>]*?\sd="([^"]+)"', svg):
            out.append((_path_points(m.group(2)), m.group(1) or ''))
        _cache[ch] = out
    return _cache[ch]


class Written:
    """one brush stroke placed on a timeline: the Stroke, when it starts and ends (s), its path in px"""

    def __init__(self, stroke, t0, t1, path, red=False):
        self.stroke, self.t0, self.t1, self.path, self.red = stroke, t0, t1, path, red
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-6)

    def tip(self, t):
        """brush tip position and direction at time t (None when the brush is not on this stroke)"""
        if not (self.t0 <= t <= self.t1):
            return None
        f = (t - self.t0) / max(self.t1 - self.t0, 1e-6)
        x, y = np.interp(f, self.s, self.path[:, 0]), np.interp(f, self.s, self.path[:, 1])
        return x, y


def write(W, H, ch, x, y, size, t0, ink=0.95, dry=0.3, weight=0.1, speed=1.0, gap=0.09, seed=0, red=(),
          wet=0.25, wobble=0.05, only=None, all_red=False):
    """Brush-write `ch` into a size x size box with top-left (x, y), starting at t0 seconds.
    weight: brush width as a fraction of the box. speed: >1 writes faster. red: stroke indices painted in
    vermilion. Returns ([Written], end time)."""
    k = size / 109.0
    t, out = t0, []
    for i, (P, typ) in enumerate(strokes_of(ch)):
        if only is not None and i not in only:
            continue
        if all_red:
            red = set(red) | {i}
        prof = PROFILES.get(typ[:1], KANA if not typ else HORIZ)
        Q = resample(P * k + (x, y), max(3.0, size * 0.035))
        if typ[:1] in '㇐㇑㇕㇖㇆㇚㇄㇗㇙':
            # 起筆: the brush lands at an angle, up and to the left of the line, then turns into it
            Q = np.vstack([Q[0] + np.float32(size * 0.028) * np.array([-1.0, -1.0], np.float32), Q])
        L = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
        w = size * weight * (0.85 if prof is DOT else 1.0)
        st = brush_stroke(W, H, [tuple(p) for p in Q], w, ink=ink, dry=dry, profile=prof, smooth=3,
                          seed=seed * 131 + i, wet=wet, wobble=wobble, red=i in red)
        dur = (0.12 + L / size * 0.42) / speed
        if st is not None:
            out.append(Written(st, t, t + dur, Q, red=i in red))
        t += dur + gap / speed
    return out, t


def write_text(W, H, text, x, y, size, t0, vertical=False, spacing=1.05, seed=0, pause=0.12, **kw):
    """write several characters in a row (or a column with vertical=True); returns ([Written], end time)"""
    allw, t = [], t0
    for j, ch in enumerate(text):
        cx, cy = (x, y + j * size * spacing) if vertical else (x + j * size * spacing, y)
        ws, t = write(W, H, ch, cx, cy, size, t, seed=seed + j * 17, **kw)
        allw += ws
        t += pause
    return allw, t

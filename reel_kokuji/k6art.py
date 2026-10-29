"""k6art — the paintings of the kokuji reel, in the same brush as the rest (poly_ink strokes and washes).
Each returns float32 densities on the full 1080x1920 page unless it says otherwise."""
import numpy as np

from art import blossom_cluster, cherry_branch
from engine import H, W, StrokeSet, blur, noise, spline, stroke
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from r5art import ronin


def _f(a):
    return np.clip(np.asarray(a, np.float32), 0, 1)


def torii(x=250, y=1230, s=1.0):
    """a vermilion torii, posts on the ground at y: two posts, the curved kasagi on top, the nuki under it
    (returns the vermilion density; the dark cap of the kasagi is in 'ink')"""
    red, ink = StrokeSet(W, H), StrokeSet(W, H)
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    for dx in (-120, 120):
        red.add(stroke(W, H, P([(dx, -400), (dx * 1.04, -200), (dx * 1.08, 0)]), 26 * s, ink=0.95, dry=0.35, profile=PRESS_DOWN, seed=11 + (dx > 0)), 0, 1)
    red.add(stroke(W, H, spline(P([(-210, -395), (0, -385), (210, -395)]), 12), 30 * s, ink=0.95, dry=0.3, profile=SWEEP, seed=13), 0, 1)
    red.add(stroke(W, H, P([(-165, -320), (165, -320)]), 20 * s, ink=0.95, dry=0.35, profile=SWEEP, seed=14), 0, 1)
    ink.add(stroke(W, H, spline(P([(-235, -418), (-100, -420), (0, -412), (100, -420), (235, -418)]), 16), 22 * s, ink=0.95, dry=0.3,
                   profile=SWEEP, seed=15), 0, 1)
    ink.add(stroke(W, H, P([(0, -395), (0, -322)]), 14 * s, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=16), 0, 1)
    return {'red': _f(red.render(9)), 'ink': _f(ink.render(9))}


def blossom_branch():
    """a cherry branch reaching in from the top right, with its blossom (ink, blossom)"""
    ss = StrokeSet(W, H)
    centres = cherry_branch(ss, [(1120, 170), (960, 230), (820, 330), (700, 360), (610, 430)], 22, 0, 1, seed=31)
    bl = np.zeros((H, W), np.float32)
    for k, (cx, cy) in enumerate(centres):
        bl = np.maximum(bl, blossom_cluster(H, W, cx, cy, 6, 30, seed=40 + k))
    return {'ink': _f(ss.render(9)), 'bloss': _f(bl)}


def mist_hills(y0=1050, seed=3):
    """three ranges of misty hills, palest at the back"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.zeros((H, W), np.float32)
    for k, (base, amp, op) in enumerate(((y0 - 120, 90, 0.25), (y0, 70, 0.4), (y0 + 130, 55, 0.6))):
        r = np.random.default_rng(seed + k)
        top = base - amp * (0.6 * np.sin(xx[0] / (170 + 40 * k) + r.uniform(0, 6)) + 0.4 * np.sin(xx[0] / 61 + r.uniform(0, 6)))
        m = np.clip((yy - top[None, :]) / 60, 0, 1) * np.clip(1 - (yy - top[None, :]) / 520, 0, 1) ** 0.8
        d = np.maximum(d, m * op * (0.85 + 0.15 * noise(H, W, 90, 3, seed + 10 + k)))
    return {'ink': _f(blur(d, 2))}


def fields():
    """rows of crops running away to a low horizon, a farmhouse and two trees on it"""
    ss = StrokeSet(W, H)
    hy, vx = 1040, 540                                          # horizon, vanishing point
    r = np.random.default_rng(5)
    for k in range(-7, 8):
        x0 = vx + k * 150
        pts = [(vx + k * 12, hy + 10), (x0, 1540)]
        ss.add(stroke(W, H, pts, 3, ink=0.45, dry=0.7, profile=PRESS_DOWN, seed=60 + k), 0, 1)
        for j in range(12):                                     # tufts along the row, bigger nearer
            u = (j + 0.5) / 12
            x, y = vx + k * 12 + (x0 - vx - k * 12) * u ** 1.6, hy + 10 + (1530 - hy) * u ** 1.6
            sz = 4 + 22 * u ** 1.6
            for a in (-0.6, 0, 0.6):
                ss.add(stroke(W, H, [(x, y), (x + sz * np.sin(a) * 0.8, y - sz * np.cos(a))], max(1.5, sz / 7), ink=0.75, dry=0.5,
                              profile=SWEEP, seed=int(r.integers(1e6))), 0, 1)
    # the farmhouse and trees on the horizon
    ss.add(pwash(W, H, [(700, hy), (700, hy - 46), (760, hy - 84), (820, hy - 46), (820, hy)], ink=0.85, soft=1.5, seed=71), 0, 1)
    ss.add(pwash(W, H, [(680, hy - 40), (760, hy - 96), (840, hy - 40)], ink=0.95, soft=1.5, seed=72), 0, 1)
    for tx in (300, 900):
        ss.add(stroke(W, H, [(tx, hy), (tx + 4, hy - 70)], 6, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=tx), 0, 1)
        ss.add(pwash(W, H, [(tx + 50 * np.cos(a), hy - 95 + 32 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)],
                     ink=0.7, soft=3, seed=tx + 1, grain=0.4), 0, 1)
    return {'ink': _f(ss.render(9))}


def flame(w=260, h=340):
    """a brushed flame (vermilion body, ink tongues) on its own small canvas"""
    red, ink = StrokeSet(w, h), StrokeSet(w, h)
    cx = w / 2
    red.add(pwash(w, h, [(cx - 90, h - 30), (cx - 60, 150), (cx - 15, 60), (cx, 15), (cx + 20, 70), (cx + 70, 140), (cx + 90, h - 30),
                         (cx, h - 10)], ink=0.9, soft=4, seed=81, grain=0.35), 0, 1)
    for k, (a, b) in enumerate((((cx - 40, h - 60), (cx - 20, 120)), ((cx + 30, h - 60), (cx + 10, 90)), ((cx, h - 50), (cx - 5, 40)))):
        ink.add(stroke(w, h, [a, ((a[0] + b[0]) / 2 + 12, (a[1] + b[1]) / 2), b], 7, ink=0.8, dry=0.5, profile=SWEEP, seed=82 + k), 0, 1)
    return {'red': _f(red.render(9)), 'ink': _f(ink.render(9))}


def walker(x, y, k=1.0, seed=0, hat=True, staff=True, pack=False):
    """a traveller (ronin) on the page, feet at (x, y); pack = a student's rucksack on the back"""
    ss = StrokeSet(W, H)
    ronin(ss, x, y, k, seed=seed, hat=hat, staff=staff)
    if pack:
        ss.add(pwash(W, H, [(x - 30 * k, y - 140 * k), (x - 8 * k, y - 146 * k), (x - 4 * k, y - 80 * k), (x - 34 * k, y - 84 * k)],
                     ink=0.95, soft=1.5 * k, seed=seed + 9), 0, 1)
    return _f(ss.render(9))


def icon(kind, w=180, h=120, seed=0):
    """a small brush motif for a kokuji card: wave, drop, crossroads, frame, scent, leaf, branch, fish"""
    ss = StrokeSet(w, h)
    c = (w / 2, h / 2)
    if kind == 'wave':                                              # 凪: a calm, flat sea
        for j, y in enumerate((50, 72, 92)):
            ss.add(stroke(w, h, spline([(20, y), (70, y - 4), (110, y + 2), (160, y - 2)], 10), 4 - j, ink=0.8, dry=0.5, profile=SWEEP,
                          seed=seed + j), 0, 1)
    elif kind == 'drop':                                            # 雫: one drop and its ring
        ss.add(pwash(w, h, [(c[0] + 16 * np.sin(a), 30 + 44 * (1 - np.cos(a)) / 2 + (12 * np.cos(a) if abs(a) > 1.2 else 0))
                            for a in np.linspace(-np.pi, np.pi, 24, endpoint=False)], ink=0.9, soft=1.2, seed=seed), 0, 1)
        ss.add(stroke(w, h, spline([(40, 100), (90, 108), (140, 100)], 10), 3, ink=0.6, dry=0.5, profile=SWEEP, seed=seed + 1), 0, 1)
    elif kind == 'cross':                                           # 辻: a crossroads
        ss.add(stroke(w, h, [(20, 70), (160, 66)], 7, ink=0.85, dry=0.6, profile=SWEEP, seed=seed), 0, 1)
        ss.add(stroke(w, h, [(92, 12), (88, 112)], 7, ink=0.85, dry=0.6, profile=SWEEP, seed=seed + 1), 0, 1)
    elif kind == 'frame':                                           # 枠: a window frame
        for a, b in (((40, 20), (140, 20)), ((140, 20), (140, 104)), ((140, 104), (40, 104)), ((40, 104), (40, 20)), ((90, 20), (90, 104))):
            ss.add(stroke(w, h, [a, b], 6, ink=0.85, dry=0.4, profile=PRESS_DOWN, seed=seed + a[0] + b[1]), 0, 1)
    elif kind == 'scent':                                           # 匂: steam rising from a cup
        ss.add(pwash(w, h, [(60, 80), (120, 80), (112, 108), (68, 108)], ink=0.85, soft=1.2, seed=seed), 0, 1)
        for j, x in enumerate((76, 92, 108)):
            ss.add(stroke(w, h, spline([(x, 72), (x - 8, 52), (x + 6, 34), (x - 2, 12)], 10), 2.5, ink=0.6, dry=0.6, profile=SWEEP,
                          seed=seed + 1 + j), 0, 1)
    elif kind == 'leaf':                                            # 笹: bamboo grass leaves
        for j, a in enumerate((-0.5, 0.0, 0.55)):
            x1, y1 = 90 + 70 * np.sin(a), 100 - 80 * np.cos(a)
            ss.add(stroke(w, h, [(90, 108), ((90 + x1) / 2, (108 + y1) / 2), (x1, y1)], 14, ink=0.8, dry=0.4,
                          profile=((0, 0.2), (0.4, 1), (1, 0.05)), seed=seed + j), 0, 1)
    elif kind == 'branch':                                          # 榊: a sakaki sprig
        ss.add(stroke(w, h, spline([(30, 100), (90, 70), (150, 30)], 10), 4, ink=0.85, dry=0.4, profile=SWEEP, seed=seed), 0, 1)
        for j, (x, y, s_) in enumerate(((60, 86, -1), (90, 70, 1), (118, 52, -1), (140, 38, 1))):
            ss.add(stroke(w, h, [(x, y), (x + 14, y + 22 * s_ - 8)], 12, ink=0.75, dry=0.4, profile=((0, 0.2), (0.5, 1), (1, 0.1)),
                          seed=seed + 1 + j), 0, 1)
    elif kind == 'fish':                                            # 鰯: a sardine
        ss.add(pwash(w, h, [(30, 62), (70, 44), (130, 50), (150, 60), (130, 72), (70, 78)], ink=0.85, soft=1.2, seed=seed), 0, 1)
        ss.add(pwash(w, h, [(146, 60), (168, 44), (164, 62), (168, 80)], ink=0.85, soft=1.0, seed=seed + 1), 0, 1)
    return _f(ss.render(9))


def ink_sweep():
    """one big wet stroke dragged across the page (the opening brush sweep)"""
    ss = StrokeSet(W, H)
    ss.add(stroke(W, H, spline([(-80, 900), (300, 860), (700, 880), (1160, 840)], 20), 230, ink=0.9, dry=0.75,
                  profile=((0, 0.6), (0.1, 1), (0.85, 0.8), (1, 0.2)), seed=91), 0, 1)
    return ss

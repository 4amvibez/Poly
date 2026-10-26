"""vignettes — small ink drawings for the example cards: a school, Tokyo Tower over a skyline, a train, a laptop,
a cup of coffee with steam.  Each returns a density patch (painted with poly_ink strokes and washes)."""
import numpy as np

from engine import StrokeSet, blur, spline, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash


def _ss(w, h):
    return StrokeSet(w, h)


def school(w=230, h=140):
    ss = _ss(w, h)
    L = lambda pts, wd, ink=0.85, sd=0: ss.add(stroke(w, h, pts, wd, ink=ink, dry=0.35, profile=BLUNT, seed=sd), 0, 1)
    ss.add(pwash(w, h, [(30, 126), (30, 62), (200, 62), (200, 126)], ink=0.18, soft=1.5, seed=1, grain=0.4), 0, 1)
    L([(18, 64), (115, 22), (212, 64)], 5, sd=2)                       # roof
    L([(30, 62), (30, 127)], 4, sd=3); L([(200, 62), (200, 127)], 4, sd=4); L([(10, 128), (220, 128)], 4, 0.6, 5)
    for row in (78, 100):
        for x in range(42, 196, 26):
            if 100 < x < 130 and row == 100:
                continue
            ss.add(pwash(w, h, [(x, row), (x + 14, row), (x + 14, row + 14), (x, row + 14)], ink=0.7, soft=0.6, seed=x + row), 0, 1)
    ss.add(pwash(w, h, [(106, 106), (124, 106), (124, 127), (106, 127)], ink=0.85, soft=0.6, seed=9), 0, 1)   # door
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    L([(115 + 9 * np.cos(t), 46 + 9 * np.sin(t)) for t in a] + [(124, 46)], 2.5, sd=6)                       # clock
    L([(115, 46), (115, 40)], 2, sd=7); L([(115, 46), (120, 46)], 2, sd=8)
    return ss.render(9)


def tokyo(w=230, h=150):
    ss = _ss(w, h)
    L = lambda pts, wd, ink=0.85, sd=0: ss.add(stroke(w, h, pts, wd, ink=ink, dry=0.35, profile=BLUNT, seed=sd), 0, 1)
    for k, (x0, x1, top) in enumerate([(8, 40, 92), (44, 70, 74), (150, 182, 82), (186, 222, 100), (74, 92, 104), (132, 148, 96)]):
        ss.add(pwash(w, h, [(x0, 146), (x0, top), (x1, top), (x1, 146)], ink=0.28, soft=1.2, seed=k), 0, 1)
    # Tokyo Tower: two legs, lattice, decks, antenna
    L([(88, 146), (106, 70), (112, 20)], 4, sd=10); L([(140, 146), (122, 70), (116, 20)], 4, sd=11)
    for y in range(140, 72, -12):
        u = (146 - y) / 76
        xl, xr = 88 + 18 * u, 140 - 18 * u
        L([(xl, y), (xr, y - 10)], 1.5, 0.6, 12 + y); L([(xr, y), (xl, y - 10)], 1.5, 0.6, 40 + y)
    ss.add(pwash(w, h, [(100, 76), (128, 76), (126, 68), (102, 68)], ink=0.85, soft=0.6, seed=20), 0, 1)
    ss.add(pwash(w, h, [(106, 46), (122, 46), (121, 40), (107, 40)], ink=0.85, soft=0.6, seed=21), 0, 1)
    L([(114, 22), (114, 4)], 2, sd=22)
    L([(4, 147), (226, 147)], 3, 0.6, 23)
    return ss.render(9)


def train(w=240, h=120):
    ss = _ss(w, h)
    L = lambda pts, wd, ink=0.85, sd=0: ss.add(stroke(w, h, pts, wd, ink=ink, dry=0.35, profile=BLUNT, seed=sd), 0, 1)
    body = [(18, 92), (14, 48), (30, 36), (222, 36), (230, 48), (226, 92)]
    ss.add(pwash(w, h, body, ink=0.2, soft=1.2, seed=1, grain=0.4), 0, 1)
    L(body + [(18, 92)], 4, sd=2)
    L([(16, 74), (228, 74)], 5, 0.75, 3)                                          # stripe
    for x in range(36, 210, 30):
        ss.add(pwash(w, h, [(x, 46), (x + 20, 46), (x + 20, 64), (x, 64)], ink=0.75, soft=0.6, seed=x), 0, 1)
    for x in (52, 80, 168, 196):
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ss.add(pwash(w, h, [(x + 9 * np.cos(t), 98 + 9 * np.sin(t)) for t in a], ink=0.9, soft=0.6, seed=x + 1), 0, 1)
    L([(100, 36), (112, 18), (132, 18)], 2.5, sd=4); L([(0, 14), (240, 12)], 2, 0.5, 5)     # pantograph, wire
    L([(0, 110), (240, 110)], 3, 0.6, 6)
    return ss.render(9)


def laptop(w=220, h=140):
    ss = _ss(w, h)
    L = lambda pts, wd, ink=0.85, sd=0: ss.add(stroke(w, h, pts, wd, ink=ink, dry=0.35, profile=BLUNT, seed=sd), 0, 1)
    L([(52, 16), (168, 16), (168, 94), (52, 94), (52, 16)], 7, sd=1)                               # bezel
    ss.add(pwash(w, h, [(58, 22), (162, 22), (162, 88), (58, 88)], ink=0.22, soft=0.8, seed=2, grain=0.4), 0, 1)
    for k, y in enumerate((36, 50, 64)):                                                            # a few lines on the screen
        L([(70, y), (70 + (80 if k != 1 else 56), y)], 3, 0.55, 5 + k)
    L([(34, 112), (52, 96), (168, 96), (186, 112), (34, 112)], 4, sd=3)
    ss.add(pwash(w, h, [(34, 112), (186, 112), (190, 120), (30, 120)], ink=0.6, soft=0.8, seed=4), 0, 1)
    return ss.render(9)


def coffee(w=420, h=420):
    """a cup of coffee on a saucer, seen slightly from above (the steam is animated by the scene)"""
    ss = _ss(w, h)
    L = lambda pts, wd, ink=0.85, sd=0, pr=BLUNT: ss.add(stroke(w, h, pts, wd, ink=ink, dry=0.4, profile=pr, seed=sd), 0, 1)
    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ell = lambda cx, cy, rx, ry: [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]
    ss.add(pwash(w, h, ell(205, 350, 175, 46), ink=0.16, soft=2, seed=1, grain=0.4), 0, 1)          # saucer
    rim = ell(205, 350, 175, 46) + [ell(205, 350, 175, 46)[0]]
    vis = [not (90 < x < 320 and y < 340) for x, y in rim]                    # the back of the rim is hidden by the cup
    run = []
    for p_, v_ in zip(rim, vis):
        if v_:
            run.append(p_)
        elif len(run) > 1:
            L(run, 5, 0.75, 2); run = []
        else:
            run = []
    if len(run) > 1:
        L(run, 5, 0.75, 2)
    cup = [(80, 210), (95, 300), (130, 336), (280, 336), (315, 300), (330, 210)]
    ss.add(pwash(w, h, cup, ink=0.22, soft=2, seed=3, grain=0.4), 0, 1)
    L(cup, 6, 0.85, 4, PRESS_DOWN)
    L([(318, 236), (368, 232), (378, 268), (352, 296), (308, 296)], 7, 0.85, 5, PRESS_DOWN)          # handle
    ss.add(pwash(w, h, ell(205, 210, 125, 30), ink=0.85, soft=1.2, seed=6, grain=0.25), 0, 1)      # the coffee
    L(ell(205, 210, 125, 30) + [ell(205, 210, 125, 30)[0]], 5, 0.9, 7)
    L([(150, 208), (190, 202), (236, 210)], 3, 0.35, 8, SWEEP)                                       # a highlight in the crema
    return ss.render(9)


def portrait(w=230, h=150):
    """a small figure waving hello, with a speech bubble (for a name card)"""
    from art import figure
    ss = _ss(w, h)
    figure(ss, 80, 140, 1.25, 0, seed=11, pose='speak', facing=1, dur=0.01)
    L = lambda pts, wd, ink=0.8, sd=0: ss.add(stroke(w, h, spline(pts, 10), wd, ink=ink, dry=0.35, profile=BLUNT, seed=sd), 0, 1)
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    bub = [(170 + 44 * np.cos(t), 46 + 28 * np.sin(t)) for t in a]
    L(bub + [bub[0]], 3.5, 0.75, 3)
    L([(140, 66), (124, 84), (150, 70)], 3, 0.75, 4)                                                  # tail
    from engine import FONT_BRUSH_EN, text_layer
    d = ss.render(9)
    hi, _ = text_layer('hi!', FONT_BRUSH_EN, 40)
    hh, hw = hi.shape
    y0, x0 = 46 - hh // 2, 172 - hw // 2
    d[y0:y0 + hh, x0:x0 + hw] = np.maximum(d[y0:y0 + hh, x0:x0 + hw], hi * 0.9)
    return d


def shout(w=230, h=150):
    """a spiky manga shout bubble with an exclamation inside"""
    ss = _ss(w, h)
    r = np.random.default_rng(5)
    n = 18
    pts = []
    for k in range(n * 2):
        a = k / (n * 2) * 2 * np.pi
        rad = (1.0 if k % 2 == 0 else 0.7) * (1 + r.uniform(-0.08, 0.08))
        pts.append((115 + 100 * rad * np.cos(a), 75 + 64 * rad * np.sin(a)))
    ss.add(pwash(w, h, pts, ink=0.1, soft=1, seed=1, grain=0.4), 0, 1)
    for k in range(len(pts)):                                         # straight edges, one brush jab per side
        p0, p1 = pts[k], pts[(k + 1) % len(pts)]
        ss.add(stroke(w, h, [p0, p1], 3.5, ink=0.85, dry=0.3, profile=BLUNT, seed=10 + k), 0, 1)
    for k, x in enumerate((100, 130)):
        ss.add(stroke(w, h, [(x + 3, 44), (x, 84)], 11, ink=0.95, dry=0.2, profile=PRESS_DOWN, seed=40 + k), 0, 1)
        a = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        ss.add(pwash(w, h, [(x - 1 + 6 * np.cos(t), 100 + 6 * np.sin(t)) for t in a], ink=0.95, soft=0.6, seed=50 + k), 0, 1)
    return ss.render(9)


def burst(r_in=150, r_out=260, n=40, seed=0):
    """manga focus lines radiating from a centre (density, square 2*r_out)"""
    s = 2 * r_out
    ss = _ss(s, s)
    rg = np.random.default_rng(seed)
    for k in range(n):
        a = k / n * 2 * np.pi + rg.uniform(-0.05, 0.05)
        r0 = r_in * rg.uniform(0.95, 1.15)
        r1 = r_out * rg.uniform(0.8, 1.0)
        ss.add(stroke(s, s, [(s / 2 + r1 * np.cos(a), s / 2 + r1 * np.sin(a)), (s / 2 + r0 * np.cos(a), s / 2 + r0 * np.sin(a))],
                      rg.uniform(3, 6), ink=0.8, dry=0.4, profile=SWEEP, seed=seed + k), 0, 1)
    return ss.render(9)

"""pics — the small ink drawings this reel needs, each a density patch painted with poly_ink strokes and washes:
a mother, a flower, a heart (likes), a bridge, chopsticks, a cliff edge, an eye, a pair of chickens in a fenced
garden, feathers, and thought clouds.  Static patches are cached by the scenes; the chickens are drawn per pose so a
scene can make them peck."""
import numpy as np

from engine import StrokeSet, spline, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash


def _circ(cx, cy, rx, ry=None, n=28, a0=0.0, a1=2 * np.pi):
    ry = rx if ry is None else ry
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(a0, a1, n, endpoint=a1 - a0 < 6.28)]


def mother(w=220, h=200):
    """a woman in a kimono, hair in a bun, holding a sprig"""
    ss = StrokeSet(w, h)
    L = lambda pts, wd, ink=0.85, sd=0, pr=PRESS_DOWN: ss.add(stroke(w, h, spline(pts, 10) if len(pts) > 2 else pts, wd, ink=ink, dry=0.35, profile=pr, seed=sd), 0, 1)
    robe = [(96, 70), (124, 70), (150, 120), (160, 190), (60, 190), (72, 120)]
    ss.add(pwash(w, h, robe, ink=0.3, soft=2, seed=1, grain=0.4), 0, 1)
    L([(96, 70), (78, 120), (62, 190)], 5, sd=2)
    L([(124, 70), (144, 120), (158, 190)], 5, sd=3)
    L([(60, 190), (160, 190)], 4, 0.7, 4, BLUNT)
    L([(100, 74), (110, 130), (122, 76)], 3, 0.6, 5)                     # collar
    ss.add(pwash(w, h, [(80, 128), (140, 128), (138, 140), (82, 140)], ink=0.75, soft=1, seed=6), 0, 1)   # obi
    ss.add(pwash(w, h, _circ(110, 50, 16, 19), ink=0.8, soft=1.2, seed=7, grain=0.15), 0, 1)
    ss.add(pwash(w, h, _circ(110, 26, 11, 9), ink=0.9, soft=1, seed=8), 0, 1)                  # bun
    L([(140, 112), (166, 84), (178, 60)], 3, 0.7, 9)                       # the sprig
    for k, (x, y) in enumerate(((176, 58), (170, 72), (184, 70))):
        ss.add(pwash(w, h, _circ(x, y, 6), ink=0.5, soft=1, seed=20 + k), 0, 1)
    return ss.render(9).astype(np.float32)


def flower(w=200, h=200, petals=5):
    """a five-petalled blossom on a stem with two leaves (density; the scene colours petals and stem separately)"""
    P, S = StrokeSet(w, h), StrokeSet(w, h)
    cx, cy = 100, 78
    for k in range(petals):
        a = -np.pi / 2 + k * 2 * np.pi / petals
        px, py = cx + 34 * np.cos(a), cy + 34 * np.sin(a)
        poly = [(cx + 8 * np.cos(a + 1.6), cy + 8 * np.sin(a + 1.6)), (px + 22 * np.cos(a + 1.2), py + 22 * np.sin(a + 1.2)),
                (px + 18 * np.cos(a), py + 18 * np.sin(a)), (px + 22 * np.cos(a - 1.2), py + 22 * np.sin(a - 1.2)),
                (cx + 8 * np.cos(a - 1.6), cy + 8 * np.sin(a - 1.6))]
        P.add(pwash(w, h, poly, ink=0.8, soft=1.5, seed=k, grain=0.35), 0, 1)
    P.add(pwash(w, h, _circ(cx, cy, 8), ink=0.9, soft=1, seed=9), 0, 1)
    S.add(stroke(w, h, spline([(100, 108), (96, 150), (104, 196)], 10), 5, ink=0.85, dry=0.3, profile=PRESS_DOWN, seed=10), 0, 1)
    for k, sx in enumerate((-1, 1)):
        S.add(pwash(w, h, [(100, 160 - 10 * k), (100 + sx * 40, 140 - 10 * k), (100 + sx * 56, 150 - 10 * k), (100 + sx * 20, 168 - 10 * k)],
                    ink=0.7, soft=1.5, seed=11 + k), 0, 1)
    return P.render(9).astype(np.float32), S.render(9).astype(np.float32)


def heart(w=140, h=130):
    ss = StrokeSet(w, h)
    t = np.linspace(0, 2 * np.pi, 40)
    pts = [(70 + 3.6 * 16 * np.sin(a) ** 3, 58 - 3.6 * (13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a))) for a in t]
    ss.add(pwash(w, h, pts, ink=0.85, soft=1.5, seed=1, grain=0.3), 0, 1)
    return ss.render(9).astype(np.float32)


def bridge(w=240, h=150):
    ss = StrokeSet(w, h)
    deck = [(10, 104), (70, 70), (120, 62), (170, 70), (230, 104)]
    ss.add(stroke(w, h, spline(deck, 16), 8, ink=0.9, dry=0.35, profile=BLUNT, seed=1), 0, 1)
    rail = [(x, y - 30) for x, y in deck]
    ss.add(stroke(w, h, spline(rail, 16), 4, ink=0.85, dry=0.35, profile=BLUNT, seed=2), 0, 1)
    for k, x in enumerate(range(24, 230, 24)):
        y = np.interp(x, [p[0] for p in deck], [p[1] for p in deck])
        ss.add(stroke(w, h, [(x, y - 30), (x, y)], 3, ink=0.8, dry=0.3, profile=BLUNT, seed=10 + k), 0, 1)
    ss.add(stroke(w, h, spline([(40, 120), (120, 92), (200, 120)], 12), 5, ink=0.6, dry=0.5, profile=BLUNT, seed=3), 0, 1)
    for k in range(4):
        x = 30 + 50 * k
        ss.add(stroke(w, h, [(x, 136), (x + 34, 136)], 2.5, ink=0.45, dry=0.6, profile=SWEEP, seed=30 + k), 0, 1)
    return ss.render(9).astype(np.float32)


def chopsticks(w=240, h=150):
    ss = StrokeSet(w, h)
    ss.add(pwash(w, h, _circ(120, 112, 90, 22), ink=0.12, soft=2, seed=1, grain=0.4), 0, 1)     # a little plate
    ss.add(stroke(w, h, [(36, 118), (204, 30)], 8, ink=0.9, dry=0.25, profile=((0, 1), (1, 0.45)), seed=2), 0, 1)
    ss.add(stroke(w, h, [(52, 128), (214, 52)], 8, ink=0.9, dry=0.25, profile=((0, 1), (1, 0.45)), seed=3), 0, 1)
    ss.add(pwash(w, h, [(48, 124), (66, 112), (74, 120), (58, 132)], ink=0.8, soft=1, seed=4), 0, 1)              # rest
    return ss.render(9).astype(np.float32)


def edge(w=240, h=150):
    """the edge of a cliff, with a pebble about to go over"""
    ss = StrokeSet(w, h)
    cliff = [(0, 60), (150, 56), (158, 70), (150, 100), (162, 150), (0, 150)]
    ss.add(pwash(w, h, cliff, ink=0.35, soft=2, seed=1, grain=0.5), 0, 1)
    ss.add(stroke(w, h, [(0, 60), (150, 56), (158, 70), (150, 100), (162, 150)], 6, ink=0.9, dry=0.4, profile=PRESS_DOWN, seed=2), 0, 1)
    ss.add(pwash(w, h, _circ(146, 50, 7), ink=0.9, soft=1, seed=3), 0, 1)
    for k, y in enumerate((80, 104, 128)):
        ss.add(stroke(w, h, [(186, y), (232, y + 2)], 2, ink=0.35, dry=0.6, profile=SWEEP, seed=4 + k), 0, 1)   # the drop
    return ss.render(9).astype(np.float32)


def eye(w=420, h=220):
    """a big brushed eye (lid strokes, iris wash, a highlight) — density; the iris also returned to tint"""
    E, I = StrokeSet(w, h), StrokeSet(w, h)
    up = spline([(30, 120), (120, 46), (210, 30), (300, 46), (390, 120)], 20)
    lo = spline([(30, 120), (120, 170), (210, 184), (300, 170), (390, 120)], 20)
    E.add(stroke(w, h, [tuple(p) for p in up], 16, ink=0.95, dry=0.35, profile=((0, 0.4), (0.3, 1), (1, 0.3)), seed=1), 0, 1)
    E.add(stroke(w, h, [tuple(p) for p in lo], 8, ink=0.85, dry=0.4, profile=((0, 0.3), (0.5, 1), (1, 0.2)), seed=2), 0, 1)
    for k in range(5):
        x = 110 + 50 * k
        E.add(stroke(w, h, [(x, 42 - 3 * abs(k - 2)), (x - 10 + 5 * k, 16 - 4 * abs(k - 2))], 4, ink=0.85, dry=0.3, profile=SWEEP, seed=10 + k), 0, 1)
    I.add(pwash(w, h, _circ(210, 108, 66), ink=0.9, soft=1.5, seed=3, grain=0.3), 0, 1)
    pupil = StrokeSet(w, h)
    pupil.add(pwash(w, h, _circ(210, 108, 28), ink=1.0, soft=1, seed=4), 0, 1)
    return E.render(9).astype(np.float32), I.render(9).astype(np.float32), pupil.render(9).astype(np.float32)


def chicken(w=260, h=240, peck=0.0, facing=1, seed=0):
    """a hen side-on (ink body, red comb drawn separately): returns (ink, comb). peck 0..1 lowers the head"""
    ss, cm = StrokeSet(w, h), StrokeSet(w, h)
    f = facing
    X = lambda x: w / 2 + f * (x - 130)
    hx, hy = X(178 + 14 * peck), 70 + 70 * peck
    body = [(X(70), 120), (X(96), 92), (X(150), 96), (X(176), 118), (X(172), 160), (X(140), 182), (X(96), 180), (X(66), 158)]
    ss.add(pwash(w, h, body, ink=0.42, soft=3, seed=seed, grain=0.45), 0, 1)
    ss.add(stroke(w, h, spline([(X(168), 110), (X(132), 92), (X(96), 92), (X(76), 108)], 10), 6, ink=0.85, dry=0.4, profile=PRESS_DOWN, seed=seed + 11), 0, 1)
    ss.add(stroke(w, h, spline([(X(172), 116), (X(184), 140), (X(172), 164)], 8), 6, ink=0.8, dry=0.4, profile=PRESS_DOWN, seed=seed + 12), 0, 1)
    ss.add(stroke(w, h, spline([(X(66), 158), (X(100), 182), (X(146), 182), (X(176), 150)], 10), 6, ink=0.85, dry=0.4, profile=PRESS_DOWN, seed=seed + 1), 0, 1)
    for k, (a, b) in enumerate((((X(80), 112), (X(30), 56)), ((X(78), 120), (X(24), 86)), ((X(82), 130), (X(36), 118)))):   # tail
        ss.add(stroke(w, h, spline([a, ((a[0] + b[0]) / 2 + f * -6, (a[1] + b[1]) / 2 - 10), b], 8), 9 - 2 * k, ink=0.9, dry=0.45, profile=SWEEP, seed=seed + 2 + k), 0, 1)
    ss.add(stroke(w, h, spline([(X(110), 120), (X(140), 128), (X(156), 146)], 8), 10, ink=0.75, dry=0.5, profile=PRESS_DOWN, seed=seed + 6), 0, 1)   # wing
    ss.add(stroke(w, h, spline([(X(166), 112), ((X(166) + hx) / 2, (112 + hy) / 2 + 4), (hx - f * 6, hy + 14)], 8), 12, ink=0.7, dry=0.4, profile=BLUNT, seed=seed + 7), 0, 1)   # neck
    ss.add(pwash(w, h, _circ(hx, hy, 16, 14), ink=0.55, soft=1.5, seed=seed + 8), 0, 1)
    ss.add(pwash(w, h, _circ(hx + f * 4, hy - 3, 3.2), ink=1.0, soft=0.5, seed=seed + 9), 0, 1)
    ss.add(pwash(w, h, [(hx + f * 14, hy - 3), (hx + f * 30, hy + 2), (hx + f * 14, hy + 6)], ink=0.9, soft=0.7, seed=seed + 10), 0, 1)   # beak
    for k in range(3):
        cm.add(pwash(w, h, _circ(hx - f * (8 - 7 * k), hy - 16 - 2 * (k == 1), 6), ink=0.95, soft=1, seed=seed + 20 + k), 0, 1)
    cm.add(pwash(w, h, _circ(hx + f * 12, hy + 14, 4, 6), ink=0.9, soft=0.8, seed=seed + 25), 0, 1)                 # wattle
    for k, x in enumerate((X(112), X(136))):
        ss.add(stroke(w, h, [(x, 180), (x + f * 4, 214)], 4, ink=0.9, dry=0.2, profile=BLUNT, seed=seed + 30 + k), 0, 1)
        ss.add(stroke(w, h, [(x - 12, 216), (x + 4, 214), (x + 16, 218)], 3, ink=0.9, dry=0.2, profile=BLUNT, seed=seed + 40 + k), 0, 1)
    return ss.render(9).astype(np.float32), cm.render(9).astype(np.float32)


def feathers(w=200, h=160):
    ss = StrokeSet(w, h)
    for k, (x0, ang) in enumerate(((70, -0.35), (130, 0.3))):
        c, s = np.cos(ang), np.sin(ang)
        R = lambda u, v: (x0 + u * s + v * c, 80 - u * c + v * s)
        ss.add(stroke(w, h, [R(-66, 0), R(64, 0)], 3, ink=0.9, dry=0.3, profile=SWEEP, seed=k), 0, 1)
        vane = [R(-40, 0)] + [R(u, 16 * np.sin(np.pi * (u + 44) / 104)) for u in np.linspace(-44, 60, 10)] + \
               [R(u, -14 * np.sin(np.pi * (u + 44) / 104)) for u in np.linspace(60, -44, 10)]
        ss.add(pwash(w, h, vane, ink=0.45, soft=1.5, seed=10 + k, grain=0.5), 0, 1)
    return ss.render(9).astype(np.float32)


def fence(w=1080, h=260, seed=0):
    """a low woven-bamboo garden fence with posts, and a little grass"""
    ss = StrokeSet(w, h)
    r = np.random.default_rng(seed)
    for k, x in enumerate(range(20, w, 150)):
        ss.add(stroke(w, h, [(x, 40), (x + 2, 230)], 10, ink=0.85, dry=0.4, profile=PRESS_DOWN, seed=seed + k), 0, 1)
    for k, y in enumerate((70, 120, 170)):
        ss.add(stroke(w, h, [(0, y), (w / 2, y + r.uniform(-4, 4)), (w, y)], 5, ink=0.7, dry=0.5, profile=SWEEP, seed=seed + 20 + k), 0, 1)
    for k in range(40):
        x = r.uniform(0, w)
        ss.add(stroke(w, h, [(x, 250), (x + r.uniform(-6, 6), 250 - r.uniform(14, 30))], 2.5, ink=0.6, dry=0.4, profile=SWEEP, seed=seed + 50 + k), 0, 1)
    return ss.render(9).astype(np.float32)


def cloud(w=420, h=260, seed=0, tail=(1, 1)):
    """a thought cloud outline (scalloped) with a pale fill; tail = direction of the little trailing bubbles"""
    ss, fill = StrokeSet(w, h), StrokeSet(w, h)
    cx, cy, rx, ry = w / 2, h / 2 - 14, w / 2 - 30, h / 2 - 44
    n = 9
    pts = []
    for k in range(n * 6 + 1):
        a = k / (n * 6) * 2 * np.pi
        bump = 1 + 0.09 * abs(np.sin(a * n / 2 * 2))
        pts.append((cx + rx * bump * np.cos(a), cy + ry * bump * np.sin(a)))
    fill.add(pwash(w, h, pts, ink=1.0, soft=2, seed=seed), 0, 1)
    ss.add(stroke(w, h, pts, 4.5, ink=0.85, dry=0.4, profile=BLUNT, seed=seed + 1), 0, 1)
    tx, ty = tail
    for k, r_ in enumerate((14, 9)):
        x, y = cx + tx * (rx * 0.6 + 26 * (k + 1)), cy + ty * (ry + 18 + 22 * k)
        ss.add(stroke(w, h, _circ(x, y, r_, n=16) + [_circ(x, y, r_, n=16)[0]], 3.5, ink=0.85, dry=0.3, profile=BLUNT, seed=seed + 2 + k), 0, 1)
        fill.add(pwash(w, h, _circ(x, y, r_), ink=1.0, soft=1, seed=seed + 4 + k), 0, 1)
    return ss.render(9).astype(np.float32), np.clip(fill.render(9), 0, 1).astype(np.float32)

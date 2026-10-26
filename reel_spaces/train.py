"""train — the hiragana scene's metaphor: the sentence as a little bullet train.  Each chunk rides in its own car
(私 | 日本語 | 勉強します, the verb up front in the locomotive), and the particles は and を are the couplers between the
cars: the small words that hold the sentence together.  The sprite is painted once (cached); the scene writes the
words onto it and moves it along a static track."""
import numpy as np

from engine import StrokeSet, blur, cached, spline, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash

TW, TH = 1040, 240
CAR1 = (40, 200)                 # x spans in sprite coordinates
CAR2 = (270, 520)
LOCO = (590, 1010)               # body to ~900, then the nose
TOP, BOT = 52, 190
TEXT_Y = 128
COUPLERS = [(235, 118), (555, 118)]
TEXT_X = [120, 395, 745]


def dense(poly, step=8.0):
    """resample a polyline into closely spaced points so the brush's spline stays on the straight edges"""
    out = []
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    return out + [tuple(poly[-1])]


def _loco_outline():
    nose = spline([(880, TOP), (930, TOP + 14), (972, TOP + 62), (1004, TOP + 118), (1008, BOT - 6)], 14)
    return [(LOCO[0], TOP)] + [tuple(p) for p in nose] + [(1004, BOT), (LOCO[0], BOT)]


def build():
    def make():
        ss = StrokeSet(TW, TH)
        body = StrokeSet(TW, TH)
        L = lambda pts, wd, ink=0.85, sd=0, pr=BLUNT, dry=0.35: ss.add(stroke(TW, TH, pts, wd, ink=ink, dry=dry, profile=pr, seed=sd), 0, 1)
        rr = lambda x0, x1, r=16: ([(x0 + r * (1 - np.cos(a)), TOP + r * (1 - np.sin(a))) for a in np.linspace(0, np.pi / 2, 5)] +
                                   [(x1 - r * (1 - np.sin(a)), TOP + r * (1 - np.cos(a))) for a in np.linspace(0, np.pi / 2, 5)] +
                                   [(x1, BOT), (x0, BOT)])
        outlines = [rr(*CAR1), rr(*CAR2), _loco_outline()]
        for k, poly in enumerate(outlines):
            body.add(pwash(TW, TH, poly, ink=1.0, soft=0.8, seed=k), 0, 1)
            ss.add(pwash(TW, TH, poly, ink=0.12, soft=1.6, seed=10 + k, grain=0.45), 0, 1)
            L(dense(poly + [poly[0]]), 5, 0.88, 20 + k, BLUNT)
        # windows along the top of each car, a windscreen on the nose
        for k, (x0, x1) in enumerate([CAR1, CAR2, (LOCO[0], 880)]):
            for x in np.arange(x0 + 20, x1 - 30, 34):
                ss.add(pwash(TW, TH, [(x, 66), (x + 20, 66), (x + 20, 88), (x, 88)], ink=0.72, soft=0.7, seed=int(x)), 0, 1)
        ss.add(pwash(TW, TH, [(900, 64), (936, 72), (966, 104), (918, 104)], ink=0.85, soft=0.9, seed=77), 0, 1)
        # skirt, wheels, couplers, pantograph
        for k, (x0, x1) in enumerate([CAR1, CAR2, (LOCO[0], 960)]):
            for x in (x0 + 30, x0 + 58, x1 - 58, x1 - 30):
                a = np.linspace(0, 2 * np.pi, 18, endpoint=False)
                ss.add(pwash(TW, TH, [(x + 13 * np.cos(t), 200 + 13 * np.sin(t)) for t in a], ink=0.92, soft=0.6, seed=int(x) + 3), 0, 1)
        for (cx, _), (x0, x1) in zip(COUPLERS, [(CAR1[1], CAR2[0]), (CAR2[1], LOCO[0])]):
            L([(x0 - 2, 176), (x1 + 2, 176)], 7, 0.9, cx)
            L([(x0 + 4, 168), (x0 + 4, 184)], 4, 0.9, cx + 1)
            L([(x1 - 4, 168), (x1 - 4, 184)], 4, 0.9, cx + 2)
        L([(700, TOP), (722, 26), (770, 26)], 3, 0.8, 90)
        L([(740, TOP), (722, 26)], 2.5, 0.7, 91)
        stripe = StrokeSet(TW, TH)
        for k, (x0, x1) in enumerate([CAR1, CAR2, (LOCO[0], 1000)]):
            stripe.add(stroke(TW, TH, [(x0 + 4, 172), (x1 - 4, 172)], 7, ink=0.9, dry=0.3, profile=SWEEP, seed=40 + k), 0, 1)
        return {'ink': ss.render(9).astype(np.float32), 'body': np.clip(blur(body.render(9), 1.0), 0, 1).astype(np.float32),
                'stripe': stripe.render(9).astype(np.float32)}
    return cached('train_v1', make)


def badge(r=31, seed=0):
    """a soft round pad for a particle (density for PINK)"""
    s = 2 * r + 12
    ss = StrokeSet(s, s)
    a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ss.add(pwash(s, s, [(s / 2 + r * np.cos(t), s / 2 + r * np.sin(t)) for t in a], ink=0.9, soft=1.2, seed=seed, grain=0.3), 0, 1)
    return ss.render(9).astype(np.float32)


def track(w, h, y_rail, y_wire, seed=0):
    """a static rail with sleepers and an overhead wire with poles, in scene coordinates"""
    def make():
        ss = StrokeSet(w, h)
        ss.add(stroke(w, h, [(-20, y_rail), (w / 2, y_rail - 1), (w + 20, y_rail)], 5, ink=0.85, dry=0.4, profile=SWEEP, seed=seed), 0, 1)
        for x in range(10, w, 46):
            ss.add(stroke(w, h, [(x, y_rail + 4), (x + 22, y_rail + 4)], 6, ink=0.45, dry=0.5, profile=BLUNT, seed=seed + x), 0, 1)
        ss.add(stroke(w, h, [(-20, y_wire), (w / 2, y_wire + 10), (w + 20, y_wire)], 2, ink=0.6, dry=0.3, profile=SWEEP, seed=seed + 1), 0, 1)
        for x in (90, 1000):
            ss.add(stroke(w, h, [(x, y_wire - 30), (x, y_rail - 4)], 6, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=seed + x + 1), 0, 1)
            ss.add(stroke(w, h, [(x - 30, y_wire - 18), (x + 30, y_wire - 18)], 4, ink=0.7, dry=0.4, profile=BLUNT, seed=seed + x + 2), 0, 1)
        return {'track': ss.render(9).astype(np.float32)}
    return cached(f'track_{w}_{h}_{y_rail}_{y_wire}_{seed}', make)['track']


def speed_lines(seed=0):
    s = StrokeSet(420, 200)
    r = np.random.default_rng(seed)
    for k in range(7):
        y = 20 + k * 26 + r.uniform(-6, 6)
        x0 = r.uniform(0, 140)
        s.add(stroke(420, 200, [(x0, y), (x0 + r.uniform(180, 280), y)], r.uniform(2, 4), ink=0.6, dry=0.6, profile=SWEEP, seed=seed + k), 0, 1)
    return s.render(9).astype(np.float32)

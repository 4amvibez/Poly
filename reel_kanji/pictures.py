"""pictures — small ink paintings of the things the first characters drew, each painted on the same grid as its
early form in ancient.py (a unit box inside a pad of 6 %), so that the pictograph traced over the painting lands
exactly on the mountain's peaks, the river's banks, the tree's branches and roots, the bird's head and tail.

    picture(name, S) -> {'ink': density, 'water': density or None}, a (S + 2P) square like ancient(ch, 'oracle', S)"""
import numpy as np

from engine import StrokeSet, blur, cached, noise, spline, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash


def _box(S):
    P = int(S * 0.06)
    D = S + 2 * P
    X = lambda u, v: (P + u * S, P + v * S)
    return P, D, X


def _foliage(ss, D, centres, r, n, seed, ink=0.45):
    rng = np.random.default_rng(seed)
    for k in range(n):
        cx, cy = centres[k % len(centres)]
        a, rr = rng.uniform(0, 2 * np.pi), abs(rng.normal(0, 0.55)) * r
        x, y = cx + np.cos(a) * rr * 1.3, cy + np.sin(a) * rr * 0.8
        b = rng.uniform(0, np.pi)
        ln = rng.uniform(0.25, 0.45) * r
        ss.add(stroke(D, D, [(x - np.cos(b) * ln, y - np.sin(b) * ln * 0.4), (x + np.cos(b) * ln, y + np.sin(b) * ln * 0.4)],
                      rng.uniform(0.28, 0.45) * r, ink=rng.uniform(0.5, 1.0) * ink, dry=0.3, profile=BLUNT, seed=seed + k), 0, 1)


def mountain(S):
    from inkpaint import Painting
    P, D, X = _box(S)
    p = Painting(D, D, seed=3)
    mid = p.peak(X(.5, 0)[0], X(0, .18)[1], 0.7 * S, 0.12 * S, 0.12 * S, power=1.2, rough=0.012 * S, seed=1)
    left = p.peak(X(.21, 0)[0], X(0, .42)[1], 0.45 * S, 0.1 * S, 0.1 * S, power=1.2, rough=0.01 * S, seed=2)
    right = p.peak(X(.79, 0)[0], X(0, .42)[1], 0.45 * S, 0.1 * S, 0.1 * S, power=1.2, rough=0.01 * S, seed=3)
    ridge = np.minimum(np.minimum(left, right), mid)
    p.wash(ridge, dens=0.8, decay=0.45 * S, tex=0.12 * S, edge=0.55, mist_y=X(0, .86)[1], mist_h=0.04 * S, seed=4)
    for sx in (.21, .5, .79):
        p.cun(ridge, X(sx - .12, 0)[0], X(sx + .12, 0)[0], n=18, shadow_x=X(sx, 0)[0], spread=0.06 * S, max_y=X(0, .8)[1], ink=0.3)
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    fade = np.clip((X(0, .9)[1] - yy) / (0.08 * S), 0, 1) * np.clip(np.minimum(xx, D - xx) / (0.05 * S), 0, 1)
    return {'ink': np.clip(p.D * fade * 1.7, 0, 1).astype(np.float32), 'water': None}


def river(S):
    from ancient import FORMS
    P, D, X = _box(S)
    L1, L2, L3 = [spline(np.array([X(*q) for q in st[1]], np.float32), 16) for st in FORMS[('川', 'oracle')]]
    water = StrokeSet(D, D)
    water.add(pwash(D, D, [tuple(q) for q in np.vstack([L1, L3[::-1]])], ink=0.75, soft=0.01 * S, seed=5, grain=0.35), 0, 1)
    ss = StrokeSet(D, D)
    rng = np.random.default_rng(6)
    for Lb, side in ((L1, -1), (L3, 1)):
        ss.add(stroke(D, D, [tuple(q) for q in Lb], 0.022 * S, ink=0.85, dry=0.45, profile=PRESS_DOWN, seed=7 + side), 0, 1)
        # grass and stones on the bank, outside the water
        for k in range(26):
            q = Lb[int(rng.uniform(0, len(Lb) - 1))]
            x, y = q[0] + side * rng.uniform(0.02, 0.09) * S, q[1]
            ss.add(stroke(D, D, [(x, y), (x + side * rng.uniform(0, 0.02) * S, y - rng.uniform(0.02, 0.05) * S)],
                          0.008 * S, ink=0.6, dry=0.5, profile=PRESS_DOWN, seed=20 + k + 40 * (side > 0)), 0, 1)
    # the current: broken strokes along the middle line, and ripples
    n = len(L2)
    for a in range(0, n - 8, 14):
        ss.add(stroke(D, D, [tuple(q) for q in L2[a:a + 8]], 0.012 * S, ink=0.45, dry=0.5, profile=SWEEP, seed=60 + a), 0, 1)
    for k in range(10):
        i = int(rng.uniform(0.1, 0.9) * len(L1))
        x0, x1 = L1[i][0], L3[i][0]
        y = (L1[i][1] + L3[i][1]) / 2
        x = rng.uniform(x0 + 0.04 * S, x1 - 0.08 * S)
        ss.add(stroke(D, D, [(x, y), (x + 0.05 * S, y + 0.002 * S)], 0.006 * S, ink=0.4, dry=0.5, profile=SWEEP, seed=90 + k), 0, 1)
    return {'ink': ss.render(9).astype(np.float32), 'water': water.render(9).astype(np.float32)}


def tree(S):
    P, D, X = _box(S)
    ss = StrokeSet(D, D)
    _foliage(ss, D, [X(.2, .13), X(.8, .13), X(.5, .08), X(.35, .22), X(.65, .22)], 0.12 * S, 90, seed=11, ink=0.42)
    ss.add(stroke(D, D, spline([X(.5, .66), X(.505, .52), X(.5, .4)], 8), 0.07 * S, ink=0.85, dry=0.45, profile=BLUNT, seed=1), 0, 1)
    ss.add(stroke(D, D, spline([X(.5, .4), X(.5, .22), X(.5, .07)], 8), 0.03 * S, ink=0.8, dry=0.45, profile=PRESS_DOWN, seed=2), 0, 1)
    for k, (a, b) in enumerate(((X(.5, .4), X(.2, .13)), (X(.5, .4), X(.8, .13)))):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03 * S)
        ss.add(stroke(D, D, spline([a, mid, b], 8), 0.035 * S, ink=0.82, dry=0.45, profile=PRESS_DOWN, seed=3 + k), 0, 1)
    for k, (a, b) in enumerate(((X(.5, .62), X(.2, .88)), (X(.5, .62), X(.8, .88)))):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 0.02 * S)
        ss.add(stroke(D, D, spline([a, mid, b], 8), 0.04 * S, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=5 + k), 0, 1)
    ss.add(stroke(D, D, [X(.08, .9), X(.5, .91), X(.92, .9)], 0.012 * S, ink=0.4, dry=0.6, profile=SWEEP, seed=9), 0, 1)
    return {'ink': ss.render(9).astype(np.float32), 'water': None}


def bird(S):
    P, D, X = _box(S)
    ss = StrokeSet(D, D)
    head = [(.31, .25), (.35, .13), (.48, .09), (.58, .15)]
    back = [(.58, .15), (.61, .3), (.69, .44), (.8, .59), (.92, .77)]
    breast = [(.32, .31), (.3, .47), (.36, .62), (.5, .71), (.66, .74)]
    body = [X(*q) for q in head + back[1:] + [(.66, .74)] + breast[::-1][1:]]
    ss.add(pwash(D, D, body, ink=0.38, soft=0.012 * S, seed=1, grain=0.4), 0, 1)
    circ = lambda c, r: [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 22, endpoint=False)]
    ss.add(pwash(D, D, circ(X(.46, .17), 0.095 * S), ink=0.78, soft=0.008 * S, seed=2, grain=0.2), 0, 1)
    for k, (a, b) in enumerate((((.49, .35), (.71, .49)), ((.47, .45), (.7, .6)), ((.47, .55), (.66, .67)))):
        ss.add(stroke(D, D, [X(*a), X(*b)], 0.05 * S, ink=0.75, dry=0.4, profile=PRESS_DOWN, seed=3 + k), 0, 1)
    ss.add(stroke(D, D, [X(.33, .26), X(.15, .28), X(.32, .31)], 0.02 * S, ink=0.95, dry=0.1, profile=BLUNT, seed=7), 0, 1)
    ss.add(stroke(D, D, spline([X(.62, .72), X(.8, .75), X(.95, .79)], 8), 0.04 * S, ink=0.8, dry=0.45, profile=PRESS_DOWN, seed=8), 0, 1)
    for k, pts in enumerate(([(.45, .69), (.41, .84), (.3, .9)], [(.41, .84), (.47, .92)], [(.55, .72), (.53, .86), (.45, .93)])):
        ss.add(stroke(D, D, [X(*q) for q in pts], 0.012 * S, ink=0.9, dry=0.2, profile=BLUNT, seed=10 + k), 0, 1)
    ss.add(stroke(D, D, [X(.12, .93), X(.5, .94), X(.8, .92)], 0.03 * S, ink=0.6, dry=0.55, profile=SWEEP, seed=14), 0, 1)
    D_ = ss.render(9)
    eye = np.zeros_like(D_)
    ey = X(.43, .19)
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    eye = np.exp(-((xx - ey[0]) ** 2 + (yy - ey[1]) ** 2) / (2 * (0.012 * S) ** 2))
    return {'ink': np.clip(D_ + eye, 0, 1).astype(np.float32), 'water': None}


def rest(S):
    """休: a person standing at rest beside a tree"""
    P, D, X = _box(S)
    ss = StrokeSet(D, D)
    _foliage(ss, D, [X(.49, .14), X(.85, .14), X(.67, .08), X(.58, .2), X(.76, .2)], 0.1 * S, 70, seed=21, ink=0.42)
    ss.add(stroke(D, D, spline([X(.67, .92), X(.672, .6), X(.67, .38), X(.67, .06)], 8), 0.05 * S, ink=0.85, dry=0.45, profile=BLUNT, seed=1), 0, 1)
    for k, (a, b) in enumerate(((X(.67, .38), X(.49, .15)), (X(.67, .38), X(.85, .15)))):
        ss.add(stroke(D, D, [a, b], 0.03 * S, ink=0.8, dry=0.45, profile=PRESS_DOWN, seed=2 + k), 0, 1)
    for k, (a, b) in enumerate(((X(.67, .63), X(.49, .87)), (X(.67, .63), X(.85, .87)))):
        ss.add(stroke(D, D, [a, b], 0.032 * S, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=4 + k), 0, 1)
    # the person: a robe along the body line, the arm, a head with a topknot
    body = spline(np.array([X(.305, .18), X(.28, .36), X(.23, .62), X(.12, .93)], np.float32), 10)
    wdt = np.linspace(0.13, 0.26, len(body)) * S
    d = np.gradient(body, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    robe = np.vstack([body + nrm * wdt[:, None] / 2, (body - nrm * wdt[:, None] / 2)[::-1]])
    ss.add(pwash(D, D, [tuple(q) for q in robe], ink=0.62, soft=0.01 * S, seed=6, grain=0.4), 0, 1)
    ss.add(pwash(D, D, [X(.27, .3), X(.35, .3), X(.44, .58), X(.4, .64)], ink=0.5, soft=0.008 * S, seed=12, grain=0.4), 0, 1)   # sleeve
    ss.add(stroke(D, D, [tuple(q) for q in body], 0.018 * S, ink=0.85, dry=0.4, profile=PRESS_DOWN, seed=7), 0, 1)
    ss.add(stroke(D, D, [X(.28, .33), X(.37, .48), X(.43, .62)], 0.03 * S, ink=0.8, dry=0.35, profile=PRESS_DOWN, seed=8), 0, 1)
    circ = lambda c, r: [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 22, endpoint=False)]
    ss.add(pwash(D, D, circ(X(.315, .1), 0.055 * S), ink=0.85, soft=0.006 * S, seed=9, grain=0.15), 0, 1)
    ss.add(pwash(D, D, circ(X(.3, .035), 0.024 * S), ink=0.9, soft=0.004 * S, seed=10), 0, 1)
    ss.add(stroke(D, D, [X(.04, .95), X(.5, .955), X(.95, .95)], 0.012 * S, ink=0.4, dry=0.6, profile=SWEEP, seed=11), 0, 1)
    return {'ink': ss.render(9).astype(np.float32), 'water': None}


PAINTERS = {'山': mountain, '川': river, '木': tree, '鳥': bird, '休': rest}


def picture(ch, S):
    def build():
        out = PAINTERS[ch](S)
        return {k: v for k, v in out.items() if v is not None}
    return cached(f'picture_{ord(ch):x}_{S}_v3', build)


if __name__ == '__main__':
    # proof: each painting with its pictograph traced over it in vermilion
    import os
    from PIL import Image
    from ancient import ancient
    from engine import INK, PAPER, SEA, VERM
    S = 300
    tiles = []
    for ch in PAINTERS:
        pc = picture(ch, S)
        g = ancient(ch, 'oracle', S)
        img = np.ones(g.dens.shape + (3,), np.float32) * PAPER
        if 'water' in pc:
            img *= 1 - (pc['water'] * 0.5)[..., None] * (1 - SEA)
        img *= 1 - pc['ink'][..., None] * (1 - INK)
        img *= 1 - (g.dens * 0.9)[..., None] * (1 - VERM)
        tiles.append((img * 255).astype(np.uint8))
    out = os.environ.get('PREVIEW_DIR', '/tmp/claude-0/preview')
    Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(out, 'pictures.png'))
    print(os.path.join(out, 'pictures.png'))

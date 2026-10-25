"""ancient — the historical forms of the characters, redrawn by hand in code.

The shapes follow the comparison charts the reel is based on (oracle bone / greater seal (bronze) / lesser seal /
clerical forms, see data/ref_*.png) and are drawn in the manner of each medium:
  oracle    甲骨文  knife-cut into bone: thin, straight-sided lines, slightly pointed ends, polygonal curves
  bronze    金文    cast in bronze: heavy rounded lines, filled masses (dots, peaks)
  seal      篆書    the Qin lesser seal: even-width, smooth, symmetrical curves, elongated
  clerical  隷書    the Han clerical hand: wide and flat, thick-thin brushwork, flared "swallow-tail" endings
Forms are simplified illustrations of the period styles, not transcriptions of single artefacts.

Each form is a list of strokes in a unit box (x right, y down):
  ('L', pts)              polyline          ('C', pts)        smooth curve through the points
  ('O', cx, cy, rx, ry)   closed loop       ('F', pts)        filled polygon        ('D', cx, cy, r)  dot
A stroke may end with a dict of options: {'w': width scale, 'flare': (start, end)} (clerical thick-thin).
"""
import cv2
import numpy as np

from engine import blur, inkify, noise, spline
from glyphs import Glyph, _poly_param, _schedule

FORMS = {
    # ------------------------------------------------------------------ 日 sun
    ('日', 'oracle'): [('O', .5, .5, .36, .33), ('L', [(.37, .5), (.63, .5)])],
    ('日', 'bronze'): [('O', .5, .5, .33, .33), ('D', .5, .5, .075)],
    ('日', 'seal'): [('C', [(.29, .87), (.29, .32), (.31, .2), (.39, .13), (.61, .13), (.69, .2), (.71, .32), (.71, .87)]),
                    ('L', [(.29, .5), (.71, .5)]), ('L', [(.29, .87), (.71, .87)])],
    ('日', 'clerical'): [('L', [(.2, .3), (.21, .76)], {'w': 0.9}),
                        ('L', [(.21, .3), (.79, .29), (.8, .76)], {'w': 0.9}),
                        ('L', [(.23, .53), (.77, .52)], {'w': 0.8}),
                        ('C', [(.12, .78), (.34, .79), (.6, .78), (.9, .74)], {'w': 1.15, 'flare': (0.6, 1.5)})],
    # ------------------------------------------------------------------ 月 moon
    ('月', 'oracle'): [('C', [(.62, .07), (.42, .18), (.29, .5), (.42, .82), (.62, .93)]),
                      ('C', [(.62, .07), (.55, .3), (.53, .5), (.55, .7), (.62, .93)]), ('L', [(.42, .42), (.43, .57)])],
    # ------------------------------------------------------------------ 山 mountain
    ('山', 'oracle'): [('L', [(.08, .78), (.21, .42), (.34, .74), (.5, .18), (.66, .74), (.79, .42), (.92, .78)]),
                      ('L', [(.08, .78), (.92, .78)])],
    ('山', 'bronze'): [('F', [(.08, .8), (.21, .4), (.34, .72), (.5, .14), (.66, .72), (.79, .4), (.92, .8)])],
    ('山', 'seal'): [('L', [(.5, .12), (.5, .82)]),
                    ('C', [(.2, .36), (.2, .66), (.26, .8), (.5, .86), (.74, .8), (.8, .66), (.8, .36)])],
    # ------------------------------------------------------------------ 川 river
    ('川', 'oracle'): [('C', [(.24, .08), (.17, .3), (.26, .52), (.17, .74), (.23, .92)]),
                      ('C', [(.5, .18), (.43, .36), (.52, .54), (.44, .72), (.5, .84)]),
                      ('C', [(.76, .08), (.69, .3), (.78, .52), (.69, .74), (.75, .92)])],
    # ------------------------------------------------------------------ 木 tree
    ('木', 'oracle'): [('L', [(.5, .05), (.5, .95)]), ('L', [(.2, .13), (.5, .4), (.8, .13)]),
                      ('L', [(.2, .88), (.5, .62), (.8, .88)])],
    # ------------------------------------------------------------------ 人 person
    ('人', 'oracle'): [('C', [(.56, .06), (.52, .36), (.45, .64), (.32, .94)]), ('L', [(.51, .36), (.63, .52), (.7, .68)])],
    # ------------------------------------------------------------------ 雨 rain
    ('雨', 'oracle'): [('L', [(.12, .14), (.88, .14)]),
                      ('L', [(.3, .26), (.3, .4)]), ('L', [(.5, .26), (.5, .44)]), ('L', [(.7, .26), (.7, .4)]),
                      ('L', [(.3, .52), (.3, .64)]), ('L', [(.5, .58), (.5, .72)]), ('L', [(.7, .52), (.7, .64)]),
                      ('L', [(.3, .76), (.3, .88)]), ('L', [(.7, .76), (.7, .88)])],
    # ------------------------------------------------------------------ 魚 fish
    ('魚', 'oracle'): [('C', [(.5, .05), (.35, .19), (.3, .4), (.37, .58), (.5, .67)]),
                      ('C', [(.5, .05), (.65, .19), (.7, .4), (.63, .58), (.5, .67)]),
                      ('L', [(.38, .25), (.62, .5)]), ('L', [(.62, .25), (.38, .5)]),
                      ('L', [(.31, .35), (.16, .46)]), ('L', [(.69, .35), (.84, .46)]),
                      ('L', [(.5, .67), (.3, .93), (.7, .93), (.5, .67)])],
    # ------------------------------------------------------------------ lightning (the early form behind 申 / 電)
    ('申', 'oracle'): [('L', [(.5, .04), (.6, .2), (.43, .36), (.59, .52), (.42, .68), (.57, .84), (.5, .96)]),
                      ('C', [(.5, .29), (.34, .25), (.27, .33), (.31, .41)]), ('C', [(.51, .61), (.67, .65), (.74, .57), (.7, .49)])],
    # ------------------------------------------------------------------ 鳥 bird (side view, facing left)
    ('鳥', 'oracle'): [('C', [(.31, .25), (.35, .13), (.48, .09), (.58, .15)]), ('L', [(.31, .25), (.15, .28), (.32, .31)]),
                      ('D', .43, .19, .028),
                      ('C', [(.32, .31), (.3, .47), (.36, .62), (.5, .71), (.66, .74)]),
                      ('C', [(.58, .15), (.61, .3), (.69, .44), (.8, .59), (.92, .77)]),
                      ('L', [(.49, .35), (.71, .49)]), ('L', [(.47, .45), (.7, .6)]), ('L', [(.47, .55), (.66, .67)]),
                      ('L', [(.66, .74), (.92, .77)]),
                      ('L', [(.45, .69), (.41, .84), (.3, .9)]), ('L', [(.41, .84), (.47, .92)]), ('L', [(.55, .72), (.53, .86), (.45, .93)])],
}

STYLE = {  # base width (fraction of the box), end taper, polygonal curves, ink, edge pooling, roughness
    'oracle': dict(w=0.034, taper=0.45, polygonal=True, ink=0.93, pool=0.05, rough=0.14),
    'bronze': dict(w=0.075, taper=0.05, polygonal=False, ink=0.95, pool=0.22, rough=0.22),
    'seal': dict(w=0.05, taper=0.0, polygonal=False, ink=0.95, pool=0.08, rough=0.08),
    'clerical': dict(w=0.07, taper=0.0, polygonal=False, ink=0.95, pool=0.12, rough=0.14),
}


def _resample(P, step):
    P = np.asarray(P, np.float32)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step) + 1)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)


def _path(st, S, P, style, rng):
    """stroke spec -> (pixel polyline, filled polygon or None, options)"""
    kind = st[0]
    opts = st[-1] if isinstance(st[-1], dict) else {}
    tf = lambda pts: [(P + x * S, P + y * S) for x, y in pts]
    if kind == 'O':
        _, cx, cy, rx, ry = st[:5]
        n = 11 if style['polygonal'] else 48
        a0 = -np.pi / 2 - 0.25
        ang = np.linspace(a0, a0 + 2 * np.pi, n + 1)
        jit = rng.normal(0, 0.012, n + 1) if style['polygonal'] else np.zeros(n + 1)
        jit[-1] = jit[0]
        pts = [(cx + (rx + j) * np.cos(a), cy + (ry + j) * np.sin(a)) for a, j in zip(ang, jit)]
        return tf(pts), None, opts
    if kind == 'D':
        _, cx, cy, r = st[:4]
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)]
        return tf([(cx - r * 0.3, cy), (cx + r * 0.3, cy)]), tf(pts), opts
    pts = st[1]
    if kind == 'F':
        return tf(pts + [pts[0]]), tf(pts), opts
    if kind == 'C' and not style['polygonal']:
        pts = [tuple(p) for p in spline(np.asarray(pts, np.float32), 14)]
    elif kind == 'C':
        pts = [tuple(p) for p in spline(np.asarray(pts, np.float32), 3)]        # a knife cuts curves as facets
    return tf(pts), None, opts


def ancient(ch, style, size, seed=0, pad=0.06, gap=0.3):
    """the character `ch` in the historical `style`, as a Glyph (density, stroke-order time map, medians)"""
    spec = FORMS[(ch, style)]
    sty = STYLE[style]
    rng = np.random.default_rng(seed)
    P = int(size * pad)
    S = size
    D = S + 2 * P
    ss = 3
    paths = [_path(st, S, P, sty, rng) for st in spec]
    medians = [pl for pl, _, _ in paths]
    spans = _schedule(medians, gap)
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    mask = np.zeros((D, D), np.float32)
    T = np.ones((D, D), np.float32) * 2
    sidx = np.full((D, D), -1, np.int16)
    for k, (pl, fill, opts) in enumerate(paths):
        m = np.zeros((D * ss, D * ss), np.uint8)
        pts = _resample(pl, 1.0)
        n = len(pts)
        u = np.linspace(0, 1, n)
        w = sty['w'] * S * opts.get('w', 1.0)
        prof = np.ones(n)
        if sty['taper'] > 0:
            tp = sty['taper']
            prof *= np.clip(np.minimum(u, 1 - u) / 0.12, 0, 1) * tp + (1 - tp)
        if 'flare' in opts:                      # clerical swallow-tail: thin into the stroke, flaring at its end
            a, b = opts['flare']
            prof *= a + (b - a) * np.clip((u - 0.55) / 0.45, 0, 1) ** 2 + (u < 0.06) * 0.25 * (1 - u / 0.06)
        prof *= 1 + rng.normal(0, 0.04, n).cumsum() * 0.02
        for (x, y), p in zip(pts, prof):
            cv2.circle(m, (int(x * ss), int(y * ss)), max(1, int(w * p * ss / 2)), 255, -1, cv2.LINE_AA)
        if fill is not None:
            cv2.fillPoly(m, [np.round(np.asarray(fill) * ss).astype(np.int32)], 255, cv2.LINE_AA)
        mk = cv2.resize(m, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        on = mk > 0.02
        if fill is not None:                     # a filled mass fills from its first edge inward
            cx, cy = np.mean(np.asarray(fill), 0)
            r = np.hypot(xx[on] - cx, yy[on] - cy)
            fr = 1 - r / (r.max() + 1e-6)
        else:
            _, fr = _poly_param(xx[on], yy[on], pl)
        a, b = spans[k]
        tk = a + fr * (b - a)
        T[on] = np.where(sidx[on] >= 0, np.minimum(T[on], tk), tk)
        sidx[on] = k
        mask = np.maximum(mask, mk)
    dens = inkify(mask, seed=seed, rough=sty['rough'], density=sty['ink'], pool=sty['pool'], grain=0.12)
    T = blur(np.where(T > 1.5, 1.0, T), 0.8)
    g = Glyph(dens, T.astype(np.float32), sidx, medians, len(medians))
    g.spans = spans
    g.size, g.pad = size, P
    return g


def carve(g, light=0.5):
    """an engraved version of a glyph's density for a bone surface: (dark groove, light lip) density pair"""
    d = g.dens
    groove = np.clip(d * 1.1, 0, 1)
    lip = np.clip(np.roll(np.roll(blur(d, 1.2), 2, 0), 2, 1) - d, 0, 1) * light
    return groove, lip


STAGES = [  # the five stages of 日 shown in the evolution scene: (style, 漢字 name, English, period)
    ('oracle', '甲骨文', 'oracle bone script', 'Shang dynasty'),
    ('bronze', '金文', 'bronze script', 'Zhou dynasty'),
    ('seal', '篆書', 'seal script', 'Qin dynasty'),
    ('clerical', '隷書', 'clerical script', 'Han dynasty'),
    ('regular', '楷書', 'regular script', "today's form"),
]

if __name__ == '__main__':
    # proof sheet: every authored form, for checking against the reference charts
    import os
    from PIL import Image
    from engine import PAPER, INK
    keys = list(FORMS)
    cols = 6
    cell = 220
    rows = (len(keys) + cols - 1) // cols
    img = np.ones((rows * (cell + 30), cols * cell, 3), np.float32) * PAPER
    from PIL import ImageDraw, ImageFont
    for i, (ch, st) in enumerate(keys):
        g = ancient(ch, st, 180, seed=i)
        y, x = (i // cols) * (cell + 30), (i % cols) * cell
        sub = img[y + 30:y + 30 + g.h, x:x + g.w]
        a = g.dens[:sub.shape[0], :sub.shape[1], None]
        sub[:] = sub * (1 - a * (1 - INK))
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc', 20)
    for i, (ch, st) in enumerate(keys):
        d.text(((i % cols) * cell + 6, (i // cols) * (cell + 30) + 4), f'{ch} {st}', fill=(0, 0, 0), font=f)
    out = os.environ.get('PREVIEW_DIR', '/tmp/claude-0/preview')
    os.makedirs(out, exist_ok=True)
    im.save(os.path.join(out, 'ancient_forms.png'))
    print(os.path.join(out, 'ancient_forms.png'))

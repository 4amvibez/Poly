"""glyphs — characters written stroke by stroke, in real stroke order.

Two sources, both verified (never image-generated):
  * calligraphy(): kanji drawn from the Make Me a Hanzi stroke outlines + medians (hanzi-writer-data, Arphic
    Public License, see data/README.md).  Every stroke is its own vector outline, revealed along its median
    in the official stroke order, so the brush can be shown writing it.
  * typeset(): glyphs from Shippori Mincho B1 (a Japanese font with verified coverage).  Kana have no stroke
    database here, so their stroke medians are authored by hand in the font's 1024 em box (KANA_MEDIANS).

Both return a Glyph: density (inked, on washi), time map (when the brush reached each pixel, 0..1),
stroke index map, and the medians in pixel coordinates for driving a brush sprite.
"""
import json
import os
import re

import cv2
import numpy as np

from engine import FONT_JP, HERE, blur, inkify, noise, sstep

_DATA = None


def hanzi(ch):
    global _DATA
    if _DATA is None:
        _DATA = json.load(open(os.path.join(HERE, 'data', 'hanzi_strokes.json'), encoding='utf8'))
    return _DATA[ch]


# medians for kana in Shippori Mincho B1's em box (x right, y down, baseline at 900), in stroke order
KANA_MEDIANS = {
    'あ': [
        [(200, 205), (300, 240), (420, 250), (520, 265), (620, 225), (700, 175)],
        [(360, 95), (405, 125), (418, 200), (430, 300), (425, 420), (420, 560), (425, 690), (440, 760)],
        [(540, 410), (515, 540), (470, 650), (400, 740), (300, 810), (200, 855), (140, 790), (150, 660),
         (240, 540), (380, 440), (550, 385), (700, 390), (840, 470), (900, 620), (860, 780), (700, 880), (400, 940)],
    ],
    'イ': [
        [(560, 90), (640, 170), (560, 300), (400, 460), (130, 640)],
        [(590, 420), (575, 600), (562, 800), (575, 930)],
    ],
}


class Glyph:
    def __init__(self, dens, time, sidx, medians, nstrokes):
        self.dens, self.time, self.sidx, self.medians, self.n = dens, time, sidx, medians, nstrokes
        self.h, self.w = dens.shape

    def at(self, f, soft=0.02, wet=0.4):
        from engine import reveal_time
        return reveal_time(self.dens, self.time, f, soft=soft, wet=wet)

    def stroke_mask(self, idxs):
        """soft mask of the given strokes (for vermilion highlights / isolating a component)"""
        m = np.isin(self.sidx, list(idxs)).astype(np.float32)
        return blur(m, 1.2) * (self.dens > 0.02)

    def brush_pos(self, f):
        """(x, y, stroke index, fraction within stroke, moving) of the brush tip at progress f"""
        for i, (a, b) in enumerate(self.spans):
            if f < a:
                # lifted between strokes: hover from the previous end to the next start
                prev = self.medians[i - 1][-1] if i > 0 else self.medians[0][0]
                nxt = self.medians[i][0]
                pa = self.spans[i - 1][1] if i > 0 else 0
                x = np.clip((f - pa) / max(a - pa, 1e-6), 0, 1)
                x = x * x * (3 - 2 * x)
                return prev[0] + (nxt[0] - prev[0]) * x, prev[1] + (nxt[1] - prev[1]) * x, i, 0.0, False
            if f <= b:
                u = (f - a) / max(b - a, 1e-6)
                return (*_along(self.medians[i], u), i, u, True)
        last = self.medians[-1][-1]
        return last[0], last[1], self.n - 1, 1.0, False


def _along(pts, u):
    P = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1] * np.clip(u, 0, 1)
    return float(np.interp(L, s, P[:, 0])), float(np.interp(L, s, P[:, 1]))


def _poly_param(px, py, pts):
    """distance from pixels to a polyline and the arc-length fraction of the nearest point"""
    P = np.asarray(pts, np.float32)
    seg = np.diff(P, axis=0)
    ln = np.linalg.norm(seg, axis=1) + 1e-6
    cum = np.concatenate([[0], np.cumsum(ln)])
    best = np.full(px.shape, np.inf, np.float32)
    frac = np.zeros(px.shape, np.float32)
    for k in range(len(seg)):
        ax, ay = P[k]
        dx, dy = seg[k]
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (ln[k] ** 2), 0, 1)
        d = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2
        better = d < best
        best = np.where(better, d, best)
        frac = np.where(better, (cum[k] + t * ln[k]) / cum[-1], frac)
    return np.sqrt(best), frac


def _schedule(medians, gap=0.35):
    """time span of each stroke within 0..1: duration ~ length (with a floor), pauses between strokes"""
    lens = [max(float(np.linalg.norm(np.diff(np.asarray(m, np.float32), axis=0), axis=1).sum()), 60) for m in medians]
    unit = np.mean(lens) * gap
    total = sum(lens) + unit * (len(lens) - 1)
    spans, t = [], 0
    for L in lens:
        spans.append((t / total, (t + L) / total))
        t += L + unit
    return spans


def _svg_polys(path, scale, ox, oy, flip=900):
    """parse a hanzi-writer SVG path (absolute M/L/Q/C/Z) into polygons in pixel coordinates"""
    toks = re.findall(r'[MLQCZ]|-?\d+(?:\.\d+)?', path)
    polys, cur, i, pos = [], [], 0, (0, 0)
    tf = lambda x, y: (ox + x * scale, oy + (flip - y) * scale)
    while i < len(toks):
        c = toks[i]; i += 1
        if c == 'M':
            if cur: polys.append(cur)
            pos = (float(toks[i]), float(toks[i + 1])); i += 2; cur = [tf(*pos)]
        elif c == 'L':
            pos = (float(toks[i]), float(toks[i + 1])); i += 2; cur.append(tf(*pos))
        elif c == 'Q':
            c1 = (float(toks[i]), float(toks[i + 1])); p = (float(toks[i + 2]), float(toks[i + 3])); i += 4
            for t in np.linspace(0, 1, 9)[1:]:
                x = (1 - t) ** 2 * pos[0] + 2 * (1 - t) * t * c1[0] + t * t * p[0]
                y = (1 - t) ** 2 * pos[1] + 2 * (1 - t) * t * c1[1] + t * t * p[1]
                cur.append(tf(x, y))
            pos = p
        elif c == 'C':
            c1 = (float(toks[i]), float(toks[i + 1])); c2 = (float(toks[i + 2]), float(toks[i + 3]))
            p = (float(toks[i + 4]), float(toks[i + 5])); i += 6
            for t in np.linspace(0, 1, 11)[1:]:
                x = (1 - t) ** 3 * pos[0] + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t * t * c2[0] + t ** 3 * p[0]
                y = (1 - t) ** 3 * pos[1] + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t * t * c2[1] + t ** 3 * p[1]
                cur.append(tf(x, y))
            pos = p
        elif c == 'Z':
            if cur: polys.append(cur)
            cur = []
    if cur:
        polys.append(cur)
    return polys


def calligraphy(ch, size, seed=0, pad=0.08, ink=0.96, gap=0.35):
    """kanji written with the brush from Make Me a Hanzi outlines; canvas is size x size (+pad)"""
    d = hanzi(ch)
    P = int(size * pad)
    S = size + 2 * P
    scale = size / 1024
    ss = 3                                                    # supersample the outlines for clean edges
    mask = np.zeros((S, S), np.float32)
    sidx = np.full((S, S), -1, np.int16)
    T = np.ones((S, S), np.float32) * 2
    medians = [[(P + x * scale, P + (900 - y) * scale) for x, y in m] for m in d['medians']]
    spans = _schedule(medians, gap)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    for k, path in enumerate(d['strokes']):
        m = np.zeros((S * ss, S * ss), np.uint8)
        for poly in _svg_polys(path, scale * ss, P * ss, P * ss):
            cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 255, cv2.LINE_AA)
        mk = cv2.resize(m, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        on = mk > 0.02
        _, fr = _poly_param(xx[on], yy[on], medians[k])
        a, b = spans[k]
        tk = a + fr * (b - a)
        # later strokes paint over earlier ones where they cross
        T[on] = np.where(sidx[on] >= 0, np.minimum(T[on], tk), tk)
        sidx[on] = k
        mask = np.maximum(mask, mk)
    dens = inkify(mask, seed=seed, rough=0.16, density=ink, pool=0.12, grain=0.14)
    # a touch of dry-brush streaking along each stroke's tail
    T = blur(np.where(T > 1.5, 1.0, T), 0.8)
    g = Glyph(dens, T.astype(np.float32), sidx, medians, len(medians))
    g.spans = spans
    g.size, g.pad = size, P
    return g


_CMAPS = {}


def check_glyph(ch, font):
    """refuse to render a character the font does not contain (it would come out as a tofu box)"""
    if font not in _CMAPS:
        from fontTools.ttLib import TTFont
        _CMAPS[font] = TTFont(font, fontNumber=0, lazy=True).getBestCmap()
    if ord(ch) not in _CMAPS[font]:
        raise ValueError(f'glyph {ch!r} (U+{ord(ch):04X}) is missing from {os.path.basename(font)}')


def typeset(ch, size, font=FONT_JP, seed=0, pad=0.08, ink=0.95, medians=None, gap=0.3, sweep=(1, 0.6)):
    """a glyph from the Japanese font. medians (1024 em box) give a real stroke-order reveal; without them
    the glyph is revealed by a brush-like diagonal sweep"""
    from PIL import Image, ImageDraw, ImageFont
    check_glyph(ch, font)
    P = int(size * pad)
    S = size + 2 * P
    ss = 2
    im = Image.new('L', (S * ss, S * ss), 0)
    f = ImageFont.truetype(font, size * ss)
    ImageDraw.Draw(im).text((P * ss, (P + size * 900 / 1024) * ss), ch, font=f, fill=255, anchor='ls')
    mask = np.asarray(im.resize((S, S), Image.LANCZOS), np.float32) / 255
    medians = medians if medians is not None else KANA_MEDIANS.get(ch)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    on = mask > 0.02
    sidx = np.full((S, S), -1, np.int16)
    T = np.ones((S, S), np.float32)
    if medians:
        sc = size / 1024
        med = [[(P + x * sc, P + y * sc) for x, y in m] for m in medians]
        spans = _schedule(med, gap)
        best = np.full(on.sum(), np.inf, np.float32)
        tt = np.zeros(on.sum(), np.float32)
        si = np.zeros(on.sum(), np.int16)
        for k, m in enumerate(med):
            dist, fr = _poly_param(xx[on], yy[on], m)
            better = dist < best - 2                       # ties go to the earlier stroke
            best = np.where(better, dist, best)
            a, b = spans[k]
            tt = np.where(better, a + fr * (b - a), tt)
            si = np.where(better, k, si)
        T[on] = tt
        sidx[on] = si
    else:
        dx, dy = sweep
        T = (xx * dx + yy * dy) / (S * (abs(dx) + abs(dy)))
        T = (T - T[on].min()) / (T[on].max() - T[on].min() + 1e-6)
        T = np.clip(T + (noise(S, S, 30, 3, seed) - 0.5) * 0.12, 0, 1)
        sidx[on] = 0
        med = [[(P, P + size / 2), (P + size, P + size / 2)]]
        spans = [(0, 1)]
    dens = inkify(mask, seed=seed, rough=0.12, density=ink, pool=0.08, grain=0.1)
    g = Glyph(dens, blur(T, 0.8).astype(np.float32), sidx, med, len(med))
    g.spans = spans
    g.size, g.pad = size, P
    return g


def text_glyphs(text, size, font=FONT_JP, seed=0, tracking=0.04, ink=0.95):
    """a horizontal word as one density patch with a left-to-right brush sweep time map; returns
    (dens, time, char boxes [(x0, x1)])"""
    gs = [typeset(c, size, font=font, seed=seed + i, pad=0.04, ink=ink, medians=False) for i, c in enumerate(text)]
    step = int(size * (1 + tracking))
    P = gs[0].pad
    w = step * len(text) + 2 * P
    h = gs[0].h
    D = np.zeros((h, w), np.float32)
    T = np.ones((h, w), np.float32)
    boxes = []
    n = len(text)
    for i, g in enumerate(gs):
        x0 = i * step
        D[:, x0:x0 + g.w] = np.maximum(D[:, x0:x0 + g.w], g.dens)
        on = g.dens > 0.02
        sub = T[:, x0:x0 + g.w]
        sub[on] = np.minimum(sub[on], (i + g.time[on] * 0.9) / n)
        boxes.append((x0 + P, x0 + P + size))
    return D, T, boxes

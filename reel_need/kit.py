"""kit — small drawing helpers shared by this reel's scenes: opaque patches (paper cards, light), anisotropic placement
(card flips), brush arrows, paper cards with a rim, text glyph rows written chunk by chunk, and the thought cloud."""
import cv2
import numpy as np

from engine import FONT_JP, INK, StrokeSet, W, H, blur, spline, stroke
from glyphs import typeset
from poly_ink.brush import PRESS_DOWN

CARD = np.array([0.97, 0.95, 0.89], np.float32)
RIM = np.array([0.55, 0.42, 0.3], np.float32)
PAPER_C = np.array([0.95, 0.92, 0.85], np.float32)
GREEN = np.array([0.36, 0.56, 0.36], np.float32)


def affine(cam, x, y, w, h, sx=1.0, sy=1.0, rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    A = np.array([[c * sx, -s * sy, 0], [s * sx, c * sy, 0]], np.float32)
    A[:, 2] = [x - (A[0, 0] * w / 2 + A[0, 1] * h / 2), y - (A[1, 0] * w / 2 + A[1, 1] * h / 2)]
    M = cam.matrix()
    return np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)


def put(fr, cam, d, x, y, color=INK, op=1.0, sx=1.0, sy=None, rot=0.0, over=False):
    """ink (or, with over=True, paint opaquely) a patch centred at (x, y), with separate x / y scale"""
    if op <= 0 or d is None:
        return
    h, w = d.shape
    a = cv2.warpAffine(d.astype(np.float32), affine(cam, x, y, w, h, sx, sx if sy is None else sy, rot), (W, H))
    if over:
        fr.over(np.asarray(color, np.float32), a, op)
    else:
        fr.ink(a, color, op)


def box(h, w, soft=2.5):
    b = np.zeros((h, w), np.float32)
    b[4:-4, 4:-4] = 1
    return blur(b, soft)


def rim(h, w, inset=10, t=3):
    a, b = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    a[inset:-inset, inset:-inset] = 1
    b[inset + t:-inset - t, inset + t:-inset - t] = 1
    return np.clip(blur(a, 0.7) - blur(b, 0.7), 0, 1)


def card(fr, cam, B, R, x, y, op, sx=1.0, sy=1.0, color=CARD, shadow=0.25):
    """a paper card: soft shadow, opaque paper, a thin brown rim"""
    put(fr, cam, blur(B, 8), x + 6, y + 10, INK, shadow * op, sx, sy)
    put(fr, cam, B, x, y, color, op, sx, sy, over=True)
    put(fr, cam, R, x, y, RIM, 0.6 * op, sx, sy)


def arrow(ss, p0, p1, t0, dur, width=5, seed=0, red=False, head=16):
    """a brush arrow from p0 to p1 with a two-stroke head"""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2 + (y1 - y0) * 0.08, (y0 + y1) / 2 - (x1 - x0) * 0.08
    ss.add(stroke(W, H, spline([p0, (mx, my), p1], 10), width, ink=0.95, dry=0.35, profile=((0, 0.5), (0.1, 1), (1, 0.7)), seed=seed, red=red), t0, dur)
    a = np.arctan2(y1 - my, x1 - mx)
    for k, s_ in enumerate((-1, 1)):
        b = a + np.pi - s_ * 0.5
        ss.add(stroke(W, H, [(x1 + head * np.cos(b), y1 + head * np.sin(b)), (x1, y1)], width * 0.9, ink=0.95, dry=0.3,
                      profile=PRESS_DOWN, seed=seed + 1 + k, red=red), t0 + dur * 0.9, 0.12)


class Row:
    """a row of Japanese characters, each its own glyph, centred on (0, 0); chars can be written in, recoloured, moved"""

    def __init__(self, text, size, font=FONT_JP, seed=0, tracking=0.04, medians=False):
        self.text, self.size = text, size
        self.g = [typeset(c, size, font=font, seed=seed + i, pad=0.04, medians=medians, sweep=(1, 0.3)) for i, c in enumerate(text)]
        self.step = size * (1 + tracking)
        n = len(text)
        self.x = [(i - (n - 1) / 2) * self.step for i in range(n)]

    def draw(self, fr, cam, cx, cy, reveal, color=None, op=None, offset=None, scale=1.0):
        for i, g in enumerate(self.g):
            f = reveal(i)
            if f <= 0:
                continue
            o = 1.0 if op is None else op(i)
            if o <= 0:
                continue
            dx, dy = (0, 0) if offset is None else offset(i)
            put(fr, cam, g.at(min(f, 1.05), soft=0.05, wet=0.35), cx + (self.x[i] + dx) * scale, cy + dy * scale,
                INK if color is None else color(i), o, scale)


def chunk_reveal(t, spans, members):
    """reveal(i) writing members[k] (lists of char indices) across spans[k] = (t0, t1)"""
    where = {}
    for (a, b), mem in zip(spans, members):
        d = (b - a) / max(len(mem), 1)
        for j, i in enumerate(mem):
            where[i] = (a + j * d * 0.85, max(d * 1.4, 0.12))
    return lambda i: (lambda s: 0.0 if s is None else float(np.clip((t - s[0]) / s[1], 0, 1.2)))(where.get(i))

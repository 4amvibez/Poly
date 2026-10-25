"""The motion engine: timed brush strokes with wet ink, eased cameras with parallax, organic ink masks,
pigment-on-paper compositing, and typography.

World art is painted into float density buffers (0 = paper, 1 = full ink), exactly like poly_ink / inkpaint:
ink D and vermilion R combine multiplicatively with the paper. A camera samples the world buffers into the
1080x1920 frame, each layer with its own parallax depth.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import poly_ink  # noqa: F401
from core import blur
from inkpaint import Painting

W, H, FPS = 1080, 1920, 30
INK = np.array([28, 26, 24], np.float32) / 255
VERM = np.array([178, 52, 38], np.float32) / 255
PAPER = np.array([241, 233, 216], np.float32) / 255
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, 'assets', 'fonts')


# ------------------------------------------------------------------ easing and keyframes
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(x):
    """ease-in-out cubic: deliberate acceleration and deceleration"""
    x = float(np.clip(x, 0, 1))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, a, b):
    """0 before a, 1 after b, eased between"""
    return ease((t - a) / max(b - a, 1e-6))


class Keys:
    """keyframed values: Keys([(t, value), ...])(t) with eased interpolation; values may be tuples"""

    def __init__(self, keys):
        self.k = sorted(keys, key=lambda kv: kv[0])

    def __call__(self, t):
        k = self.k
        if t <= k[0][0]:
            return np.asarray(k[0][1], np.float64)
        for (t0, v0), (t1, v1) in zip(k, k[1:]):
            if t <= t1:
                f = ease((t - t0) / max(t1 - t0, 1e-6))
                return np.asarray(v0, np.float64) * (1 - f) + np.asarray(v1, np.float64) * f
        return np.asarray(k[-1][1], np.float64)


# ------------------------------------------------------------------ camera
def affine(cx, cy, z, out=(W, H)):
    """PIL affine data that shows world point (cx, cy) at the frame centre, z frame px per world px"""
    return (1 / z, 0, cx - out[0] / (2 * z), 0, 1 / z, cy - out[1] / (2 * z))


def view(arr, cam, out=(W, H), fill=0.0):
    """sample a float world buffer through camera cam=(cx, cy, z)"""
    im = Image.fromarray(np.ascontiguousarray(arr, np.float32), 'F')
    return np.array(im.transform(out, Image.AFFINE, affine(*cam, out=out), resample=Image.BILINEAR,
                                   fillcolor=fill), np.float32)


def view_rgb(img, cam, out=(W, H)):
    """sample a uint8 RGB world image (PIL) through the camera; returns float 0..1"""
    return np.array(img.transform(out, Image.AFFINE, affine(*cam, out=out), resample=Image.BILINEAR),
                      np.float32) / 255


def parallax(cam, ref, depth):
    """camera for a layer at `depth` (1 = main plane, <1 farther and slower, >1 nearer and faster)"""
    cx, cy, z = cam
    rx, ry, rz = ref
    return (rx + (cx - rx) * depth, ry + (cy - ry) * depth, rz * (z / rz) ** depth)


def to_screen(cam, x, y):
    cx, cy, z = cam
    return (x - cx) * z + W / 2, (y - cy) * z + H / 2


# ------------------------------------------------------------------ paper and compositing
def paper(w, h, seed=0, tone=PAPER, fibers=None):
    """aged washi: the inkpaint paper plus a faint sepia mottling; returns a PIL RGB image"""
    p = Painting(w, h, seed=seed)
    p.paper(fibers=fibers or int(w * h / 800), tone=tone)
    rgb = p.paper_rgb
    mott = p.noise2d(420, 4, seed + 3)
    rgb = rgb * (1 - 0.045 * mott[..., None] * (1 - np.array([0.2, 0.45, 0.85])))
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def over(buf, a):
    np.copyto(buf, 1 - (1 - buf) * (1 - np.clip(a, 0, 1)))


def compose(pap, D, R=None, ink=INK):
    """pigment on paper: ink darkens multiplicatively, vermilion tints"""
    img = pap * (1 - D[..., None] * (1 - ink))
    if R is not None:
        img = img * (1 - R[..., None] * (1 - VERM / PAPER))
    return img


def glaze(img, a, color):
    """transparent watercolour glaze (multiply) of `color` with alpha map a"""
    return img * (1 - a[..., None] * (1 - np.asarray(color, np.float32)))


def glow(img, a, color):
    """light (screen blend) — lantern glow, fire"""
    c = np.asarray(color, np.float32)
    return 1 - (1 - img) * (1 - a[..., None] * c)


# ------------------------------------------------------------------ timed strokes
class Track:
    """brush strokes in world coordinates, each written between t0 and t1 (seconds).
    density(t) gives the ink at time t, with wet ink that blooms slightly and darkens as it lands, then settles."""

    def __init__(self, w, h, wet=True):
        self.w, self.h, self.wet = w, h, wet
        self.items = []
        self._t, self._D, self._R, self._done = None, None, None, set()

    def add(self, written):
        for x in (written if isinstance(written, list) else [written]):
            self.items.append(x)
        self._t = None
        return self

    def stroke(self, st, t0, t1, red=False, path=None):
        from .glyph import Written
        if st is None:
            return self
        if isinstance(st, list):
            for s in st:
                self.stroke(s, t0, t1, red, path)
            return self
        p = np.asarray(path if path is not None else [(st.x0, st.y0), (st.x0 + 1, st.y0)], np.float32)
        return self.add(Written(st, t0, t1, p, red=red or st.red))

    @property
    def end(self):
        return max((x.t1 for x in self.items), default=0)

    def _paint(self, D, R, x, f, t):
        st = x.stroke
        dens = st.reveal(f, soft=0.04)
        h, w = dens.shape
        if self.wet and t - x.t1 < 0.6:
            # wet ink: the part the brush has just reached is darker and bleeds a little further
            reached = x.t0 + st.time * (x.t1 - x.t0)
            age = t - reached
            wetness = np.where(age >= 0, np.exp(-np.maximum(age, 0) / 0.28), 0).astype(np.float32)
            if not hasattr(st, 'halo'):
                st.halo = blur(st.dens, 1.5 + 0.03 * max(h, w) ** 0.5 * 3)
            dens = np.clip(dens * (1 + 0.1 * wetness) + 0.22 * st.halo * wetness, 0, 1)
        buf = (R if x.red else D)[st.y0:st.y0 + h, st.x0:st.x0 + w]
        np.copyto(buf, 1 - (1 - buf) * (1 - dens))

    def density(self, t):
        """(D, R) world buffers at time t; finished strokes are cached, so rendering forward in time is cheap"""
        if self._t is None or t < self._t:
            self._D = np.zeros((self.h, self.w), np.float32)
            self._R = np.zeros((self.h, self.w), np.float32)
            self._done = set()
        self._t = t
        for i, x in enumerate(self.items):
            if getattr(x, 'fade', None) is not None:
                continue
            if i not in self._done and t >= x.t1 + 0.6:
                self._paint(self._D, self._R, x, 1.0, 1e9)
                self._done.add(i)
        D, R = self._D.copy(), self._R.copy()
        for i, x in enumerate(self.items):
            if i in self._done or t <= x.t0:
                continue
            f = (t - x.t0) / max(x.t1 - x.t0, 1e-6)
            if getattr(x, 'fade', None) is not None:
                self._paint_faded(D, R, x, min(f, 1.0), t)
            else:
                self._paint(D, R, x, f, t)
        return D, R

    def _paint_faded(self, D, R, x, f, t):
        k = float(x.fade(t))
        if k <= 0.001:
            return
        st = x.stroke
        dens = st.reveal(f, soft=0.04) * k
        h, w = dens.shape
        buf = (R if x.red else D)[st.y0:st.y0 + h, st.x0:st.x0 + w]
        np.copyto(buf, 1 - (1 - buf) * (1 - dens))

    def tip(self, t):
        """(x, y) of the brush tip if a stroke with a path is being written at t"""
        live = [x for x in self.items if x.t0 <= t <= x.t1 and len(x.path) > 2]
        return live[-1].tip(t) if live else None

    def last_tip(self, t, hold=0.25):
        """tip position, or where the last stroke ended if it ended less than `hold` s ago (brush lifting)"""
        tp = self.tip(t)
        if tp is not None:
            return tp, 1.0
        done = [x for x in self.items if x.t1 < t <= x.t1 + hold and len(x.path) > 2]
        if done:
            x = max(done, key=lambda x: x.t1)
            return tuple(x.path[-1]), 1 - (t - x.t1) / hold
        return None, 0.0


# ------------------------------------------------------------------ organic masks
def noise(h, w, scale, octaves=4, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, int(h / scale * 2 ** o) + 2), max(2, int(w / scale * 2 ** o) + 2)
        g = Image.fromarray(r.random((gh, gw)).astype(np.float32), 'F')
        out += amp * np.asarray(g.resize((w, h), Image.BICUBIC), np.float32)
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


class InkFront:
    """an ink wash spreading across an area: field = distance from the origin (a point, a line or a direction)
    pushed around by fractal noise, so the front is a soft irregular bleed rather than a wipe.
    mask(p) is 0..1, with p = 0 nothing covered and p = 1 everything covered."""

    def __init__(self, h, w, origin=None, direction=None, seeds=None, rough=0.22, scale=180, soft=0.06, seed=0):
        YY, XX = np.mgrid[0:h, 0:w].astype(np.float32)
        if seeds is not None:                                  # several contact points (min distance)
            f = np.min([np.hypot(XX - x, YY - y) / r for x, y, r in seeds], axis=0)
        elif origin is not None:
            f = np.hypot(XX - origin[0], YY - origin[1])
        else:
            dx, dy = direction
            f = XX * dx + YY * dy
        f = (f - f.min()) / (f.max() - f.min() + 1e-6)
        n = noise(h, w, scale, 4, seed)
        fine = noise(h, w, max(4, scale / 6), 2, seed + 1)
        self.f = (f * (1 - rough) + rough * (0.88 * n + 0.12 * fine)).astype(np.float32)
        self.f = (self.f - self.f.min()) / (self.f.max() - self.f.min() + 1e-6)
        self.soft = soft

    def mask(self, p):
        p = float(p) * (1 + 2 * self.soft) - self.soft
        return np.clip((p - self.f) / self.soft + 0.5, 0, 1)

    def edge(self, p):
        """the wet rim at the front, where the ink has just arrived"""
        m = self.mask(p)
        return 4 * m * (1 - m)


# ------------------------------------------------------------------ typography
_fonts = {}


def font(name, size):
    key = (name, size)
    if key not in _fonts:
        path = {'serif': 'EBGaramond[wght].ttf', 'italic': 'EBGaramond-Italic[wght].ttf',
                'mincho': 'ShipporiMincho-Regular.ttf', 'mincho-bold': 'ShipporiMincho-Bold.ttf'}[name]
        f = ImageFont.truetype(os.path.join(FONTS, path), size)
        if name in ('serif', 'italic'):
            try:
                f.set_variation_by_axes([500])
            except Exception:
                pass
        _fonts[key] = f
    return _fonts[key]


def is_cjk(ch):
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF


def runs(text):
    out = []
    for ch in text:
        c = is_cjk(ch)
        if out and out[-1][0] == c:
            out[-1][1] += ch
        else:
            out.append([c, ch])
    return out


def text_width(text, size, latin='serif', cjk='mincho'):
    return sum(font(cjk if c else latin, int(size * (0.92 if c else 1))).getlength(s) for c, s in runs(text))


def draw_text(draw, xy, text, size, fill, latin='serif', cjk='mincho', anchor='ls'):
    """mixed Latin / Japanese text on one baseline (Japanese set in Shippori Mincho, Latin in EB Garamond)"""
    x, y = xy
    if anchor[0] == 'm':
        x -= text_width(text, size, latin, cjk) / 2
    for c, s in runs(text):
        f = font(cjk if c else latin, int(size * (0.92 if c else 1)))
        draw.text((x, y), s, font=f, fill=fill, anchor='ls')
        x += f.getlength(s)


def text_mask(w, h, text, xy, size, latin='serif', cjk='mincho', anchor='ms'):
    """a float mask (0..1) of typeset text, for typography that sits in the world as ink"""
    im = Image.new('L', (w, h), 0)
    draw_text(ImageDraw.Draw(im), xy, text, size, 255, latin, cjk, anchor)
    return np.asarray(im, np.float32) / 255


def wrap(text, size, max_w, balance=False):
    """greedy word wrap; balance=True splits a two-line caption where the lines come out most even"""
    if balance and text_width(text, size) > max_w:
        words = text.split(' ')
        best = min(range(1, len(words)), key=lambda k: max(text_width(' '.join(words[:k]), size),
                                                           text_width(' '.join(words[k:]), size)))
        a, b = ' '.join(words[:best]), ' '.join(words[best:])
        if max(text_width(a, size), text_width(b, size)) <= max_w:
            return [a, b]
    words, lines, cur = text.split(' '), [], ''
    for wd in words:
        trial = (cur + ' ' + wd).strip()
        if cur and text_width(trial, size) > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines

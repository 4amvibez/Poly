"""engine — the motion system shared by every scene of the reel.

Everything is computed with numpy / OpenCV: a frame is aged washi paper multiplied by layers of ink density,
exactly like pigment on paper (see poly_ink).  A layer lives in its scene's *world* coordinates and is
projected to the 1080x1920 frame by a camera (centre, zoom) with a per-layer parallax depth.

Motion vocabulary (reused across scenes):
    reveal_time()   a mark appears in the order it was painted (brush strokes, glyphs written stroke by stroke),
                    with a darker wet front that settles
    wash_map()      an organic reveal front (distance + fractal noise) for ink washes spreading across paper
    bloom()         ink spreading and being absorbed after it touches the paper
    wipe()          ink transitions between scenes driven by a time map (stroke shaped, curve, dissolve ...)
"""
import functools
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import poly_ink  # noqa: E402,F401  (puts the claude-drawing lib on sys.path)
from poly_ink import brush as pbrush  # noqa: E402
from core import noise2d as _noise2d, spline  # noqa: E402

W, H, FPS = 1080, 1920, 30
CACHE = os.path.join(HERE, 'cache')
os.makedirs(CACHE, exist_ok=True)

# palette: aged ivory paper, sumi ink, restrained vermilion, sepia, muted sea
PAPER = np.array([240, 232, 214], np.float32) / 255
INK = np.array([22, 20, 19], np.float32) / 255
GREY = np.array([92, 88, 84], np.float32) / 255
VERM = np.array([176, 56, 40], np.float32) / 255
SEPIA = np.array([142, 104, 66], np.float32) / 255
SEA = np.array([52, 94, 98], np.float32) / 255
SAKURA = np.array([206, 128, 124], np.float32) / 255

FONT_JP = os.path.join(HERE, 'fonts', 'ShipporiMinchoB1-ExtraBold-subset.ttf')
FONT_JP_FALLBACK = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
FONT_EN = os.path.join(HERE, 'fonts', 'CormorantGaramond-SemiBold.ttf')
FONT_EN_BOLD = os.path.join(HERE, 'fonts', 'CormorantGaramond-Bold.ttf')
FONT_EN_ITALIC = os.path.join(HERE, 'fonts', 'CormorantGaramond-MediumItalic.ttf')
FONT_HEAD = os.path.join(HERE, 'fonts', 'NotoSerifJP-Black-latin.ttf')          # headline serif (Latin)
FONT_BRUSH_EN = os.path.join(HERE, 'fonts', 'KaushanScript-Regular.ttf')         # brush-script Latin
FONT_BRUSH_JP = os.path.join(HERE, 'fonts', 'YujiSyuku-subset.ttf')              # brush-calligraphy kanji (JP forms)

# Instagram Reels safe area: keep key content inside these margins
SAFE_TOP, SAFE_BOTTOM, SAFE_LEFT, SAFE_RIGHT = 212, 1498, 80, 960   # Reels UI: top ~11%, bottom ~22%, right buttons


# ------------------------------------------------------------------ timing helpers
def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def ramp(t, a, b):
    """0 before a, 1 after b, linear between"""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp01((t - a) / (b - a))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    """cubic ease in-out: deliberate acceleration and deceleration"""
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def ease_sine(x):
    x = clamp01(x)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def lerp(a, b, x):
    return a + (b - a) * x


def keyframes(t, keys, ease=ease_io):
    """piecewise interpolation through [(time, value or tuple), ...] with easing on each segment"""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            x = ease((t - t0) / max(t1 - t0, 1e-6))
            if isinstance(v0, tuple):
                return tuple(lerp(a, b, x) for a, b in zip(v0, v1))
            return lerp(v0, v1, x)
    return keys[-1][1]


# ------------------------------------------------------------------ image helpers
def blur(a, s):
    if s <= 0.05:
        return a
    return cv2.GaussianBlur(a, (0, 0), float(s), borderType=cv2.BORDER_REFLECT)


@functools.lru_cache(maxsize=64)
def noise(h, w, scale, octaves=4, seed=0):
    """2D fractal value noise in [0, 1] (cached, treat as read-only)"""
    return _noise2d(h, w, scale, octaves, seed).astype(np.float32)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def cached(name, fn):
    """memoise an expensive painted asset on disk (npz of arrays, or dict of arrays)"""
    path = os.path.join(CACHE, name + '.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    out = fn()
    np.savez_compressed(path, **out)
    return out


# ------------------------------------------------------------------ paper
def make_paper(w, h, seed=0, tone=PAPER, stains=0.06):
    """aged washi: warm mottling, sepia age stains, fibres, fine grain"""
    def build():
        r = np.random.default_rng(seed)
        p = np.ones((h, w, 3), np.float32) * np.asarray(tone, np.float32)
        p *= (0.955 + 0.06 * noise(h, w, 300, 5, seed + 1))[..., None]
        # sepia age blotches, very soft
        st = sstep(0.55, 0.95, noise(h, w, 520, 4, seed + 2)) * stains
        p = p * (1 - st[..., None] * (1 - np.array([0.86, 0.76, 0.62], np.float32)))
        # fibres
        fib = np.zeros((h, w), np.float32)
        n = int(w * h / 1400)
        for _ in range(n):
            x, y = r.uniform(0, w), r.uniform(0, h)
            ang, ln = r.uniform(0, np.pi), r.uniform(10, 46)
            pts = np.array([(x + np.cos(ang + 0.3 * np.sin(t)) * t, y + np.sin(ang + 0.3 * np.sin(t)) * t)
                            for t in np.linspace(0, ln, 6)], np.int32)
            cv2.polylines(fib, [pts], False, float(r.uniform(0.015, 0.045)), 1, cv2.LINE_AA)
        p *= (1 - blur(fib, 0.8))[..., None]
        # a few brighter fibres (washi kozo strands)
        fib2 = np.zeros((h, w), np.float32)
        for _ in range(n // 6):
            x, y = r.uniform(0, w), r.uniform(0, h)
            ang, ln = r.uniform(0, np.pi), r.uniform(20, 70)
            pts = np.array([(x + np.cos(ang) * t + 3 * np.sin(t / 9), y + np.sin(ang) * t) for t in np.linspace(0, ln, 8)], np.int32)
            cv2.polylines(fib2, [pts], False, 1.0, 1, cv2.LINE_AA)
        p += blur(fib2, 0.7)[..., None] * 0.025
        p *= (1 + 0.018 * blur(r.normal(0, 1, (h, w)).astype(np.float32), 0.7))[..., None]
        return {'rgb': np.clip(p, 0, 1).astype(np.float32)}
    return cached(f'paper_{w}x{h}_{seed}_{int(stains * 100)}', build)['rgb']


# ------------------------------------------------------------------ camera
class Cam:
    """camera over a scene's world: (cx, cy) world point at screen centre, zoom = screen px per world px"""

    def __init__(self, cx, cy, zoom=1.0, rot=0.0):
        self.cx, self.cy, self.zoom, self.rot = cx, cy, zoom, rot

    def matrix(self, depth=1.0, ref=None):
        """affine world->screen for a layer at parallax `depth` (1 = moves with the camera, <1 farther,
        >1 nearer).  ref = world point where all depths coincide (defaults to the camera itself)."""
        rx, ry = ref if ref is not None else (self.cx, self.cy)
        cx = rx + (self.cx - rx) * depth
        cy = ry + (self.cy - ry) * depth
        z = self.zoom ** depth if depth != 1 else self.zoom
        c, s = np.cos(self.rot), np.sin(self.rot)
        # screen = R * z * (world - c) + centre
        return np.array([[z * c, -z * s, W / 2 - z * (c * cx - s * cy)],
                         [z * s, z * c, H / 2 - z * (s * cx + c * cy)]], np.float32)


def warp(src, M, border=0.0, interp=cv2.INTER_LINEAR, size=(W, H)):
    """project a world-space array (HxW or HxWxC float32) through affine M to the frame"""
    mode = cv2.BORDER_REFLECT if border == 'reflect' else cv2.BORDER_CONSTANT
    val = 0 if border == 'reflect' else border
    return cv2.warpAffine(src, M, size, flags=interp, borderMode=mode, borderValue=val)


def shift_matrix(M, ox, oy):
    """M for a sub-image whose top-left sits at world (ox, oy)"""
    M2 = M.copy()
    M2[:, 2] += M[:, 0] * ox + M[:, 1] * oy
    return M2


# ------------------------------------------------------------------ frame compositing
class Frame:
    """float RGB frame; ink layers multiply into it like pigment on paper"""

    def __init__(self, base):
        self.img = base.astype(np.float32, copy=True)

    def ink(self, dens, color=INK, opacity=1.0):
        if opacity <= 0:
            return
        a = np.clip(dens * opacity, 0, 1)[..., None]
        self.img *= 1 - a * (1 - np.asarray(color, np.float32))

    def over(self, rgb, alpha, opacity=1.0):
        if opacity <= 0:
            return
        a = np.clip(alpha * opacity, 0, 1)[..., None]
        self.img = self.img * (1 - a) + rgb * a

    def tint(self, color, amount):
        self.img = self.img * (1 - amount) + self.img * np.asarray(color, np.float32) * amount

    def finish(self, t, vignette=0.22, grain=0.012, seed=0):
        img = self.img * _vignette(vignette)[..., None]
        if grain > 0:
            g = _grain(int(t * FPS) % 24)
            img = img * (1 + grain * g[..., None])
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


@functools.lru_cache(maxsize=4)
def _vignette(strength):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    return (1 - strength * sstep(0.55, 1.35, d)).astype(np.float32)


@functools.lru_cache(maxsize=24)
def _grain(k):
    r = np.random.default_rng(1000 + k)
    return blur(r.normal(0, 1, (H, W)).astype(np.float32), 0.6)


# ------------------------------------------------------------------ reveal effects
def reveal_time(dens, T, f, soft=0.035, wet=0.35):
    """show the part of `dens` painted by progress f (T = when each pixel was painted, 0..1).
    The leading edge is darker and wetter, then settles into the stable mark."""
    if f <= 0:
        return np.zeros_like(dens)
    d = f - T
    m = np.clip(d / soft + 0.5, 0, 1)
    out = dens * m
    if wet > 0 and f < 1.0 + soft * 4:
        front = np.exp(-(d / (soft * 2.5)) ** 2) * m
        out = out + wet * dens * front * (1 - out)
    return np.clip(out, 0, 1)


def wash_map(h, w, origin, scale=1.0, noise_amp=0.35, noise_scale=140, seed=0, direction=None):
    """time map for an ink wash spreading from `origin` (x, y), or sweeping along `direction` (dx, dy):
    0 where the ink lands first, 1 where it arrives last, with an organic fractal front"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if direction is not None:
        dx, dy = direction
        n = np.hypot(dx, dy)
        d = (xx * dx + yy * dy) / n
    else:
        d = np.hypot(xx - origin[0], yy - origin[1])
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    nz = noise(h, w, noise_scale, 5, seed) - 0.5
    T = d * scale + nz * noise_amp
    T = (T - T.min()) / (T.max() - T.min() + 1e-6)
    return T.astype(np.float32)


def bloom(dens, age, sigma=6.0, amount=0.32, dur=0.9, halo=None):
    """ink spreading into the fibres after it lands: a feathered halo grows, then is absorbed and stays faint"""
    if age <= 0:
        return dens
    g = ease_out(age / dur)
    hal = halo if halo is not None else blur(dens, sigma)
    a = amount * g * (1 - 0.45 * smooth((age - dur) / (dur * 1.5)))
    return np.clip(dens + hal * a * (1 - dens), 0, 1)


def wipe_mix(a, b, T, f, soft=0.05, edge=0.5, edge_color=INK):
    """transition frame: show b where T < f, a elsewhere; an ink-dark wet edge rides the front"""
    d = f - T
    m = np.clip(d / soft + 0.5, 0, 1)[..., None]
    out = a * (1 - m) + b * m
    if edge > 0:
        ring = np.exp(-(d / (soft * 1.6)) ** 2)[..., None] * edge
        out = out * (1 - ring * (1 - np.asarray(edge_color, np.float32)))
    return out


# ------------------------------------------------------------------ brush strokes (from poly_ink)
def stroke(w, h, pts, width, **kw):
    """a poly_ink bristle stroke on a w x h canvas -> (dens, time, x0, y0) or None"""
    s = pbrush.brush_stroke(w, h, pts, width, **kw)
    if s is None:
        return None
    return s


def paste(canvas, st, value=None):
    """composite a poly_ink Stroke (or density patch) into a full canvas with ink 'over'"""
    d = st.dens if value is None else value
    hh, ww = d.shape
    buf = canvas[st.y0:st.y0 + hh, st.x0:st.x0 + ww]
    np.copyto(buf, 1 - (1 - buf) * (1 - d[:buf.shape[0], :buf.shape[1]]))


class StrokeSet:
    """strokes with their own timing, rendered into a full canvas at any time.
    Each entry: (Stroke, start_s, dur_s).  Supports bloom after landing."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []

    def add(self, st, start, dur, bloom_amt=0.0):
        for s in (st if isinstance(st, list) else [st]):
            if s is not None:
                halo = blur(s.dens, 2.5 + s.dens.shape[0] * 0.01) if bloom_amt > 0 else None
                self.items.append((s, start, dur, bloom_amt, halo))
        return start + dur

    @property
    def end(self):
        return max((s + d for _, s, d, _, _ in self.items), default=0)

    def render(self, t, red=None):
        if t > self.end + 3.0 and red is None:
            if getattr(self, '_final', None) is None:
                self._final = self.render(self.end + 2.99)
            return self._final
        D = np.zeros((self.h, self.w), np.float32)
        for s, st, du, bam, halo in self.items:
            if red is not None and bool(s.red) != red:
                continue
            f = (t - st) / max(du, 1e-6)
            if f <= 0:
                continue
            d = reveal_time(s.dens, s.time, f, soft=0.05, wet=0.3) if f < 1.2 else s.dens
            if bam > 0:
                d = bloom(d, t - st - du * 0.5, amount=bam, halo=halo)
            paste(D, s, d)
        return D


# ------------------------------------------------------------------ text
def text_layer(text, font_path, size, color=(1, 1, 1), anchor='mm', stroke_w=0, index=0, tracking=0):
    """render text to (alpha, (w, h)) with PIL, tight-cropped with padding"""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(font_path, size, index=index)
    if tracking:
        widths = [f.getlength(c) for c in text]
        tw = int(sum(widths) + tracking * (len(text) - 1)) + 40
    else:
        tw = int(f.getlength(text)) + 40
    th = int(size * 1.6) + 20
    im = Image.new('L', (tw, th), 0)
    d = ImageDraw.Draw(im)
    if tracking:
        x = 20
        for c, cw in zip(text, widths):
            d.text((x, th / 2), c, font=f, fill=255, anchor='lm')
            x += cw + tracking
    else:
        d.text((20, th / 2), text, font=f, fill=255, anchor='lm', stroke_width=stroke_w, stroke_fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.01)
    if len(xs) == 0:
        return a, (0, 0)
    pad = 6
    a = a[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
    return a, a.shape[::-1]


def place(frame_arr, patch, cx, cy, op='max'):
    """place patch (h, w) centred at (cx, cy) into a frame-sized array; returns the array"""
    ph, pw = patch.shape[:2]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    fh, fw = frame_arr.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(fw, x0 + pw), min(fh, y0 + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return frame_arr
    sub = patch[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    if op == 'max':
        np.maximum(frame_arr[dy0:dy1, dx0:dx1], sub, out=frame_arr[dy0:dy1, dx0:dx1])
    else:
        frame_arr[dy0:dy1, dx0:dx1] = sub
    return frame_arr


def inkify(mask, seed=0, rough=0.22, density=0.95, pool=0.10, grain=0.12):
    """turn a crisp glyph / shape mask into an ink mark on washi: irregular feathered edge, uneven density,
    ink pooling slightly at the edges"""
    h, w = mask.shape
    m = blur(mask, 1.1)
    n1 = noise(h, w, 9, 3, seed) - 0.5
    m = sstep(0.32, 0.68, m + n1 * rough)
    body = (1 - grain) + grain * noise(h, w, 40, 4, seed + 1)
    d = m * body * density
    if pool > 0:
        d = d + pool * np.clip(blur(m, 1.5) - blur(m, 4.5), 0, None) * 2.0
    d = np.maximum(d, blur(m, 2.4) * 0.12)                      # faint absorption halo into the fibres
    return np.clip(d, 0, 1).astype(np.float32)

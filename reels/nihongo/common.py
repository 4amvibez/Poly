"""Shared scene machinery: the Scene base class, sprites (the writing brush, the ship), screen-space pasting."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import poly_ink  # noqa: F401
from core import blur
from poly_ink.brush import BLUNT, PRESS_DOWN, brush_stroke, splatter, wash

from .ink import (FPS, H, PAPER, W, InkFront, Track, compose, ease, noise, over, paper, ramp, text_mask, to_screen,
                  view, view_rgb, parallax)


class Scene:
    """A scene paints its world once (build) and then renders any moment of itself (frame).
    World art is drawn at S x the frame size, so camera push-ins stay sharp. Coordinates given in frame units
    (0..1080, 0..1920) go through self.u()."""
    S = 1.3
    PAD = 0          # extra world margin (world px) around the frame-unit layout, for wide camera moves
    seed = 0

    def __init__(self, tl, n):
        self.tl, self.n = tl, n
        info = tl.scenes[n]
        self.dur, self.L, self.trans = info['dur'], info['lines'], info['trans']
        self.ww, self.wh = self.world_size()
        self.track = Track(self.ww, self.wh)
        self.build()

    def world_size(self):
        return int(W * self.S) + 2 * self.PAD, int(H * self.S) + 2 * self.PAD

    # helpers
    def u(self, *v):
        """frame units -> world px: a point (x, y) is offset by the padding, a single length is only scaled"""
        return tuple(float(a) * self.S + self.PAD for a in v) if len(v) > 1 else float(v[0]) * self.S

    def U(self, pts):
        return [(x * self.S + self.PAD, y * self.S + self.PAD) for x, y in pts]

    def wt(self, i, word, after=0):
        """local time of `word` in global line i"""
        return self.tl.word_time(i, word, self.n, after)

    def clamp(self, cam):
        """keep the camera inside the painted world"""
        cx, cy, z = (float(v) for v in cam)
        z = max(z, W / self.ww, H / self.wh)
        hw, hh = W / (2 * z), H / (2 * z)
        return (float(np.clip(cx, hw, self.ww - hw)), float(np.clip(cy, hh, self.wh - hh)), z)

    def base_cam(self):
        return (self.ww / 2, self.wh / 2, 1 / self.S)

    def cam(self, t):
        return self.base_cam()

    def build(self):
        pass

    def frame(self, t):
        raise NotImplementedError

    def enter(self, t, p):
        """mask (frame size) of how much of this scene shows while it enters; p goes 0 -> 1"""
        if not hasattr(self, '_front'):
            self._front = InkFront(H // 4, W // 4, origin=(W // 8, H // 8), rough=0.3, scale=80, soft=0.12,
                                   seed=self.n)
        m = self._front.mask(ease(p))
        return np.asarray(Image.fromarray(m, 'F').resize((W, H), Image.BILINEAR), np.float32)

    # rendering helpers
    def ink_view(self, cam, D, R=None):
        return view(D, cam), (view(R, cam) if R is not None else None)


def paste(img, rgb, a, x, y):
    """alpha-paste a float sprite (rgb h x w x 3, a h x w) into img with its top-left at (x, y)"""
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return img
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - sa) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * sa
    return img


def sprite_from(im):
    """PIL RGBA -> (rgb float, alpha float)"""
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3]


def make_brush(length=760, angle=-28):
    """a calligraphy brush: bamboo handle with nodes, dark ferrule, a black wet tip. Returns (PIL RGBA, tip
    offset (x, y) within the image) with the brush leaning up to the right."""
    s = 2
    L, Wd = length * s, 44 * s
    im = Image.new('RGBA', (Wd + 40 * s, L + 40 * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = im.width // 2
    hair_len, fer_len = int(L * 0.17), int(L * 0.05)
    top = 20 * s
    # handle: bamboo, warm tan with a darker edge, node rings
    hx0, hx1 = cx - Wd // 2 + 6 * s, cx + Wd // 2 - 6 * s
    handle_bottom = top + L - hair_len - fer_len
    for i in range(hx1 - hx0):
        f = i / (hx1 - hx0)
        shade = 0.62 + 0.38 * np.sin(np.pi * (0.15 + 0.7 * f))
        col = tuple(int(c * shade) for c in (176, 136, 84))
        d.line([(hx0 + i, top), (hx0 + i, handle_bottom)], fill=col + (255,))
    for yy in np.linspace(top + 120 * s, handle_bottom - 60 * s, 3):
        d.rectangle([hx0, yy - 3 * s, hx1, yy + 3 * s], fill=(110, 80, 46, 255))
        d.line([(hx0, yy + 4 * s), (hx1, yy + 4 * s)], fill=(214, 178, 120, 255), width=s)
    d.rounded_rectangle([hx0, top - 4 * s, hx1, top + 18 * s], 8 * s, fill=(60, 34, 22, 255))      # end cap
    # ferrule
    d.rectangle([hx0 - 2 * s, handle_bottom, hx1 + 2 * s, handle_bottom + fer_len], fill=(52, 30, 22, 255))
    d.line([(hx0 - 2 * s, handle_bottom + 6 * s), (hx1 + 2 * s, handle_bottom + 6 * s)], fill=(120, 80, 50, 255),
           width=2 * s)
    # hair: a full belly tapering to a sharp, wet point
    y0 = handle_bottom + fer_len
    tip_y = y0 + hair_len
    pts = []
    for k in np.linspace(0, 1, 40):
        w = (Wd * 0.5) * (1 + 0.28 * np.sin(np.pi * min(1.0, k * 1.3))) * (1 - k ** 2.4) ** 0.75 + 0.5
        pts.append((cx + w, y0 + k * hair_len))
    pts += [(2 * cx - x, y) for x, y in reversed(pts)]
    d.polygon(pts, fill=(18, 16, 15, 255))
    d.line([(cx - 6 * s, y0 + 12 * s), (cx - 4 * s, y0 + hair_len * 0.6)], fill=(64, 60, 56, 255), width=2 * s)   # sheen
    im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    # find the tip after rotation: rotate the tip point about the image centre the way PIL does
    # (simpler: rotate a marker image the same way)
    mk = Image.new('L', (Wd + 40 * s, L + 40 * s), 0)
    ImageDraw.Draw(mk).ellipse([cx - 3 * s, tip_y - 6 * s, cx + 3 * s, tip_y], fill=255)
    mk = np.asarray(mk.rotate(angle, resample=Image.BILINEAR, expand=True), np.float32)
    ys, xs = np.nonzero(mk > 128)
    tip = (xs.mean() / s, ys.max() / s)
    im = im.resize((im.width // s, im.height // s), Image.LANCZOS)
    return im, tip


class BrushTool:
    """the writing brush that follows a Track's stroke tip in screen space, with a soft shadow on the paper"""

    def __init__(self, scale=1.0, angle=-28):
        im, self.tip = make_brush(int(760 * scale), angle)
        self.rgb, self.a = sprite_from(im)
        sh = Image.fromarray((self.a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9))
        self.shadow = np.asarray(sh, np.float32) / 255 * 0.28
        self.last = None

    def draw(self, img, track, t, cam, presence=1.0):
        pos, a = track.last_tip(t, hold=0.35)
        if pos is None:
            # between strokes: hover over where the next stroke starts
            nxt = [x for x in track.items if x.t0 > t and len(x.path) > 2 and x.t0 - t < 0.35]
            if nxt:
                x = min(nxt, key=lambda x: x.t0)
                pos, a = tuple(x.path[0]), 1.0
        if pos is None:
            return img
        sx, sy = to_screen(cam, *pos)
        lift = 0.0 if track.tip(t) is not None else 10.0
        alpha = presence * min(1.0, a)
        tx, ty = self.tip
        sh_a = self.shadow * alpha
        img = paste(img, np.zeros_like(self.rgb) + 0.15, sh_a, sx - tx + 26, sy - ty + 14)
        return paste(img, self.rgb, self.a * alpha, sx - tx, sy - ty - lift)


def ink_sprite(w, h, draw_fn, seed=0, ink_rgb=(28, 26, 24)):
    """paint with poly_ink strokes into a transparent sprite: draw_fn(W, H) -> list of Stroke / (Stroke, red).
    Returns (rgb, alpha) floats with the ink as alpha."""
    D = np.zeros((h, w), np.float32)
    R = np.zeros((h, w), np.float32)
    for st in draw_fn(w, h):
        red = False
        if isinstance(st, tuple):
            st, red = st
        if st is None:
            continue
        dh, dw = st.dens.shape
        buf = (R if red or st.red else D)[st.y0:st.y0 + dh, st.x0:st.x0 + dw]
        np.copyto(buf, 1 - (1 - buf) * (1 - st.dens))
    a = 1 - (1 - D) * (1 - R)
    ink = np.asarray(ink_rgb, np.float32) / 255
    rgb = np.where(a[..., None] > 1e-4, (D[..., None] * ink + R[..., None] * np.array([0.7, 0.2, 0.15])) /
                   np.maximum(D + R, 1e-4)[..., None], ink)
    return rgb.astype(np.float32), np.clip(a, 0, 1).astype(np.float32)


def mist_layer(h, w, seed=0):
    """a wide tileable mist texture to drift across a frame"""
    n = noise(h, w * 2, 260, 4, seed)
    return n


class Labels:
    """typeset text placed in the world, revealed by a soft left-to-right ink wipe and drawn through the camera"""

    def __init__(self, ww, wh):
        self.ww, self.wh = ww, wh
        self.items = []

    def add(self, text, x, y, size, t0, dur=0.7, latin='serif', cjk='mincho', anchor='ms', red=False, ink=0.85,
            t_out=None, color=None):
        from .ink import text_mask
        pad = int(size * 1.2)
        w = int(max(64, size * len(text) * 1.1 + 2 * pad))
        h = int(size * 2.2)
        ox = {'m': w / 2, 'l': pad, 'r': w - pad}[anchor[0]]
        mk = text_mask(w, h, text, (ox, h * 0.68), size, latin, cjk, anchor)
        ys, xs = np.nonzero(mk > 0.01)
        if len(xs) == 0:
            return self
        x0, y0, x1, y1 = xs.min() - 3, ys.min() - 3, xs.max() + 4, ys.max() + 4
        crop = mk[y0:y1, x0:x1] * ink
        wx, wy = x - ox + x0, y - h * 0.68 + y0
        self.items.append(dict(m=crop, x=wx, y=wy, t0=t0, dur=dur, red=red, t_out=t_out, color=color))
        return self

    def draw(self, img, cam, t):
        from .ink import INK, PAPER, VERM, to_screen
        z = cam[2]
        for it in self.items:
            a = ramp(t, it['t0'], it['t0'] + it['dur'])
            if it['t_out'] is not None:
                a *= 1 - ramp(t, it['t_out'], it['t_out'] + 0.5)
            if a <= 0:
                continue
            m = it['m']
            reveal = np.clip((a * 1.35 - np.linspace(0, 1, m.shape[1])[None, :]) / 0.35, 0, 1) * min(1, a * 1.5)
            sx, sy = to_screen(cam, it['x'], it['y'])
            sw, sh = int(round(m.shape[1] * z)), int(round(m.shape[0] * z))
            if sw < 2 or sh < 2:
                continue
            ms = np.asarray(Image.fromarray((m * reveal).astype(np.float32), 'F').resize((sw, sh), Image.BILINEAR))
            xa, ya = int(round(sx)), int(round(sy))
            xb, yb, xe, ye = max(0, xa), max(0, ya), min(W, xa + sw), min(H, ya + sh)
            if xe <= xb or ye <= yb:
                continue
            k = ms[yb - ya:ye - ya, xb - xa:xe - xa, None]
            sl = img[yb:ye, xb:xe]
            if it['color'] is not None:          # opaque paint (light text on a dark band)
                sl[:] = sl * (1 - k) + np.asarray(it['color'], np.float32) * k
            else:
                col = VERM / PAPER if it['red'] else INK
                sl[:] = sl * (1 - k * (1 - col))
        return img


def directional_front(angle_deg=20, streak=True, soft=0.07, rough=0.3, seed=0):
    """an InkFront that sweeps across the frame like one huge dry brush: bristle streaks along the stroke"""
    from .ink import InkFront
    h, w = H // 4, W // 4
    a = np.radians(angle_deg)
    fr = InkFront(h, w, direction=(np.cos(a), np.sin(a)), rough=0.0, soft=soft, seed=seed)
    if streak:
        n = noise(h, 6, 3, 3, seed + 5)
        n = np.asarray(Image.fromarray(n, 'F').resize((w, h), Image.BICUBIC), np.float32)
        n = np.asarray(Image.fromarray(n, 'F').rotate(-angle_deg, resample=Image.BILINEAR), np.float32)
        fr.f = (fr.f * (1 - rough) + rough * n).astype(np.float32)
        fr.f = (fr.f - fr.f.min()) / (fr.f.max() - fr.f.min() + 1e-6)
    return fr


def aged_paper(w, h, seed=0, tone=(0.905, 0.835, 0.70)):
    """manuscript paper: sepia tone, foxing stains, darker worn edges; returns PIL RGB"""
    from .ink import paper
    p = np.asarray(paper(w, h, seed=seed, tone=np.array(tone, np.float32)), np.float32) / 255
    n = noise(h // 4, w // 4, 40, 5, seed + 7)
    n = np.asarray(Image.fromarray(n, 'F').resize((w, h), Image.BICUBIC), np.float32)
    stain = np.clip((n - 0.62) * 4, 0, 1) ** 1.5 * 0.16
    rim = np.clip((n - 0.62) * 4, 0, 1) - np.clip((n - 0.66) * 4, 0, 1)
    p = p * (1 - (stain + rim * 0.05)[..., None] * (1 - np.array([0.62, 0.45, 0.25])))
    YY, XX = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum(np.minimum(XX, w - XX), np.minimum(YY, h - YY)) / (0.12 * min(w, h))
    p = p * (1 - 0.12 * np.clip(1 - edge, 0, 1)[..., None] * (1 - np.array([0.7, 0.55, 0.38])))
    return Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8))

"""A fast bristle brush for calligraphy and big expressive strokes, plus a scene that can be "written" stroke
by stroke.

Each stroke is many bristles dragged along a smoothed path. Bristles run dry towards the end of the stroke and
skip in patches (飞白, flying white). The pressure profile swells and thins the stroke (顿笔 / 出锋), and wet
ink bleeds into the paper. Every stroke also records *when* the brush reached each pixel, so an animation can
reveal it in the order it was painted instead of fading it in.
"""
import numpy as np
from core import blur, fbm1d, spline
from inkpaint import Painting

# pressure profiles: (t along the stroke 0..1, width factor)
PRESS_DOWN = ((0, 0.55), (0.06, 1.0), (0.75, 0.9), (1, 0.15))       # press, then lift into a sharp tip (出锋)
BLUNT = ((0, 0.6), (0.06, 1.0), (0.9, 0.95), (1, 0.7))              # press and stop (藏锋)
SWEEP = ((0, 0.25), (0.25, 1.0), (0.7, 0.85), (1, 0.1))             # light entry, full belly, thin exit
HOOK = ((0, 0.6), (0.06, 1.0), (0.8, 0.95), (0.88, 1.1), (1, 0.1))   # heavy turn into a flicked hook (钩)
RISE = ((0, 0.08), (0.18, 0.9), (0.3, 1.0), (0.6, 0.85), (1, 0.12))  # fast pointed entry, full belly, thin dry curl


class Stroke:
    """one painted stroke, stored only over its bounding box"""

    def __init__(self, x0, y0, dens, time, red=False):
        self.x0, self.y0, self.dens, self.time, self.red = x0, y0, dens, time, red

    def reveal(self, f, soft=0.03):
        """density with only the part painted by fraction f (0..1) of the stroke shown"""
        if f >= 1:
            return self.dens
        if f <= 0:
            return np.zeros_like(self.dens)
        return self.dens * np.clip((f - self.time) / soft + 0.5, 0, 1)


def brush_stroke(W, H, pts, width, ink=0.95, dry=0.3, profile=PRESS_DOWN, bristles=None, wobble=0.06,
                 wet=0.25, fill=1.0, smooth=24, seed=0, red=False):
    """Paint a bristle stroke along the control points `pts` (Catmull-Rom smoothed).
    width: full width in px. ink: darkness 0..1. dry: 0 = fully loaded, 0.7 = very dry, broken flying white.
    bristles: fewer than about width/3 leaves visible bristle streaks (default: dense).
    fill: how much ink pools between the bristles (1 = solid calligraphy, 0 = separate hair lines).
    wet: ink bleed into the paper. Returns a Stroke, or None if it falls off the canvas."""
    r = np.random.default_rng(seed)
    path = spline(pts, smooth) if len(pts) > 2 else np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    L = float(seg.sum())
    if L < 2:
        return None
    pa = np.asarray(profile, np.float32)
    s = np.concatenate([[0], np.cumsum(seg)])
    # sample so that no bristle moves more than ~0.7 px between dabs, even on the outside of a tight turn
    fine = np.linspace(0, L, int(L * 2) + 2)
    fx_, fy_ = np.interp(fine, s, path[:, 0]), np.interp(fine, s, path[:, 1])
    ang = np.unwrap(np.arctan2(np.gradient(fy_), np.gradient(fx_)))
    fw = width * np.interp(fine / L, pa[:, 0], pa[:, 1])
    eff = np.concatenate([[0], np.cumsum(np.diff(fine) + np.abs(np.diff(ang)) * fw[1:] * 0.55 + np.abs(np.diff(fw)) * 0.5)])
    ss = np.interp(np.arange(0, eff[-1], 0.7), eff, fine).astype(np.float32)
    n = len(ss)
    px, py = np.interp(ss, s, path[:, 0]), np.interp(ss, s, path[:, 1])
    tx, ty = np.gradient(px), np.gradient(py)
    tl = np.hypot(tx, ty) + 1e-6
    tx, ty = tx / tl, ty / tl
    # the brush body can't turn on the spot: smooth its direction over about a third of its width so the
    # bristles sweep round a corner instead of fanning out from it
    k = max(1, int(width * 0.35 / 0.7))
    ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    sm = lambda a: np.convolve(np.pad(a, 3 * k, mode='edge'), ker / ker.sum(), mode='valid')
    tx, ty = sm(tx), sm(ty)
    tl = np.hypot(tx, ty) + 1e-6
    nx, ny = -ty / tl, tx / tl
    t = ss / L
    w = width * np.interp(t, pa[:, 0], pa[:, 1])
    step = np.gradient(ss)[None, :]

    nb = int(bristles or np.clip(width * 1.4, 6, 320))
    offs = np.linspace(-0.5, 0.5, nb) + r.normal(0, 0.35 / nb, nb)
    edge = np.abs(offs) * 2                                               # 0 at the centre, 1 at the edge
    drift = np.stack([fbm1d(n, 90, 3, int(r.integers(1 << 30))) for _ in range(nb)]) * wobble * edge[:, None]
    gate = np.stack([fbm1d(n, r.uniform(18, 60), 3, int(r.integers(1 << 30))) for _ in range(nb)])
    load0 = r.uniform(0.7, 1.0, nb)[:, None]
    run_dry = dry * (t[None, :] ** 1.15) * r.uniform(0.6, 1.4, nb)[:, None] * (0.7 + 0.6 * edge[:, None])
    load = np.clip(load0 * (1 - run_dry), 0, 1)
    alive = load * (0.78 + 0.45 * gate) > dry * (0.55 + 0.5 * edge[:, None])
    # rounded ends: the brush tip lands and lifts at the centre first, the edge bristles touch later / leave sooner
    land = (edge[:, None] ** 2) * w[None, :1] * 0.45
    lift = (edge[:, None] ** 2) * w[None, -1:] * 0.45
    alive &= (ss[None, :] >= land) & (ss[None, :] <= L - lift)
    amt = (load * alive * step).astype(np.float32)
    # pooled ink between the bristles runs out faster than the bristles do, so dry strokes break into streaks
    pool_w = np.clip(1 - 2.2 * dry * t ** 1.5, 0, 1)[None, :] * (load > 0.35)

    X =px[None, :] + nx[None, :] * (offs[:, None] + drift) * w[None, :]
    Y = py[None, :] + ny[None, :] * (offs[:, None] + drift) * w[None, :]
    pad = int(width * 0.6 + 12)
    x0, y0 = int(max(0, X.min() - pad)), int(max(0, Y.min() - pad))
    x1, y1 = int(min(W, X.max() + pad)), int(min(H, Y.max() + pad))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    h, wd = y1 - y0, x1 - x0
    acc = np.zeros(h * wd, np.float32)
    tacc = np.zeros(h * wd, np.float32)
    pacc = np.zeros(h * wd, np.float32)
    lx, ly = X.ravel() - x0, Y.ravel() - y0
    a, tt = amt.ravel(), np.broadcast_to(t, X.shape).ravel()
    pw = np.broadcast_to(pool_w, X.shape).ravel()
    ix, iy = np.floor(lx).astype(np.int64), np.floor(ly).astype(np.int64)
    fx, fy = lx - ix, ly - iy
    for dx, dy, k in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        jx, jy = ix + dx, iy + dy
        ok = (jx >= 0) & (jx < wd) & (jy >= 0) & (jy < h) & (a > 0)
        idx = jy[ok] * wd + jx[ok]
        np.add.at(acc, idx, a[ok] * k[ok])
        np.add.at(tacc, idx, a[ok] * k[ok] * tt[ok])
        if fill > 0:
            np.add.at(pacc, idx, a[ok] * k[ok] * pw[ok])
    acc, tacc, pacc = acc.reshape(h, wd), tacc.reshape(h, wd), pacc.reshape(h, wd)

    spacing = max(width / nb, 0.5)
    sig = 0.55 + 0.12 * min(spacing, 3)
    acc_b, tacc_b = blur(acc, sig), blur(tacc, sig)
    if fill > 0:                                                          # ink pooled between the bristles
        acc_b = acc_b + fill * blur(pacc, max(1.0, spacing * 1.2)) * spacing
    if wet > 0:                                                           # ink bleeding into the paper
        acc_b = acc_b + wet * blur(acc_b, 2.5 + width * 0.02)
    dens = ink * (1 - np.exp(-2.2 * acc_b))
    # when the brush reached each pixel: deposit-weighted mean of t, spread wide enough to cover the pooled ink
    spread = 2 + spacing + (2.5 + width * 0.02 if wet > 0 else 0)
    tw, aw = blur(tacc, spread), blur(acc, spread)
    time = np.where(aw > 1e-6, tw / np.maximum(aw, 1e-9), 1.0)
    return Stroke(x0, y0, dens.astype(np.float32), np.clip(time, 0, 1).astype(np.float32), red)


def splatter(W, H, cx, cy, spread, n=40, size=(1.5, 6), ink=0.9, seed=0, direction=None):
    """ink flicked off the brush: small irregular drops, a few with a tail, scattered around (cx, cy).
    spread: (sx, sy) radius; direction: angle in radians the drops fly along (elongates them)."""
    r = np.random.default_rng(seed)
    pad = int(size[1] * 4 + 6)
    x0, y0 = int(max(0, cx - spread[0] - pad)), int(max(0, cy - spread[1] - pad))
    x1, y1 = int(min(W, cx + spread[0] + pad)), int(min(H, cy + spread[1] + pad))
    dens = np.zeros((y1 - y0, x1 - x0), np.float32)
    time = np.ones_like(dens)
    for i in range(n):
        dx, dy = np.clip(r.normal(0, 0.45), -1, 1) * spread[0], np.clip(r.normal(0, 0.45), -1, 1) * spread[1]
        x, y = cx + dx, cy + dy
        rad = r.uniform(*size) * (0.45 if r.random() < 0.55 else 1)
        ang = (direction if direction is not None else np.arctan2(dy, dx)) + r.normal(0, 0.35)
        stretch = r.uniform(1, 1.5) if r.random() < 0.7 else r.uniform(1.8, 2.8)
        bx0, by0 = int(max(x0, x - rad * 3.5 - 2)), int(max(y0, y - rad * 3.5 - 2))
        bx1, by1 = int(min(x1, x + rad * 3.5 + 3)), int(min(y1, y + rad * 3.5 + 3))
        if bx1 <= bx0 or by1 <= by0:
            continue
        YY, XX = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        u = (XX - x) * np.cos(ang) + (YY - y) * np.sin(ang)
        v = -(XX - x) * np.sin(ang) + (YY - y) * np.cos(ang)
        theta = np.arctan2(v, u)
        lumpy = 1 + 0.18 * np.sin(3 * theta + r.uniform(0, 6)) + 0.1 * np.sin(5 * theta + r.uniform(0, 6))
        d = np.sqrt((u / stretch) ** 2 + v ** 2) - rad * lumpy
        drop = np.clip(0.6 - d, 0, 1) * ink * r.uniform(0.8, 1)
        sl = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
        hit = drop > 0.02
        time[sl][hit] = np.minimum(time[sl][hit], i / n)
        np.maximum(dens[sl], drop, out=dens[sl])
        if r.random() < 0.5:                                          # a speck or two thrown off the drop
            for _ in range(int(r.integers(1, 3))):
                ex, ey = x + r.normal(0, rad * 2.5), y + r.normal(0, rad * 2.5)
                jx, jy = int(ex) - x0, int(ey) - y0
                if 1 <= jx < dens.shape[1] - 1 and 1 <= jy < dens.shape[0] - 1:
                    dens[jy - 1:jy + 1, jx - 1:jx + 1] = np.maximum(dens[jy - 1:jy + 1, jx - 1:jx + 1], ink * 0.8)
                    time[jy - 1:jy + 1, jx - 1:jx + 1] = np.minimum(time[jy - 1:jy + 1, jx - 1:jx + 1], i / n)
    return Stroke(x0, y0, dens, time)


def sweep(W, H, pts, width, ink=0.85, dry=0.55, seed=0, wash=0.3, profile=SWEEP):
    """a big expressive dry-brush sweep: a pale wet wash for the body with dark broken bristle streaks over it.
    Returns [wash stroke, bristle stroke] to be added with the same timing."""
    body = brush_stroke(W, H, pts, width, ink=wash, dry=0.2, profile=profile, fill=1.0, wet=0.35,
                        wobble=0.1, seed=seed) if wash > 0 else None
    hair = brush_stroke(W, H, pts, width * 0.9, ink=ink, dry=dry, profile=profile, bristles=max(8, int(width * 0.6)),
                        fill=0.15, wet=0.05, wobble=0.08, seed=seed + 1)
    return [s for s in (body, hair) if s is not None]


def wash(W, H, pts, ink=0.3, soft=2.0, seed=0, grain=0.25):
    """a flat ink wash filling the polygon `pts` (hat, robe, rock); revealed top to bottom when animated"""
    from PIL import Image, ImageDraw
    r = np.random.default_rng(seed)
    P = np.asarray(pts, np.float32)
    pad = int(soft * 3 + 4)
    x0, y0 = int(max(0, P[:, 0].min() - pad)), int(max(0, P[:, 1].min() - pad))
    x1, y1 = int(min(W, P[:, 0].max() + pad)), int(min(H, P[:, 1].max() + pad))
    m = Image.new('L', (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in P], fill=255)
    m = blur(np.asarray(m, np.float32) / 255, soft)
    tex = 1 - grain + grain * blur(r.random(m.shape).astype(np.float32), 3) * 2
    dens = np.clip(m * ink * np.clip(tex, 0, 1.3), 0, 1).astype(np.float32)
    time = np.broadcast_to(np.linspace(0, 1, m.shape[0], dtype=np.float32)[:, None], m.shape).copy()
    return Stroke(x0, y0, dens, time)


class InkScene:
    """A painting built from brush strokes that can be shown at any point of its writing.

        s = InkScene(1920, 1080, seed=1)
        s.add(brush_stroke(...), start=0, dur=400)          # times in ms
        img = s.render()                                     # finished painting
        img = s.render(at=800)                               # what the paper looks like 0.8 s in

    s.p is the underlying claude-drawing Painting, used for the paper, background washes and the seal."""

    def __init__(self, W=1920, H=1080, seed=0):
        self.W, self.H = W, H
        self.p = Painting(W, H, seed=seed)
        self.p.paper()
        self.items = []                                   # (Stroke, start_ms, dur_ms)
        self.background = []                              # (density array, start_ms, dur_ms): faded in, not written
        self.texts = []                                   # (text item, start_ms)

    def add(self, stroke, start, dur):
        """add a Stroke (or a list of them, e.g. from sweep()) painted from `start` ms over `dur` ms;
        returns the end time, handy for chaining strokes one after another"""
        for st in (stroke if isinstance(stroke, list) else [stroke]):
            if st is not None:
                self.items.append((st, start, dur))
        return start + dur

    def add_background(self, start=0, dur=400):
        """everything drawn on self.p.D so far becomes a background layer that fades in"""
        self.background.append((self.p.D.copy(), start, dur))
        self.p.D[:] = 0

    def seal(self, text, x, y, size=40, start=0):
        self.p.seal(text, x, y, size)
        self.texts.append((self.p.texts.pop(), start))

    @property
    def length(self):
        ends = [s + d for _, s, d in self.items] + [s + d for _, s, d in self.background] + [s for _, s in self.texts]
        return max(ends) if ends else 0

    def render(self, at=None):
        """PIL Image of the painting at time `at` ms (None = finished)"""
        D = np.zeros((self.H, self.W), np.float32)
        R = np.zeros((self.H, self.W), np.float32)
        ease = lambda f: f * f * (3 - 2 * f)
        for dens, s, d in self.background:
            f = 1.0 if at is None else float(np.clip((at - s) / max(d, 1), 0, 1))
            if f > 0:
                np.copyto(D, 1 - (1 - D) * (1 - dens * ease(f)))
        for st, s, d in self.items:
            f = 1.0 if at is None else float(np.clip((at - s) / max(d, 1), 0, 1))
            if f <= 0:
                continue
            dens = st.reveal(f)
            h, w = dens.shape
            buf = (R if st.red else D)[st.y0:st.y0 + h, st.x0:st.x0 + w]
            np.copyto(buf, 1 - (1 - buf) * (1 - dens))
        self.p.D, self.p.R = D, R
        self.p.texts = [txt for txt, s in self.texts if at is None or at >= s]
        return self.p.composite()

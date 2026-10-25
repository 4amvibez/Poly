"""Scenes 3-6: the map, kanji, hiragana, katakana."""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import poly_ink  # noqa: F401
from core import blur, curve, spline
from poly_ink.brush import BLUNT, HOOK, PRESS_DOWN, RISE, SWEEP, brush_stroke, splatter, sweep, wash

from .common import BrushTool, Scene, ink_sprite, paste, sprite_from
from .glyph import Written, resample, strokes_of, write, write_text
from .ink import (FPS, H, INK, PAPER, VERM, W, InkFront, Keys, Track, compose, ease, glaze, glow, noise, over,
                  paper, parallax, ramp, smooth, text_mask, to_screen, view, view_rgb)

HERE = os.path.dirname(os.path.abspath(__file__))

SEA = np.array([0.45, 0.67, 0.67], np.float32)        # muted turquoise
SEA_DEEP = np.array([0.30, 0.50, 0.53], np.float32)
LAND = np.array([0.95, 0.88, 0.74], np.float32)       # parchment


def _blit(buf, st, k=1.0):
    if st is None:
        return
    h, w = st.dens.shape
    b = buf[st.y0:st.y0 + h, st.x0:st.x0 + w]
    np.copyto(b, 1 - (1 - b) * (1 - st.dens * k))


# ===================================================================== 3. the map
class Map(Scene):
    """An antique relief map of East Asia (Natural Earth coastlines). A ship carries writing from the Chinese
    coast along the Korean peninsula, across the strait by Tsushima and Iki to Kyushu, and through the Inland Sea,
    leaving a vermilion route; the camera follows it, then pulls back to show all three."""
    LON0, LON1, LAT0, LAT1, K = 108.0, 148.0, 20.0, 48.0, 130.0
    ROUTE = [(120.9, 35.75), (122.4, 36.05), (124.0, 36.3), (125.55, 36.15), (125.85, 35.3), (125.9, 34.55),
             (126.6, 34.0), (127.6, 34.1), (128.6, 34.35), (129.55, 33.98), (130.15, 33.85), (130.75, 33.98),
             (131.25, 33.93), (132.0, 33.85), (132.8, 34.0), (133.6, 34.3), (134.4, 34.5), (135.05, 34.5)]

    kx = K * np.cos(np.radians(34))

    def world_size(self):
        return int((self.LON1 - self.LON0) * self.kx), int((self.LAT1 - self.LAT0) * self.K)

    def ll(self, lon, lat):
        return (lon - self.LON0) * self.kx, (self.LAT1 - lat) * self.K

    def build(self):
        ww, wh = self.ww, self.wh
        self.track = Track(ww, wh, wet=False)
        geo = json.load(open(os.path.join(HERE, 'assets', 'geo', 'east_asia.json')))
        rings = [np.array([self.ll(*p) for p in r], np.float32) for r in geo['land']]
        # land mask (supersampled edge)
        m = Image.new('L', (ww, wh), 0)
        dr = ImageDraw.Draw(m)
        for r in rings:
            if len(r) > 2:
                dr.polygon([tuple(p) for p in r], fill=255)
        land = np.asarray(m, np.float32) / 255
        self.land = blur(land, 1.0)
        self.near_coast = blur(land, 40)                 # 0..1 field around the coasts
        sea = 1 - self.land
        tex = noise(wh // 4, ww // 4, 60, 5, 31)
        tex = np.asarray(Image.fromarray(tex, 'F').resize((ww, wh), Image.BICUBIC))
        # sea: turquoise glaze, deeper offshore, with watercolour blooms
        deep = 1 - blur(land, 140)
        self.sea_a = (sea * (0.42 + 0.3 * deep) * (0.82 + 0.3 * (tex - 0.5))).astype(np.float32)
        self.sea_deep = (sea * deep * 0.5).astype(np.float32)
        # land: parchment glaze, a little darker towards the coast like an old hand-tinted map
        self.land_a = (self.land * (0.55 + 0.25 * (tex - 0.5)) + 0.25 * self.land * (1 - blur(land, 18))).astype(np.float32)

        D = np.zeros((wh, ww), np.float32)
        # coastlines: bristle brush along every ring (skip the edges of the clipping box)
        lo, hi = np.array(self.ll(self.LON0 - 0.5, self.LAT1 + 0.5)), np.array(self.ll(self.LON1 + 0.5, self.LAT0 - 0.5))
        k = 0
        for r in rings:
            inside = np.all((r > lo) & (r < hi), axis=1)
            runs, cur = [], []
            for p, ok in zip(r, inside):
                if ok:
                    cur.append(p)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            for run in runs:
                P = np.array(run)
                if len(P) < 3:
                    continue
                L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
                if L < 14:
                    continue
                Q = resample(P, 7.0)
                for i in range(0, len(Q) - 1, 90):
                    seg = Q[i:i + 92]
                    if len(seg) < 2:
                        continue
                    _blit(D, brush_stroke(ww, wh, [tuple(p) for p in seg], 4.6, ink=0.72, dry=0.35, profile=BLUNT,
                                          smooth=2, seed=k, wet=0.15, wobble=0.04))
                    k += 1
        # coastal ripple lines on the sea side (antique map engraving)
        g = blur(land, 10)
        for lvl, a in ((0.05, 0.16), (0.012, 0.1)):
            band = np.exp(-((g - lvl) / (lvl * 0.35)) ** 2) * sea
            over(D, band * a)
        # inner shading along the coast on the land side
        over(D, self.land * (1 - blur(land, 9)) * 0.22)
        # rivers
        for rv in geo['rivers']:
            P = np.array([self.ll(*p) for p in rv['pts']], np.float32)
            if len(P) < 3:
                continue
            Q = resample(P, 8.0)
            for i in range(0, len(Q) - 1, 80):
                _blit(D, brush_stroke(ww, wh, [tuple(p) for p in Q[i:i + 82]], 2.6, ink=0.45, dry=0.45,
                                      profile=BLUNT, smooth=2, seed=5000 + i, wet=0.1))
        # relief: little brush peaks along the mountain ranges, shaded on the left
        rng = np.random.default_rng(7)
        ranges = [((112.0, 113.8), (35.0, 39.0), 26), ((106.5, 111.5), (33.3, 34.6), 22), ((116.5, 118.8), (25.5, 28.5), 18),
                  ((116.6, 117.6), (35.9, 36.5), 6), ((127.2, 128.8), (40.5, 42.3), 16), ((128.3, 129.3), (35.8, 38.4), 14),
                  ((127.3, 128.2), (35.2, 36.4), 8), ((137.0, 138.6), (35.4, 36.7), 16), ((139.2, 140.6), (37.0, 40.0), 12),
                  ((131.0, 133.5), (34.4, 35.1), 8), ((130.6, 131.4), (32.4, 33.4), 6), ((132.6, 134.0), (33.5, 33.9), 6),
                  ((119.5, 121.0), (28.0, 30.0), 10), ((124.0, 126.0), (40.0, 41.5), 10)]
        for (x0, x1), (y0, y1), n in ranges:
            for j in range(n):
                lon, lat = rng.uniform(x0, x1), rng.uniform(y0, y1)
                x, y = self.ll(lon, lat)
                if not (0 < x < ww and 0 < y < wh) or land[int(y), int(x)] < 0.5 or self.near_coast[int(y), int(x)] < 0.93:
                    continue
                s = rng.uniform(18, 30)
                _blit(D, brush_stroke(ww, wh, [(x - s, y + s * 0.15), (x - s * 0.2, y - s * 0.75), (x + s * 0.1, y - s * 0.85),
                                               (x + s, y + s * 0.15)], 3.0, ink=0.55, dry=0.4, profile=PRESS_DOWN,
                                      seed=6000 + j * 7 + int(lon * 10)))
                for q in range(3):
                    hx = x - s * (0.55 - q * 0.17)
                    _blit(D, brush_stroke(ww, wh, [(hx, y - s * (0.3 - q * 0.08)), (hx - s * 0.12, y + s * 0.1)], 1.8,
                                          ink=0.35, dry=0.5, profile=PRESS_DOWN, seed=7000 + j * 3 + q))
        self.D = D

        # labels: English in spaced capitals, the Japanese name beneath, revealed when the story reaches them
        self.labels = []
        for name, ja, (lon, lat), tw in [('CHINA', '中国', (115.2, 32.4), 0.5),
                                         ('KOREA', '朝鮮半島', (127.6, 39.6), self.wt(4, 'Korean')),
                                         ('JAPAN', '日本', (134.2, 36.75), self.wt(4, 'Japan'))]:
            x, y = self.ll(lon, lat)
            spaced = ' '.join(name)
            mk = text_mask(ww, wh, spaced, (x, y), 74, latin='serif', anchor='ms')
            mk = np.maximum(mk, text_mask(ww, wh, ja, (x, y + 70), 46, cjk='mincho', anchor='ms'))
            ys, xs = np.nonzero(mk > 0.02)
            box = (xs.min() - 6, ys.min() - 6, xs.max() + 6, ys.max() + 6)
            self.labels.append((mk[box[1]:box[3], box[0]:box[2]] * 0.8, box, tw))
        # the route: vermilion dashes, revealed behind the ship
        R = np.array([self.ll(*p) for p in self.ROUTE], np.float32)
        self.route = spline(R, 12)
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.route_s = np.concatenate([[0], np.cumsum(seg)])
        self.route_len = self.route_s[-1]
        self.ship_t0, self.ship_t1 = 2.5, 11.4
        n = int(self.route_len / 30)
        for j in range(n):
            a, b = j * 30, j * 30 + 17
            pts = [(np.interp(s, self.route_s, self.route[:, 0]), np.interp(s, self.route_s, self.route[:, 1]))
                   for s in np.linspace(a, b, 4)]
            st = brush_stroke(ww, wh, pts, 5.5, ink=0.9, dry=0.2, profile=BLUNT, seed=8000 + j, red=True, wet=0.1)
            ta = self.ship_time(b + 50)
            self.track.stroke(st, ta, ta + 0.15, red=True)
        self.pap = paper(ww, wh, seed=33, fibers=int(ww * wh / 1400))
        self.ship = Ship()
        # the transition stroke: the Chinese coast traced at the opening camera
        cam0 = self.cam(-1.0)
        ring = max(rings, key=len)
        sx, sy = to_screen(cam0, ring[:, 0], ring[:, 1])
        ok = (sx > -40) & (sx < W + 40) & (sy > -40) & (sy < H + 40)
        best, cur = [], []
        for i in range(len(ring)):
            if ok[i]:
                cur.append((sx[i], sy[i]))
            else:
                if len(cur) > len(best):
                    best = cur
                cur = []
        if len(cur) > len(best):
            best = cur
        self.coast_screen = np.array(best, np.float32)
        Q = resample(self.coast_screen, 6.0)
        self.coast_stroke = brush_stroke(W, H, [tuple(p) for p in Q], 9, ink=0.92, dry=0.4, profile=SWEEP, smooth=2,
                                         seed=77, wet=0.2)
        self.coast_front = InkFront(H // 4, W // 4, seeds=[(x / 4, y / 4, 1.0) for x, y in Q[::6]], rough=0.4,
                                    scale=45, soft=0.1, seed=78)

    def ship_time(self, s):
        """the time the ship reaches arc length s (inverse of ship_s)"""
        f = np.clip(s / self.route_len, 0, 1)
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if self._prog(mid) < f:
                lo = mid
            else:
                hi = mid
        return self.ship_t0 + (lo + hi) / 2 * (self.ship_t1 - self.ship_t0)

    @staticmethod
    def _prog(x):
        """eased progress, gentle start and finish, steady in the middle"""
        x = np.clip(x, 0, 1)
        a = 0.12
        if x < a:
            return 0.5 * x * x / a / (1 - a)
        if x > 1 - a:
            return 1 - 0.5 * (1 - x) ** 2 / a / (1 - a)
        return (x - a / 2) / (1 - a)

    def ship_s(self, t):
        return self._prog((t - self.ship_t0) / (self.ship_t1 - self.ship_t0)) * self.route_len

    def ship_pos(self, t):
        s = self.ship_s(t)
        x = np.interp(s, self.route_s, self.route[:, 0])
        y = np.interp(s, self.route_s, self.route[:, 1])
        x2 = np.interp(s + 30, self.route_s, self.route[:, 0])
        y2 = np.interp(s + 30, self.route_s, self.route[:, 1])
        return x, y, np.degrees(np.arctan2(y2 - y, x2 - x + 1e-6))

    def cam(self, t):
        ll = self.ll
        start = (*ll(117.6, 34.6), 1.22)
        if t < 2.2:
            c = Keys([(0.5, start), (2.2, (*ll(118.8, 34.9), 1.05))])(t)
        else:
            # follow the ship, gently ahead of it; zoom out a little as it travels; then the wide view
            sx, sy, _ = self.ship_pos(min(t + 0.6, self.ship_t1))
            follow = np.array([sx, sy - 40, 0.82 - 0.12 * ramp(t, 5, 10)])
            pre = np.array((*ll(118.8, 34.9), 1.05))
            c = pre * (1 - ramp(t, 2.2, 4.2)) + follow * ramp(t, 2.2, 4.2)
            wide = np.array((*ll(129.3, 35.4), 0.53))
            c = c * (1 - ramp(t, 10.4, self.dur + 0.9)) + wide * ramp(t, 10.4, self.dur + 0.9)
        return self.clamp(c)

    def frame(self, t):
        cam = self.cam(t)
        img = view_rgb(self.pap, cam)
        img = glaze(img, view(self.land_a, cam), LAND)
        sea_a = view(self.sea_a, cam)
        img = glaze(img, sea_a, SEA)
        img = glaze(img, view(self.sea_deep, cam) * 0.6, SEA_DEEP)
        # currents and foam, computed in screen space from world coordinates
        cx, cy, z = cam
        YY, XX = np.mgrid[0:H:2, 0:W:2].astype(np.float32)
        wx, wy = (XX - W / 2) / z + cx, (YY - H / 2) / z + cy
        phase = (wy + 34 * np.sin(wx / 290 + wy / 900 + t * 0.35) + 11 * np.sin(wx / 83 - t * 0.9)) / 92 + t * 0.05
        line = np.exp(-((phase - np.round(phase)) / 0.045) ** 2)
        # broken into drifting streaks, like brushed current lines on an old chart
        patch = np.clip((np.sin(wx / 140 + np.round(phase) * 2.1 - t * 0.5) - 0.2) * 1.6, 0, 1) * \
            np.clip(np.sin(wx / 520 + wy / 700 + t * 0.12) * 0.6 + 0.5, 0, 1)
        nc = view(self.near_coast, cam)[::2, ::2]
        sea_s = 1 - view(self.land, cam)[::2, ::2]
        cur = line * patch * sea_s * (1 - np.clip(nc * 1.6, 0, 1)) * 0.3
        foam_ph = nc * 7 - t * 0.45
        foam = np.exp(-((foam_ph - np.round(foam_ph)) / 0.07) ** 2) * np.clip((nc - 0.08) * 4, 0, 1) * np.clip((0.75 - nc) * 3, 0, 1) * sea_s
        up = lambda a: np.asarray(Image.fromarray(a.astype(np.float32), 'F').resize((W, H), Image.BILINEAR))
        img = glaze(img, up(cur), SEA_DEEP * 0.75)
        img = glow(img, up(foam) * 0.45, (1.0, 0.98, 0.92))
        # ink: the painted map + the route so far
        Dt, Rt = self.track.density(t)
        Dv = view(self.D, cam)
        img = img * (1 - Dv[..., None] * (1 - INK))
        img = img * (1 - view(Rt, cam)[..., None] * (1 - VERM / PAPER))
        # labels: ink spreading in from the left
        for mk, (x0, y0, x1, y1), tw in self.labels:
            a = ramp(t, tw, tw + 0.9)
            if a <= 0:
                continue
            reveal = np.clip((a * 1.3 - np.linspace(0, 1, mk.shape[1])[None, :]) / 0.3, 0, 1)
            sx0, sy0 = to_screen(cam, x0, y0)
            m = Image.fromarray((mk * reveal).astype(np.float32), 'F')
            sw, sh = int(mk.shape[1] * z), int(mk.shape[0] * z)
            if sw < 2 or sh < 2:
                continue
            ms = np.asarray(m.resize((sw, sh), Image.BILINEAR))
            xa, ya = int(sx0), int(sy0)
            xb, yb = max(0, xa), max(0, ya)
            xe, ye = min(W, xa + sw), min(H, ya + sh)
            if xe > xb and ye > yb:
                sl = img[yb:ye, xb:xe]
                sl[:] = sl * (1 - ms[yb - ya:ye - ya, xb - xa:xe - xa, None] * (1 - INK))
        # the ship
        if self.ship_t0 - 0.5 < t:
            x, y, ang = self.ship_pos(t)
            sx, sy = to_screen(cam, x, y)
            a = ramp(t, self.ship_t0 - 0.5, self.ship_t0 + 0.3)
            img = self.ship.draw(img, sx, sy, z, ang, t, a, cam, self)
        # the opening stroke along the coast fades into the painted coastline
        if t < 1.2:
            f = np.clip((t + self.trans / 2) / (self.trans * 0.5), 0, 1)
            d = self.coast_stroke.reveal(f, soft=0.05) * (1 - ramp(t, -0.1, 0.5))
            h, w = d.shape
            s = self.coast_stroke
            sl = img[s.y0:s.y0 + h, s.x0:s.x0 + w]
            sl[:] = sl * (1 - d[..., None] * (1 - INK))
        return img

    def enter(self, t, p):
        """the coast is traced over the previous scene by one stroke, then the map floods out from it"""
        s = self.coast_stroke
        cov = np.zeros((H, W), np.float32)
        d = s.reveal(np.clip(p / 0.5, 0, 1), soft=0.05)
        h, w = d.shape
        cov[s.y0:s.y0 + h, s.x0:s.x0 + w] = np.clip(d * 1.5, 0, 1)
        m = self.coast_front.mask(np.clip((p - 0.35) / 0.65, 0, 1))
        m = np.asarray(Image.fromarray(m, 'F').resize((W, H), Image.BILINEAR))
        return np.maximum(m, cov)


class Ship:
    """a two-masted ship of the kind that crossed the Korea Strait, in ink, with mat sails and a red pennant;
    it pitches gently, its sails breathe, and it leaves a wake"""

    def __init__(self):
        def hull(w, h):
            B = lambda pts, wd, **k: brush_stroke(w, h, pts, wd, **k)
            out = [wash(w, h, [(40, 222), (80, 252), (300, 254), (352, 214), (330, 222), (60, 214)], ink=0.72, soft=1.5, seed=1),
                   B([(30, 206), (70, 226), (200, 230), (330, 222), (366, 196)], 7, ink=0.92, dry=0.25, profile=BLUNT, seed=2),
                   B([(64, 250), (200, 258), (316, 250)], 5, ink=0.85, dry=0.4, profile=BLUNT, seed=3)]
            for j in range(5):
                x = 100 + j * 40
                out.append(B([(x, 236), (x - 22, 282)], 3, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=10 + j))
            out.append(B([(150, 214), (152, 40)], 5, ink=0.9, dry=0.2, profile=BLUNT, seed=20))
            out.append(B([(262, 214), (263, 80)], 4.5, ink=0.9, dry=0.2, profile=BLUNT, seed=21))
            return out

        def sails(w, h):
            B = lambda pts, wd, **k: brush_stroke(w, h, pts, wd, **k)
            out = [wash(w, h, [(96, 52), (204, 46), (214, 196), (88, 200)], ink=0.3, soft=1.2, seed=30, grain=0.4),
                   wash(w, h, [(220, 90), (300, 86), (308, 200), (214, 202)], ink=0.26, soft=1.2, seed=31, grain=0.4)]
            for j in range(7):
                y = 56 + j * 22
                out.append(B([(92, y + 2), (150, y - 1), (208, y)], 2.2, ink=0.7, dry=0.35, profile=BLUNT, seed=40 + j))
            for j in range(5):
                y = 94 + j * 22
                out.append(B([(218, y + 1), (262, y - 1), (304, y)], 2.0, ink=0.7, dry=0.35, profile=BLUNT, seed=50 + j))
            out.append(B([(96, 52), (92, 200)], 3, ink=0.8, dry=0.3, profile=BLUNT, seed=60))
            out.append(B([(204, 46), (214, 196)], 3, ink=0.8, dry=0.3, profile=BLUNT, seed=61))
            out.append((B([(152, 40), (176, 34), (196, 40)], 6, ink=0.95, dry=0.1, profile=PRESS_DOWN, seed=62), True))
            return out

        self.hull = ink_sprite(400, 300, hull)
        self.sails = ink_sprite(400, 300, sails)
        self.rng = np.random.default_rng(3)

    def draw(self, img, sx, sy, z, heading, t, alpha, cam, scene):
        s = 0.62 * z / 0.8
        pitch = 2.2 * np.sin(t * 1.9) + 1.2 * np.sin(t * 3.1)
        bob = 3.5 * np.sin(t * 2.3) * s
        breathe = 1 + 0.025 * np.sin(t * 2.7)
        # wake: pale lines spreading behind the ship along its route
        s_now = scene.ship_s(t)
        if s_now > 5:
            im = Image.new('L', (W, H), 0)
            d = ImageDraw.Draw(im)
            for side in (-1, 1):
                pts = []
                for q in np.linspace(0, 160, 12):
                    ss = max(0.0, s_now - 18 - q)
                    x = np.interp(ss, scene.route_s, scene.route[:, 0])
                    y = np.interp(ss, scene.route_s, scene.route[:, 1])
                    px, py = to_screen(cam, x, y)
                    pts.append((px, py + side * (5 + q * 0.22) * z + 4 * s))
                for i in range(len(pts) - 1):
                    d.line([pts[i], pts[i + 1]], fill=int(170 * (1 - i / len(pts)) * alpha), width=max(1, int(3 * s)))
            wk = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
            img = glow(img, wk * 0.55, (1.0, 0.98, 0.93))
        layers = []
        for rgb, a in (self.hull, self.sails):
            im = Image.fromarray(np.dstack([rgb, a[..., None]]).__mul__(255).astype(np.uint8), 'RGBA')
            if a is self.sails[1]:
                im = im.resize((int(400 * breathe), 300), Image.BILINEAR).crop((int(200 * breathe) - 200, 0,
                                                                                  int(200 * breathe) + 200, 300))
            layers.append(im)
        comp = Image.alpha_composite(layers[0], layers[1])
        comp = comp.resize((max(2, int(400 * s)), max(2, int(300 * s))), Image.LANCZOS)
        comp = comp.rotate(pitch, resample=Image.BICUBIC, expand=False)
        rgb, a = sprite_from(comp)
        # waterline foam around the hull
        img = paste(img, rgb, a * alpha, sx - comp.width / 2, sy - comp.height * 0.78 + bob)
        fy = sy - comp.height * 0.78 + bob + comp.height * 0.86
        YY, XX = np.ogrid[int(fy) - 20:int(fy) + 20, int(sx - comp.width * 0.6):int(sx + comp.width * 0.6)]
        if YY.size and XX.size and 0 <= int(fy) - 20 and int(fy) + 20 < H:
            dx = (XX - sx) / (comp.width * 0.5)
            ring = np.exp(-((YY - fy) / (3 * s + 1)) ** 2) * np.clip(1 - dx ** 2, 0, 1) * (0.6 + 0.4 * np.sin(XX * 0.15 + t * 8))
            x0 = int(sx - comp.width * 0.6)
            xa, xb = max(0, x0), min(W, x0 + ring.shape[1])
            sl = img[int(fy) - 20:int(fy) + 20, xa:xb]
            sl[:] = glow(sl, ring[:, xa - x0:xb - x0] * 0.5 * alpha, (1, 0.98, 0.94))
        return img


from .common import Labels, aged_paper, directional_front  # noqa: E402

ANALECTS = '子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎'


# ===================================================================== 4. kanji
class Kanji(Scene):
    """An old manuscript. A brush writes 漢字 large; columns of classical Chinese (the opening of the Analects)
    surface around it. Then two examples of how Japan used kanji: 山 for its meaning (yama, 'mountain') and
    安 purely for its sound ('a'), the man'yōgana use that leads to hiragana."""
    S = 1.3

    def build(self):
        u, ww, wh = self.u, self.ww, self.wh
        self.pap = aged_paper(ww, wh, seed=41)
        T = self.track
        ws, t = write_text(ww, wh, '漢字', *u(282, 232), u(320), 0.1, vertical=True, spacing=1.02, weight=0.1, dry=0.38,
                           ink=0.96, speed=2.0, seed=40, pause=0.15)
        T.add(ws)
        self.kanji_end = t
        # faint classical text, typeset in columns right to left
        cols = [ANALECTS[i:i + 12] for i in range(0, len(ANALECTS), 12)]
        txt = np.zeros((wh, ww), np.float32)
        for c, col in enumerate(cols):
            x = 1010 - c * 74
            for j, ch in enumerate(col):
                txt = np.maximum(txt, text_mask(ww, wh, ch, u(x, 240 + j * 60), u(48), anchor='ms'))
        # column rules
        im = Image.new('L', (ww, wh), 0)
        d = ImageDraw.Draw(im)
        for c in range(len(cols) + 1):
            x = u(1047 - c * 74)
            d.line([(x, u(170)), (x, u(960))], fill=90, width=2)
        txt = np.maximum(txt, blur(np.asarray(im, np.float32) / 255, 0.8))
        self.text = (txt * 0.42).astype(np.float32)
        self.text_front = InkFront(wh // 4, ww // 4, direction=(-1, 0.25), rough=0.35, scale=40, soft=0.1, seed=42)
        # the two examples
        t_m = self.wt(6, 'meaning') - 1.3
        t_s = self.wt(6, 'sound') - 1.5
        ws, e1 = write(ww, wh, '山', *u(210, 1010), u(230), t_m, weight=0.1, dry=0.3, speed=1.4, seed=43)
        T.add(ws)
        self.an_box = (580, 1010, 230)
        ws, e2 = write(ww, wh, '安', *u(*self.an_box[:2]), u(self.an_box[2]), t_s, weight=0.1, dry=0.3, speed=1.6,
                       seed=44)
        T.add(ws)
        self.an_end = e2
        L = self.labels = Labels(ww, wh)
        L.add('M E A N I N G', *u(325, 985), u(26), t_m - 0.2, red=True, ink=0.9)
        L.add('S O U N D', *u(695, 985), u(26), t_s - 0.2, red=True, ink=0.9)
        L.add('yama · “mountain”', *u(325, 1288), u(38), e1 - 0.1, latin='italic')
        L.add('“a” · sound only', *u(695, 1288), u(38), e2 - 0.1, latin='italic')
        self.brush = BrushTool(scale=0.9)
        an_c = (u(self.an_box[0] + self.an_box[2] / 2, self.an_box[1] + self.an_box[2] / 2))
        self.camk = Keys([(-0.6, (*u(340, 330), 1.25)), (0.4, (*u(360, 360), 1.2)), (2.6, (*u(420, 560), 1.0)),
                          (self.kanji_end + 0.6, (*u(560, 760), 0.8)), (t_m - 0.6, (*u(540, 860), 0.8)),
                          (self.dur - 1.9, (*u(560, 900), 0.82)), (self.dur - 0.55, (*an_c, 1.15))])

    def cam(self, t):
        return self.clamp(self.camk(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        m = self.text_front.mask(ramp(t, 1.0, 4.8))
        m = np.asarray(Image.fromarray(m, 'F').resize((self.ww, self.wh), Image.BILINEAR))
        img = view_rgb(self.pap, cam)
        img = img * (1 - view(self.text * m, cam)[..., None] * (1 - np.array([0.30, 0.22, 0.16])))
        img = compose(img, view(D, cam), view(R, cam))
        img = self.labels.draw(img, cam, t)
        img = self.brush.draw(img, self.track, t, cam, presence=1.0)
        return img

    def enter(self, t, p):
        """the parchment of the map turns into the manuscript: a slow, fibrous paper dissolve"""
        if not hasattr(self, '_front'):
            self._front = InkFront(H // 4, W // 4, direction=(0.2, 1), rough=0.75, scale=30, soft=0.18, seed=45)
        return np.asarray(Image.fromarray(self._front.mask(ease(p)), 'F').resize((W, H), Image.BILINEAR))


# ===================================================================== 5. hiragana
# the cursive (草書) form of 安, drawn as a simplified illustration: the dot and roof of 宀 run together, and 女
# becomes one descending stroke and a loop — the shape that was simplified into あ
CURSIVE_AN = [
    ([(52, 9), (55, 17), (54, 21)], ((0, 0.4), (0.5, 1.0), (1, 0.5))),
    ([(29, 26), (26, 37), (31, 33), (52, 29), (74, 26), (84, 25), (79, 35)], ((0, 0.5), (0.15, 0.8), (0.4, 0.75), (0.85, 0.9), (1, 0.2))),
    ([(53, 36), (50, 52), (47, 70), (48, 86), (52, 97)], ((0, 0.6), (0.2, 1.0), (0.8, 0.7), (1, 0.25))),
    ([(71, 46), (64, 63), (51, 79), (37, 88), (28, 83), (33, 70), (50, 61), (70, 60), (85, 68), (87, 83), (76, 95), (62, 100)],
     ((0, 0.5), (0.12, 1.0), (0.45, 0.6), (0.7, 0.9), (0.9, 0.7), (1, 0.15))),
]


class Hiragana(Scene):
    """安 → cursive 安 → あ, one stage under the other. The strokes of 女 that become the loop of あ are
    picked out in vermilion; the cursive form is written in one flowing movement; あ resolves at the bottom."""
    S = 1.3
    PAD = 700

    def build(self):
        u, ww, wh = self.u, self.ww, self.wh
        self.pap = paper(ww, wh, seed=51)
        T = self.track
        x, sz = 400, 280
        self.box = {'an': (x, 185, sz), 'cur': (x, 560, sz), 'a': (x, 935, sz)}
        # 安, already on the page (it was written at the end of the last scene)
        ws, _ = write(ww, wh, '安', *u(*self.box['an'][:2]), u(sz), -3.0, weight=0.1, dry=0.3, speed=1.6, seed=44)
        for k, w_ in enumerate(ws):
            if k >= 3:          # the strokes of 女 give way to vermilion as they are re-traced
                w_.fade = (lambda tt, k=k: 1 - ramp(tt, 0.9 + 0.35 * (k - 3), 1.35 + 0.35 * (k - 3)))
        T.add(ws)
        # the strokes of 女 (3-5), which become the loop of あ, re-traced in vermilion
        ws, _ = write(ww, wh, '安', *u(*self.box['an'][:2]), u(sz), 0.9, weight=0.1, dry=0.3, speed=1.3, seed=44,
                      only={3, 4, 5}, all_red=True, ink=0.95)
        T.add(ws)
        # arrows
        for k, (y0, y1, ta) in enumerate([(482, 545, 1.9), (857, 920, self.wt(8, 'simplified') - 0.6)]):
            T.add(_arrow(ww, wh, u, 540, y0, y1, ta, seed=53 + k))
        # the cursive form: quick, wet, connected
        t = 2.3
        bx, by, _ = self.box['cur']
        kk = u(sz) / 109
        ox, oy = u(bx, by)
        for i, (pts, prof) in enumerate(CURSIVE_AN):
            Q = resample(np.array([(ox + px * kk, oy + py * kk) for px, py in pts], np.float32), 3.0)
            L = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
            st = brush_stroke(ww, wh, [tuple(p) for p in Q], u(sz) * 0.085, ink=0.95, dry=0.35, profile=prof, smooth=4,
                              seed=60 + i, wet=0.3)
            dur = 0.12 + L / u(sz) * 0.3
            T.add(Written(st, t, t + dur, Q))
            if i < len(CURSIVE_AN) - 1:
                # the brush barely lifts: a hairline joins one stroke to the next (連綿)
                nxt = CURSIVE_AN[i + 1][0][0]
                a, b = Q[-1], (ox + nxt[0] * kk, oy + nxt[1] * kk)
                hl = brush_stroke(ww, wh, [tuple(a), ((a[0] + b[0]) / 2 + u(6), (a[1] + b[1]) / 2), b], u(2.2),
                                  ink=0.55, dry=0.5, profile=((0, 0.6), (0.5, 1), (1, 0.4)), seed=70 + i, wet=0.05)
                T.add(Written(hl, t + dur, t + dur + 0.1, np.array([a, b], np.float32)))
                t += dur + 0.12
            else:
                t += dur
        self.cur_end = t
        # あ: written stroke by stroke, landing on "hiragana"
        t_a = self.wt(8, 'hiragana') - 2.0
        ws, self.a_end = write(ww, wh, 'あ', *u(*self.box['a'][:2]), u(sz), t_a, weight=0.095, dry=0.28, speed=0.9,
                               seed=54)
        T.add(ws)
        # labels at the right of each stage
        L = self.labels = Labels(ww, wh)
        lx = 720
        for key, ja, en, tl in [('an', '漢字', 'kanji', 0.6), ('cur', '草書', 'cursive', self.cur_end - 0.4),
                                ('a', 'ひらがな', 'hiragana', self.a_end - 0.2)]:
            _, y, _ = self.box[key]
            L.add(ja, *u(lx, y + 140), u(46), tl, anchor='ls', red=(key == 'a'))
            L.add(en, *u(lx, y + 192), u(38), tl + 0.15, latin='italic', anchor='ls', red=(key == 'a'))
        # the final beat: vermilion ticks beside 安 and the cursive form, an ensō round あ
        t_an = self.wt(9, '安')
        t_cu = self.wt(9, 'cursive')
        t_ah = self.wt(9, 'あ')
        for key, tt, sd in (('an', t_an, 80), ('cur', t_cu, 81)):
            _, y, _ = self.box[key]
            T.add(Written(brush_stroke(ww, wh, [u(360, y + 70), u(352, y + 140), u(358, y + 210)], u(9), ink=0.9,
                                       dry=0.3, profile=PRESS_DOWN, seed=sd, red=True), tt - 0.2, tt + 0.1,
                          np.array([u(360, y + 70), u(358, y + 210)], np.float32), red=True))
        cx, cy = 540, 935 + sz / 2
        ens = [(cx + np.cos(a) * 180, cy + np.sin(a) * 172) for a in np.linspace(-2.2, 3.7, 26)]
        T.add(Written(brush_stroke(ww, wh, self.U(ens), u(11), ink=0.85, dry=0.55,
                                   profile=((0, 0.5), (0.1, 1.0), (0.7, 0.8), (1, 0.15)), seed=82, red=True),
                      t_ah - 0.3, t_ah + 0.5, np.array(self.U(ens[::5]), np.float32), red=True))
        self.brush = BrushTool(scale=0.9)
        an_c = u(x + sz / 2, 185 + sz / 2)
        z0 = 1.15 * 230 / sz           # 安 at the same size as the last frame of the previous scene
        self.camk = Keys([(-0.6, (*an_c, z0)), (0.55, (*an_c, z0)), (2.2, (*u(540, 640), 0.9)),
                          (self.cur_end + 0.3, (*u(540, 700), 0.86)), (t_a - 0.3, (*u(540, 860), 0.86)),
                          (self.a_end + 0.4, (*u(540, 820), 0.82)), (t_an - 0.4, (*u(540, 760), 0.8)),
                          (self.dur + 1, (*u(540, 900), 0.9))])

    def cam(self, t):
        return self.clamp(self.camk(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        img = compose(view_rgb(self.pap, cam), view(D, cam), view(R, cam))
        img = self.labels.draw(img, cam, t)
        img = self.brush.draw(img, self.track, t, cam)
        return img

    def enter(self, t, p):
        """a straight soft dissolve: the camera move lines 安 up with itself across the cut"""
        return np.full((H, W), smooth(p), np.float32)


def _arrow(ww, wh, u, x, y0, y1, t, seed=0):
    sh = brush_stroke(ww, wh, [u(x, y0), u(x + 2, (y0 + y1) / 2), u(x, y1)], u(4.5), ink=0.6, dry=0.5,
                      profile=PRESS_DOWN, seed=seed)
    hl = brush_stroke(ww, wh, [u(x - 16, y1 - 22), u(x, y1 + 2)], u(4), ink=0.6, dry=0.4, profile=BLUNT, seed=seed + 1)
    hr = brush_stroke(ww, wh, [u(x + 16, y1 - 22), u(x, y1 + 2)], u(4), ink=0.6, dry=0.4, profile=BLUNT, seed=seed + 2)
    return [Written(sh, t, t + 0.3, np.array([u(x, y0), u(x, y1)], np.float32)),
            Written(hl, t + 0.3, t + 0.4, np.array([u(x - 16, y1 - 22), u(x, y1)], np.float32)),
            Written(hr, t + 0.4, t + 0.5, np.array([u(x + 16, y1 - 22), u(x, y1)], np.float32))]


# ===================================================================== 6. katakana
SUTRA = '観自在菩薩行深般若波羅蜜多時照見五蘊皆空度一切苦厄舎利子色不異空空不異色色即是空空即'
# reading glosses (kunten) beside the characters, as monks wrote them, for the kundoku reading
# 観自在菩薩、深般若波羅蜜多ヲ行ズル時、五蘊皆空ナリト照見シテ、一切ノ苦厄ヲ度シタマフ。
# 舎利子ヨ、色ハ空ニ異ナラズ、空ハ色ニ異ナラズ: index in SUTRA -> katakana
GLOSSES = {5: 'ズル', 12: 'ヲ', 15: 'シテ', 19: 'ナリト', 20: 'シタマフ', 22: 'ノ', 24: 'ヲ', 27: 'ヨ', 28: 'ハ',
           30: 'ナラズ', 31: 'ニ', 32: 'ハ', 34: 'ナラズ', 35: 'ニ'}


class Katakana(Scene):
    """A sutra page with monks' reading notes in small vermilion katakana beside the Chinese text. Then 伊 is
    written large; its left side 亻 is picked out and boxed, the rest fades, and イ is written beside it."""
    S = 1.25
    PAD = 300

    def build(self):
        u, ww, wh = self.u, self.ww, self.wh
        self.pap = aged_paper(ww, wh, seed=61, tone=(0.92, 0.86, 0.73))
        T = self.track
        # the sutra: 6 columns of 7 characters, right to left, with ruled columns
        n_per = 7
        cols = [SUTRA[i:i + n_per] for i in range(0, len(SUTRA), n_per)]
        txt = np.zeros((wh, ww), np.float32)
        im = Image.new('L', (ww, wh), 0)
        d = ImageDraw.Draw(im)
        self.char_pos = {}
        for c, col in enumerate(cols):
            x = 925 - c * 140
            d.line([u(x + 70, 180), u(x + 70, 880)], fill=110, width=2)
            for j, ch in enumerate(col):
                y = 270 + j * 92
                self.char_pos[c * n_per + j] = (x, y)
                txt = np.maximum(txt, text_mask(ww, wh, ch, u(x, y), u(70), cjk='mincho-bold', anchor='ms'))
        d.line([u(925 - (len(cols) - 1) * 140 - 70, 180), u(925 - (len(cols) - 1) * 140 - 70, 880)], fill=110, width=2)
        self.text = np.maximum(txt * 0.78, blur(np.asarray(im, np.float32) / 255, 0.8) * 0.5).astype(np.float32)
        self.text_front = InkFront(wh // 4, ww // 4, direction=(-1, 0.15), rough=0.3, scale=40, soft=0.08, seed=62)
        # glosses: small vermilion katakana, brushed in one after another
        L = self.labels = Labels(ww, wh)
        # 伊 is written on "shorthand"; the glosses are paced to finish, and the camera to arrive, before it
        t_i = self.wt(11, 'shorthand') - 0.5
        t = 2.4
        step = float(np.clip((t_i - 1.5 - t) / len(GLOSSES), 0.18, 0.42))
        for idx in sorted(GLOSSES):
            x, y = self.char_pos[idx]
            for j, ch in enumerate(GLOSSES[idx]):
                L.add(ch, *u(x + 52, y - 50 + j * 25), u(24), t + j * 0.06, dur=0.22, red=True, ink=0.95,
                      cjk='mincho-bold')
            t += step
        self.gloss_end = t
        # 伊, large; its left side 亻 (strokes 0, 1) picked out; the rest fades; イ written beside it
        self.i_box = (100, 940, 330)
        ws, self.i_end = write(ww, wh, '伊', *u(*self.i_box[:2]), u(self.i_box[2]), t_i, weight=0.1, dry=0.32,
                               speed=1.7, seed=63)
        t_part = self.wt(11, 'one part') - 0.2
        t_fade = self.wt(12, 'left side') + 0.2
        for k, w_ in enumerate(ws):
            if k >= 2:
                w_.fade = (lambda tf, tp: (lambda tt: 1 - 0.7 * ramp(tt, tp, tp + 0.6) - 0.22 * ramp(tt, tf, tf + 0.8)))(t_fade, t_part)
        T.add(ws)
        ws, _ = write(ww, wh, '伊', *u(*self.i_box[:2]), u(self.i_box[2]), t_part, weight=0.1, dry=0.32, speed=2.2,
                      seed=63, only={0, 1}, all_red=True)
        T.add(ws)
        # a box drawn round 亻
        bx, by, bs = self.i_box
        box = [(bx + 8, by + 18), (bx + 140, by + 12), (bx + 146, by + bs - 10), (bx + 6, by + bs - 4), (bx + 10, by + 24)]
        T.add(Written(brush_stroke(ww, wh, self.U(box), u(4), ink=0.85, dry=0.5, profile=BLUNT, smooth=6, seed=64,
                                   red=True), t_part + 0.5, t_part + 1.2, np.array(self.U(box), np.float32), red=True))
        # arrow and イ
        t_k = self.wt(12, 'katakana') - 0.6
        sh = brush_stroke(ww, wh, [u(470, 1110), u(530, 1104), u(575, 1110)], u(5), ink=0.6, dry=0.5,
                          profile=PRESS_DOWN, seed=65)
        T.add(Written(sh, t_k - 0.45, t_k - 0.2, np.array([u(470, 1110), u(575, 1110)], np.float32)))
        T.add(Written(brush_stroke(ww, wh, [u(552, 1088), u(577, 1110)], u(4), ink=0.6, dry=0.4, profile=BLUNT, seed=66),
                      t_k - 0.2, t_k - 0.12, np.array([u(552, 1088), u(577, 1110)], np.float32)))
        T.add(Written(brush_stroke(ww, wh, [u(552, 1132), u(577, 1110)], u(4), ink=0.6, dry=0.4, profile=BLUNT, seed=67),
                      t_k - 0.12, t_k - 0.04, np.array([u(552, 1132), u(577, 1110)], np.float32)))
        ws, self.k_end = write(ww, wh, 'イ', *u(620, 940), u(330), t_k, weight=0.1, dry=0.3, speed=1.1, seed=68)
        T.add(ws)
        L.add('K A N J I', *u(265, 915), u(26), t_i, ink=0.75)
        L.add('伊  i', *u(265, 1312), u(42), self.i_end, ink=0.8)
        L.add('K A T A K A N A', *u(785, 915), u(26), t_k, red=True)
        L.add('イ  i', *u(785, 1312), u(42), self.k_end, red=True)
        self.brush = BrushTool(scale=0.85)
        self.camk = Keys([(-0.5, (*u(880, 520), 1.05)), (0.3, (*u(860, 520), 1.05)),
                          (self.gloss_end - 0.2, (*u(300, 560), 1.0)), (max(t_i - 0.3, self.gloss_end + 0.6), (*u(540, 900), 0.84)),
                          (t_part, (*u(500, 1000), 0.9)), (t_k - 0.8, (*u(540, 1010), 0.86)),
                          (self.dur + 1, (*u(540, 1000), 0.88))])

    def cam(self, t):
        return self.clamp(self.camk(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        m = self.text_front.mask(ramp(t, -0.7, 1.6))
        m = np.asarray(Image.fromarray(m, 'F').resize((self.ww, self.wh), Image.BILINEAR))
        img = view_rgb(self.pap, cam)
        img = img * (1 - view(self.text * m, cam)[..., None] * (1 - np.array([0.17, 0.15, 0.14])))
        img = compose(img, view(D, cam), view(R, cam))
        img = self.labels.draw(img, cam, t)
        img = self.brush.draw(img, self.track, t, cam)
        return img

    def enter(self, t, p):
        """the flowing hiragana stroke turns into one sharp, angular sweep that carries the next page in"""
        if not hasattr(self, '_front'):
            self._front = directional_front(-18, rough=0.35, soft=0.05, seed=69)
        return np.asarray(Image.fromarray(self._front.mask(ease(p)), 'F').resize((W, H), Image.BILINEAR))

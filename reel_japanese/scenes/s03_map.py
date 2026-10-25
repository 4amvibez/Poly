"""Scene 3 — China, the Korean Peninsula and Japan.  An antique relief map built from real coastline data
(global-land-mask, 1 km GLOBE data): parchment land with hillshade and brushed mountain symbols, engraved
water-lines along the coasts, a muted turquoise sea with moving currents and foam.  A small wooden ship sails
a plausible route (Shandong -> the west coast of Korea -> Tsushima -> northern Kyushu) drawing a vermilion
route; fainter routes appear at the end, because the contact spanned centuries and many crossings."""
import cv2
import numpy as np

from engine import (FONT_EN, FONT_EN_BOLD, GREY, H, INK, PAPER, SEA, SEPIA, VERM, W, Cam, StrokeSet, blur, cached,
                    ease_io, ease_out, keyframes, noise, ramp, reveal_time, smooth, spline, sstep, stroke, text_layer,
                    warp, wipe_mix)
from poly_ink.brush import BLUNT, PRESS_DOWN
from scenes.base import SceneBase
from timeline import at

MW, MH = 2400, 3600                 # world canvas (map + parchment margins)
LON0, LON1, LAT0, LAT1 = 114.0, 146.0, 46.0, 24.0
KX = np.cos(np.radians(35))
K = MW / ((LON1 - LON0) * KX)       # px per degree of latitude
MAP_Y0 = int((MH - (LAT0 - LAT1) * K) / 2)


def proj(lon, lat):
    return (lon - LON0) * K * KX, MAP_Y0 + (LAT0 - lat) * K


ROUTE = [(122.4, 37.35), (123.6, 37.6), (124.9, 37.45), (125.9, 36.9), (126.05, 35.9), (125.95, 34.95),
         (126.6, 34.25), (127.6, 34.3), (128.6, 34.6), (129.15, 34.35), (129.55, 33.85), (130.25, 33.62)]
ROUTES_FAINT = [
    [(121.95, 31.0), (124.5, 31.4), (127.0, 31.9), (128.9, 32.4), (129.75, 32.75)],      # across the East China Sea
    [(129.5, 35.9), (130.6, 35.85), (131.8, 35.6), (132.75, 35.45)],                      # Korean east coast -> Izumo
    [(120.9, 38.6), (121.8, 39.2), (123.0, 39.7), (124.3, 39.75)],                        # along the Bohai coast
]


def _route_px(r):
    return np.array([proj(lo, la) for lo, la in r], np.float32)


class Scene(SceneBase):
    paper_size = (MW, MH)
    paper_seed = 13
    paper_stains = 0.1

    def build(self):
        A = cached('s03_map_v2', self._paint)
        self.land = A['land']
        self.sea_d, self.lines, self.relief, self.coast = A['sea'], A['lines'], A['relief'], A['coast']
        self.mounts, self.coast_band, self.grat = A['mounts'], A['band'], A['grat']
        self.sea_mask = 1 - self.land
        # current streaks: anisotropic noise, tiled and scrolled along the flow
        r = np.random.default_rng(5)
        st = blur(r.random((512, 512)).astype(np.float32), 1.0)
        st = cv2.resize(cv2.resize(st, (512, 64)), (512, 512), interpolation=cv2.INTER_CUBIC)
        st = (st - st.min()) / (st.max() - st.min())
        self.streak = np.tile(sstep(0.62, 0.9, st) * 0.5, (1, 1)).astype(np.float32)
        self.foam_n = noise(512, 512, 24, 3, 17)
        self.route = spline(_route_px(ROUTE), 16)
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.route_s = np.concatenate([[0], np.cumsum(seg)])
        self.faint = [spline(_route_px(r), 14) for r in ROUTES_FAINT]
        self.ship = self._ship()

        # labels
        self.labels = []
        for text, lon, lat, size, track, rot, t in [
                ('CHINA', 119.0, 35.75, 84, 24, 0, at('China')[0] - 0.2),
                ('KOREAN', 127.15, 40.55, 44, 10, 0, at('Korean Peninsula')[0] - 0.1),
                ('PENINSULA', 127.15, 39.95, 44, 10, 0, at('Korean Peninsula')[0] + 0.15),
                ('JAPAN', 139.0, 33.6, 80, 22, 33, at('Japanese islands')[0] - 0.3)]:
            a, _ = text_layer(text, FONT_EN_BOLD, size, tracking=track)
            if rot:
                h0, w0 = a.shape
                M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), rot, 1)
                c, s = abs(M[0, 0]), abs(M[0, 1])
                nw, nh = int(h0 * s + w0 * c), int(h0 * c + w0 * s)
                M[0, 2] += nw / 2 - w0 / 2
                M[1, 2] += nh / 2 - h0 / 2
                a = cv2.warpAffine(a, M, (nw, nh))
            x, y = proj(lon, lat)
            hh, ww = a.shape
            T = np.tile(np.linspace(0, 1, ww, dtype=np.float32), (hh, 1))
            self.labels.append((a * 0.85, T, x - ww / 2, y - hh / 2, t))

        self.t_ship0 = at('Through centuries')[0] - 0.4
        self.t_ship1 = at('Japanese islands')[0] + 0.2
        self._track = self._precompute_track()

    # ------------------------------------------------------------ painting the map (cached)
    def _paint(self):
        from global_land_mask import globe
        xs = np.arange(MW, dtype=np.float32)
        ys = np.arange(MH, dtype=np.float32)
        lon = LON0 + xs / (K * KX)
        lat = LAT0 - (ys - MAP_Y0) / K
        LO, LA = np.meshgrid(lon, np.clip(lat, -89, 89))
        land = globe.is_land(LA, LO).astype(np.float32)
        inside = ((ys >= MAP_Y0) & (ys < MAP_Y0 + (LAT0 - LAT1) * K))[:, None] * np.ones((1, MW), np.float32)
        land = blur(land, 1.2)
        land = sstep(0.4, 0.6, land) * inside
        lb = (land > 0.5).astype(np.uint8)
        # drop specks (single noisy pixels would grow bullseye water-lines); keep real small islands
        n, lab, stats, _ = cv2.connectedComponentsWithStats(lb, 8)
        small = np.isin(lab, np.where(stats[:, cv2.CC_STAT_AREA] < 70)[0][1:])
        lb[small] = 0
        land = land * lb + blur(lb.astype(np.float32), 0.8) * (1 - lb) * 0.0
        land = np.where(lb > 0, np.maximum(land, 0.5), np.minimum(land, 0.49) * 0)
        d_sea = cv2.distanceTransform(1 - lb, cv2.DIST_L2, 5).astype(np.float32)   # distance from land (in sea)
        d_land = cv2.distanceTransform(lb, cv2.DIST_L2, 5).astype(np.float32)      # distance from sea (on land)
        # sea: deeper offshore, textured
        sea = (0.28 + 0.30 * sstep(0, 300, d_sea)) * (0.93 + 0.07 * noise(MH, MW, 160, 3, 3)) * (0.96 + 0.04 * noise(MH, MW, 14, 3, 9))
        sea = sea * (1 - land) * inside
        # engraved water-lines following the coast
        lines = np.zeros_like(sea)
        for k, d in enumerate((7, 14, 23, 35, 50)):
            lines = np.maximum(lines, np.clip(1 - np.abs(d_sea - d) / 1.1, 0, 1) * (0.42 - 0.07 * k))
        lines *= (1 - land) * inside * (0.75 + 0.25 * noise(MH, MW, 30, 3, 4))
        # relief: inland height + mountain zones, hillshade from the north-west
        zones = np.zeros_like(sea)
        for lo0, lo1, la0, la1, s in [(136.3, 139.2, 35.2, 37.0, 1.0), (139.4, 141.6, 37.3, 40.8, 0.6),
                                      (142.0, 143.8, 42.5, 44.2, 0.6), (130.4, 131.6, 31.8, 33.6, 0.6),
                                      (132.6, 134.2, 33.4, 34.0, 0.5), (127.6, 129.3, 36.3, 39.6, 0.8),
                                      (126.4, 129.4, 39.6, 42.6, 0.8), (126.5, 130.5, 41.5, 44.5, 0.6),
                                      (114.0, 120.3, 24.0, 30.2, 0.75), (117.4, 121.0, 35.6, 37.0, 0.45),
                                      (114.0, 114.9, 34.5, 41.5, 0.8), (131.5, 135.5, 34.4, 35.4, 0.5)]:
            x0, y0 = proj(lo0, la1)
            x1, y1 = proj(lo1, la0)
            box = np.zeros_like(sea)
            box[int(y0):int(y1), int(x0):int(x1)] = s
            zones = np.maximum(zones, blur(box, 30))
        hgt = (1 - np.exp(-d_land / 50)) * (0.35 + 0.65 * zones) * (0.5 + 0.5 * noise(MH, MW, 70, 5, 6)) * land
        gy, gx = np.gradient(blur(hgt, 2) * 120)
        shade = np.clip(-(gx * -0.7 + gy * -0.7), -1.2, 1.2)
        relief = np.clip(0.07 * land + 0.18 * np.clip(-shade, 0, None) + 0.10 * hgt, 0, 0.5) * land
        coast = np.clip(1 - np.abs(d_sea + (1 - lb) * 0 - 0.5) / 1.3, 0, 1) * (lb == 0) + np.clip(1 - d_land / 1.6, 0, 1) * lb
        coast = blur(coast, 0.6) * inside * (0.75 + 0.25 * noise(MH, MW, 12, 3, 7))
        band = np.clip(1 - d_sea / 9, 0, 1) * (1 - land) * inside
        # mountain symbols, brushed: a light sunlit slope, a dark shadowed slope, hatching
        ss = StrokeSet(MW, MH)
        r = np.random.default_rng(11)
        pts = []
        for _ in range(9000):
            x, y = r.uniform(0, MW), r.uniform(MAP_Y0, MAP_Y0 + (LAT0 - LAT1) * K)
            xi, yi = int(x), int(y)
            if d_land[yi, xi] < 9 or r.random() > zones[yi, xi] * 0.9:
                continue
            if all((x - a) ** 2 + (y - b) ** 2 > 30 ** 2 for a, b in pts[-60:]):
                pts.append((x, y))
        for k, (x, y) in enumerate(pts[:520]):
            s = r.uniform(0.75, 1.3)
            w2, hh = 15 * s, 13 * s
            ss.add([stroke(MW, MH, [(x - w2, y), (x - w2 * 0.2, y - hh), (x, y - hh * 1.05)], 2.0 * s, ink=0.42, dry=0.4,
                           profile=PRESS_DOWN, seed=k * 7),
                    stroke(MW, MH, [(x, y - hh * 1.05), (x + w2 * 0.35, y - hh * 0.5), (x + w2, y)], 2.6 * s, ink=0.62,
                           dry=0.35, profile=PRESS_DOWN, seed=k * 7 + 1)] +
                   [stroke(MW, MH, [(x + w2 * (0.1 + 0.2 * j), y - hh * (0.75 - 0.2 * j)), (x + w2 * (0.05 + 0.2 * j), y - 1)],
                           1.3 * s, ink=0.35, dry=0.5, profile=PRESS_DOWN, seed=k * 7 + 2 + j) for j in range(3)], 0, 1)
        mounts = ss.render(10) * inside
        # graticule every 5 degrees
        grat = np.zeros_like(sea)
        for lo in range(115, 146, 5):
            x = int(proj(lo, 0)[0])
            grat[:, max(0, x - 1):x + 1] = 1
        for la in range(25, 46, 5):
            y = int(proj(0, la)[1])
            grat[y - 1:y + 1, :] = 1
        grat = blur(grat, 0.7) * inside * 0.16
        # map border: double rule
        bd = np.zeros_like(sea)
        y0, y1 = MAP_Y0, int(MAP_Y0 + (LAT0 - LAT1) * K)
        for off, wdt in ((-14, 5), (-26, 2)):
            cv2.rectangle(bd, (12 - off, y0 + off), (MW - 13 + off, y1 - off), 1.0, wdt, cv2.LINE_AA)
        grat = np.maximum(grat, blur(bd, 0.6) * 0.7)
        # compass rose in the open Pacific
        cx, cy = proj(143.2, 27.6)
        R = 120
        rose = np.zeros_like(sea)
        cv2.circle(rose, (int(cx), int(cy)), R, 1.0, 3, cv2.LINE_AA)
        cv2.circle(rose, (int(cx), int(cy)), R - 12, 0.6, 2, cv2.LINE_AA)
        for k in range(8):
            a = k * np.pi / 4 - np.pi / 2
            L = R * (1.25 if k % 2 == 0 else 0.75)
            tip = (cx + np.cos(a) * L, cy + np.sin(a) * L)
            for side in (-1, 1):
                b = a + side * np.pi / 2
                base = (cx + np.cos(b) * 16, cy + np.sin(b) * 16)
                tri = np.array([tip, base, (cx, cy)], np.int32)
                cv2.fillPoly(rose, [tri], 1.0 if side > 0 else 0.35, cv2.LINE_AA)
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new('L', (MW, MH), 0)
        ImageDraw.Draw(im).text((cx, cy - R * 1.25 - 40), 'N', font=ImageFont.truetype(FONT_EN_BOLD, 64), fill=255, anchor='mm')
        rose = np.maximum(rose, np.asarray(im, np.float32) / 255)
        grat = np.maximum(grat, blur(rose, 0.7) * 0.75)
        f16 = lambda a: a.astype(np.float32)
        return {'land': f16(land), 'sea': f16(sea), 'lines': f16(lines), 'relief': f16(relief), 'coast': f16(coast),
                'mounts': f16(mounts), 'band': f16(band), 'grat': f16(grat)}

    def _ship(self):
        """a small wooden sailing ship facing right: rgb, alpha (world px ~70 wide)"""
        S = 3                                    # drawn at 3x, sampled down by the camera
        w, h = 90 * S, 96 * S
        ss = StrokeSet(w, h)
        hull = [(8, 70), (20, 82), (60, 84), (82, 72), (86, 64), (70, 68), (20, 68)]
        from poly_ink.brush import wash as pwash
        hullw = pwash(w, h, [(x * S, y * S) for x, y in hull], ink=0.85, soft=1.2 * S, seed=2)
        ss.add(hullw, 0, 1)
        ss.add(stroke(w, h, [(10 * S, 68 * S), (45 * S, 70 * S), (86 * S, 62 * S)], 3 * S, ink=0.9, dry=0.2, profile=BLUNT, seed=3), 0, 1)
        ss.add(stroke(w, h, [(44 * S, 70 * S), (45 * S, 8 * S)], 2.2 * S, ink=0.9, dry=0.1, profile=BLUNT, seed=4), 0, 1)
        D = ss.render(5)
        sail = np.zeros((h, w), np.float32)
        poly = np.array([(28, 14), (60, 10), (66, 60), (24, 62)], np.float32) * S
        cv2.fillPoly(sail, [poly.astype(np.int32)], 1.0, cv2.LINE_AA)
        sail = blur(sail, 0.8 * S)
        bat = np.zeros_like(sail)
        for yy in np.linspace(18, 56, 6):
            cv2.line(bat, (int(26 * S), int(yy * S)), (int(64 * S), int((yy - 3) * S)), 1.0, int(1.4 * S), cv2.LINE_AA)
        flag = np.zeros_like(sail)
        cv2.fillPoly(flag, [np.array([(45, 6), (56, 9), (45, 12)], np.int32) * S], 1.0, cv2.LINE_AA)
        rgb = np.ones((h, w, 3), np.float32) * PAPER
        sail_col = np.array([0.80, 0.70, 0.55], np.float32) * (0.9 + 0.1 * noise(h, w, 20, 3, 8))[..., None]
        rgb = rgb * (1 - sail[..., None]) + sail_col * sail[..., None]
        rgb *= (1 - (bat * sail * 0.6)[..., None])
        rgb *= (1 - D[..., None] * (1 - INK))
        rgb = rgb * (1 - flag[..., None]) + VERM * flag[..., None]
        alpha = np.clip(np.maximum.reduce([sail, D * 1.1, flag]), 0, 1)
        return rgb.astype(np.float32), alpha.astype(np.float32), S

    # ------------------------------------------------------------ motion
    def ship_s(self, t):
        """arc length along the route: eases off the coast, steady at sea, eases into harbour"""
        u = (t - self.t_ship0) / (self.t_ship1 - self.t_ship0)
        u = np.clip(u, 0, 1)
        e = np.where(u < 0.12, 0.5 * u ** 2 / 0.12, np.where(u > 0.88, 1 - 0.5 * (1 - u) ** 2 / 0.12 - 0.12 * 0.5 * 0 - 0.06 + 0.06, u - 0.06))
        e = float(np.clip(e / 0.94, 0, 1))
        return e * self.route_s[-1]

    def ship_xy(self, s):
        return (float(np.interp(s, self.route_s, self.route[:, 0])), float(np.interp(s, self.route_s, self.route[:, 1])))

    def _precompute_track(self):
        ts = np.arange(self.t0 - 1, self.t1 + 1, 1 / 30)
        xy = np.array([self.ship_xy(self.ship_s(t)) for t in ts])
        k = np.exp(-0.5 * (np.arange(-30, 31) / 12) ** 2)
        k /= k.sum()
        sm = np.stack([np.convolve(np.pad(xy[:, i], 30, mode='edge'), k, mode='valid') for i in range(2)], 1)
        return ts, sm

    def cam(self, t):
        ts, sm = self._track
        cx0, cy0 = proj(119.6, 36.6)
        sx, sy = proj(122.3, 37.0)
        fx = float(np.interp(t, ts, sm[:, 0])) + 40
        fy = float(np.interp(t, ts, sm[:, 1])) + 60
        wx, wy = proj(129.4, 35.2)
        a = self.t0
        if t < self.t_ship0 + 0.6:
            cx, cy, z = keyframes(t, [(a, (cx0, cy0, 1.7)), (self.t_ship0 + 0.6, (sx + 80, sy + 60, 1.18))], ease=ease_io)
        elif t < self.t_ship1 - 0.3:
            # follow the ship, gently settling the zoom
            cx, cy = fx, fy
            z = keyframes(t, [(self.t_ship0 + 0.6, 1.18), (self.t_ship1 - 0.3, 1.0)], ease=ease_io)
            blend = smooth((t - self.t_ship0 - 0.6) / 1.2)
            cx = sx + 80 + (cx - sx - 80) * blend
            cy = sy + 60 + (cy - sy - 60) * blend
        else:
            px, py = float(np.interp(self.t_ship1 - 0.3, ts, sm[:, 0])) + 40, float(np.interp(self.t_ship1 - 0.3, ts, sm[:, 1])) + 60
            cx, cy, z = keyframes(t, [(self.t_ship1 - 0.3, (px, py, 1.0)), (self.t1 - 0.6, (wx, wy, 0.5))], ease=ease_io)
        return Cam(cx, cy, z)

    # ------------------------------------------------------------ render
    def render(self, t):
        cam = self.cam(t)
        M = cam.matrix()
        fr = self.frame(cam)
        P = lambda a: warp(a, M)
        land = P(self.land)
        sea_mask = P(self.sea_mask) * (1 - P(self.grat) * 0)
        fr.ink(P(self.relief), SEPIA)
        fr.ink(P(self.sea_d), SEA)
        # currents: streaks scrolled north-east, lightening the water
        Ms = M.copy()
        Ms[:, 2] += M[:, :2] @ np.array([-t * 26.0, t * 14.0])
        st = cv2.warpAffine(self.streak, Ms, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        Ms2 = M.copy()
        Ms2[:, 2] += M[:, :2] @ np.array([-t * 15.0 + 200, t * 9.0 + 100])
        st2 = cv2.warpAffine(self.streak, Ms2 @ np.array([[1.6, 0.3, 0], [-0.2, 1.4, 0], [0, 0, 1]], np.float32)[:2] if False else Ms2,
                             (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        cur = (st * 0.6 + st2 * 0.4) * sea_mask
        fr.img = fr.img * (1 - cur[..., None] * 0.38) + np.array([0.86, 0.9, 0.86]) * cur[..., None] * 0.38
        fr.ink(P(self.lines), np.array([0.16, 0.28, 0.30]))
        # foam breathing along the coast
        Mf = M.copy()
        Mf[:, 2] += M[:, :2] @ np.array([np.sin(t * 0.7) * 6, t * 5.0])
        fn = cv2.warpAffine(self.foam_n, Mf, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        band = P(self.coast_band)
        foam = band * sstep(0.45 - 0.15 * band, 0.75, fn) * 0.7
        fr.img = fr.img * (1 - foam[..., None]) + np.array([0.95, 0.93, 0.87]) * foam[..., None]
        fr.ink(P(self.grat), SEPIA)
        fr.ink(P(self.mounts), INK, 0.85)
        fr.ink(P(self.coast), INK, 0.9)

        # labels written in as they are named
        for a, T, x0, y0, ts in self.labels:
            f = (t - ts) / 0.7
            if f > 0:
                d = reveal_time(a, T, f * 1.1, soft=0.06, wet=0.2)
                halo = np.clip(blur(cv2.dilate(d, np.ones((9, 9), np.uint8)), 6) * 1.6, 0, 1)
                hs = self.project(cam, halo, x0, y0)
                fr.img = fr.img * (1 - hs[..., None] * 0.75) + np.array([0.93, 0.89, 0.8]) * hs[..., None] * 0.75
                self.layer(fr, cam, d, x0, y0, color=np.array([0.24, 0.17, 0.11]))

        # fainter routes during the closing wide view
        ff = ramp(t, self.t_ship1 + 0.1, self.t_ship1 + 1.5)
        if ff > 0:
            for k, rt in enumerate(self.faint):
                self._route(fr, M, rt, np.clip(ff * 1.3 - k * 0.15, 0, 1), GREY, 3.0, 0.55, dash=(14, 12))

        # the voyage
        if t > self.t_ship0 - 0.2:
            s = self.ship_s(t)
            n = int(np.searchsorted(self.route_s, s))
            if n > 1:
                self._route(fr, M, np.vstack([self.route[:n], self.ship_xy(s)]), 1.0, VERM, 4.2, 0.95, dash=(18, 10))
            self._draw_ship(fr, M, t, s)
        return fr.img

    def _route(self, fr, M, pts, frac, color, width, ink, dash=(18, 10)):
        P = (np.asarray(pts, np.float32) @ M[:, :2].T) + M[:, 2]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        L = s[-1] * frac
        D = np.zeros((H, W), np.float32)
        z = np.hypot(M[0, 0], M[0, 1])
        on, off = dash[0] * z, dash[1] * z
        a = 0.0
        while a < L:
            b = min(a + on, L)
            q = np.linspace(a, b, 6)
            xy = np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)
            cv2.polylines(D, [np.round(xy * 4).astype(np.int32)], False, 1.0, max(1, int(width * z)), cv2.LINE_AA, shift=2)
            a = b + off
        fr.ink(blur(D, 0.8) * ink, color)

    def _draw_ship(self, fr, M, t, s):
        rgb, alpha, S = self.ship
        x, y = self.ship_xy(s)
        x2, y2 = self.ship_xy(min(s + 20, self.route_s[-1]))
        appear = smooth((t - self.t_ship0 + 0.2) / 0.6)
        if appear <= 0:
            return
        heading = np.arctan2(y2 - y, x2 - x)
        roll = 0.05 * np.sin(t * 2.2) + 0.02 * np.sin(t * 3.7)
        bob = 1.5 * np.sin(t * 2.6)
        # wake: two foam lines trailing behind
        z = np.hypot(M[0, 0], M[0, 1])
        wake = np.zeros((H, W), np.float32)
        back = [self.ship_xy(max(s - d, 0)) for d in range(0, 120, 10)]
        for side in (-1, 1):
            pts = []
            for k, (bx, by) in enumerate(back):
                nx, ny = -np.sin(heading), np.cos(heading)
                spread = 2 + k * 1.8
                pts.append(M[:, :2] @ np.array([bx + side * nx * spread, by + side * ny * spread + 10]) + M[:, 2])
            cv2.polylines(wake, [np.round(np.array(pts) * 4).astype(np.int32)], False, 1.0, max(1, int(2 * z)), cv2.LINE_AA, shift=2)
        fade = np.ones((H, W), np.float32)
        wake = blur(wake, 1.2) * 0.55 * appear
        fr.img = fr.img * (1 - wake[..., None]) + np.array([0.95, 0.94, 0.88]) * wake[..., None]
        # ship sprite: anchored at its keel, scaled to world size, rolling
        sc = z / S * 1.15
        ang = roll + np.clip(heading, -0.35, 0.35) * 0.25
        c, sn = np.cos(ang), np.sin(ang)
        A = np.array([[sc * c, -sc * sn], [sc * sn, sc * c]], np.float32)
        anchor = np.array([45 * S, 76 * S], np.float32)
        sx, sy = M[:, :2] @ np.array([x, y + bob]) + M[:, 2]
        Ms = np.concatenate([A, (np.array([sx, sy]) - A @ anchor)[:, None]], 1)
        al = cv2.warpAffine(alpha, Ms, (W, H), flags=cv2.INTER_AREA) * appear
        co = cv2.warpAffine(rgb, Ms, (W, H), flags=cv2.INTER_LINEAR)
        sh = blur(cv2.warpAffine(alpha, Ms + np.array([[0, 0, 4], [0, 0, 5]], np.float32), (W, H)), 3) * 0.25 * appear
        fr.img *= (1 - sh)[..., None]
        fr.img = fr.img * (1 - al[..., None]) + co * al[..., None]

    # ------------------------------------------------------------ transition in from the landscape
    def enter(self, prev_img, cur_img, f, prev_scene):
        """a curved ink line travels across the landscape, then the map opens out from it: the line becomes
        the Chinese coastline"""
        if not hasattr(self, '_coast_line'):
            cam = self.cam(self.t0)
            M = cam.matrix()
            lb = (warp(self.land, M) > 0.5).astype(np.uint8)
            cs, _ = cv2.findContours(lb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=lambda q: cv2.arcLength(q, False))[:, 0, :].astype(np.float32)
            keep = (c[:, 0] > 2) & (c[:, 0] < W - 3) & (c[:, 1] > 2) & (c[:, 1] < H - 3)
            c = c[keep]
            c = c[::6]
            if c[0, 1] > c[-1, 1]:
                c = c[::-1]
            st = stroke(W, H, c, 7, ink=0.9, dry=0.3, profile=((0, 0.4), (0.1, 1), (0.9, 1), (1, 0.4)), wet=0.15, smooth=4, seed=3)
            canvas = np.zeros((H, W), np.float32)
            from engine import paste
            paste(canvas, st)
            T = np.ones((H, W), np.float32) * 2
            hh, ww = st.dens.shape
            T[st.y0:st.y0 + hh, st.x0:st.x0 + ww] = np.where(st.dens > 0.02, st.time, 2)
            core = (canvas > 0.2).astype(np.uint8)
            dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5).astype(np.float32)
            dmap = dist / dist.max()
            dmap = np.clip(dmap ** 0.7 + (noise(H, W, 110, 4, 33) - 0.5) * 0.25, 0, 1)
            self._coast_line = (canvas, T, dmap)
        canvas, T, dmap = self._coast_line
        line = reveal_time(canvas, T, f / 0.5, soft=0.04, wet=0.4)
        from engine import Frame
        fa = Frame(prev_img)
        fa.ink(line, INK)
        g = np.clip((f - 0.42) / 0.58, 0, 1)
        if g <= 0:
            return fa.img
        out = wipe_mix(fa.img, cur_img, dmap, ease_io(g) * 1.12, soft=0.06, edge=0.35)
        return out

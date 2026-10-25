"""Scene 10 — the journey of kanji: "Over time, Chinese characters developed into a sophisticated writing system.
They later reached Japan through historical cultural contacts, and were adapted to write the Japanese language."

A flat cartographic map in the manner of the reference @raj_in_jp sent (tan land, fine coastlines, a graticule,
rivers and mountain marks, place labels with small dots), a header with an era counter, and dark caption cards.
A dark-red trail is drawn behind the moving marker: overland from Anyang (Shang oracle bones, c. 1200 BCE) to
Chang'an (the Han dynasty, which gives kanji — "Han characters" — its name), on to the Shandong coast; there a
boat takes over and sails, trail behind it, past Baekje on the Korean Peninsula, across to Kyushu and through the
Inland Sea to Naniwa, and the marker reaches Nara (the Kojiki, 712).  Real coastlines (global-land-mask); the
rivers are simplified by hand and the route is a plausible illustration, not a single documented voyage."""
import cv2
import numpy as np

from engine import (FONT_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, H, PAPER, W, blur, cached, ease_io, keyframes,
                    noise, smooth, spline, sstep, text_layer, wash_map, wipe_mix)
from scenes.base import SceneBase
from timeline import at

LON0, LON1, LAT_T, LAT_B = 104.0, 142.0, 44.0, 26.0
KX = np.cos(np.radians(35))
K = 160                                         # world px per degree of latitude
MW, MH = int((LON1 - LON0) * KX * K), int((LAT_T - LAT_B) * K)

LAND = np.array([0.80, 0.68, 0.50], np.float32)
SEA = np.array([0.93, 0.90, 0.82], np.float32)
COAST = np.array([0.34, 0.25, 0.17], np.float32)
ROUTE = np.array([0.55, 0.11, 0.09], np.float32)
CARD = np.array([0.24, 0.19, 0.15], np.float32)
CARD_HEAD = np.array([0.80, 0.68, 0.48], np.float32)
IVORY = np.array([0.96, 0.93, 0.86], np.float32)
BROWN = np.array([0.25, 0.18, 0.12], np.float32)


def proj(lon, lat):
    return (lon - LON0) * KX * K, (LAT_T - lat) * K


YELLOW_R = [(103.8, 36.1), (105.2, 37.5), (106.4, 39.2), (107.5, 40.6), (109.8, 40.6), (111.1, 40.3), (110.6, 38.6),
            (110.5, 36.8), (110.4, 35.6), (111.3, 34.8), (113.6, 34.9), (115.4, 35.4), (116.9, 36.7), (118.9, 37.7)]
YANGTZE = [(104.6, 28.8), (106.5, 29.6), (108.4, 30.8), (110.4, 31.0), (111.3, 30.7), (112.2, 30.2), (113.2, 29.6),
           (114.3, 30.6), (116.0, 29.8), (117.0, 30.5), (118.4, 31.4), (118.8, 32.0), (120.3, 32.0), (121.8, 31.4)]
MOUNTAINS = [(108.0, 112.0, 33.4, 34.0), (113.0, 114.0, 35.6, 39.0), (104.5, 107.0, 32.5, 35.0), (128.3, 129.3, 36.3, 39.0),
             (136.6, 138.4, 35.4, 36.8), (110.0, 112.5, 26.5, 28.5), (127.2, 128.6, 39.5, 41.5)]

LEGS = [  # (lon/lat points, start word offset, by boat?)
    ([(114.35, 36.10), (112.6, 36.25), (110.6, 35.5), (108.94, 34.27)], False),
    ([(108.94, 34.27), (110.7, 34.2), (112.45, 34.62), (114.4, 34.85), (116.9, 36.4), (118.9, 37.2), (120.75, 37.78)], False),
    ([(120.75, 37.78), (120.95, 37.86), (122.4, 37.62), (124.2, 37.35), (125.75, 36.85), (125.85, 35.95), (125.6, 35.0),
      (125.75, 34.25), (126.5, 33.88), (127.5, 34.0), (128.5, 34.42), (129.0, 34.05), (129.55, 33.97), (129.95, 33.72),
      (130.25, 33.68)], True),
    ([(130.25, 33.68), (130.45, 33.95), (130.72, 34.06), (130.95, 33.98), (131.2, 33.92), (131.6, 33.85), (132.3, 33.92),
      (133.1, 34.18), (133.9, 34.3), (134.45, 34.38), (134.85, 34.55), (135.1, 34.62), (135.38, 34.64)], True),
    ([(135.38, 34.64), (135.62, 34.66), (135.83, 34.68)], False),
]


class Scene(SceneBase):
    paper_seed = 50

    def build(self):
        A = cached('k10_map_v1', self._paint)
        self.mip = [A['rgb0'], A['rgb1']]
        # route legs in world px, each with its own timing
        o = at('Over time')[0]
        self.leg_t = [(o + 1.3, o + 2.2), (at('sophisticated')[0] - 0.1, at('system', 1)[0] + 0.3),
                      (at('They later')[0] - 0.2, at('historical')[0] + 0.6), (at('cultural contacts')[1] + 0.1, at('adapted')[0] + 0.2),
                      (at('adapted')[0] + 0.25, at('write the')[0] + 0.2)]
        self.legs = []
        for (pts, boat), (a, b) in zip(LEGS, self.leg_t):
            P = spline(np.array([proj(*p) for p in pts], np.float32), 16)
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            self.legs.append(dict(P=P, s=np.concatenate([[0], np.cumsum(seg)]), boat=boat, a=a, b=b))
        self.t_start = o + 0.2
        # header
        self.hdr = [(text_layer('@RAJ_IN_JP', FONT_EN, 30, tracking=6)[0], 255, BROWN * 1.6, 0.0),
                    (text_layer('THE JOURNEY OF KANJI', FONT_EN_BOLD, 76, tracking=4)[0], 312, BROWN, 0.1),
                    (text_layer('CHINA · KOREA · JAPAN', FONT_EN, 38, tracking=7)[0], 378, BROWN * 1.4, 0.2)]
        self.eras = [(self.t_start, 'c. 1200 BCE · Shang dynasty'), (self.leg_t[0][1] + 0.1, '206 BCE – 220 CE · Han dynasty'),
                     (at('Japan through')[0], '4th–5th century · Baekje, Korea'), (self.leg_t[4][1], '8th century · Nara, Japan')]
        self.era_txt = [(t, text_layer(s, FONT_EN_ITALIC, 44)[0]) for t, s in self.eras]
        # caption cards
        self.cards = [
            (self.t_start + 0.4, self.leg_t[0][0] + 0.5, 'C. 1200 BCE · ANYANG', ['Oracle bones of the Shang dynasty']),
            (self.leg_t[0][1] + 0.15, self.leg_t[1][1] - 0.2, '206 BCE – 220 CE · CHANG’AN',
             ['The Han dynasty: kanji means', '“Han characters”']),
            (at('Japan through')[0] + 0.05, self.leg_t[3][0] + 0.4, '4TH–5TH CENTURY · BAEKJE',
             ['Books and scholars cross', 'from Korea to Japan']),
            (self.leg_t[4][1] + 0.05, self.t1 + 1, '8TH CENTURY · NARA',
             ['The Kojiki (712), Japan’s oldest', 'book, is written in kanji'])]
        self.card_img = [self._card(h, lines) for _, _, h, lines in self.cards]
        # labels: (text, font, size, tracking, lon, lat, appear time, colour, dot)
        t_wide = self.leg_t[4][1] + 0.2
        L = [('CHINA', FONT_EN_BOLD, 64, 14, 112.6, 32.2, self.t_start, BROWN, False),
             ('Anyang 安陽', FONT_EN, 40, 0, 114.35, 36.1, self.t_start + 0.2, BROWN, True),
             ('Chang’an 長安', FONT_EN, 40, 0, 108.94, 34.27, self.leg_t[0][1] - 0.2, BROWN, 'below'),
             ('Yellow River', FONT_EN_ITALIC, 34, 0, 111.6, 39.6, self.t_start + 0.6, np.array([0.3, 0.38, 0.42]), False),
             ('Yellow Sea', FONT_EN_ITALIC, 40, 2, 123.6, 35.6, self.leg_t[2][0], np.array([0.42, 0.36, 0.28]), False),
             ('KOREAN\nPENINSULA', FONT_EN_BOLD, 38, 8, 127.3, 40.0, self.leg_t[2][0] + 0.4, BROWN, False),
             ('Baekje 百済', FONT_EN, 40, 0, 127.12, 36.46, at('Japan through')[0], BROWN, True),
             ('JAPAN', FONT_EN_BOLD, 64, 14, 135.0, 36.9, self.leg_t[3][0], BROWN, False),
             ('Sea of Japan', FONT_EN_ITALIC, 40, 2, 133.6, 39.6, self.leg_t[3][0] + 0.3, np.array([0.42, 0.36, 0.28]), False),
             ('Nara 奈良', FONT_EN_BOLD, 44, 0, 135.83, 34.68, self.leg_t[4][1] - 0.1, BROWN, 'left')]
        self.labels = []
        for text, font, size, tr, lon, lat, t, col, dot in L:
            if any(ord(c) > 0x3000 for c in text):
                en, jp = text.rsplit(' ', 1)
                a1, _ = text_layer(en, font, size)
                a2, _ = text_layer(jp, FONT_JP, int(size * 0.82))
                h = max(a1.shape[0], a2.shape[0])
                a = np.zeros((h, a1.shape[1] + a2.shape[1] + 8), np.float32)
                a[(h - a1.shape[0]) // 2:(h - a1.shape[0]) // 2 + a1.shape[0], :a1.shape[1]] = a1
                a[(h - a2.shape[0]) // 2:(h - a2.shape[0]) // 2 + a2.shape[0], a1.shape[1] + 8:] = a2
            elif '\n' in text:
                rows = [text_layer(r, font, size, tracking=tr)[0] for r in text.split('\n')]
                a = np.zeros((sum(r.shape[0] for r in rows) - 8, max(r.shape[1] for r in rows)), np.float32)
                y = 0
                for r in rows:
                    x = (a.shape[1] - r.shape[1]) // 2
                    a[y:y + r.shape[0], x:x + r.shape[1]] = np.maximum(a[y:y + r.shape[0], x:x + r.shape[1]], r[:a.shape[0] - y])
                    y += r.shape[0] - 8
            else:
                a, _ = text_layer(text, font, size, tracking=tr)
            halo = np.clip(blur(cv2.dilate(a, np.ones((7, 7), np.uint8)), 4) * 1.6, 0, 1)
            minor = font == FONT_EN_ITALIC                      # sea and river names leave the closing wide view
            self.labels.append(dict(a=a, halo=halo, xy=proj(lon, lat), t=t, col=np.asarray(col, np.float32), dot=dot,
                                    t_out=t_wide if minor else 1e9))
        self.ship = self._ship()

    # ------------------------------------------------------------ the map, painted once
    def _paint(self):
        from global_land_mask import globe
        lon = LON0 + np.arange(MW, dtype=np.float32) / (KX * K)
        lat = LAT_T - np.arange(MH, dtype=np.float32) / K
        LO, LA = np.meshgrid(lon, lat)
        lb = globe.is_land(LA, LO).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lb, 8)
        lb[np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] < 40)[0][1:])] = 0
        land = blur(lb.astype(np.float32), 0.9)
        d_sea = cv2.distanceTransform(1 - lb, cv2.DIST_L2, 5).astype(np.float32)
        d_land = cv2.distanceTransform(lb, cv2.DIST_L2, 5).astype(np.float32)
        mot = noise(MH, MW, 220, 4, 3)
        rgb = SEA * (0.97 + 0.03 * noise(MH, MW, 400, 3, 1))[..., None]
        rgb = rgb * (1 - 0.06 * np.exp(-d_sea / 26))[..., None]                       # sea darkens a touch at the coast
        land_rgb = LAND * (0.94 + 0.08 * mot)[..., None] * (1 + 0.05 * np.clip(1 - d_land / 6, 0, 1))[..., None]
        rgb = rgb * (1 - land[..., None]) + land_rgb * land[..., None]
        # graticule every 2 degrees
        g = np.zeros((MH, MW), np.float32)
        for lo in range(106, 142, 2):
            x = int(proj(lo, 0)[0]); g[:, x:x + 2] = 1
        for la in range(28, 44, 2):
            y = int(proj(0, la)[1]); g[y:y + 2, :] = 1
        rgb *= (1 - 0.10 * blur(g, 0.8))[..., None]
        # rivers
        rv = np.zeros((MH, MW), np.float32)
        for river in (YELLOW_R, YANGTZE):
            P = spline(np.array([proj(*p) for p in river], np.float32), 12)
            cv2.polylines(rv, [np.round(P * 4).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA, shift=2)
        rv = blur(rv, 0.7) * land
        rgb = rgb * (1 - rv[..., None] * 0.75) + np.array([0.42, 0.52, 0.56], np.float32) * rv[..., None] * 0.75
        # mountain marks: small shaded triangles
        r = np.random.default_rng(4)
        mt_dark, mt_light = np.zeros((MH, MW), np.float32), np.zeros((MH, MW), np.float32)
        for lo0, lo1, la0, la1 in MOUNTAINS:
            for _ in range(int((lo1 - lo0) * (la1 - la0) * 9)):
                x, y = proj(r.uniform(lo0, lo1), r.uniform(la0, la1))
                if lb[int(np.clip(y, 0, MH - 1)), int(np.clip(x, 0, MW - 1))] == 0 or d_land[int(y), int(x)] < 10:
                    continue
                s = r.uniform(9, 14)
                cv2.fillPoly(mt_light, [np.array([(x - s, y), (x, y - s * 1.2), (x, y)], np.int32)], 1.0, cv2.LINE_AA)
                cv2.fillPoly(mt_dark, [np.array([(x, y - s * 1.2), (x + s, y), (x, y)], np.int32)], 1.0, cv2.LINE_AA)
        rgb = rgb * (1 - 0.18 * mt_dark[..., None]) + IVORY * (0.25 * mt_light[..., None])
        # coastline: a dark line on the edge and a pale line just inside it
        edge = np.clip(1 - np.abs(d_sea - 0.5) / 1.2, 0, 1) * (lb == 0) + np.clip(1 - d_land / 1.4, 0, 1) * lb
        inner = np.clip(1 - np.abs(d_land - 4.0) / 1.0, 0, 1) * lb
        rgb = rgb * (1 - 0.35 * blur(inner, 0.6)[..., None]) + IVORY * 0.35 * blur(inner, 0.6)[..., None]
        e = blur(edge, 0.6)[..., None]
        rgb = rgb * (1 - e * 0.85) + COAST * e * 0.85
        rgb0 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb1 = cv2.resize(rgb0, (MW // 2, MH // 2), interpolation=cv2.INTER_AREA)
        return {'rgb0': rgb0, 'rgb1': rgb1}

    def _card(self, head, lines):
        h_a, _ = text_layer(head, FONT_EN, 30, tracking=3)
        ls = [text_layer(l, FONT_EN, 44)[0] for l in lines]
        w = max([h_a.shape[1]] + [a.shape[1] for a in ls]) + 64
        hh = 26 + h_a.shape[0] + 8 + sum(a.shape[0] for a in ls) + 4 * (len(ls) - 1) + 26
        return dict(head=h_a, lines=ls, w=w, h=hh)

    def _ship(self):
        S = 3
        w, h = 90 * S, 96 * S
        hull = np.zeros((h, w), np.float32)
        cv2.fillPoly(hull, [np.array([(8, 68), (20, 82), (62, 84), (84, 70), (86, 62), (70, 66), (20, 66)], np.int32) * S], 1.0, cv2.LINE_AA)
        mast = np.zeros_like(hull)
        cv2.line(mast, (45 * S, 68 * S), (45 * S, 6 * S), 1.0, int(2.4 * S), cv2.LINE_AA)
        sail = np.zeros_like(hull)
        cv2.fillPoly(sail, [np.array([(26, 14), (62, 10), (68, 58), (22, 60)], np.int32) * S], 1.0, cv2.LINE_AA)
        bat = np.zeros_like(hull)
        for yy in np.linspace(18, 54, 5):
            cv2.line(bat, (int(24 * S), int(yy * S)), (int(66 * S), int((yy - 3) * S)), 1.0, int(1.3 * S), cv2.LINE_AA)
        flag = np.zeros_like(hull)
        cv2.fillPoly(flag, [np.array([(45, 4), (57, 8), (45, 12)], np.int32) * S], 1.0, cv2.LINE_AA)
        sail = blur(sail, 0.6 * S)
        rgb = np.ones((h, w, 3), np.float32) * np.array([0.93, 0.86, 0.72], np.float32)
        rgb = rgb * (1 - (bat * sail * 0.45)[..., None])
        dark = np.clip(hull + mast, 0, 1)
        rgb = rgb * (1 - dark[..., None]) + BROWN * dark[..., None]
        rgb = rgb * (1 - flag[..., None]) + ROUTE * flag[..., None]
        outline = np.clip(blur(sail, 1.2 * S) - sail * 0.0, 0, 1)
        alpha = np.clip(np.maximum.reduce([sail, dark, flag]), 0, 1)
        ring = np.clip(cv2.dilate(alpha, np.ones((5, 5), np.uint8)) - alpha, 0, 1)
        rgb = rgb * (1 - ring[..., None]) + BROWN * ring[..., None]
        alpha = np.maximum(alpha, ring)
        return rgb.astype(np.float32), blur(alpha, 0.6).astype(np.float32), S

    # ------------------------------------------------------------ motion
    def head(self, t):
        """(trail polylines so far, head xy, heading, on a boat?)"""
        drawn = []
        for i, L in enumerate(self.legs):
            if t < L['a']:
                break
            u = ease_io((t - L['a']) / (L['b'] - L['a'])) if t < L['b'] else 1.0
            s = u * L['s'][-1]
            n = int(np.searchsorted(L['s'], s))
            p = (float(np.interp(s, L['s'], L['P'][:, 0])), float(np.interp(s, L['s'], L['P'][:, 1])))
            drawn.append(np.vstack([L['P'][:n], p]))
            q = L['P'][min(n + 1, len(L['P']) - 1)]
            heading = np.arctan2(q[1] - p[1], q[0] - p[0])
            nxt_boat = i + 1 < len(self.legs) and self.legs[i + 1]['boat']
            boat = L['boat'] and (t < L['b'] + 0.05 or nxt_boat)
            last = (p, heading, boat)
            if t < L['b']:
                return drawn, p, heading, boat
        if not drawn:
            return [], tuple(self.legs[0]['P'][0]), 0.0, False
        return drawn, last[0], last[1], last[2]

    def cam(self, t):
        if not hasattr(self, '_track'):
            ts = np.arange(self.t0 - 0.5, self.t1 + 0.5, 1 / 30)
            xy = np.array([self.head(tt)[1] for tt in ts], np.float32)
            k = np.exp(-0.5 * (np.arange(-24, 25) / 9) ** 2)
            k /= k.sum()
            sm = np.stack([np.convolve(np.pad(xy[:, i], 24, mode='edge'), k, mode='valid') for i in range(2)], 1)
            self._track = (ts, sm)
        ts, sm = self._track
        hx, hy = float(np.interp(t, ts, sm[:, 0])), float(np.interp(t, ts, sm[:, 1]))
        z = keyframes(t, [(self.t0, 0.78), (self.t_start + 1.2, 1.0), (self.leg_t[0][1], 1.0), (self.leg_t[1][0], 1.0), (self.leg_t[1][1], 0.62),
                          (self.leg_t[2][1], 0.56), (self.leg_t[3][1], 0.62), (self.leg_t[4][1] + 0.2, 0.62),
                          (self.t1 - 0.3, 0.275)], ease=ease_io)
        wide = smooth((t - self.leg_t[4][1] - 0.2) / 1.6)
        cx, cy = proj(122.4, 35.2)
        # the head sits a little above the centre of the frame (the card is below)
        return hx + (cx - hx) * wide, hy + 90 / z + (cy - hy) * wide, z

    def render(self, t):
        cx, cy, z = self.cam(t)
        lvl = 1 if z < 0.55 else 0
        src = self.mip[lvl]
        zs = z * (2 ** lvl)
        M = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)
        Ms = np.array([[zs, 0, W / 2 - z * cx], [0, zs, H / 2 - z * cy]], np.float32)
        base = cv2.warpAffine(src, Ms, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
        from engine import Frame
        fr = Frame(base * (self.paper / np.asarray(PAPER, np.float32)))
        P = lambda xy: M[:, :2] @ np.asarray(xy, np.float32) + M[:, 2]
        # the trail and the marker / boat
        drawn, p, heading, boat = self.head(t)
        if t > self.t_start - 0.1:
            tr = np.zeros((H, W), np.float32)
            for seg in drawn:
                S = (seg @ M[:, :2].T) + M[:, 2]
                cv2.polylines(tr, [np.round(S * 4).astype(np.int32)], False, 1.0, 5, cv2.LINE_AA, shift=2)
            fr.over(IVORY, blur(cv2.dilate(tr, np.ones((3, 3), np.uint8)), 1.2), 0.5)
            fr.ink(blur(tr, 0.5), ROUTE, 0.95)
            x, y = P(p)
            ap = smooth((t - self.t_start + 0.1) / 0.4)
            if boat:
                self._draw_ship(fr, x, y, t, heading, ap)
            else:
                d = np.zeros((H, W), np.float32)
                ring = np.zeros((H, W), np.float32)
                pulse = 1 + 0.35 * np.exp(-((t - self.t_start) % 99) * 3) if t < self.t_start + 1.5 else 1
                cv2.circle(ring, (int(x * 4), int(y * 4)), int(4 * 15 * pulse), 1.0, -1, cv2.LINE_AA, shift=2)
                cv2.circle(d, (int(x * 4), int(y * 4)), 4 * 10, 1.0, -1, cv2.LINE_AA, shift=2)
                fr.over(IVORY, blur(ring, 0.8), ap)
                fr.over(ROUTE, blur(d, 0.6), ap)
        # labels and dots (above the trail), shrinking a little in the wide views
        ls = float(np.clip((z / 0.62) ** 0.5, 0.72, 1.1))
        for L in self.labels:
            op = smooth((t - L['t']) / 0.45) * (1 - smooth((t - L['t_out']) / 0.5))
            if op <= 0:
                continue
            x, y = P(L['xy'])
            a, halo = L['a'], L['halo']
            if abs(ls - 1) > 0.02:
                a = cv2.resize(a, None, fx=ls, fy=ls, interpolation=cv2.INTER_AREA)
                halo = cv2.resize(halo, None, fx=ls, fy=ls, interpolation=cv2.INTER_AREA)
            h, w = a.shape
            if L['dot']:
                d = np.zeros((H, W), np.float32)
                cv2.circle(d, (int(x * 4), int(y * 4)), 4 * 6, 1.0, -1, cv2.LINE_AA, shift=2)
                fr.ink(blur(d, 0.6), BROWN, op)
                x0 = {'left': int(x - w - 16), 'below': int(x - w / 2)}.get(L['dot'], int(x + 16))
                y0 = int(y + 14) if L['dot'] == 'below' else int(y - h / 2 - 2)
            else:
                x0, y0 = int(x - w / 2), int(y - h / 2)
            self._blit(fr, halo, x0, y0, LAND * 1.08, 0.6 * op, over=True)
            self._blit(fr, a, x0, y0, L['col'], op)
        # header over a paper fade at the top
        yy = np.arange(H, dtype=np.float32)[:, None]
        fade = np.clip(1 - (yy - 430) / 170, 0, 1) * np.ones((1, W), np.float32)
        fr.over(self.paper, fade, 0.92)
        for a, y, col, dt in self.hdr:
            op = smooth((t - self.t0 - dt) / 0.5)
            self._blit(fr, a, int(540 - a.shape[1] / 2), int(y - a.shape[0] / 2), col, op)
        rule = np.zeros((H, W), np.float32)
        rule[410:413, 490:590] = 1
        fr.ink(rule, ROUTE, smooth((t - self.t0 - 0.3) / 0.5))
        for k, (te, a) in enumerate(self.era_txt):
            nxt = self.era_txt[k + 1][0] if k + 1 < len(self.era_txt) else 1e9
            op = smooth((t - te) / 0.3) * (1 - smooth((t - nxt + 0.15) / 0.25))
            self._blit(fr, a, int(540 - a.shape[1] / 2), int(458 - a.shape[0] / 2), BROWN * 1.3, op)
        # caption cards
        for (ta, tb, _, _), c in zip(self.cards, self.card_img):
            op = smooth((t - ta) / 0.3) * (1 - smooth((t - tb) / 0.3))
            if op > 0:
                self._draw_card(fr, c, 1190 + 14 * (1 - op), op)
        return fr.img

    def _blit(self, fr, a, x0, y0, col, op, over=False):
        if op <= 0:
            return
        h, w = a.shape
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb <= xa or yb <= ya:
            return
        sub = a[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * op
        reg = fr.img[ya:yb, xa:xb]
        col = np.asarray(col, np.float32)
        if over:
            reg[:] = reg * (1 - sub) + col * sub
        else:
            reg[:] = reg * (1 - sub * (1 - col))

    def _draw_card(self, fr, c, cy, op):
        w, h = c['w'], c['h']
        x0, y0 = int(540 - w / 2), int(cy - h / 2)
        box = np.zeros((H, W), np.float32)
        box[y0:y0 + h, x0:x0 + w] = 1
        sh = blur(np.roll(box, 10, 0), 12)
        fr.img *= (1 - 0.3 * op * sh)[..., None]
        fr.over(CARD, blur(box, 0.7), 0.94 * op)
        edge = np.zeros_like(box)
        edge[y0:y0 + h, x0:x0 + 6] = 1
        fr.over(ROUTE, edge, op)
        y = y0 + 26
        self._blit(fr, c['head'], x0 + 32, y, CARD_HEAD, op, over=True)
        y += c['head'].shape[0] + 8
        for a in c['lines']:
            self._blit(fr, a, x0 + 32, y, IVORY, op, over=True)
            y += a.shape[0] + 4

    def _draw_ship(self, fr, x, y, t, heading, op):
        rgb, alpha, S = self.ship
        sc = 0.95 / S * 1.0
        roll = 0.05 * np.sin(t * 2.4)
        c, s = np.cos(roll), np.sin(roll)
        A = np.array([[sc * c, -sc * s], [sc * s, sc * c]], np.float32)
        anchor = np.array([45 * S, 76 * S], np.float32)
        Ms = np.concatenate([A, (np.array([x, y + 1.5 * np.sin(t * 2.7)]) - A @ anchor)[:, None]], 1)
        al = cv2.warpAffine(alpha, Ms, (W, H), flags=cv2.INTER_AREA) * op
        co = cv2.warpAffine(rgb, Ms, (W, H), flags=cv2.INTER_LINEAR)
        sh = blur(cv2.warpAffine(alpha, Ms + np.array([[0, 0, 5], [0, 0, 6]], np.float32), (W, H)), 3) * 0.25 * op
        fr.img *= (1 - sh)[..., None]
        fr.img = fr.img * (1 - al[..., None]) + co * al[..., None]

    def enter(self, prev_img, cur_img, f, prev_scene):
        from scenes.base import ink_bloom
        return ink_bloom(prev_img, cur_img, f, (540, 900), seed=self.paper_seed)

"""Scene 2 — why (short cut).  "Japan already had its own words when Chinese characters arrived.  So many kanji kept
both kinds of reading: a Chinese-based sound, on'yomi, and a Japanese word, kun'yomi."
The sea chart of East Asia from real coastlines.  "Its own words": やま ひ みず ひと rise off the islands in indigo.
"When Chinese characters arrived": the junk sails from the Chinese coast past Korea to Kyushu, the camera following
it, a dotted vermilion route behind it.  "Kept both kinds": 山 日 水 人 lift off it onto an open book; on "a Chinese-
based sound" each takes its on'yomi below it (vermilion) and 音読み ON'YOMI is set over the sea, on "a Japanese
word" its kun'yomi above it (indigo) and 訓読み KUN'YOMI."""
import cv2
import numpy as np

import kit
import r5art
from cards import INDIGO
from engine import (FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, H, INK, VERM, W, Cam, Frame, StrokeSet, cached,
                    ease_io, ease_out, keyframes, lerp, smooth, spline, stroke, text_layer, warp)
from glyphs import typeset
from scenes.kscene import KanjiScene as SceneBase
from timeline import JA, at

KANJI = '山日水人'
KUN = ['やま', 'ひ', 'みず', 'ひと']
ON = ['サン', 'ニチ', 'スイ', 'ジン']
SLOTS = [175, 395, 685, 905]          # the four characters on the book's two pages
BOOK_Y, KJ_Y, KUN_Y, ON_Y = 1150, 1150, 1030, 1268
ROUTE = [(120.9, 36.0), (122.6, 36.4), (124.6, 36.2), (125.7, 35.0), (126.1, 34.0), (127.3, 33.95), (128.6, 34.45),
         (129.3, 34.1), (129.9, 33.75), (130.25, 33.68)]      # Shandong coast, past the Korean peninsula, to Hakata
PUFF = [(136.0, 35.3), (132.2, 34.6), (139.6, 37.2), (130.8, 32.4)]   # where each word rises from, on the islands
CARD = np.array([0.97, 0.95, 0.89], np.float32)


class Scene(SceneBase):
    paper_seed = 92

    def build(self):
        C = cached('s02_chart_v2', r5art.east_asia_chart)
        self.chart = C['rgb'].astype(np.float32)
        self.J = cached('s02_junk_v1', lambda: dict(zip(('ink', 'red'), r5art.junk())))
        self.B = cached('s02_book_v1', lambda: dict(zip(('fill', 'ink'), r5art.manuscript())))
        self.kj = [typeset(c, 118, font=FONT_BRUSH_JP, seed=200 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate(KANJI)]
        self.kun = [text_layer(r, FONT_JP, 50)[0] for r in KUN]
        self.on = [text_layer(r, FONT_JP, 50)[0] for r in ON]
        self.puff = [text_layer(r, FONT_JP, 46)[0] for r in KUN]
        self.on_big, _ = text_layer('音読み', FONT_JP, 96)
        self.kun_big, _ = text_layer('訓読み', FONT_JP, 96)
        self.on_en, _ = text_layer("ON'YOMI", FONT_EN_BOLD, 40, tracking=6)
        self.kun_en, _ = text_layer("KUN'YOMI", FONT_EN_BOLD, 40, tracking=6)
        self.on_def, _ = text_layer('Chinese-derived reading', FONT_EN_ITALIC, 34)
        self.kun_def, _ = text_layer('Native Japanese reading', FONT_EN_ITALIC, 34)
        self.t_spoken = at('its own words')[0] - 0.3
        self.t_ship = at('when Chinese')[0] - 0.2
        self.t_dock = at('arrived.')[1]                               # the junk docks as Raj says "arrived" (wood knock)
        self.t_fly = at('So many kanji')[0]
        self.t_pres = at('a Chinese-based')[0] - 0.2
        self.t_on = JA['onyomi']['start']
        self.t_own = at('a Japanese word')[0] - 0.2
        self.t_kun = JA['kunyomi']['start']
        self.t_split = self.t1 + 5                                    # (no split in the short cut)
        self.route = np.array(spline([tuple(r5art.chart_xy(lo, la)) for lo, la in ROUTE], 24), np.float32)
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(self.route, axis=0).T))]
        self.route_s = seg / seg[-1]
        self.dots = np.array([self.route_at(u) for u in np.linspace(0, 1, int(seg[-1] / 20))])
        self.puff_xy = [r5art.chart_xy(lo, la) for lo, la in PUFF]
        wide = (r5art.chart_xy(129.0, 31.5)[0], r5art.chart_xy(129.0, 31.5)[1], 0.62)
        start = self.route[0]
        dock = self.route[-1]
        hold = (dock[0] + 290, dock[1] + 330, 1.0)                     # Japan in the top half once the book is up
        self.keys = [(self.t0, wide), (self.t_ship - 0.5, wide), (self.t_ship + 0.6, (start[0] + 240, start[1] + 120, 1.0)),
                     (self.t_dock - 0.2, (dock[0] + 60, dock[1] + 60, 1.0)), (self.t_dock + 1.1, hold)]
        # the screen-space strokes: on-lines back to the docked ship, the splitting brushstroke
        ds = self.mcam(self.t_pres).matrix() @ np.array([dock[0], dock[1], 1.0])
        self.dock_screen = ds
        ss = StrokeSet(W, H)
        for k, x in enumerate(SLOTS):
            ss.add(stroke(W, H, spline([(ds[0], ds[1] + 20), (lerp(ds[0], x, 0.5) + 30, 880), (x, ON_Y - 40)], 14), 2.4,
                          ink=0.75, dry=0.6, profile=((0, 0.3), (0.2, 1), (1, 0.6)), seed=300 + k, red=True), self.t_pres + 0.15 * k, 0.4)
        self.lines = ss
        sp = StrokeSet(W, H)
        sp.add(stroke(W, H, spline([(540, 250), (536, 340), (540, 420)], 8), 30, ink=0.92, dry=0.5,
                      profile=((0, 0.5), (0.1, 1), (1, 0.9)), seed=310), self.t_split, 0.35)
        sp.add(stroke(W, H, spline([(540, 410), (470, 470), (330, 520), (230, 560)], 14), 24, ink=0.95, dry=0.55,
                      profile=((0, 1), (0.9, 0.8), (1, 0.2)), seed=311, red=True), self.t_split + 0.3, 0.45)
        sp.add(stroke(W, H, spline([(540, 410), (610, 470), (750, 520), (850, 560)], 14), 24, ink=0.95, dry=0.55,
                      profile=((0, 1), (0.9, 0.8), (1, 0.2)), seed=312), self.t_split + 0.3, 0.45)
        self.split = sp
        tick = StrokeSet(W, H)
        for k, x in enumerate(SLOTS):
            tick.add(stroke(W, H, [(x - 40, KUN_Y + 34), (x + 40, KUN_Y + 33)], 4, ink=0.9, dry=0.4, profile=((0, 0.3), (0.2, 1), (1, 0.4)),
                            seed=320 + k), self.t_own + 0.15 * k + 0.15, 0.25)
        self.tick = tick
        self.scr = Cam(540, 960, 1.0)

    def route_at(self, u):
        return np.array([np.interp(u, self.route_s, self.route[:, 0]), np.interp(u, self.route_s, self.route[:, 1])], np.float32)

    def ship_u(self, t):
        return ease_io((t - self.t_ship) / (self.t_dock - self.t_ship))

    def mcam(self, t):
        x, y, z = keyframes(t, self.keys, ease=ease_io)
        return Cam(x, y, z)

    def render(self, t):
        mc = self.mcam(t)
        scr = self.scr
        paper = self.frame(scr).img
        chart = warp(self.chart, mc.matrix(), border='reflect')
        fade = smooth((t - self.t_split + 0.1) / 0.6)
        fr = Frame(chart * (1 - fade) + paper * fade)
        mp = 1 - fade
        M = mc.matrix()
        proj = lambda p: M @ np.array([p[0], p[1], 1.0])
        # the words people already said, rising off the islands
        for k, (g, p) in enumerate(zip(self.puff, self.puff_xy)):
            f = (t - self.t_spoken - 0.35 * k) / 0.5
            if f > 0:
                x, y = proj(p)
                o = smooth(f) * (1 - smooth((t - self.t_ship - 0.2) / 0.5)) * mp
                kit.put(fr, scr, g, x, y - 30 * ease_out(min(f / 3, 1)), INDIGO, o)
        # the route and the junk
        if t > self.t_ship - 0.3 and mp > 0:
            u = self.ship_u(t)
            d = np.zeros((H, W), np.float32)
            for q in self.dots[: int(u * len(self.dots))]:
                x, y = proj(q)
                if -10 < x < W + 10 and -10 < y < H + 10:
                    cv2.circle(d, (int(x), int(y)), 4, 1.0, -1, cv2.LINE_AA)
            fr.ink(d, VERM, 0.85 * mp)
            x, y = proj(self.route_at(u))
            bob = 3 * np.sin(t * 4) + 7 * np.exp(-max(t - self.t_dock, 0) * 6) * np.sin(max(t - self.t_dock, 0) * 20) * (t > self.t_dock)
            o = smooth((t - self.t_ship + 0.3) / 0.4) * mp
            sc = 0.6 * mc.zoom
            kit.put(fr, scr, self.J['ink'], x, y - 60 * sc + bob, INK, o, sc)
            kit.put(fr, scr, self.J['red'], x, y - 60 * sc + bob, VERM, 0.9 * o, sc)
            if t > self.t_dock:                                       # a ripple ring as it docks
                r = (t - self.t_dock) * 160
                ring = np.zeros((H, W), np.float32)
                cv2.ellipse(ring, (int(x), int(y + 4)), (int(r), int(r * 0.35)), 0, 0, 360, 1.0, 2, cv2.LINE_AA)
                fr.ink(ring, INK, 0.5 * np.exp(-(t - self.t_dock) * 3) * mp)
        # the book slides up over the sea
        bo = ease_io((t - self.t_fly + 0.4) / 0.6)
        if bo > 0:
            y = lerp(BOOK_Y + 500, BOOK_Y, bo)
            kit.put(fr, scr, kit.blur(self.B['fill'], 12), 548, y + 16, INK, 0.25 * bo)
            kit.put(fr, scr, self.B['fill'], 540, y, CARD, smooth(bo * 2), over=True)
            kit.put(fr, scr, self.B['ink'], 540, y, INK, smooth(bo * 2))
        # the four characters lift off the ship and settle on the book
        ds = self.dock_screen
        for k, g in enumerate(self.kj):
            f = (t - self.t_fly - 0.18 * k) / 0.8
            if f <= 0:
                continue
            u = ease_io(min(f, 1))
            x = lerp(ds[0], SLOTS[k], u)
            y = lerp(ds[1] - 60, KJ_Y, u) - 160 * np.sin(np.pi * u)
            kit.put(fr, scr, g.dens, x, y, INK, smooth(f * 3), lerp(0.4, 1.0, u))
        # their own words above (indigo), their Chinese-derived sounds below (vermilion)
        for k in range(4):
            fk = (t - self.t_own - 0.15 * k) / 0.3
            if fk > 0:
                kit.put(fr, scr, self.kun[k], SLOTS[k], KUN_Y - 6, INDIGO, smooth(fk), 1 + 0.3 * np.exp(-fk * 2.5))
            fo = (t - self.t_pres - 0.15 * k) / 0.3
            if fo > 0:
                kit.put(fr, scr, self.on[k], SLOTS[k], ON_Y, VERM, smooth(fo), 1 + 0.3 * np.exp(-fo * 2.5))
        self.layer(fr, scr, self.tick.render(t), color=INDIGO, opacity=0.9)
        self.layer(fr, scr, self.lines.render(t, red=True), color=VERM, opacity=0.85 * mp)
        # 音読み ON'YOMI / 訓読み KUN'YOMI over the sea, as Raj names them
        for y, big, en, de, col, t0 in ((330, self.on_big, self.on_en, self.on_def, VERM, self.t_on - 0.1),
                                        (560, self.kun_big, self.kun_en, self.kun_def, INDIGO, self.t_kun - 0.1)):
            f = (t - t0) / 0.3
            if f > 0:
                self.plate(fr, scr, 540, y + 20, 760, 190, smooth(f))
                kit.put(fr, scr, big, 400, y, col, smooth(f), 1 + 0.2 * np.exp(-f * 2.5))
                kit.put(fr, scr, en, 700, y - 22, col, smooth(f - 0.4))
                kit.put(fr, scr, de, 700, y + 32, INK, smooth(f - 0.7))
        return fr.img

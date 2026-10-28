"""Scene 1 — the hook: "One Japanese character can have multiple readings.  But why?  The answer goes back to how
Japan adapted Chinese writing."
Frame 1 is 山 being brushed across the whole screen, ink still flying.  Then 日, 水 and 人 slam in the same way, one
after another (each with a taiko hit and a camera jolt), and each flies up into its own strip of paper, where it
turns into what it means: the ink rises into a mountain, a vermilion sun blooms, a river starts to flow, a traveller
stands on a ridge.  The four freeze side by side, and as Raj says "multiple readings" each one's readings are set
under it in two rows: the kun'yomi (native Japanese words, indigo, 訓) やま ひ みず ひと and the on'yomi
(Chinese-derived, vermilion, 音) サン ニチ スイ ジン.  "ONE CHARACTER. / MULTIPLE READINGS." is set under them, and on "why?" a vermilion WHY? is stamped across the page.  On "adapted Chinese writing" a long dry brush
sweeps across the bottom, carrying us into the history."""
import numpy as np

import kit
from art import blossom_cluster, cherry_branch, figure, sun_disc
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_HEAD, FONT_JP, GREY, H, INK, SAKURA, SEA, VERM, W, Cam,
                    StrokeSet, blur, cached, ease_io, ease_out, lerp, noise, reveal_time, smooth, spline, stroke, text_layer,
                    wash_map)
from glyphs import typeset
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from timeline import at

KANJI = '山日水人'
LABELS = ['MOUNTAIN', 'SUN / DAY', 'WATER', 'PERSON']
COLS = [138, 406, 674, 942]
PW, PH = 252, 540            # one strip of paper
TOP = 400                    # top of the strips (world y)
KY = 300                     # the kanji row
S = 0.8                      # the strips are painted at PW x PH and shown at this scale
STRIP_CY, BOTTOM = TOP + PH * S / 2, TOP + PH * S
KUN = ['やま', 'ひ', 'みず', 'ひと']          # kun'yomi: the native Japanese words
ON = ['サン', 'ニチ', 'スイ', 'ジン']         # on'yomi: Chinese-derived (in katakana, as dictionaries write them)
KUN_Y, ON_Y = 928, 1000
INDIGO = np.array([40, 62, 118], np.float32) / 255
IVORY = np.array([0.98, 0.95, 0.88], np.float32)
BIG, SMALL = 820, 210        # glyph size when it slams in, and in its strip
SLAM = (540, 900)
T_SLAM = [0.0, 0.42, 0.84, 1.26]
HOLD, FLY = 0.26, 0.24       # how long each kanji fills the screen, and its flight up to its strip


class Scene(SceneBase):
    paper_seed = 91
    paper_stains = 0.12

    def build(self):
        self.big = [typeset(c, BIG, font=FONT_BRUSH_JP, seed=10 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate(KANJI)]
        self.A = cached('s01_strips_v5', _paint_strips)
        self.splat = [cached(f's01_splat_v1_{k}', lambda k=k: {'d': _splat(k)})['d'] for k in range(4)]
        self.lbl = [text_layer(s, FONT_EN_BOLD, 30, tracking=5)[0] for s in LABELS]
        self.kun = [text_layer(r, FONT_JP, 58)[0] for r in KUN]
        self.on = [text_layer(r, FONT_JP, 58)[0] for r in ON]
        sq = kit.blur(np.pad(np.ones((46, 46), np.float32), 4), 1.2)
        self.tags = [(sq, text_layer(c, FONT_JP, 32)[0], col) for c, col in (('訓', INDIGO), ('音', VERM))]
        self.l1, _ = text_layer('ONE CHARACTER.', FONT_HEAD, 66)
        self.l2, _ = text_layer('MULTIPLE READINGS.', FONT_HEAD, 66)
        self.why, _ = text_layer('WHY?', FONT_BRUSH_EN, 190)
        self.why = self.why * (0.85 + 0.15 * noise(*self.why.shape, 6, 2, 4))
        self.tmap = [wash_map(PH, PW, (PW / 2, 0 if k != 1 else PH * 0.35), noise_amp=0.3, noise_scale=90, seed=40 + k) for k in range(4)]
        self.t_l1 = max(at('character')[0], 1.75)
        self.t_l2 = at('multiple readings')[0]
        self.t_read = at('multiple readings')[0] - 0.05
        self.t_why = at('But why?')[0] + 0.12
        self.t_sweep = at('adapted')[0]
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, spline([(1060, 1215), (760, 1190), (420, 1225), (60, 1195)], 14), 34, ink=0.85, dry=0.7,
                      profile=((0, 0.3), (0.08, 1), (0.85, 0.8), (1, 0.15)), seed=5), self.t_sweep, 0.7)
        ss.add(stroke(W, H, [(340, 1388), (540, 1380), (750, 1386)], 12, ink=0.95, dry=0.4,
                      profile=((0, 0.3), (0.1, 1), (0.9, 0.8), (1, 0.1)), seed=6, red=True), self.t_why + 0.12, 0.22)
        self.strokes = ss

    def landed(self, k):
        return T_SLAM[k] + HOLD + FLY

    def cam(self, t):
        jolt = sum(0.035 * np.exp(-max(t - T, 0) * 11) * (t >= T) for T in T_SLAM[1:])
        punch = 0.03 * np.exp(-max(t - self.t_why, 0) * 6) * (t > self.t_why)
        drift = 0.02 * smooth((t - 2.0) / 7)
        return Cam(540, 960 - 30 * smooth((t - self.t_sweep) / 2), 1.0 + jolt + punch + drift)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        # the cherry branch in the corner, once the page has settled
        self.layer(fr, cam, A['branch'], opacity=0.85 * smooth((t - 1.6) / 0.6))
        self.layer(fr, cam, A['bloss'], color=SAKURA, opacity=0.9 * smooth((t - 1.8) / 0.6))
        # the strips: a hairline rule between them, then each illustration as its kanji lands
        self.layer(fr, cam, A['rules'], opacity=0.55 * smooth((t - self.landed(0) + 0.2) / 0.4))
        for k in range(4):
            f = (t - self.landed(k) + 0.05) / 0.75
            if f <= 0:
                continue
            ink = reveal_time(A['ink'][k], self.tmap[k], f, soft=0.06, wet=0.3)
            if k == 2:                                     # the river flows
                yy = np.arange(PH, dtype=np.float32)[:, None]
                ink = ink + reveal_time(A['flow'], self.tmap[k], f, soft=0.06) * (0.55 + 0.45 * np.sin(yy / 15 - t * 7 + np.arange(PW)[None] / 40))
            if k == 2:
                kit.put(fr, cam, reveal_time(A['sea'], self.tmap[k], f), COLS[k], STRIP_CY, SEA, 0.5, S)
            kit.put(fr, cam, ink, COLS[k], STRIP_CY, INK, 1.0, S)
            if k == 1:
                kit.put(fr, cam, A['sun'] * smooth(f * 1.3), COLS[k], STRIP_CY, VERM, 0.95, S)
            kit.put(fr, cam, self.lbl[k], COLS[k], BOTTOM + 28, GREY, smooth((t - self.landed(k) - 0.5) / 0.3))
        # the readings, two rows: kun'yomi (indigo) and on'yomi (vermilion), as Raj says "multiple readings"
        for row, (gl, y, t0) in enumerate(((self.kun, KUN_Y, self.t_read), (self.on, ON_Y, self.t_read + 0.4))):
            sq, ch, col = self.tags[row]
            ft = smooth((t - t0 + 0.1) / 0.25)
            kit.put(fr, cam, sq, 34, y, col, 0.92 * ft)
            kit.put(fr, cam, ch, 34, y, IVORY, ft, over=True)
            for k, g in enumerate(gl):
                fk = (t - t0 - 0.07 * k) / 0.22
                if fk > 0:
                    kit.put(fr, cam, g, COLS[k] + 14, y, col, smooth(fk), 1 + 0.25 * np.exp(-fk * 2.5))
        # the splash round each kanji as it slams in
        for k, T in enumerate(T_SLAM):
            o = 1 - smooth((t - T - 0.15) / 0.5)
            if t >= T - 0.05 and o > 0:
                self.layer(fr, cam, self.splat[k], color=VERM if k == 1 else INK, opacity=0.9 * o)
        # the four kanji: full screen, then up into their strips
        for k, g in enumerate(self.big):
            T = T_SLAM[k]
            if t < T - 0.05:
                continue
            u = ease_io((t - T - HOLD) / FLY)
            x, y = lerp(SLAM[0], COLS[k], u), lerp(SLAM[1], KY, u)
            sc = lerp(1.0 + 0.1 * np.exp(-max(t - T, 0) * 12), SMALL / BIG, u)
            d = g.at((t - T + 0.17) / 0.17, soft=0.05)          # whole at the moment of impact
            kit.put(fr, cam, d, x, y, INK, 1.0, sc)
        # ONE CHARACTER. / MULTIPLE READINGS. / WHY?
        out = 1 - smooth((t - self.t_sweep + 0.4) / 0.5)
        kit.put(fr, cam, self.l1, 540, 1085, INK, smooth((t - self.t_l1) / 0.25) * out)
        kit.put(fr, cam, self.l2, 540, 1155, INK, smooth((t - self.t_l2) / 0.25) * out)
        fw = (t - self.t_why) / 0.16
        if fw > 0:
            kit.put(fr, cam, self.why, 540, 1285, VERM, smooth(fw) * out, 1 + 0.35 * np.exp(-fw * 2.2), rot=-0.04)
        self.layer(fr, cam, self.strokes.render(t, red=True), color=VERM, opacity=out)
        self.layer(fr, cam, self.strokes.render(t), opacity=0.9)
        return fr.img


def _paint_strips():
    """the four little paintings (each a PW x PH density) plus the rules between the strips and the corner branch"""
    r = np.random.default_rng(7)
    ink, sun, sea, flow = [], None, None, None
    # 山: a tall peak with a wet dark summit and cun texture, mist at its foot, a darker near ridge with a pine
    p = Painting(PW, PH, seed=1)
    r0 = p.ridge(PH - 300, peaks=[(30, 90, 60), (225, 120, 55)], rough=6, seed=6)
    p.wash(r0, dens=0.18, decay=60, tex=40, edge=0.3, seed=7)
    r1 = p.peak(128, 96, height=330, left_w=62, right_w=80, power=1.25, rough=8, seed=2)
    p.wash(r1, dens=0.62, decay=95, tex=40, edge=0.5, mist_y=PH - 205, mist_h=36, seed=3)
    p.cun(r1, 80, 190, n=36, shadow_x=140, spread=30, max_y=PH - 240, ink=0.2)
    r2 = p.ridge(PH - 70, peaks=[(40, 70, 70), (200, 40, 80)], rough=8, seed=4)
    p.wash(r2, dens=0.85, decay=50, tex=30, edge=0.5, seed=5)
    ink.append(_fade_bottom(p.D))
    # 日: a low horizon, the sun above it and its reflection broken on the water
    ss = StrokeSet(PW, PH)
    p = Painting(PW, PH, seed=11)
    r1 = p.ridge(PH - 190, peaks=[(40, 60, 90), (190, 80, 110)], rough=6, seed=12)
    p.wash(r1, dens=0.3, decay=60, tex=40, edge=0.3, mist_y=PH - 200, mist_h=20, seed=13)
    d = p.D.copy()
    for i in range(7):
        y = PH - 150 + i * 20
        w = 70 - i * 7
        ss.add(stroke(PW, PH, [(PW / 2 - w, y), (PW / 2 + w, y + 2)], 4, ink=0.5, dry=0.6, profile=SWEEP, seed=20 + i), 0, 1)
    ink.append(_fade_bottom(np.maximum(d, ss.render(9))))
    sun = sun_disc(PH, PW, PW / 2, PH - 300, 82, seed=14).astype(np.float32)
    refl = np.zeros((PH, PW), np.float32)
    for i in range(6):
        y = int(PH - 150 + i * 20 + 8)
        w = int(52 - i * 7)
        refl[y - 3:y + 3, PW // 2 - w:PW // 2 + w] = 0.8 * (1 - i / 7)
    sun = np.maximum(sun, blur(refl, 1.5) * (0.6 + 0.4 * noise(PH, PW, 8, 2, 15)))
    # 水: a river winding down out of the hills, rocks on its banks; the current lines are animated in render()
    p = Painting(PW, PH, seed=31)
    r0 = p.ridge(120, peaks=[(60, 60, 50), (190, 80, 60)], rough=6, seed=32)
    p.wash(r0, dens=0.22, decay=50, tex=40, edge=0.3, mist_y=150, mist_h=25, seed=33)
    ys = np.arange(PH, dtype=np.float32)[:, None]
    xs = np.arange(PW, dtype=np.float32)[None]
    centre = PW / 2 + 40 * np.sin(ys / 110 + 0.6)
    half = 20 + 70 * np.clip((ys - 120) / (PH - 120), 0, 1)
    dist = np.abs(xs - centre) - half
    top = np.clip((ys - 130) / 70, 0, 1)
    bank = np.clip(dist / 6, 0, 1) * top * np.exp(-np.clip(dist, 0, None) / 46)
    p.D = np.maximum(p.D * (1 - (dist <= 0) * top), bank * (0.45 + 0.4 * noise(PH, PW, 20, 3, 34)))
    rocks = StrokeSet(PW, PH)
    for j in range(5):
        y = 230 + j * 60 + r.uniform(-12, 12)
        side = 1 if j % 2 else -1
        c = PW / 2 + 40 * np.sin(y / 110 + 0.6) + side * (20 + 70 * (y - 120) / (PH - 120) + r.uniform(-6, 14))
        rw, rh = r.uniform(10, 18), r.uniform(5, 9)
        poly = [(c + rw * np.cos(a) * (1 + 0.2 * np.sin(3 * a + j)), y + rh * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)]
        rocks.add(pwash(PW, PH, poly, ink=0.6, soft=2.2, seed=35 + j, grain=0.5), 0, 1)
    ink.append(_fade_bottom(np.maximum(p.D, rocks.render(9))))
    ss = StrokeSet(PW, PH)
    for i in range(8):
        off = (i - 3.5) / 3.5
        pts = [(PW / 2 + 40 * np.sin(y / 110 + 0.6) + off * (16 + 60 * (y - 120) / (PH - 120)), y) for y in np.linspace(175, PH, 14)]
        ss.add(stroke(PW, PH, spline(pts, 6), 2.2, ink=0.45, dry=0.75, profile=SWEEP, seed=40 + i), 0, 1)
    flow = _fade_bottom(ss.render(9))
    sea = _fade_bottom(np.clip(-dist / 14, 0, 1) * top * (0.45 + 0.55 * noise(PH, PW, 30, 3, 41)))
    # 人: a traveller in a straw hat standing on a ridge, looking out
    ss = StrokeSet(PW, PH)
    p = Painting(PW, PH, seed=51)
    r1 = p.ridge(PH - 60, peaks=[(60, 140, 160), (230, 40, 80)], rough=8, seed=52)
    p.wash(r1, dens=0.75, decay=60, tex=40, edge=0.45, seed=53)
    r2 = p.ridge(PH - 260, peaks=[(200, 90, 140)], rough=8, seed=54)
    p.wash(r2, dens=0.15, decay=60, tex=60, edge=0.25, mist_y=PH - 250, mist_h=30, seed=55)
    fx, fy, k = 104, PH - 150, 1.25
    P = lambda pts: [(fx + x * k, fy + y * k) for x, y in pts]
    ss.add(pwash(PW, PH, P([(-14, -150), (14, -150), (30, -120), (26, -100), (20, -90), (30, -40), (40, 0), (-38, 0), (-28, -40),
                            (-20, -90), (-27, -104), (-30, -122)]), ink=0.92, soft=1.6, seed=56, grain=0.3), 0, 1)   # robe and sleeves
    ss.add(stroke(PW, PH, spline(P([(-4, -86), (-6, -40), (-10, -2)]), 6), 1.6, ink=0.35, dry=0.6, profile=SWEEP, seed=60), 0, 1)
    ss.add(stroke(PW, PH, P([(-34, -2), (36, -2)]), 3, ink=0.9, dry=0.5, profile=SWEEP, seed=61), 0, 1)
    ss.add(pwash(PW, PH, [(fx + 10 * k * np.cos(a), fy - 162 * k + 11 * k * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)],
                 ink=0.95, soft=1, seed=57), 0, 1)                                                    # head
    ss.add(pwash(PW, PH, P([(-40, -164), (0, -196), (40, -164), (0, -170)]), ink=0.97, soft=1.2, seed=58), 0, 1)   # straw hat
    ss.add(stroke(PW, PH, P([(26, -128), (40, 6)]), 4, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=59), 0, 1)     # staff
    ink.append(_fade_bottom(np.maximum(p.D, ss.render(9))))
    # the rules between the strips
    rs = StrokeSet(W, H)
    for k, x in enumerate((272, 540, 808)):
        rs.add(stroke(W, H, [(x + 2, TOP - 170), (x - 1, TOP + 240), (x + 1, TOP + 470)], 3, ink=0.6, dry=0.6,
                      profile=((0, 0.2), (0.1, 1), (0.9, 1), (1, 0.2)), seed=60 + k), 0, 1)
    # a cherry branch reaching in from the top-left corner
    bs = StrokeSet(W, H)
    tips = cherry_branch(bs, [(-20, 120), (90, 95), (190, 60), (260, 30)], 12, 0, 0.01, seed=70)
    bl = np.zeros((H, W), np.float32)
    for k, (bx, by) in enumerate(tips + [(140, 70), (215, 40)]):
        bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 6, 26, seed=80 + k))
    return {'ink': [np.clip(a, 0, 1).astype(np.float32) for a in ink], 'sun': sun.astype(np.float32), 'sea': sea.astype(np.float32),
            'flow': flow.astype(np.float32), 'rules': rs.render(9).astype(np.float32), 'branch': bs.render(9).astype(np.float32),
            'bloss': bl.astype(np.float32)}


def _fade_bottom(d):
    d = np.asarray(d, np.float32).copy()
    n = 60
    d[-n:] *= np.linspace(1, 0, n)[:, None] ** 1.5
    return d


def _splat(k):
    """ink flung off the brush as a kanji slams in: droplets and a few flicks round the centre"""
    import cv2
    r = np.random.default_rng(100 + k)
    d = np.zeros((H, W), np.float32)
    cx, cy = SLAM
    for _ in range(170):
        a = r.uniform(0, 2 * np.pi)
        rr = r.uniform(330, 560)
        x, y = cx + rr * np.cos(a) * 0.95, cy + rr * np.sin(a) * 1.1
        cv2.circle(d, (int(x), int(y)), int(r.uniform(2, 13)), float(r.uniform(0.6, 1)), -1, cv2.LINE_AA)
    for _ in range(12):
        a = r.uniform(0, 2 * np.pi)
        x0, y0 = cx + 360 * np.cos(a), cy + 400 * np.sin(a)
        x1, y1 = x0 + 150 * np.cos(a), y0 + 150 * np.sin(a)
        cv2.line(d, (int(x0), int(y0)), (int(x1), int(y1)), 0.85, int(r.uniform(3, 8)), cv2.LINE_AA)
    return blur(d, 1.0)

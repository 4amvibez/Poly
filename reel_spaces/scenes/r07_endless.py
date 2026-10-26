"""Scene 7 — "At first, Japanese can look like one endless line.  But once you understand the patterns, the sentence
starts to make sense."
A single unbroken line of Japanese flows across the page like a river, past a reader sitting on the bank: and it is
made of everything this reel has taught (私は日本語を勉強します。毎朝コーヒーを飲みます。東京の学校まで電車で行きます。).
On "patterns" the scripts show their colours as it flows (kanji in ink, hiragana rose, katakana blue); the river slows
and stops on 毎朝コーヒーを飲みます。, which parts into its chunks with their meanings: every morning · coffee · drink."""
import numpy as np

from art import figure_back, sun_disc
from backdrop import backdrop, draw_backdrop
from engine import (FONT_EN, FONT_JP_FALLBACK, GREY, H, INK, SEA, VERM, W, Cam, StrokeSet, cached, ease_io, keyframes, lerp,
                    noise, smooth, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import PRESS_DOWN, SWEEP
from scenes.base import SceneBase
from sentence import BLUE, PINK
from timeline import WORDS, at

PASSAGE = '私は日本語を勉強します。毎朝コーヒーを飲みます。東京の学校まで電車で行きます。'
LINE = PASSAGE * 2
TARGET = len(PASSAGE) + PASSAGE.index('毎朝')                    # 毎朝コーヒーを飲みます。 in the second copy
TCHUNK = [(0, 2), (2, 7), (7, 12)]                               # 毎朝 | コーヒーを | 飲みます。
DXC = [-34, 0, 34]
LY = 800
SIZE, STEP = 64, 68
V0 = 390.0


def kind(c):
    o = ord(c)
    if 0x3040 <= o <= 0x309F:
        return 'H'
    if 0x30A0 <= o <= 0x30FF:
        return 'A'
    return 'P' if c in '。、' else 'K'


def after(word, t):
    w = next(w for w in WORDS if w['start'] > t and ''.join(c for c in w['text'].lower() if c.isalnum()) == word)
    return w['start'], w['end']


class Scene(SceneBase):
    paper_seed = 69

    def build(self):
        self.B = backdrop(seed=150, branch='left', pagoda_x=880, ground=1300)

        def paint():
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            y0, y1 = LY - 78, LY + 78
            band = np.clip((yy - y0) / 22, 0, 1) * np.clip((y1 - yy) / 22, 0, 1) * (0.55 + 0.45 * noise(H, W, 90, 4, 3))
            ss = StrokeSet(W, H)
            r = np.random.default_rng(4)
            for k in range(30):
                x, y = r.uniform(-40, 1040), r.choice([r.uniform(y0 + 8, y0 + 26), r.uniform(y1 - 26, y1 - 8)])
                ss.add(stroke(W, H, [(x, y), (x + r.uniform(60, 150), y + r.normal(0, 1))], r.uniform(2, 3.5), ink=0.45, dry=0.6,
                              profile=SWEEP, seed=40 + k), 0, 1)
            fig = StrokeSet(W, H)
            figure_back(fig, 190, 1236, 1.15, 0, seed=12, dur=0.01, pose='sit')
            bank = StrokeSet(W, H)
            bank.add(stroke(W, H, [(-30, 1244), (200, 1236), (420, 1248), (640, 1256)], 9, ink=0.6, dry=0.55, profile=PRESS_DOWN, seed=7), 0, 1)
            for k, x in enumerate((360, 382, 402, 590, 612)):
                bank.add(stroke(W, H, [(x, 1248), (x + 8, 1204 - 10 * (k % 2)), (x + 18, 1186)], 3, ink=0.6, dry=0.4, profile=SWEEP,
                                seed=80 + k), 0, 1)
            for k, (bx, by, sz) in enumerate([(300, 430, 1.0), (352, 396, 0.8), (408, 448, 0.9)]):        # swallows
                bank.add(stroke(W, H, [(bx - 22 * sz, by - 6 * sz), (bx - 8 * sz, by - 2 * sz), (bx, by + 6 * sz)], 3.2 * sz, ink=0.8, dry=0.3,
                                profile=PRESS_DOWN, seed=60 + k), 0, 1)
                bank.add(stroke(W, H, [(bx, by + 6 * sz), (bx + 9 * sz, by - 3 * sz), (bx + 24 * sz, by - 8 * sz)], 3.2 * sz, ink=0.8, dry=0.3,
                                profile=PRESS_DOWN, seed=70 + k), 0, 1)
            return {'river': (band * 0.5).astype(np.float32), 'rip': ss.render(5).astype(np.float32),
                    'fig': fig.render(9).astype(np.float32), 'bank': bank.render(9).astype(np.float32),
                    'sun': sun_disc(H, W, 800, 430, 70, seed=5).astype(np.float32)}
        self.A = cached('r07_paint_v2', paint)
        self.g = [typeset(c, SIZE, font=FONT_JP_FALLBACK, seed=300 + i, pad=0.04, medians=False, sweep=(1, 0.3)) for i, c in enumerate(LINE)]
        self.kind = [kind(c) for c in LINE]
        # the river's motion: steady, then slowing to a stop with the target sentence centred
        self.t_slow = at('patterns')[0] - 0.2
        self.t_stop = after('sentence', self.t_slow)[0] + 0.15
        self.t_pat = at('once you understand')[0]
        self.t_part = after('starts', self.t_slow)[0] - 0.1
        self.t_line = at('endless line')[0]
        tc = TARGET + (12 - 1) / 2
        self.C = 540 - tc * STEP + self.dist(self.t_stop)
        self.gloss = [text_layer(s, FONT_EN, 44)[0] for s in ('every morning', 'coffee', 'drink')]
        under = StrokeSet(W, H)
        for k, (a, b) in enumerate(TCHUNK):
            xa = 540 + (a - 5.5) * STEP + DXC[k] - STEP * 0.42
            xb = 540 + (b - 1 - 5.5) * STEP + DXC[k] + STEP * 0.42
            if k == 2:
                xb -= STEP * 0.7                                            # not under the 。
            under.add(stroke(W, H, [(xa, LY + 50), ((xa + xb) / 2, LY + 46), (xb, LY + 50)], 6, ink=0.95, dry=0.4,
                             profile=((0, 0.5), (0.1, 1), (0.9, 0.9), (1, 0.4)), seed=90 + k, red=True), self.t_part + 0.25 + 0.15 * k, 0.3)
        self.under = under

    def dist(self, t):
        """how far the line has travelled (px) by time t"""
        D = self.t_stop - self.t_slow
        if t < self.t_slow:
            return V0 * (t - self.t0)
        u = min((t - self.t_slow) / D, 1.0)
        return V0 * (self.t_slow - self.t0) + V0 * D * (1 - (1 - u) ** 3) / 3

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.04)), (self.t_stop, (540, 950, 1.0)), (self.t1, (540, 945, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        self.layer(fr, cam, A['sun'], color=VERM, opacity=0.8, depth=0.9)
        self.layer(fr, cam, A['river'], color=SEA, opacity=0.55)
        self.layer(fr, cam, np.roll(A['rip'], -int(self.dist(t) * 0.4) % W, axis=1), color=SEA * 0.6, opacity=0.65)   # drifting
        self.layer(fr, cam, A['bank'], opacity=0.85)
        self.layer(fr, cam, A['fig'], opacity=0.95)
        # --- the line
        pat = smooth((t - self.t_pat) / 0.8)
        part = ease_io((t - self.t_part) / 0.55)
        focus = smooth((t - self.t_stop + 0.4) / 0.6)
        x0 = self.C - self.dist(t)
        tint = {'K': INK, 'H': PINK * 0.78, 'A': BLUE * 0.75, 'P': GREY}
        for i, g in enumerate(self.g):
            x = x0 + i * STEP
            j = i - TARGET
            inside = 0 <= j < 12
            if inside:
                k = next(k for k, (a, b) in enumerate(TCHUNK) if a <= j < b)
                x += DXC[k] * part
            if x < -STEP or x > W + STEP:
                continue
            col = lerp(INK, tint[self.kind[i]], pat)
            op = 1.0 if inside else 1 - 0.85 * focus
            self.place(fr, cam, g.dens, x, LY, color=col, opacity=op)
        self.layer(fr, cam, self.under.render(t, red=True), color=VERM)
        cx = [540 + ((a + b - 1) / 2 - 5.5) * STEP + DXC[k] for k, (a, b) in enumerate(TCHUNK)]
        cx[2] -= STEP * 0.5
        for k, gl in enumerate(self.gloss):
            self.place(fr, cam, gl, cx[k], LY + 104, color=INK, opacity=smooth((t - self.t_part - 0.35 - 0.15 * k) / 0.35))
        return fr.img

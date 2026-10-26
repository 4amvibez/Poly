"""Scene 2 — "In English, spaces tell us where one word ends, and the next begins.  But Japanese?  Words often run
together, without spaces.  So how do readers know where one word ends, and another begins?"
English: I / love / learning on stepping stones across a stream; the water between them is the space, and a small
figure hops from word to word.  Japanese: 私は日本語を勉強します。 above an arched bridge, one unbroken path; a single
vermilion stroke runs under it on "run together".  Then the figure stops on the bridge, and dashed split lines try
themselves between the characters — where does one word end?"""
import numpy as np

from art import figure
from backdrop import backdrop, draw_backdrop
from engine import (FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, H, SEA, VERM, W, Cam, StrokeSet, blur,
                    cached, ease_io, keyframes, lerp, noise, smooth, spline, stroke, text_layer)
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from sentence import Sentence
from timeline import at, quiet_gap

STONES = [(150, 640, 'I', 52), (410, 646, 'love', 106), (785, 640, 'learning', 168)]    # x, y, word, half-width
GAPS = [(259, 'space'), (566, 'space')]
SY = 945                                   # the Japanese sentence's line
IVORY = np.array([0.97, 0.94, 0.87], np.float32)


def tag(text):
    """a dark brush banner with the label in ivory: (ink density, text alpha)"""
    a, _ = text_layer(text, FONT_EN_BOLD, 50, tracking=1)
    h, w = a.shape
    bw, bh = w + 90, h + 34
    ss = StrokeSet(bw, bh)
    ss.add(stroke(bw, bh, [(10, bh / 2 + 3), (bw / 2, bh / 2 - 2), (bw - 10, bh / 2 + 2)], bh * 0.95, ink=0.95, dry=0.5,
                  profile=((0, 0.6), (0.08, 1), (0.9, 0.95), (1, 0.4)), seed=len(text)), 0, 1)
    txt = np.zeros((bh, bw), np.float32)
    txt[(bh - h) // 2:(bh - h) // 2 + h, (bw - w) // 2:(bw - w) // 2 + w] = a
    return ss.render(5), txt


class Scene(SceneBase):
    paper_seed = 64

    def build(self):
        def paint():
            out = {}
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            # the stream under the stepping stones, and the river under the bridge
            for key, (y0, y1) in (('stream', (585, 705)), ('river', (1160, 1330))):
                band = np.clip((yy - y0) / 18, 0, 1) * np.clip((y1 - yy) / 18, 0, 1)
                band *= 0.55 + 0.45 * noise(H, W, 90, 4, len(key))
                ss = StrokeSet(W, H)
                r = np.random.default_rng(len(key))
                for k in range(26):
                    x, y = r.uniform(40, 1000), r.uniform(y0 + 15, y1 - 10)
                    ss.add(stroke(W, H, [(x, y), (x + r.uniform(50, 130), y + r.normal(0, 1))], r.uniform(2, 3.5), ink=0.45,
                                  dry=0.6, profile=SWEEP, seed=40 + k), 0, 1)
                out[key] = (band * 0.55).astype(np.float32)
                out[key + '_rip'] = ss.render(5).astype(np.float32)
            # stepping stones, each its own layer so it can be set down on its word
            for k, (x, y, word, rw) in enumerate(STONES):
                ss = StrokeSet(W, H)
                pts = [(x + rw * np.cos(a) * (1 + 0.06 * np.sin(3 * a + k)), y + 44 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 18, endpoint=False)]
                ss.add(pwash(W, H, pts, ink=0.2, soft=3, seed=10 + k, grain=0.5), 0, 1)
                ss.add(stroke(W, H, [(x - rw * 0.95, y + 20), (x, y + 40), (x + rw * 0.95, y + 20)], 8, ink=0.75, dry=0.5,
                              profile=PRESS_DOWN, seed=20 + k), 0, 1)
                ss.add(stroke(W, H, [(x - rw * 0.8, y - 24), (x - rw * 0.2, y - 36)], 4, ink=0.35, dry=0.6, profile=PRESS_DOWN, seed=30 + k), 0, 1)
                out[f'stone{k}'] = ss.render(9).astype(np.float32)
            # the arched bridge: deck, rail, posts, the arch beneath, and its reflection
            ss = StrokeSet(W, H)
            deck = spline([(60, 1178), (300, 1110), (540, 1088), (780, 1110), (1020, 1178)], 20)
            rail = deck - np.array([0, 58], np.float32)
            ss.add(stroke(W, H, [tuple(p) for p in deck], 16, ink=0.85, dry=0.4, profile=BLUNT, seed=1), 0, 1)
            ss.add(stroke(W, H, [tuple(p) for p in rail], 7, ink=0.8, dry=0.4, profile=BLUNT, seed=2), 0, 1)
            for k in range(0, len(deck), 6):
                (x, y), (_, yr) = deck[k], rail[k]
                ss.add(stroke(W, H, [(x, yr - 6), (x, y)], 6, ink=0.8, dry=0.3, profile=BLUNT, seed=3 + k), 0, 1)
            arch = spline([(150, 1215), (340, 1140), (540, 1118), (740, 1140), (930, 1215)], 20)
            ss.add(stroke(W, H, [tuple(p) for p in arch], 9, ink=0.7, dry=0.5, profile=BLUNT, seed=5), 0, 1)
            out['bridge'] = ss.render(9).astype(np.float32)
            refl = np.flipud(out['bridge'][900:1200])                  # mirrored into the river, broken by ripples
            R = np.zeros((H, W), np.float32)
            R[1185:1185 + 300] = refl[:300] * 0.25
            R *= 0.6 + 0.4 * noise(H, W, 6, 2, 9)
            R[1330:] = 0
            out['refl'] = blur(R, 1.5).astype(np.float32)
            return out
        self.A = cached('r02_paint_v3', paint)
        self.B = backdrop(seed=30, branch='left', ground=1300)
        self.t_en = at('In English')[0]
        self.t_jp = at('But Japanese')[0]
        self.t_run = at('run together')[0]
        self.t_how = at('So how')[0]
        c, n = quiet_gap(at('another begins')[1] - 0.2, at('The first clue')[0])
        self.t_hit = c - n / 2 + 0.1                                     # sync: the dot lands (+ bowl)
        self.t_q = self.t_hit - 0.36
        self.tag_en, self.tag_jp = tag('English'), tag('Japanese')
        self.words = [(text_layer(w, FONT_EN_BOLD, 80)[0], x, y) for x, y, w, _ in STONES]
        self.space = text_layer('space', FONT_EN_ITALIC, 42)[0]
        self.t_stone = [self.t_en + 0.25 + 0.3 * k for k in range(3)]
        e1 = at('where one word ends')
        self.hops = [(e1[0], e1[1] - 0.2), (at('and the next')[0], at('and the next')[1] + 0.3)]
        fs = StrokeSet(300, 300)
        figure(fs, 150, 280, 1.1, 0, seed=5, pose='stand', facing=1, dur=0.01)
        self.fig = fs.render(9)
        self.sent = Sentence(size=72)
        line = StrokeSet(W, H)
        a, b = self.sent.span(range(12))
        line.add(stroke(W, H, [(540 + a, SY + 54), (540, SY + 48), (540 + b, SY + 54)], 9, ink=0.95, dry=0.4,
                        profile=((0, 0.5), (0.05, 1), (0.95, 0.9), (1, 0.3)), seed=8, red=True), self.t_run, 0.8)
        self.line = line
        # candidate splits that "try themselves" between characters
        self.splits = [(i, self.t_how + 0.25 + 0.32 * k) for k, i in enumerate([3, 7, 1, 9, 5, 2, 10, 6])]
        self.dash = self._dash()
        q = StrokeSet(W, H)
        P = lambda pts: [(540 + (x - 548) * 0.62, 470 + (y - 880) * 0.62) for x, y in pts]
        q.add(stroke(W, H, spline(P([(392, 720), (420, 612), (535, 548), (668, 592), (700, 706), (628, 800), (560, 868), (544, 965),
                                       (548, 1048)]), 14), 46, ink=0.95, dry=0.42, profile=((0, 0.55), (0.08, 1.0), (0.7, 0.9), (1, 0.45)),
                     seed=7, red=True), self.t_q, 0.35)
        cx, cy = 540, 470 + (1166 - 880) * 0.62
        q.add(pwash(W, H, [(cx + 30 * np.cos(a), cy + 31 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 22, endpoint=False)],
                    ink=0.96, soft=1.5, seed=8, grain=0.18), self.t_q + 0.36, 0.08)
        q.items[-1][0].red = True
        self.q = q

    def _dash(self):
        d = np.zeros((170, 30), np.float32)
        for y in range(40, 170, 24):
            d[y:y + 14, 11:19] = 1
        q, _ = text_layer('？', FONT_JP, 34)
        qh, qw = q.shape
        d[0:min(qh, 40), max(0, 15 - qw // 2):max(0, 15 - qw // 2) + qw] = np.maximum(d[0:min(qh, 40), max(0, 15 - qw // 2):max(0, 15 - qw // 2) + qw], q[:40, :30 - max(0, 15 - qw // 2)])
        return blur(d, 0.8)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 900, 1.0)), (self.t_jp - 0.4, (540, 880, 1.02)), (self.t_how, (540, 940, 1.02)),
                                  (self.t1, (540, 960, 1.06))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        draw_backdrop(self, fr, cam, self.B, 0.75 * smooth((t - self.t0) / 0.8))
        # --- English
        oe_out = 1 - smooth((t - self.t_q + 1.0) / 0.6)            # English steps aside for the closing question
        oe = smooth((t - self.t_en + 0.2) / 0.5) * oe_out
        if oe > 0:
            self.layer(fr, cam, A['stream'], color=SEA, opacity=0.5 * oe)
            self.layer(fr, cam, A['stream_rip'], color=SEA * 0.6, opacity=0.6 * oe)
            ink, txt = self.tag_en
            self.place(fr, cam, ink, 250, 440, opacity=oe)
            self._ivory(fr, cam, txt, 250, 440, oe)
            for k, (a_, x, y) in enumerate(self.words):
                f = smooth((t - self.t_stone[k]) / 0.35)
                if f <= 0:
                    continue
                self.layer(fr, cam, A[f'stone{k}'], opacity=f * oe_out)
                self.place(fr, cam, a_, x, y - 4 + 12 * (1 - f), opacity=f * oe_out)
            for k, (gx, lbl) in enumerate(GAPS):
                f = smooth((t - self.t_stone[2] - 0.4 - 0.25 * k) / 0.35) * oe_out
                if f > 0:
                    self.place(fr, cam, self.space, gx, 760, color=VERM, opacity=f)
                    ar = np.zeros((70, 20), np.float32)
                    ar[0:54, 9:12] = 1
                    for j in range(10):
                        ar[54 + j, 9 - j:12 + j] = 1 if j < 8 else 0
                    self.place(fr, cam, blur(ar, 0.7), gx, 708, color=VERM, opacity=f)
            # the hopping reader
            fo = smooth((t - self.t_stone[0] - 0.2) / 0.4) * (1 - smooth((t - self.t_jp - 0.6) / 0.5))
            if fo > 0:
                x, y = STONES[0][0], STONES[0][1] - 40
                for k, (h0, h1) in enumerate(self.hops):
                    u = np.clip((t - h0) / (h1 - h0), 0, 1)
                    xa, xb = STONES[k][0], STONES[k + 1][0]
                    if u > 0:
                        x = lerp(xa, xb, ease_io(u))
                        y = STONES[0][1] - 40 - 110 * np.sin(np.pi * u)
                self.place(fr, cam, self.fig, x, y - 150, opacity=fo * oe_out, scale=1.15)
        # --- Japanese
        oj = smooth((t - self.t_jp + 0.2) / 0.5)
        if oj > 0:
            self.layer(fr, cam, A['river'], color=SEA, opacity=0.5 * oj)
            self.layer(fr, cam, A['river_rip'], color=SEA * 0.6, opacity=0.6 * oj)
            self.layer(fr, cam, A['refl'], opacity=oj)
            self.layer(fr, cam, A['bridge'], opacity=oj)
            ink, txt = self.tag_jp
            self.place(fr, cam, ink, 270, 850, opacity=oj)
            self._ivory(fr, cam, txt, 270, 850, oj)
            rv = lambda i: smooth((t - self.t_jp - 0.3 - 0.06 * i) / 0.3)
            self.sent.draw(self, fr, cam, 540, SY, rv)
            self.layer(fr, cam, self.line.render(t, red=True), color=VERM)
            # the reader crosses to the middle of the bridge and stops
            fo = smooth((t - self.t_run) / 0.4)
            if fo > 0:
                u = ease_io((t - self.t_run) / 2.4)
                x = lerp(120, 520, u)
                y = 1180 - (1180 - 1088) * np.sin(np.pi * (x - 60) / 960) - 4
                self.place(fr, cam, self.fig, x, y - 125, opacity=fo, scale=0.9)
            for i, ts in self.splits:
                f = (t - ts) / 0.3
                if 0 < f < 3.5:
                    op = smooth(f) * (1 - smooth((t - ts - 0.6) / 0.35)) if ts < self.splits[-1][1] else smooth(f) * (1 - smooth((t - self.t_q) / 0.4))
                    x = 540 + (self.sent.x[i] + self.sent.x[i + 1]) / 2
                    self.place(fr, cam, self.dash, x, SY - 18, color=VERM, opacity=0.95 * op)
            self.layer(fr, cam, self.q.render(t, red=True), color=VERM)
        return fr.img

    def _ivory(self, fr, cam, a, cx, cy, op):
        import cv2
        h, w = a.shape
        M = cam.matrix()
        A = np.array([[1, 0, cx - w / 2], [0, 1, cy - h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(IVORY, cv2.warpAffine(a, Mt, (W, H)), op)

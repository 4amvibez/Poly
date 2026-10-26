"""Scene 8 — the end card.  "Follow me to learn Japanese, one step at a time."  最後まで見てくれてありがとう。またね！
FOLLOW · @raj_in_jp stamps in on a vermilion brush banner.  Below, a callback to the stepping stones of the opening:
four stones across a stream carry 一 歩 ず つ (ippo zutsu, one step at a time), and the little reader hops across them
one per word, each stone's character turning vermilion as it lands.  Then the Japanese sign-off is brushed in as Raj
says it, the reader turns and waves on またね, and the page holds."""
import numpy as np

from art import figure, sun_disc
from backdrop import backdrop, draw_backdrop
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, GREY, H, INK, SEA, VERM, W, Cam, StrokeSet, blur,
                    cached, ease_io, keyframes, lerp, noise, smooth, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from timeline import JA, at, quiet_gap

IVORY = np.array([0.97, 0.94, 0.86], np.float32)
SX = [205, 420, 640, 860]
SYS = 760
STONE_CH = '一歩ずつ'
LINES = [('最後まで見てくれて', 1010, 76), ('ありがとう。', 1120, 76), ('またね！', 1262, 104)]


class Scene(SceneBase):
    paper_seed = 70

    def build(self):
        self.B = backdrop(seed=170, branch='left', pagoda_x=900, ground=1600)
        self.t_follow = at('Follow me')[0]
        s0 = at('one step')[0]
        c, n = quiet_gap(at('a time')[0], JA['start'])
        self.t_hit = c - n / 2 + 0.35                                     # sync: the last leap lands on つ (+ bowl)
        self.t_hop = [s0 - 0.1, s0 + 0.2, s0 + 0.52, self.t_hit - 0.45]
        self.hop_d = [0.3, 0.3, 0.3, 0.45]

        def paint():
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            y0, y1 = SYS - 64, SYS + 64
            band = np.clip((yy - y0) / 18, 0, 1) * np.clip((y1 - yy) / 18, 0, 1) * (0.55 + 0.45 * noise(H, W, 90, 4, 21))
            ss = StrokeSet(W, H)
            r = np.random.default_rng(22)
            for k in range(22):
                x, y = r.uniform(-20, 1040), r.uniform(y0 + 10, y1 - 10)
                ss.add(stroke(W, H, [(x, y), (x + r.uniform(50, 120), y + r.normal(0, 1))], r.uniform(2, 3.2), ink=0.45, dry=0.6,
                              profile=SWEEP, seed=60 + k), 0, 1)
            out = {'stream': (band * 0.5).astype(np.float32), 'rip': ss.render(5).astype(np.float32),
                   'sun': sun_disc(H, W, 870, 585, 62, seed=7).astype(np.float32)}
            for k, x in enumerate(SX):
                st = StrokeSet(W, H)
                rw = 84
                pts = [(x + rw * np.cos(a) * (1 + 0.06 * np.sin(3 * a + k)), SYS + 40 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 18, endpoint=False)]
                st.add(pwash(W, H, pts, ink=0.16, soft=3, seed=30 + k, grain=0.5), 0, 1)
                st.add(stroke(W, H, [(x - rw * 0.95, SYS + 18), (x, SYS + 36), (x + rw * 0.95, SYS + 18)], 8, ink=0.75, dry=0.5,
                              profile=PRESS_DOWN, seed=40 + k), 0, 1)
                out[f'stone{k}'] = st.render(9).astype(np.float32)
            banner = StrokeSet(W, H)
            banner.add(stroke(W, H, [(250, 392), (540, 380), (835, 388)], 118, ink=0.96, dry=0.42,
                              profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=9, red=True), 0, 1)
            out['banner'] = banner.render(9).astype(np.float32)
            return out
        self.A = cached('r08_paint_v2', paint)
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 44, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 82)
        self.stone_g = [typeset(c, 78, font=FONT_BRUSH_JP, seed=80 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate(STONE_CH)]
        self.ippo, _ = text_layer('ippo zutsu · one step at a time', FONT_EN_ITALIC, 40)
        fs, fw = StrokeSet(300, 300), StrokeSet(300, 300)
        figure(fs, 150, 280, 1.1, 0, seed=5, pose='stand', facing=1, dur=0.01)
        figure(fw, 150, 280, 1.1, 0, seed=5, pose='speak', facing=-1, dur=0.01)
        self.fig, self.wave = fs.render(9), fw.render(9)
        # the sign-off, line by line as it is spoken
        a, e = JA['start'], JA['arigatou_end']
        split = a + (e - a) * 11 / 16                                    # saigo made mite kurete | arigatou
        spans = [(a - 0.05, split), (split, e), (JA['matane'] - 0.05, JA['end'])]
        self.lines = []
        for k, ((txt, y, sz), (t0, t1)) in enumerate(zip(LINES, spans)):
            gl = [typeset(c, sz, font=FONT_BRUSH_JP, seed=120 + 10 * k + j) for j, c in enumerate(txt)]
            step = sz * 1.0
            xs = [540 + (j - (len(txt) - 1) / 2) * step for j in range(len(txt))]
            per = (t1 - t0) / len(txt)
            self.lines.append([(g, x, y, t0 + j * per * 0.9, max(per * 1.6, 0.18)) for j, (g, x) in enumerate(zip(gl, xs))])
        self.tr1, _ = text_layer('Thanks for watching till the end.', FONT_EN_ITALIC, 42)
        self.tr2, _ = text_layer('See you next time!', FONT_EN_ITALIC, 42)
        self.t_tr1, self.t_tr2 = split + 0.3, JA['matane'] + 0.35

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.03)), (self.t_hop[-1] + 0.4, (540, 950, 1.0)), (self.t1, (540, 950, 1.02))],
                              ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        self.layer(fr, cam, A['sun'], color=VERM, opacity=0.8, depth=0.9)
        # --- FOLLOW @raj_in_jp
        f = (t - self.t_follow + 0.1) / 0.3
        if f > 0:
            punch = 1 + 0.12 * np.exp(-f * 2.5)
            m = np.zeros_like(A['banner'])
            m[:, :int(W * min(1, f * 1.2))] = 1
            self.layer(fr, cam, A['banner'] * blur(m, 10), color=VERM, opacity=0.95)
            self._over(fr, cam, self.handle, 540, 380, IVORY, smooth((t - self.t_follow - 0.1) / 0.3), punch)
            self.place(fr, cam, self.follow, 540, 268, color=VERM, opacity=smooth((t - self.t_follow) / 0.35))
        # --- 一歩ずつ: stepping stones across a stream, and the reader hopping them
        fs = smooth((t - self.t0 - 0.2) / 0.6)
        self.layer(fr, cam, A['stream'], color=SEA, opacity=0.5 * fs)
        self.layer(fr, cam, np.roll(A['rip'], int((t - self.t0) * 12) % W, axis=1), color=SEA * 0.6, opacity=0.6 * fs)
        for k, (x, g) in enumerate(zip(SX, self.stone_g)):
            fk = smooth((t - self.t0 - 0.35 - 0.15 * k) / 0.4)
            if fk <= 0:
                continue
            self.layer(fr, cam, A[f'stone{k}'], opacity=fk)
            land = self.t_hop[k] + self.hop_d[k]
            lit = smooth((t - land + 0.02) / 0.15)
            col = lerp(INK, VERM, lit)
            self.place(fr, cam, g.at((t - self.t0 - 0.5 - 0.15 * k) / 0.4, soft=0.06), x, SYS - 6, color=col,
                       scale=1 + 0.14 * np.exp(-max(t - land, 0) * 5) * (t > land))
        self.place(fr, cam, self.ippo, 540, SYS + 108, color=GREY, opacity=smooth((t - self.t_hop[0]) / 0.4))
        x, y = 60, SYS - 18
        for k, (h0, hd) in enumerate(zip(self.t_hop, self.hop_d)):
            u = np.clip((t - h0) / hd, 0, 1)
            if u > 0:
                xa = 60 if k == 0 else SX[k - 1]
                x = lerp(xa, SX[k], ease_io(u))
                y = SYS - 18 - (80 + 50 * (k == 3)) * np.sin(np.pi * u)
        fo = smooth((t - self.t0 - 0.3) / 0.5)
        sprite = self.wave if t > JA['matane'] else self.fig
        self.place(fr, cam, sprite, x, y - 137, opacity=fo, scale=1.02)
        # --- the sign-off
        for line in self.lines:
            for g, gx, gy, ts, d in line:
                fk = (t - ts) / d
                if fk > 0:
                    col = VERM if gy == LINES[2][1] else INK
                    self.place(fr, cam, g.at(min(fk, 1.05), soft=0.05, wet=0.35), gx, gy, color=col)
        self.place(fr, cam, self.tr1, 540, 1372, color=GREY, opacity=smooth((t - self.t_tr1) / 0.4))
        self.place(fr, cam, self.tr2, 540, 1426, color=GREY, opacity=smooth((t - self.t_tr2) / 0.4))
        return fr.img

    def _over(self, fr, cam, a, x, y, color, op, scale=1.0):
        import cv2
        if op <= 0:
            return
        h, w = a.shape
        M = cam.matrix()
        A = np.array([[scale, 0, x - scale * w / 2], [0, scale, y - scale * h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(color, cv2.warpAffine(a, Mt, (W, H)), op)

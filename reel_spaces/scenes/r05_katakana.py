"""Scene 5 — the third clue, katakana.  "And katakana?  It often helps readers recognize foreign-origin words, names,
and certain emphasized expressions.  For example, 'kohee' means coffee."
CLUE 3 · カタカナ; three cards deal in on their words: パソコン (a borrowed word, with a laptop), ラージ (a name: Raj,
waving hello) and スゴイ！ (emphasis, with manga focus lines bursting round it).  Then the cards lift away and a big
cup of coffee steams in the middle of the page under コーヒー brushed large: kōhī means coffee."""
import cv2
import numpy as np

import vignettes
from backdrop import backdrop, draw_backdrop
from clues import Clue
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, blur, cached,
                    ease_io, ease_out, keyframes, lerp, smooth, text_layer)
from glyphs import text_glyphs, typeset
from scenes.base import SceneBase
from timeline import at, quiet_gap

CARDS = [('パソコン', 'pasokon', 'computer', 'laptop'), ('ラージ', 'rāji', 'Raj (a name)', 'portrait'),
         ('スゴイ！', 'sugoi!', 'amazing!', 'shout')]
CX = [190, 540, 890]
CY = 800
BH, BW = 450, 300
CUP = (540, 1075)


class Scene(SceneBase):
    paper_seed = 67

    def build(self):
        self.clue = Clue(3, 'カタカナ', 'katakana', at('And katakana')[0] + 0.2, at('It often')[0] - 0.1)
        self.B = backdrop(seed=110, branch='left', pagoda_x=930, ground=1360)
        self.t_card = [at('foreign-origin')[0] - 0.1, at('names')[0] - 0.1, at('emphasized')[0] - 0.05]
        self.t_cof = at('For example')[0]
        self.t_kohee = at('kohee')[0]
        self.t_hit = quiet_gap(self.t_kohee - 0.45, self.t_kohee + 0.1)[0]   # sync: the cup lands (+ clink)
        self.t_coffee = at('means coffee')[1] - 0.45

        def pics():
            return {k: getattr(vignettes, k)().astype(np.float32) for k in ('laptop', 'portrait', 'shout')} | \
                   {'cup': vignettes.coffee().astype(np.float32), 'burst': vignettes.burst(150, 250, 44, seed=3).astype(np.float32)}
        self.P = cached('r05_pics_v1', pics)
        self.card = []
        for k, (kana, ro, en, pic) in enumerate(CARDS):
            D, _, _ = text_glyphs(kana, 54, font=FONT_JP, seed=20 + k, tracking=0.02)
            self.card.append(dict(kana=D, ro=text_layer(ro, FONT_EN_ITALIC, 36)[0], en=text_layer(en, FONT_EN, 40)[0], pic=self.P[pic]))
        box = np.zeros((BH, BW), np.float32)
        box[4:-4, 4:-4] = 1
        self.box = blur(box, 2.5)
        rim, inner = np.zeros_like(box), np.zeros_like(box)
        rim[10:-10, 10:-10] = 1
        inner[13:-13, 13:-13] = 1
        self.rim = np.clip(blur(rim, 0.7) - blur(inner, 0.7), 0, 1)
        # the backs of the cards: a vermilion diamond with a question mark, dealt face down before each word
        yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
        dia = np.abs(xx - BW / 2) / (BW * 0.36) + np.abs(yy - BH / 2) / (BH * 0.36)
        q, _ = text_layer('？', FONT_JP, 120)
        back = np.clip((1 - dia) * 30, 0, 1) * 0.85
        qh, qw = q.shape
        back[BH // 2 - qh // 2:BH // 2 - qh // 2 + qh, BW // 2 - qw // 2:BW // 2 - qw // 2 + qw] *= 1 - q
        frame_ = np.clip(blur(rim, 0.7) - blur(inner, 0.7), 0, 1)
        inner2, inner3 = np.zeros_like(box), np.zeros_like(box)
        inner2[22:-22, 22:-22] = 1
        inner3[25:-25, 25:-25] = 1
        self.back = np.clip(blur(back, 0.8) + 0.9 * np.clip(blur(inner2, 0.7) - blur(inner3, 0.7), 0, 1), 0, 1)
        self.t_deal = [at('It often')[0] + 0.12 + 0.16 * k for k in range(3)]
        # coffee
        self.kohi = [typeset(c, 150, font=FONT_BRUSH_JP, seed=60 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('コーヒー')]
        self.ro, _ = text_layer('kōhī', FONT_EN_ITALIC, 62)
        self.eq, _ = text_layer('=', FONT_EN, 60)
        self.coffee, _ = text_layer('coffee', FONT_BRUSH_EN, 76)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 950, 1.02)), (self.t_card[0], (540, 940, 1.0)), (self.t_cof, (540, 945, 1.0)),
                                  (self.t1, (540, 960, 1.04))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        self.clue.draw(self, fr, cam, t)
        out = ease_io((t - self.t_cof + 0.15) / 0.55)                 # the cards lift away for the coffee
        # emphasis: focus lines burst out round the third card (drawn first, so the cards sit on top of them)
        if t > self.t_card[2] + 0.2 and out < 1:
            jolt = np.exp(-max(t - self.t_card[2] - 0.3, 0) * 5) * (t > self.t_card[2] + 0.3)
            bo = smooth((t - self.t_card[2] - 0.2) / 0.2) * (1 - out)
            self._put(fr, cam, self.P['burst'], CX[2], CY - 90 * out, 1.25 + 0.08 * jolt, 1.25 + 0.08 * jolt, INK, 0.6 * bo)
        for k, C in enumerate(self.card):
            f = (t - self.t_deal[k]) / 0.45
            if f <= 0 or out >= 1:
                continue
            e = ease_out(f)
            op = smooth(f * 1.5) * (1 - out)
            x, y = CX[k], CY + 140 * (1 - e) - 90 * out
            u = np.clip((t - self.t_card[k]) / 0.4, 0, 1)               # the flip: face down -> face up
            sx = max(abs(np.cos(np.pi * u)), 0.02)
            up = u >= 0.5
            sy = 1 + 0.06 * np.sin(np.pi * u)
            P = lambda d, dy=0.0, col=INK, o=1.0, s=1.0: self._put(fr, cam, d, x, y + dy * sy, sx * s, sy * s, col, op * o)
            jolt = np.exp(-max(t - self.t_card[k] - 0.3, 0) * 5) * (t > self.t_card[k] + 0.3) if k == 2 and up else 0.0
            P(blur(self.box, 8), 10, col=INK, o=0.25)
            self._put(fr, cam, self.box, x, y, sx, sy, np.array([0.97, 0.95, 0.89], np.float32), op, over=True)
            P(self.rim, col=np.array([0.55, 0.42, 0.3], np.float32), o=0.6)
            if not up:
                P(self.back, col=VERM, o=0.9)
                continue
            P(C['kana'], -150, s=1 + 0.18 * jolt)
            P(C['ro'], -86, col=GREY)
            P(C['en'], -40)
            P(C['pic'], 100, s=1.2)
        # --- kōhī: a cup of coffee, steaming
        fc = smooth((t - self.t_cof - 0.1) / 0.4)
        if fc > 0:
            cx, cy = CUP
            u = np.clip((t - self.t_cof - 0.1) / (self.t_hit - self.t_cof - 0.1), 0, 1)
            drop = -160 * (1 - u ** 2)                                     # falls, lands on the hit, settles with a bounce
            if t > self.t_hit:
                drop = -10 * np.sin((t - self.t_hit) * 18) * np.exp(-(t - self.t_hit) * 8)
            self.place(fr, cam, blur(self.P['cup'], 10), cx + 10, cy + 18, opacity=0.18 * fc * (0.4 + 0.6 * u), scale=1.25)
            self.place(fr, cam, self.P['cup'], cx, cy + drop, opacity=fc, scale=1.25)
            self.layer(fr, cam, self._steam(t, cx, cy - 75), color=GREY, opacity=0.55 * smooth((t - self.t_cof - 0.4) / 0.8))
        for k, g in enumerate(self.kohi):
            f = (t - self.t_hit - 0.08 - 0.14 * k) / 0.35
            if f > 0:
                self.place(fr, cam, g.at(f, soft=0.06), 540 + (k - 1.5) * 150, 585)
        fm = smooth((t - self.t_coffee) / 0.35)
        sl = ease_io((t - self.t_coffee) / 0.4)
        self.place(fr, cam, self.ro, lerp(540, 380, sl), 716, color=GREY, opacity=smooth((t - self.t_kohee - 0.45) / 0.35))
        self.place(fr, cam, self.eq, 478, 712, color=GREY, opacity=fm)
        self.place(fr, cam, self.coffee, 640, 712 + 12 * (1 - fm), color=VERM, opacity=fm)
        return fr.img

    def _steam(self, t, cx, y0):
        """three wisps rising from the cup, curling as they go"""
        d = np.zeros((H, W), np.float32)
        for k, dx in enumerate((-58, 0, 58)):
            ys = np.linspace(y0, y0 - 230, 50)
            u = (y0 - ys) / 230
            xs = cx + dx * (1 + 0.3 * u) + (6 + 14 * u) * np.sin(u * 6 - t * 2.0 + k * 0.4) + 4 * np.sin(t * 0.9 + k)
            pts = np.stack([xs, ys], 1).astype(np.int32)
            layer = np.zeros_like(d)
            cv2.polylines(layer, [pts], False, 1.0, 9, cv2.LINE_AA)
            fade = np.zeros((H, 1), np.float32)
            a, b = int(y0 - 230), int(y0)
            fade[a:b, 0] = np.sin(np.linspace(np.pi, 0, b - a)) ** 1.2
            d = np.maximum(d, layer * fade)
        return blur(d, 5)

    def _put(self, fr, cam, d, x, y, sx, sy, color, op, over=False):
        """place a patch centred at (x, y) with separate x / y scale (for the card flip); over=True paints it opaque"""
        if op <= 0:
            return
        h, w = d.shape
        M = cam.matrix()
        A = np.array([[sx, 0, x - sx * w / 2], [0, sy, y - sy * h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        a = cv2.warpAffine(d, Mt, (W, H))
        if over:
            fr.over(color, a, op)
        else:
            fr.ink(a, color, op)

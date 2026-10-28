"""Scene 7 — the tip worth saving.  "Quick tip: alone, a kanji is often read with kun'yomi.  In compound words, it's
often on'yomi.  Not always, so learn the word."
QUICK TIP is brushed at the top.  Two columns recap every word of the reel: ALONE / often 訓読み (indigo) with
山 やま, 日 ひ, 水 みず, 人 ひと, and IN WORDS / often 音読み (vermilion) with 富士山 さん, 日曜日 にち, 水曜日 すい, 日本人 じん,
each column writing itself in as Raj says it and its label pulsing on its reading's name.  On "Not always" a small
grey NOT ALWAYS is set under them, and LEARN THE WORD. is stamped in vermilion with a brush underline."""
import numpy as np

import kit
from cards import INDIGO
from engine import (FONT_BRUSH_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, noise, smooth,
                    spline, stroke, text_layer)
from scenes.kscene import KanjiScene
from timeline import JA, at

LEFT = [('山', 'やま'), ('日', 'ひ'), ('水', 'みず'), ('人', 'ひと')]
RIGHT = [('富士山', 'さん', 2), ('日曜日', 'にち', 0), ('水曜日', 'すい', 0), ('日本人', 'じん', 2)]
ROWS = [600, 730, 860, 990]
XL, XR = 270, 800


class Scene(KanjiScene):
    paper_seed = 107

    def build(self):
        self.head, _ = text_layer('QUICK TIP', FONT_BRUSH_EN, 130)
        self.head = self.head * (0.85 + 0.15 * noise(*self.head.shape, 6, 2, 6))
        often, _ = text_layer('often', FONT_EN_ITALIC, 44)
        self.often = often
        self.lab = [(text_layer('ALONE', FONT_EN_BOLD, 46, tracking=8)[0], text_layer('訓読み', FONT_JP, 46)[0]),
                    (text_layer('IN WORDS', FONT_EN_BOLD, 46, tracking=8)[0], text_layer('音読み', FONT_JP, 46)[0])]
        self.lk = [(text_layer(k, FONT_JP, 88)[0], text_layer(r, FONT_JP, 62)[0]) for k, r in LEFT]
        self.rk = [(text_layer(k, FONT_JP, 64)[0], text_layer(r, FONT_JP, 50)[0], i) for k, r, i in RIGHT]
        self.na, _ = text_layer('NOT ALWAYS', FONT_EN_BOLD, 40, tracking=10)
        self.learn, _ = text_layer('LEARN THE WORD.', FONT_BRUSH_EN, 112)
        self.learn = self.learn * (0.85 + 0.15 * noise(*self.learn.shape, 6, 2, 7))
        self.t_tip = at('Quick tip')[0]
        self.t_alone = at('alone a kanji')[0]
        self.t_kun = JA['kunyomi_tip']['start']
        self.t_comp = at('In compound')[0]
        self.t_on = JA['onyomi_tip']['start']
        self.t_not = at('Not always')[0]
        self.t_learn = at('learn the word')[0]
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [(540, 430), (538, 760), (541, 1060)], 3, ink=0.55, dry=0.6, profile=((0, 0.2), (0.1, 1), (0.9, 1), (1, 0.2)),
                      seed=150), self.t_comp - 0.1, 0.4)
        ss.add(stroke(W, H, [(260, 1262), (540, 1252), (820, 1260)], 12, ink=0.95, dry=0.4, profile=((0, 0.3), (0.1, 1), (0.9, 0.8), (1, 0.1)),
                      seed=151, red=True), self.t_learn + 0.25, 0.3)
        self.ss = ss

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.02 * smooth((t - self.t0) / 9) + 0.03 * np.exp(-max(t - self.t_learn, 0) * 6) * (t > self.t_learn))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fh = (t - self.t_tip) / 0.18
        if fh > 0:
            kit.put(fr, cam, self.head, 540, 250, VERM, smooth(fh), 1 + 0.3 * np.exp(-fh * 2.5), rot=-0.03)
        self.layer(fr, cam, self.ss.render(t), opacity=0.8)
        self.layer(fr, cam, self.ss.render(t, red=True), color=VERM)
        for side, (x, t0, tp, col) in enumerate(((XL, self.t_alone, self.t_kun, INDIGO), (XR, self.t_comp, self.t_on, VERM))):
            p = np.exp(-max(t - tp, 0) * 4) * (t > tp)
            a, b = self.lab[side]
            kit.put(fr, cam, a, x, 420, col, smooth((t - t0 + 0.1) / 0.25))
            kit.put(fr, cam, self.often, x - 90, 482, INK, smooth((t - t0) / 0.25))
            kit.put(fr, cam, b, x + 40, 480, col, smooth((t - t0) / 0.25), 1 + 0.3 * p)
            for k, y in enumerate(ROWS):
                f = smooth((t - t0 - 0.12 * k) / 0.25)
                if f <= 0:
                    continue
                if side == 0:
                    g, r = self.lk[k]
                    kit.put(fr, cam, g, x - 70, y, INK, f)
                    kit.put(fr, cam, r, x + 70, y, INDIGO, f, 1 + 0.2 * p)
                else:
                    g, r, _ = self.rk[k]
                    kit.put(fr, cam, g, x - 60, y, INK, f)
                    kit.put(fr, cam, r, x + 110, y, VERM, f, 1 + 0.2 * p)
        kit.put(fr, cam, self.na, 540, 1100, GREY, smooth((t - self.t_not) / 0.25))
        fl = (t - self.t_learn) / 0.15
        if fl > 0:
            kit.put(fr, cam, self.learn, 540, 1200, VERM, smooth(fl), 1 + 0.3 * np.exp(-fl * 2.5), rot=-0.03)
        return fr.img

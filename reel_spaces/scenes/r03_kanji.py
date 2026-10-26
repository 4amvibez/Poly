"""Scene 3 — the first clue, kanji.  "The first clue is kanji.  Look at this sentence. [私は日本語を勉強します。]  Kanji
carries important meaning.  When readers see 'nihongo', they can recognize the word for Japanese language."
CLUE 1 · 漢字 stamps in and settles as a header; the sentence is written chunk by chunk exactly as Raj says it in
Japanese, with "I study Japanese." beneath; gold swashes sweep behind the kanji (私, 日本語, 勉強) while the hiragana
step back; 日本語 lifts out into a card with its reading にほんご and meaning; then three more kanji words show
themselves at a glance: 学校 school, 東京 Tokyo, 電車 train, each with a small ink drawing."""
import numpy as np

import vignettes
from backdrop import backdrop, draw_backdrop
from clues import Clue
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_io,
                    ease_out, keyframes, lerp, smooth, stroke, text_layer)
from glyphs import text_glyphs
from scenes.base import SceneBase
from sentence import GOLD, Sentence, chunk_reveal
from timeline import JA_SENT, at, quiet_gap

SY = 760
CARDS = [('学校', 'がっこう', 'school', vignettes.school), ('東京', 'とうきょう', 'Tokyo', vignettes.tokyo),
         ('電車', 'でんしゃ', 'train', vignettes.train)]
CX = [225, 540, 855]
CY = 1170


class Scene(SceneBase):
    paper_seed = 65

    def build(self):
        self.clue = Clue(1, '漢字', 'kanji', at('The first clue')[0] + 0.2, at('Look at')[0] - 0.1)
        self.B = backdrop(seed=70, branch='right', pagoda_x=880, reader=True)
        self.sent = Sentence(size=74)
        self.t_meaning_en = JA_SENT['end'] + 0.1
        self.meaning, _ = text_layer('"I study Japanese."', FONT_EN_ITALIC, 50)
        self.t_gold = at('Kanji carries')[0] + 0.1
        self.gold = [(self.sent.swash(idx, color_seed=k), self.t_gold + 0.32 * k) for k, idx in enumerate(([0], [2, 3, 4], [6, 7]))]
        self.t_lift = at('nihongo')[0] - 0.1
        self.t_hit = quiet_gap(at('nihongo')[0] + 0.3, at('recognize the word')[0])[0]   # sync: frame closes (+ bowl)
        D, T, _ = text_glyphs('にほんご', 40, font=FONT_JP, seed=3, tracking=0.08)
        self.furi = D
        self.jl, _ = text_layer('Japanese language', FONT_EN, 50)
        self.t_cards = [at('recognize the word')[0] + 0.1 + 0.42 * k for k in range(3)]

        def cards():
            out = {}
            for k, (kj, kana, en, fn) in enumerate(CARDS):
                out[f'pic{k}'] = fn().astype(np.float32)
            return out
        self.pics = cached('r03_cards_v1', cards)
        self.card = []
        for k, (kj, kana, en, fn) in enumerate(CARDS):
            D1, _, _ = text_glyphs(kj, 88, font=FONT_JP, seed=20 + k, tracking=0.02)
            D2, _, _ = text_glyphs(kana, 30, font=FONT_JP, seed=30 + k, tracking=0.06)
            self.card.append(dict(kj=D1, kana=D2, en=text_layer(en, FONT_EN, 44)[0], pic=self.pics[f'pic{k}']))
        box = np.zeros((360, 250), np.float32)
        box[4:-4, 4:-4] = 1
        self.box = blur(box, 2.5)
        rim = np.zeros_like(box)
        rim[10:-10, 10:-10] = 1
        inner = np.zeros_like(box)
        inner[13:-13, 13:-13] = 1
        self.rim = np.clip(blur(rim, 0.7) - blur(inner, 0.7), 0, 1)
        # the 日本語 card's vermilion frame
        fr_ = StrokeSet(W, H)
        pts = [(282, 404), (798, 400), (802, 722), (278, 726), (282, 400)]
        fr_.add(stroke(W, H, pts, 6, ink=0.95, dry=0.4, profile=((0, 0.6), (0.05, 1), (0.95, 0.9), (1, 0.5)), seed=9, red=True),
                self.t_hit - 0.45, 0.45)
        self.frame_red = fr_

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 940, 1.02)), (self.t_lift, (540, 930, 1.0)), (self.t1, (540, 950, 1.02))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        self.clue.draw(self, fr, cam, t)
        lift = ease_io((t - self.t_lift) / 0.7)
        # gold swashes behind the kanji
        for k, (D, ts) in enumerate(self.gold):
            f = (t - ts) / 0.35
            if f > 0:
                a = min(1, f)
                w_ = D.shape[1]
                m = np.zeros_like(D)
                m[:, :int(w_ * a)] = 1
                self.place(fr, cam, D * blur(m, 6), 540, SY + 80 * lift, color=GOLD, opacity=0.55 * (1 - (1.0 if k == 1 else 0.6) * lift))
        # the sentence, written as it is spoken; then the hiragana step back and 日本語 lifts out
        rv = chunk_reveal(t, JA_SENT['chunks'])
        hira = smooth((t - self.t_gold) / 0.5)
        def opac(i):
            if 2 <= i <= 4:
                return 1 - lift
            o = 1 - 0.55 * hira * (self.sent.kind[i] != 'K')
            return o * (1 - 0.55 * lift)
        sy = SY + 80 * lift                                          # the rest of the sentence makes room for the card
        self.sent.draw(self, fr, cam, 540, sy, rv, opacity=opac)
        if lift > 0:
            for i in (2, 3, 4):
                g = self.sent.g[i]
                x = lerp(540 + self.sent.x[i], 540 + (i - 3) * 74 * 1.9, lift)
                y = lerp(SY, 540, lift)
                self.place(fr, cam, g.dens, x, y, scale=lerp(1, 1.9, lift))
            lo = smooth((t - self.t_lift - 0.5) / 0.35)
            self.place(fr, cam, self.furi, 540, 448, color=GREY, opacity=lo)
            self.place(fr, cam, self.jl, 540, 668, color=INK, opacity=smooth((t - self.t_lift - 0.7) / 0.35))
            self.layer(fr, cam, self.frame_red.render(t, red=True), color=VERM, opacity=1)
        # the meaning, under the sentence
        om = smooth((t - self.t_meaning_en) / 0.4) * (1 - lift)
        self.place(fr, cam, self.meaning, 540, SY + 100, color=GREY, opacity=om)
        # three more words, each with a picture
        for k, C in enumerate(self.card):
            f = (t - self.t_cards[k]) / 0.4
            if f <= 0:
                continue
            e = ease_out(f)
            x, y = CX[k], CY + 40 * (1 - e)
            self.place(fr, cam, blur(self.box, 8), x + 6, y + 10, opacity=0.25 * e)
            fr_op = smooth(f * 1.5)
            self._paper_card(fr, cam, x, y, fr_op)
            self.place(fr, cam, self.rim, x, y, color=np.array([0.55, 0.42, 0.3], np.float32), opacity=0.6 * fr_op)
            self.place(fr, cam, C['kana'], x, y - 140, color=GREY, opacity=fr_op)
            self.place(fr, cam, C['kj'], x, y - 78, opacity=fr_op)
            self.place(fr, cam, C['en'], x, y - 8, color=GREY, opacity=fr_op)
            self.place(fr, cam, C['pic'], x, y + 92, opacity=fr_op, scale=0.95)
        return fr.img

    def _paper_card(self, fr, cam, x, y, op):
        import cv2
        h, w = self.box.shape
        M = cam.matrix()
        A = np.array([[1, 0, x - w / 2], [0, 1, y - h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(np.array([0.97, 0.95, 0.89], np.float32), cv2.warpAffine(self.box, Mt, (W, H)), op)

"""Scene 6 — the reveal.  "But here's the real secret.  Japanese readers don't rely on just one clue.  They combine
kanji, kana, grammar, vocabulary, and context, to recognize meaningful chunks of a sentence.  Their brains learn to
see the boundaries, even when spaces aren't written."
秘密 (the secret) stamps in and settles; a reader's head is brushed in profile.  "Just one clue": a lone clue seal
appears and shakes its head.  Then five clue badges land one per word (漢字 kanji, かな kana, 文法 grammar, 語彙
vocabulary, 文脈 context), each pouring a coloured stream into the reader's head, where the brain warms with every
one.  Below, the sentence is written and its meaningful chunks light up; on "boundaries" they part at the joins, and
on "spaces aren't written" they close up again, the colours still showing where each chunk ends."""
import numpy as np

import head as HD
from backdrop import backdrop, draw_backdrop
from clues import Clue, seal
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, VERM, W, Cam, StrokeSet, blur, ease_io, ease_out, keyframes,
                    lerp, smooth, spline, stroke, text_layer)
from glyphs import text_glyphs
from poly_ink.brush import PRESS_DOWN
from scenes.base import SceneBase
from sentence import CHUNK_COL, GOLD, PINK, BLUE, Sentence
from timeline import WORDS, at, quiet_gap

GREEN = np.array([0.42, 0.62, 0.46], np.float32)
RUST = np.array([0.86, 0.46, 0.34], np.float32)
BADGES = [('漢字', 'kanji', GOLD), ('かな', 'kana', PINK), ('文法', 'grammar', BLUE), ('語彙', 'vocabulary', GREEN),
          ('文脈', 'context', RUST)]
BX = [160, 350, 540, 730, 920]
BY = 535
HEAD = (520, 905)
SY = 1262
DX = [-24, 0, 24]
GLOW = np.array([1.0, 0.86, 0.52], np.float32)


def after(word, t):
    """(start, end) of the first narration word matching `word` after time t"""
    w = next(w for w in WORDS if w['start'] > t and ''.join(c for c in w['text'].lower() if c.isalnum()) == word)
    return w['start'], w['end']


class Scene(SceneBase):
    paper_seed = 68

    def build(self):
        self.t_in = quiet_gap(at('real secret')[1] - 0.3, at('Japanese readers')[0] + 0.1)[0]   # sync: the seal stamps (+ taiko)
        self.clue = Clue('!', '秘密', 'himitsu · the real secret', self.t_in, self.t_in + 0.75, label='THE SECRET', big=(350, 0.85))
        self.B = backdrop(seed=130, branch='right', ground=1430)
        self.head = HD.build()
        self.bx, self.by = HEAD[0] - HD.HW / 2 + HD.BRAIN[0], HEAD[1] - HD.HH / 2 + HD.BRAIN[1]
        self.t_head = self.t0 + 0.5                                     # the reader is brushed in while Raj sets up the secret
        self.t_one = at('just one clue')[0]
        self.seal1 = seal(1, 120)
        t = at('They combine')[0]
        self.t_badge = []
        for _, en, _ in BADGES:
            s, _ = after(en, t)
            self.t_badge.append(s - 0.08)
            t = s
        self.badge = []
        for k, (jp, en, col) in enumerate(BADGES):
            D, _, _ = text_glyphs(jp, 54, font=FONT_JP, seed=10 + k, tracking=0.02)
            self.badge.append((D, text_layer(en, FONT_EN_ITALIC, 36)[0], col))
        yy, xx = np.mgrid[0:170, 0:170].astype(np.float32)
        rr = np.hypot(xx - 85, yy - 85)
        self.disc = np.clip((72 - rr) / 2.5, 0, 1) * (0.85 + 0.15 * np.cos(xx * 0.09) * np.sin(yy * 0.07))
        self.ring = np.clip(1 - np.abs(rr - 72) / 2.2, 0, 1)
        # a stream from each badge into the brain
        self.streams = []
        for k, x in enumerate(BX):
            ss = StrokeSet(W, H)
            pts = [(x, BY + 112), (x + (self.bx - x) * 0.15, BY + 170), (lerp(x, self.bx, 0.6), self.by - 70), (self.bx + (x - self.bx) * 0.12, self.by - 8)]
            ss.add(stroke(W, H, spline(pts, 12), 9, ink=0.9, dry=0.45, profile=((0, 0.9), (0.6, 0.7), (1, 0.2)), seed=50 + k),
                   self.t_badge[k] + 0.3, 0.55)
            self.streams.append(ss)
        # the sentence and its chunks
        self.sent = Sentence(size=66, seed=200)
        S = self.sent
        self.t_write = at('to recognize')[0] - 0.1
        self.t_brain = quiet_gap(at('of a sentence')[1] - 0.3, at('Their brains')[0] + 0.1)[0]   # sync: the brain lights up (+ bowl)
        self.t_chunk = at('chunks')[0]
        self.t_bound = at('boundaries')[0] - 0.1
        self.t_close = after('spaces', self.t_bound)[0]
        self.sw = [S.swash(idx, color_seed=80 + k, width=1.05) for k, idx in enumerate(([0, 1], [2, 3, 4, 5], [6, 7, 8, 9, 10]))]
        ticks = StrokeSet(W, H)
        for k, (a, b) in enumerate([(1, 2), (5, 6)]):
            x = 540 + (S.x[a] + DX[S.chunk[a]] + S.x[b] + DX[S.chunk[b]]) / 2
            ticks.add(stroke(W, H, [(x + 2, SY - 48), (x, SY), (x - 2, SY + 46)], 7, ink=0.95, dry=0.4, profile=PRESS_DOWN,
                             seed=60 + k, red=True), self.t_bound + 0.15 * k, 0.3)
        self.ticks = ticks
        self.gloss = [text_layer(s, FONT_EN, 42)[0] for s in ('I', 'Japanese', 'study')]
        self.gx = [(S.x[0] + S.x[1]) / 2, (S.x[2] + S.x[5]) / 2, (S.x[6] + S.x[10]) / 2]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 950, 1.0)), (self.t_badge[0], (540, 930, 1.0)), (self.t_write, (540, 950, 1.0)),
                                  (self.t1, (540, 975, 1.04))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, 0.8 * smooth((t - self.t0) / 0.8))
        self.clue.draw(self, fr, cam, t)
        # --- the reader's head, and the brain warming with each clue
        fh = (t - self.t_head) / 1.0
        if fh > 0:
            m = np.zeros_like(self.head)
            yy = np.arange(HD.HH, dtype=np.float32)[:, None]
            m[:] = np.clip((fh * (HD.HH + 120) - yy) / 60, 0, 1)
            self.place(fr, cam, self.head * m, *HEAD)
            lvl = sum(smooth((t - tb - 0.8) / 0.4) for tb in self.t_badge) / len(self.t_badge)
            pulse = sum(np.exp(-max(t - tb - 0.85, 0) * 3) * (t > tb + 0.85) for tb in self.t_badge)
            pulse += 1.6 * np.exp(-max(t - self.t_brain, 0) * 2.2) * (t > self.t_brain)
            g = np.clip(0.15 * smooth(fh) + 0.45 * lvl + 0.3 * min(pulse, 1.4), 0, 0.92)
            yy, xx = np.mgrid[0:240, 0:240].astype(np.float32)
            disc = np.exp(-((xx - 120) ** 2 + (yy - 120) ** 2) / (2 * (52 + 10 * lvl) ** 2))
            self._over(fr, cam, disc, self.bx, self.by, GLOW, g)
        # --- just one clue?  a single seal pops up and shakes its head
        fo = (t - self.t_one) / 0.3
        if 0 < fo and t < self.t_badge[0] + 0.2:
            shake = np.sin((t - self.t_one - 0.6) * 22) * 0.12 * np.exp(-max(t - self.t_one - 0.6, 0) * 2) * (t > self.t_one + 0.6)
            op = smooth(fo) * (1 - smooth((t - self.t_badge[0] + 0.5) / 0.4))
            self.place(fr, cam, self.seal1, 540, BY, color=VERM, opacity=0.92 * op, scale=0.8 + 0.2 * ease_out(fo), rot=shake)
        # --- five clues, poured into the head
        for k, ((D, lbl, col), x) in enumerate(zip(self.badge, BX)):
            f = (t - self.t_badge[k]) / 0.35
            if f <= 0:
                continue
            e = ease_out(f)
            sc = 0.55 + 0.45 * e + 0.12 * np.sin(np.pi * min(f, 1))
            y = BY - 40 * (1 - e)
            self.place(fr, cam, self.disc, x, y, color=col, opacity=0.5 * smooth(f), scale=sc)
            self.place(fr, cam, self.ring, x, y, color=col * 0.7, opacity=0.8 * smooth(f), scale=sc)
            self.place(fr, cam, D, x, y, opacity=smooth(f * 1.4), scale=sc)
            self.place(fr, cam, lbl, x, BY + 92, color=GREY, opacity=smooth((t - self.t_badge[k] - 0.15) / 0.3))
            self.layer(fr, cam, self.streams[k].render(t), color=col * 0.85, opacity=0.8)
        # --- the sentence: chunks light up, part at the joins, close again
        S = self.sent
        part = ease_io((t - self.t_bound) / 0.6) * (1 - ease_io((t - self.t_close) / 0.8))
        for k, D in enumerate(self.sw):
            f = (t - self.t_chunk - 0.28 * k) / 0.35
            if f > 0:
                m = np.zeros_like(D)
                m[:, :int(D.shape[1] * min(1, f))] = 1
                self.place(fr, cam, D * blur(m, 6), 540 + DX[k] * part, SY, color=CHUNK_COL[k], opacity=0.5)
        rv = lambda i: smooth((t - self.t_write - 0.04 * i) / 0.3)
        S.draw(self, fr, cam, 540, SY, rv, offset=lambda i: (DX[S.chunk[i]] * part, 0))
        self.layer(fr, cam, self.ticks.render(t, red=True), color=VERM, opacity=1 - smooth((t - self.t_close) / 0.6))
        for k, (g, x) in enumerate(zip(self.gloss, self.gx)):
            self.place(fr, cam, g, 540 + x + DX[k] * part, SY + 84, color=GREY, opacity=smooth((t - self.t_bound - 0.3 - 0.15 * k) / 0.35))
        return fr.img

    def _over(self, fr, cam, mask, x, y, color, op):
        import cv2
        h, w = mask.shape
        M = cam.matrix()
        A = np.array([[1, 0, x - w / 2], [0, 1, y - h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(color, cv2.warpAffine(mask.astype(np.float32), Mt, (W, H)), op)

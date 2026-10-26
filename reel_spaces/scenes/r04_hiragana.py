"""Scene 4 — the second clue, hiragana.  "Next comes hiragana.  Characters like 'wa', 'o', and 'shimasu' help readers
understand how the sentence is structured.  They show grammatical relationships, and how words work together."
CLUE 2 · ひらがな; the sentence returns with its kanji stepped back and は, を, します picked out in colour, each with its
job written beneath (topic marker, object marker, polite verb ending).  On "structured" the sentence parts at the
particles into its three chunks; on "relationships" two brush arcs ask who? and what? of the verb.  Underneath, a
little bullet train has rolled in: the chunks ride in its cars, the particles are its couplers, and on "work together"
the couplers snap tight and the train pulls away."""
import cv2
import numpy as np

import train as TR
from backdrop import backdrop, draw_backdrop
from clues import Clue
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, ease_in, ease_io, ease_out,
                    keyframes, lerp, smooth, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import PRESS_DOWN
from scenes.base import SceneBase
from sentence import BLUE, PINK, Sentence
from timeline import at, quiet_gap

SY = 650
TY = 1070                         # screen y of the train sprite's centre
DX = [-26, 0, 26]                 # how far each chunk moves when the sentence parts
CAR_PAPER = np.array([0.95, 0.93, 0.87], np.float32)


class Scene(SceneBase):
    paper_seed = 66

    def build(self):
        self.clue = Clue(2, 'ひらがな', 'hiragana', at('Next comes')[0] + 0.2, at('Characters like')[0] - 0.1)
        self.B = backdrop(seed=90, branch='right', ground=1340)
        self.T = TR.build()
        self.track = TR.track(W, H, TY + 96, TY - 94, seed=3)
        self.speed = TR.speed_lines(5)
        self.badge = TR.badge(31, seed=4)
        self.sent = Sentence(size=74)
        S = self.sent
        self.t_write = at('Characters like')[0] - 0.6
        self.t_mark = [at('wa')[0], at('o')[0], at('shimasu')[0]]
        self.marks = [([1], PINK, 'topic marker'), ([5], PINK, 'object marker'), ([8, 9, 10], BLUE, 'polite verb ending')]
        self.sw = [S.swash(idx, color_seed=60 + k) for k, (idx, _, _) in enumerate(self.marks)]
        self.lbl = [text_layer(s, FONT_EN_ITALIC, 40)[0] for _, _, s in self.marks]
        self.t_part = at('help readers')[0]
        self.t_struct = at('structured')[0] - 0.15
        self.t_rel = at('grammatical')[0]
        self.t_tog = at('how words')[1]
        self.t_arrive = at('Characters like')[0] - 0.5
        self.t_depart = at('work together')[0] + 0.1
        self.t_hit = quiet_gap(at('structured')[1] - 0.3, at('They show')[0] + 0.1)[0]   # sync: couplers snap (+ clack)
        # chunk glosses
        self.gloss = [text_layer(s, FONT_EN, 44)[0] for s in ('I', 'Japanese', 'study')]
        self.gx = [(S.x[0] + S.x[1]) / 2 + DX[0], (S.x[2] + S.x[5]) / 2 + DX[1], (S.x[6] + S.x[10]) / 2 + DX[2]]
        # vermilion ticks where the chunks part
        ticks = StrokeSet(W, H)
        for k, (a, b) in enumerate([(1, 2), (5, 6)]):
            x = 540 + (S.x[a] + DX[S.chunk[a]] + S.x[b] + DX[S.chunk[b]]) / 2
            ticks.add(stroke(W, H, [(x + 2, SY - 52), (x, SY), (x - 2, SY + 50)], 7, ink=0.95, dry=0.4, profile=PRESS_DOWN,
                             seed=30 + k, red=True), self.t_struct + 0.12 * k, 0.3)
        self.ticks = ticks
        # who? / what? arcs over the sentence, ending on the verb
        arcs = StrokeSet(W, H)
        cx = lambda i: 540 + S.x[i] + DX[S.chunk[i]]
        top = SY - 52
        for k, (i0, i1, apex) in enumerate([(0, 7, 492), (3, 6, 562)]):
            x0, x1 = cx(i0), cx(i1)
            pts = [(x0 + (x1 - x0) * u, top - (top - apex) * 4 * u * (1 - u)) for u in np.linspace(0, 1, 9)]
            t = self.t_rel + 0.45 * k
            arcs.add(stroke(W, H, pts, 6, ink=0.95, dry=0.35, profile=((0, 0.5), (0.1, 1), (0.9, 0.9), (1, 0.6)), seed=40 + k, red=True), t, 0.55)
            for s_ in (-1, 1):
                arcs.add(stroke(W, H, [(x1 + s_ * 16, top - 22), (x1, top - 2)], 5, ink=0.95, dry=0.3, profile=PRESS_DOWN, seed=44 + k + s_,
                                red=True), t + 0.5, 0.15)
        self.arcs = arcs
        self.arc_lbl = [(text_layer('who?', FONT_EN_ITALIC, 42)[0], (cx(0) + cx(7)) / 2, 466, self.t_rel + 0.5),
                        (text_layer('what?', FONT_EN_ITALIC, 42)[0], (cx(3) + cx(6)) / 2, 590, self.t_rel + 0.95)]
        # the words in the cars and the particles on the couplers
        self.car_words = []
        for k, (word, x) in enumerate(zip(['私', '日本語', '勉強します'], TR.TEXT_X)):
            gl = [typeset(c, 54, font=FONT_JP, seed=70 + 5 * k + j, pad=0.04, medians=False, sweep=(1, 0.3)) for j, c in enumerate(word)]
            xs = [x + (j - (len(word) - 1) / 2) * 56 for j in range(len(word))]
            self.car_words.append((gl, xs, self.t_struct + 0.25 + 0.3 * k))
        self.part = [typeset(c, 42, font=FONT_JP, seed=90 + k, pad=0.04, medians=False, sweep=(1, 0.3)) for k, c in enumerate('はを')]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 945, 1.02)), (self.t_part, (540, 930, 1.0)), (self.t1, (540, 950, 1.03))], ease=ease_io))

    def train_x(self, t):
        """horizontal offset of the train and its speed (0..1) at time t"""
        a = np.clip((t - self.t_arrive) / 1.7, 0, 1)
        d = np.clip((t - self.t_depart) / 1.2, 0, 1)
        off = lerp(-1180, 0, ease_out(a)) + 1300 * ease_in(d)
        sp = (1 - a) ** 1.5 * (a > 0) + d ** 0.8 * (d > 0)
        return off, float(np.clip(sp, 0, 1))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        S = self.sent
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        self.layer(fr, cam, self.track, opacity=0.9 * smooth((t - self.t0 - 0.2) / 0.6))
        self.clue.draw(self, fr, cam, t)
        # --- the sentence: particles marked, then parted into chunks
        part = ease_io((t - self.t_part) / 0.7)
        lbl_out = 1 - smooth((t - self.t_part) / 0.4)
        for k, ((idx, col, _), D) in enumerate(zip(self.marks, self.sw)):
            f = (t - self.t_mark[k] + 0.05) / 0.3
            if f <= 0:
                continue
            m = np.zeros_like(D)
            m[:, :int(D.shape[1] * min(1, f))] = 1
            dx = DX[S.chunk[idx[0]]] * part
            self.place(fr, cam, D * m, 540 + dx, SY, color=col, opacity=0.55)
            lo = smooth((t - self.t_mark[k] - 0.15) / 0.3) * lbl_out
            if lo > 0:
                x = 540 + np.mean([S.x[i] for i in idx])
                tick = np.zeros((34, 6), np.float32)
                tick[:, 2:4] = 1
                self.place(fr, cam, tick, x, SY + 64, color=col * 0.8, opacity=lo)
                self.place(fr, cam, self.lbl[k], x, SY + 106, color=col * 0.75, opacity=lo)
        dim = smooth((t - self.t_mark[0] + 0.6) / 0.4) * (1 - part)
        rv = lambda i: smooth((t - self.t_write - 0.045 * i) / 0.3)
        self.sent.draw(self, fr, cam, 540, SY, rv, opacity=lambda i: 1 - 0.62 * dim * (S.kind[i] == 'K'),
                       offset=lambda i: (DX[S.chunk[i]] * part, 0))
        self.layer(fr, cam, self.ticks.render(t, red=True), color=VERM)
        for k, (g, x) in enumerate(zip(self.gloss, self.gx)):
            self.place(fr, cam, g, 540 + x, SY + 92, color=GREY, opacity=smooth((t - self.t_struct - 0.1 - 0.15 * k) / 0.35))
        self.layer(fr, cam, self.arcs.render(t, red=True), color=VERM)
        for g, x, y, ts in self.arc_lbl:
            self.place(fr, cam, g, x, y, color=VERM, opacity=smooth((t - ts) / 0.3))
        # --- the train
        off, sp = self.train_x(t)
        if off > -1170:
            ox = 540 - TR.TW / 2 + off                                  # sprite left edge on screen
            oy = TY - TR.TH / 2
            if sp > 0.02:
                self.place(fr, cam, self.speed, ox - 150, TY - 10, opacity=0.8 * sp)
            self._over(fr, cam, self.T['body'], ox, oy, CAR_PAPER, 1.0)
            self.layer(fr, cam, self.T['ink'], ox, oy)
            self.layer(fr, cam, self.T['stripe'], ox, oy, color=BLUE, opacity=0.85)
            for gl, xs, ts in self.car_words:
                for j, (g, x) in enumerate(zip(gl, xs)):
                    f = (t - ts - 0.07 * j) / 0.3
                    if f > 0:
                        self.place(fr, cam, g.at(min(f, 1.05), soft=0.05, wet=0.35), ox + x, oy + TR.TEXT_Y)
            snap = np.exp(-max(t - self.t_tog, 0) * 4.0) * (t > self.t_tog)
            for k, ((bx, by), g) in enumerate(zip(TR.COUPLERS, self.part)):
                f = smooth((t - self.t_hit) / 0.12)
                if f > 0:
                    hit = np.exp(-max(t - self.t_hit, 0) * 6)
                    sc = (0.6 + 0.4 * ease_out(f)) * (1 + 0.28 * snap + 0.3 * hit)
                    self.place(fr, cam, self.badge, ox + bx, oy + by, color=PINK, opacity=0.8 * f, scale=sc)
                    self.place(fr, cam, g.dens, ox + bx, oy + by, opacity=f, scale=sc)
        return fr.img

    def _over(self, fr, cam, mask, x0, y0, color, op):
        M = cam.matrix()
        A = np.array([[1, 0, x0], [0, 1, y0]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(color, cv2.warpAffine(mask, Mt, (W, H)), op)

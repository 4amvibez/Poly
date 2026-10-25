"""Scene 5 — hiragana.  Sounds written with whole kanji (安 加 左 太 奈 = a ka sa ta na, the man'yogana behind
あかさたな); 安 written in full, its parts picked out in vermilion; a flowing cursive form brushed over it;
then the typeset hiragana あ.  The three stages settle into a row, marked as a simplified illustration (the real
development ran through many intermediate hands).  For "the Heian court": the Iroha poem, a Heian-period verse
that uses each kana once, flows down in vertical columns."""
import numpy as np

from art import brush_sprite, draw_sprite
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP_FALLBACK, GREY, H, INK, SEPIA, VERM, W, Cam, Frame, StrokeSet,
                    blur, bloom, ease_io, ease_out, keyframes, noise, ramp, reveal_time, shift_matrix, smooth, spline,
                    sstep, stroke, text_layer, warp)
from glyphs import calligraphy, typeset
from poly_ink.brush import PRESS_DOWN, RISE, SWEEP
from scenes.base import SceneBase
from timeline import at

S = 520
IROHA = ['いろはにほへと', 'ちりぬるを', 'わかよたれそ', 'つねならむ']


class Scene(SceneBase):
    paper_seed = 15
    paper_stains = 0.08

    def build(self):
        # A: sounds written with whole characters
        self.row = [typeset(c, 128, font=FONT_JP_FALLBACK, seed=20 + i, medians=False) for i, c in enumerate('安加左太奈')]
        self.row_r = [text_layer(r, FONT_EN_ITALIC, 42)[0] for r in ('a', 'ka', 'sa', 'ta', 'na')]
        self.tA = at('Writing sounds')[0] + 0.1
        # B: 安 written in full, then its parts highlighted
        self.an = calligraphy('安', S, seed=9)
        self.an_halo = blur(self.an.dens, 5)
        self.tB = at('As scribes')[0] - 0.2
        self.tB_end = self.tB + 2.2
        self.t_hl = self.tB_end + 0.1
        self.roof = self.an.stroke_mask([0, 1, 2])
        self.woman = self.an.stroke_mask([3, 4, 5])
        # C: the cursive form, brushed over the ghost of 安
        n = self.an.w
        P = self.an.pad
        sc = S / 1024
        q = lambda pts: [(P + x * sc, P + y * sc) for x, y in pts]
        self.tC = at('in flowing cursive')[0] - 0.1
        cur = StrokeSet(n, n)
        t = self.tC
        t = cur.add(stroke(n, n, spline(q([(470, 90), (505, 150), (520, 190)]), 8), 34, ink=0.92, dry=0.3, profile=PRESS_DOWN, seed=1), t, 0.35) + 0.08
        t = cur.add(stroke(n, n, spline(q([(250, 390), (252, 310), (330, 268), (560, 252), (770, 240), (790, 270), (725, 340)]), 10), 30,
                           ink=0.92, dry=0.45, profile=RISE, seed=2), t, 0.6) + 0.06
        t = cur.add(stroke(n, n, spline(q([(560, 330), (520, 450), (430, 610), (320, 770), (250, 860), (320, 872), (470, 762),
                                           (640, 662), (790, 662), (862, 760), (790, 868), (560, 925)]), 12), 32,
                           ink=0.92, dry=0.55, profile=((0, 0.5), (0.1, 1), (0.6, 0.85), (0.85, 0.9), (1, 0.12)), seed=3), t, 1.2)
        self.cursive = cur
        self.tC_end = t
        # D: the typeset hiragana
        self.a = typeset('あ', S, seed=11)
        self.a_halo = blur(self.a.dens, 5)
        self.tD = at('came hiragana')[0] - 0.6
        self.lbl = [typeset(c, 86, seed=30 + i, medians=False) for i, c in enumerate('ひらがな')]
        self.lbl_en, _ = text_layer('hiragana', FONT_EN_ITALIC, 46)
        self.t_lbl = at('came hiragana')[1] - 0.6
        # E: the row  安 → 草 → あ
        self.tE = at('The kanji an')[0] - 0.1
        self.cap = {k: text_layer(s, f, z)[0] for k, s, f, z in [
            ('an', 'an', FONT_EN_ITALIC, 42), ('cur', 'cursive', FONT_EN_ITALIC, 42), ('a', 'a', FONT_EN_ITALIC, 42),
            ('note', 'a simplified illustration', FONT_EN, 32)]}
        arrows = StrokeSet(W, H)
        for k, x in enumerate((385, 695)):
            tt = self.tE + 0.7 + 0.5 * k
            arrows.add(stroke(W, H, [(x - 40, 1195), (x + 40, 1190)], 6, ink=0.7, dry=0.4, profile=SWEEP, seed=40 + k), tt, 0.3)
            arrows.add(stroke(W, H, [(x + 20, 1172), (x + 42, 1190), (x + 20, 1210)], 5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=50 + k), tt + 0.25, 0.25)
        self.arrows = arrows
        # F: the Iroha poem, flowing down in vertical columns
        self.tF = at('Hiragana flourished')[0]
        cols = []
        for j, line in enumerate(IROHA):
            col = [typeset(c, 60, font=FONT_JP_FALLBACK, seed=60 + 10 * j + i, medians=False, pad=0.05) for i, c in enumerate(line)]
            cols.append(col)
        self.iroha = cols
        self.iroha_lbl, _ = text_layer('Iroha poem · Heian period (794–1185)', FONT_EN_ITALIC, 34)
        self.brush = brush_sprite(seed=5)

    # ------------------------------------------------------------ placement helper
    def put(self, fr, cam, dens, cx, cy, scale=1.0, color=INK, opacity=1.0):
        h, w = dens.shape
        A = np.array([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]], np.float32)
        M = cam.matrix()
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(dens, Mt), color, opacity)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.0)), (self.tB, (540, 900, 1.0)), (self.tC_end, (540, 830, 1.12)),
                                  (self.tE - 0.2, (540, 870, 1.08)), (self.tE + 1.4, (540, 960, 1.0)),
                                  (self.t1, (540, 960, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        CX, CY = 540, 760

        # A: the row of man'yogana, fading as 安 takes the stage
        fa = 1 - smooth((t - self.tB + 0.3) / 0.6)
        if t > self.tA and fa > 0:
            for i, (g, r) in enumerate(zip(self.row, self.row_r)):
                x = 156 + 192 * i
                f = (t - self.tA - 0.32 * i) / 0.45
                if f > 0:
                    self.put(fr, cam, g.at(f, soft=0.05), x, 790, opacity=fa)
                    self.put(fr, cam, r, x, 655, color=GREY, opacity=fa * smooth(f - 0.5))

        # B: 安 in full; C: highlighted, then ghosted under the cursive form
        fb = (t - self.tB) / (self.tB_end - self.tB)
        ghost = 1 - 0.78 * smooth((t - self.tC) / 0.8)
        gone = 1 - smooth((t - self.tD) / 0.7)
        mv = smooth((t - self.tE) / 1.0)
        if fb > 0 and gone > 0:
            d = self.an.at(min(fb, 1.05), soft=0.012, wet=0.45)
            d = bloom(d, t - self.tB_end, amount=0.1, halo=self.an_halo)
            hl = smooth((t - self.t_hl) / 0.5) * (1 - smooth((t - self.tC - 0.6) / 0.6))
            self.put(fr, cam, d * (1 - hl * np.maximum(self.roof, self.woman)), CX, CY, opacity=ghost * gone)
            if hl > 0:
                self.put(fr, cam, d * self.woman, CX, CY, color=VERM, opacity=hl * gone)
                self.put(fr, cam, d * self.roof, CX, CY, color=SEPIA, opacity=hl * gone * 0.9)
        cg = 1 - smooth((t - self.tD) / 1.1)
        if t > self.tC and mv < 1:
            self.put(fr, cam, self.cursive.render(t), CX, CY, opacity=cg * (1 - mv))

        # D: あ written over the fading cursive
        fd = (t - self.tD) / 1.5
        if fd > 0:
            d = bloom(self.a.at(min(fd, 1.05), soft=0.015, wet=0.45), t - self.tD - 1.5, amount=0.1, halo=self.a_halo)
            # E: あ moves into the third slot of the row
            cx = CX + (850 - CX) * ease_io(mv)
            cy = CY + (1190 - CY) * ease_io(mv)
            sc = 1 + (0.36 - 1) * ease_io(mv)
            self.put(fr, cam, d, cx, cy, sc)
        fl = (t - self.t_lbl) / 0.45
        if fl > 0:
            out = 1 - smooth((t - self.tE) / 0.6)
            for k, g in enumerate(self.lbl):
                self.put(fr, cam, g.at(fl - k * 0.3, soft=0.05), 400 + k * 93, 300, opacity=out)
            self.put(fr, cam, self.lbl_en, 540, 395, color=GREY, opacity=smooth(fl - 1.2) * out)

        # E: the row 安 → cursive → あ
        if mv > 0:
            self.put(fr, cam, self.an.dens, 230, 1190, 0.36, opacity=mv)
            self.put(fr, cam, self.cursive.render(99), 540, 1190, 0.36, opacity=mv)
            self.layer(fr, cam, self.arrows.render(t), color=GREY)
            for k, x in (('an', 230), ('cur', 540), ('a', 850)):
                self.put(fr, cam, self.cap[k], x, 1305, color=GREY, opacity=smooth((t - self.tE - 0.8) / 0.5))
            self.put(fr, cam, self.cap['note'], 540, 1350, color=GREY, opacity=0.8 * smooth((t - self.tE - 1.6) / 0.6))

        # F: the Iroha poem, flowing down in vertical columns
        self.tF = at('Hiragana flourished')[0]
        cols = []
        for j, line in enumerate(IROHA):
            col = [typeset(c, 60, font=FONT_JP_FALLBACK, seed=60 + 10 * j + i, medians=False, pad=0.05) for i, c in enumerate(line)]
            cols.append(col)
        self.iroha = cols
        self.iroha_lbl, _ = text_layer('Iroha poem · Heian period (794–1185)', FONT_EN_ITALIC, 34)
        self.brush = brush_sprite(seed=5)

    # ------------------------------------------------------------ placement helper
    def put(self, fr, cam, dens, cx, cy, scale=1.0, color=INK, opacity=1.0):
        h, w = dens.shape
        A = np.array([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]], np.float32)
        M = cam.matrix()
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(dens, Mt), color, opacity)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.0)), (self.tB, (540, 900, 1.0)), (self.tC_end, (540, 830, 1.12)),
                                  (self.tE - 0.2, (540, 870, 1.08)), (self.tE + 1.4, (540, 960, 1.0)),
                                  (self.t1, (540, 960, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        CX, CY = 540, 760

        # A: the row of man'yogana, fading as 安 takes the stage
        fa = 1 - smooth((t - self.tB + 0.3) / 0.6)
        if t > self.tA and fa > 0:
            for i, (g, r) in enumerate(zip(self.row, self.row_r)):
                x = 156 + 192 * i
                f = (t - self.tA - 0.32 * i) / 0.45
                if f > 0:
                    self.put(fr, cam, g.at(f, soft=0.05), x, 790, opacity=fa)
                    self.put(fr, cam, r, x, 655, color=GREY, opacity=fa * smooth(f - 0.5))

        # B: 安 in full; C: highlighted, then ghosted under the cursive form
        fb = (t - self.tB) / (self.tB_end - self.tB)
        ghost = 1 - 0.78 * smooth((t - self.tC) / 0.8)
        gone = 1 - smooth((t - self.tD) / 0.7)
        mv = smooth((t - self.tE) / 1.0)
        if fb > 0 and gone > 0:
            d = self.an.at(min(fb, 1.05), soft=0.012, wet=0.45)
            d = bloom(d, t - self.tB_end, amount=0.1, halo=self.an_halo)
            hl = smooth((t - self.t_hl) / 0.5) * (1 - smooth((t - self.tC - 0.6) / 0.6))
            self.put(fr, cam, d * (1 - hl * np.maximum(self.roof, self.woman)), CX, CY, opacity=ghost * gone)
            if hl > 0:
                self.put(fr, cam, d * self.woman, CX, CY, color=VERM, opacity=hl * gone)
                self.put(fr, cam, d * self.roof, CX, CY, color=SEPIA, opacity=hl * gone * 0.9)
        cg = 1 - smooth((t - self.tD) / 1.1)
        if t > self.tC and mv < 1:
            self.put(fr, cam, self.cursive.render(t), CX, CY, opacity=cg * (1 - mv))

        # D: あ written over the fading cursive
        fd = (t - self.tD) / 1.5
        if fd > 0:
            d = bloom(self.a.at(min(fd, 1.05), soft=0.015, wet=0.45), t - self.tD - 1.5, amount=0.1, halo=self.a_halo)
            # E: あ moves into the third slot of the row
            cx = CX + (850 - CX) * ease_io(mv)
            cy = CY + (1190 - CY) * ease_io(mv)
            sc = 1 + (0.36 - 1) * ease_io(mv)
            self.put(fr, cam, d, cx, cy, sc)
        fl = (t - self.t_lbl) / 0.45
        if fl > 0:
            out = 1 - smooth((t - self.tE) / 0.6)
            for k, g in enumerate(self.lbl):
                self.put(fr, cam, g.at(fl - k * 0.3, soft=0.05), 400 + k * 93, 300, opacity=out)
            self.put(fr, cam, self.lbl_en, 540, 395, color=GREY, opacity=smooth(fl - 1.2) * out)

        # E: the row 安 → cursive → あ
        if mv > 0:
            self.put(fr, cam, self.an.dens, 230, 1190, 0.36, opacity=mv)
            self.put(fr, cam, self.cursive.render(99), 540, 1190, 0.36, opacity=mv)
            self.layer(fr, cam, self.arrows.render(t), color=GREY)
            for k, x in (('an', 230), ('cur', 540), ('a', 850)):
                self.put(fr, cam, self.cap[k], x, 1305, color=GREY, opacity=smooth((t - self.tE - 0.8) / 0.5))
            self.put(fr, cam, self.cap['note'], 540, 1350, color=GREY, opacity=0.8 * smooth((t - self.tE - 1.6) / 0.6))

        # F: the Iroha poem, flowing down in vertical columns (right to left), very quiet
        ff = t - self.tF
        if ff > 0:
            for j, col in enumerate(self.iroha):
                x = 720 - 120 * j
                for i, g in enumerate(col):
                    f = (ff - j * 0.55 - i * 0.12) / 0.3
                    if f > 0:
                        self.put(fr, cam, g.at(f, soft=0.05), x, 270 + i * 64, color=SEPIA, opacity=0.55)
            self.put(fr, cam, self.iroha_lbl, 540, 790, color=SEPIA, opacity=0.9 * smooth((ff - 1.2) / 0.8))

        # the brush, while 安 and the cursive are written
        if self.tB - 0.6 <= t <= self.tC_end + 0.8:
            self._brush(fr, cam, t, CX, CY)
        return fr.img

    def _brush(self, fr, cam, t, CX, CY):
        ox, oy = CX - self.an.w / 2, CY - self.an.h / 2
        if t < self.tC:
            f = (t - self.tB) / (self.tB_end - self.tB)
            x, y, k, u, moving = self.an.brush_pos(np.clip(f, 0, 1))
            lift = 0 if moving and 0 < f < 1 else 30
            if f < 0:
                lift = 40 + 500 * (-f) * 3
        else:
            # follow the cursive strokes: the latest stroke being painted
            moving, x, y, lift = False, None, None, 30
            for st, s0, du, _, _ in self.cursive.items:
                if s0 <= t <= s0 + du:
                    u = (t - s0) / du
                    on = st.time <= u
                    ys, xs = np.nonzero((np.abs(st.time - u) < 0.03) & (st.dens > 0.3))
                    if len(xs):
                        x, y, moving, lift = st.x0 + xs.mean(), st.y0 + ys.mean(), True, 0
            if x is None:
                last = [it for it in self.cursive.items if it[1] <= t]
                st = (last[-1] if last else self.cursive.items[0])[0]
                ys, xs = np.nonzero(st.time > 0.95)
                x, y = (st.x0 + xs.mean(), st.y0 + ys.mean()) if len(xs) else (self.an.w / 2, self.an.h / 2)
                if t > self.tC_end:
                    lift = 30 + 600 * ease_out((t - self.tC_end) / 0.8)
        M = cam.matrix()
        sx, sy = M[:, :2] @ np.array([ox + x, oy + y]) + M[:, 2]
        draw_sprite(fr, *self.brush, sx, sy - lift * 0.5, -0.32, scale=0.62 * cam.zoom, shadow=0.3, lift=lift)

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the camera follows a curving stroke from the kanji page down onto the next page"""
        if not hasattr(self, '_seam'):
            c = np.zeros((2 * H, W), np.float32)
            st = stroke(W, 2 * H, spline([(820, 1380), (900, 1600), (760, 1900), (420, 2080), (260, 2300), (380, 2520)], 14),
                        44, ink=0.9, dry=0.55, profile=SWEEP, seed=77)
            from engine import paste
            paste(c, st)
            T = np.ones_like(c) * 2
            hh, ww = st.dens.shape
            T[st.y0:st.y0 + hh, st.x0:st.x0 + ww] = np.where(st.dens > 0.01, st.time, 2)
            self._seam = (c, T)
        c, T = self._seam
        e = ease_io(f)
        off = int(e * H)
        big = np.concatenate([prev_img, cur_img], 0)
        stroke_now = reveal_time(c, T, f * 1.4, soft=0.05, wet=0.4) * (1 - smooth((f - 0.75) / 0.25))
        fr = Frame(big[off:off + H])
        fr.ink(stroke_now[off:off + H], INK)
        return fr.img

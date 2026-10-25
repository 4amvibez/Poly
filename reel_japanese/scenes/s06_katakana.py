"""Scene 6 — katakana.  A Buddhist sutra page (the opening of the Heart Sutra, in Xuanzang's Chinese) gains
small vermilion reading notes squeezed beside the columns.  Then 伊 is written; its left component 亻 is ringed
in vermilion, isolated, and becomes the katakana イ.  Two more standard derivations follow, crisp and compact:
加 -> カ (from 力) and 多 -> タ (from 夕)."""
import cv2
import numpy as np

from art import brush_sprite, draw_sprite
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP_FALLBACK, GREY, H, INK, SEPIA, VERM, W, Cam, Frame, StrokeSet,
                    blur, bloom, ease_io, ease_out, keyframes, noise, ramp, reveal_time, smooth, spline, sstep, stroke,
                    text_layer, warp, wipe_mix)
from glyphs import calligraphy, typeset
from poly_ink.brush import PRESS_DOWN, SWEEP
from scenes.base import SceneBase
from timeline import at

SUTRA = ['觀自在菩薩', '行深般若波羅蜜多時', '照見五蘊皆空', '度一切苦厄']
GLOSS = [(1, 2, 'ニ'), (1, 8, 'ノ'), (2, 1, 'シテ'), (2, 5, 'ナリト'), (3, 0, 'シ'), (3, 4, 'ヲ')]   # (column, char, note)
S = 500


class Scene(SceneBase):
    paper_seed = 16
    paper_stains = 0.1

    def build(self):
        # A: sutra page with notes
        page = np.zeros((H, W), np.float32)
        self.char_xy = {}
        for j, col in enumerate(SUTRA):
            x = 780 - 140 * j
            for i, c in enumerate(col):
                g = typeset(c, 76, font=FONT_JP_FALLBACK, seed=200 + 10 * j + i, pad=0.05, medians=False)
                y = 270 + i * 88
                page[y:y + g.h, x - g.w // 2:x - g.w // 2 + g.w] = np.maximum(page[y:y + g.h, x - g.w // 2:x - g.w // 2 + g.w], g.dens)
                self.char_xy[(j, i)] = (x, y + g.h // 2)
        self.page = page
        from engine import wash_map
        self.page_T = wash_map(H, W, None, direction=(-0.3, 1), noise_amp=0.25, noise_scale=80, seed=3)
        self.gloss = []
        for k, (j, i, note) in enumerate(GLOSS):
            gs = [typeset(c, 34, font=FONT_JP_FALLBACK, seed=300 + k * 5 + m, pad=0.04, medians=False) for m, c in enumerate(note)]
            x, y = self.char_xy[(j, i)]
            self.gloss.append((gs, x + 58, y - 20))
        self.tA = self.t0 + 0.6
        self.t_gloss = at('quick notes')[0] - 0.2

        # B/C: 伊, its component, イ
        self.i_k = calligraphy('伊', S, seed=12)
        self.i_halo = blur(self.i_k.dens, 5)
        self.comp = self.i_k.stroke_mask([0, 1])
        self.rest = self.i_k.stroke_mask([2, 3, 4, 5])
        self.tB = at('They took')[0] - 0.3
        self.tB_end = self.tB + 1.9
        self.t_ring = at('left side')[0] - 0.15
        self.t_iso = at('and created')[0] - 0.4
        self.t_kana = at('katakana', 0)[0] - 0.2
        # ring around the component: a thin vermilion brush loop
        n = self.i_k.w
        ys, xs = np.nonzero(self.comp > 0.3)
        cx, cy = xs.mean(), ys.mean()
        rx, ry = (xs.max() - xs.min()) / 2 + 26, (ys.max() - ys.min()) / 2 + 24
        ring = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(-2.2, -2.2 + 2 * np.pi * 1.04, 40)]
        rs = StrokeSet(n, n)
        rs.add(stroke(n, n, ring, 6, ink=0.9, dry=0.35, profile=((0, 0.4), (0.1, 1), (0.9, 0.9), (1, 0.2)), seed=4), self.t_ring, 0.7)
        self.ring = rs
        self.comp_c = (cx, cy)
        self.ka = typeset('イ', 430, seed=14)
        self.lbl = [typeset(c, 86, seed=40 + i, medians=False) for i, c in enumerate('カタカナ')]
        self.lbl_en, _ = text_layer('katakana', FONT_EN_ITALIC, 46)

        # D: more derivations
        self.t_sharp = at('Sharp')[0] - 0.1
        self.t_comp = at('compact')[0] - 0.1
        self.pairs = []
        for k, (kan, idx, kana, t) in enumerate([('加', [0, 1], 'カ', self.t_comp), ('多', [0, 1, 2], 'タ', at('precise')[0] - 0.15)]):
            g = calligraphy(kan, 150, seed=50 + k)
            self.pairs.append((g, g.stroke_mask(idx), typeset(kana, 130, seed=60 + k, medians=False), t))
        arrows = StrokeSet(W, H)
        for k, (x, y, t) in enumerate([(545, 760, self.t_sharp + 0.1), (300, 1215, self.t_comp + 0.25), (780, 1215, at('precise')[0] + 0.1)]):
            arrows.add(stroke(W, H, [(x - 34, y + 2), (x + 34, y - 2)], 5, ink=0.7, dry=0.4, profile=SWEEP, seed=70 + k), t, 0.22)
            arrows.add(stroke(W, H, [(x + 16, y - 16), (x + 36, y - 1), (x + 16, y + 14)], 4.5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=80 + k), t + 0.18, 0.18)
        self.arrows = arrows
        self.brush = brush_sprite(seed=8)

    def put(self, fr, cam, dens, cx, cy, scale=1.0, color=INK, opacity=1.0):
        if opacity <= 0:
            return
        h, w = dens.shape
        A = np.array([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]], np.float32)
        M = cam.matrix()
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(dens, Mt), color, opacity)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (560, 820, 1.18)), (self.t_gloss + 1.6, (600, 760, 1.32)),
                                  (self.tB - 0.1, (560, 850, 1.05)), (self.t_ring + 0.4, (500, 800, 1.18)),
                                  (self.t_kana + 0.8, (540, 840, 1.05)), (self.t1, (540, 950, 1.0))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        # A: the sutra page fades back as 伊 is written
        pa = 1 - smooth((t - self.tB + 0.5) / 0.7)
        fa = ramp(t, self.tA - 0.6, self.tA + 1.2)
        if fa > 0 and pa > 0:
            self.layer(fr, cam, reveal_time(self.page, self.page_T, fa * 1.1, soft=0.08, wet=0.2), opacity=pa)
            for k, (gs, x, y) in enumerate(self.gloss):
                f = (t - self.t_gloss - 0.32 * k) / 0.3
                for m, g in enumerate(gs):
                    if f - m * 0.5 > 0:
                        self.put(fr, cam, g.at(f - m * 0.5, soft=0.06), x, y + m * 36, color=VERM, opacity=pa)

        CX, CY = 540, 760
        fb = (t - self.tB) / (self.tB_end - self.tB)
        iso = smooth((t - self.t_iso) / 0.6)
        kana = smooth((t - self.t_kana) / 0.7)
        pair = smooth((t - self.t_sharp) / 0.8)
        if fb > 0:
            d = bloom(self.i_k.at(min(fb, 1.05), soft=0.012, wet=0.45), t - self.tB_end, amount=0.1, halo=self.i_halo)
            hl = smooth((t - self.t_ring) / 0.4)
            # the rest of 伊 ghosts away; the component stays, in vermilion, and slides to centre
            dx = (CX - (CX - self.i_k.w / 2 + self.comp_c[0])) * ease_io(iso)
            px = (CX + dx) * (1 - pair) + (CX - 210) * pair
            sc = 1 - 0.4 * pair
            rest_op = (1 - iso) if pair == 0 else 0.45 * pair
            self.put(fr, cam, d * self.rest, CX - 210 * pair, CY, sc, opacity=rest_op)
            comp_op = (1 - kana) if pair == 0 else pair
            self.put(fr, cam, d * self.comp * (1 - hl), px, CY, sc, opacity=comp_op)
            self.put(fr, cam, d * self.comp * hl, px, CY, sc, color=VERM, opacity=comp_op)
            ro = (1 - smooth((t - self.t_iso) / 0.4))
            if ro > 0 and t > self.t_ring:
                self.put(fr, cam, self.ring.render(t), CX, CY, color=VERM, opacity=ro)
        if t > self.t_kana:
            # イ written where the component now stands, then moving to the right of 伊
            fk = (t - self.t_kana) / 0.8
            d = self.ka.at(min(fk, 1.05), soft=0.02, wet=0.4)
            self.put(fr, cam, d, CX + 230 * pair, CY + 10, 1 - 0.4 * pair)
        fl = (t - self.t_kana - 0.5) / 0.4
        if fl > 0:
            for k, g in enumerate(self.lbl):
                self.put(fr, cam, g.at(fl - k * 0.25, soft=0.05), 400 + k * 93, 300)
            self.put(fr, cam, self.lbl_en, 540, 395, color=GREY, opacity=smooth(fl - 1.2))

        # D: 加 -> カ, 多 -> タ
        self.layer(fr, cam, self.arrows.render(t), color=GREY)
        for k, (g, m, kn, ts) in enumerate(self.pairs):
            f = (t - ts) / 0.35
            if f <= 0:
                continue
            x = 300 if k == 0 else 780
            d = g.at(min(f, 1.05), soft=0.02)
            self.put(fr, cam, d * (1 - m), x - 100, 1210, opacity=0.45)
            self.put(fr, cam, d * m, x - 100, 1210, color=VERM)
            self.put(fr, cam, kn.at((f - 0.8) * 1.5, soft=0.03), x + 105, 1210)
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the flowing cursive line sweeps on and turns into a sharp, angular cut that opens the next page"""
        if not hasattr(self, '_cut'):
            flow = stroke(W, H, spline([(-40, 1250), (260, 1120), (520, 1180), (760, 1060), (1120, 980)], 14), 30,
                          ink=0.9, dry=0.5, profile=SWEEP, seed=91)
            ang = [(1120, 980), (820, 700), (640, 1000), (380, 640), (180, 880), (-40, 560)]
            sharp = stroke(W, H, ang, 16, ink=0.95, dry=0.25, smooth=2, profile=((0, 1), (0.95, 1), (1, 0.2)), seed=92)
            from engine import paste
            A = np.zeros((H, W), np.float32); Ta = np.ones((H, W), np.float32) * 2
            B = np.zeros((H, W), np.float32); Tb = np.ones((H, W), np.float32) * 2
            for st, buf, T in ((flow, A, Ta), (sharp, B, Tb)):
                paste(buf, st)
                hh, ww = st.dens.shape
                T[st.y0:st.y0 + hh, st.x0:st.x0 + ww] = np.where(st.dens > 0.01, st.time, 2)
            core = ((A + B) > 0.3).astype(np.uint8)
            dist = cv2.distanceTransform(1 - core, cv2.DIST_L1, 3).astype(np.float32)
            dmap = np.clip(dist / dist.max(), 0, 1) ** 0.75
            self._cut = (A, Ta, B, Tb, dmap)
        A, Ta, B, Tb, dmap = self._cut
        fa = Frame(prev_img)
        fa.ink(reveal_time(A, Ta, f / 0.35, soft=0.04, wet=0.4), INK)
        fa.ink(reveal_time(B, Tb, (f - 0.3) / 0.25, soft=0.03, wet=0.3), INK)
        g = np.clip((f - 0.5) / 0.5, 0, 1)
        if g <= 0:
            return fa.img
        out = wipe_mix(fa.img, cur_img, dmap, ease_io(g) * 1.1, soft=0.02, edge=0.12)
        return out

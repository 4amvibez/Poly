"""Scene 5 — "Here's another example: にわにはにわにわとりがいる。  Now look at it with kanji: 庭には二羽鶏がいる。
'There are two chickens in the garden.'  See how the kanji helps make the meaning clearer?"
The hiragana line is written as the Japanese voice reads it, and three thought clouds pop up on the three にわ sounds:
にわ? (a garden), にわ? (two feathers: a counter for birds), にわとり? (a chicken).  Then the kanji line takes its
place, colour-coded as it is spoken: 庭 garden, 二羽 two (birds), 鶏 chicken; a garden is painted below, and in the
breath before "There are two chickens…" two hens pop into it (wood knock) and peck, counted 一羽, 二羽.  On "See how…"
the page splits: without kanji (grey, a ？) and with kanji (bright, a lit bulb)."""
import cv2
import numpy as np

import kit
import pics
from backdrop import backdrop, draw_backdrop
from engine import (FONT_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, cached,
                    ease_io, ease_out, keyframes, lerp, noise, smooth, spline, stroke, text_layer)
from glyphs import typeset
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, wash as pwash
from scenes.base import SceneBase
from sentence import BLUE, GOLD
from timeline import JA, at, quiet_gap

SY = 380
HIRA, KAN = 'にわにはにわにわとりがいる。', '庭には二羽鶏がいる。'
CH = [(0.02, 0.80), (1.05, 1.48), (1.70, 2.70)]             # にわには | にわ | にわとりがいる (s into the Japanese take)
HM = [[0, 1, 2, 3], [4, 5], list(range(6, 14))]
KM = [[0, 1, 2], [3, 4], list(range(5, 10))]
KCOL = {0: kit.GREEN, 3: VERM, 4: VERM, 5: BLUE}
CLOUDS = [(190, 760, 'にわ？', 'garden'), (540, 700, 'にわ？', 'feathers'), (890, 760, 'にわとり？', 'chicken')]
HENS = [(400, 1215, 1), (700, 1250, -1)]
IVORY = np.array([0.97, 0.94, 0.86], np.float32)


class Scene(SceneBase):
    paper_seed = 85

    def build(self):
        self.B = backdrop(seed=240, branch='right', ground=1520)
        self.hira = kit.Row(HIRA, 66, seed=200)
        self.kan = kit.Row(KAN, 92, seed=230)
        a, b = JA['niwa_kana']['start'], JA['niwa_kanji']['start']
        self.sp_h = [(a + p, a + q) for p, q in CH]
        self.sp_k = [(b + p, b + q) for p, q in CH]
        self.t_out = at('Now look at it')[0]
        c, n = quiet_gap(JA['niwa_kanji']['end'] - 0.1, at('There are two')[0] + 0.1)
        self.t_hit = c - n / 2 + 0.08                                # sync: the hens pop in (+ knock)
        self.t_two = at('two chickens')[0]
        self.t_split = at('See how')[0] - 0.15

        def paint():
            p = Painting(W, H, seed=12)
            r = p.ridge(1080, peaks=[(200, 60, 240), (760, 90, 260)], rough=8, seed=13)
            p.wash(r, dens=0.12, decay=80, tex=60, edge=0.25, mist_y=1110, mist_h=30, seed=14)
            ground = np.zeros((H, W), np.float32)
            yy = np.arange(H, dtype=np.float32)[:, None]
            ground[:] = np.clip((yy - 1110) / 40, 0, 1) * np.clip((1330 - yy) / 60, 0, 1) * 0.22
            ground *= 0.7 + 0.3 * noise(H, W, 40, 3, 15)
            ss = StrokeSet(W, H)
            from art import ink_tree
            ink_tree(ss, 110, 1120, 1.5, 0, seed=16, dur=0.01, ink=0.75)
            fe = pics.fence(W, 260, seed=3)
            F = np.zeros((H, W), np.float32)
            F[990:1250] = fe
            hens = {}
            for k, (x, y, f) in enumerate(HENS):
                for j, pk in enumerate((0.0, 0.5, 1.0)):
                    ink, comb = pics.chicken(peck=pk, facing=f, seed=10 * k)
                    hens[f'hen{k}_{j}'], hens[f'comb{k}_{j}'] = ink, comb
            cl, cf = pics.cloud(330, 300, seed=21, tail=(0, 1))
            return {'land': (p.D + ground).astype(np.float32), 'tree': ss.render(9).astype(np.float32), 'fence': F,
                    'feathers': pics.feathers(), 'cl': cl, 'cf': cf, 'mini_fence': pics.fence(300, 140, seed=4)} | hens
        self.A = cached('s05_paint_v2', paint)
        self.cloud_lbl = [kit.Row(s, 42, seed=260 + k) for k, (_, _, s, _) in enumerate(CLOUDS)]
        self.mini_chicken = cached('s05_mini_v1', lambda: dict(zip(('i', 'c'), pics.chicken(seed=40))))
        # labels under the kanji
        self.k_lbl = [(0, 1, 'garden', kit.GREEN), (3, 4, 'two (birds)', VERM), (5, 5, 'chicken', BLUE)]
        self.k_lbl_t = [text_layer(s, FONT_EN_ITALIC, 44)[0] for _, _, s, _ in self.k_lbl]
        arr = StrokeSet(W, H)
        for k, (i0, i1, _, _) in enumerate(self.k_lbl):
            x = 540 + (self.kan.x[i0] + self.kan.x[i1]) / 2
            lx = [200, 540, 880][k]
            kit.arrow(arr, (x, SY + 64), (lx, SY + 150), self.sp_k[k][0] + 0.15, 0.25, 5, seed=60 + 3 * k, red=True)
        self.arr = arr
        self.count = [kit.Row(s, 64, seed=290 + k) for k, s in enumerate(('一羽', '二羽'))]
        # the split
        self.tag_l, _ = text_layer('Without kanji', FONT_EN_BOLD, 46)
        self.tag_r, _ = text_layer('With kanji', FONT_EN_BOLD, 46)
        self.lines_l = [kit.Row(s, 62, seed=300 + k) for k, s in enumerate(('にわには', 'にわにわとり', 'がいる。'))]
        self.lines_r = [kit.Row(s, 70, seed=320 + k) for k, s in enumerate(('庭には', '二羽鶏が', 'いる。'))]
        self.q, _ = text_layer('？', FONT_JP, 150)
        self.bulb = cached('s05_bulb_v1', _bulb)
        from engine import FONT_BRUSH_EN
        self.another, _ = text_layer('another example…', FONT_BRUSH_EN, 80)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 955, 1.02)), (self.t_out, (540, 960, 1.0)), (self.t_hit, (540, 975, 1.0)),
                                  (self.t_split, (540, 960, 1.0)), (self.t1, (540, 960, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        split = smooth((t - self.t_split) / 0.45)
        keep = 1 - split
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8) * keep)
        kit.put(fr, cam, self.another, 540, 230, VERM, smooth((t - self.t0 - 0.2) / 0.4) * (1 - smooth((t - self.t_out) / 0.5)))
        # 1. the hiragana line and its three guesses
        ho = 1 - smooth((t - self.t_out) / 0.5)
        if ho > 0 and keep > 0:
            self.hira.draw(fr, cam, 540, SY, kit.chunk_reveal(t, self.sp_h, HM), op=lambda i: ho)
            for k, (x, y, lbl, pic) in enumerate(CLOUDS):
                f = (t - self.sp_h[k][0] - 0.1) / 0.3
                if f <= 0:
                    continue
                sc = 1.12 * (0.6 + 0.4 * ease_out(min(f, 1)))
                o = smooth(f) * ho
                bob = 6 * np.sin(t * 2.4 + k)
                kit.put(fr, cam, A['cf'], x, y + bob, kit.PAPER_C, 0.96 * o, sc, over=True)
                kit.put(fr, cam, A['cl'], x, y + bob, INK, o, sc)
                self.cloud_lbl[k].draw(fr, cam, x, y - 70 * sc + bob, lambda i: 1.0, op=lambda i: o, scale=sc)
                if pic == 'garden':
                    kit.put(fr, cam, A['mini_fence'], x, y + 30 * sc + bob, INK, o, 0.85 * sc)
                elif pic == 'feathers':
                    kit.put(fr, cam, A['feathers'], x, y + 30 * sc + bob, INK, o, 0.95 * sc)
                else:
                    kit.put(fr, cam, self.mini_chicken['i'], x, y + 34 * sc + bob, INK, o, 0.62 * sc)
                    kit.put(fr, cam, self.mini_chicken['c'], x, y + 34 * sc + bob, VERM, o, 0.62 * sc)
        # 2. the kanji line, colour-coded, with the garden below
        if keep > 0:
            col = lambda i: KCOL.get(i, INK)
            self.kan.draw(fr, cam, 540, SY, kit.chunk_reveal(t, self.sp_k, KM), color=col, op=lambda i: keep)
            self.layer(fr, cam, self.arr.render(t, red=True), color=VERM, opacity=keep)
            for k, ((_, _, _, c), g) in enumerate(zip(self.k_lbl, self.k_lbl_t)):
                kit.put(fr, cam, g, [200, 540, 880][k], SY + 190, c * 0.85, smooth((t - self.sp_k[k][0] - 0.35) / 0.3) * keep)
            fg = smooth((t - self.sp_k[0][0]) / 0.8) * keep
            self.layer(fr, cam, A['land'], opacity=fg, depth=0.95)
            self.layer(fr, cam, A['tree'], opacity=fg)
            self.layer(fr, cam, A['fence'], opacity=0.9 * fg)
            for k, (x, y, f) in enumerate(HENS):
                g = (t - self.t_hit - 0.1 * k) / 0.25
                if g <= 0:
                    continue
                ph = (t * 1.3 + 0.5 * k) % 2
                j = 0 if ph < 1.2 else (1 if ph < 1.35 or ph > 1.85 else 2)
                pop = 1.3 * (1 + 0.25 * np.exp(-g * 3))
                hop = -30 * np.exp(-g * 4) * np.sin(min(g, 1) * np.pi)
                kit.put(fr, cam, A[f'hen{k}_{j}'], x, y + hop, INK, smooth(g) * keep, pop)
                kit.put(fr, cam, A[f'comb{k}_{j}'], x, y + hop, VERM, smooth(g) * keep, pop)
                cc = smooth((t - self.t_two - 0.25 * k) / 0.3) * keep
                self.count[k].draw(fr, cam, x, y - 200, lambda i: 1.0, color=lambda i: VERM, op=lambda i: cc)
        # 3. without kanji / with kanji
        if split > 0:
            sl = ease_out(split)
            dark = np.zeros((H, W), np.float32)
            dark[:, :int(540 * sl)] = 1
            fr.over(np.array([0.55, 0.53, 0.5], np.float32), blur(dark, 3), 0.6 * split)
            fr.over(np.array([0.99, 0.97, 0.9], np.float32), np.clip(1 - blur(dark, 3), 0, 1), 0.5 * split)
            line = np.zeros((H, W), np.float32)
            line[260:1360, 538:542] = 1
            self.layer(fr, cam, blur(line, 1), opacity=0.8 * split)
            for x, tg, dk in ((270, self.tag_l, True), (810, self.tag_r, False)):
                plate = kit.box(80, tg.shape[1] + 60, 4)
                kit.put(fr, cam, plate, x, 330, INK if dk else VERM, 0.95 * split, over=True)
                kit.put(fr, cam, tg, x, 330, IVORY, split, over=True)
            for k, R in enumerate(self.lines_l):
                R.draw(fr, cam, 270, 560 + 100 * k, lambda i: 1.0, color=lambda i: np.array([0.25, 0.24, 0.23], np.float32),
                       op=lambda i: smooth((t - self.t_split - 0.2 - 0.12 * k) / 0.3))
            for k, R in enumerate(self.lines_r):
                cmap = {0: {0: kit.GREEN}, 1: {0: VERM, 1: VERM, 2: BLUE}}.get(k, {})
                R.draw(fr, cam, 810, 560 + 105 * k, lambda i: 1.0, color=lambda i, c=cmap: c.get(i, INK),
                       op=lambda i: smooth((t - self.t_split - 0.3 - 0.12 * k) / 0.3))
            qf = smooth((t - self.t_split - 0.6) / 0.3)
            kit.put(fr, cam, self.q, 270, 1020 + 8 * np.sin(t * 3), INK, qf, 1 + 0.2 * np.sin(t * 2))
            bf = smooth((t - self.t_split - 0.8) / 0.3)
            glow = 0.6 + 0.4 * np.sin(t * 5) ** 2
            kit.put(fr, cam, self.bulb['glow'], 810, 1000, GOLD, 0.7 * bf * glow, over=False)
            kit.put(fr, cam, self.bulb['bulb'], 810, 1000, INK, bf)
        return fr.img


def _bulb():
    w, h = 260, 320
    ss = StrokeSet(w, h)
    a = np.linspace(-0.85 * np.pi, 0.85 * np.pi, 30) - np.pi / 2
    glass = [(130 + 80 * np.cos(u), 120 + 80 * np.sin(u)) for u in a]
    ss.add(stroke(w, h, glass + [(100, 214), (160, 214)][::-1][:0] + [(100, 200)], 6, ink=0.9, dry=0.35, profile=PRESS_DOWN, seed=1), 0, 1)
    ss.add(stroke(w, h, [(160, 200), (154, 186), glass[-1]], 6, ink=0.9, dry=0.35, profile=PRESS_DOWN, seed=2), 0, 1)
    for k, y in enumerate((214, 234, 254)):
        ss.add(stroke(w, h, [(100, y), (160, y)], 7, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=3 + k), 0, 1)
    ss.add(stroke(w, h, spline([(112, 190), (118, 130), (130, 150), (142, 130), (148, 190)], 10), 3.5, ink=0.8, dry=0.3, profile=PRESS_DOWN, seed=7), 0, 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.exp(-((xx - 130) ** 2 + (yy - 120) ** 2) / (2 * 70 ** 2)).astype(np.float32)
    return {'bulb': ss.render(9).astype(np.float32), 'glow': glow}

"""Scene 2 — the experiment.  "Look at this sentence: 母は花が好き。  It means, 'Mother likes flowers.'  Now, let's
remove the kanji, and write everything in hiragana: ははははながすき。"
A hanging scroll unrolls; the sentence is brushed onto it chunk by chunk as Raj says it in Japanese.  On the meaning,
brush arrows run from 母, 好 and 花 to little paintings below: a mother (mother), a heart (likes), a flower (flowers).
Then, in the breath after "hiragana", the three kanji are wiped away in a puff of ink (bowl), the kana that are left
slide apart, and the gaps are written in hiragana as the Japanese voice reads the sentence again.  A beat later the
new line gives a little jolt and a vermilion ？ pops up over it: it suddenly looks so confusing."""
import numpy as np

import kit
import pics
from backdrop import backdrop, draw_backdrop
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_io,
                    ease_out, keyframes, lerp, smooth, text_layer)
from scenes.base import SceneBase
from sentence import BLUE, PINK
from timeline import JA, at, quiet_gap

SY = 700
OLD, NEW = '母は花が好き。', 'ははははながすき。'
MOVE = {1: 2, 3: 5, 5: 7, 6: 8}            # old index -> new index for the kana that stay
FILL = [0, 1, 3, 4, 6]                     # new kana written into the gaps the kanji left
SIZE = 90
# where the speech falls inside the Japanese takes (s from the take's start): 母は | 花が | 好き, and the kana-only line
# read literally, one syllable at a time: は は は は な が す き
CH = [(0.0, 0.53), (0.63, 1.15), (1.16, 1.48)]
SYL = [0.0, 0.45, 0.84, 1.18, 1.61, 1.8, 2.02, 2.11]


class Scene(SceneBase):
    paper_seed = 82

    def build(self):
        self.B = backdrop(seed=210, branch='left', ground=1420)
        self.old = kit.Row(OLD, SIZE, seed=100)
        self.new = kit.Row(NEW, SIZE, seed=120)
        a = JA['haha_kanji']['start']
        self.sp_old = [(a + p, a + q) for p, q in CH]
        b = JA['haha_kana']['start']
        self.t_syl = [b + s for s in SYL]                          # when each kana of ははははながすき is read
        self.sp_new = [(self.t_syl[0], self.t_syl[2]), (self.t_syl[3], self.t_syl[5]), (self.t_syl[6], self.t_syl[7])]
        self.dot = kit.blur(np.pad(np.ones((10, 46), np.float32), 4), 2)
        self.t_unroll = at('Look at this')[0] - 0.1
        self.t_mean = [at('Mother')[0], at('likes')[0], at('flowers')[0]]
        c, n = quiet_gap(at('everything in hiragana')[1] - 0.2, JA['haha_kana']['start'])
        self.t_hit = c - n / 2 + 0.08                              # sync: the kanji are wiped away (+ bowl)
        self.t_slide = self.t_hit + 0.15
        c2, n2 = quiet_gap(JA['haha_kana']['end'] - 0.1, at('Wait')[0] + 0.1)
        self.t_huh = c2 - n2 / 2 + 0.2                             # sync: the jolt and the ？ (+ wood knock)
        # the scroll
        self.sheet = kit.box(330, 980, 3)
        self.prim = kit.rim(330, 980, 18, 3)
        rod = np.zeros((34, 1040), np.float32)
        rod[6:-6, 10:-10] = 1
        self.rod = blur(rod, 1.5)

        def paint():
            fl, st = pics.flower()
            return {'mother': pics.mother(), 'flower': fl, 'stem': st, 'heart': pics.heart()}
        self.P = cached('s02_pics_v1', paint)
        # arrows and labels: 母 -> mother, 好 -> likes, 花 -> flowers
        xs = [540 + self.old.x[i] for i in (0, 4, 2)]
        self.items = [(xs[0], 'mother', 'mother', VERM, 0), (xs[1], 'likes', 'heart', BLUE, 1), (xs[2], 'flowers', 'flower', PINK, 2)]
        self.lab_x = [200, 860, 530]
        arr = StrokeSet(W, H)
        for k, (x, txt, pic, col, j) in enumerate(self.items):
            kit.arrow(arr, (x, SY + 80), (self.lab_x[k], SY + 250), self.t_mean[k] - 0.1, 0.3, 6, seed=40 + 3 * k, red=True)
        self.arr = arr
        self.labels = [text_layer(txt, FONT_EN_ITALIC, 62)[0] for _, txt, _, _, _ in self.items]
        self.puff = []
        r = np.random.default_rng(5)
        for i in (0, 2, 4):
            for _ in range(26):
                self.puff.append((540 + self.old.x[i] + r.normal(0, 30), SY + r.normal(0, 30), r.uniform(0, 2 * np.pi), r.uniform(60, 220), r.uniform(4, 11)))
        self.q, _ = text_layer('？', FONT_JP, 130)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.03)), (self.t_mean[0], (540, 975, 1.0)), (self.t_hit, (540, 960, 1.0)),
                                  (self.t1, (540, 950, 1.04))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        # the scroll unrolls from the top rod
        u = ease_out((t - self.t_unroll) / 0.7)
        if u > 0:
            sy = 0.05 + 0.95 * u
            cy = SY - 165 * (1 - sy)
            kit.card(fr, cam, self.sheet, self.prim, 540, cy, 1.0, 1.0, sy)
            for y in (SY - 175, cy + 165 * sy + 8):
                kit.put(fr, cam, self.rod, 540, y, np.array([0.3, 0.19, 0.11], np.float32), 0.95)
        # the sentence with kanji, then without
        gone = smooth((t - self.t_hit) / 0.18)
        slide = ease_io((t - self.t_slide) / 0.5)
        rv_old = kit.chunk_reveal(t, self.sp_old, [[0, 1], [2, 3], [4, 5, 6]])
        isk = lambda i: OLD[i] in '母花好'

        def op_old(i):
            return (1 - gone) if isk(i) else 1.0

        def off_old(i):
            if i in MOVE:
                return ((self.new.x[MOVE[i]] - self.old.x[i]) * slide, 0)
            return (0, 0)
        jolt = np.exp(-max(t - self.t_huh, 0) * 6) * (t > self.t_huh)
        shake = (8 * np.sin((t - self.t_huh) * 60) * jolt, 0)
        tint = {0: VERM, 2: PINK, 4: BLUE}
        tk = [smooth((t - self.t_mean[k] + 0.05) / 0.3) for k in range(3)]
        tw = {0: tk[0], 4: tk[1], 2: tk[2]}
        col = lambda i: lerp(INK, tint[i] * 0.8, tw[i]) if i in tint else INK
        self.old.draw(fr, cam, 540 + shake[0], SY, rv_old, op=op_old, offset=off_old, color=col)
        # the kana-only line is read one kana at a time: each is written (or lit) as it is said
        rv_new = lambda i: float(np.clip((t - self.t_syl[i] + 0.05) / 0.2, 0, 1.2)) if i < 8 else 0.0
        self.new.draw(fr, cam, 540 + shake[0], SY, lambda i: rv_new(i) if i in FILL else 0.0)
        for i, ts in enumerate(self.t_syl):
            p = (t - ts) / 0.35
            if 0 < p < 1.6:
                kit.put(fr, cam, self.dot, 540 + shake[0] + self.new.x[i], SY + 70, VERM, smooth(p * 3) * (1 - smooth(p - 0.6)))
        # ink puff where the kanji were
        db = t - self.t_hit
        if 0 < db < 1.2:
            import cv2
            d = np.zeros((H, W), np.float32)
            e = ease_out(min(db / 0.8, 1))
            for x0, y0, a, dist, sz in self.puff:
                cv2.circle(d, (int(x0 + np.cos(a) * dist * e), int(y0 + np.sin(a) * dist * e)), max(1, int(sz * (1 - 0.6 * e))), 1.0, -1, cv2.LINE_AA)
            self.layer(fr, cam, blur(d, 1.5), opacity=0.85 * (1 - smooth((db - 0.4) / 0.7)))
        # the meanings
        mo = 1 - smooth((t - self.t_hit + 0.6) / 0.5)
        self.layer(fr, cam, self.arr.render(t, red=True), color=VERM, opacity=mo)
        for k, (x, txt, pic, col, j) in enumerate(self.items):
            f = smooth((t - self.t_mean[k] + 0.05) / 0.35) * mo
            if f <= 0:
                continue
            lx = self.lab_x[k]
            pop = 0.8 + 0.2 * ease_out((t - self.t_mean[k] + 0.05) / 0.35)
            if pic == 'flower':
                kit.put(fr, cam, self.P['flower'], lx, SY + 470, PINK, f, pop * 1.6)
                kit.put(fr, cam, self.P['stem'], lx, SY + 470, kit.GREEN, f, pop * 1.6)
            elif pic == 'heart':
                kit.put(fr, cam, self.P['heart'], lx, SY + 450, BLUE, f, pop * 1.7)
            else:
                kit.put(fr, cam, self.P['mother'], lx, SY + 470, INK, f, pop * 1.5)
            kit.put(fr, cam, self.labels[k], lx, SY + 290, col * 0.85, f)
        # huh?
        if t > self.t_huh:
            g = (t - self.t_huh) / 0.25
            kit.put(fr, cam, self.q, 880, SY - 260, VERM, smooth(g), 1.3 + 0.6 * np.exp(-g * 2), rot=0.2)
        return fr.img

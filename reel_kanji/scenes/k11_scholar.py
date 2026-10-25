"""Scene 11 — "These characters became known in Japanese as kanji."
A scholar seen from behind, bowed over a long scroll of classical text, a pagoda and a blossom branch beyond.
漢字 is brushed in large above (Yuji Syuku, a Japanese brush face), and read as kan "Han (China)" + ji "character"."""
import numpy as np

from art import blossom_cluster, cherry_branch, figure_back, pagoda
from engine import (FONT_BRUSH_JP, FONT_EN, FONT_EN_ITALIC, FONT_JP_FALLBACK, GREY, H, INK, SAKURA, SEPIA, VERM, W, Cam,
                    StrokeSet, blur, bloom, cached, ease_io, keyframes, noise, reveal_time, smooth, text_layer)
from glyphs import typeset
from scenes.base import SceneBase
from timeline import at

TEXT = '子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎'


class Scene(SceneBase):
    paper_seed = 51
    paper_stains = 0.12

    def build(self):
        def paint():
            from inkpaint import Painting
            p = Painting(W, H, seed=52)
            r1 = p.ridge(1120, peaks=[(200, 160, 200), (760, 220, 230)], rough=12, seed=1)
            p.wash(r1, dens=0.16, decay=110, tex=80, edge=0.3, mist_y=1180, mist_h=50, seed=2)
            D = p.D.copy()
            ss = StrokeSet(W, H)
            pagoda(ss, 860, 1180, 1.2, 0, 0.01, seed=3, ink=0.8)
            tips = cherry_branch(ss, [(-40, 220), (90, 262), (190, 240), (300, 300), (380, 284)], 18, 0, 0.01, seed=4)
            deco = ss.render(9)
            fs = StrokeSet(W, H)
            figure_back(fs, 540, 1450, 1.75, 0, seed=5, dur=0.01, pose='read')
            fig = fs.render(9)
            import cv2
            occl = np.clip(blur(cv2.dilate((fig > 0.12).astype(np.float32), np.ones((9, 9), np.uint8)), 3), 0, 1)
            # fill the silhouette's interior (the robe wash is translucent)
            cs, _ = cv2.findContours((occl > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solid = np.zeros((H, W), np.uint8)
            cv2.drawContours(solid, cs, -1, 1, -1)
            occl = np.maximum(occl, blur(solid.astype(np.float32), 2))
            bl = np.zeros((H, W), np.float32)
            for k, (bx, by) in enumerate(tips):
                bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 12, 40, seed=10 + k))
            # the scroll on the desk in front of the scholar: paper, rollers, columns of text
            sc = np.zeros((H, W), np.float32)
            sc[1222:1338, 120:960] = 1
            sc = blur(sc, 1.5)
            txt = np.zeros((H, W), np.float32)
            k = 0
            for x in range(915, 150, -30):                         # columns read top to bottom, right to left
                for j in range(4):
                    g = typeset(TEXT[k % len(TEXT)], 24, font=FONT_JP_FALLBACK, seed=k, pad=0.05, ink=0.85, medians=False)
                    k += 1
                    y = 1232 + j * 26
                    txt[y:y + g.h, x:x + g.w] = np.maximum(txt[y:y + g.h, x:x + g.w], g.dens)
            rollers = np.zeros((H, W), np.float32)
            rollers[1212:1348, 108:124] = 1
            rollers[1212:1348, 956:972] = 1
            sc, txt, rollers = sc * (1 - occl), txt * (1 - occl), rollers * (1 - occl)
            return {'land': D.astype(np.float32), 'deco': deco.astype(np.float32), 'fig': fig.astype(np.float32),
                    'occl': occl.astype(np.float32), 'bl': bl.astype(np.float32),
                    'scroll': sc.astype(np.float32), 'txt': txt.astype(np.float32), 'rollers': blur(rollers, 1.0).astype(np.float32)}
        self.A = cached('k11_scene_v2', paint)
        self.T = {k: None for k in self.A}
        self.kan = typeset('漢', 300, font=FONT_BRUSH_JP, seed=7, medians=False, sweep=(0.6, 1))
        self.ji = typeset('字', 300, font=FONT_BRUSH_JP, seed=8, medians=False, sweep=(0.6, 1))
        self.t_kan = at('These characters became')[0] + 0.2
        self.t_ji = self.t_kan + 0.55
        self.t_lbl = at('as kanji')[0]
        self.lbl = [(text_layer('kan', FONT_EN_ITALIC, 62)[0], text_layer('Han (China)', FONT_EN, 42)[0], 380),
                    (text_layer('ji', FONT_EN_ITALIC, 62)[0], text_layer('character', FONT_EN, 42)[0], 700)]
        self.kanji, _ = text_layer('kanji', FONT_EN_ITALIC, 76)
        self.wash_T = __import__('engine').wash_map(H, W, (540, 1300), noise_amp=0.5, noise_scale=120, seed=9)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.06)), (self.t1, (540, 940, 1.0))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        fw = (t - self.t0) / 1.2
        self.layer(fr, cam, reveal_time(A['land'], self.wash_T, fw, soft=0.1, wet=0.2), depth=0.9)
        self.layer(fr, cam, A['scroll'], color=SEPIA, opacity=0.18)
        self.layer(fr, cam, A['txt'], color=INK, opacity=0.7)
        self.layer(fr, cam, A['rollers'], color=np.array([0.35, 0.22, 0.14], np.float32), opacity=0.9)
        self.layer(fr, cam, reveal_time(A['deco'], self.wash_T, fw * 1.1, soft=0.1, wet=0.25))
        fr.over(np.asarray(self.paper[1300, 540], np.float32), self.project(cam, A['occl']), 0.55 * smooth(fw))   # the robe hides what is behind it
        self.layer(fr, cam, reveal_time(A['fig'], self.wash_T, fw * 1.1, soft=0.1, wet=0.25))
        self.layer(fr, cam, A['bl'], color=SAKURA, opacity=0.9 * smooth(fw))
        for g, x, t0 in ((self.kan, 380, self.t_kan), (self.ji, 700, self.t_ji)):
            f = (t - t0) / 0.5
            if f > 0:
                d = bloom(g.at(min(f, 1.05), soft=0.05, wet=0.45), t - t0 - 0.5, amount=0.14)
                self.place(fr, cam, d, x, 470)
        for k, (a1, a2, x) in enumerate(self.lbl):
            op = smooth((t - self.t_kan - 1.0 - 0.3 * k) / 0.4)
            self.place(fr, cam, a1, x, 680, color=VERM, opacity=op)
            self.place(fr, cam, a2, x, 735, color=GREY, opacity=op)
        op = smooth((t - self.t_lbl) / 0.4)
        self.place(fr, cam, self.kanji, 540, 840, opacity=op, scale=1 + 0.06 * np.exp(-max(t - self.t_lbl, 0) * 5))
        return fr.img

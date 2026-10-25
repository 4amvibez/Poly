"""Scene 3 — "In ancient China, people carved symbols onto oracle bones and turtle shells.  Some of these early
characters represented things they could see in everyday life."
The two oracle-bone photographs (supplied by @raj_in_jp) on the washi: the first fills the frame while the camera
reads down its columns, with 甲骨文 brushed vertically in vermilion; on "turtle shells" the second slides in beside
it.  Then the photographs step back and eight pictographs are carved, one after another, into small bone tiles,
each with its meaning: sun, moon, mountain, tree, river, rain, person, fish (simplified oracle-bone forms)."""
import numpy as np

from ancient import ancient
from engine import (FONT_EN_ITALIC, FONT_JP, GREY, H, INK, SEPIA, VERM, W, Cam, StrokeSet, blur, ease_io, ease_out,
                    keyframes, noise, reveal_time, smooth, text_layer)
from glyphs import typeset
from photos import cutout, place_photo
from poly_ink.brush import wash as pwash
from scenes.base import SceneBase, museum_card
from timeline import at

TILES = [('日', 'sun'), ('月', 'moon'), ('山', 'mountain'), ('木', 'tree'),
         ('川', 'river'), ('雨', 'rain'), ('人', 'person'), ('魚', 'fish')]
BONE = np.array([0.91, 0.83, 0.66], np.float32)
GROOVE = np.array([0.24, 0.16, 0.10], np.float32)


class Scene(SceneBase):
    paper_seed = 43
    paper_stains = 0.12

    def build(self):
        self.b1, self.b2 = cutout('bone1'), cutout('bone2')
        self.t_shell = at('turtle shells')[0] - 0.25
        self.t_some = at('Some of these')[0] - 0.1
        # vertical 甲骨文 in vermilion, and the museum card
        self.kkb = [typeset(c, 92, seed=5 + i, medians=False) for i, c in enumerate('甲骨文')]
        self.card = [(text_layer('Oracle bones · Shang dynasty, China', FONT_EN_ITALIC, 44)[0], GREY * 0.55),
                     (text_layer('c. 1200 BCE', FONT_EN_ITALIC, 38)[0], GREY)]
        # the bone tiles: an irregular bone-coloured wash with fine cracks, a pictograph carved into each
        r = np.random.default_rng(3)
        self.tiles = []
        for k, (ch, en) in enumerate(TILES):
            col, row = k % 2, k // 2
            cx, cy = 300 + 460 * col, 470 + 232 * row
            tw, th = 410, 206
            pts = []
            for a in np.linspace(0, 2 * np.pi, 13, endpoint=False):
                rr = 1 + r.normal(0, 0.09)
                pts.append((tw / 2 + 10 + np.cos(a) * tw / 2 * rr * (1 + 0.15 * abs(np.cos(a)) ** 6),
                            th / 2 + 10 + np.sin(a) * th / 2 * rr))
            from engine import paste
            full = np.zeros((th + 20, tw + 20), np.float32)
            st = pwash(tw + 20, th + 20, pts, ink=0.9, soft=2.0, seed=40 + k, grain=0.3)
            paste(full, st)
            tex = 0.8 + 0.2 * noise(th + 20, tw + 20, 30, 4, 50 + k)
            cracks = np.zeros_like(full)
            import cv2
            for j in range(3):
                x = r.uniform(30, tw - 10)
                p = [(x, r.uniform(5, 40))]
                for _ in range(5):
                    p.append((p[-1][0] + r.normal(0, 14), p[-1][1] + r.uniform(20, 45)))
                cv2.polylines(cracks, [np.array(p, np.int32)], False, 1.0, 1, cv2.LINE_AA)
            g = ancient(ch, 'oracle', 150, seed=60 + k)
            lbl, _ = text_layer(en, FONT_EN_ITALIC, 50)
            self.tiles.append(dict(cx=cx, cy=cy, tile=np.clip(full * tex, 0, 1), cracks=blur(cracks, 0.6) * full, g=g,
                                   lbl=lbl, t=self.t_some + 0.1 + 0.26 * k))

    def photo_keys(self, t):
        """(x, y, scale, opacity) for each photograph"""
        a, s, z = self.t0, self.t_shell, self.t_some
        k1 = keyframes(t, [(a, (470, 930, 0.74, 1.0)), (s, (470, 900, 0.76, 1.0)), (s + 0.7, (330, 860, 0.5, 1.0)),
                           (z, (330, 840, 0.5, 1.0)), (z + 0.7, (300, 520, 0.3, 0.0))], ease=ease_io)
        k2 = keyframes(t, [(a, (1500, 1000, 0.55, 1.0)), (s, (1500, 1000, 0.55, 1.0)), (s + 0.7, (790, 990, 0.55, 1.0)),
                           (z, (780, 980, 0.55, 1.0)), (z + 0.7, (800, 560, 0.33, 0.0))], ease=ease_io)
        return k1, k2

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.0)), (self.t_shell, (540, 930, 1.08)),
                                  (self.t_some, (540, 930, 1.0)), (self.t1, (540, 900, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        M = cam.matrix()
        for ph, (x, y, sc, op) in zip((self.b1, self.b2), self.photo_keys(t)):
            if op <= 0.01:
                continue
            hh, ww = ph['alpha'].shape
            A = np.array([[sc, 0, x - sc * ww / 2], [0, sc, y - sc * hh / 2]], np.float32)
            Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
            place_photo(fr, ph, Mt, opacity=op, shadow=0.45, lift=1.4)
        # 甲骨文, brushed vertically in vermilion at the top right
        fk = (t - self.t0 - 0.5) / 0.9
        out = 1 - smooth((t - self.t_some) / 0.5)
        if fk > 0 and out > 0:
            for i, g in enumerate(self.kkb):
                self.place(fr, cam, g.at(fk * 1.4 - i * 0.3, soft=0.06), 905, 300 + i * 104, color=VERM, opacity=0.9 * out)
        museum_card(fr, self.card, 540, 1310, smooth((t - self.t0 - 1.0) / 0.45) * out)
        # the pictographs, carved into bone tiles
        for T in self.tiles:
            f = (t - T['t']) / 0.55
            if f <= 0:
                continue
            e = ease_out(f * 1.4)
            yoff = 30 * (1 - e)
            self.place(fr, cam, blur(T['tile'], 6), T['cx'] + 8, T['cy'] + 12 + yoff, opacity=0.35 * e)        # shadow
            self.place(fr, cam, T['tile'], T['cx'], T['cy'] + yoff, color=BONE, opacity=0.9 * e)
            self.place(fr, cam, T['cracks'], T['cx'], T['cy'] + yoff, color=GROOVE, opacity=0.35 * e)
            g = T['g']
            d = g.at(f - 0.25, soft=0.03, wet=0.25)
            gx = T['cx'] - 105
            self.place(fr, cam, blur(d, 2.5), gx + 2, T['cy'] + yoff + 3, color=BONE * 0.7, opacity=0.5)   # groove depth
            self.place(fr, cam, d, gx, T['cy'] + yoff, color=GROOVE)
            self.place(fr, cam, T['lbl'], T['cx'] + 75, T['cy'] + yoff, color=GROOVE, opacity=smooth(f * 2 - 0.8))
        return fr.img

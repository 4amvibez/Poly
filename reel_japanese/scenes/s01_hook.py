"""Scene 1 — the hook: "Japanese has three scripts. But... why?"
日本語 is written as "Japanese" is spoken; 漢字, ひらがな and カタカナ stamp onto the page one after another on
"three scripts", each with its name; on "But..." they fall back into the paper, and a huge vermilion question
mark is brushed across them on "why?".  The ink of the question mark then spreads into the next scene."""
import cv2
import numpy as np

from engine import (FONT_EN_ITALIC, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, bloom, ease_io, ease_out,
                    keyframes, noise, reveal_time, smooth, spline, stroke, text_layer, warp)
from glyphs import text_glyphs
from poly_ink.brush import splatter, wash as pwash
from scenes.base import SceneBase, dof
from timeline import at

ROWS = [('漢字', 'kanji', 640, 170), ('ひらがな', 'hiragana', 885, 132), ('カタカナ', 'katakana', 1120, 132)]


class Scene(SceneBase):
    paper_seed = 11

    def build(self):
        D, T, _ = text_glyphs('日本語', 124, seed=31, tracking=0.04)
        self.title = (D, T, blur(D, 4))
        self.t_title = at('Japanese has')[0] - 0.15
        s3, e3 = at('three scripts')
        self.rows = []
        for k, (jp, en, y, size) in enumerate(ROWS):
            D, T, _ = text_glyphs(jp, size, seed=40 + 10 * k, tracking=0.03)
            lbl, _ = text_layer(en, FONT_EN_ITALIC, 46)
            self.rows.append(dict(D=D, T=T, halo=blur(D, 5), lbl=lbl, y=y, t=s3 - 0.12 + 0.36 * k))
        self.t_but = at('But')[0]
        self.t_why = at('why')[0]
        # the question mark: one sweeping vermilion stroke and a heavy dot
        q = StrokeSet(W, H)
        path = spline([(392, 720), (420, 612), (535, 548), (668, 592), (700, 706), (628, 800), (560, 868), (544, 965),
                       (548, 1048)], 14)
        q.add(stroke(W, H, path, 82, ink=0.95, dry=0.42, profile=((0, 0.55), (0.08, 1.0), (0.7, 0.9), (1, 0.45)),
                     seed=7, red=True), self.t_why - 0.62, 0.5)
        circ = [(548 + 44 * np.cos(a), 1166 + 46 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 28, endpoint=False)]
        dot = pwash(W, H, circ, ink=0.96, soft=2.0, seed=8, grain=0.18)
        dot.red = True
        q.add(dot, self.t_why + 0.02, 0.12, bloom_amt=0.3)
        sp = splatter(W, H, 560, 1170, (90, 60), n=14, size=(1.5, 4.5), seed=9)
        sp.red = True
        q.add(sp, self.t_why + 0.08, 0.15)
        self.q = q

    def cam(self, t):
        return Cam(*keyframes(t, [(0.0, (540, 900, 1.08)), (self.t_but, (540, 880, 1.0)), (self.t1, (540, 860, 1.05))],
                              ease=ease_io))

    def put(self, fr, cam, dens, cx, cy, scale=1.0, color=INK, opacity=1.0):
        if opacity <= 0:
            return
        h, w = dens.shape
        A = np.array([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]], np.float32)
        M = cam.matrix()
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(dens, Mt), color, opacity)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        back = smooth((t - self.t_but) / 0.6)                  # "But..." — the scripts fall back into the paper
        D, T, halo = self.title
        ft = (t - self.t_title) / 0.55
        if ft > 0:
            d = bloom(reveal_time(D, T, ft * 1.1, soft=0.05, wet=0.45), t - self.t_title - 0.5, amount=0.15, halo=halo)
            self.put(fr, cam, d, 540, 380, opacity=1 - 0.55 * back)
        for k, r in enumerate(self.rows):
            f = (t - r['t']) / 0.26
            if f <= 0:
                continue
            punch = 1 + 0.12 * np.exp(-max(f, 0) * 2.2) * (f < 3)      # a stamp: lands big, settles
            d = bloom(reveal_time(r['D'], r['T'], f * 1.15, soft=0.06, wet=0.5), t - r['t'] - 0.3, amount=0.22, halo=r['halo'])
            sc = punch * (1 - 0.08 * back)
            self.put(fr, cam, d, 540, r['y'], sc, opacity=1 - 0.68 * back)
            self.put(fr, cam, r['lbl'], 540, r['y'] + (100 if k == 0 else 84), color=GREY,
                     opacity=smooth(f - 0.8) * (1 - 0.75 * back))
        self.layer(fr, cam, self.q.render(t, red=True), color=VERM)
        img = dof(fr.img, 1 - smooth(t / 0.9), focus_y=0.45, band=0.15)
        return img

    def exit_map(self):
        """screen-space time map for the next scene: the ink spreads outward from the question mark"""
        cam = self.cam(self.t1)
        m = self.project(cam, self.q.render(99, red=True))
        core = (m > 0.3).astype(np.uint8)
        dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5).astype(np.float32)
        T = (dist / dist.max()) ** 0.8 + (noise(H, W, 120, 4, 91) - 0.5) * 0.22
        return np.clip((T - T.min()) / (T.max() - T.min()), 0, 1)

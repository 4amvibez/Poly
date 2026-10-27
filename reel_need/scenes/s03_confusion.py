"""Scene 3 — "Wait… why does this suddenly look so confusing?"
ははははながすき。 hangs at the top; below it a reader's head in profile, a spiral of confusion over it, and thought
clouds pop up around it, one guess after another, each a kanji with what it would mean: 母? mother, 歯? teeth,
花? flower, 鼻? nose, 葉? leaf, 羽? feather.  On "confusing" the worst guess lands in vermilion: "Mother likes…
noses?!" (母は鼻が好き)."""
import numpy as np

import head as HD
import kit
import pics
from backdrop import backdrop, draw_backdrop
from engine import (FONT_BRUSH_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, cached, ease_out, smooth,
                    spline, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

GUESS = [('母', 'mother', 205, 640), ('花', 'flower', 875, 640), ('歯', 'teeth', 160, 1010), ('鼻', 'nose', 920, 1000),
         ('葉', 'leaf', 205, 1290), ('羽', 'feather', 875, 1280)]
HEAD = (560, 1060)


class Scene(SceneBase):
    paper_seed = 83

    def build(self):
        self.B = backdrop(seed=220, branch='right', ground=1500)
        self.row = kit.Row('ははははながすき。', 84, seed=120)
        self.head = HD.build()
        cl = cached('s03_cloud_v2', lambda: dict(zip(('l', 'f'), pics.cloud(330, 260, seed=9, tail=(0.3, 1)))))
        self.cl, self.cf = cl['l'], cl['f']
        self.g = [(typeset(k, 92, font=FONT_JP, seed=150 + i, pad=0.04, medians=False), text_layer(e + '?', FONT_EN_ITALIC, 38)[0], x, y)
                  for i, (k, e, x, y) in enumerate(GUESS)]
        tw, tc = at('Wait')[0], at('confusing')[0]
        why = at('why does')[0]
        self.t_pop = [tw + 0.1, tw + 0.45, why, why + 0.35, at('suddenly')[0] + 0.2, at('look so')[0]]
        self.t_joke = tc + 0.1
        self.joke1, _ = text_layer('"Mother likes…', FONT_BRUSH_EN, 70)
        self.joke2, _ = text_layer('noses?!"', FONT_BRUSH_EN, 92)
        self.jjp = kit.Row('母は鼻が好き？', 50, seed=170)
        sp = StrokeSet(W, H)
        a = np.linspace(0, 5 * np.pi, 60)
        cx, cy = HEAD[0] - HD.HW / 2 + HD.BRAIN[0] + 10, HEAD[1] - HD.HH / 2 - 30
        pts = [(cx + (8 + 6 * u) * np.cos(u) * 1.4, cy + (8 + 6 * u) * np.sin(u) * 0.5) for u in a]
        sp.add(stroke(W, H, pts, 4, ink=0.85, dry=0.4, profile=SWEEP, seed=3), self.t0 + 0.4, 0.8)
        self.spiral = sp
        self.drop = cached('s03_drop_v1', lambda: {'d': _drop()})['d']

    def cam(self, t):
        wob = np.sin(t * 3) * 0.006
        return Cam(540, 960, 1.0 + 0.02 * smooth((t - self.t0) / 4), wob)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.6))
        self.row.draw(fr, cam, 540, 330, lambda i: 1.0)
        # the head, confused
        kit.put(fr, cam, self.head, *HEAD, INK, 1.0)
        self.layer(fr, cam, self.spiral.render(t), opacity=0.9)
        fd = smooth((t - self.t_pop[2]) / 0.3)
        kit.put(fr, cam, self.drop, HEAD[0] + 120, HEAD[1] - 160 + 30 * ((t * 0.8) % 1), np.array([0.35, 0.5, 0.65], np.float32), fd * 0.8)
        # guesses
        for k, (g, gl, x, y) in enumerate(self.g):
            f = (t - self.t_pop[k]) / 0.3
            if f <= 0:
                continue
            e = ease_out(min(f, 1))
            sc = 0.6 + 0.4 * e + 0.08 * np.sin(np.pi * min(f, 1))
            bob = 6 * np.sin(t * 2.5 + k)
            hot = k == 3 and t > self.t_joke
            kit.put(fr, cam, self.cf, x, y + bob, kit.PAPER_C, 0.96 * smooth(f), sc, over=True)
            kit.put(fr, cam, self.cl, x, y + bob, VERM if hot else INK, smooth(f), sc)
            kit.put(fr, cam, g.dens, x, y - 40 + bob, VERM if hot else INK, smooth(f), sc)
            kit.put(fr, cam, gl, x, y + 32 + bob, GREY, smooth(f), sc)
        # the worst guess
        fj = (t - self.t_joke) / 0.3
        if fj > 0:
            kit.put(fr, cam, self.joke1, 540, 470, VERM, smooth(fj))
            kit.put(fr, cam, self.joke2, 560, 560, VERM, smooth(fj - 0.6), 1 + 0.3 * np.exp(-max(fj - 0.6, 0) * 2))
        return fr.img


def _drop():
    ss = StrokeSet(60, 90)
    from poly_ink.brush import wash
    ss.add(wash(60, 90, [(30, 6), (46, 50), (40, 74), (20, 74), (14, 50)], ink=0.9, soft=1.2, seed=1), 0, 1)
    return ss.render(9).astype(np.float32)

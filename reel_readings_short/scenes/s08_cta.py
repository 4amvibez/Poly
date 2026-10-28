"""Scene 8 — the question to comment on.  "Now try this one.  How do you read it?  Comment below, and follow me for more!"
今日 is brushed huge with a vermilion ？ after it (no answer is given: it is read きょう, a special reading, or こんにち,
and that is the comment).  HOW DO YOU READ IT? is set above; on "Comment below" COMMENT BELOW appears with a brush
arrow bouncing down towards the comments; on "follow me" the black banner FOLLOW @raj_in_jp sweeps in and holds."""
import numpy as np

import kit
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_HEAD, FONT_JP, H, INK, VERM, W, Cam, StrokeSet, noise, smooth,
                    spline, stroke, text_layer)
from glyphs import typeset
from scenes.kscene import KanjiScene
from timeline import at

IVORY = np.array([0.98, 0.95, 0.88], np.float32)


class Scene(KanjiScene):
    paper_seed = 108

    def build(self):
        self.kj = [typeset(c, 340, font=FONT_BRUSH_JP, seed=160 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('今日')]
        self.q, _ = text_layer('？', FONT_JP, 300)
        self.how, _ = text_layer('HOW DO YOU READ IT?', FONT_HEAD, 62)
        self.com, _ = text_layer('COMMENT BELOW', FONT_BRUSH_EN, 104)
        self.com = self.com * (0.85 + 0.15 * noise(*self.com.shape, 6, 2, 8))
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 40, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 84)
        self.t_now = at('Now try')[0]
        self.t_how = at('How do you read it')[0]
        self.t_com = at('Comment below')[0]
        self.t_follow = at('follow me')[0]
        ar = StrokeSet(W, H)
        kit.arrow(ar, (540, 1010), (540, 1120), self.t_com + 0.3, 0.3, width=9, seed=170, red=True, head=26)
        self.ar = ar
        bn = StrokeSet(W, H)
        bn.add(stroke(W, H, [(250, 1336), (540, 1324), (830, 1332)], 118, ink=0.96, dry=0.45,
                      profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=171), self.t_follow - 0.05, 0.4)
        self.banner = bn

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.04 * np.exp(-max(t - self.t_now, 0) * 8) * (t > self.t_now) + 0.02 * smooth((t - self.t0) / 6))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        for k, (g, x) in enumerate(zip(self.kj, (330, 690))):
            kit.put(fr, cam, g.at((t - self.t_now + 0.1 - 0.12 * k) / 0.3, soft=0.05), x, 640, INK, 1.0)
        fq = (t - self.t_how) / 0.18
        if fq > 0:
            wob = 1 + 0.4 * np.exp(-fq * 2.5) + 0.04 * np.sin(t * 5)
            kit.put(fr, cam, self.q, 930, 560, VERM, smooth(fq), wob, rot=0.1)
        kit.put(fr, cam, self.how, 540, 320, INK, smooth((t - self.t_how) / 0.25))
        fc = (t - self.t_com) / 0.2
        if fc > 0:
            kit.put(fr, cam, self.com, 540, 930, VERM, smooth(fc), 1 + 0.3 * np.exp(-fc * 2.5), rot=-0.03)
            bob = 10 * np.sin(max(t - self.t_com - 0.6, 0) * 7) * (t > self.t_com + 0.6)
            self.layer(fr, cam, self.ar.render(t, red=True), 0, bob, color=VERM)
        fb = (t - self.t_follow + 0.05) / 0.4
        if fb > 0:
            self.layer(fr, cam, self.banner.render(t), opacity=0.96)
            kit.put(fr, cam, self.handle, 540, 1324, IVORY, smooth((t - self.t_follow - 0.15) / 0.3), 1 + 0.1 * np.exp(-fb * 2.5), over=True)
            kit.put(fr, cam, self.follow, 540, 1238, INK, smooth(fb))
        return fr.img

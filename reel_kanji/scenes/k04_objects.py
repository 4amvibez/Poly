"""Scene 4 — "The sun. The moon. Mountains. Trees."
Four painted panels divided by two brush strokes, each opening with an ink wash exactly on its word:
a vermilion sun with brushed rays, a crescent moon in a night wash, misty mountains, a broadleaf tree."""
import numpy as np

from art import ink_tree, soft_disc, sun_disc
from engine import (FONT_EN_ITALIC, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_io, keyframes, noise,
                    reveal_time, smooth, stroke, text_layer, wash_map)
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

PANELS = [('sun', 'The sun', (305, 620)), ('moon', 'The moon', (775, 620)),
          ('mountains', 'Mountains', (305, 1085)), ('trees', 'Trees', (775, 1085))]


class Scene(SceneBase):
    paper_seed = 44

    def build(self):
        def paint():
            out = {}
            # sun with rays
            cx, cy = 305, 600
            out['sun_r'] = sun_disc(H, W, cx, cy, 100, seed=3)
            ss = StrokeSet(W, H)
            for k in range(12):
                a = k / 12 * 2 * np.pi + 0.1
                r0, r1 = 132, 132 + (46 if k % 2 == 0 else 30)
                ss.add(stroke(W, H, [(cx + np.cos(a) * r0, cy + np.sin(a) * r0), (cx + np.cos(a) * r1, cy + np.sin(a) * r1)],
                              9, ink=0.9, dry=0.4, profile=((0, 1), (1, 0.3)), seed=10 + k), 0, 1)
            out['sun_r'] = np.maximum(out['sun_r'], ss.render(9))
            # moon: a night wash with a pale crescent
            mx, my = 775, 600
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            cloud = np.exp(-(((xx - mx) / 190) ** 2 + ((yy - my) / 150) ** 2) ** 1.6) * (0.6 + 0.4 * noise(H, W, 60, 4, 7))
            out['moon_d'] = np.clip(cloud * 0.75, 0, 1)
            disc = soft_disc(H, W, mx, my, 74, edge=2)
            bite = soft_disc(H, W, mx + 34, my - 22, 66, edge=2)
            out['moon_c'] = np.clip(disc - bite, 0, 1)
            # mountains: painted on their own small sheet so they stay inside the panel
            from inkpaint import Painting
            pw, ph = 440, 400
            p = Painting(pw, ph, seed=5)
            far = np.minimum(p.peak(130, 70, 330, 80, 100, rough=8, seed=1), p.peak(330, 40, 360, 90, 110, rough=8, seed=2))
            p.wash(far, dens=0.24, decay=120, tex=60, edge=0.35, mist_y=300, mist_h=36, seed=3)
            near = np.minimum(p.peak(220, 150, 260, 95, 120, rough=10, seed=4), p.peak(400, 230, 200, 70, 90, rough=8, seed=5))
            p.wash(near, dens=0.52, decay=70, tex=50, edge=0.45, mist_y=360, mist_h=26, seed=6)
            p.cun(near, 120, 330, n=45, shadow_x=220, spread=40, max_y=360, ink=0.22)
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            fade = np.clip(np.minimum(xx, pw - xx) / 50, 0, 1) * np.clip((ph - yy) / 70, 0, 1)
            D = np.zeros((H, W), np.float32)
            D[1085 - 220:1085 - 220 + ph, 305 - pw // 2:305 - pw // 2 + pw] = p.D * fade
            out['mount_d'] = D
            # tree
            ts = StrokeSet(W, H)
            ink_tree(ts, 780, 1245, 1.4, 0, seed=21, dur=0.01, ink=0.8)
            out['tree_d'] = ts.render(9)
            return {k: v.astype(np.float32) for k, v in out.items()}
        self.A = cached('k04_panels_v2', paint)
        self.times = {k: at(w)[0] - 0.12 for k, w, _ in PANELS}
        self.T = {k: wash_map(H, W, c, noise_amp=0.4, noise_scale=70, seed=i) for i, (k, _, c) in enumerate(PANELS)}
        self.lbl = {k: text_layer(k, FONT_EN_ITALIC, 52)[0] for k, _, _ in PANELS}
        div = StrokeSet(W, H)
        div.add(stroke(W, H, [(540, 390), (536, 850), (542, 1300)], 6, ink=0.55, dry=0.6, profile=SWEEP, seed=31), self.t0 + 0.1, 0.5)
        div.add(stroke(W, H, [(110, 856), (540, 850), (970, 858)], 6, ink=0.55, dry=0.6, profile=SWEEP, seed=32), self.t0 + 0.3, 0.5)
        self.div = div

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 0.98)), (self.times['trees'] + 0.6, (540, 950, 1.02)),
                                  (self.t1, (330, 640, 1.5))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.div.render(t), color=GREY)
        A, S = self.A, self.times
        f = lambda k: (t - S[k]) / 0.55
        if f('sun') > 0:
            self.layer(fr, cam, reveal_time(A['sun_r'], self.T['sun'], f('sun') * 1.1, soft=0.08, wet=0.3), color=VERM, opacity=0.92)
        if f('moon') > 0:
            self.layer(fr, cam, reveal_time(A['moon_d'], self.T['moon'], f('moon') * 1.1, soft=0.1, wet=0.2), color=INK, opacity=0.9)
            fr.over(np.array([0.97, 0.93, 0.78], np.float32), self.project(cam, A['moon_c']), smooth(f('moon') * 1.5 - 0.3))
        if f('mountains') > 0:
            self.layer(fr, cam, reveal_time(A['mount_d'], self.T['mountains'], f('mountains') * 1.1, soft=0.08, wet=0.3))
        if f('trees') > 0:
            self.layer(fr, cam, reveal_time(A['tree_d'], self.T['trees'], f('trees') * 1.1, soft=0.08, wet=0.3))
        for k, _, (cx, cy) in PANELS:
            op = smooth(f(k) * 1.5 - 0.5)
            if op > 0:
                self.place(fr, cam, self.lbl[k], cx, cy + 178, color=GREY, opacity=op)
        return fr.img

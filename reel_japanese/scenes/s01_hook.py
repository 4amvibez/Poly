"""Scene 1 — opening hook.  Macro paper; a bold stroke lands and sweeps across; ink blooms at the contact point;
three small scripts flicker past ("three different scripts"); 日本語 emerges through the ink over a vermilion
sun while mountains rise from the stroke; the camera pulls back to the full composition."""
import numpy as np

from art import soft_disc, sun_disc
from engine import (GREY, H, INK, VERM, W, Cam, StrokeSet, blur, bloom, cached, ease_io, ease_out, keyframes,
                    noise, ramp, reveal_time, smooth, wash_map)
from glyphs import text_glyphs, typeset
from poly_ink.brush import splatter, sweep
from scenes.base import SceneBase, dof
from timeline import at


class Scene(SceneBase):
    paper_seed = 11

    def build(self):
        # the sweeping stroke: lands at the left, sweeps up and across, flicks off the right edge
        self.ss = StrokeSet(W, H)
        pts = [(96, 1318), (260, 1296), (520, 1270), (800, 1236), (1010, 1206), (1130, 1190)]
        self.stroke_pts = pts
        self.ss.add(sweep(W, H, pts, 132, ink=0.93, dry=0.56, seed=5, wash=0.32), 0.35, 1.05, bloom_amt=0.25)
        self.ss.add(splatter(W, H, 1050, 1200, (70, 26), n=16, size=(1.4, 4.5), seed=9, direction=-0.15), 1.25, 0.25)
        self.contact = (96, 1318)

        def paint_mountains():
            from inkpaint import Painting
            p = Painting(W, H, seed=21)
            far = p.ridge(1110, peaks=[(130, 150, 170), (470, 95, 220), (930, 190, 160)], rough=12, seed=3)
            p.wash(far, dens=0.15, decay=120, tex=90, edge=0.3, mist_y=1215, mist_h=70, seed=4)
            near = p.ridge(1215, peaks=[(330, 120, 240), (800, 70, 230)], rough=18, seed=5)
            p.wash(near, dens=0.3, decay=70, tex=70, edge=0.45, mist_y=1290, mist_h=50, seed=6)
            p.mist(1180, 50, 0.6, seed=7)
            D = p.D.copy()
            D[1330:] = 0
            return {'D': D.astype(np.float32)}
        self.mount = cached('s01_mountains', paint_mountains)['D']
        self.mount_T = wash_map(H, W, None, direction=(0.25, -1), noise_amp=0.5, noise_scale=120, seed=8)

        self.sun_c, self.sun_r = (540, 830), 255
        self.sun = sun_disc(H, W, *self.sun_c, self.sun_r, seed=12)
        self.sun_T = wash_map(H, W, self.sun_c, noise_amp=0.25, noise_scale=40, seed=13)

        # title 日本語, written across the sun
        D, T, _ = text_glyphs('日本語', 230, seed=31, tracking=0.02)
        self.title, self.title_T = D, T
        self.title_halo = blur(D, 5)
        hh, ww = D.shape
        self.title_xy = (540 - ww // 2, 830 - hh // 2)

        # three scripts glimpsed during "three different scripts"
        self.three = [typeset(c, 92, seed=40 + i, medians=False) for i, c in enumerate('漢かカ')]
        s, e = at('three different scripts')
        self.three_t = [s - 0.05, s + 0.38, s + 0.8]

        self.t_title = at('Japanese does')[0] - 1.55          # title completes as "Japanese" is spoken

    def cam(self, t):
        cx, cy, z = keyframes(t, [(0.0, (250, 1300, 2.15)), (1.5, (470, 1250, 1.75)), (3.4, (540, 1010, 1.22)),
                                  (5.8, (540, 960, 1.0)), (max(self.t1, 6.0), (540, 955, 1.035))], ease=ease_io)
        return Cam(cx, cy, z)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)

        # mountains rising out of the stroke
        fm = ramp(t, self.t_title + 0.6, self.t_title + 3.2)
        if fm > 0:
            self.layer(fr, cam, reveal_time(self.mount, self.mount_T, fm * 1.05, soft=0.08, wet=0.25), depth=1.0)

        # vermilion sun spreading from its centre
        fs = ramp(t, self.t_title + 0.5, self.t_title + 1.9)
        if fs > 0:
            self.layer(fr, cam, reveal_time(self.sun, self.sun_T, ease_out(fs) * 1.05, soft=0.06, wet=0.2), color=VERM)

        # the stroke, and the ink blooming where the brush first touched
        age = t - 0.35
        if age > 0:
            g = ease_out(age / 1.1)
            core = soft_disc(H, W, *self.contact, 8 + 34 * g, edge=4, rough=18, seed=3)
            halo = soft_disc(H, W, self.contact[0] - 6, self.contact[1] + 4, 10 + 62 * g, edge=10, rough=46, seed=4)
            blot = np.maximum(core * (0.78 + 0.15 * np.exp(-age * 2)), blur(halo, 3) * 0.26)
            fr.ink(self.project(cam, blot), INK)
            fr.ink(self.project(cam, self.ss.render(t)), INK)

        # the three scripts flicker past, then make way for the title
        for g, ts, x in zip(self.three, self.three_t, (380, 540, 700)):
            f = (t - ts) / 0.42
            if f <= 0:
                continue
            fade = 1 - smooth((t - (self.t_title - 0.1)) / 0.8)
            if fade <= 0:
                continue
            d = g.at(min(f, 1.2), soft=0.05)
            self.layer(fr, cam, d, x - g.w // 2, 1075 - g.h // 2, color=GREY, opacity=fade * 0.9)

        # 日本語 emerging through the ink, then blooming slightly into the paper
        ft = (t - self.t_title) / 1.5
        if ft > 0:
            d = reveal_time(self.title, self.title_T, ft * 1.05, soft=0.05, wet=0.45)
            d = bloom(d, t - self.t_title - 1.2, amount=0.18, halo=self.title_halo)
            self.layer(fr, cam, d, *self.title_xy)

        img = fr.img
        img = dof(img, 1 - smooth(t / 2.6), focus_y=0.62, band=0.12)
        return img

    def exit_map(self):
        """screen-space time map for the next scene: spreads outward from the stroke"""
        cam = self.cam(self.t1)
        m = self.project(cam, self.ss.render(99))
        from engine import sstep
        core = (m > 0.25).astype(np.float32)
        import cv2
        dist = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        T = dist / dist.max()
        T = T ** 0.8 + (noise(H, W, 120, 4, 91) - 0.5) * 0.22
        return np.clip((T - T.min()) / (T.max() - T.min()), 0, 1)

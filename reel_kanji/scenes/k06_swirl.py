"""Scene 6 — "And the idea doesn't stop there."  A transition beat: a misty peak and a low vermilion sun, and a big
brush sweeping one spiralling stroke of ink across the foreground; the next scene opens out of the wet ink."""
import cv2
import numpy as np

from art import brush_sprite, draw_sprite, sun_disc
from engine import H, VERM, W, Cam, StrokeSet, cached, ease_io, ease_out, keyframes, noise, smooth, stroke
from scenes.base import SceneBase
from timeline import at

C = (560, 1130)


class Scene(SceneBase):
    paper_seed = 46

    def build(self):
        def land():
            from inkpaint import Painting
            p = Painting(W, H, seed=47)
            pk = p.peak(470, 640, 520, 300, 360, power=1.2, rough=14, seed=1)
            p.wash(pk, dens=0.34, decay=150, tex=90, edge=0.4, mist_y=930, mist_h=70, seed=2)
            p.cun(pk, 300, 640, n=60, shadow_x=470, spread=60, max_y=900, ink=0.2)
            r2 = p.ridge(1000, peaks=[(150, 90, 160), (900, 120, 180)], rough=10, seed=3)
            p.wash(r2, dens=0.2, decay=80, tex=70, edge=0.3, mist_y=1060, mist_h=40, seed=4)
            D = p.D.copy()
            D[1150:] *= np.linspace(1, 0, H - 1150)[:, None] ** 0.5
            return {'D': D}
        self.land = cached('k06_land', land)['D']
        self.sun = sun_disc(H, W, 790, 560, 90, seed=5)
        # the spiral: a little more than one and a half turns, flattened like a stroke laid on the ground
        th = np.linspace(0, 2 * np.pi * 1.6, 120)
        rr = 60 + 330 * (th / th[-1]) ** 0.9
        self.path = np.stack([C[0] + rr * np.cos(th + 2.2), C[1] + rr * np.sin(th + 2.2) * 0.42], 1)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.path_s = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
        self.t_sw = at('And the idea')[0] + 0.05
        self.t_sw_end = at('stop there')[1] - 0.1
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [tuple(p) for p in self.path], 70, ink=0.95, dry=0.55,
                      profile=((0, 0.35), (0.1, 0.8), (0.6, 1.0), (0.92, 0.75), (1, 0.2)), seed=7),
               self.t_sw, self.t_sw_end - self.t_sw, bloom_amt=0.12)
        self.ss = ss
        self.brush = brush_sprite(length=820, seed=8)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.0)), (self.t1, (560, 1040, 1.12))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.sun, color=VERM, opacity=0.8, depth=0.85)
        self.layer(fr, cam, self.land, depth=0.9)
        self.layer(fr, cam, self.ss.render(t))
        f = (t - self.t_sw) / (self.t_sw_end - self.t_sw)
        if -0.3 < f < 1.4:
            u = np.clip(f, 0, 1)
            x = float(np.interp(u, self.path_s, self.path[:, 0]))
            y = float(np.interp(u, self.path_s, self.path[:, 1]))
            M = cam.matrix()
            sx, sy = M[:, :2] @ np.array([x, y]) + M[:, 2]
            lift = 80 * (1 - smooth((f + 0.3) / 0.3)) + 900 * ease_out((f - 1) / 0.4) * (f > 1)
            draw_sprite(fr, *self.brush, sx, sy - lift * 0.5, -0.45 + 0.1 * np.sin(u * 9), scale=0.95, shadow=0.3, lift=lift)
        return fr.img

    def exit_map(self):
        """time map for the next scene: it opens out of the spiral's wet ink"""
        cam = self.cam(self.t1)
        m = self.project(cam, self.ss.render(99))
        core = (m > 0.3).astype(np.uint8)
        dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5).astype(np.float32)
        T = (dist / dist.max()) ** 0.7 + (noise(H, W, 120, 4, 61) - 0.5) * 0.2
        return np.clip((T - T.min()) / (T.max() - T.min()), 0, 1)

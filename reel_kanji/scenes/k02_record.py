"""Scene 2 — "Thousands of years ago, people needed a way to record the world around them."
Dawn by a river cliff in ancient China: two people sit at a small fire while a third paints what they see onto the
smooth face of the rock — the sun, mountains, a fish, a tree — in red ochre, one picture after another as the line
is spoken; the camera moves in on the pictures, which lead straight into the oracle bones."""
import numpy as np

from ancient import ancient
from art import figure, sun_disc
from engine import (H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_io, keyframes, noise, reveal_time, smooth,
                    stroke, wash_map)
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from timeline import at

GROUND = 1385
OCHRE = np.array([0.56, 0.27, 0.15], np.float32)
DRAWINGS = [('日', (655, 985), 150), ('山', (850, 975), 160), ('魚', (650, 1195), 150), ('木', (845, 1200), 155)]


class Scene(SceneBase):
    paper_seed = 42

    def build(self):
        def paint():
            from inkpaint import Painting
            p = Painting(W, H, seed=43)
            far = p.ridge(930, peaks=[(130, 150, 170), (420, 90, 190)], rough=12, seed=1)
            p.wash(far, dens=0.14, decay=90, tex=80, edge=0.3, mist_y=990, mist_h=50, seed=2)
            land = p.D.copy()
            ss = StrokeSet(W, H)
            r = np.random.default_rng(4)
            # the cliff: a heavy wash at the edges, the smooth face kept pale where the pictures go
            cliff = [(470, GROUND + 20), (488, 1240), (522, 1040), (540, 820), (585, 640), (655, 520), (760, 470), (880, 455),
                     (990, 490), (1100, 470), (1100, GROUND + 20)]
            ss.add(pwash(W, H, cliff, ink=0.62, soft=6, seed=5, grain=0.6), 0, 1)
            for k in range(70):
                x = r.uniform(490, 1080)
                y = r.uniform(480, GROUND)
                face = ((x - 760) / 250) ** 2 + ((y - 1090) / 330) ** 2 < 1
                if face and r.random() < 0.8:
                    continue
                ss.add(stroke(W, H, [(x, y), (x + r.normal(-6, 6), y + r.uniform(30, 90))], r.uniform(3, 7), ink=r.uniform(0.35, 0.7),
                              dry=0.7, profile=PRESS_DOWN, seed=10 + k), 0, 1)
            for k in range(30):                                             # moss on the top of the cliff
                x = r.uniform(560, 1080)
                y = 470 + 30 * np.sin(x / 90) + r.uniform(-10, 10)
                ss.add(stroke(W, H, [(x, y), (x + 4, y - r.uniform(8, 18))], 3, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=90 + k), 0, 1)
            # ground and river
            for k in range(14):
                x, y = r.uniform(0, 1000), r.uniform(GROUND, GROUND + 60)
                ss.add(stroke(W, H, [(x, y), (x + r.uniform(60, 160), y + r.normal(0, 1))], r.uniform(2.5, 4.5), ink=0.3, dry=0.7,
                              profile=SWEEP, seed=130 + k), 0, 1)
            for k in range(10):
                x, y = r.uniform(0, 900), r.uniform(GROUND + 120, GROUND + 260)
                ss.add(stroke(W, H, [(x, y), (x + r.uniform(80, 200), y)], r.uniform(2, 3.5), ink=0.2, dry=0.7, profile=SWEEP,
                              seed=150 + k), 0, 1)
            D = ss.render(9)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            face = np.exp(-(((xx - 760) / 260) ** 2 + ((yy - 1090) / 340) ** 2) ** 2)
            D = D * (1 - 0.72 * face)
            # people: one painting on the rock, two by the fire
            fs = StrokeSet(W, H)
            figure(fs, 520, GROUND, 2.35, 0, seed=200, pose='speak', facing=1, dur=0.01)
            figure(fs, 175, GROUND + 4, 2.0, 0, seed=210, pose='sit', facing=1, dur=0.01)
            figure(fs, 360, GROUND + 6, 2.0, 0, seed=220, pose='sit', facing=-1, dur=0.01)
            fs.add([stroke(W, H, [(238 + 11 * k, GROUND + 8 + (k % 2) * 2), (246 + 11 * k, GROUND + 9)], 8, ink=0.6, dry=0.3,
                           profile=PRESS_DOWN, seed=230 + k) for k in range(6)], 0, 1)
            return {'land': land.astype(np.float32), 'rock': D.astype(np.float32), 'people': fs.render(9).astype(np.float32)}
        self.A = cached('k02_record_v1', paint)
        self.wash_T = wash_map(H, W, (700, 1000), noise_amp=0.45, noise_scale=120, seed=6)
        self.sun = sun_disc(H, W, 250, 640, 85, seed=7)
        t_rec = at('record the world')[0] - 0.9
        self.drawings = []
        for k, (ch, xy, size) in enumerate(DRAWINGS):
            g = ancient(ch, 'oracle', size, seed=30 + k)
            tex = 0.7 + 0.3 * noise(g.h, g.w, 6, 2, 40 + k)
            self.drawings.append(dict(g=g, tex=tex, xy=xy, t=t_rec + 0.42 * k))
        self.fire = (272, GROUND)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 960, 1.0)), (self.drawings[0]['t'], (600, 1000, 1.05)),
                                  (self.t1, (660, 1080, 1.3))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fw = (t - self.t0) / 1.0
        self.layer(fr, cam, self.sun, color=VERM, opacity=0.55 * smooth(fw), depth=0.8)
        self.layer(fr, cam, reveal_time(self.A['land'], self.wash_T, fw, soft=0.1, wet=0.2), depth=0.85)
        self.layer(fr, cam, reveal_time(self.A['rock'], self.wash_T, fw * 1.1, soft=0.1, wet=0.25))
        for Dw in self.drawings:
            f = (t - Dw['t']) / 0.45
            if f > 0:
                d = Dw['g'].at(f, soft=0.04, wet=0.3) * Dw['tex']
                self.place(fr, cam, blur(d, 1.2), *Dw['xy'], color=OCHRE, opacity=0.92)
        self.layer(fr, cam, reveal_time(self.A['people'], self.wash_T, fw * 1.2, soft=0.1, wet=0.25), depth=1.02)
        self._fire(fr, cam, t)
        return fr.img

    def _fire(self, fr, cam, t):
        fx, fy = self.fire
        h, w = 200, 130
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        fl = np.zeros((h, w), np.float32)
        for j, (dx, ph) in enumerate(((-5, 0.0), (6, 1.9), (0, 3.3))):
            ht = (44 + 10 * np.sin(t * 8.3 + ph) + 6 * np.sin(t * 13.1 + ph * 2)) * (0.8 if j == 2 else 1)
            cx = w / 2 + dx + 3 * np.sin(t * 5 + ph)
            u = np.clip((h - 12 - yy) / ht, 0, 1.2)
            body = np.exp(-((xx - cx - 4 * np.sin(u * 2.5 + t * 6)) / (10 * (1 - u) + 1.0)) ** 2) * (u < 1) * (yy < h - 10)
            fl = np.maximum(fl, body * (1 - u * 0.4))
        glow = np.exp(-(((xx - w / 2) / 60) ** 2 + ((yy - h + 14) / 18) ** 2)) * 0.4
        x0, y0 = fx - w // 2, fy - h + 14
        self.layer(fr, cam, glow, x0, y0, color=np.array([0.93, 0.7, 0.5], np.float32), depth=1.02)
        self.layer(fr, cam, blur(fl, 1.5) * 0.8, x0, y0, color=VERM, depth=1.02)

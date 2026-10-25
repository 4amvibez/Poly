"""Scene 12 — "So the next time you write a kanji, remember: behind that single character may lie centuries of
history."  A figure sits on a rock facing a vermilion sun over a misty peak and a lake, a pagoda on the far shore,
petals drifting.  On "write a kanji" the brush writes 日 in the sky; on "behind that single character" its older
forms come out from behind it and recede into the distance, oldest farthest (隷書, 篆書, 金文, 甲骨文), and on
"history" the trail warms to vermilion."""
import cv2
import numpy as np

from ancient import ancient
from art import brush_sprite, draw_sprite, figure_back, pagoda, sun_disc
from engine import (H, INK, SAKURA, SEPIA, VERM, W, Cam, StrokeSet, blur, bloom, cached, ease_io, ease_out, keyframes,
                    lerp, reveal_time, smooth, wash_map)
from glyphs import calligraphy
from scenes.base import SceneBase
from timeline import at

JI = (310, 560)
ECHOES = [('clerical', (575, 450), 0.66, 0.5, 1.2), ('seal', (758, 372), 0.47, 0.4, 2.2),
          ('bronze', (875, 320), 0.34, 0.32, 3.2), ('oracle', (950, 286), 0.25, 0.25, 4.0)]   # style, xy, scale, ink, blur


class Scene(SceneBase):
    paper_seed = 53

    def build(self):
        def paint():
            from inkpaint import Painting
            p = Painting(W, H, seed=54)
            pk = p.peak(600, 820, 380, 260, 300, power=1.2, rough=12, seed=1)
            p.wash(pk, dens=0.3, decay=140, tex=90, edge=0.4, mist_y=1060, mist_h=60, seed=2)
            p.cun(pk, 450, 760, n=50, shadow_x=600, spread=50, max_y=1040, ink=0.18)
            r2 = p.ridge(1120, peaks=[(120, 80, 160), (980, 100, 170)], rough=10, seed=3)
            p.wash(r2, dens=0.22, decay=60, tex=60, edge=0.35, mist_y=1170, mist_h=30, seed=4)
            land = p.D.copy()
            # the lake: a few dry horizontal strokes catching light, and the far shore
            ss = StrokeSet(W, H)
            from engine import stroke
            from poly_ink.brush import SWEEP
            r = np.random.default_rng(5)
            for k in range(16):
                y = r.uniform(1200, 1330)
                x = r.uniform(80, 900)
                ss.add(stroke(W, H, [(x, y), (x + r.uniform(60, 180), y + r.normal(0, 1))], r.uniform(2, 4), ink=0.22,
                              dry=0.7, profile=SWEEP, seed=10 + k), 0, 1)
            pagoda(ss, 905, 1150, 0.8, 0, 0.01, seed=6, ink=0.85)
            # the rock and the figure facing the sun
            from poly_ink.brush import wash as pwash
            rock = [(100, 1580), (120, 1520), (170, 1478), (230, 1452), (300, 1446), (370, 1458), (430, 1490), (470, 1540), (490, 1580)]
            ss.add(pwash(W, H, rock, ink=0.62, soft=4, seed=7, grain=0.6), 0, 1)
            ss.add(pwash(W, H, [(150, 1500), (230, 1462), (320, 1458), (300, 1490), (210, 1510)], ink=0.25, soft=6, seed=17, grain=0.6), 0, 1)
            from poly_ink.brush import PRESS_DOWN
            for k in range(14):
                x = r.uniform(140, 460)
                y = 1470 + abs(x - 300) * 0.25 + r.uniform(0, 30)
                ss.add(stroke(W, H, [(x, y), (x + r.normal(-10, 6), y + r.uniform(25, 60))], r.uniform(2.5, 5), ink=0.45, dry=0.7,
                              profile=PRESS_DOWN, seed=30 + k), 0, 1)
            figure_back(ss, 300, 1462, 1.35, 0, seed=8, dur=0.01)
            deco = ss.render(9)
            return {'land': land.astype(np.float32), 'deco': deco.astype(np.float32)}
        self.A = cached('k12_scene_v2', paint)
        self.sun = sun_disc(H, W, 700, 760, 165, seed=9)
        self.wash_T = wash_map(H, W, (540, 1100), noise_amp=0.5, noise_scale=120, seed=10)
        self.g = calligraphy('日', 420, seed=11)
        self.t_write = at('write a kanji')[0] - 0.05
        self.t_write_end = at('remember')[0] - 0.1
        self.brush = brush_sprite(seed=12)
        t_b = at('behind that')[0]
        self.echo = []
        for k, (st, xy, sc, op, bl) in enumerate(ECHOES):
            g = ancient('日', st, 400, seed=20 + k)
            self.echo.append(dict(d=blur(g.dens, bl / sc), xy=xy, sc=sc, op=op, t=t_b + 0.15 + 0.42 * k))
        from timeline import quiet_gap
        self.t_hit, _ = quiet_gap(at('history')[1] - 0.15, at('Follow Raj')[0] + 0.05)
        self.t_hist = self.t_hit - 0.3
        r = np.random.default_rng(13)
        self.petals = [(r.uniform(-50, 1100), r.uniform(-300, 1700), r.uniform(0.6, 1.2), r.uniform(0, 6)) for _ in range(16)]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.04)), (self.t_write_end, (540, 960, 1.0)),
                                  (self.t1, (560, 900, 1.06))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fw = (t - self.t0) / 1.1
        self.layer(fr, cam, reveal_time(self.sun, self.wash_T, fw * 1.2, soft=0.08, wet=0.2), color=VERM, opacity=0.85, depth=0.8)
        self.layer(fr, cam, reveal_time(self.A['land'], self.wash_T, fw, soft=0.1, wet=0.25), depth=0.9)
        self.layer(fr, cam, reveal_time(self.A['deco'], self.wash_T, fw * 1.1, soft=0.1, wet=0.25), depth=1.05)
        # the echoes behind 日 (drawn first: they are behind it)
        warm = smooth((t - self.t_hist) / 0.6)
        flash = float(np.exp(-max(t - self.t_hit, 0) * 4) * (t > self.t_hit))     # the trail glows as it warms
        for E in self.echo:
            f = (t - E['t']) / 0.6
            if f <= 0:
                continue
            e = ease_out(f)
            x, y = lerp(JI[0], E['xy'][0], e), lerp(JI[1], E['xy'][1], e)
            sc = lerp(0.95, E['sc'], e)
            col = np.asarray(SEPIA) * (1 - warm) + np.asarray(VERM) * warm
            self.place(fr, cam, E['d'], x, y, color=col, opacity=E['op'] * smooth(f * 2) * (1 + 0.4 * warm + 0.6 * flash), scale=sc, depth=0.95)
        f = (t - self.t_write) / (self.t_write_end - self.t_write)
        if f > 0:
            g = self.g
            d = bloom(g.at(min(f, 1.05), soft=0.012, wet=0.45), t - self.t_write_end, amount=0.12)
            halo = np.clip(blur(d, 8) * 1.6, 0, 1)
            fr.over(np.asarray(self.paper[300, 300], np.float32), self.project(cam, halo, JI[0] - g.w // 2, JI[1] - g.h // 2), 0.5)
            self.layer(fr, cam, d, JI[0] - g.w // 2, JI[1] - g.h // 2)
            if t < self.t_write_end + 0.9:
                self._brush(fr, cam, t, f)
        # drifting petals
        P = np.zeros((H, W), np.float32)
        for x0, y0, sp, ph in self.petals:
            y = (y0 + (t - self.t0) * 60 * sp) % 1900
            x = x0 + (t - self.t0) * 18 * sp + 22 * np.sin(t * 1.1 + ph)
            cv2.ellipse(P, (int(x), int(y)), (7, 4), (t * 2 + ph) * 57, 0, 360, 0.5, -1, cv2.LINE_AA)
        fr.ink(blur(P, 0.7), SAKURA, smooth(fw))
        return fr.img

    def _brush(self, fr, cam, t, f):
        g = self.g
        x, y, k, u, moving = g.brush_pos(np.clip(f, 0, 1))
        wx, wy = JI[0] - g.w // 2 + x, JI[1] - g.h // 2 + y
        M = cam.matrix()
        sx, sy = M[:, :2] @ np.array([wx, wy]) + M[:, 2]
        lift = 0 if moving else 30
        if f <= 0:
            lift = 60
        if f > 1:
            lift = 30 + 700 * ease_out((t - self.t_write_end) / 0.9)
        draw_sprite(fr, *self.brush, sx, sy - lift * 0.5, -0.3, scale=0.62 * cam.zoom, shadow=0.3, lift=lift)

"""Scene 8 — the end card.  "Follow me, and let's master Japanese together!"  またね！
Sunrise over misty peaks: the traveller from the staircase stands on the near ridge with their back to us, a pack on
their shoulders, facing a huge red sun.  "Let's master Japanese together!" is brushed across the sky; FOLLOW ·
@raj_in_jp on a black brush banner over the sun; 一歩ずつ (one step at a time) brushed down the right side.  On またね
it is stamped in vermilion (bowl) and the page holds."""
import numpy as np

import kit
from art import sun_disc
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, H, INK, SAKURA, VERM, W, Cam, StrokeSet, blur, cached, ease_out,
                    smooth, spline, stroke, text_layer)
from glyphs import typeset
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, wash as pwash
from scenes.base import SceneBase
from timeline import JA, at

IVORY = np.array([0.98, 0.95, 0.88], np.float32)


class Scene(SceneBase):
    paper_seed = 88

    def build(self):
        def paint():
            p = Painting(W, H, seed=40)
            r1 = p.ridge(1150, peaks=[(140, 200, 180), (520, 150, 240), (930, 230, 200)], rough=12, seed=41)
            p.wash(r1, dens=0.14, decay=110, tex=80, edge=0.25, mist_y=1180, mist_h=50, seed=42)
            r2 = p.ridge(1360, peaks=[(300, 120, 300), (700, 70, 260)], rough=8, seed=43)
            p.wash(r2, dens=0.3, decay=80, tex=60, edge=0.3, mist_y=1420, mist_h=40, seed=44)
            land = p.D.copy()
            land[1500:] *= np.linspace(1, 0, H - 1500)[:, None] ** 2
            ss = StrokeSet(W, H)
            _traveller(ss, 300, 1330, 1.15)
            bn = StrokeSet(W, H)
            bn.add(stroke(W, H, [(250, 902), (540, 890), (830, 898)], 116, ink=0.96, dry=0.45,
                          profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=9), 0, 1)
            return {'land': land.astype(np.float32), 'fig': ss.render(9).astype(np.float32), 'banner': bn.render(9).astype(np.float32),
                    'sun': sun_disc(H, W, 560, 930, 230, seed=45).astype(np.float32)}
        self.A = cached('s08_paint_v2', paint)
        self.l1, _ = text_layer("Let's master Japanese", FONT_BRUSH_EN, 92)
        self.l2, _ = text_layer('together!', FONT_BRUSH_EN, 110)
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 40, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 80)
        self.ippo = [typeset(c, 84, font=FONT_BRUSH_JP, seed=60 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('一歩ずつ')]
        self.matane = [typeset(c, 150, font=FONT_BRUSH_JP, seed=70 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('またね！')]
        self.t_follow = at('Follow me')[0]
        self.t_tog = at('master Japanese')[0]
        self.t_hit = JA['matane']['start']

    def cam(self, t):
        punch = 0.035 * np.exp(-max(t - self.t_hit, 0) * 5) * (t > self.t_hit)
        return Cam(540, 960 - 20 * smooth((t - self.t0) / 4), 1.03 - 0.03 * smooth((t - self.t0) / 4) + punch)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        rise = ease_out((t - self.t0) / 2.5)
        self.layer(fr, cam, A['sun'], 0, 90 * (1 - rise), color=VERM, opacity=0.88, depth=0.9)
        self.layer(fr, cam, A['land'], depth=0.95)
        self.layer(fr, cam, A['fig'], opacity=0.95)
        # FOLLOW @raj_in_jp
        f = (t - self.t_follow + 0.05) / 0.3
        if f > 0:
            m = np.zeros_like(A['banner'])
            m[:, :int(W * min(1, f * 1.2))] = 1
            self.layer(fr, cam, A['banner'] * blur(m, 10), color=INK, opacity=0.95)
            kit.put(fr, cam, self.handle, 540, 888, IVORY, smooth((t - self.t_follow - 0.1) / 0.3), 1 + 0.1 * np.exp(-f * 2.5), over=True)
            kit.put(fr, cam, self.follow, 540, 795, INK, smooth(f))
        kit.put(fr, cam, self.l1, 540, 250, INK, smooth((t - self.t_tog + 0.2) / 0.35))
        kit.put(fr, cam, self.l2, 540, 360, VERM, smooth((t - self.t_tog - 0.4) / 0.35))
        for k, g in enumerate(self.ippo):
            kit.put(fr, cam, g.at((t - self.t0 - 0.6 - 0.15 * k) / 0.35, soft=0.06), 960, 560 + 92 * k, INK, 0.9)
        # またね！
        for k, g in enumerate(self.matane):
            fk = (t - self.t_hit - 0.08 * k) / 0.22
            if fk > 0:
                kit.put(fr, cam, g.at(min(fk, 1.05), soft=0.06), 540 + (k - 1.5) * 150, 570, VERM, 1.0, 1 + 0.2 * np.exp(-fk * 2.5))
        return fr.img


def _traveller(ss, x, y, s):
    """a traveller seen from behind, standing, a pack on their back and a staff (feet at (x, y))"""
    W_, H_ = ss.w, ss.h
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    circ = lambda cx, cy, rx, ry: [(x + (cx + rx * np.cos(a)) * s, y + (cy + ry * np.sin(a)) * s) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    robe = P([(-26, -150), (26, -150), (44, -60), (52, 0), (-52, 0), (-44, -60)])
    ss.add(pwash(W_, H_, robe, ink=0.6, soft=3 * s, seed=1, grain=0.4), 0, 1)
    ss.add(stroke(W_, H_, spline(P([(-26, -150), (-42, -70), (-52, -2)]), 8), 6 * s, ink=0.9, dry=0.35, profile=PRESS_DOWN, seed=2), 0, 1)
    ss.add(stroke(W_, H_, spline(P([(26, -150), (42, -70), (52, -2)]), 8), 6 * s, ink=0.9, dry=0.35, profile=PRESS_DOWN, seed=3), 0, 1)
    ss.add(pwash(W_, H_, P([(-30, -146), (30, -146), (34, -78), (-34, -78)]), ink=0.85, soft=2 * s, seed=4, grain=0.3), 0, 1)   # the pack
    for k, sx in enumerate((-1, 1)):
        ss.add(stroke(W_, H_, P([(sx * 18, -152), (sx * 26, -120)]), 4 * s, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=5 + k), 0, 1)
    ss.add(pwash(W_, H_, circ(0, -172, 17, 20), ink=0.92, soft=1.5 * s, seed=8, grain=0.15), 0, 1)
    ss.add(pwash(W_, H_, circ(0, -196, 8, 7), ink=1.0, soft=1 * s, seed=9), 0, 1)
    ss.add(stroke(W_, H_, P([(60, -150), (70, 4)]), 5 * s, ink=0.85, dry=0.35, profile=PRESS_DOWN, seed=10), 0, 1)               # staff

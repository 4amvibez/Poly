"""Scene 7 — "So, don't look at kanji as just another obstacle to overcome.  Every character you learn helps you
understand a little more of the Japanese language.  Don't fear kanji.  Learn to see the meaning behind it."
A wall of stone blocks, a kanji carved on each, drops into place in front of a small traveller ("obstacle": the wall
rumbles, a ？! over their head).  On "Every character you learn" the blocks lift out and rebuild themselves as a
staircase; the traveller climbs it, and each step turns gold as it is learned, while a red sun rises over the top.
In the breath before "Don't fear kanji." the line is stamped across the sky (taiko).  On "the meaning behind it" the
stairs fade and 花 is brushed large, a cherry tree in full bloom painting itself behind it: the meaning behind the
character."""
import numpy as np

import kit
from art import blossom_cluster, cherry_branch, figure, sun_disc
from backdrop import backdrop, draw_backdrop
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, SAKURA, VERM, W, Cam, StrokeSet,
                    blur, cached, ease_in, ease_io, ease_out, keyframes, lerp, smooth, spline, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import PRESS_DOWN, wash as pwash
from scenes.base import SceneBase
from sentence import GOLD
from timeline import at, quiet_gap

CHARS = '漢字日本語学読書山川水火木月人心花道光空'
COLS, ROWS = 4, 5
BW_, BH_ = 150, 118
WALL = [(600 + (c - 1.5) * (BW_ + 6), 1200 - BH_ / 2 - r * (BH_ + 4)) for r in range(ROWS) for c in range(COLS)]
STEPS = [(230 + 102 * k, 1230 - 74 * k) for k in range(8)]          # where 8 of the blocks end up, as a staircase
STEP_FROM = [16, 17, 12, 13, 8, 9, 4, 5]                            # which wall blocks become the steps


class Scene(SceneBase):
    paper_seed = 87

    def build(self):
        self.B = backdrop(seed=250, branch='left', ground=1360)
        self.blk = cached('s07_block_v1', _block)
        self.kj = [typeset(c, 70, font=FONT_JP, seed=500 + i, pad=0.04, medians=False) for i, c in enumerate(CHARS)]
        fs = StrokeSet(300, 300)
        figure(fs, 150, 280, 1.0, 0, seed=7, pose='stand', facing=1, dur=0.01)
        fw = StrokeSet(300, 300)
        figure(fw, 150, 280, 1.0, 0, seed=7, pose='speak', facing=1, dur=0.01)
        self.walker, self.cheer = fs.render(9), fw.render(9)
        self.t_wall = at('So dont look')[0]
        self.t_obst = at('obstacle')[0]
        self.t_build = at('Every character')[0]
        self.t_climb = [self.t_build + 0.6 + (at('of the Japanese language')[1] - self.t_build - 0.9) * k / 7 for k in range(8)]
        c, n = quiet_gap(at('Japanese language')[1] - 0.3, at('Dont fear')[0] + 0.1)
        self.t_hit = c - n / 2 + 0.06                               # sync: "Don't fear kanji." stamps (+ taiko)
        self.t_mean = at('Learn to see')[0] - 0.1
        self.fear, _ = text_layer("Don't fear kanji.", FONT_BRUSH_EN, 120)
        self.mean1, _ = text_layer('Learn to see the', FONT_BRUSH_EN, 76)
        self.mean2, _ = text_layer('meaning', FONT_BRUSH_EN, 110)
        self.mean3, _ = text_layer('behind it.', FONT_BRUSH_EN, 76)
        self.hana = typeset('花', 360, font=FONT_BRUSH_JP, seed=9, medians=False, sweep=(0.7, 1))
        self.q, _ = text_layer('？！', FONT_JP, 76)

        def paint():
            ss = StrokeSet(W, H)
            trunk = [(540, 1400), (530, 1200), (560, 1020), (540, 880)]
            ss.add(stroke(W, H, spline(trunk, 12), 46, ink=0.85, dry=0.45, profile=((0, 1), (1, 0.4)), seed=1), 0, 1)
            tips = []
            for k, pts in enumerate(([(545, 1040), (420, 940), (290, 900), (180, 820)], [(550, 980), (680, 880), (820, 850), (930, 760)],
                                     [(540, 900), (500, 760), (430, 650)], [(545, 900), (620, 760), (700, 640)], [(530, 1130), (390, 1100), (270, 1110)],
                                     [(555, 1100), (700, 1060), (840, 1070)])):
                tips += cherry_branch(ss, pts, 20 - 2 * k, 0, 0.01, seed=10 + k)
            bl = np.zeros((H, W), np.float32)
            r = np.random.default_rng(4)
            for k, (bx, by) in enumerate(tips):
                bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 14, 52, seed=30 + k))
            for k in range(60):
                bx, by = r.uniform(180, 920), r.uniform(600, 1120)
                bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 6, 40, seed=200 + k) * 0.9)
            return {'tree': ss.render(9).astype(np.float32), 'bl': bl.astype(np.float32),
                    'sun': sun_disc(H, W, 860, 520, 120, seed=11).astype(np.float32)}
        self.A = cached('s07_paint_v1', paint)

    def cam(self, t):
        rumble = 4 * np.sin(t * 50) * np.exp(-max(t - self.t_obst, 0) * 3) * (t > self.t_obst)
        punch = 0.04 * np.exp(-max(t - self.t_hit, 0) * 5) * (t > self.t_hit)
        return Cam(540 + rumble, 960, 1.0 + punch + 0.01 * smooth((t - self.t0) / 10))

    def walker_pos(self, t):
        x, y = 120, 1240
        for k, tc in enumerate(self.t_climb):
            u = np.clip((t - tc) / 0.3, 0, 1)
            if u > 0:
                (xa, ya) = (120, 1240) if k == 0 else (STEPS[k - 1][0], STEPS[k - 1][1] - BH_ / 2 + 6)
                xb, yb = STEPS[k][0], STEPS[k][1] - BH_ / 2 + 6
                x, y = lerp(xa, xb, ease_io(u)), lerp(ya, yb, u) - 40 * np.sin(np.pi * u)
        return x, y

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        mean = smooth((t - self.t_mean) / 0.5)
        keep = 1 - mean
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        rise = ease_out((t - self.t_build - 0.5) / 3.5)
        self.layer(fr, cam, A['sun'], 0, 300 * (1 - rise), color=VERM, opacity=0.85 * smooth((t - self.t_build) / 0.6) * keep)
        # the wall, then the staircase
        build = (t - self.t_build) / 0.7
        for i, (wx, wy) in enumerate(WALL):
            fd = (t - self.t_wall - 0.04 * i) / 0.35
            if fd <= 0:
                continue
            drop = -260 * (1 - ease_in(min(fd, 1))) if fd < 1 else 0.0
            if i in STEP_FROM:
                k = STEP_FROM.index(i)
                u = ease_io(np.clip(build - 0.12 * k, 0, 1))
                x, y = lerp(wx, STEPS[k][0], u), lerp(wy + drop, STEPS[k][1], u) - 60 * np.sin(np.pi * u)
                lit = smooth((t - self.t_climb[k] - 0.25) / 0.2)
                op = keep
            else:
                x, y = wx, wy + drop + 900 * ease_in(np.clip(build, 0, 1))
                lit, op = 0.0, (1 - smooth(build)) * keep
            if op <= 0:
                continue
            kit.put(fr, cam, self.blk['fill'], x, y, lerp(np.array([0.82, 0.8, 0.76], np.float32), np.array([0.98, 0.86, 0.5], np.float32), lit), op, over=True)
            kit.put(fr, cam, self.blk['ink'], x, y, INK, op)
            kit.put(fr, cam, self.kj[i].dens, x, y, lerp(INK, VERM * 0.9, lit), op, 1 + 0.15 * np.exp(-max(t - self.t_climb[STEP_FROM.index(i)] - 0.25, 0) * 5) * (lit > 0) if i in STEP_FROM else 1)
        # the traveller
        x, y = self.walker_pos(t)
        top = t > self.t_climb[-1] + 0.35
        kit.put(fr, cam, self.cheer if top else self.walker, x, y - 182, INK, keep * smooth((t - self.t0 - 0.2) / 0.4), 1.4)
        qo = smooth((t - self.t_obst) / 0.25) * (1 - smooth((t - self.t_build) / 0.3))
        kit.put(fr, cam, self.q, x + 10, y - 330, VERM, qo, 1 + 0.3 * np.exp(-max(t - self.t_obst, 0) * 3))
        # Don't fear kanji.
        ff = (t - self.t_hit) / 0.2
        if ff > 0:
            kit.put(fr, cam, self.fear, 540, 260, INK, smooth(ff), 1 + 0.25 * np.exp(-ff * 2.5))
        # the meaning behind it: 花, and the tree it means, blooming behind it
        if mean > 0:
            g = ease_out((t - self.t_mean) / 1.4)
            m = np.zeros((H, W), np.float32)
            yy = np.arange(H, dtype=np.float32)[:, None]
            m[:] = np.clip((1450 - yy - (1 - g) * 900) / 120, 0, 1)
            self.layer(fr, cam, A['tree'] * m, opacity=0.45 * mean)
            self.layer(fr, cam, A['bl'] * m, color=SAKURA, opacity=0.95 * mean)
            kit.put(fr, cam, self.hana.at((t - self.t_mean - 0.2) / 0.6, soft=0.06), 540, 900, INK, mean)
            kit.put(fr, cam, self.mean1, 540, 440, INK, smooth((t - self.t_mean - 0.1) / 0.3))
            kit.put(fr, cam, self.mean2, 540, 540, VERM, smooth((at('meaning behind')[0] - 0.05 - t) * -1 / 0.3), 1 + 0.2 * np.exp(-max(t - at('meaning behind')[0], 0) * 4))
            kit.put(fr, cam, self.mean3, 540, 640, INK, smooth((t - at('behind it')[0]) / 0.3))
        return fr.img


def _block():
    w, h = BW_, BH_
    ss = StrokeSet(w, h)
    poly = [(6, 10), (w - 8, 6), (w - 4, h - 8), (8, h - 6)]
    fill = StrokeSet(w, h)
    fill.add(pwash(w, h, poly, ink=1.0, soft=1.2, seed=1), 0, 1)
    ss.add(pwash(w, h, poly, ink=0.1, soft=2, seed=2, grain=0.6), 0, 1)
    ss.add(stroke(w, h, poly + [poly[0]], 4, ink=0.75, dry=0.5, profile=PRESS_DOWN, seed=3), 0, 1)
    return {'fill': np.clip(fill.render(9), 0, 1).astype(np.float32), 'ink': ss.render(9).astype(np.float32)}

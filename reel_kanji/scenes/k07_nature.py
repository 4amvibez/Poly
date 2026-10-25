"""Scene 7 — "'Yama' means mountain. 'Kawa' means river. 'Ki' means tree.  Look closely, and you can see how some
of these characters resemble the things they represent."
Three rows open on their words, each a small painting → its early (oracle-bone) form → the kanji written by
brush, with its reading.  On "Look closely" the camera travels down the page to the clearest case of all, the bird
(after the reference sheets @raj_in_jp sent): a painted sparrow → the early bird pictograph → 鳥 tori; then a
vermilion ghost of the sparrow slides onto the pictograph, and the resemblance is plain."""
import numpy as np

from ancient import ancient
from art import ink_tree, sparrow
from engine import (FONT_EN, FONT_EN_ITALIC, GREY, H, INK, SEA, VERM, W, Cam, StrokeSet, blur, bloom, cached, ease_io,
                    keyframes, reveal_time, smooth, spline, stroke, text_layer, wash_map, wipe_mix)
from glyphs import calligraphy
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from timeline import at

WH = 3300                              # the page continues below the frame for the bird
ROWS = [('山', 'yama', 'mountain', 'Yama', 440), ('川', 'kawa', 'river', 'Kawa', 765), ('木', 'ki', 'tree', 'Ki', 1090)]
PX, AX, MX = 205, 525, 815             # picture, ancient form, modern kanji
BIRD_Y = (1915, 2225, 2525)            # sparrow, pictograph, 鳥


def _arrow(ss, x0, x1, y, t, seed):
    ss.add([stroke(ss.w, ss.h, [(x0, y), (x1, y - 2)], 4, ink=0.55, dry=0.5, profile=PRESS_DOWN, seed=seed),
            stroke(ss.w, ss.h, [(x1 - 14, y - 13), (x1, y - 2), (x1 - 13, y + 11)], 4, ink=0.55, dry=0.4, profile=PRESS_DOWN,
                   seed=seed + 1)], t, 0.25)


def _down(ss, x, y0, y1, t, seed):
    ss.add([stroke(ss.w, ss.h, [(x, y0), (x + 2, y1)], 4, ink=0.55, dry=0.5, profile=PRESS_DOWN, seed=seed),
            stroke(ss.w, ss.h, [(x - 13, y1 - 15), (x + 2, y1), (x + 15, y1 - 14)], 4, ink=0.55, dry=0.4, profile=PRESS_DOWN,
                   seed=seed + 1)], t, 0.25)


class Scene(SceneBase):
    paper_size = (W, WH)
    paper_seed = 47

    def build(self):
        def paint():
            out = {}
            # mountain
            from inkpaint import Painting
            pw, ph = 280, 240
            p = Painting(pw, ph, seed=3)
            far = p.peak(90, 70, 200, 60, 70, rough=6, seed=1)
            p.wash(far, dens=0.22, decay=80, tex=40, edge=0.35, mist_y=190, mist_h=22, seed=2)
            near = np.minimum(p.peak(170, 40, 220, 70, 80, rough=6, seed=3), p.peak(250, 120, 160, 50, 60, rough=5, seed=4))
            p.wash(near, dens=0.5, decay=55, tex=40, edge=0.45, mist_y=220, mist_h=18, seed=5)
            p.cun(near, 110, 230, n=30, shadow_x=170, spread=30, max_y=220, ink=0.2)
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            out['mount'] = (p.D * np.clip(np.minimum(xx, pw - xx) / 30, 0, 1) * np.clip((ph - yy) / 40, 0, 1)).astype(np.float32)
            # river: a winding band, narrow and far at the top, wide and near at the bottom
            rw, rh = 280, 250
            ss = StrokeSet(rw, rh)
            c = spline([(150, 8), (118, 60), (170, 130), (120, 200), (140, 246)], 20)
            wdt = np.linspace(14, 100, len(c))
            d = np.gradient(c, axis=0)
            n = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
            L, R = c + n * wdt[:, None] / 2, c - n * wdt[:, None] / 2
            water = pwash(rw, rh, [tuple(q) for q in np.vstack([L, R[::-1]])], ink=0.55, soft=2, seed=7, grain=0.3)
            ss.add(water, 0, 1)
            wd = ss.render(9)
            bank = StrokeSet(rw, rh)
            bank.add([stroke(rw, rh, [tuple(q) for q in L], 4.5, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=8),
                      stroke(rw, rh, [tuple(q) for q in R], 4.5, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=9)], 0, 1)
            r = np.random.default_rng(4)
            for k in range(9):
                i = int(len(c) * (0.2 + 0.08 * k))
                x, y = c[min(i, len(c) - 1)]
                ww = wdt[min(i, len(c) - 1)] * 0.3
                bank.add(stroke(rw, rh, [(x - ww, y), (x + ww, y + r.normal(0, 1))], 2.2, ink=0.5, dry=0.4, profile=SWEEP,
                                seed=20 + k), 0, 1)
            out['river_w'] = wd.astype(np.float32)
            out['river_b'] = bank.render(9).astype(np.float32)
            # tree
            ts = StrokeSet(300, 280)
            ink_tree(ts, 150, 275, 0.95, 0, seed=31, dur=0.01, ink=0.8)
            out['tree'] = ts.render(9).astype(np.float32)
            # sparrow
            sp = StrokeSet(420, 300)
            sparrow(sp, 200, 150, 1.55, 0, 0.01, seed=41)
            out['bird'] = sp.render(9).astype(np.float32)
            return out
        self.A = cached('k07_pictures_v1', paint)
        self.rows = []
        arrows = StrokeSet(W, WH)
        for k, (ch, rd, en, word, y) in enumerate(ROWS):
            t = at(word)[0] - 0.1
            lbl_r, _ = text_layer(rd, FONT_EN_ITALIC, 58)
            lbl_e, _ = text_layer(en, FONT_EN, 44)
            pic = {'山': ['mount'], '川': ['river_w', 'river_b'], '木': ['tree']}[ch]
            self.rows.append(dict(ch=ch, y=y, t=t, pic=pic, anc=ancient(ch, 'oracle', 200, seed=70 + k),
                                  mod=calligraphy(ch, 215, seed=80 + k), lr=lbl_r, le=lbl_e,
                                  T=wash_map(250, 290, (145, 125), noise_amp=0.4, noise_scale=40, seed=k)))
            _arrow(arrows, 365, 410, y, t + 0.35, 100 + 10 * k)
            _arrow(arrows, 650, 695, y, t + 0.75, 105 + 10 * k)
        # the bird
        self.t_look = at('Look closely')[0] - 0.15
        self.t_bird = self.t_look + 0.45
        self.bird_anc = ancient('鳥', 'oracle', 250, seed=90)
        self.bird_mod = calligraphy('鳥', 220, seed=91)
        self.bird_T = wash_map(300, 420, (200, 150), noise_amp=0.4, noise_scale=50, seed=9)
        _down(arrows, 540, BIRD_Y[0] + 118, BIRD_Y[1] - 132, self.t_bird + 0.6, 200)
        _down(arrows, 540, BIRD_Y[1] + 132, BIRD_Y[2] - 118, self.t_bird + 1.4, 210)
        self.arrows = arrows
        self.tori, _ = text_layer('tori', FONT_EN_ITALIC, 64)
        self.bird_en, _ = text_layer('bird', FONT_EN, 48)
        from timeline import quiet_gap
        self.t_hit, _ = quiet_gap(at('represent')[1] - 0.15, at('But heres')[0] + 0.05)   # the ghost lands in the breath
        self.t_ghost = self.t_hit - 1.0

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 900, 1.0)), (self.rows[2]['t'] + 1.2, (540, 880, 1.0)),
                                  (self.t_look, (540, 900, 1.0)), (self.t_look + 1.0, (540, 2225, 1.0)),
                                  (self.t1, (540, 2215, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        for R in self.rows:
            f = (t - R['t']) / 0.5
            if f <= 0:
                continue
            y = R['y']
            for k in R['pic']:
                d = A[k]
                T = R['T'] if d.shape == R['T'].shape else wash_map(d.shape[0], d.shape[1], (d.shape[1] / 2, d.shape[0] / 2), seed=3)
                col = SEA if k == 'river_w' else INK
                self.place(fr, cam, reveal_time(d, T, f * 1.1, soft=0.08, wet=0.3), PX, y, color=col,
                           opacity=0.38 if k == 'river_w' else 1.0, scale=1.12)
            self.place(fr, cam, R['anc'].at((t - R['t'] - 0.4) / 0.45, soft=0.03, wet=0.3), AX, y)
            fm = (t - R['t'] - 0.8) / 0.6
            if fm > 0:
                d = bloom(R['mod'].at(min(fm, 1.05), soft=0.015, wet=0.45), t - R['t'] - 1.4, amount=0.12)
                self.place(fr, cam, d, MX, y - 10)
            lo = smooth((t - R['t'] - 1.2) / 0.4)
            if lo > 0:
                self.place(fr, cam, R['lr'], MX - 4 - R['lr'].shape[1] / 2, y + 136, color=VERM, opacity=lo)
                self.place(fr, cam, R['le'], MX + 14 + R['le'].shape[1] / 2, y + 140, color=GREY, opacity=lo)
        # the bird, down the page
        fb = (t - self.t_bird) / 0.6
        if fb > 0:
            self.place(fr, cam, reveal_time(A['bird'], self.bird_T, fb * 1.1, soft=0.08, wet=0.3), 540, BIRD_Y[0])
            self.place(fr, cam, self.bird_anc.at((t - self.t_bird - 0.8) / 0.6, soft=0.03, wet=0.3), 540, BIRD_Y[1])
            fm = (t - self.t_bird - 1.6) / 0.8
            if fm > 0:
                d = bloom(self.bird_mod.at(min(fm, 1.05), soft=0.015, wet=0.45), t - self.t_bird - 2.4, amount=0.12)
                self.place(fr, cam, d, 540, BIRD_Y[2])
            lo = smooth((t - self.t_bird - 2.3) / 0.4)
            self.place(fr, cam, self.tori, 780, BIRD_Y[2] - 30, color=VERM, opacity=lo)
            self.place(fr, cam, self.bird_en, 780, BIRD_Y[2] + 34, color=GREY, opacity=lo)
            # the resemblance: a vermilion ghost of the sparrow slides down onto the pictograph
            g = smooth((t - self.t_ghost) / 1.0)
            if g > 0:
                # sparrow -> pictograph: rotate +0.32 rad and scale 0.69 so head and tail land on the drawing
                gx, gy = 540 + 11 * g, BIRD_Y[0] + (BIRD_Y[1] - 29 - BIRD_Y[0]) * g
                self.place(fr, cam, A['bird'], gx, gy, color=VERM, opacity=0.45 * smooth(g * 3),
                           scale=1 - 0.31 * g, rot=0.32 * g)
        self.layer(fr, cam, self.arrows.render(t), color=GREY)
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        if not hasattr(self, '_T'):
            self._T = prev_scene.exit_map()
        return wipe_mix(prev_img, cur_img, self._T, ease_io(f) * 1.15 - 0.05, soft=0.05, edge=0.5)

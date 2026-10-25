"""Scene 2 — before writing.  An ancient settlement emerges through an ink wash: far mountains, misty hills,
pit dwellings, trees, a fire.  People are brushed in; delicate ink lines rise between speakers (conversation,
song, story) and then dissolve into the air — speech that leaves no written trace.  Not a specific historical
event: a general evocation of a spoken language before writing."""
import numpy as np

from art import figure, granary, ink_tree, mist_band, pit_dwelling
from engine import (GREY, H, INK, PAPER, SEPIA, VERM, W, Cam, StrokeSet, blur, cached, ease_io, keyframes,
                    noise, ramp, reveal_time, smooth, spline, sstep, stroke, wash_map)
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP
from scenes.base import SceneBase
from timeline import at

WW, WH = 1500, 1920


class Scene(SceneBase):
    paper_size = (WW, WH)
    paper_seed = 12

    def build(self):
        def paint():
            from inkpaint import Painting
            p = Painting(WW, WH, seed=31)
            far = p.ridge(860, peaks=[(160, 300, 230), (620, 200, 260), (1120, 360, 250), (1450, 230, 200)], rough=16, seed=1)
            p.wash(far, dens=0.17, decay=160, tex=100, edge=0.3, mist_y=950, mist_h=80, seed=2)
            p.cun(far, 900, 1350, n=60, shadow_x=1120, spread=60, max_y=1000, ink=0.12)
            D_far = p.D.copy(); p.D[:] = 0
            mid = p.ridge(1075, peaks=[(250, 110, 300), (880, 70, 330), (1330, 130, 240)], rough=14, seed=3)
            p.wash(mid, dens=0.25, decay=85, tex=80, edge=0.4, mist_y=1160, mist_h=50, seed=4)
            p.moss(mid, 60, 1450, n=60, depth=(2, 18))
            D_mid = p.D.copy(); p.D[:] = 0
            ground = p.ridge(1215, peaks=[(700, 18, 400)], rough=6, seed=5)
            p.wash(ground, dens=0.10, decay=160, tex=60, edge=0.25, seed=6)
            D_gnd = p.D.copy()
            D_gnd[1560:] *= np.linspace(1, 0, WH - 1560)[:, None]
            return {'far': D_far, 'mid': D_mid, 'gnd': D_gnd}
        L = cached('s02_land', paint)
        self.far, self.mid, self.gnd = L['far'], L['mid'], L['gnd']
        self.wash_T = wash_map(WH, WW, (700, 1300), noise_amp=0.55, noise_scale=150, seed=9)

        t0 = self.t0
        self.ss = StrokeSet(WW, WH)
        ss = self.ss
        # dwellings and trees, brushed in as the wash settles
        pit_dwelling(ss, 330, 1236, 1.1, t0 + 1.0, seed=100)
        pit_dwelling(ss, 545, 1196, 0.62, t0 + 1.9, seed=170)
        granary(ss, 1235, 1222, 0.95, t0 + 1.4, seed=140)
        pit_dwelling(ss, 1030, 1180, 0.5, t0 + 2.1, seed=150)
        ink_tree(ss, 120, 1235, 1.2, t0 + 1.3, seed=200)
        ink_tree(ss, 1420, 1215, 1.0, t0 + 1.8, seed=230)
        ink_tree(ss, 760, 1170, 0.55, t0 + 2.2, seed=260, ink=0.45)
        # ground texture: a few horizontal dry strokes
        r = np.random.default_rng(7)
        g = [stroke(WW, WH, [(x, y), (x + r.uniform(50, 120), y + r.normal(0, 2))], r.uniform(2.5, 4.5), ink=0.16, dry=0.7,
                    profile=SWEEP, seed=300 + k)
             for k, (x, y) in enumerate(zip(r.uniform(0, 1400, 10), r.uniform(1365, 1460, 10)))]
        ss.add(g, t0 + 2.0, 1.0)

        # people, brushed in as "spoken language" is heard
        tf = at('living spoken')[0] - 0.3
        S = 1.0
        figure(ss, 650, 1335, S, tf, seed=400, pose='speak', facing=1)
        figure(ss, 772, 1340, S, tf + 0.35, seed=410, pose='listen', facing=-1)
        figure(ss, 925, 1340, S, tf + 0.7, seed=420, pose='sit', facing=-1)
        figure(ss, 1045, 1336, S, tf + 1.0, seed=430, pose='speak', facing=-1)
        figure(ss, 1135, 1332, S * 0.92, tf + 1.25, seed=440, pose='listen', facing=-1)
        self.fire = (850, 1345)
        # fire pit stones
        ss.add([stroke(WW, WH, [(822 + 9 * k, 1349 + (k % 2) * 2), (828 + 9 * k, 1350)], 7, ink=0.6, dry=0.3, profile=BLUNT,
                       seed=460 + k) for k in range(6)], tf + 0.5, 0.4)

        # the speech lines: conversation, song, story
        self.lines = StrokeSet(WW, WH)
        c0 = at('conversation')[0]
        s0 = at('song')[0]
        k0 = at('story')[0]
        prof = ((0, 0.15), (0.25, 1), (0.8, 0.7), (1, 0.1))
        mk = lambda pts, w, sd, ink=0.5: stroke(WW, WH, spline(pts, 12), w, ink=ink, dry=0.2, profile=prof, wobble=0.03,
                                                wet=0.12, seed=sd)
        # conversation: arcs passing from the speaker to the listener and back
        self.lines.add([mk([(676, 1246), (700, 1222), (728, 1214), (752, 1226), (764, 1240)], 2.8, 500),
                        mk([(680, 1236), (706, 1208), (734, 1202), (758, 1216)], 2.0, 501, 0.38)], c0 - 0.1, 0.8)
        self.lines.add(mk([(762, 1262), (740, 1280), (712, 1282), (688, 1268)], 2.2, 502, 0.42), c0 + 0.55, 0.6)
        # song: rising, gently waving lines
        for k, dx in enumerate((-9, 0, 9)):
            pts = [(930 + dx + 7 * np.sin(j * 1.25 + k), 1268 - j * 34) for j in range(9)]
            self.lines.add(mk(pts, 2.4 - 0.4 * abs(k - 1), 510 + k, 0.5 - 0.1 * abs(k - 1)), s0 - 0.1 + k * 0.12, 1.0)
        # story: a wide arc carried over the listeners
        self.lines.add([mk([(1026, 1240), (1000, 1200), (950, 1172), (880, 1166), (815, 1182), (780, 1205)], 2.8, 520),
                        mk([(1030, 1228), (1004, 1186), (950, 1156), (884, 1150), (826, 1162)], 2.0, 521, 0.36)], k0 - 0.1, 1.1)
        self.lines.add(mk([(1060, 1236), (1090, 1214), (1118, 1224)], 2.2, 522, 0.42), k0 + 0.5, 0.5)
        self.lines_gone = (at('no script')[0] - 1.6, at('no script')[1] + 0.2)

        # foreground reeds (own layer: they sway and sit nearer the lens)
        def reeds():
            R = StrokeSet(WW, WH)
            r = np.random.default_rng(81)
            for k in range(26):
                left = k < 14
                bx = r.uniform(10, 210) if left else r.uniform(1300, 1490)
                top = r.uniform(1180, 1380)
                lean = r.normal(0, 30) + (25 if left else -25)
                pts = [(bx, 1760), (bx + lean * 0.3, (1760 + top) / 2), (bx + lean, top)]
                R.add(stroke(WW, WH, spline(pts, 10), r.uniform(4, 8), ink=r.uniform(0.45, 0.8), dry=0.4,
                             profile=PRESS_DOWN, seed=600 + k), 0, 1)
                if r.random() < 0.5:                                  # seed head
                    R.add(stroke(WW, WH, [(bx + lean, top), (bx + lean * 1.25, top - 40)], 7, ink=0.6, dry=0.5,
                                 profile=((0, 0.6), (0.5, 1), (1, 0.1)), seed=700 + k), 0, 1)
            return {'D': R.render(10)}
        self.reeds = cached('s02_reeds', reeds)['D']
        self.reed_t = at('living spoken')[0] - 1.0

        # smoke texture, scrolled upward every frame
        self.smoke_tex = noise(1400, 160, 38, 4, 55)
        self.mist = mist_band(WH, WW, 985, 70, seed=60, strength=0.75)

    def cam(self, t):
        a = self.t0
        wf = at('conversation')[0]
        return Cam(*keyframes(t, [(a, (640, 1000, 1.0)), (a + 2.6, (680, 1040, 1.04)), (wf - 0.6, (860, 1130, 1.5)),
                                  (wf + 2.2, (880, 1120, 1.55)), (at('debated')[0], (860, 1060, 1.2)),
                                  (self.t1, (920, 1000, 1.02))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        ref = (750, 1250)
        fr = self.frame(cam, ref=ref)
        fw = ramp(t, self.t0 - 0.2, self.t0 + 2.6)
        rv = lambda D: reveal_time(D, self.wash_T, fw * 1.08, soft=0.07, wet=0.3) if fw < 1 else D
        self.layer(fr, cam, rv(self.far), depth=0.55, ref=ref)
        # drifting mist between the far range and the hills
        m = self.project(cam, np.roll(self.mist, int(t * 9) % WW, axis=1), depth=0.65, ref=ref)
        fr.img = fr.img * (1 - m[..., None] * 0.8) + (PAPER * 0.99) * m[..., None] * 0.8
        self.layer(fr, cam, rv(self.mid), depth=0.78, ref=ref)
        self.layer(fr, cam, rv(self.gnd), depth=1.0, ref=ref)

        # fire: a glow and flickering flame tongues, then rising smoke
        fx, fy = self.fire
        fon = ramp(t, at('living spoken')[0], at('living spoken')[0] + 0.6)
        if fon > 0:
            self._fire(fr, cam, t, fon, ref)

        self.layer(fr, cam, self.ss.render(t), depth=1.0, ref=ref)
        self._birds(fr, cam, t, ref)

        # speech lines appear, linger, then dissolve upward into the air
        a, b = self.lines_gone
        g = smooth((t - a) / (b - a))
        if g < 1:
            L = self.lines.render(t)
            if g > 0:
                n = noise(WH, WW, 26, 3, 70)
                L = blur(L, 1 + 4 * g) * sstep(g - 0.15, g + 0.15, n) * (1 - g * 0.5)
                L = np.roll(L, -int(40 * g), axis=0)
            self.layer(fr, cam, L, depth=1.0, ref=ref, opacity=1.0)

        # reeds, swaying, nearest the lens
        ro = smooth((t - self.reed_t) / 1.2)
        if ro > 0:
            k = 0.035 * np.sin(t * 1.1) + 0.012 * np.sin(t * 2.7 + 1)
            M = cam.matrix(1.3, ref)
            S = np.array([[1, -k, k * 1760], [0, 1, 0]], np.float32)
            Mt = np.concatenate([M[:, :2] @ S[:, :2], (M[:, :2] @ S[:, 2] + M[:, 2])[:, None]], 1)
            from engine import warp
            fr.ink(warp(self.reeds, Mt), INK, ro)
        return fr.img

    def _birds(self, fr, cam, t, ref):
        import cv2
        D = np.zeros((H, W), np.float32)
        M = cam.matrix(0.6, ref)
        for k, (x0, y0, sp) in enumerate([(300, 640, 22), (360, 600, 20), (430, 660, 24)]):
            x = x0 + (t - self.t0) * sp
            y = y0 + 6 * np.sin(t * 0.8 + k)
            flap = 0.5 + 0.5 * np.sin(t * 6 + k * 2)
            sx, sy = M[:, :2] @ np.array([x, y]) + M[:, 2]
            sz = 9 * M[0, 0]
            pts = np.array([(sx - sz, sy - sz * 0.5 * flap), (sx - sz * 0.4, sy - sz * 0.15), (sx, sy + 1),
                            (sx + sz * 0.4, sy - sz * 0.15), (sx + sz, sy - sz * 0.5 * flap)], np.int32)
            cv2.polylines(D, [pts], False, 0.7, 2, cv2.LINE_AA)
        fr.ink(blur(D, 0.6) * ramp(t, self.t0 + 2.5, self.t0 + 3.5), INK)

    def enter(self, prev_img, cur_img, f, prev_scene):
        """a portion of the opening stroke expands across the screen and becomes the landscape"""
        from engine import ease_io, wipe_mix
        if not hasattr(self, '_T'):
            self._T = prev_scene.exit_map()
        return wipe_mix(prev_img, cur_img, self._T, ease_io(f) * 1.12 - 0.04, soft=0.035, edge=0.4)

    def _fire(self, fr, cam, t, on, ref):
        fx, fy = self.fire
        h, w = 260, 160
        x0, y0 = fx - w // 2, fy - h + 12
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # glow on the ground
        glow = np.exp(-(((xx - w / 2) / 46) ** 2 + ((yy - h + 16) / 14) ** 2)) * 0.35
        # flame: two soft flickering tongues of vermilion over a dark ember bed
        fl = np.zeros((h, w), np.float32)
        for j, (dx, ph) in enumerate(((-5, 0.0), (6, 1.9), (0, 3.3))):
            ht = (26 + 7 * np.sin(t * 8.3 + ph) + 4 * np.sin(t * 13.1 + ph * 2)) * (0.8 if j == 2 else 1)
            cx = w / 2 + dx + 2.5 * np.sin(t * 5 + ph)
            u = np.clip((h - 14 - yy) / ht, 0, 1.2)
            body = np.exp(-((xx - cx - 3 * np.sin(u * 2.5 + t * 6)) / (6.5 * (1 - u) + 0.8)) ** 2) * (u < 1) * (yy < h - 11)
            fl = np.maximum(fl, body * (1 - u * 0.4))
        fl = blur(fl, 1.4) * 0.7
        ember = np.exp(-(((xx - w / 2) / 16) ** 2 + ((yy - h + 12) / 4) ** 2)) * 0.8
        self.layer(fr, cam, ember * on, x0, y0, depth=1.0, ref=ref, color=INK)
        # smoke: scrolled noise in a widening column, drifting right
        s = self.smoke_tex
        off = int(t * 40) % (s.shape[0] - h)
        tex = s[s.shape[0] - h - off:s.shape[0] - off]
        u = (h - yy) / h
        col = np.exp(-((xx - w / 2 - 30 * u ** 1.5) / (6 + 40 * u)) ** 2) * sstep(0.12, 0.3, u) * (1 - sstep(0.75, 1.0, u))
        smoke = col * sstep(0.45, 0.8, tex) * 0.28
        self.layer(fr, cam, smoke * on, x0, y0, depth=1.0, ref=ref, color=GREY)
        self.layer(fr, cam, glow * on, x0, y0, depth=1.0, ref=ref, color=np.array([0.93, 0.7, 0.5]))
        self.layer(fr, cam, fl * on, x0, y0, depth=1.0, ref=ref, color=VERM)

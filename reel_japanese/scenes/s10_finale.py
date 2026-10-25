"""Scene 10 — the final ink composition.  A brushstroke passes across the street and leaves an open sheet of
paper.  A black ink wave builds from layered dry-brush strokes and flung spray; a vermilion sun appears behind it;
日本語 is written down the left in the negative space, signed @raj_in_jp with a seal, like an inscription on a
painting.  The last drops settle and the composition holds."""
import numpy as np

from art import sun_disc
from engine import (FONT_EN_BOLD, GREY, H, INK, VERM, W, Cam, Frame, StrokeSet, blur, bloom, ease_io, ease_out, keyframes,
                    noise, ramp, reveal_time, smooth, spline, stroke, text_layer, wash_map, wipe_mix)
from glyphs import calligraphy, typeset
from poly_ink.brush import RISE, SWEEP, splatter, sweep
from scenes.base import SceneBase
from timeline import at


class Scene(SceneBase):
    paper_seed = 20

    def build(self):
        t = at('Three scripts')[0] - 0.15
        self.t_wave = t
        ss = StrokeSet(W, H)
        face = [
            ([(70, 1500), (250, 1330), (430, 1150), (600, 985), (740, 885), (860, 852), (935, 892), (905, 962), (842, 962)], 125, 0.88, 0.6),
            ([(250, 1525), (420, 1352), (580, 1182), (722, 1042), (832, 962)], 98, 0.9, 0.62),
            ([(430, 1540), (580, 1382), (722, 1232), (842, 1122), (935, 1062)], 82, 0.8, 0.66),
            ([(30, 1405), (200, 1255), (360, 1095), (500, 965)], 60, 0.55, 0.62),
        ]
        for k, (pts, w, ink, dry) in enumerate(face):
            ss.add(sweep(W, H, pts, w, ink=ink, dry=dry, seed=100 + 7 * k, wash=0.3, profile=RISE), t + 0.13 * k, 0.55)
        t2 = t + 0.13 * len(face) + 0.35
        for k, (pts, w, ink, dry) in enumerate([
                ([(862, 852), (965, 905), (1045, 1005), (1110, 1130)], 82, 0.75, 0.6),
                ([(905, 885), (885, 822), (822, 792), (752, 802), (702, 852), (714, 902)], 50, 0.9, 0.5)]):
            ss.add(sweep(W, H, pts, w, ink=ink, dry=dry, seed=150 + 7 * k, wash=0.25), t2 + 0.12 * k, 0.4)
        ss.add(splatter(W, H, 800, 815, (170, 60), n=46, seed=41), t2 + 0.2, 0.3)
        # curling fingers of spray off the lip
        for k, (pts, w) in enumerate([([(760, 800), (720, 770), (690, 780), (688, 805), (705, 812)], 13),
                                      ([(820, 790), (800, 748), (770, 735), (752, 750), (762, 768)], 11),
                                      ([(700, 850), (650, 840), (630, 860), (642, 880)], 10)]):
            ss.add(stroke(W, H, spline(pts, 10), w, ink=0.85, dry=0.5, profile=((0, 0.5), (0.2, 1), (0.8, 0.7), (1, 0.1)),
                          seed=170 + k), t2 + 0.3 + 0.08 * k, 0.3)
        t3 = t2 + 0.55
        for k, (pts, w, ink, dry) in enumerate([
                ([(-20, 1570), (300, 1548), (700, 1588), (1100, 1562)], 112, 0.92, 0.62),
                ([(-20, 1668), (500, 1652), (1100, 1690)], 62, 0.6, 0.68)]):
            ss.add(sweep(W, H, pts, w, ink=ink, dry=dry, seed=200 + 7 * k, wash=0.22), t3 + 0.15 * k, 0.45)
        ss.add(splatter(W, H, 980, 1560, (120, 40), n=24, seed=43, direction=0.1), t3 + 0.35, 0.25)
        self.wave = ss

        self.t_sun = at('One language')[0] - 0.1
        self.sun = sun_disc(H, W, 700, 690, 230, seed=21)
        self.sun_T = wash_map(H, W, (700, 690), noise_amp=0.25, noise_scale=40, seed=22)

        self.t_title = at('Still written today')[0] - 0.05
        self.title = [calligraphy(c, 200, seed=60 + i) for i, c in enumerate('日本語')]
        self.title_dur = [0.42, 0.42, 0.9]
        self.halos = [blur(g.dens, 4) for g in self.title]
        self.t_brand = at('Still written today')[1] + 0.15
        self.brand, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 56, tracking=2)
        g = typeset('墨', 46, seed=3, medians=False, pad=0.12)
        hh, ww = g.h + 12, g.w + 12
        sq = np.zeros((hh, ww), np.float32)
        sq[3:-3, 3:-3] = 1
        sq = blur(sq, 1.0) * (0.82 + 0.18 * noise(hh, ww, 6, 2, 9))
        inner = np.zeros_like(sq)
        inner[6:6 + g.h, 6:6 + g.w] = g.dens
        self.seal = np.clip(sq * (1 - inner * 1.1), 0, 1)
        r = np.random.default_rng(8)
        self.motes = [(r.uniform(600, 1050), r.uniform(700, 950), r.uniform(0.6, 1.4), r.uniform(1.2, 2.6)) for _ in range(9)]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (600, 1000, 1.08)), (self.t1 - 1.5, (540, 960, 1.0)), (self.t1, (540, 960, 1.0))],
                              ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fs = (t - self.t_sun) / 1.1
        if fs > 0:
            self.layer(fr, cam, reveal_time(self.sun, self.sun_T, ease_out(fs) * 1.05, soft=0.06, wet=0.2), color=VERM)
        self.layer(fr, cam, self.wave.render(t))
        # drops still drifting down after the spray, settling onto the paper
        import cv2
        P = np.zeros((H, W), np.float32)
        M = cam.matrix()
        for x, y0, sp, rad in self.motes:
            age = t - self.t_wave - 1.0
            if age <= 0:
                continue
            y = y0 + 120 * sp * (1 - np.exp(-age * 0.9))
            sx, sy = M[:, :2] @ np.array([x, y]) + M[:, 2]
            cv2.circle(P, (int(sx), int(sy)), int(rad * cam.zoom + 0.5), 0.75, -1, cv2.LINE_AA)
        fr.ink(blur(P, 0.6), INK)
        # 日本語 down the left, written in the negative space
        ts = self.t_title
        for k, (g, du, halo) in enumerate(zip(self.title, self.title_dur, self.halos)):
            f = (t - ts) / du
            if f > 0:
                d = bloom(g.at(min(f, 1.05), soft=0.015, wet=0.45), t - ts - du, amount=0.12, halo=halo)
                self.layer(fr, cam, d, 205 - g.w // 2, 470 + 215 * k - g.h // 2)
            ts += du + 0.05
        fb = smooth((t - self.t_brand) / 0.7)
        if fb > 0:
            a = self.brand
            self.layer(fr, cam, a, 205 - a.shape[1] // 2, 1150 - a.shape[0] // 2, opacity=fb)
            self.layer(fr, cam, self.seal, 205 - self.seal.shape[1] // 2, 1215, color=VERM, opacity=smooth((t - self.t_brand - 0.4) / 0.5) * 0.9)
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        """a last brushstroke passes across the street; behind it, the street dissolves into open paper"""
        if not hasattr(self, '_pass'):
            st = stroke(W, H, spline([(-80, 1180), (300, 1020), (700, 960), (1160, 820)], 14), 260, ink=0.9, dry=0.62,
                        profile=SWEEP, seed=301)
            from engine import paste
            c = np.zeros((H, W), np.float32)
            paste(c, st)
            T = np.ones((H, W), np.float32) * 2
            hh, ww = st.dens.shape
            T[st.y0:st.y0 + hh, st.x0:st.x0 + ww] = np.where(st.dens > 0.01, st.time, 2)
            xx = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
            sweep_T = np.clip(xx * 0.85 + (noise(H, W, 100, 4, 61) - 0.5) * 0.3, 0, 1)
            self._pass = (c, T, sweep_T)
        c, T, sweep_T = self._pass
        head = f * 1.5
        vis = c * np.clip((head - T) / 0.04 + 0.5, 0, 1) * np.clip((T - (head - 0.55)) / 0.15, 0, 1)
        out = wipe_mix(prev_img, cur_img, sweep_T, f * 1.25 - 0.15, soft=0.1, edge=0.2)
        fr = Frame(out)
        fr.ink(vis, INK)
        return fr.img

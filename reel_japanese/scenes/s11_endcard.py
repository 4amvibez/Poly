"""Scene 11 — the end card.  Back to washi paper: a faint vermilion sun, misty ink mountains, drifting petals.
"Follow for more Japanese content" and @raj_in_jp with a seal; then, as Raj says it in Japanese,
最後まで見てくれてありがとう is written by brush, and またね！ is stamped in vermilion."""
import cv2
import numpy as np

from art import sun_disc
from engine import (FONT_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, SAKURA, VERM, W, Cam, StrokeSet,
                    blur, bloom, cached, ease_io, ease_out, keyframes, noise, reveal_time, smooth, stroke, text_layer,
                    wash_map, wipe_mix)
from glyphs import text_glyphs, typeset
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import JA, at


class Scene(SceneBase):
    paper_seed = 22

    def build(self):
        self.sun = sun_disc(H, W, 540, 640, 300, seed=31)
        self.sun_T = wash_map(H, W, (540, 640), noise_amp=0.25, noise_scale=40, seed=32)

        def mountains():
            from inkpaint import Painting
            p = Painting(W, H, seed=23)
            far = p.ridge(1300, peaks=[(150, 130, 180), (520, 80, 220), (940, 170, 170)], rough=12, seed=3)
            p.wash(far, dens=0.13, decay=110, tex=90, edge=0.3, mist_y=1400, mist_h=70, seed=4)
            near = p.ridge(1400, peaks=[(330, 100, 240), (820, 60, 230)], rough=16, seed=5)
            p.wash(near, dens=0.24, decay=70, tex=70, edge=0.4, mist_y=1470, mist_h=50, seed=6)
            D = p.D.copy()
            D[1560:] *= np.linspace(1, 0, H - 1560)[:, None]
            return {'D': D.astype(np.float32)}
        self.mount = cached('s11_mountains', mountains)['D']
        self.mount_T = wash_map(H, W, None, direction=(0.2, -1), noise_amp=0.5, noise_scale=120, seed=8)

        self.t_follow = at('Follow for more')[0] - 0.1
        self.follow, _ = text_layer('Follow for more Japanese content', FONT_EN, 50)
        self.handle, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 72, tracking=2)
        ul = StrokeSet(W, H)
        ul.add(stroke(W, H, [(380, 1246), (540, 1240), (700, 1247)], 7, ink=0.9, dry=0.45, profile=SWEEP, seed=4, red=True),
               self.t_follow + 0.5, 0.4)
        self.ul = ul
        g = typeset('墨', 46, seed=3, medians=False, pad=0.12)
        hh, ww = g.h + 12, g.w + 12
        sq = np.zeros((hh, ww), np.float32)
        sq[3:-3, 3:-3] = 1
        sq = blur(sq, 1.0) * (0.82 + 0.18 * noise(hh, ww, 6, 2, 9))
        inner = np.zeros_like(sq)
        inner[6:6 + g.h, 6:6 + g.w] = g.dens
        self.seal = np.clip(sq * (1 - inner * 1.1), 0, 1)

        # the Japanese sign-off, written in time with the voice
        a, b, m = JA['start'], JA['arigatou_end'], JA['matane']
        split = a + (b - a) * 0.6                               # 最後まで見てくれて | ありがとう
        self.ja = []
        for text, size, y, t0, t1, col in (('最後まで見てくれて', 90, 500, a - 0.05, split, INK),
                                            ('ありがとう', 124, 655, split - 0.05, b, INK),
                                            ('またね！', 140, 855, m - 0.08, m + 0.35, VERM)):
            D, T, _ = text_glyphs(text, size, font=FONT_JP, seed=hash(text) % 997, tracking=0.03)
            self.ja.append(dict(D=D, T=T, halo=blur(D, 4), y=y, t0=t0, t1=t1, col=col))
        self.tr, _ = text_layer('"Thanks for watching till the end. See you!"', FONT_EN_ITALIC, 42)
        self.t_tr = m + 0.4
        r = np.random.default_rng(12)
        self.petals = [(r.uniform(-50, 1100), r.uniform(-300, 1700), r.uniform(0.6, 1.2), r.uniform(0, 6)) for _ in range(14)]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.06)), (self.t1, (540, 950, 1.0))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fs = (t - self.t0 - 0.2) / 1.4
        if fs > 0:
            rise = 110 * (1 - ease_out((t - self.t0) / (JA['start'] - self.t0 + 0.6)))     # the sun climbs while we wait
            self.layer(fr, cam, reveal_time(self.sun, self.sun_T, ease_out(fs) * 1.05, soft=0.06, wet=0.2), 0, int(rise),
                       color=VERM, opacity=0.55)
        fm = (t - self.t0 - 0.3) / 2.0
        if fm > 0:
            self.layer(fr, cam, reveal_time(self.mount, self.mount_T, fm * 1.05, soft=0.08, wet=0.25))
        for L in self.ja:
            f = (t - L['t0']) / (L['t1'] - L['t0'])
            if f > 0:
                d = bloom(reveal_time(L['D'], L['T'], f * 1.08, soft=0.05, wet=0.45), t - L['t1'], amount=0.15, halo=L['halo'])
                punch = 1 + 0.1 * np.exp(-max(t - L['t0'], 0) * 5) if L['col'] is VERM else 1
                h, w = d.shape
                dd = cv2.resize(d, (int(w * punch), int(h * punch))) if punch > 1.002 else d
                self.layer(fr, cam, dd, 540 - dd.shape[1] // 2, L['y'] - dd.shape[0] // 2, color=L['col'])
        ot = smooth((t - self.t_tr) / 0.5)
        if ot > 0:
            self.layer(fr, cam, self.tr, 540 - self.tr.shape[1] // 2, 975 - self.tr.shape[0] // 2, color=GREY, opacity=ot)
        of = smooth((t - self.t_follow) / 0.5)
        if of > 0:
            self.layer(fr, cam, self.follow, 540 - self.follow.shape[1] // 2, 1105 - self.follow.shape[0] // 2, color=GREY, opacity=of)
            oh = smooth((t - self.t_follow - 0.2) / 0.5)
            self.layer(fr, cam, self.handle, 540 - self.handle.shape[1] // 2, 1190 - self.handle.shape[0] // 2, opacity=oh)
            self.layer(fr, cam, self.ul.render(t, red=True), color=VERM)
            self.layer(fr, cam, self.seal, 540 + self.handle.shape[1] // 2 + 26, 1190 - self.seal.shape[0] // 2, color=VERM,
                       opacity=smooth((t - self.t_follow - 0.6) / 0.4) * 0.9)
        # drifting petals
        P = np.zeros((H, W), np.float32)
        for x0, y0, sp, ph in self.petals:
            y = (y0 + (t - self.t0) * 60 * sp) % 1900
            x = x0 + (t - self.t0) * 18 * sp + 22 * np.sin(t * 1.1 + ph)
            cv2.ellipse(P, (int(x), int(y)), (7, 4), (t * 2 + ph) * 57, 0, 360, 0.5, -1, cv2.LINE_AA)
        fr.ink(blur(P, 0.7), SAKURA, smooth((t - self.t0) / 1.0))
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the night street dissolves through an ink wash into open paper"""
        if not hasattr(self, '_T'):
            self._T = wash_map(H, W, (540, 900), noise_amp=0.45, noise_scale=110, seed=77)
        return wipe_mix(prev_img, cur_img, self._T, ease_io(f) * 1.15 - 0.05, soft=0.04, edge=0.45)

"""Scene 1 — the hook: "Did you know that some Japanese kanji started as pictures?"
The title card of the storyboard, alive: How was KANJI created? over a vermilion sun, misty mountains, a pagoda,
a blossom branch and a figure seen from behind.  On "pictures?" a brush rings the sun with the oracle-bone sun
(a circle with a mark inside), and 日 is written into it: the picture becomes the character."""
import numpy as np

from art import cherry_branch, figure_back, blossom_cluster, pagoda, sun_disc
from engine import (FONT_BRUSH_EN, FONT_HEAD, H, INK, SAKURA, VERM, W, Cam, StrokeSet, blur, bloom, cached, ease_io,
                    ease_out, keyframes, noise, reveal_time, smooth, stroke, text_layer, wash_map)
from glyphs import calligraphy
from scenes.base import SceneBase, dof
from timeline import at

SUN = (540, 1035, 180)


class Scene(SceneBase):
    paper_seed = 41
    paper_stains = 0.1

    def build(self):
        cx, cy, r = SUN
        self.sun = sun_disc(H, W, cx, cy, r, seed=41)

        def land():
            from inkpaint import Painting
            p = Painting(W, H, seed=42)
            far = p.ridge(1150, peaks=[(120, 140, 200), (560, 230, 260), (980, 120, 200)], rough=14, seed=1)
            p.wash(far, dens=0.22, decay=130, tex=90, edge=0.3, mist_y=1240, mist_h=60, seed=2)
            D_far = p.D.copy(); p.D[:] = 0
            near = p.ridge(1290, peaks=[(160, 70, 260), (900, 90, 280)], rough=12, seed=3)
            p.wash(near, dens=0.42, decay=70, tex=70, edge=0.4, mist_y=1360, mist_h=40, seed=4)
            D_near = p.D.copy()
            D_near[1520:] *= np.linspace(1, 0, H - 1520)[:, None]
            return {'far': D_far, 'near': D_near}
        L = cached('k01_land_v2', land)
        self.far, self.near = L['far'], L['near']

        # pagoda on the right ridge, blossom branch from the top-right corner, figure in the foreground
        dec = StrokeSet(W, H)
        pagoda(dec, 868, 1262, 0.95, 0, 0.01, seed=60)
        tips = cherry_branch(dec, [(1130, 110), (1015, 170), (905, 158), (815, 228), (725, 214), (650, 262)], 22, 0, 0.01, seed=70)
        self.deco = dec.render(9)
        bl = np.zeros((H, W), np.float32)
        for k, (bx, by) in enumerate(tips):
            bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 14, 46, seed=80 + k))
        self.blossom = bl
        fig = StrokeSet(W, H)
        figure_back(fig, 250, 1530, 1.65, 0, seed=90, dur=0.01)
        self.figure = fig.render(9)

        # title: How was / KANJI / created?
        self.t_title = 0.0
        self.l1, _ = text_layer('How was', FONT_HEAD, 104)
        self.l3, _ = text_layer('created?', FONT_HEAD, 118)
        k, _ = text_layer('KANJI', FONT_BRUSH_EN, 290)
        hh, ww = k.shape
        # a dry-brush reveal, left to right, with streaks along the stroke direction
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        T = xx / ww * 0.85 + (noise(hh, ww, 18, 3, 5) - 0.5) * 0.18 + (yy / hh - 0.5) * 0.08
        self.kanji = (k * (0.86 + 0.14 * noise(hh, ww, 6, 2, 6)), np.clip(T, 0, 1), blur(k, 6))
        self.t_kanji = -0.25                     # half brushed in the very first frame
        self.t_pulse = at('kanji')[0]

        # "pictures?": the oracle-bone sun painted round the real one, then 日 written inside
        p0, p1 = at('pictures')
        from timeline import quiet_gap
        self.t_hit, _ = quiet_gap(p1 - 0.1, at('Thousands')[0] + 0.05)     # 日 lands in the breath after "pictures?"
        self.t_ring = self.t_hit - 0.45 - 0.62
        ring = StrokeSet(W, H)
        a = np.linspace(-2.2, -2.2 + 2 * np.pi * 1.02, 40)
        rr = r * 1.28 * (1 + 0.02 * np.sin(a * 3))
        ring.add(stroke(W, H, [(cx + q * np.cos(b), cy + q * np.sin(b) * 0.97) for q, b in zip(rr, a)], 24, ink=0.95, dry=0.5,
                        profile=((0, 0.6), (0.08, 1), (0.85, 0.9), (1, 0.4)), seed=21), self.t_ring, 0.42, bloom_amt=0.15)
        ring.add(stroke(W, H, [(cx - 70, cy + 4), (cx, cy - 2), (cx + 72, cy + 3)], 26, ink=0.95, dry=0.3,
                        profile=((0, 0.7), (0.2, 1), (1, 0.6)), seed=22), self.t_ring + 0.45, 0.14)
        self.ring = ring
        self.g = calligraphy('日', 300, seed=23)
        self.t_ji = self.t_ring + 0.62
        self.t_ji_end = self.t_hit

    def cam(self, t):
        return Cam(*keyframes(t, [(0.0, (540, 930, 1.04)), (self.t_ring, (540, 960, 1.0)), (self.t1, (540, 1000, 1.06))],
                              ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        arrive = smooth(t / 0.5) * 0.25 + 0.75                     # everything is already on the page in frame 1
        self.layer(fr, cam, self.sun, color=VERM, opacity=0.95 * arrive * (1 - 0.35 * smooth((t - self.t_ji) / 0.4)),
                   depth=0.85)
        self.layer(fr, cam, self.far, depth=0.88)
        self.layer(fr, cam, self.near, depth=0.95)
        self.layer(fr, cam, self.deco, opacity=0.9)
        self.layer(fr, cam, self.blossom, color=SAKURA, opacity=0.9)
        self.layer(fr, cam, self.figure, depth=1.1)
        # title
        self.place(fr, cam, self.l1, 540, 300)
        D, T, halo = self.kanji
        fk = (t - self.t_kanji) / 0.55
        if fk > 0:
            d = bloom(reveal_time(D, T, fk * 1.1, soft=0.06, wet=0.5), t - self.t_kanji - 0.5, amount=0.18, halo=halo)
            pulse = 1 + 0.05 * np.exp(-max(t - self.t_pulse, 0) * 6) * (t > self.t_pulse)
            self.place(fr, cam, d, 535, 505, color=VERM, scale=pulse)
        self.place(fr, cam, self.l3, 540, 700, opacity=smooth((t - 0.2) / 0.35))
        # the picture becomes the character
        self.layer(fr, cam, self.ring.render(t), opacity=1 - 0.7 * smooth((t - self.t_ji_end) / 0.3))
        fj = (t - self.t_ji) / (self.t_ji_end - self.t_ji)
        if fj > 0:
            cx, cy, _ = SUN
            d = self.g.at(min(fj, 1.05), soft=0.02, wet=0.45)
            halo = np.clip(blur(d, 6) * 1.5, 0, 1)
            fr.over(np.array([0.95, 0.92, 0.85], np.float32), self.project(cam, halo, cx - self.g.w // 2, cy - self.g.h // 2), 0.25)
            self.layer(fr, cam, d, cx - self.g.w // 2, cy - self.g.h // 2)
        return dof(fr.img, 0.35 * (1 - smooth(t / 0.8)), focus_y=0.3, band=0.2)

    def exit_map(self):
        cx, cy, _ = SUN
        return wash_map(H, W, (cx, cy), noise_amp=0.4, noise_scale=120, seed=5)

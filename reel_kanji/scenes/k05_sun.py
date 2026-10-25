"""Scene 5 — 日, "hi", the sun: from picture to character.
  A  "Look at this character: 'hi', meaning sun."  the brush writes 日 large, in stroke order; hi · "sun"
  B  "Early forms resembled a drawing of the sun, often with a mark inside."  a painted sun beside the oracle-bone
     form; the mark inside is ringed in vermilion
  C  "Over centuries, its shape changed through different writing styles, until it became the character we
     recognize today."  the oracle form moves to the top of a column and the later styles follow, one per row,
     each with its name and period: 甲骨文 → 金文 → 篆書 → 隷書 → 楷書; a seal-red frame lands on today's form."""
import numpy as np

from ancient import STAGES, ancient
from art import brush_sprite, draw_sprite, sun_disc
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, bloom, cached,
                    ease_io, ease_out, keyframes, lerp, reveal_time, smooth, stroke, text_layer, wash_map)
from glyphs import calligraphy, typeset
from poly_ink.brush import PRESS_DOWN
from scenes.base import SceneBase
from timeline import at

ROW_Y = [360, 565, 770, 975, 1180]
GS = 150                                   # glyph size in the column
GX = 330                                   # glyph column
LX = 470                                   # label column (left edge)


class Scene(SceneBase):
    paper_seed = 45
    paper_stains = 0.1

    def build(self):
        # A: the big 日
        self.big = calligraphy('日', 440, seed=3)
        self.t_write = at('Look at this')[0] + 0.25
        self.t_write_end = at('hi')[0] + 0.2
        self.brush = brush_sprite(seed=4)
        self.hi, _ = text_layer('hi', FONT_EN_ITALIC, 96)
        self.hi_en, _ = text_layer('"sun"', FONT_EN, 54)
        self.t_hi = at('hi')[0] - 0.05
        self.top_sun = sun_disc(H, W, 540, 330, 112, seed=8)
        def land():
            from inkpaint import Painting
            p = Painting(W, H, seed=46)
            r1 = p.ridge(560, peaks=[(150, 120, 160), (520, 70, 200), (930, 140, 150)], rough=10, seed=2)
            p.wash(r1, dens=0.16, decay=90, tex=70, edge=0.3, mist_y=600, mist_h=40, seed=3)
            D = p.D.copy()
            D[700:] = 0
            return {'D': D}
        self.hills = cached('k05_hills', land)['D']
        # B: a painted sun beside the oracle form
        self.t_early = at('Early forms')[0]
        self.t_mark = at('mark inside')[0] - 0.05
        self.p_sun = sun_disc(H, W, 300, 880, 150, seed=9)
        self.p_sun_T = wash_map(H, W, (300, 880), noise_amp=0.3, noise_scale=40, seed=10)
        self.stages = []
        for k, (style, jp, en, period) in enumerate(STAGES):
            g = calligraphy('日', GS, seed=20 + k) if style == 'regular' else ancient('日', style, GS, seed=20 + k)
            self.stages.append(dict(g=g, jp=text_layer(jp, FONT_JP, 54)[0], en=text_layer(en, FONT_EN, 48)[0],
                                    per=text_layer(period, FONT_EN_ITALIC, 38)[0]))
        self.oracle_big = ancient('日', 'oracle', 300, seed=20)
        ring = StrokeSet(W, H)
        a = np.linspace(-2.5, -2.5 + 2 * np.pi * 1.05, 36)
        ring.add(stroke(W, H, [(760 + 86 * np.cos(b), 880 + 50 * np.sin(b)) for b in a], 9, ink=0.95, dry=0.45,
                        profile=((0, 0.5), (0.1, 1), (0.9, 0.9), (1, 0.3)), seed=11, red=True), self.t_mark, 0.4)
        self.ring = ring
        arr = StrokeSet(W, H)
        arr.add([stroke(W, H, [(482, 884), (560, 878), (612, 882)], 5, ink=0.7, dry=0.5, profile=PRESS_DOWN, seed=12),
                 stroke(W, H, [(592, 864), (614, 882), (592, 900)], 5, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=13)],
                self.t_early + 0.75, 0.3)
        self.arrow_b = arr
        # C: the column
        self.t_over = at('Over centuries')[0] - 0.2
        self.t_rows = [self.t_over + 0.15] + [at('its shape')[0] + 0.1 + 0.75 * k for k in range(4)]
        from timeline import quiet_gap
        self.t_today = at('today')[0]
        self.t_hit, _ = quiet_gap(at('today')[1] - 0.15, at('And the idea')[0] + 0.05)   # the frame closes in the breath
        arrows = StrokeSet(W, H)
        for k in range(4):
            y0, y1 = ROW_Y[k] + 84, ROW_Y[k + 1] - 84
            arrows.add([stroke(W, H, [(GX, y0), (GX + 2, y1)], 4, ink=0.6, dry=0.5, profile=PRESS_DOWN, seed=30 + k),
                        stroke(W, H, [(GX - 12, y1 - 14), (GX + 2, y1), (GX + 15, y1 - 13)], 4, ink=0.6, dry=0.4,
                               profile=PRESS_DOWN, seed=40 + k)], self.t_rows[k + 1] - 0.3, 0.3)
        self.arrows = arrows
        frame = StrokeSet(W, H)
        y = ROW_Y[4]
        pts = [(GX - 92, y - 90), (GX + 92, y - 92), (GX + 94, y + 92), (GX - 90, y + 94), (GX - 92, y - 86)]
        frame.add(stroke(W, H, pts, 8, ink=0.95, dry=0.5, profile=((0, 0.6), (0.05, 1), (0.95, 0.9), (1, 0.5)), seed=50,
                         red=True), self.t_hit - 0.35, 0.35, bloom_amt=0.15)
        self.frame_red = frame

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.06)), (self.t_write_end, (540, 960, 1.0)),
                                  (self.t_over, (540, 950, 1.0)), (self.t_rows[-1] + 0.6, (540, 900, 1.0)),
                                  (self.t_today, (540, 900, 1.0)), (self.t1, (540, 880, 1.06))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        # A
        outA = 1 - smooth((t - self.t_early + 0.1) / 0.45)
        sA = smooth((t - self.t0) / 0.6) * (1 - smooth((t - self.t_early) / 0.8))
        if sA > 0:
            self.layer(fr, cam, self.top_sun, color=VERM, opacity=0.75 * sA, depth=0.9)
            self.layer(fr, cam, self.hills, opacity=sA, depth=0.92)
        f = (t - self.t_write) / (self.t_write_end - self.t_write)
        if f > 0 and outA > 0:
            g = self.big
            d = bloom(g.at(min(f, 1.05), soft=0.012, wet=0.45), t - self.t_write_end, amount=0.12)
            self.layer(fr, cam, d, 540 - g.w // 2, 880 - g.h // 2, opacity=outA)
            oh = smooth((t - self.t_hi) / 0.35) * outA
            self.place(fr, cam, self.hi, 470, 1180, color=VERM, opacity=oh)
            self.place(fr, cam, self.hi_en, 610, 1190, color=GREY, opacity=smooth((t - self.t_hi - 0.5) / 0.4) * outA)
            if t < self.t_write_end + 0.9:
                self._brush(fr, cam, t, f)
        # B
        fb = (t - self.t_early - 0.15) / 0.8
        outB = 1 - smooth((t - self.t_over) / 0.6)
        if fb > 0 and outB > 0:
            self.layer(fr, cam, reveal_time(self.p_sun, self.p_sun_T, fb * 1.1, soft=0.08, wet=0.3), color=VERM, opacity=0.9 * outB)
        # the oracle form: carved large at the right in B, then it travels to the top of the column
        fo = (t - self.t_early - 0.9) / 0.9
        if fo > 0:
            mv = ease_io((t - self.t_over) / 0.9)
            g = self.oracle_big
            x, y, sc = lerp(760, GX, mv), lerp(880, ROW_Y[0], mv), lerp(1.0, GS / 300, mv)
            self.place(fr, cam, g.at(fo, soft=0.03, wet=0.3), x, y, scale=sc)
            self.layer(fr, cam, self.ring.render(t, red=True), color=VERM, opacity=outB)
        self.layer(fr, cam, self.arrow_b.render(t), color=GREY, opacity=outB)
        # C: the column of styles
        for k, st in enumerate(self.stages):
            tr = self.t_rows[k]
            fr_ = (t - tr) / (0.6 if k < 4 else 0.7)
            if fr_ <= 0:
                continue
            y = ROW_Y[k]
            if k > 0:
                self.place(fr, cam, st['g'].at(fr_, soft=0.03, wet=0.35), GX, y)
            lo = smooth(fr_ * 1.4 - 0.4)
            for a, dy, col in ((st['jp'], -52, VERM), (st['en'], 6, INK), (st['per'], 56, GREY)):
                h, w = a.shape
                self.layer(fr, cam, a, LX, y + dy - h // 2, color=col, opacity=lo)
        self.layer(fr, cam, self.arrows.render(t), color=GREY)
        self.layer(fr, cam, self.frame_red.render(t, red=True), color=VERM)
        return fr.img

    def _brush(self, fr, cam, t, f):
        g = self.big
        x, y, k, u, moving = g.brush_pos(np.clip(f, 0, 1))
        wx, wy = 540 - g.w // 2 + x, 880 - g.h // 2 + y
        M = cam.matrix()
        sx, sy = M[:, :2] @ np.array([wx, wy]) + M[:, 2]
        lift = 0 if moving else 30
        if f <= 0:
            lift = 60
        if f > 1:
            lift = 30 + 700 * ease_out((t - self.t_write_end) / 0.9)
        draw_sprite(fr, *self.brush, sx, sy - lift * 0.5, -0.3, scale=0.62 * cam.zoom, shadow=0.3, lift=lift)

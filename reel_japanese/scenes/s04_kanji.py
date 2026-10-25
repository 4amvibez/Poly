"""Scene 4 — the adoption of kanji.  A manuscript page: the brush writes 字 ("character") in its proper stroke
order, beginning in close-up on the brush tip; the camera pulls back over ruled columns of classical text (the
opening of the Thousand Character Classic, 千字文).  Then the adaptation, with a real Man'yoshu-style example:
山 written for its meaning (yama, "mountain") beside 夜麻 written purely for its sound (ya + ma)."""
import numpy as np

from art import brush_sprite, draw_sprite
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP_FALLBACK, GREY, H, INK, SEPIA, VERM, W, Cam, Frame, StrokeSet,
                    blur, bloom, ease_io, ease_out, keyframes, noise, ramp, reveal_time, smooth, sstep, stroke,
                    text_layer, wipe_mix)
from glyphs import calligraphy, typeset
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

CLASSIC = '天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰致雨露結為霜金生麗水玉出崑岡劍號巨闕珠稱夜光'


class Scene(SceneBase):
    paper_seed = 14
    paper_stains = 0.14

    def build(self):
        self.g = calligraphy('字', 520, seed=7)
        self.gx, self.gy = 540 - self.g.w // 2, 745 - self.g.h // 2
        self.halo = blur(self.g.dens, 5)
        self.t_write = at('These characters')[0] + 0.25
        self.t_write_end = at('kanji')[1] + 1.6
        self.brush = brush_sprite(seed=3)

        # ruled columns and faint classical text
        rule = np.zeros((H, W), np.float32)
        for x in range(90, 1000, 112):
            rule[200:1420, x:x + 2] = 1
        rule[198:201, 90:992] = 1
        rule[1418:1421, 90:992] = 1
        self.rules = blur(rule, 0.7) * 0.5
        txt = np.zeros((H, W), np.float32)
        k = 0
        for col in (0, 1, 7):
            x = 90 + 112 * col + 56
            for j in range(17):
                g = typeset(CLASSIC[k % len(CLASSIC)], 62, font=FONT_JP_FALLBACK, seed=k, pad=0.05, ink=0.9, medians=False)
                k += 1
                y = 240 + j * 69
                txt[y:y + g.h, x - g.w // 2:x - g.w // 2 + g.w] = np.maximum(txt[y:y + g.h, x - g.w // 2:x - g.w // 2 + g.w], g.dens)
        self.classic = txt
        from engine import wash_map
        self.classic_T = wash_map(H, W, (540, 745), noise_amp=0.4, noise_scale=90, seed=4)

        # label: 漢字 kanji
        self.kanji_lbl = typeset('漢', 92, seed=1, medians=False), typeset('字', 92, seed=2, medians=False)
        self.kanji_en, _ = text_layer('kanji  ·  "Han characters"', FONT_EN_ITALIC, 46)
        self.t_lbl = at('kanji')[0] - 0.1

        # meaning vs sound
        self.yama = typeset('山', 165, font=FONT_JP_FALLBACK, seed=5, medians=False)
        self.ya = typeset('夜', 128, font=FONT_JP_FALLBACK, seed=6, medians=False)
        self.ma = typeset('麻', 128, font=FONT_JP_FALLBACK, seed=8, medians=False)
        self.t_mean = at('some for their meaning')[0] - 0.1
        self.t_sound = at('others purely')[0] - 0.1
        self.txt = {k: text_layer(s, f, z)[0] for k, s, f, z in [
            ('yama', 'yama', FONT_EN_ITALIC, 46), ('mean', 'by meaning: "mountain"', FONT_EN, 34),
            ('yama2', 'ya + ma', FONT_EN_ITALIC, 46), ('sound', 'by sound alone', FONT_EN, 34)]}
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [(552, 1070), (550, 1190), (554, 1300)], 4, ink=0.5, dry=0.5, profile=SWEEP, seed=9), self.t_mean + 0.4, 0.5)
        self.divider = ss
        self.seal = self._seal()

    def _seal(self):
        g = typeset('墨', 64, seed=3, medians=False, pad=0.12)
        h, w = g.h + 16, g.w + 16
        sq = np.zeros((h, w), np.float32)
        sq[4:-4, 4:-4] = 1
        sq = blur(sq, 1.2) * (0.82 + 0.18 * noise(h, w, 6, 2, 3))
        inner = np.zeros_like(sq)
        inner[8:8 + g.h, 8:8 + g.w] = g.dens
        return np.clip(sq * (1 - inner * 1.1), 0, 1)

    def cam(self, t):
        x0, y0 = self.g.medians[0][0]
        sx, sy = self.gx + x0, self.gy + y0
        a = self.t_write
        return Cam(*keyframes(t, [(self.t0, (sx + 40, sy + 120, 2.6)), (a, (sx + 30, sy + 90, 2.4)),
                                  (a + 1.2, (520, 760, 1.6)), (self.t_write_end, (540, 900, 1.06)),
                                  (self.t1, (540, 930, 1.0))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fc = ramp(t, self.t_write + 0.8, self.t_write_end + 0.8)
        if fc > 0:
            dim = 1 - 0.45 * smooth((t - self.t_mean + 0.6) / 0.8)
            self.layer(fr, cam, self.rules * smooth(fc * 1.5), color=VERM, opacity=0.5 * dim)
            self.layer(fr, cam, reveal_time(self.classic, self.classic_T, fc * 1.1, soft=0.1, wet=0.1), color=SEPIA, opacity=0.5 * dim)
            self.layer(fr, cam, self.seal, 128, 1300, color=VERM, opacity=smooth(fc * 2 - 1) * 0.85)

        f = (t - self.t_write) / (self.t_write_end - self.t_write)
        if f > 0:
            d = self.g.at(min(f, 1.05), soft=0.012, wet=0.45)
            d = bloom(d, t - self.t_write_end, amount=0.12, halo=self.halo)
            self.layer(fr, cam, d, self.gx, self.gy)

        # 漢字 label
        fl = (t - self.t_lbl) / 0.5
        if fl > 0:
            for k, g in enumerate(self.kanji_lbl):
                self.layer(fr, cam, g.at(fl - k * 0.4, soft=0.05), 455 + k * 95 - g.w // 2, 300 - g.h // 2)
            op = smooth(fl - 1.0)
            if op > 0:
                a = self.kanji_en
                self.layer(fr, cam, a, 540 - a.shape[1] // 2, 400 - a.shape[0] // 2, color=GREY, opacity=op)

        # meaning | sound
        fm = (t - self.t_mean) / 0.6
        if fm > 0:
            self.layer(fr, cam, self.yama.at(fm, soft=0.05), 410 - self.yama.w // 2, 1055)
            for i, (k, y) in enumerate((('yama', 1252), ('mean', 1296))):
                a = self.txt[k]
                self.layer(fr, cam, a, 410 - a.shape[1] // 2, y - a.shape[0] // 2, color=INK if i == 0 else GREY, opacity=smooth(fm - 0.6 - 0.3 * i))
            self.layer(fr, cam, self.divider.render(t), color=GREY)
        fs = (t - self.t_sound) / 0.6
        if fs > 0:
            self.layer(fr, cam, self.ya.at(fs, soft=0.05), 632 - self.ya.w // 2, 1078)
            self.layer(fr, cam, self.ma.at(fs - 0.5, soft=0.05), 764 - self.ma.w // 2, 1078)
            for i, (k, y) in enumerate((('yama2', 1252), ('sound', 1296))):
                a = self.txt[k]
                self.layer(fr, cam, a, 698 - a.shape[1] // 2, y - a.shape[0] // 2, color=INK if i == 0 else GREY, opacity=smooth(fs - 0.8 - 0.3 * i))

        # the brush, writing
        if self.t0 <= t <= self.t_write_end + 0.9:
            self._brush(fr, cam, t, f)
        return fr.img

    def _brush(self, fr, cam, t, f):
        x, y, k, u, moving = self.g.brush_pos(np.clip(f, 0, 1))
        wx, wy = self.gx + x, self.gy + y
        M = cam.matrix()
        sx, sy = M[:, :2] @ np.array([wx, wy]) + M[:, 2]
        lift = 0 if moving else 30
        if f <= 0:
            lift = 60 * (1 - smooth((t - self.t0) / max(self.t_write - self.t0, 0.1)))
        if f > 1:
            lift = 30 + 600 * ease_out((t - self.t_write_end) / 0.9)
        # lean into the direction of travel
        x2, y2, *_ = self.g.brush_pos(np.clip(f + 0.01, 0, 1))
        lean = np.clip((x2 - x) * 0.01, -0.12, 0.12)
        draw_sprite(fr, *self.brush, sx, sy - lift * cam.zoom * 0.5, -0.32 + lean, scale=0.62 * cam.zoom, shadow=0.3, lift=lift)

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the map's parchment becomes the manuscript: a sepia-edged dissolve through paper"""
        T = noise(H, W, 150, 5, 404)
        T = (T - T.min()) / (T.max() - T.min())
        return wipe_mix(prev_img, cur_img, T, ease_io(f) * 1.2 - 0.1, soft=0.12, edge=0.12, edge_color=SEPIA)

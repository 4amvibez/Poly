"""Scene 8 — modern Japanese in practice.  私は日本語を勉強します。 "I study Japanese."  Written group by group as
the romaji is spoken; then the roles: kanji word groups (私, 日本語, 勉強) underlined in ink as "kanji form the core
words", and the hiragana (は, を, します) warming to vermilion as "hiragana link them, and complete the verb".
A quiet legend notes that this sentence happens to use no katakana."""
import numpy as np

from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, Frame, StrokeSet, blur, bloom, ease_io,
                    keyframes, noise, ramp, reveal_time, smooth, spline, stroke, text_layer, warp)
from glyphs import typeset
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

LINES = [  # (chars, kind, romaji, narration word)
    [('私', 'k', 'watashi', 'Watashi'), ('は', 'h', 'wa', 'wa'), ('日本語', 'k', 'nihongo', 'nihongo'), ('を', 'h', 'o', 'o')],
    [('勉強', 'k', 'benkyō', 'benkyou'), ('します', 'h', 'shimasu', 'shimasu'), ('。', 'p', '', 'shimasu')],
]
SIZE = 128
STEP = 136


class Scene(SceneBase):
    paper_seed = 18

    def build(self):
        self.groups = []
        t_wat = at('Watashi')[0]
        for li, line in enumerate(LINES):
            n = sum(len(c) for c, *_ in line)
            x = 540 - STEP * n / 2 + STEP / 2
            y = 620 + li * 250
            for chars, kind, rom, word in line:
                ts = at(word)[0] - 0.12 if word != 'wa' else at('Watashi wa')[1] - 0.25
                if word == 'o':
                    ts = at('nihongo o')[1] - 0.2
                if chars == '。':
                    ts = at('shimasu')[1] - 0.1
                gl = []
                for i, c in enumerate(chars):
                    gl.append((typeset(c, SIZE, font=FONT_JP, seed=hash(c) % 1000, medians=None if c in 'あイ' else False), x + i * STEP))
                x0, x1 = gl[0][1] - SIZE / 2, gl[-1][1] + SIZE / 2
                rom_a = text_layer(rom, FONT_EN_ITALIC, 40)[0] if rom else None
                self.groups.append(dict(gl=gl, kind=kind, y=y, ts=ts, dur=0.22 * len(chars) + 0.15, x0=x0, x1=x1, rom=rom_a))
                x += STEP * len(chars)
        self.t_trans = at('I study')[0] - 0.1
        self.trans, _ = text_layer('"I study Japanese."', FONT_EN_ITALIC, 66)
        self.t_kanji = at('Kanji form')[0] - 0.05
        self.t_hira = at('Hiragana link')[0] - 0.05
        self.t_verb = at('complete the verb')[0]
        ul = StrokeSet(W, H)
        k = 0
        for g in self.groups:
            if g['kind'] == 'k':
                yy = g['y'] + 82
                ul.add(stroke(W, H, [(g['x0'] + 6, yy), ((g['x0'] + g['x1']) / 2, yy - 3), (g['x1'] - 6, yy + 1)], 7, ink=0.85,
                              dry=0.4, profile=SWEEP, seed=300 + k), self.t_kanji + 0.25 * k, 0.35)
                k += 1
        self.ul_k = ul
        ulh = StrokeSet(W, H)
        k = 0
        for g in self.groups:
            if g['kind'] == 'h':
                yy = g['y'] + 82
                ulh.add(stroke(W, H, [(g['x0'] + 6, yy), ((g['x0'] + g['x1']) / 2, yy - 3), (g['x1'] - 6, yy + 1)], 7, ink=0.9,
                               dry=0.4, profile=SWEEP, seed=320 + k, red=True), self.t_hira + 0.25 * k, 0.35)
                k += 1
        self.ul_h = ulh
        self.leg = {k: text_layer(s, f, z)[0] for k, s, f, z in [
            ('k', 'kanji: 私 · 日本語 · 勉強', FONT_EN, 40), ('h', 'hiragana: は · を · します', FONT_EN, 40),
            ('kk', 'katakana: none in this sentence', FONT_EN_ITALIC, 36)]}
        # legend glyph runs need the Japanese font: render the legend as English label + Japanese glyph strip
        self.leg_jp = {}
        for key, txt in (('k', '私・日本語・勉強'), ('h', 'は・を・します')):
            gs = [typeset(c, 44, font=FONT_JP, seed=400 + i, pad=0.04, medians=False) for i, c in enumerate(txt)]
            w = sum(g.w for g in gs)
            strip = np.zeros((gs[0].h, w), np.float32)
            x = 0
            for g in gs:
                strip[:, x:x + g.w] = np.maximum(strip[:, x:x + g.w], g.dens)
                x += g.w
            self.leg_jp[key] = strip
        self.leg_en = {k: text_layer(s, FONT_EN, 40)[0] for k, s in (('k', 'kanji'), ('h', 'hiragana'))}

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 880, 1.06)), (self.t_trans, (540, 900, 1.0)), (self.t1, (540, 920, 1.03))],
                              ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        hira = smooth((t - self.t_hira - 0.1) / 0.5)
        kan = smooth((t - self.t_kanji) / 0.4) * (1 - smooth((t - self.t_hira) / 0.6))
        for g in self.groups:
            f = (t - g['ts']) / g['dur']
            if f <= 0:
                continue
            n = len(g['gl'])
            for i, (gl, x) in enumerate(g['gl']):
                fi = f * n - i
                if fi <= 0:
                    continue
                d = gl.at(min(fi, 1.05), soft=0.05, wet=0.4)
                if g['kind'] == 'h' and hira > 0:
                    self.layer(fr, cam, d, int(x - gl.w / 2), int(g['y'] - gl.h / 2), opacity=1 - hira)
                    self.layer(fr, cam, d, int(x - gl.w / 2), int(g['y'] - gl.h / 2), color=VERM, opacity=hira)
                else:
                    op = 1.0 if g['kind'] == 'k' or kan == 0 else 1 - 0.55 * kan
                    self.layer(fr, cam, d, int(x - gl.w / 2), int(g['y'] - gl.h / 2), opacity=op)
            if g['rom'] is not None:
                a = g['rom']
                self.layer(fr, cam, a, int((g['x0'] + g['x1']) / 2 - a.shape[1] / 2), int(g['y'] + 120 - a.shape[0] / 2),
                           color=GREY, opacity=smooth(f - 0.4))
        self.layer(fr, cam, self.ul_k.render(t), color=INK)
        self.layer(fr, cam, self.ul_h.render(t, red=True), color=VERM)
        ft = smooth((t - self.t_trans) / 0.6)
        if ft > 0:
            a = self.trans
            self.layer(fr, cam, a, 540 - a.shape[1] // 2, 1075 - a.shape[0] // 2, opacity=ft)
        # legend
        for k, (key, tt, col) in enumerate((('k', self.t_kanji + 0.6, INK), ('h', self.t_hira + 0.6, VERM))):
            op = smooth((t - tt) / 0.5)
            if op > 0:
                y = 1195 + 54 * k
                en, jp = self.leg_en[key], self.leg_jp[key]
                tot = en.shape[1] + 24 + jp.shape[1]
                x0 = 540 - tot // 2
                self.layer(fr, cam, en, x0, y - en.shape[0] // 2, color=GREY, opacity=op)
                self.layer(fr, cam, jp, x0 + en.shape[1] + 24, y - jp.shape[0] // 2, color=col, opacity=op)
        op = smooth((t - self.t_verb - 0.3) / 0.5)
        if op > 0:
            a = self.leg['kk']
            self.layer(fr, cam, a, 540 - a.shape[1] // 2, 1305 - a.shape[0] // 2, color=GREY, opacity=op * 0.9)
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the three bands converge toward the centre and dissolve into the sentence"""
        import cv2
        e = ease_io(f)
        sy = 1 - 0.9 * e
        cy = 760                                      # the bands converge on the line where the sentence will be written
        M = np.array([[1 - 0.12 * e, 0, W / 2 * 0.12 * e], [0, sy, cy * (1 - sy)]], np.float32)
        squeezed = cv2.warpAffine(prev_img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        band = np.zeros((H, W), np.float32)
        top, bot = int(cy - (cy - 0) * sy), int(cy + (H - cy) * sy)
        band[max(0, top):min(H, bot)] = 1
        band = cv2.GaussianBlur(band, (0, 0), 6 + 30 * e)
        a = band[..., None] * (1 - e) ** 0.6
        return squeezed * a + cur_img * (1 - a)


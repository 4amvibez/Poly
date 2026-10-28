"""Scene 6 — kanji made in Japan.  "And here's something surprising.  Some kanji weren't made in China at all.  Japan
created its own, by combining parts.  They're called こくじ, or わせいかんじ.  Person, plus move, makes はたらく: to
work.  Mountain, up, and down, makes とうげ: a mountain pass.  And most of them have only a Japanese reading."
"Some kanji weren't made in China at all": NOT FROM CHINA is set, and the small junk turns back off the page.  "Japan
created its own": a vermilion sun blooms over misty hills and MADE IN JAPAN is stamped across it.  On こくじ the
word 国字 is brushed in large with こくじ · KOKUJI under it, on わせいかんじ 和製漢字 わせいかんじ · WASEI KANJI below it.
"Person, plus move": the sun clears and the parts come in one after the other, 亻 PERSON + 動 MOVE; on "makes"
they slide into one another and become 働, brushed large, and its card 働 / はたらく (indigo) / to work is written
with a 訓 seal as Raj says it.  The same for 山 MOUNTAIN + 上 UP + 下
DOWN, a dotted path climbing over a ridge and down the far side behind them, -> 峠 / とうげ / mountain pass.  "And
most of them have only a Japanese reading": 働 はたらく and 峠 とうげ side by side, each with its 訓 seal, and
KUN'YOMI ONLY (MOSTLY) under them."""
import numpy as np

import kit
import r5art
from art import sun_disc
from cards import INDIGO
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet,
                    cached, ease_io, lerp, noise, smooth, spline, stroke, text_layer)
from glyphs import typeset
from scenes.kscene import KanjiScene
from timeline import JA, at

EQ_Y = 700                                   # the row of parts
BIG_Y = 700


class Scene(KanjiScene):
    paper_seed = 99
    chapter = 3

    def build(self):
        self.L = cached('s04_land_v1', r5art.sun_landscape)['ink']
        self.J = cached('s02_junk_v1', lambda: dict(zip(('ink', 'red'), r5art.junk())))
        self.sun = sun_disc(H, W, 540, 640, 250, seed=66).astype(np.float32)
        self.notch, _ = text_layer('NOT FROM CHINA', FONT_EN_BOLD, 46, tracking=10)
        self.made, _ = text_layer('MADE IN JAPAN', FONT_BRUSH_EN, 104)
        self.made = self.made * (0.85 + 0.15 * noise(*self.made.shape, 6, 2, 9))
        self.kokuji = [typeset(c, 230, font=FONT_BRUSH_JP, seed=230 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('国字')]
        self.kk_r = text_layer('こくじ', FONT_JP, 50)[0]
        self.kk_e = text_layer('KOKUJI', FONT_EN_BOLD, 34, tracking=10)[0]
        self.wasei = text_layer('和製漢字', FONT_JP, 104)[0]
        self.ws_r = text_layer('わせいかんじ', FONT_JP, 44)[0]
        self.ws_e = text_layer('WASEI KANJI', FONT_EN_BOLD, 34, tracking=10)[0]
        self.ws_g = text_layer('kanji made in Japan', FONT_EN_ITALIC, 36)[0]
        self.plus, _ = text_layer('+', FONT_EN_BOLD, 90)
        self.eq, _ = text_layer('=', FONT_EN_BOLD, 90)
        self.parts = [dict(g=[text_layer(c, FONT_JP, 190)[0] for c in cs], lab=[text_layer(l, FONT_EN_BOLD, 32, tracking=8)[0] for l in ls])
                      for cs, ls in (('亻動', ('PERSON', 'MOVE')), ('山上下', ('MOUNTAIN', 'UP', 'DOWN')))]
        self.big = [typeset(c, 400, font=FONT_BRUSH_JP, seed=240 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('働峠')]
        self.small = [text_layer(c, FONT_JP, 170)[0] for c in '働峠']
        self.small_r = [text_layer(r, FONT_JP, 56)[0] for r in ('はたらく', 'とうげ')]
        self.only, _ = text_layer("KUN'YOMI ONLY", FONT_EN_BOLD, 52, tracking=8)
        self.mostly, _ = text_layer('(mostly)', FONT_EN_ITALIC, 40)
        self.c_work = self.card('働', ['はたらく'], ['kun'], 'hataraku', 'to work', seed=250)
        self.c_pass = self.card('峠', ['とうげ'], ['kun'], 'tōge', 'mountain pass', seed=254)
        self.t_some = at('Some kanji')[0]
        self.t_japan = at('Japan created')[0]
        self.t_kokuji = JA['kokuji']['start']
        self.t_wasei = JA['wasei']['start']
        self.t_part = [[at('Person')[0], at('move')[0]], None]
        self.t_mount = at('Mountain up')[0]
        self.t_part[1] = [self.t_mount, at('up', after=self.t_mount)[0], at('down', after=self.t_mount)[0]]
        self.t_makes = [at('makes', after=self.t_part[0][0])[0], at('makes', after=self.t_mount)[0]]
        self.t_read = [JA['hataraku']['start'], JA['touge']['start']]
        self.t_most = at('And most')[0]
        self.t_hit = self.t_makes[0]
        # the path over the pass, behind 山 上 下
        ps = StrokeSet(W, H)
        pts = spline([(120, 1010), (300, 930), (470, 820), (560, 790), (650, 820), (820, 930), (980, 1010)], 40)
        for k in range(0, len(pts) - 4, 6):
            ps.add(stroke(W, H, pts[k:k + 3], 6, ink=0.8, dry=0.5, seed=260 + k, red=True), self.t_mount + 0.2 + k * 0.012, 0.08)
        ps.add(stroke(W, H, spline([(60, 1060), (330, 900), (540, 760), (760, 900), (1020, 1060)], 30), 26, ink=0.5, dry=0.8,
                      profile=((0, 0.3), (0.1, 1), (0.9, 1), (1, 0.3)), seed=270), self.t_mount, 0.5)
        self.path = ps

    def cam(self, t):
        p = sum(0.03 * np.exp(-max(t - tm, 0) * 6) * (t > tm) for tm in self.t_makes)
        return Cam(540, 960, 1.0 + 0.015 * smooth((t - self.t0) / 20) + p)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        a_out = 1 - smooth((t - self.t_part[0][0] + 0.2) / 0.4)          # the "made in Japan" part clears for 働
        # --- not from China: the junk turns back
        fn = smooth((t - self.t_some) / 0.3) * (1 - smooth((t - self.t_japan) / 0.4))
        if fn > 0:
            u = ease_io((t - self.t_some) / 2.2)
            kit.put(fr, cam, self.J['ink'], lerp(560, -150, u), 760 + 4 * np.sin(t * 4), INK, fn, -0.8, 0.8)
            kit.put(fr, cam, self.J['red'], lerp(560, -150, u), 760 + 4 * np.sin(t * 4), VERM, 0.9 * fn, -0.8, 0.8)
            kit.put(fr, cam, self.notch, 540, 560, INK, fn)
        # --- Japan made its own: the sun over the hills, MADE IN JAPAN, then 国字 / 和製漢字
        fs = smooth((t - self.t_japan) / 0.7) * a_out
        if fs > 0:
            self.layer(fr, cam, self.sun, color=VERM, opacity=0.55 * fs)
            self.layer(fr, cam, self.L, 0, 240, opacity=0.7 * fs)
            fm = (t - self.t_japan - 0.5) / 0.15
            gone = smooth((t - self.t_kokuji + 0.4) / 0.3)
            if fm > 0:
                kit.put(fr, cam, self.made, 540, 450, INK, smooth(fm) * (1 - gone) * a_out, 1 + 0.35 * np.exp(-fm * 2.5), rot=-0.05)
            fk = (t - self.t_kokuji + 0.25) / 0.5
            if fk > 0:
                for j, g in enumerate(self.kokuji):
                    kit.put(fr, cam, g.at(fk - 0.25 * j, soft=0.05), 420 + 240 * j, 560, INK, a_out)
                kit.put(fr, cam, self.kk_r, 540, 730, INK, smooth(fk - 0.4) * a_out, 1 + 0.25 * np.exp(-max(fk - 0.5, 0) * 3))
                kit.put(fr, cam, self.kk_e, 540, 785, GREY, smooth(fk - 0.7) * a_out)
            fw = (t - self.t_wasei + 0.1) / 0.3
            if fw > 0:
                kit.put(fr, cam, self.wasei, 540, 900, VERM, smooth(fw) * a_out, 1 + 0.3 * np.exp(-fw * 2.5))
                kit.put(fr, cam, self.ws_r, 540, 1005, INK, smooth(fw - 0.5) * a_out)
                kit.put(fr, cam, self.ws_e, 540, 1055, GREY, smooth(fw - 0.7) * a_out)
                kit.put(fr, cam, self.ws_g, 540, 1105, GREY, smooth(fw - 1.0) * a_out)
        # --- the two kokuji, part by part
        for k in range(2):
            self.compound(fr, cam, t, k)
        # --- most of them: only a Japanese reading
        fo = smooth((t - self.t_most) / 0.4)
        if fo > 0:
            for k, x in enumerate((300, 780)):
                f = smooth((t - self.t_most - 0.15 * k) / 0.35)
                kit.put(fr, cam, self.small[k], x, 640, INK, f, 1 + 0.2 * np.exp(-max(t - self.t_most, 0) * 5))
                kit.put(fr, cam, self.small_r[k], x, 800, INDIGO, f)
                self.reading_seal(fr, cam, t, self.t_most + 0.35 + 0.2 * k, '訓', x=x + 130, y=540)
            kit.put(fr, cam, self.only, 540, 960, INDIGO, smooth((t - self.t_most - 0.6) / 0.3))
            kit.put(fr, cam, self.mostly, 540, 1030, GREY, smooth((t - self.t_most - 0.9) / 0.3))
        self.header(fr, t)
        return fr.img

    def compound(self, fr, cam, t, k):
        """parts -> one kanji -> its card, for 働 (k = 0) and 峠 (k = 1)"""
        ts = self.t_part[k]
        t_in = ts[0] - 0.15
        t_end = (self.t_mount if k == 0 else self.t_most) - 0.15
        if t < t_in or t > t_end + 0.4:
            return
        out = 1 - smooth((t - t_end) / 0.35)
        P = self.parts[k]
        n = len(P['g'])
        xs = [270, 790] if n == 2 else [200, 540, 880]
        merge = ease_io((t - self.t_makes[k]) / 0.45)
        if k == 1:
            self.layer(fr, cam, self.path.render(t, red=True), color=VERM, opacity=0.8 * out * (1 - 0.6 * merge))
            self.layer(fr, cam, self.path.render(t), opacity=0.35 * out)
        for j in range(n):
            f = (t - ts[j] + 0.1) / 0.2
            if f <= 0:
                continue
            x = lerp(xs[j], 540, merge)
            o = smooth(f) * (1 - smooth((merge - 0.6) / 0.4)) * out
            kit.put(fr, cam, P['g'][j], x, EQ_Y, INK, o, 1 + 0.3 * np.exp(-f * 2.5))
            kit.put(fr, cam, P['lab'][j], x, EQ_Y + 150, GREY, o * (1 - merge))
            if j > 0:
                kit.put(fr, cam, self.plus, (xs[j - 1] + xs[j]) / 2, EQ_Y, VERM, smooth(f) * (1 - merge) * out)
        # the kanji they make, brushed over the merge, and its card
        fb = (t - self.t_makes[k] - 0.25) / 0.5
        if fb > 0:
            kit.put(fr, cam, self.big[k].at(fb, soft=0.05), 540, BIG_Y, INK, out, 1 + 0.12 * np.exp(-fb * 3))
        card = (self.c_work, self.c_pass)[k]
        tr = self.t_read[k]
        self.draw_card(fr, cam, t, card, tr - 0.35, op=out, pulse=tr)
        self.reading_seal(fr, cam, t, tr, '訓', op=out)

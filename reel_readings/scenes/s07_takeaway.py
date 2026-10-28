"""Scene 7 — the takeaway.  "So don't memorize kanji as isolated symbols.  Learn them through real words.  On'yomi
and kun'yomi reflect different histories, and the word tells you which reading to use.  Every kanji has a story.
Follow Raj in JP to discover more."
The four kanji come back, each shut in its own box ("isolated symbols").  On "real words" the words open under them
in two rows, as in the hook: the kun'yomi words (indigo, 訓) やま ひ みず ひと and the on'yomi words (vermilion, 音)
富士山 日曜日 水曜日 日本人 with their readings, only the part that belongs to the kanji coloured.  On "on'yomi" a vermilion
brush band runs under the on row, on "kun'yomi" an indigo band under the kun row; on "the word tells you which
reading to use" the words pulse column by column.  "Every kanji has a story": the page clears and the ink of all
four becomes one landscape (a mountain under the sun, a river running down from it, a traveller beside the water);
a vermilion seal of 山日水人 is stamped.  DON'T JUST MEMORIZE KANJI. / LEARN THE WORDS., then FOLLOW @raj_in_jp on
a black brush banner, held to the end."""
import numpy as np

import kit
import r5art
from cards import INDIGO
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_HEAD, FONT_JP, GREY, H, INK, SEA, VERM, W, Cam, StrokeSet,
                    blur, cached, ease_io, noise, smooth, spline, stroke, text_layer)
from glyphs import typeset
from scenes.kscene import KanjiScene
from timeline import at, quiet_gap

KANJI = '山日水人'
COLS = [138, 406, 674, 942]
BOX_Y, KUN_Y, ON_Y, ON_R_Y = 470, 740, 905, 978
KUN = ['やま', 'ひ', 'みず', 'ひと']
ON_W = ['富士山', '日曜日', '水曜日', '日本人']
ON_R = [('ふじ', 'さん', 1), ('にち', 'ようび', 0), ('すい', 'ようび', 0), ('にほん', 'じん', 1)]   # (a, b, which part is the kanji's)
IVORY = np.array([0.98, 0.95, 0.88], np.float32)
CARD = np.array([0.97, 0.95, 0.89], np.float32)


class Scene(KanjiScene):
    paper_seed = 97

    def build(self):
        self.kj = [typeset(c, 160, font=FONT_BRUSH_JP, seed=120 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate(KANJI)]
        self.box, self.rim = kit.box(220, 220), kit.rim(220, 220, inset=8, t=3)
        self.kun = [text_layer(r, FONT_JP, 64)[0] for r in KUN]
        self.onw = [text_layer(r, FONT_JP, 60)[0] for r in ON_W]
        self.onr = []
        for a, b, which in ON_R:
            ga, gb = text_layer(a, FONT_JP, 38)[0], text_layer(b, FONT_JP, 38)[0]
            self.onr.append((ga, gb, which))
        sq = kit.blur(np.pad(np.ones((46, 46), np.float32), 4), 1.2)
        self.tags = [(sq, text_layer(c, FONT_JP, 32)[0], col) for c, col in (('訓', INDIGO), ('音', VERM))]
        self.F = cached('s07_final_v1', r5art.final_landscape)
        self.l1, _ = text_layer("DON'T JUST MEMORIZE KANJI.", FONT_HEAD, 58)
        self.l2, _ = text_layer('LEARN THE WORDS.', FONT_BRUSH_EN, 98)
        self.l2 = self.l2 * (0.85 + 0.15 * noise(*self.l2.shape, 6, 2, 4))
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 40, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 84)
        self.seal = _seal()
        self.t_so = at('So dont')[0]
        self.t_learn = at('Learn them')[0]
        self.t_on = at("On'yomi")[0]
        self.t_kun = at("kun'yomi")[0]
        self.t_word = at('and the word')[0]
        self.t_story = at('Every kanji')[0]
        c, n = quiet_gap(at('story.')[1] - 0.2, at('Follow')[0] + 0.05)
        self.t_stamp = self.t_hit = c - n / 2 + 0.03                  # sync: the seal is stamped (+ stamp, taiko)
        self.t_follow = at('Follow')[0]
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [(70, ON_Y + 4), (540, ON_Y - 6), (1040, ON_Y + 2)], 70, ink=0.5, dry=0.75,
                      profile=((0, 0.4), (0.06, 1), (0.92, 0.9), (1, 0.2)), seed=130, red=True), self.t_on, 0.6)
        ss.add(stroke(W, H, [(70, KUN_Y + 2), (540, KUN_Y - 4), (1040, KUN_Y + 4)], 62, ink=0.5, dry=0.75,
                      profile=((0, 0.4), (0.06, 1), (0.92, 0.9), (1, 0.2)), seed=131), self.t_kun, 0.6)
        self.bands = ss
        ul = StrokeSet(W, H)
        ul.add(stroke(W, H, [(230, 420), (540, 410), (850, 418)], 12, ink=0.95, dry=0.4, profile=((0, 0.3), (0.1, 1), (0.9, 0.8), (1, 0.1)),
                      seed=132, red=True), self.t_story + 1.25, 0.3)
        bn = StrokeSet(W, H)
        bn.add(stroke(W, H, [(250, 1336), (540, 1324), (830, 1332)], 118, ink=0.96, dry=0.45,
                      profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=133), self.t_follow - 0.05, 0.4)
        self.ul, self.banner = ul, bn
        self.T_land = self.wipe_map((540, 300), seed=10)

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.02 * smooth((t - self.t_so) / 10) + 0.03 * np.exp(-max(t - self.t_stamp, 0) * 5) * (t > self.t_stamp))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        out = 1 - smooth((t - self.t_story + 0.2) / 0.5)
        if out > 0:
            self.layer(fr, cam, self.bands.render(t, red=True), color=VERM, opacity=0.35 * out)
            self.layer(fr, cam, self.bands.render(t), color=INDIGO, opacity=0.3 * out)
            for k, g in enumerate(self.kj):
                f = (t - self.t_so - 0.12 * k) / 0.35
                if f <= 0:
                    continue
                o = smooth(f) * out
                shut = 1 - smooth((t - self.t_learn) / 0.4)           # the box opens on "real words"
                kit.card(fr, cam, self.box, self.rim, COLS[k], BOX_Y, o * (0.35 + 0.65 * shut))
                kit.put(fr, cam, g.at(f, soft=0.05), COLS[k], BOX_Y, INK, o)
                # the words
                fw = (t - self.t_learn - 0.15 * k) / 0.35
                if fw > 0:
                    pk = np.exp(-max(t - self.t_word - 0.3 * k, 0) * 4) * (t > self.t_word + 0.3 * k)
                    kit.put(fr, cam, self.kun[k], COLS[k] + 12, KUN_Y, INDIGO, smooth(fw) * out, 1 + 0.25 * pk)
                    fo = smooth(fw - 0.6) * out
                    kit.put(fr, cam, self.onw[k], COLS[k] + 12, ON_Y, INK, fo, 1 + 0.15 * pk)
                    ga, gb, which = self.onr[k]
                    wa, wb = ga.shape[1], gb.shape[1]
                    x0 = COLS[k] + 12 - (wa + wb) / 2
                    kit.put(fr, cam, ga, x0 + wa / 2, ON_R_Y, VERM if which == 0 else GREY, fo, 1 + (0.3 * pk if which == 0 else 0))
                    kit.put(fr, cam, gb, x0 + wa + wb / 2, ON_R_Y, VERM if which == 1 else GREY, fo, 1 + (0.3 * pk if which == 1 else 0))
            for row, (y, t0) in enumerate(((KUN_Y, self.t_learn), (ON_Y + 30, self.t_learn + 0.6))):
                sq, ch, col = self.tags[row]
                ft = smooth((t - t0) / 0.3) * out
                kit.put(fr, cam, sq, 30, y, col, 0.92 * ft)
                kit.put(fr, cam, ch, 30, y, IVORY, ft, over=True)
        # one landscape
        if t > self.t_story - 0.2:
            m = self.wipe(self.T_land, ease_io((t - self.t_story + 0.25) / 0.8))
            self.layer(fr, cam, self.F['sun'] * m, color=VERM, opacity=0.85)
            self.layer(fr, cam, self.F['water'] * m, color=SEA, opacity=0.45)
            self.layer(fr, cam, self.F['ink'] * m)
            kit.put(fr, cam, self.l1, 540, 230, INK, smooth((t - self.t_story - 0.5) / 0.35))
            kit.put(fr, cam, self.l2, 540, 335, VERM, smooth((t - self.t_story - 0.9) / 0.3), 1 + 0.2 * np.exp(-max(t - self.t_story - 0.9, 0) * 4))
            self.layer(fr, cam, self.ul.render(t, red=True), color=VERM)
            fs = (t - self.t_stamp) / 0.15
            if fs > 0:
                kit.put(fr, cam, self.seal, 165, 640, VERM, smooth(fs) * 0.95, 1 + 0.5 * np.exp(-fs * 2), rot=-0.05)
        # FOLLOW @raj_in_jp
        fb = (t - self.t_follow + 0.05) / 0.4
        if fb > 0:
            self.layer(fr, cam, self.banner.render(t), opacity=0.96)
            kit.put(fr, cam, self.handle, 540, 1324, IVORY, smooth((t - self.t_follow - 0.15) / 0.3), 1 + 0.1 * np.exp(-fb * 2.5), over=True)
            kit.put(fr, cam, self.follow, 540, 1238, INK, smooth(fb))
        return fr.img


def _seal():
    """a square vermilion seal with 山日 / 水人 cut out in white (density: 1 = vermilion)"""
    s = 150
    d = np.zeros((s, s), np.float32)
    d[8:-8, 8:-8] = 1
    d = blur(d, 1.2) * (0.82 + 0.18 * noise(s, s, 10, 3, 140))
    for k, c in enumerate(KANJI):
        g, _ = text_layer(c, FONT_JP, 56)
        gh, gw = g.shape
        cx, cy = (106, 44) if k == 0 else (106, 106) if k == 1 else (44, 44) if k == 2 else (44, 106)   # read top-right, down
        ys, xs = int(cy - gh / 2), int(cx - gw / 2)
        d[ys:ys + gh, xs:xs + gw] *= 1 - 0.95 * g
    return d

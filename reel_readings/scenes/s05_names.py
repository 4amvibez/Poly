"""Scene 5 — the names.  "These two kinds of readings have names.  The Chinese-based reading, like さん and にち, is
called おんよみ.  The native Japanese reading, like やま and ひ, is called くんよみ."
Over a faint landscape, TWO READINGS. TWO NAMES. is set under the header.  On "The Chinese-based reading" a hanging
scroll with a vermilion mount unrolls on the left: a 音 seal is stamped at its head, the little junk from the map
sails across under it (FROM CHINA), and as Raj says さん and にち the words 富士山 さん and 日曜日 にち drop in, the
reading in vermilion; on "おんよみ" its name 音読み ON'YOMI is brushed in at the top and pulses.  Then the same on
the right in indigo: 訓 seal, FROM JAPAN with a small vermilion sun, 山 やま and 日 ひ, and on "くんよみ" 訓読み
KUN'YOMI."""
import numpy as np

import kit
import r5art
from cards import INDIGO
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, GREY, INK, VERM, Cam, cached, ease_io, smooth, text_layer
from art import sun_disc
from scenes.kscene import KanjiScene
from timeline import JA, at

XL, XR = 290, 790
SY, SH, SW = 850, 760, 430                 # the scrolls: centre y, height, width
SIDES = [  # (x, colour, name, romaji, gloss, origin, examples [(word, reading, key)], seal)
    (XL, VERM, '音読み', "ON'YOMI", 'Chinese-based reading', 'FROM CHINA', [('富士山', 'さん', 'san_n'), ('日曜日', 'にち', 'nichi_n')], '音', 'onyomi_n'),
    (XR, INDIGO, '訓読み', "KUN'YOMI", 'native Japanese reading', 'FROM JAPAN', [('山', 'やま', 'yama_n'), ('日', 'ひ', 'hi_n')], '訓', 'kunyomi_n'),
]
ROW_Y = [955, 1065]
NAME_Y, ROM_Y, SEAL_Y, ICON_Y, ORIGIN_Y, GLOSS_Y = 565, 632, 712, 795, 857, 1170


class Scene(KanjiScene):
    paper_seed = 98
    chapter = 2

    def build(self):
        self.F = cached('s07_final_v1', r5art.final_landscape)
        self.J = cached('s02_junk_v1', lambda: dict(zip(('ink', 'red'), r5art.junk())))
        self.sun = sun_disc(120, 120, 60, 60, 44, seed=61).astype(np.float32)
        self.title, _ = text_layer('TWO READINGS. TWO NAMES.', FONT_HEAD, 50)
        self.L = []
        for x, col, name, rom, gloss, origin, ex, seal, key in SIDES:
            self.L.append(dict(name=text_layer(name, FONT_JP, 96)[0], rom=text_layer(rom, FONT_EN_BOLD, 40, tracking=8)[0],
                               gloss=text_layer(gloss, FONT_EN_ITALIC, 34)[0], origin=text_layer(origin, FONT_EN_BOLD, 30, tracking=8)[0],
                               ex=[(text_layer(w, FONT_JP, 66)[0], text_layer(r, FONT_JP, 56)[0], k) for w, r, k in ex]))
        self.t_title = at('These two kinds')[0]
        self.t_side = [at('Chinese-based')[0] - 0.25, at('native Japanese')[0] - 0.35]
        self.t_name = [JA['onyomi_n']['start'], JA['kunyomi_n']['start']]
        self.t_hit = self.t_name[0]

    def cam(self, t):
        p = sum(0.02 * np.exp(-max(t - tn, 0) * 6) * (t > tn) for tn in self.t_name)
        return Cam(540, 960, 1.0 + 0.015 * smooth((t - self.t0) / 12) + p)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        haze = 0.13 * smooth((t - self.t0) / 0.8)
        self.layer(fr, cam, self.F['sun'], color=VERM, opacity=haze * 1.4)
        self.layer(fr, cam, self.F['ink'], opacity=haze)
        kit.put(fr, cam, self.title, 540, 385, INK, smooth((t - self.t_title) / 0.3))
        for k, ((x, col, *_rest), L) in enumerate(zip(SIDES, self.L)):
            t0 = self.t_side[k]
            self.scroll(fr, cam, t, t0, x, SY, SW, SH, col, vertical=True)
            if t < t0 + 0.3:
                continue
            seal = _rest[5]
            self.reading_seal(fr, cam, t, t0 + 0.35, seal, x=x, y=SEAL_Y)
            # where it came from: the junk sailing in from China / the sun of Japan
            fo = smooth((t - t0 - 0.55) / 0.3)
            if k == 0:
                u = ease_io((t - t0 - 0.55) / 1.6)
                kit.put(fr, cam, self.J['ink'], x - 70 + 140 * u, ICON_Y + 3 * np.sin(t * 4), INK, fo, 0.32)
                kit.put(fr, cam, self.J['red'], x - 70 + 140 * u, ICON_Y + 3 * np.sin(t * 4), VERM, 0.9 * fo, 0.32)
            else:
                kit.put(fr, cam, self.sun, x, ICON_Y, VERM, 0.85 * fo, 0.8 + 0.2 * fo)
            kit.put(fr, cam, L['origin'], x, ORIGIN_Y, GREY, fo)
            for j, (gw, gr, rk) in enumerate(L['ex']):
                tr = JA[rk]['start']
                fw = smooth((t - tr + 0.25) / 0.25)
                p = np.exp(-max(t - tr, 0) * 4) * (t > tr)
                kit.put(fr, cam, gw, x - 70, ROW_Y[j], INK, fw)
                kit.put(fr, cam, gr, x + 110, ROW_Y[j], col, fw, 1 + 0.35 * p)
            # its name, at the head of the scroll
            tn = self.t_name[k]
            fn = (t - tn + 0.15) / 0.2
            if fn > 0:
                p = np.exp(-max(t - tn, 0) * 3.5) * (t > tn)
                kit.put(fr, cam, L['name'], x, NAME_Y, col, smooth(fn), 1 + 0.3 * np.exp(-fn * 2.5) + 0.06 * p)
                kit.put(fr, cam, L['rom'], x, ROM_Y, col, smooth(fn - 0.4))
                kit.put(fr, cam, L['gloss'], x, GLOSS_Y, INK, smooth(fn - 0.8))
        self.header(fr, t)
        return fr.img

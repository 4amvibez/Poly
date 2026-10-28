"""Scene 3 — 山.  "First, this kanji means mountain.  On its own, we say やま.  But in 富士山, we say さん."
then Raj in Japanese: 「山は『やま』とも読みます。でも、富士山は『ふじさん』と読みます。」
A huge 山 is brushed onto the paper.  On "means mountain" its ink rises off the page and becomes a mountain in mist
with pines.  The card writes itself: 山 / やま (indigo, kun'yomi) / yama / MOUNTAIN, and やま pulses as Raj says
it.  "But in 富士山": an ink sweep across the page reveals Mount Fuji under a vermilion sun, its reflection in a lake,
a pagoda on the shore; the card becomes 富士山 / ふ・じ・さん with さん in vermilion (on'yomi), pulsing on "san".
During the Japanese line both words stand side by side, each lit while it is being said."""
import numpy as np

import kit
import r5art
from engine import INK, VERM, Cam, cached, ease_io, ease_out, lerp, smooth
from scenes.kscene import CARD_Y, KanjiScene
from timeline import JA, at, quiet_gap


class Scene(KanjiScene):
    paper_seed = 93

    def build(self):
        self.big = self.big_kanji('山', seed=31)
        self.Mt = cached('s03_mountain_v1', r5art.mountain_scene)
        self.Fj = cached('s03_fuji_v3', r5art.fuji_scene)
        self.c_yama = self.card('山', ['やま'], ['kun'], 'yama', 'mountain', seed=40)
        self.c_fuji = self.card('富士山', ['ふ', 'じ', 'さん'], [None, None, 'on'], 'Fujisan', 'Mount Fuji', focus=2, seed=44)
        self.t_first = at('First')[0]
        self.t_means = at('means mountain')[0]
        self.t_own = at('On its own')[0] - 0.1
        self.t_yama = JA['yama']['start']
        self.t_but = at('But in')[0]
        self.t_fuji = JA['fujisan']['start']
        self.t_san = JA['san']['start']
        self.t_line = JA['yama_line']['start']
        self.t_line_end = JA['yama_line']['end']
        self.t_mid, _ = quiet_gap(self.t_line + 1.2, self.t_line_end - 1.5)
        c, n = quiet_gap(JA['san']['end'] - 0.05, self.t_line + 0.05)
        self.t_hit = c - n / 2 + 0.04                                  # sync: the two cards part (+ bowl, koto)
        self.T_rise = self.wipe_map((540, 1150), seed=3)
        self.T_sweep = self.wipe_map((-200, 300), seed=4)

    def cam(self, t):
        z = 1.0 + 0.03 * smooth((t - self.t_means) / 3) - 0.03 * smooth((t - self.t_but) / 1.0) + 0.04 * smooth((t - self.t_line) / 6)
        return Cam(540, 960 - 20 * smooth((t - self.t_line) / 6), z)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        rise = self.wipe(self.T_rise, ease_out((t - self.t_means + 0.1) / 1.3))
        sweep = self.wipe(self.T_sweep, ease_io((t - self.t_but + 0.05) / 0.8))
        if rise.any():
            self.layer(fr, cam, self.Mt['ink'] * rise * (1 - sweep))
        if sweep.any():
            self.layer(fr, cam, self.Fj['sun'] * sweep, color=VERM, opacity=0.9)
            self.layer(fr, cam, self.Fj['ink'] * sweep)
            self.layer(fr, cam, self.Fj['sun_refl'] * sweep, color=VERM, opacity=0.6)
        # the big 山, written, then rising into the mountain
        f = (t - self.t_first) / 0.75
        if f > 0:
            u = ease_io((t - self.t_means - 0.1) / 0.9)
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, lerp(640, 470, u), INK, 1 - smooth(u * 1.2), lerp(1.0, 1.15, u))
        # the cards
        line = smooth((t - self.t_hit) / 0.45)
        o_y = 1 - smooth((t - self.t_but) / 0.3)
        first = t < self.t_mid
        if line <= 0:
            self.draw_card(fr, cam, t, self.c_yama, self.t_own, op=o_y, pulse=self.t_yama)
            if t > self.t_but + 0.2:
                self.draw_card(fr, cam, t, self.c_fuji, self.t_but + 0.25, pulse=self.t_san)
        else:
            u = ease_io(line)
            self.draw_card(fr, cam, t, self.c_yama, self.t_line - 0.6, x=lerp(540, 300, u), y=lerp(CARD_Y, 1070, u), scale=lerp(1, 0.72, u),
                           op=u * (1.0 if first else 0.5))
            self.draw_card(fr, cam, t, self.c_fuji, self.t_but + 0.25, x=lerp(540, 770, u), y=lerp(CARD_Y, 1070, u), scale=lerp(1, 0.72, u),
                           op=1.0 if not first else lerp(1, 0.5, u), pulse=self.t_mid + 0.9 if not first else None)
        return fr.img

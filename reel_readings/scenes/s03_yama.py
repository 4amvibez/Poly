"""Scene 3 — 山.  "First, this kanji means mountain.  On its own, we say やま.  But in 富士山, we say さん."
A huge 山 is brushed onto the paper.  On "means mountain" its ink rises off the page and becomes a mountain in mist
with pines.  The card writes itself: 山 / やま (indigo, kun'yomi) / yama / MOUNTAIN, and やま pulses as Raj says
it.  "But in 富士山": an ink sweep across the page reveals Mount Fuji under a vermilion sun, its reflection in a lake,
a pagoda on the shore; the card becomes 富士山 / ふ・じ・さん with さん in vermilion (on'yomi), pulsing on "san",
and once more, with the sun, in the breath after it (sync)."""
import numpy as np

import kit
import r5art
from engine import INK, VERM, Cam, cached, ease_io, ease_out, lerp, smooth
from scenes.kscene import KanjiScene
from timeline import JA, at, quiet_gap


class Scene(KanjiScene):
    paper_seed = 93
    chapter = 0

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
        c, n = quiet_gap(JA['san']['end'] - 0.05, at('Next')[0] + 0.05)
        self.t_hit = c - min(n / 2, 0.1)                               # sync: さん and the sun pulse (+ bowl, koto), mid-breath
        self.T_rise = self.wipe_map((540, 1150), seed=3)
        self.origin('山', 520, self.t_first - 0.1, seed=5)
        self.T_sweep = self.wipe_map((-200, 300), seed=4)

    def cam(self, t):
        z = 1.0 + 0.03 * smooth((t - self.t_means) / 3) - 0.03 * smooth((t - self.t_but) / 1.0)
        return Cam(540, 960, z)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        rise = self.wipe(self.T_rise, ease_out((t - self.t_means + 0.1) / 1.3))
        sweep = self.wipe(self.T_sweep, ease_io((t - self.t_but + 0.05) / 0.8))
        if rise.any():
            self.layer(fr, cam, self.Mt['ink'] * rise * (1 - sweep))
        if sweep.any():
            glow = 0.9 + 0.1 * np.exp(-max(t - self.t_hit, 0) * 4) * (t > self.t_hit)
            self.layer(fr, cam, self.Fj['sun'] * sweep, color=VERM, opacity=glow)
            self.layer(fr, cam, self.Fj['ink'] * sweep)
            self.layer(fr, cam, self.Fj['sun_refl'] * sweep, color=VERM, opacity=0.6)
        # the big 山, written, then rising into the mountain
        self.draw_origin(fr, cam, t, 540, 640)
        f = self.modern(t)
        if f > 0:
            u = ease_io((t - self.t_means - 0.1) / 0.9)
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, lerp(640, 470, u), INK, 1 - smooth(u * 1.2), lerp(1.0, 1.15, u))
        # the cards
        self.draw_card(fr, cam, t, self.c_yama, self.t_own, op=1 - smooth((t - self.t_but) / 0.3), pulse=self.t_yama)
        if t > self.t_but + 0.2:
            self.draw_card(fr, cam, t, self.c_fuji, self.t_but + 0.25, pulse=self.t_san if t < self.t_hit else self.t_hit)
        self.reading_seal(fr, cam, t, self.t_yama, '訓', op=1 - smooth((t - self.t_but) / 0.3))
        self.reading_seal(fr, cam, t, self.t_san, '音')
        self.header(fr, t)
        return fr.img

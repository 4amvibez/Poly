"""Scene 4 — 日.  "Next, this kanji can mean the sun, or a day.  On its own, it's ひ.  But in 日曜日, Sunday, it's にち."
A black 日 is brushed in one gesture over a misty landscape.  On "the sun" it fills with vermilion and becomes the
sun; on "or a day" it starts across the sky.  Card: 日 / ひ (indigo) / hi / SUN · DAY.  "But in 日曜日": the sun runs
across the sky and drops into a calendar page that swings in, landing on the 日 (Sunday) column, which is circled in
vermilion.  Card: 日曜日 / にち・よう・び with にち in vermilion (on'yomi, pulsing on "nichi") and the last 日's び in
indigo (kun'yomi ひ, softened), so the word itself shows both kinds of reading."""
import numpy as np

import kit
import r5art
from art import sun_disc
from engine import INK, VERM, Cam, cached, ease_in, ease_io, lerp, smooth
from scenes.kscene import KanjiScene
from timeline import JA, at, quiet_gap

CAL = (540, 660)


class Scene(KanjiScene):
    paper_seed = 94
    chapter = 1

    def build(self):
        self.big = self.big_kanji('日', seed=51)
        self.L = cached('s04_land_v1', r5art.sun_landscape)
        self.C = cached('s04_cal_v1', r5art.calendar)
        self.sun = sun_disc(420, 420, 210, 210, 175, seed=52).astype(np.float32)
        self.c_hi = self.card('日', ['ひ'], ['kun'], 'hi', 'sun / day', seed=60)
        self.c_nichi = self.card('日曜日', ['にち', 'よう', 'び'], ['on', None, 'kun'], 'nichiyōbi', 'Sunday', focus=0, seed=64)
        self.t_next = at('Next')[0]
        self.t_sun = at('the sun')[0]
        self.origin('日', 520, self.t_next - 0.1, seed=6)
        self.t_day = at('or a day')[0]
        self.t_own = at('On its own', after=self.t0)[0] - 0.1
        self.t_hi = JA['hi']['start']
        self.t_but = at('But in', after=self.t0)[0]
        self.t_nichi = JA['nichi']['start']
        w, h = self.C['size']
        cx, cy = self.C['cells'][0]
        self.cell = (CAL[0] + cx - w / 2, CAL[1] + cy - h / 2)
        c, n = quiet_gap(at('Sunday')[1] - 0.15, at("it's", after=at('Sunday')[0])[0] + 0.05)
        self.t_hit = c - n / 2 + 0.04                                  # sync: the circle closes round 日 (+ bowl)
        self.T_in = self.wipe_map((540, 600), seed=5)

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.025 * smooth((t - self.t_next) / 4) - 0.025 * smooth((t - self.t_but) / 1.2))

    def sun_pos(self, t):
        """centre and scale of the sun: blooming where 日 was, drifting on "or a day", then into the calendar"""
        x, y, s = 540, 620, 1.0
        u = ease_io((t - self.t_day) / 3.0)
        x, y = lerp(540, 650, u), lerp(620, 560, u)
        v = ease_in((t - self.t_but - 0.1) / 0.75)
        x, y, s = lerp(x, self.cell[0], v), lerp(y, self.cell[1], v), lerp(1.0, 0.22, v)
        return x, y, s, v

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        land = self.wipe(self.T_in, smooth((t - self.t0) / 1.0))
        cal_t = self.t_but + 0.35
        dim = 1 - 0.55 * smooth((t - cal_t) / 0.5)
        self.layer(fr, cam, self.L['ink'] * land, opacity=dim)
        # 日, brushed, then turning into the sun
        self.draw_origin(fr, cam, t, 540, 620)
        f = self.modern(t)
        red = smooth((t - self.t_sun + 0.1) / 0.5)
        if f > 0:
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, 620, INK, 1 - red)
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, 620, VERM, red * (1 - smooth((t - self.t_sun - 0.3) / 0.5)))
        self.calendar(fr, cam, t, self.C, CAL[0], CAL[1], cal_t, 0, t_mark=self.t_hit)
        if red > 0:
            x, y, s, v = self.sun_pos(t)
            bloom = smooth((t - self.t_sun) / 0.6)
            kit.put(fr, cam, self.sun, x, y, VERM, 0.92 * bloom * (1 - smooth((v - 0.85) / 0.15)), s * (0.6 + 0.4 * bloom))
        # the cards
        self.draw_card(fr, cam, t, self.c_hi, self.t_own, op=1 - smooth((t - self.t_but) / 0.3), pulse=self.t_hi)
        if t > self.t_but + 0.2:
            self.draw_card(fr, cam, t, self.c_nichi, self.t_but + 0.3, pulse=self.t_nichi)
        self.reading_seal(fr, cam, t, self.t_hi, '訓', op=1 - smooth((t - self.t_but) / 0.3))
        self.reading_seal(fr, cam, t, self.t_nichi, '音')
        self.header(fr, t)
        return fr.img

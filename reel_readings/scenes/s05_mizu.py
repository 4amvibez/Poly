"""Scene 5 — 水.  "Now, this kanji means water.  On its own, it's みず.  But in 水曜日, Wednesday, it's すい.  The character
stays the same; the word decides the reading."
水 is brushed; on "means water" a pool of black ink spreads out from it across the paper and becomes a river winding
down out of the hills, whose current keeps flowing.  Card: 水 / みず (indigo) / mizu / WATER.  "But in 水曜日": the river
curls up into one long brushstroke that sweeps onto a calendar, where the 水 (Wednesday) column is circled.  Card:
水曜日 / すい・よう・び, すい in vermilion.  "The character stays the same": the page clears to one 水 in the middle
with its two words on either side, みず (indigo) and すいようび (vermilion), each joined to it by a brush arrow; on
"the word decides the reading" each reading pulses in turn."""
import numpy as np

import kit
import r5art
from engine import H, INK, SEA, VERM, W, Cam, StrokeSet, blur, cached, ease_io, smooth, spline, stroke
from cards import INDIGO
from scenes.kscene import KanjiScene
from timeline import JA, at, quiet_gap

CAL = (540, 660)


class Scene(KanjiScene):
    paper_seed = 95

    def build(self):
        self.big = self.big_kanji('水', seed=71)
        self.mid = self.big_kanji('水', size=300, seed=72)
        self.R = cached('s05_river_v1', r5art.river_scene)
        self.C = cached('s05_cal_v1', r5art.calendar)
        self.c_mizu = self.card('水', ['みず'], ['kun'], 'mizu', 'water', seed=80)
        self.c_sui = self.card('水曜日', ['すい', 'よう', 'び'], ['on', None, 'kun'], 'suiyōbi', 'Wednesday', focus=0, seed=84)
        self.t_now = at('Now')[0]
        self.t_means = at('means water')[0]
        self.t_own = at('On its own', after=self.t0)[0] - 0.1
        self.t_mizu = JA['mizu']['start']
        self.t_but = at('But in', after=self.t0)[0]
        self.t_sui = JA['sui']['start']
        self.t_same = at('The character stays')[0]
        self.t_word = at('the word decides')[0]
        c, n = quiet_gap(at('Wednesday')[1] - 0.15, at("it's", after=at('Wednesday')[0])[0] + 0.05)
        self.t_hit = c - n / 2 + 0.04                                  # sync: the circle closes round 水 (+ bowl)
        self.T_pool = self.wipe_map((540, 760), seed=6)
        self.T_river = self.wipe_map((540, 560), seed=7)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.pool = (np.clip(1 - np.hypot((xx - 540) / 420, (yy - 800) / 330), 0, 1) ** 0.6 * 0.9).astype(np.float32)
        self.yy = yy
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, spline([(560, 1560), (700, 1250), (820, 980), (760, 780), (560, 700), (330, 560)], 20), 34, ink=0.9, dry=0.65,
                      profile=((0, 0.3), (0.1, 1), (0.8, 0.9), (1, 0.15)), seed=90), self.t_but + 0.05, 0.6)
        self.curl = ss
        ar = StrokeSet(W, H)
        kit.arrow(ar, (450, 640), (310, 770), self.t_same + 0.4, 0.35, width=6, seed=91)
        kit.arrow(ar, (630, 640), (770, 770), self.t_same + 0.7, 0.35, width=6, seed=94, red=True)
        self.arrows = ar

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.03 * smooth((t - self.t_means) / 4) - 0.03 * smooth((t - self.t_but) / 1))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        clear = smooth((t - self.t_same + 0.1) / 0.5)
        # the ink pool, then the river it becomes
        pf = (t - self.t_means + 0.1) / 0.7
        rf = (t - self.t_means - 0.35) / 1.2
        cal_t = self.t_but + 0.45
        dim = (1 - 0.6 * smooth((t - cal_t) / 0.5)) * (1 - clear)
        if pf > 0:
            pool = self.pool * self.wipe(self.T_pool, smooth(pf)) * (1 - smooth(rf * 1.4))
            self.layer(fr, cam, pool, opacity=dim)
        if rf > 0:
            m = self.wipe(self.T_river, smooth(rf))
            self.layer(fr, cam, self.R['water'] * m, color=SEA, opacity=0.45 * dim)
            self.layer(fr, cam, self.R['ink'] * m, opacity=dim)
            flow = self.R['flow'] * m * (0.5 + 0.5 * np.sin(self.yy / 16 - t * 6))
            self.layer(fr, cam, flow, opacity=0.9 * dim)
        f = (t - self.t_now) / 0.6
        if f > 0:
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, 620, INK, 1 - smooth((t - self.t_means - 0.2) / 0.6))
        # the river curls into one stroke and onto the calendar
        self.layer(fr, cam, self.curl.render(t), opacity=(1 - smooth((t - cal_t - 0.3) / 0.5)))
        self.calendar(fr, cam, t, self.C, CAL[0], CAL[1], cal_t, 3, op=1 - clear, t_mark=self.t_hit)
        # the cards
        self.draw_card(fr, cam, t, self.c_mizu, self.t_own, op=1 - smooth((t - self.t_but) / 0.3), pulse=self.t_mizu)
        if t > self.t_but + 0.2:
            self.draw_card(fr, cam, t, self.c_sui, self.t_but + 0.35, op=1 - clear, pulse=self.t_sui)
        # one character, two words
        if clear > 0:
            kit.put(fr, cam, self.mid.dens, 540, 520, INK, clear, 0.9 + 0.1 * clear)
            self.layer(fr, cam, self.arrows.render(t), color=INDIGO)
            self.layer(fr, cam, self.arrows.render(t, red=True), color=VERM)
            self.draw_card(fr, cam, t, self.c_mizu, self.t_same + 0.55, x=270, y=880, scale=0.8, pulse=self.t_word + 0.2)
            self.draw_card(fr, cam, t, self.c_sui, self.t_same + 0.85, x=790, y=880, scale=0.8, pulse=self.t_word + 0.7)
        return fr.img

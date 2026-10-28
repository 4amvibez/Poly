"""Scene 5 — the first quiz: 水.  "Your turn!  Water is みず.  So how do you read this?  [pause]  すい！ すいようび."
YOUR TURN! is stamped at the top.  水 is brushed and a pool of ink spreads out from it into a river winding down out of
the hills, its current flowing; card 水 / みず (indigo).  "How do you read this?": the river dims and 水曜日 comes up
large with ？ ？ ？ for its readings.  In the silence left for the viewer a big ？ is stamped (sync) and a vermilion ring
draws itself round the word like a timer; on "すい！" the readings drop in, すい in vermilion, and the whole word pulses
on "すいようび"."""
import numpy as np

import kit
import r5art
from cards import INDIGO
from engine import FONT_BRUSH_EN, FONT_JP, H, INK, SEA, VERM, W, Cam, cached, noise, smooth, text_layer
from scenes.kscene import KanjiScene
from timeline import JA, PAUSES, at

QY = 760                    # the quiz word


class Scene(KanjiScene):
    paper_seed = 105

    def build(self):
        self.big = self.big_kanji('水', size=420, seed=71)
        self.R = cached('s05_river_v1', r5art.river_scene)
        self.c_mizu = self.card('水', ['みず'], ['kun'], 'mizu', 'water', seed=80)
        self.c_q = self.card('水曜日', ['？', '？', '？'], [None, None, None], '?', 'how do you read it?', seed=84)
        self.c_sui = self.card('水曜日', ['すい', 'よう', 'び'], ['on', None, 'kun'], 'suiyōbi', 'Wednesday', focus=0, seed=84)
        self.turn, _ = text_layer('YOUR TURN!', FONT_BRUSH_EN, 150)
        self.turn = self.turn * (0.85 + 0.15 * noise(*self.turn.shape, 6, 2, 5))
        self.q, _ = text_layer('？', FONT_JP, 260)
        self.t_turn = at('Your turn')[0]
        self.t_water = at('Water is')[0]
        self.t_mizu = JA['mizu']['start']
        self.t_how = at('So how')[0]
        p = PAUSES['quiz_water']
        self.p0, self.p1 = p['start'], p['end']
        self.t_hit = self.p0 + 0.1                                     # sync: ？ stamped in the silence (+ taiko)
        self.t_sui = JA['sui']['start']
        self.t_suiyoubi = JA['suiyoubi']['start']
        self.T_pool = self.wipe_map((540, 760), seed=6)
        self.T_river = self.wipe_map((540, 560), seed=7)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.pool = (np.clip(1 - np.hypot((xx - 540) / 420, (yy - 800) / 330), 0, 1) ** 0.6 * 0.9).astype(np.float32)
        self.yy = yy

    def cam(self, t):
        punch = 0.03 * np.exp(-max(t - self.t_hit, 0) * 6) * (t > self.t_hit) + 0.03 * np.exp(-max(t - self.t_sui, 0) * 6) * (t > self.t_sui)
        return Cam(540, 960, 1.0 + punch + 0.02 * smooth((t - self.t0) / 8))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        quiz = smooth((t - self.t_how + 0.1) / 0.4)
        dim = 1 - 0.65 * quiz
        pf = (t - self.t_water - 0.2) / 0.6
        rf = (t - self.t_water - 0.5) / 1.0
        if pf > 0:
            pool = self.pool * self.wipe(self.T_pool, smooth(pf)) * (1 - smooth(rf * 1.4))
            self.layer(fr, cam, pool, opacity=dim)
        if rf > 0:
            m = self.wipe(self.T_river, smooth(rf))
            self.layer(fr, cam, self.R['water'] * m, color=SEA, opacity=0.45 * dim)
            self.layer(fr, cam, self.R['ink'] * m, opacity=dim)
            self.layer(fr, cam, self.R['flow'] * m * (0.5 + 0.5 * np.sin(self.yy / 16 - t * 6)), opacity=0.9 * dim)
        f = (t - self.t_water) / 0.4
        if f > 0:
            kit.put(fr, cam, self.big.at(f, soft=0.05), 540, 640, INK, 1 - smooth((t - self.t_water - 0.6) / 0.5))
        ft = (t - self.t_turn) / 0.15
        if ft > 0:
            kit.put(fr, cam, self.turn, 540, 270, VERM, smooth(ft), 1 + 0.35 * np.exp(-ft * 2.2), rot=-0.04)
        # みず, then the quiz
        self.draw_card(fr, cam, t, self.c_mizu, self.t_mizu - 0.35, op=1 - quiz, pulse=self.t_mizu)
        if quiz > 0:
            card = self.c_q if t < self.t_sui else self.c_sui
            t0 = self.t_how if t < self.t_sui else self.t_sui - 0.3
            self.draw_card(fr, cam, t, card, t0, y=QY, scale=1.35, pulse=self.t_sui if card is self.c_sui else None,
                           pulses={1: self.t_suiyoubi, 2: self.t_suiyoubi} if card is self.c_sui else None)
            self.ring(fr, cam, t, self.p0, 540, QY + 90, VERM, r=4.4, dur=self.p1 - self.p0)
            fq = (t - self.t_hit) / 0.15
            if fq > 0 and t < self.t_sui + 0.2:
                kit.put(fr, cam, self.q, 900, QY - 210, VERM, smooth(fq) * (1 - smooth((t - self.t_sui) / 0.2)), 1 + 0.4 * np.exp(-fq * 2.5), rot=0.08)
        return fr.img

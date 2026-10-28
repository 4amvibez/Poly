"""Scene 6 — the second quiz: 人.  "Person is ひと.  And this one?  [pause]  にほんじん！"
人 is brushed on a ridge and settles into the straw-hat traveller; card 人 / ひと (indigo).  "And this one?": the
mist opens on a small group of travellers under a large vermilion circle, a pagoda and cherry blossom, and 日本人
comes up with ？ ？ ？.  In the silence a big ？ is stamped (sync) and a vermilion ring draws round the word like a
timer; on "にほんじん！" the readings drop in, じん in vermilion."""
import numpy as np

import kit
import r5art
from engine import FONT_JP, INK, SAKURA, VERM, Cam, cached, ease_io, smooth, text_layer
from scenes.kscene import KanjiScene
from timeline import JA, PAUSES, at


class Scene(KanjiScene):
    paper_seed = 106

    def build(self):
        self.big = self.big_kanji('人', size=470, seed=101)
        self.P = cached('s06_person_v1', r5art.person_scene)
        self.G = cached('s06_group_v1', r5art.group_scene)
        self.c_hito = self.card('人', ['ひと'], ['kun'], 'hito', 'person', seed=110)
        self.c_q = self.card('日本人', ['？', '？', '？'], [None, None, None], 'nihonjin?', 'how do you read it?', seed=114)
        self.c_jin = self.card('日本人', ['に', 'ほん', 'じん'], [None, None, 'on'], 'nihonjin', 'Japanese person', focus=2, seed=114)
        self.q, _ = text_layer('？', FONT_JP, 240)
        self.t_fin = at('Person is')[0] - 0.1
        self.t_means = at('Person is')[0] + 0.4
        self.t_hito = JA['hito']['start']
        self.t_but = at('And this one')[0]
        p = PAUSES['quiz_person']
        self.p0, self.p1 = p['start'], p['end']
        self.t_hit = self.p0 + 0.1                                     # sync: ？ stamped in the silence (+ taiko)
        self.t_jin = JA['nihonjin']['start']
        self.T_in = self.wipe_map((540, 1000), seed=8)
        self.T_open = self.wipe_map((540, 820), seed=9)

    def cam(self, t):
        punch = 0.03 * np.exp(-max(t - self.t_hit, 0) * 6) * (t > self.t_hit) + 0.03 * np.exp(-max(t - self.t_jin, 0) * 6) * (t > self.t_jin)
        return Cam(540, 960, 1.0 + punch + 0.02 * smooth((t - self.t0) / 6))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        land = self.wipe(self.T_in, smooth((t - self.t0) / 0.8))
        op = self.wipe(self.T_open, ease_io((t - self.t_but + 0.05) / 0.7))
        self.layer(fr, cam, self.P['land'] * land * (1 - op))
        if op.any():
            kit.put(fr, cam, self.G['sun'] * op, 540, 850, VERM, 0.85)
            self.layer(fr, cam, self.G['land'] * op, 0, -110)
            self.layer(fr, cam, self.G['figs'] * op, 0, -110)
            self.layer(fr, cam, self.G['branch'] * op, 0, -110, opacity=0.9)
            self.layer(fr, cam, self.G['bloss'] * op, 0, -110, color=SAKURA, opacity=0.9)
        f = (t - self.t_fin) / 0.4
        become = smooth((t - self.t_means) / 0.6)
        if f > 0:
            kit.put(fr, cam, self.big.at(f, soft=0.05), 380, 760, INK, 1 - become, 1 - 0.15 * become)
        self.layer(fr, cam, self.P['fig'] * (1 - op), opacity=become)
        self.draw_card(fr, cam, t, self.c_hito, self.t_hito - 0.35, op=1 - smooth((t - self.t_but) / 0.3), pulse=self.t_hito)
        if t > self.t_but + 0.15:
            card = self.c_q if t < self.t_jin else self.c_jin
            t0 = self.t_but + 0.2 if t < self.t_jin else self.t_jin - 0.3
            self.draw_card(fr, cam, t, card, t0, pulse=self.t_jin + 0.3 if card is self.c_jin else None)
            self.ring(fr, cam, t, self.p0, 540, 1150 + 60, VERM, r=3.4, dur=self.p1 - self.p0)
            fq = (t - self.t_hit) / 0.15
            if fq > 0 and t < self.t_jin + 0.2:
                kit.put(fr, cam, self.q, 890, 1000, VERM, smooth(fq) * (1 - smooth((t - self.t_jin) / 0.2)), 1 + 0.4 * np.exp(-fq * 2.5), rot=0.08)
        return fr.img

"""Scene 6 — 人.  "Finally, this kanji means person.  Alone, it's often ひと.  But in 日本人, a Japanese person, it
becomes じん.  And in 三人, three people, it's にん."
One 人 in black ink, standing on a ridge.  On "means person" its two strokes settle into the silhouette of a
traveller in a straw hat (the same traveller as the hook).  Card: 人 / ひと (indigo) / hito / PERSON.  "But in 日本人":
the mist opens on a small group of travellers on a shore under a large vermilion circle, a pagoda and cherry
blossom.  Card: 日本人 / に・ほん・じん with じん in vermilion (on'yomi), pulsing on "jin".  "And in 三人": the card
becomes 三人 / さん・にん with にん in vermilion, three of the travellers step forward, and as Raj says にん the three
readings of 人 are set in a row above them, ひと (indigo) じん にん (vermilion), with a 音 seal by the card."""
import numpy as np

import kit
import r5art
from engine import INK, SAKURA, VERM, Cam, cached, ease_io, smooth
from scenes.kscene import KanjiScene
from timeline import JA, at, quiet_gap


class Scene(KanjiScene):
    paper_seed = 96
    chapter = 2

    def build(self):
        self.big = self.big_kanji('人', size=470, seed=101)
        self.P = cached('s06_person_v1', r5art.person_scene)
        self.G = cached('s06_group_v1', r5art.group_scene)
        self.c_hito = self.card('人', ['ひと'], ['kun'], 'hito', 'person', seed=110)
        self.c_jin = self.card('日本人', ['に', 'ほん', 'じん'], [None, None, 'on'], 'nihonjin', 'Japanese person', focus=2, seed=114)
        self.t_fin = at('Finally')[0]
        self.t_means = at('means person')[0]
        self.origin('人', 470, self.t_fin - 0.1, seed=8)
        self.t_alone = at('Alone', after=self.t0)[0] - 0.1
        self.t_hito = JA['hito']['start']
        self.t_but = at('But in', after=self.t0)[0]
        self.t_jin = JA['jin']['start']
        self.t_sannin = JA['sannin']['start']
        self.t_nin = JA['nin']['start']
        self.c_nin = self.card('三人', ['さん', 'にん'], [None, 'on'], 'sannin', 'three people', focus=1, seed=118)
        from engine import FONT_JP, text_layer
        self.three = [text_layer(r, FONT_JP, 72)[0] for r in ('ひと', 'じん', 'にん')]
        self.dot = text_layer('・', FONT_JP, 72)[0]
        c, n = quiet_gap(at('Japanese person')[1] - 0.15, at('it becomes')[0] + 0.05)
        self.t_hit = c - n / 2 + 0.04                                  # sync: the red circle swells (+ taiko, bowl)
        self.T_in = self.wipe_map((540, 1000), seed=8)
        self.T_open = self.wipe_map((540, 820), seed=9)

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.03 * smooth((t - self.t_fin) / 4) - 0.03 * smooth((t - self.t_but) / 1) + 0.02 * smooth((t - self.t_but - 1) / 4))

    def petals(self, u):
        """cherry petals blown off the branch, tumbling down and to the right"""
        import cv2
        from engine import H, W
        d = np.zeros((H, W), np.float32)
        rng = np.random.default_rng(12)
        for k in range(22):
            x0, y0, ph, sp = rng.uniform(40, 420), rng.uniform(560, 900), rng.uniform(0, 6.3), rng.uniform(0.6, 1.2)
            v = (u * sp + rng.uniform(0, 6)) % 6
            x = x0 + 95 * v + 30 * np.sin(v * 2.2 + ph)
            y = y0 + 70 * v + 18 * np.sin(v * 3.1 + ph)
            if 0 < x < W and 0 < y < H:
                a = np.degrees(v * 2.5 + ph)
                cv2.ellipse(d, (int(x), int(y)), (int(9 * abs(np.cos(v * 3 + ph)) + 3), 5), a, 0, 360, 0.9, -1, cv2.LINE_AA)
        return d

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        land = self.wipe(self.T_in, smooth((t - self.t0) / 1.1))
        op = self.wipe(self.T_open, ease_io((t - self.t_but + 0.05) / 0.9))
        self.layer(fr, cam, self.P['land'] * land * (1 - op))
        if op.any():
            sw = 1 + 0.04 * np.exp(-max(t - self.t_hit, 0) * 3) * (t > self.t_hit)
            kit.put(fr, cam, self.G['sun'] * op, 540, 960 - 110 + (1 - sw) * -260, VERM, 0.85, sw)
            self.layer(fr, cam, self.G['land'] * op, 0, -110)
            self.layer(fr, cam, self.G['figs'] * op, 0, -110)
            self.layer(fr, cam, self.G['branch'] * op, 0, -110, opacity=0.9)
            self.layer(fr, cam, self.G['bloss'] * op, 0, -110, color=SAKURA, opacity=0.9)
            self.layer(fr, cam, self.petals(t - self.t_but) * op, color=SAKURA, opacity=0.95)
        # 人, then the traveller it becomes
        self.draw_origin(fr, cam, t, 380, 760, label_y=1040)
        f = self.modern(t)
        become = smooth((t - self.t_means - 0.1) / 0.7)
        if f > 0:
            kit.put(fr, cam, self.big.at(f, soft=0.05), 380, 760, INK, 1 - become, 1 - 0.15 * become)
        self.layer(fr, cam, self.P['fig'] * (1 - op), opacity=become)
        # the cards
        self.draw_card(fr, cam, t, self.c_hito, self.t_alone, op=1 - smooth((t - self.t_but) / 0.3), pulse=self.t_hito)
        if t > self.t_but + 0.2:
            self.draw_card(fr, cam, t, self.c_jin, self.t_but + 0.3, op=1 - smooth((t - self.t_sannin + 0.4) / 0.3), pulse=self.t_jin)
        if t > self.t_sannin - 0.4:
            self.draw_card(fr, cam, t, self.c_nin, self.t_sannin - 0.3, pulse=self.t_nin)
        # one kanji, three readings
        from cards import INDIGO
        for j, (g, col) in enumerate(zip(self.three, (INDIGO, VERM, VERM))):
            f = (t - self.t_nin - 0.15 - 0.15 * j) / 0.25
            if f > 0:
                kit.put(fr, cam, g, 300 + 240 * j, 420, col, smooth(f), 1 + 0.3 * np.exp(-f * 2.5))
                if j:
                    kit.put(fr, cam, self.dot, 180 + 240 * j, 420, INK, smooth(f))
        self.reading_seal(fr, cam, t, self.t_hito, '訓', op=1 - smooth((t - self.t_but) / 0.3))
        self.reading_seal(fr, cam, t, self.t_jin, '音')
        self.reading_seal(fr, cam, t, self.t_nin, '音')
        self.header(fr, t)
        return fr.img

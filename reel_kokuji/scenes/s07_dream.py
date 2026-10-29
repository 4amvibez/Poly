"""Scene 7 — the motivational ending.  "Every kanji has a story.  Every mistake helps you grow.  Keep learning, one step
at a time.  Your Japanese journey is still being written!"  then, in Japanese, 一歩ずつ、夢に近づこう。
The kanji of the reel (働 峠 畑 凪 雫 辻 枠 匂) float up off the page into the mist as small paper tags.  A lone student
with a rucksack walks along a winding path through the hills towards a vermilion sunrise behind the mountain.  On
"Keep learning" a final brushstroke writes 夢 large in the sky with ゆめ / DREAM beside it; as Raj says the Japanese
line it is set under it, 一歩ずつ、夢に近づこう。, with Ippo zutsu, yume ni chikazukō. and "Step by step, move closer
to your dream." below."""
import numpy as np

import k6art
import kit
import r5art
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, INK, SEA, VERM, Cam, cached, ease_io, smooth, text_layer
from scenes.kscene import KanjiScene
from timeline import JA, at

TAGS = [('働', 160, 1150), ('峠', 900, 1080), ('畑', 280, 980), ('凪', 760, 1220), ('雫', 470, 1120), ('辻', 980, 900),
        ('枠', 120, 860), ('匂', 620, 960)]


class Scene(KanjiScene):
    paper_seed = 207

    def build(self):
        self.F = cached('s07_final_v1', r5art.final_landscape)
        self.stud = cached('k6_student_v1', lambda: {'d': k6art.walker(320, 1330, 0.9, seed=9, hat=False, staff=False, pack=True)})['d']
        self.tag = kit.box(110, 84, 2.5)
        self.tags = [(text_layer(c, FONT_JP, 60)[0], x, y) for c, x, y in TAGS]
        self.yume = self.big_kanji('夢', size=360, seed=271)
        self.yr, _ = text_layer('ゆめ', FONT_JP, 64)
        self.ye, _ = text_layer('DREAM', FONT_EN_BOLD, 40, tracking=10)
        self.line, _ = text_layer('一歩ずつ、夢に近づこう。', FONT_JP, 66)
        self.rom, _ = text_layer('Ippo zutsu, yume ni chikazukō.', FONT_EN_ITALIC, 40)
        self.tr, _ = text_layer('"Step by step, move closer to your dream."', FONT_EN_ITALIC, 34)
        self.t_story = at('Every kanji has')[0]
        self.t_keep = at('Keep learning')[0]
        self.t_line = JA['dream_line']['start']
        self.t_hit = self.t_keep
        self.T_in = self.wipe_map((540, 300), seed=41)

    def cam(self, t):
        return Cam(540, 960 - 40 * smooth((t - self.t0) / 14), 1.0 + 0.04 * smooth((t - self.t0) / 14))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        m = self.wipe(self.T_in, ease_io((t - self.t0 + 0.1) / 1.2))
        rise = 60 * (1 - smooth((t - self.t0) / 8))
        self.layer(fr, cam, self.F['sun'] * m, 0, rise, color=VERM, opacity=0.85)
        self.layer(fr, cam, self.F['water'] * m, color=SEA, opacity=0.4)
        self.layer(fr, cam, self.F['ink'] * m)
        # the kanji float away into the mist
        for k, (g, x, y) in enumerate(self.tags):
            u = (t - self.t0 - 0.2 * k) / 6.0
            if 0 < u < 1:
                o = np.sin(u * np.pi) * 0.85
                yy = y - 420 * u
                sw = 0.08 * np.sin(t * 1.5 + k)
                kit.put(fr, cam, self.tag, x, yy, np.array([0.96, 0.93, 0.86], np.float32), o, rot=sw, over=True)
                kit.put(fr, cam, g, x, yy, INK, o, rot=sw)
        # the student walking towards the sunrise
        w = ease_io((t - self.t0) / 14)
        self.layer(fr, cam, self.stud, 260 * w, -120 * w + 2 * np.sin(t * 8), opacity=smooth((t - self.t0 - 0.5) / 0.6))
        # 夢
        fy = (t - self.t_keep + 0.2) / 0.6
        if fy > 0:
            kit.put(fr, cam, self.yume.at(fy, soft=0.05), 400, 560, INK, 1.0, 1 + 0.08 * np.exp(-fy * 3))
            kit.put(fr, cam, self.yr, 400, 785, VERM, smooth(fy - 0.8))
            kit.put(fr, cam, self.ye, 400, 850, INK, smooth(fy - 1.1))
        fl = (t - self.t_line + 0.1) / 0.4
        if fl > 0:
            self.plate(fr, cam, 540, 1060, 900, 250, smooth(fl))
            kit.put(fr, cam, self.line, 540, 990, INK, smooth(fl), 1 + 0.15 * np.exp(-fl * 3))
            kit.put(fr, cam, self.rom, 540, 1075, VERM, smooth(fl - 0.6))
            kit.put(fr, cam, self.tr, 540, 1135, GREY, smooth(fl - 1.0))
        return fr.img

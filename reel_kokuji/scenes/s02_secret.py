"""Scene 2 — the secret.  "They're called こくじ, Japanese-created kanji.  And some of them are hiding in plain sight!"
The mist clears on a scenic page: misty mountains, a vermilion torii, a cherry branch reaching in from the top
right.  国字 is brushed in large in the centre; on こくじ its furigana こくじ is set above it, pulsing; on
"Japanese-created kanji" JAPANESE-CREATED KANJI comes in under it.  On "hiding in plain sight" five everyday
kokuji (畑 峠 働 込 枠) slip out of the mist around it, one after another, and sink back."""
import numpy as np

import k6art
import kit
import r5art
from engine import FONT_EN_BOLD, FONT_JP, INK, SAKURA, VERM, Cam, cached, ease_io, smooth, text_layer
from scenes.kscene import KanjiScene
from timeline import JA, at

HIDE = [('畑', 150, 560), ('峠', 930, 600), ('働', 140, 1030), ('込', 950, 1010), ('枠', 820, 380)]


class Scene(KanjiScene):
    paper_seed = 202

    def build(self):
        self.Mt = cached('s03_mountain_v1', r5art.mountain_scene)['ink']
        self.T = cached('k6_torii_v2', lambda: k6art.torii(170, 1330, 0.72))
        self.B = cached('k6_branch_v1', k6art.blossom_branch)
        self.kj = [self.big_kanji(c, size=340, seed=210 + k) for k, c in enumerate('国字')]
        self.furi, _ = text_layer('こくじ', FONT_JP, 74)
        self.sub, _ = text_layer('JAPANESE-CREATED KANJI', FONT_EN_BOLD, 50, tracking=6)
        self.hide = [(text_layer(c, FONT_JP, 110)[0], x, y) for c, x, y in HIDE]
        self.t_kokuji = JA['kokuji']['start']
        self.t_jc = at('Japanese-created')[0]
        self.t_hide = at('hiding')[0]
        self.t_hit = self.t_kokuji
        self.T_in = self.wipe_map((540, 760), seed=21)

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.04 * smooth((t - self.t0) / 9) + 0.02 * np.exp(-max(t - self.t_kokuji, 0) * 6) * (t > self.t_kokuji))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        m = self.wipe(self.T_in, ease_io((t - self.t0) / 1.0))
        self.layer(fr, cam, self.Mt * m, 0, 120, opacity=0.35, depth=0.85)
        self.layer(fr, cam, self.T['red'] * m, color=VERM, opacity=0.9)
        self.layer(fr, cam, self.T['ink'] * m, opacity=0.95)
        sway = 4 * np.sin(t * 1.3)
        self.layer(fr, cam, self.B['ink'], sway, 0, opacity=0.9 * smooth((t - self.t0 - 0.3) / 0.6))
        self.layer(fr, cam, self.B['bloss'], sway, 0, color=SAKURA, opacity=0.95 * smooth((t - self.t0 - 0.5) / 0.6))
        for k, g in enumerate(self.kj):
            kit.put(fr, cam, g.at((t - self.t_kokuji + 0.55 - 0.25 * k) / 0.45, soft=0.05), 375 + 330 * k, 700, INK, 1.0)
        f = (t - self.t_kokuji) / 0.25
        if f > 0:
            kit.put(fr, cam, self.furi, 540, 460, VERM, smooth(f), 1 + 0.3 * np.exp(-f * 2.5))
        kit.put(fr, cam, self.sub, 540, 930, INK, smooth((t - self.t_jc) / 0.3))
        # hiding in plain sight
        for k, (g, x, y) in enumerate(self.hide):
            u = (t - self.t_hide - 0.3 * k) / 1.8
            if 0 < u < 1:
                kit.put(fr, cam, g, x, y + 20 * (1 - u), INK, 0.55 * np.sin(u * np.pi), 1 + 0.05 * u)
        return fr.img

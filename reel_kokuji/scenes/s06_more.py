"""Scene 6 — more kokuji.  "Dictionaries list hundreds of kokuji.  Here are a few more!"
HUNDREDS is brushed large in vermilion with KOKUJI under it and "listed in dictionaries" below.  On "Here are a few more!" eight small washi cards are
laid down one by one in a 4 x 2 grid, each with its kanji, a little brush motif, its reading and its meaning:
凪 なぎ calm sea · 雫 しずく drop · 辻 つじ crossroads · 枠 わく frame · 匂 におい scent · 笹 ささ bamboo grass ·
榊 さかき sakaki tree · 鰯 いわし sardine — then held still long enough to read (the narration pauses here)."""
import numpy as np

import k6art
import kit
from cards import INDIGO
from engine import FONT_BRUSH_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, GREY, INK, VERM, Cam, cached, noise, smooth, text_layer
from scenes.kscene import KanjiScene
from timeline import JA, at

CARDS = [('凪', 'なぎ', 'calm sea', 'wave'), ('雫', 'しずく', 'drop', 'drop'), ('辻', 'つじ', 'crossroads', 'cross'),
         ('枠', 'わく', 'frame', 'frame'), ('匂', 'におい', 'scent', 'scent'), ('笹', 'ささ', 'bamboo grass', 'leaf'),
         ('榊', 'さかき', 'sakaki tree', 'branch'), ('鰯', 'いわし', 'sardine', 'fish')]
CW, CH = 228, 400
XS = [148, 386, 624, 862]
YS = [735, 1150]
CARD_RGB = np.array([0.96, 0.93, 0.86], np.float32)


class Scene(KanjiScene):
    paper_seed = 206

    def build(self):
        self.over = text_layer('HUNDREDS', FONT_BRUSH_EN, 150)[0]
        self.over = self.over * (0.86 + 0.14 * noise(*self.over.shape, 6, 2, 3))
        self.kk, _ = text_layer('KOKUJI', FONT_HEAD, 56)
        self.sub, _ = text_layer('listed in dictionaries', FONT_EN_ITALIC, 36)
        self.box, self.rim = kit.box(CH, CW, 3.0), kit.rim(CH, CW, inset=10, t=3)
        self.icons = cached('k6_icons_v1', lambda: {k: k6art.icon(k, seed=40 + j) for j, (_, _, _, k) in enumerate(CARDS)})
        self.C = [(text_layer(c, FONT_JP, 120)[0], text_layer(r, FONT_JP, 40)[0], text_layer(m.upper(), FONT_EN_BOLD, 22, tracking=3)[0], ic)
                  for c, r, m, ic in CARDS]
        self.t_some = at('Dictionaries list')[0]
        self.t_over = at('hundreds')[0]
        self.t_few = at('Here are a few')[0]
        self.t_hit = self.t_over
        r = np.random.default_rng(66)
        pool = '凪雫辻枠匂笹榊鰯働峠畑込'
        self.ghost = [(text_layer(pool[k % len(pool)], FONT_JP, int(r.uniform(70, 130)))[0], r.uniform(60, 1020), r.uniform(200, 1400),
                       r.uniform(0, 1.5)) for k in range(34)]

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.02 * smooth((t - self.t0) / 8))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        # a sky of faint kokuji behind it all, thinning out as the cards come
        for g, x, y, d in self.ghost:
            o = 0.1 * smooth((t - self.t0 - d) / 0.6) * (1 - 0.6 * smooth((t - self.t_few) / 0.6))
            kit.put(fr, cam, g, x, y - 30 * smooth((t - self.t0) / 10), INK, o)
        fo = (t - self.t_over) / 0.18
        top = 1 - 0.35 * smooth((t - self.t_few) / 0.5)                    # the headline shrinks up as the cards arrive
        y0 = 380 - 90 * smooth((t - self.t_few) / 0.5)
        if fo > 0:
            kit.put(fr, cam, self.over, 540, y0, VERM, smooth(fo), top * (1 + 0.3 * np.exp(-fo * 2.5)), rot=-0.03)
        kit.put(fr, cam, self.kk, 540, y0 + 120 * top, INK, smooth((t - JA['kokuji2']['start']) / 0.3), top * (1 + 0.15 * np.exp(-max(t - JA['kokuji2']['start'], 0) * 4)))
        kit.put(fr, cam, self.sub, 540, y0 + 185 * top, GREY, smooth((t - self.t_over - 0.4) / 0.3) * (1 - smooth((t - self.t_few) / 0.4)))
        for k, (g, r, m, ic) in enumerate(self.C):
            f = (t - self.t_few - 0.3 * k) / 0.3
            if f <= 0:
                continue
            x, y = XS[k % 4], YS[k // 4]
            o = smooth(f)
            dy = 40 * (1 - smooth(f))
            kit.card(fr, cam, self.box, self.rim, x, y + dy, o, color=CARD_RGB)
            kit.put(fr, cam, g, x, y - 95 + dy, INK, o, 1 + 0.2 * np.exp(-f * 3))
            kit.put(fr, cam, self.icons[ic], x, y + 30 + dy, INK, 0.85 * o)
            kit.put(fr, cam, r, x, y + 105 + dy, INDIGO, o)
            kit.put(fr, cam, m, x, y + 150 + dy, GREY, o)
        return fr.img

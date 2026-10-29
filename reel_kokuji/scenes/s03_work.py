"""Scene 3 — 働, built.  "First, look at this character: うごく.  It means to move.  Now, add ひと, a person.  Combine
them, and you get はたらく, meaning to work!"
動 is brushed in large and pushes forward, leaving a trail of fading ink behind it; card 動 / うごく / to move.
"Now, add ひと": 動 steps aside to the right, 人 is brushed on the left and a traveller walks in along the hills
below; card 人 / ひと / person.  "Combine them": 人 folds into its narrow left-side form 亻 (the two shapes cross-
fade as it slims), the row reads 亻 + 動, and on "you get" the two slide together and 働 is brushed over them,
large, with an ink-stamp thud; card 働 / はたらく / to work.  The header seals (働 峠 畑) show where we are."""
import numpy as np

import k6art
import kit
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, INK, VERM, cached, ease_io, lerp, smooth, text_layer
from scenes.compound import BIG_Y, CompoundScene
from timeline import JA, at


class Scene(CompoundScene):
    paper_seed = 203
    chapter = 0

    def build(self):
        self.hills = cached('k6_hills_v2', lambda: k6art.mist_hills(1240, seed=8))['ink']
        self.walker = cached('k6_walker_v1', lambda: {'d': k6art.walker(0 + 140, 1290, 1.1, seed=5)})['d']
        self.dou = self.big_kanji('動', size=400, seed=231)
        self.hito = self.big_kanji('人', size=300, seed=232)
        self.nin = text_layer('亻', FONT_JP, 240)[0]
        self.dou_s = text_layer('動', FONT_JP, 240)[0]
        self.lab = {k: text_layer(v, FONT_EN_BOLD, 34, tracking=8)[0] for k, v in (('p', 'PERSON'), ('m', 'TO MOVE'))}
        self.plus, _ = text_layer('+', FONT_EN_BOLD, 96)
        self.c_dou = self.card('動', ['うごく'], ['kun'], 'ugoku', 'to move', seed=240)
        self.c_hito = self.card('人', ['ひと'], ['kun'], 'hito', 'person', seed=241)
        self.c_work = self.card('働', ['はたらく'], ['kun'], 'hataraku', 'to work', seed=242)
        self.t_first = at('First look')[0]
        self.t_ugoku = JA['ugoku']['start']
        self.t_add = at('Now add')[0]
        self.t_hito = JA['hito']['start']
        self.t_comb = at('Combine them')[0]
        self.t_get = at('you get')[0]
        self.t_read = JA['hataraku']['start']
        self.t_hit = self.t_read
        self.result('働', self.t_get, self.c_work, self.t_read, size=460, seed=243)
        self._note, _ = text_layer('a memory aid, not the full history', FONT_EN_ITALIC, 30)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.hills, opacity=0.8)
        # 動, pushing forward with a trail
        side = ease_io((t - self.t_add) / 0.6)                          # 動 steps aside for 人
        comb = ease_io((t - self.t_comb) / 0.5)                         # the row 亻 + 動
        merge = ease_io((t - self.t_get) / 0.5)
        fd = (t - self.t_first + 0.15) / 0.5
        if fd > 0:
            push = 30 * ease_io((t - self.t_first) / 1.6)
            x = lerp(540 + push, 760, side)
            x = lerp(x, 760, comb)
            x = lerp(x, 540, merge)
            s = lerp(1.0, 0.62, side)
            o = 1 - smooth((comb - 0.4) / 0.4)
            for k in range(3, 0, -1):                                   # the trail
                kit.put(fr, cam, self.dou.dens, x - 40 * k * (1 - side), 700, INK, 0.08 * k / 3 * smooth(fd) * (1 - side), s)
            kit.put(fr, cam, self.dou.at(fd, soft=0.05), x, 700, INK, o, s)
            # in the row: the plain 動 beside 亻
            if comb > 0:
                kit.put(fr, cam, self.dou_s, x, 700, INK, smooth((comb - 0.3) / 0.4) * (1 - smooth((merge - 0.55) / 0.45)), 1.0)
                kit.put(fr, cam, self.lab['m'], x, 860, GREY, comb * (1 - merge))
        # 人, then 亻
        fh = (t - self.t_add - 0.2) / 0.45
        if fh > 0:
            x = lerp(300, 330, comb)
            x = lerp(x, 540, merge)
            kit.put(fr, cam, self.hito.at(fh, soft=0.05), x, 700, INK, 1 - smooth((comb - 0.1) / 0.5), lerp(1.0, 0.7, comb), 1.0)
            fn = smooth((comb - 0.2) / 0.5) * (1 - smooth((merge - 0.55) / 0.45))
            kit.put(fr, cam, self.nin, x, 700, INK, fn, lerp(1.4, 1.0, smooth((comb - 0.2) / 0.5)), 1.0)
            kit.put(fr, cam, self.lab['p'], x, 860, GREY, smooth(fh - 1) * (1 - merge))
            kit.put(fr, cam, self.plus, 545, 700, VERM, smooth((comb - 0.5) / 0.3) * (1 - merge))
        # the traveller walking in on "a person"
        fw = (t - self.t_hito) / 2.5
        if 0 < fw:
            self.layer(fr, cam, self.walker, -60 + 200 * ease_io(min(fw, 1)), 2 * np.sin(t * 9), opacity=smooth(fw * 4) * (1 - merge))
        # 働, brushed over the merge
        fb = (t - self.t_get - 0.3) / 0.55
        if fb > 0:
            kit.put(fr, cam, self._big.at(fb, soft=0.05), 540, BIG_Y, INK, 1.0, 1 + 0.1 * np.exp(-fb * 3))
        # the cards
        self.draw_card(fr, cam, t, self.c_dou, self.t_ugoku - 0.35, op=1 - smooth((t - self.t_add) / 0.3), pulse=self.t_ugoku)
        if t > self.t_add:
            self.draw_card(fr, cam, t, self.c_hito, self.t_hito - 0.35, op=1 - smooth((t - self.t_comb) / 0.3), pulse=self.t_hito)
        if t > self.t_get:
            self.draw_card(fr, cam, t, self.c_work, self.t_read - 0.35, pulse=self.t_read)
        kit.put(fr, cam, self._note, 540, 380, GREY, 0.8 * smooth((t - self.t_comb) / 0.4))
        self.header(fr, t)
        return fr.img

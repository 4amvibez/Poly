"""compound — what the three build scenes (働 峠 畑) share: the parts set in a row with + between them and a label
under each, the row sliding together into one brushed kanji (the parts fade as it is written over them), its word
card writing itself in as Raj says the reading, and the small note that the build is a memory aid."""
import numpy as np

import kit
from cards import INDIGO
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, INK, VERM, Cam, ease_io, lerp, smooth, text_layer
from scenes.kscene import KanjiScene

ROW_Y = 700
BIG_Y = 700
NOTE = 'a memory aid, not the full history'


class CompoundScene(KanjiScene):
    def parts(self, chars, labels, times, xs, size=200, color=INK):
        self._parts = [dict(g=text_layer(c, FONT_JP, size)[0], lab=text_layer(l, FONT_EN_BOLD, 34, tracking=8)[0], t=t, x=x)
                       for c, l, t, x in zip(chars, labels, times, xs)]
        self._plus, _ = text_layer('+', FONT_EN_BOLD, 96)
        self._note, _ = text_layer(NOTE, FONT_EN_ITALIC, 30)
        self._pcol = color

    def result(self, ch, t_merge, card, t_read, size=440, seed=0):
        self._big = self.big_kanji(ch, size=size, seed=seed)
        self.t_merge, self._card, self.t_read = t_merge, card, t_read

    def cam(self, t):
        p = 0.035 * np.exp(-max(t - self.t_merge - 0.3, 0) * 6) * (t > self.t_merge + 0.3)
        return Cam(540, 960, 1.0 + 0.03 * smooth((t - self.t0) / 8) + p)

    def draw_build(self, fr, cam, t, y=ROW_Y, out=1.0):
        merge = ease_io((t - self.t_merge) / 0.5)
        P = self._parts
        for j, p in enumerate(P):
            f = (t - p['t'] + 0.1) / 0.22
            if f <= 0:
                continue
            x = lerp(p['x'], 540, merge)
            o = smooth(f) * (1 - smooth((merge - 0.55) / 0.45)) * out
            col = self._pcol[j] if isinstance(self._pcol, (list, tuple)) else self._pcol
            kit.put(fr, cam, p['g'], x, y, col, o, 1 + 0.3 * np.exp(-f * 2.5))
            kit.put(fr, cam, p['lab'], x, y + 160, GREY, o * (1 - merge))
            if j:
                kit.put(fr, cam, self._plus, (P[j - 1]['x'] + p['x']) / 2, y, VERM, smooth(f) * (1 - merge) * out)
        fb = (t - self.t_merge - 0.3) / 0.55
        if fb > 0:
            kit.put(fr, cam, self._big.at(fb, soft=0.05), 540, BIG_Y, INK, out, 1 + 0.1 * np.exp(-fb * 3))
        self.draw_card(fr, cam, t, self._card, self.t_read - 0.35, op=out, pulse=self.t_read)
        kit.put(fr, cam, self._note, 540, 380, GREY, 0.8 * smooth((t - self.t_merge - 0.8) / 0.4) * out)

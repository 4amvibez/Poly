"""cards — a word the way this reel teaches it: the kanji large, the reading of each character set under it in its
colour (kun'yomi indigo, on'yomi vermilion, the rest grey), the romaji, and the English meaning smallest.
The card writes itself in (kanji, then readings, then romaji and meaning) and can pulse its focus reading when Raj
says it."""
import numpy as np

import kit
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, INK, VERM, smooth, text_layer

INDIGO = np.array([40, 62, 118], np.float32) / 255
KIND = {'kun': INDIGO, 'on': VERM, None: GREY}


class WordCard:
    def __init__(self, chars, readings, kinds, romaji, meaning, size=150, seed=0, focus=None):
        """chars '富士山', readings ['ふ', 'じ', 'さん'], kinds [None, None, 'on'] (one per character); focus = the index
        of the character being taught (defaults to the first with a kind)"""
        self.n, self.size = len(chars), size
        self.kanji = kit.Row(chars, size, font=FONT_JP, seed=seed)
        self.read = [text_layer(r, FONT_JP, int(size * 0.42))[0] for r in readings]
        self.kinds = kinds
        self.focus = focus if focus is not None else next((i for i, k in enumerate(kinds) if k), 0)
        self.rom, _ = text_layer(romaji, FONT_EN_ITALIC, int(size * 0.33))
        self.mean, _ = text_layer(meaning.upper(), FONT_EN_BOLD, int(size * 0.2), tracking=4)
        self.bar = kit.blur(np.pad(np.ones((6, int(size * 0.8)), np.float32), 4), 1.5)

    def draw(self, fr, cam, x, y, t0, op=1.0, pulse=None, scale=1.0, dim=1.0, pulses=None):
        """write the card in from t0 at (x, y) = the centre of the kanji row; pulse = when Raj says the focus reading;
        pulses = {character index: time} to pulse other readings too"""
        t = self._t
        s = self.size * scale
        if op <= 0 or t < t0:
            return
        self.kanji.draw(fr, cam, x, y, lambda i: (t - t0 - 0.08 * i) / 0.3, op=lambda i: op * (1.0 if i == self.focus or self.n == 1 else dim),
                        scale=scale)
        pt = dict(pulses or {})
        if pulse is not None:
            pt[self.focus] = pulse
        pp = {i: (0.0 if t < tp else np.exp(-(t - tp) * 4)) for i, tp in pt.items()}
        for i, (g, kind) in enumerate(zip(self.read, self.kinds)):
            p = pp.get(i, 0.0)
            f = smooth((t - t0 - 0.3 - 0.06 * i) / 0.25)
            xi = x + self.kanji.x[i] * scale
            foc = i == self.focus
            kit.put(fr, cam, g, xi, y + 0.8 * s, KIND[kind], op * f * (1.0 if foc or kind else 0.85), scale * (1 + 0.3 * p))
            if foc and kind:
                kit.put(fr, cam, self.bar, xi, y + 1.06 * s, KIND[kind], op * f * (0.7 + 0.3 * p), scale)
        kit.put(fr, cam, self.rom, x, y + 1.3 * s, INK, op * smooth((t - t0 - 0.5) / 0.3), scale)
        kit.put(fr, cam, self.mean, x, y + 1.58 * s, GREY, op * smooth((t - t0 - 0.65) / 0.3), scale)

    def at(self, t):
        self._t = t
        return self

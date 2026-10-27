"""clues — the recurring "clue" title used for kanji, hiragana and katakana: a carved vermilion seal with the clue's
number, the script's name brushed in Japanese, and its romaji.  It stamps in large, then settles as a header at the
top of the page while the clue is explained."""
import numpy as np

from engine import FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, GREY, VERM, blur, ease_io, lerp, noise, smooth, text_layer
from glyphs import typeset


def seal(n, size=150):
    """a square vermilion seal with the number cut out of it (density for VERM)"""
    a, _ = text_layer(str(n), FONT_EN_BOLD, int(size * 0.78))
    sd = n if isinstance(n, int) else sum(map(ord, str(n)))
    sq = np.zeros((size, size), np.float32)
    sq[6:-6, 6:-6] = 1
    sq = blur(sq, 1.4) * (0.8 + 0.2 * noise(size, size, 9, 3, sd))
    edge = (noise(size, size, 5, 2, sd + 7) - 0.5) * 0.25
    sq = np.clip(sq + edge * (sq > 0.05), 0, 1)
    h, w = a.shape
    inner = np.zeros_like(sq)
    inner[(size - h) // 2 + 4:(size - h) // 2 + 4 + h, (size - w) // 2:(size - w) // 2 + w] = a[:size, :size]
    return np.clip(sq * (1 - inner * 1.15), 0, 1)


class Clue:
    def __init__(self, n, jp, romaji, t_in, t_settle, label=None, big=(620, 1.0)):
        self.seal = seal(n)
        self.lbl, _ = text_layer(label or f'CLUE {n}', FONT_EN_BOLD, 46, tracking=8)
        self.jp = [typeset(c, 150, font=FONT_BRUSH_JP, seed=40 + 7 * (n if isinstance(n, int) else 9) + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate(jp)]
        self.ro, _ = text_layer(romaji, FONT_EN_ITALIC, 56)
        self.t_in, self.t_settle = t_in, t_settle
        self.big = big                                               # where (y) and how large it first stamps

    def draw(self, scene, fr, cam, t, op=1.0):
        f = (t - self.t_in) / 0.35
        if f <= 0 or op <= 0:
            return
        m = ease_io((t - self.t_settle) / 0.6)                      # big in the middle -> header at the top
        cx, cy = 540, lerp(self.big[0], 300, m)
        sc = lerp(self.big[1], 0.5, m)
        punch = 1 + 0.25 * np.exp(-max(f, 0) * 2.5) * (f < 3)
        n = len(self.jp)
        jw = 150 * 1.02
        tot = 150 + 40 + n * jw                                      # seal + gap + characters
        x0 = cx - tot * sc / 2
        scene.place(fr, cam, self.seal, x0 + 75 * sc, cy, color=VERM, opacity=0.92 * smooth(f) * op, scale=sc * punch)
        for k, g in enumerate(self.jp):
            scene.place(fr, cam, g.at((t - self.t_in - 0.15 - 0.12 * k) / 0.35, soft=0.06), x0 + (190 + jw * (k + 0.5)) * sc, cy,
                        opacity=op, scale=sc)
        lo = smooth((t - self.t_in - 0.4) / 0.35) * op * (1 - m)
        scene.place(fr, cam, self.lbl, cx, cy - 135 * sc, color=VERM, opacity=lo)
        scene.place(fr, cam, self.ro, cx, cy + 130 * sc + 10, color=GREY, opacity=smooth((t - self.t_in - 0.5) / 0.35) * op, scale=lerp(self.big[1], 0.75, m))

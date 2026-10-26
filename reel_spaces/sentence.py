"""sentence — the reel's key sentence, 私は日本語を勉強します。, as a row of separately controllable characters.

Each character is a glyph from the Japanese font with its type (K kanji, H hiragana, P punctuation) and its chunk
(私は | 日本語を | 勉強します。), so a scene can write it in, colour the kanji or the hiragana, lift 日本語 out, or let the
three chunks drift apart.  Highlights are soft brush swashes laid behind the characters, like a marker in ink."""
import numpy as np

from engine import FONT_JP, StrokeSet, W, H, smooth, stroke
from glyphs import typeset
from poly_ink.brush import SWEEP

TEXT = '私は日本語を勉強します。'
KIND = 'KHKKKHKKHHHP'
CHUNK = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

# highlight colours, muted to sit in the ink palette (after the storyboard's yellow / pink / blue markers)
GOLD = np.array([0.86, 0.66, 0.24], np.float32)
PINK = np.array([0.86, 0.47, 0.47], np.float32)
BLUE = np.array([0.40, 0.55, 0.78], np.float32)
CHUNK_COL = [PINK, GOLD, BLUE]


class Sentence:
    def __init__(self, size=74, tracking=0.04, seed=0, text=TEXT, kind=KIND, chunk=CHUNK):
        self.size, self.text, self.kind, self.chunk = size, text, kind, chunk
        self.g = [typeset(c, size, font=FONT_JP, seed=seed + i, pad=0.04, medians=False, sweep=(1, 0.3)) for i, c in enumerate(text)]
        self.step = size * (1 + tracking)
        n = len(text)
        self.x = [(i - (n - 1) / 2) * self.step for i in range(n)]       # centre of each character, relative

    def span(self, idxs, gap=None):
        """(left, right) x of a run of characters, relative to the sentence centre"""
        xs = [self.x[i] for i in idxs]
        return min(xs) - self.size * 0.55, max(xs) + self.size * 0.55

    def swash(self, idxs, color_seed=0, dy=0.0, width=1.15):
        """a brush swash behind characters idxs: returns a StrokeSet in sentence-local coords (W x H canvas, centred)"""
        a, b = self.span(idxs)
        y = H / 2 + dy
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [(W / 2 + a - 6, y + 3), (W / 2 + (a + b) / 2, y - 2), (W / 2 + b + 6, y + 2)], self.size * width,
                      ink=0.95, dry=0.35, profile=SWEEP, seed=color_seed), 0, 1)
        return ss.render(5)

    def draw(self, scene, fr, cam, cx, cy, reveal, color=None, offset=None, opacity=None, scale=1.0):
        """reveal(i) -> progress 0..1 of character i; color(i) -> rgb; offset(i) -> (dx, dy); opacity(i) -> 0..1"""
        from engine import INK
        for i, g in enumerate(self.g):
            f = reveal(i)
            if f <= 0:
                continue
            op = 1.0 if opacity is None else opacity(i)
            if op <= 0:
                continue
            dx, dy = (0, 0) if offset is None else offset(i)
            col = INK if color is None else color(i)
            scene.place(fr, cam, g.at(min(f, 1.05), soft=0.05, wet=0.35), cx + (self.x[i] + dx) * scale, cy + dy * scale,
                        color=col, opacity=op, scale=scale)


def chunk_reveal(t, chunks, per=None):
    """a reveal function that writes the characters of each chunk across that chunk's time span (synced to speech)"""
    def f(i):
        c = CHUNK[i]
        a, b = chunks[c]
        members = [k for k in range(len(CHUNK)) if CHUNK[k] == c]
        j = members.index(i)
        d = (b - a) / len(members)
        return smooth((t - (a + j * d * 0.85)) / max(d * 1.4, 0.12))
    return f

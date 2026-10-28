"""subtitles — captions cut from the narration's word timestamps.

English cues are short (<= ~34 characters, breaking after punctuation), shown from the first word to the last, on
a feathered paper plate, each word lighting up as it is spoken, inside the Reels safe area (clear of the right-hand buttons and the caption area).  The
Japanese words Raj says inside the English (spoken by Raj in Japanese) are set in italic and coloured the way the
reel colours readings: kun'yomi indigo, on'yomi vermilion, whole words ink."""
import functools
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import FONT_EN, FONT_EN_ITALIC, VERM, blur
from timeline import WORDS

MAX_CHARS = 28
SIZE = 58
CX, CY = 520, 1436                 # slightly left of centre: clear of the like/comment/share column
INK_RGB = np.array([24, 22, 21], np.float32) / 255
PLATE_RGB = np.array([244, 238, 224], np.float32) / 255
INDIGO = np.array([40, 62, 118], np.float32) / 255
KUN = {'yama', 'hi', 'mizu', 'hito', 'kunyomi'}
ON = {'san', 'nichi', 'sui', 'jin', 'onyomi'}
COLOR = {'ink': INK_RGB, 'kun': INDIGO, 'on': VERM}
def _style(w):
    """(italic, colour key) for a word"""
    k = w.get('ja')
    if not k:
        return False, 'ink'
    return True, 'kun' if k in KUN else 'on' if k in ON else 'ink'


def _split(sent):
    """split one sentence into balanced cues of <= MAX_CHARS, preferring breaks after punctuation"""
    import itertools
    L = lambda ws: len(' '.join(x['text'] for x, _ in ws))
    if L(sent) <= MAX_CHARS:
        return [sent]
    best, bc = None, 1e18
    n = len(sent)
    for k in range(2, 5):
        for cuts in itertools.combinations(range(1, n), k - 1):
            parts = [sent[a:b] for a, b in zip((0,) + cuts, cuts + (n,))]
            lens = [L(p) for p in parts]
            if max(lens) > MAX_CHARS + 2:
                continue
            avg = sum(lens) / k
            cost = sum((x - avg) ** 2 for x in lens) + 60 * k
            for p in parts[:-1]:
                if p[-1][0]['text'][-1] not in ',:;':
                    cost += 90
                if len(p) == 1 or len(parts[parts.index(p) + 1]) == 1:
                    cost += 400                       # no orphan single words
            if cost < bc:
                best, bc = parts, cost
        if best:
            return best
    # a very long sentence: break at the punctuation (or greedily) and balance each piece
    out, cur = [], []
    for x in sent:
        cur.append(x)
        if x[0]['text'][-1] in ',:;' and L(cur) > MAX_CHARS // 2:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    res = []
    for p in out:
        if L(p) <= MAX_CHARS + 2:
            res.append(p)
            continue
        sub = _split(p) if len(p) < len(sent) else None
        if sub and all(L(q) <= MAX_CHARS + 2 for q in sub):
            res += sub
            continue
        line = []
        for x in p:
            if line and L(line + [x]) > MAX_CHARS:
                res.append(line)
                line = []
            line.append(x)
        res.append(line)
    return res


def build_cues():
    cues, cur = [], []
    for i, w in enumerate(WORDS):
        cur.append((w, i))
        t = w['text']
        if t[-1] in '.?!;' or t.endswith('."'):
            cues += _split(cur)
            cur = []
    if cur:
        cues += _split(cur)
    out = []
    for k, c in enumerate(cues):
        start = c[0][0]['start'] - 0.06
        end = c[-1][0]['end'] + 0.25
        if k + 1 < len(cues):
            end = min(end, cues[k + 1][0][0]['start'] - 0.08)
        words = [(w['text'], _style(w), w['start']) for w, i in c]
        out.append(dict(start=start, end=max(end, start + 0.6), words=words))
    return out


CUES = build_cues()


@functools.lru_cache(maxsize=128)
def _render(k):
    """(alpha of all the text, plate); the per-colour alphas are kept in _COLOURED"""
    words = CUES[k]['words']
    fr, fi = ImageFont.truetype(FONT_EN, SIZE), ImageFont.truetype(FONT_EN_ITALIC, SIZE + 2)
    space = fr.getlength(' ')
    widths = [(fi if it else fr).getlength(w) for w, (it, _), _ in words]
    tw = int(sum(widths) + space * (len(words) - 1))
    padx, pady = 34, 20
    w, h = tw + 2 * padx, int(SIZE * 1.25) + 2 * pady
    per = []
    x = padx
    for (word, (it, col), t0), ww in zip(words, widths):
        im = Image.new('L', (w, h), 0)
        ImageDraw.Draw(im).text((x, h / 2 + 2), word, font=fi if it else fr, fill=255, anchor='lm')
        per.append((np.asarray(im, np.float32) / 255, col, t0))
        x += ww + space
    _COLOURED[k] = per
    al = {'all': sum(a for a, _, _ in per)}
    plate = np.zeros((h, w), np.float32)
    plate[6:-6, 10:-10] = 1
    return sum(al.values()), blur(plate, 7) * 0.80


_COLOURED = {}


def draw(fr, t):
    for k, c in enumerate(CUES):
        if c['start'] <= t <= c['end']:
            fade = min(np.clip((t - c['start']) / 0.1, 0, 1), np.clip((c['end'] - t) / 0.08, 0, 1))
            if fade <= 0:
                continue
            _, plate = _render(k)
            h, w = plate.shape
            rise = int((1 - fade) * 6)
            x0, y0 = int(CX - w / 2), int(c.get('cy', CY) - h / 2) + rise
            sl = (slice(y0, y0 + h), slice(x0, x0 + w))
            img = fr.img[sl]
            pa = plate[..., None] * fade
            img[:] = img * (1 - pa) + PLATE_RGB * pa
            for a, key, t0 in _COLOURED[k]:                              # each word lights up as Raj says it
                lit = 0.3 + 0.7 * float(np.clip((t - t0 + 0.06) / 0.12, 0, 1))
                ta = a[..., None] * (fade * lit)
                img[:] = img * (1 - ta) + COLOR[key] * ta

"""subtitles — English captions cut from the narration's word timestamps.

Cues are short (<= ~34 characters, breaking after punctuation), shown from the first word to the last word,
on a feathered paper plate inside the Reels safe area (clear of the right-hand buttons and the caption area).
Romanised Japanese terms are set in italic."""
import functools
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import FONT_EN, FONT_EN_ITALIC, H, W, blur
from timeline import WORDS

MAX_CHARS = 34
SIZE = 50
CX, CY = 520, 1444                 # slightly left of centre: clear of the like/comment/share column
ITALIC = {'kanji', 'kana', 'hiragana', 'katakana'}
INK_RGB = np.array([24, 22, 21], np.float32) / 255
PLATE_RGB = np.array([244, 238, 224], np.float32) / 255


def _is_jp(word, i):
    """romanised Japanese: the script names and the quoted words ("nihongo", "wa", "o", "shimasu", "kōhī")"""
    w = re.sub(r'[^a-z]', '', word.lower())
    if w in ('kanji', 'kana', 'hiragana', 'katakana'):
        return True
    return w in ITALIC and '"' in word


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
    return [sent]


def build_cues():
    cues, cur = [], []
    for i, w in enumerate(WORDS):
        cur.append((w, i))
        t = w['text']
        if t[-1] in '.?!' or t.endswith('."'):
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
        out.append(dict(start=start, end=max(end, start + 0.6), words=[(w['text'], _is_jp(w['text'], i)) for w, i in c]))
    return out


def _fix(cues):
    """the Japanese lines (spoken by the Japanese voice) are captioned with their reading, in italic"""
    from timeline import JA
    READ = {'haha_kanji': 'Haha wa hana ga suki.', 'haha_kana': 'Haha wa hana ga suki.',
            'niwa_kana': 'Niwa ni wa niwa niwatori ga iru.', 'niwa_kanji': 'Niwa ni wa niwa niwatori ga iru.', 'matane': 'Mata ne!'}
    for key, txt in READ.items():
        a, b = JA[key]['start'], JA[key]['end']
        k = next((i for i, c in enumerate(cues) if c['start'] > a - 0.3), len(cues))
        if k > 0:
            cues[k - 1]['end'] = min(cues[k - 1]['end'], a - 0.1)
        cues.insert(k, dict(start=a - 0.06, end=b + 0.3, words=[(txt, True)]))
    return cues


CUES = _fix(build_cues())


@functools.lru_cache(maxsize=128)
def _render(k):
    words = CUES[k]['words']
    fr, fi = ImageFont.truetype(FONT_EN, SIZE), ImageFont.truetype(FONT_EN_ITALIC, SIZE + 2)
    space = fr.getlength(' ')
    widths = [(fi if jp else fr).getlength(w) for w, jp in words]
    tw = int(sum(widths) + space * (len(words) - 1))
    padx, pady = 34, 20
    w, h = tw + 2 * padx, int(SIZE * 1.25) + 2 * pady
    txt = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(txt)
    x = padx
    for (word, jp), ww in zip(words, widths):
        d.text((x, h / 2 + 2), word, font=fi if jp else fr, fill=255, anchor='lm')
        x += ww + space
    a = np.asarray(txt, np.float32) / 255
    plate = np.zeros((h, w), np.float32)
    plate[6:-6, 10:-10] = 1
    plate = blur(plate, 7) * 0.80
    return a, plate


def draw(fr, t):
    for k, c in enumerate(CUES):
        if c['start'] <= t <= c['end']:
            fade = min(np.clip((t - c['start']) / 0.1, 0, 1), np.clip((c['end'] - t) / 0.08, 0, 1))
            if fade <= 0:
                continue
            a, plate = _render(k)
            h, w = a.shape
            rise = int((1 - fade) * 6)
            x0, y0 = int(CX - w / 2), int(CY - h / 2) + rise
            sl = (slice(y0, y0 + h), slice(x0, x0 + w))
            img = fr.img[sl]
            pa = plate[..., None] * fade
            img[:] = img * (1 - pa) + PLATE_RGB * pa
            ta = a[..., None] * fade
            img[:] = img * (1 - ta) + INK_RGB * ta

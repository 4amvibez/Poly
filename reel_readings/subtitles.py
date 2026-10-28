"""subtitles — captions cut from the narration's word timestamps.

English cues are short (<= ~34 characters, breaking after punctuation), shown from the first word to the last, on
a feathered paper plate inside the Reels safe area (clear of the right-hand buttons and the caption area).  The
Japanese words Raj says inside the English (spoken by Raj in Japanese) are set in italic and coloured the way the
reel colours readings: kun'yomi indigo, on'yomi vermilion, whole words ink.
The Japanese line gets a three-tier plate, split at でも: the Japanese largest, the romaji smaller, the English
meaning smallest."""
import functools
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import FONT_EN, FONT_EN_ITALIC, FONT_JP, H, VERM, W, blur
from timeline import JA, WORDS, quiet_gap

MAX_CHARS = 34
SIZE = 50
CX, CY = 520, 1444                 # slightly left of centre: clear of the like/comment/share column
INK_RGB = np.array([24, 22, 21], np.float32) / 255
PLATE_RGB = np.array([244, 238, 224], np.float32) / 255
INDIGO = np.array([40, 62, 118], np.float32) / 255
KUN = {'yama', 'hi', 'mizu', 'hito', 'kunyomi'}
ON = {'san', 'nichi', 'sui', 'jin', 'onyomi'}
COLOR = {'ink': INK_RGB, 'kun': INDIGO, 'on': VERM}
# the Japanese line, in two halves: (Japanese, romaji, English)
LINE = [('山は「やま」とも読みます。', 'Yama wa "yama" tomo yomimasu.', 'This kanji can also be read "yama".'),
        ('でも、富士山は「ふじさん」と読みます。', 'Demo, Fujisan wa "Fujisan" to yomimasu.', 'But Mount Fuji is read "Fujisan".')]
LINE_CY = 1392


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
    return [sent]


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
        words = [(w['text'].replace('JP', '@raj_in_jp') if w['text'] == 'JP' else w['text'], _style(w)) for w, i in c]
        out.append(dict(start=start, end=max(end, start + 0.6), words=words))
    return _handle(out)


def _handle(cues):
    """"Follow Raj in JP" is captioned with the handle itself"""
    for c in cues:
        txt = [w for w, _ in c['words']]
        for i in range(len(txt) - 2):
            if txt[i] == 'Raj' and txt[i + 1] == 'in' and txt[i + 2].startswith('@raj_in_jp'):
                c['words'] = c['words'][:i] + [('@raj_in_jp', (False, 'ink'))] + c['words'][i + 3:]
                break
    return cues


def _line_cues(cues):
    a, b = JA['yama_line']['start'], JA['yama_line']['end']
    mid, _ = quiet_gap(a + 1.2, b - 1.5)
    k = next((i for i, c in enumerate(cues) if c['start'] > a - 0.3), len(cues))
    if k > 0:
        cues[k - 1]['end'] = min(cues[k - 1]['end'], a - 0.1)
    for j, (s0, s1) in enumerate(((a - 0.06, mid), (mid, b + 0.35))):
        cues.insert(k + j, dict(start=s0, end=s1, words=[(LINE[j][0], (False, 'ink'))], tiers=LINE[j], cy=LINE_CY))
    return cues


CUES = _line_cues(build_cues())


@functools.lru_cache(maxsize=128)
def _render(k):
    """(alpha of all the text, plate); the per-colour alphas are kept in _COLOURED"""
    c = CUES[k]
    if 'tiers' in c:
        return _render_tiers(k)
    words = c['words']
    fr, fi = ImageFont.truetype(FONT_EN, SIZE), ImageFont.truetype(FONT_EN_ITALIC, SIZE + 2)
    space = fr.getlength(' ')
    widths = [(fi if it else fr).getlength(w) for w, (it, _) in words]
    tw = int(sum(widths) + space * (len(words) - 1))
    padx, pady = 34, 20
    w, h = tw + 2 * padx, int(SIZE * 1.25) + 2 * pady
    layers = {key: Image.new('L', (w, h), 0) for key in COLOR}
    x = padx
    for (word, (it, col)), ww in zip(words, widths):
        ImageDraw.Draw(layers[col]).text((x, h / 2 + 2), word, font=fi if it else fr, fill=255, anchor='lm')
        x += ww + space
    al = {key: np.asarray(im, np.float32) / 255 for key, im in layers.items()}
    _COLOURED[k] = al
    plate = np.zeros((h, w), np.float32)
    plate[6:-6, 10:-10] = 1
    return sum(al.values()), blur(plate, 7) * 0.80


def _render_tiers(k):
    ja, rom, en = CUES[k]['tiers']
    fj, fr, fe = ImageFont.truetype(FONT_JP, 42), ImageFont.truetype(FONT_EN_ITALIC, 36), ImageFont.truetype(FONT_EN, 31)
    tw = int(max(fj.getlength(ja), fr.getlength(rom), fe.getlength(en)))
    padx, pady = 34, 18
    w, h = tw + 2 * padx, 62 + 46 + 38 + 2 * pady
    layers = {key: Image.new('L', (w, h), 0) for key in COLOR}
    ImageDraw.Draw(layers['ink']).text((w / 2, pady + 30), ja, font=fj, fill=255, anchor='mm')
    ImageDraw.Draw(layers['ink']).text((w / 2, pady + 62 + 21), rom, font=fr, fill=255, anchor='mm')
    ImageDraw.Draw(layers['ink']).text((w / 2, pady + 62 + 46 + 18), en, font=fe, fill=200, anchor='mm')
    al = {key: np.asarray(im, np.float32) / 255 for key, im in layers.items()}
    _COLOURED[k] = al
    plate = np.zeros((h, w), np.float32)
    plate[6:-6, 10:-10] = 1
    return sum(al.values()), blur(plate, 7) * 0.84


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
            for key, a in _COLOURED[k].items():
                ta = a[..., None] * fade
                img[:] = img * (1 - ta) + COLOR[key] * ta

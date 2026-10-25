"""thumbnail — Instagram Reel covers in the film's own style (1080x1920).

The Reels feed shows the full 9:16 frame; the profile grid crops the centre 3:4 (1080x1440, y 240-1680), so every
word and the main motif sit inside y ~280-1640.  Two designs:
  A  "hook":  Why does Japanese have 3 scripts? — 漢字 / ひらがな / カタカナ over a brushed vermilion ？
  B  "then & now":  brush-written 字 on a manuscript meeting the neon GiGO sign, "3 scripts · 1 language"
    python thumbnail.py   -> output/thumbnail_A.jpg, output/thumbnail_B.jpg (+ crop previews in the scratch dir)"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from art import sun_disc  # noqa: E402
from engine import (FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, PAPER, SEPIA, VERM, W, Frame, StrokeSet,  # noqa: E402
                    blur, cached, make_paper, noise, spline, stroke, text_layer)
from glyphs import calligraphy, text_glyphs, typeset  # noqa: E402
from poly_ink.brush import splatter, sweep, wash as pwash  # noqa: E402

FONT_HEAD = os.path.join(HERE, 'fonts', 'NotoSerifJP-Black-latin.ttf')
OUT = os.path.join(HERE, 'output')


def full(a, cx, cy, anchor='c'):
    """a patch placed into a frame-sized array, centred at (cx, cy) (or left-aligned at cx)"""
    h, w = a.shape
    x = int(cx - w / 2) if anchor == 'c' else int(cx)
    y = int(cy - h / 2)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    D = np.zeros((H, W), np.float32)
    D[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
    return D


def put(fr, a, cx, cy, color=INK, op=1.0, anchor='c'):
    fr.ink(full(a, cx, cy, anchor), color, op)


def seal(text='墨', size=46):
    g = typeset(text, size, seed=3, medians=False, pad=0.12)
    hh, ww = g.h + 12, g.w + 12
    sq = np.zeros((hh, ww), np.float32)
    sq[3:-3, 3:-3] = 1
    sq = blur(sq, 1.0) * (0.82 + 0.18 * noise(hh, ww, 6, 2, 9))
    inner = np.zeros_like(sq)
    inner[6:6 + g.h, 6:6 + g.w] = g.dens
    return np.clip(sq * (1 - inner * 1.1), 0, 1)


def mountains():
    return cached('s11_mountains', lambda: None)['D']


def question_mark(cx, cy, s):
    """the hook's vermilion ？, brushed: one sweeping stroke and a heavy dot, scaled about (cx, cy)"""
    P = lambda pts: [(cx + (x - 548) * s, cy + (y - 880) * s) for x, y in pts]
    q = StrokeSet(W, H)
    q.add(stroke(W, H, spline(P([(392, 720), (420, 612), (535, 548), (668, 592), (700, 706), (628, 800), (560, 868),
                                  (544, 965), (548, 1048)]), 14), 82 * s, ink=0.95, dry=0.42,
                 profile=((0, 0.55), (0.08, 1.0), (0.7, 0.9), (1, 0.45)), seed=7), 0, 1)
    dcx, dcy = cx, cy + (1166 - 880) * s
    q.add(pwash(W, H, [(dcx + 46 * s * np.cos(a), dcy + 48 * s * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 28, endpoint=False)],
                ink=0.96, soft=2.0, seed=8, grain=0.18), 0, 1)
    q.add(splatter(W, H, dcx + 12, dcy, (110 * s, 70 * s), n=16, size=(1.5, 5), seed=9), 0, 1)
    return q.render(5)


def design_a():
    fr = Frame(make_paper(W, H, 30, stains=0.08))
    fr.ink(sun_disc(H, W, 540, 1000, 430, seed=31), VERM, 0.30)
    fr.ink(np.roll(mountains(), 290, axis=0) * 0.85, INK)
    fr.ink(question_mark(600, 930, 1.32), VERM, 0.92)
    # headline: the hook question, with the 3 in vermilion
    l1, _ = text_layer('Why does Japanese', FONT_HEAD, 82)
    put(fr, l1, 540, 318)
    pre, _ = text_layer('have', FONT_HEAD, 112)
    three, _ = text_layer('3', FONT_HEAD, 150)
    post, _ = text_layer('scripts?', FONT_HEAD, 112)
    gap = 26                                            # word spaces (text_layer trims them)
    tot = pre.shape[1] + three.shape[1] + post.shape[1] + 2 * gap
    x = 540 - tot / 2
    put(fr, pre, x, 448, anchor='l')
    put(fr, three, x + pre.shape[1] + gap, 436, VERM, anchor='l')
    put(fr, post, x + pre.shape[1] + three.shape[1] + 2 * gap, 448, anchor='l')
    # the three scripts, big, each with its name
    for txt, size, y, en in (('漢字', 230, 760, 'kanji'), ('ひらがな', 162, 1040, 'hiragana'), ('カタカナ', 162, 1275, 'katakana')):
        D, _, _ = text_glyphs(txt, size, seed=hash(txt) % 997, tracking=0.03)
        halo = np.clip(blur(cv2.dilate(D, np.ones((17, 17), np.uint8)), 7) * 1.4, 0, 1)
        fr.over(PAPER, full(halo, 540, y), 0.55)          # lift the letters off the red brushstroke
        put(fr, D, 540, y)
        lab, _ = text_layer(en, FONT_EN_ITALIC, 46)
        put(fr, lab, 540, y + size * 0.62 + 6, GREY)
    h, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 60, tracking=2)
    put(fr, h, 520, 1555)
    s = seal()
    put(fr, s, 520 + h.shape[1] / 2 + 36, 1555, VERM, 0.9)
    return fr.finish(0, vignette=0.18, grain=0.01)


def design_b():
    # bottom: the neon GiGO sign from the user's footage; top: the manuscript with the brush-written 字
    paper = make_paper(W, H, 31, stains=0.14)
    fr = Frame(paper)
    freeze = cv2.imread(os.path.join(HERE, 'footage', 'gigo_freeze.png'))[:, :, ::-1].astype(np.float32) / 255
    g = freeze.mean(2, keepdims=True)
    freeze = np.clip((freeze * 0.9 + g * 0.1) * np.array([1.03, 1.0, 0.96]), 0, 1)
    # place the sign so its words sit in the lower half of the cover
    M = np.array([[1.5, 0, 540 - 1.5 * 360], [0, 1.5, 1190 - 1.5 * 250]], np.float32)
    photo = cv2.warpAffine(freeze, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    # a diagonal dry-brush edge between paper and photo
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    edge = 860 + (xx - 540) * -0.18 + (noise(H, W, 40, 4, 12) - 0.5) * 70
    m = np.clip((yy - edge) / 14, 0, 1)
    fr.img = fr.img * (1 - m[..., None]) + photo * m[..., None]
    ss = StrokeSet(W, H)
    ss.add(sweep(W, H, [(-60, 975), (300, 905), (700, 830), (1140, 760)], 120, ink=0.92, dry=0.62, seed=4, wash=0.25), 0, 1)
    fr.ink(ss.render(5), INK)
    # manuscript: faint ruled columns and the brush-written 字
    rule = np.zeros((H, W), np.float32)
    for x in range(110, 1000, 124):
        rule[230:820, x:x + 2] = 1
    fr.ink(blur(rule, 0.7) * 0.35 * (1 - m), VERM)
    gz = calligraphy('字', 380, seed=7)
    D = np.zeros((H, W), np.float32)
    y0, x0 = 300, 250 - gz.w // 2
    D[y0:y0 + gz.h, x0:x0 + gz.w] = gz.dens
    fr.ink(D, INK)
    # headline on the paper, right of 字
    for txt, size, y, col in (('3 scripts', 92, 395, INK), ('1 language', 92, 505, VERM)):
        a, _ = text_layer(txt, FONT_HEAD, size)
        put(fr, a, 470, y, col, anchor='l')
    sub, _ = text_layer('the story of Japanese writing', FONT_EN_ITALIC, 44)
    put(fr, sub, 474, 590, GREY, anchor='l')
    # handle on a paper plate over the photo
    h, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 60, tracking=2)
    plate = np.zeros((H, W), np.float32)
    plate[1515:1600, 540 - h.shape[1] // 2 - 70:540 + h.shape[1] // 2 + 90] = 1
    fr.over(PAPER, blur(plate, 6), 0.92)
    put(fr, h, 525, 1557)
    s = seal()
    put(fr, s, 525 + h.shape[1] / 2 + 36, 1557, VERM, 0.9)
    return fr.finish(0, vignette=0.18, grain=0.01)


def crop_preview(img, path):
    """the cover as the feed shows it, the 3:4 profile-grid crop and the old 1:1 crop, side by side"""
    full = Image.fromarray(img).resize((360, 640))
    g34 = Image.fromarray(img[240:1680]).resize((360, 480))
    g11 = Image.fromarray(img[420:1500]).resize((360, 360))
    out = Image.new('RGB', (1120, 680), (30, 30, 30))
    out.paste(full, (10, 30)); out.paste(g34, (380, 30)); out.paste(g11, (750, 30))
    d = ImageDraw.Draw(out)
    for x, t in ((10, 'Reels feed 9:16'), (380, 'profile grid 3:4'), (750, 'square 1:1 (older grid)')):
        d.text((x, 8), t, fill=(255, 255, 255))
    out.save(path)


if __name__ == '__main__':
    scratch = os.environ.get('PREVIEW_DIR', '/tmp/claude-0/preview')
    os.makedirs(scratch, exist_ok=True)
    for name, fn in (('A', design_a), ('B', design_b)):
        img = fn()
        Image.fromarray(img).save(os.path.join(OUT, f'thumbnail_{name}.jpg'), quality=95)
        crop_preview(img, os.path.join(scratch, f'thumb_{name}_crops.png'))
        print('wrote', os.path.join(OUT, f'thumbnail_{name}.jpg'))

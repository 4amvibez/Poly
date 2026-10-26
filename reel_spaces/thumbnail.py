"""thumbnail — Instagram Reel covers in the film's own style (1080x1920).

The Reels feed shows the full 9:16 frame; the profile grid crops the centre 3:4 (y 240-1680), so every word and the
main motif sit inside y ~280-1640.  Two designs:
  A  "hook":    How do Japanese people read WITHOUT SPACES? over 私は日本語を勉強します。 with the ？ split lines trying
                themselves between the characters, above the river and the arched bridge from the film
  B  "secret":  the reader's head in profile with the five clue badges (漢字 かな 文法 語彙 文脈) pouring into a glowing
                brain, "The secret to reading Japanese", the sentence in its coloured chunks below
    python thumbnail.py   -> output/thumbnail_A.jpg, output/thumbnail_B.jpg (+ crop previews in the scratch dir)"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import head as HD  # noqa: E402
from art import figure, sun_disc  # noqa: E402
from backdrop import backdrop  # noqa: E402
from engine import (FONT_BRUSH_EN, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, GREY, H, INK, SAKURA, SEA, VERM, W,  # noqa: E402
                    Frame, StrokeSet, blur, cached, make_paper, noise, spline, stroke, text_layer)
from glyphs import text_glyphs  # noqa: E402
from poly_ink.brush import PRESS_DOWN  # noqa: E402
from sentence import BLUE, CHUNK_COL, GOLD, PINK, Sentence  # noqa: E402

OUT = os.path.join(HERE, 'output')


def full(a, cx, cy):
    """a patch placed into a frame-sized array, centred at (cx, cy)"""
    h, w = a.shape
    x, y = int(cx - w / 2), int(cy - h / 2)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    D = np.zeros((H, W), np.float32)
    D[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
    return D


def put(fr, a, cx, cy, color=INK, op=1.0):
    fr.ink(full(a, cx, cy), color, op)


def handle(fr, y):
    h, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 62, tracking=2)
    bn = StrokeSet(W, H)
    bn.add(stroke(W, H, [(300, y + 6), (540, y - 2), (780, y + 4)], 96, ink=0.95, dry=0.55,
                  profile=((0, 0.7), (0.06, 1.0), (0.85, 0.95), (1, 0.35)), seed=13), 0, 1)
    fr.ink(bn.render(5), VERM, 0.95)
    fr.over(np.array([0.98, 0.95, 0.88], np.float32), full(h, 540, y - 4), 1.0)


def deco(fr, B, op=1.0):
    fr.ink(B['land'], INK, op)
    fr.ink(B['deco'], INK, op * 0.9)
    fr.ink(B['bl'], SAKURA, op * 0.9)


def sentence(fr, S, cx, cy, swash=True):
    if swash:
        for k, idx in enumerate(([0, 1], [2, 3, 4, 5], [6, 7, 8, 9, 10])):
            put(fr, S.swash(idx, color_seed=80 + k, width=1.05), cx, cy, CHUNK_COL[k], 0.5)
    for i, g in enumerate(S.g):
        put(fr, g.dens, cx + S.x[i], cy)


def design_a():
    fr = Frame(make_paper(W, H, 61, stains=0.1))
    A = cached('r02_paint_v3', lambda: None)
    fr.ink(sun_disc(H, W, 850, 760, 80, seed=9), VERM, 0.85)
    fr.ink(A['river'], SEA, 0.5)
    fr.ink(A['river_rip'], SEA * 0.6, 0.6)
    fr.ink(A['refl'], INK)
    fr.ink(A['bridge'], INK)
    fs = StrokeSet(W, H)
    figure(fs, 540, 1084, 1.0, 0, seed=5, pose='stand', facing=1, dur=0.01)
    fr.ink(fs.render(9), INK)
    # title
    put(fr, text_layer('How do Japanese', FONT_HEAD, 92)[0], 540, 320)
    put(fr, text_layer('people read', FONT_HEAD, 92)[0], 540, 428)
    k, _ = text_layer('without spaces?', FONT_BRUSH_EN, 150)
    put(fr, k * (0.86 + 0.14 * noise(*k.shape, 6, 2, 6)), 540, 590, VERM)
    # the sentence, and the question of where to cut it
    S = Sentence(size=78, seed=400)
    sy = 905
    for i, g in enumerate(S.g):
        put(fr, g.dens, 540 + S.x[i], sy)
    q, _ = text_layer('？', FONT_JP, 40)
    for i in (1, 5):
        x = 540 + (S.x[i] + S.x[i + 1]) / 2
        d = np.zeros((150, 12), np.float32)
        for y in range(0, 150, 24):
            d[y:y + 14, 4:8] = 1
        put(fr, blur(d, 0.8), x, sy + 4, VERM, 0.95)
        put(fr, q, x, sy - 100, VERM)
    handle(fr, 1560)
    return fr.finish(0, vignette=0.18, grain=0.01)


def design_b():
    fr = Frame(make_paper(W, H, 68, stains=0.12))
    deco(fr, backdrop(seed=130, branch='right', ground=1430), 0.8)
    hx, hy = 540, 960
    hd = HD.build()
    put(fr, hd, hx - 20, hy)
    bx, by = hx - 20 - HD.HW / 2 + HD.BRAIN[0], hy - HD.HH / 2 + HD.BRAIN[1]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fr.over(np.array([1.0, 0.86, 0.52], np.float32), np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 62 ** 2)), 0.85)
    badges = [('漢字', 'kanji', GOLD), ('かな', 'kana', PINK), ('文法', 'grammar', BLUE),
              ('語彙', 'vocabulary', np.array([0.42, 0.62, 0.46], np.float32)), ('文脈', 'context', np.array([0.86, 0.46, 0.34], np.float32))]
    BX, BY = [160, 350, 540, 730, 920], 570
    rr = np.hypot(xx - 0, yy - 0)
    for k, ((jp, en, col), x) in enumerate(zip(badges, BX)):
        r = np.hypot(xx - x, yy - BY)
        fr.ink(np.clip((74 - r) / 2.5, 0, 1), col, 0.5)
        fr.ink(np.clip(1 - np.abs(r - 74) / 2.2, 0, 1), col * 0.7, 0.8)
        D, _, _ = text_glyphs(jp, 56, font=FONT_JP, seed=10 + k, tracking=0.02)
        put(fr, D, x, BY)
        put(fr, text_layer(en, FONT_EN_ITALIC, 36)[0], x, BY + 96, GREY)
        ss = StrokeSet(W, H)
        pts = [(x, BY + 118), (x + (bx - x) * 0.15, BY + 176), (x + (bx - x) * 0.6, by - 70), (bx + (x - bx) * 0.12, by - 8)]
        ss.add(stroke(W, H, spline(pts, 12), 9, ink=0.9, dry=0.45, profile=((0, 0.9), (0.6, 0.7), (1, 0.2)), seed=50 + k), 0, 1)
        fr.ink(ss.render(5), col * 0.85, 0.8)
    _ = rr
    put(fr, text_layer('The secret to', FONT_HEAD, 84)[0], 540, 300)
    k, _ = text_layer('reading Japanese', FONT_BRUSH_EN, 124)
    put(fr, k * (0.86 + 0.14 * noise(*k.shape, 6, 2, 6)), 540, 420, VERM)
    sentence(fr, Sentence(size=70, seed=420), 540, 1290)
    put(fr, text_layer('…without spaces', FONT_EN_ITALIC, 48)[0], 540, 1380, GREY)
    handle(fr, 1570)
    return fr.finish(0, vignette=0.18, grain=0.01)


def crop_preview(img, path):
    """the cover as the feed shows it, the 3:4 profile-grid crop and the old 1:1 crop, side by side"""
    full_ = Image.fromarray(img).resize((360, 640))
    g34 = Image.fromarray(img[240:1680]).resize((360, 480))
    g11 = Image.fromarray(img[420:1500]).resize((360, 360))
    out = Image.new('RGB', (1120, 680), (30, 30, 30))
    out.paste(full_, (10, 30)); out.paste(g34, (380, 30)); out.paste(g11, (750, 30))
    d = ImageDraw.Draw(out)
    for x, t in ((10, 'Reels feed 9:16'), (380, 'profile grid 3:4'), (750, 'square 1:1 (older grid)')):
        d.text((x, 8), t, fill=(255, 255, 255))
    out.save(path)


if __name__ == '__main__':
    scratch = os.environ.get('PREVIEW_DIR', '/tmp/claude-0/preview')
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('A', design_a), ('B', design_b)):
        img = fn()
        Image.fromarray(img).save(os.path.join(OUT, f'thumbnail_{name}.jpg'), quality=95)
        crop_preview(img, os.path.join(scratch, f'sthumb_{name}_crops.png'))
        print('wrote', os.path.join(OUT, f'thumbnail_{name}.jpg'))

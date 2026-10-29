"""thumbnail — covers for the merged reel (1080x1920), painted with the reel's own ink engine.
The profile grid shows the centre 3:4 (y 240-1680), so every word and the motif sit inside it.
  A  "one kanji":  ONE KANJI. / 3 READINGS?! over a huge brushed 人 on the vermilion sun, its readings round it,
                   ひと in indigo (KUN'YOMI), じん にん in vermilion (ON'YOMI), misty hills below
  B  "same kanji": SAME KANJI. DIFFERENT READING?! over 日曜日 with にち and び under the two 日, ringed
    python thumbnail.py   -> output/thumbnail_A.jpg, output/thumbnail_B.jpg
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REEL = os.path.join(HERE, '..', 'reel_readings')
sys.path.insert(0, REEL)
import kit  # noqa: E402
import r5art  # noqa: E402
from art import sun_disc  # noqa: E402
from cards import INDIGO  # noqa: E402
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, H, INK, VERM, W, Cam,  # noqa: E402
                    Frame, StrokeSet, cached, make_paper, noise, stroke, text_layer)
from glyphs import typeset  # noqa: E402

OUT = os.path.join(HERE, 'output')
IVORY = np.array([0.98, 0.95, 0.88], np.float32)
CAM = Cam(540, 960, 1.0)


def brush_text(s, size, seed):
    d, _ = text_layer(s, FONT_BRUSH_EN, size)
    return d * (0.86 + 0.14 * noise(*d.shape, 6, 2, seed))


def banner(fr, y=1590):
    bn = StrokeSet(W, H)
    bn.add(stroke(W, H, [(270, y + 6), (540, y - 4), (810, y + 4)], 112, ink=0.96, dry=0.5,
                  profile=((0, 0.6), (0.06, 1.0), (0.85, 0.95), (1, 0.4)), seed=171), 0, 1)
    fr.ink(bn.render(5), INK, 0.96)
    kit.put(fr, CAM, text_layer('@raj_in_jp', FONT_BRUSH_EN, 86)[0], 540, y - 6, IVORY, 1.0, over=True)
    kit.put(fr, CAM, text_layer('FOLLOW FOR MORE', FONT_EN_BOLD, 34, tracking=10)[0], 540, y - 100, INK, 1.0)


def tag(fr, x, y, name, gloss, col):
    n, _ = text_layer(name, FONT_EN_BOLD, 40, tracking=6)
    g, _ = text_layer(gloss, FONT_EN_ITALIC, 30)
    kit.put(fr, CAM, n, x, y, col, 1.0)
    kit.put(fr, CAM, g, x, y + 40, INK, 0.9)


def thumb_a():
    fr = Frame(make_paper(W, H, 77, stains=0.08))
    land = cached('s04_land_v1', r5art.sun_landscape)['ink']
    fr.ink(np.roll(land, 420, axis=0) * (np.arange(H)[:, None] > 420), INK, 0.75)
    fr.ink(sun_disc(H, W, 540, 880, 330, seed=55).astype(np.float32), VERM, 0.8)
    big = typeset('人', 640, font=FONT_BRUSH_JP, seed=101, medians=False, sweep=(0.7, 1))
    kit.put(fr, CAM, big.dens, 540, 900, INK, 1.0)
    # the title, inside the grid crop
    kit.put(fr, CAM, text_layer('ONE KANJI.', FONT_HEAD, 96)[0], 540, 330, INK, 1.0)
    kit.put(fr, CAM, brush_text('3 READINGS?!', 128, 7), 540, 470, VERM, 1.0, rot=-0.04)
    # the three readings round the kanji
    for r, x, y, col, rot in (('ひと', 180, 800, INDIGO, -0.08), ('じん', 905, 690, VERM, 0.07), ('にん', 925, 935, VERM, -0.06)):
        g, _ = text_layer(r, FONT_JP, 118)
        kit.put(fr, CAM, kit.blur(g, 6), x + 6, y + 8, IVORY, 0.9, 1.08, over=True)
        kit.put(fr, CAM, g, x, y, col, 1.0, rot=rot)
    tag(fr, 200, 1300, "KUN'YOMI", 'Japanese reading', INDIGO)
    tag(fr, 860, 1300, "ON'YOMI", 'Chinese-based', VERM)
    banner(fr)
    return fr


def thumb_b():
    fr = Frame(make_paper(W, H, 78, stains=0.08))
    land = cached('s04_land_v1', r5art.sun_landscape)['ink']
    fr.ink(np.roll(land, 360, axis=0) * (np.arange(H)[:, None] > 360), INK, 0.7)
    fr.ink(sun_disc(H, W, 540, 820, 280, seed=56).astype(np.float32), VERM, 0.75)
    for k, (c, x) in enumerate(zip('日曜日', (205, 540, 875))):
        g = typeset(c, 300, font=FONT_BRUSH_JP, seed=20 + k, medians=False, sweep=(0.7, 1))
        kit.put(fr, CAM, g.dens, x, 820, INK, 1.0)
    from scenes.kscene import KanjiScene
    ring = KanjiScene._circle(type('R', (), {})())
    for x, col in ((205, VERM), (875, INDIGO)):
        kit.put(fr, CAM, ring.render(1.3, red=True), x, 820, col, 1.0, 3.3)
    for r, x, col in (('にち', 205, VERM), ('び', 875, INDIGO)):
        kit.put(fr, CAM, text_layer(r, FONT_JP, 120)[0], x, 1110, col, 1.0)
    kit.put(fr, CAM, text_layer('SAME KANJI.', FONT_HEAD, 92)[0], 540, 330, INK, 1.0)
    kit.put(fr, CAM, brush_text('DIFFERENT READING?!', 86, 8), 540, 460, VERM, 1.0, rot=-0.04)
    banner(fr)
    return fr


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('A', thumb_a), ('B', thumb_b)):
        fr = fn()
        Image.fromarray((np.clip(fr.img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, f'thumbnail_{name}.jpg'), quality=95)
        print('wrote', name)

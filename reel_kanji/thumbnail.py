"""thumbnail — Instagram Reel covers in the film's own style (1080x1920).

The Reels feed shows the full 9:16 frame; the profile grid crops the centre 3:4 (y 240-1680), so every word and the
main motif sit inside y ~280-1640.  Two designs:
  A  "hook":     How was KANJI created? over the vermilion sun with 日 written in it, and the five-stage strip
                 oracle → bronze → seal → clerical → regular
  B  "artefact": the Shang oracle bone (photo supplied by @raj_in_jp) beside 漢字 brushed large, "3,000 years of history"
    python thumbnail.py   -> output/thumbnail_A.jpg, output/thumbnail_B.jpg (+ crop previews in the scratch dir)"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ancient import ancient  # noqa: E402
from art import sun_disc  # noqa: E402
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, GREY, H, INK, PAPER, VERM, W,  # noqa: E402
                    Frame, StrokeSet, blur, cached, make_paper, noise, stroke, text_layer)
from glyphs import calligraphy, typeset  # noqa: E402
from photos import cutout, place_photo  # noqa: E402
from poly_ink.brush import PRESS_DOWN  # noqa: E402

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


def handle(fr, y):
    h, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 62, tracking=2)
    bn = StrokeSet(W, H)
    bn.add(stroke(W, H, [(300, y + 6), (540, y - 2), (780, y + 4)], 96, ink=0.95, dry=0.55,
                  profile=((0, 0.7), (0.06, 1.0), (0.85, 0.95), (1, 0.35)), seed=13), 0, 1)
    fr.ink(bn.render(5), VERM, 0.95)
    fr.over(np.array([0.98, 0.95, 0.88], np.float32), full(h, 540, y - 4), 1.0)


def design_a():
    fr = Frame(make_paper(W, H, 41, stains=0.1))
    land = cached('k01_land_v2', lambda: None)
    fr.ink(sun_disc(H, W, 540, 1010, 215, seed=41), VERM, 0.95)
    fr.ink(land['far'] * 0.9, INK)
    fr.ink(land['near'], INK)
    # 日 written in the sun
    g = calligraphy('日', 330, seed=23)
    D = full(g.dens, 540, 1010)
    fr.over(np.array([0.95, 0.92, 0.85], np.float32), np.clip(blur(D, 7) * 1.5, 0, 1), 0.25)
    fr.ink(D, INK)
    # title
    put(fr, text_layer('How was', FONT_HEAD, 108)[0], 540, 330)
    k, _ = text_layer('KANJI', FONT_BRUSH_EN, 300)
    put(fr, k * (0.86 + 0.14 * noise(*k.shape, 6, 2, 6)), 535, 535, VERM)
    put(fr, text_layer('created?', FONT_HEAD, 122)[0], 540, 735)
    # the five stages, as a strip
    xs = [150, 345, 540, 735, 930]
    for k_, (st, x) in enumerate(zip(['oracle', 'bronze', 'seal', 'clerical', 'regular'], xs)):
        gg = calligraphy('日', 120, seed=30) if st == 'regular' else ancient('日', st, 120, seed=20 + k_)
        plate = np.clip(blur(cv2.dilate(gg.dens, np.ones((15, 15), np.uint8)), 6) * 1.5, 0, 1)
        fr.over(PAPER, full(plate, x, 1385), 0.7)
        put(fr, gg.dens, x, 1385, INK if st == 'regular' else np.array([0.32, 0.22, 0.14], np.float32))
        if k_ < 4:
            ar = StrokeSet(W, H)
            ar.add([stroke(W, H, [(x + 70, 1385), (x + 118, 1383)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=k_),
                    stroke(W, H, [(x + 104, 1372), (x + 119, 1383), (x + 104, 1395)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=9 + k_)], 0, 1)
            fr.ink(ar.render(5), GREY)
    handle(fr, 1560)
    return fr.finish(0, vignette=0.18, grain=0.01)


def design_b():
    fr = Frame(make_paper(W, H, 43, stains=0.14))
    ph = cutout('bone1')
    hh, ww = ph['alpha'].shape
    sc = 0.56
    ang = np.radians(-6)
    c, s = np.cos(ang) * sc, np.sin(ang) * sc
    cx, cy = 320, 960
    M = np.array([[c, -s, cx - (c * ww / 2 - s * hh / 2)], [s, c, cy - (s * ww / 2 + c * hh / 2)]], np.float32)
    place_photo(fr, ph, M, shadow=0.5, lift=1.6)
    # 漢字 brushed vertically on the right, 甲骨文 in vermilion beside the bone
    for k, ch in enumerate('漢字'):
        g = typeset(ch, 300, font=FONT_BRUSH_JP, seed=7 + k, medians=False)
        put(fr, g.dens, 800, 680 + k * 300)
    for k, ch in enumerate('甲骨文'):
        g = typeset(ch, 66, seed=5 + k, medians=False)
        put(fr, g.dens, 722 + k * 78, 1225, VERM, 0.9)
    put(fr, text_layer('oracle-bone script', FONT_EN_ITALIC, 38)[0], 800, 1295, GREY)
    put(fr, text_layer('How kanji began', FONT_HEAD, 92)[0], 540, 300)
    put(fr, text_layer('3,000 years of history', FONT_EN_ITALIC, 58)[0], 540, 400, GREY)
    handle(fr, 1580)
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
    for name, fn in (('A', design_a), ('B', design_b)):
        img = fn()
        Image.fromarray(img).save(os.path.join(OUT, f'thumbnail_{name}.jpg'), quality=95)
        crop_preview(img, os.path.join(scratch, f'kthumb_{name}_crops.png'))
        print('wrote', os.path.join(OUT, f'thumbnail_{name}.jpg'))

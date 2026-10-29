"""thumbnail — the cover (1080x1920), painted with the reel's own ink engine.  The profile grid shows the centre 3:4
(y 240-1680), so the title, the motif and the handle all sit inside it.
  JAPAN / CREATED ITS OWN / KANJI?! over a huge brushed 働 on the vermilion sun, 亻 + 動 = 働 under it, the torii and
  misty hills below, the @raj_in_jp banner at the bottom.
    python thumbnail.py   -> output/thumbnail.jpg
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k6art  # noqa: E402
import kit  # noqa: E402
from art import sun_disc  # noqa: E402
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_HEAD, FONT_JP, H, INK, VERM, W, Cam, Frame, StrokeSet,  # noqa: E402
                    cached, make_paper, noise, stroke, text_layer)
from glyphs import typeset  # noqa: E402

CAM = Cam(540, 960, 1.0)
IVORY = np.array([0.98, 0.95, 0.88], np.float32)


def brush(s, size, seed):
    d, _ = text_layer(s, FONT_BRUSH_EN, size)
    return d * (0.86 + 0.14 * noise(*d.shape, 6, 2, seed))


def main():
    fr = Frame(make_paper(W, H, 88, stains=0.08))
    fr.ink(cached('k6_hills_v4', lambda: k6art.mist_hills(1330, seed=21))['ink'], INK, 0.85)
    fr.ink(sun_disc(H, W, 540, 930, 320, seed=55).astype(np.float32), VERM, 0.8)
    T = cached('k6_torii_v3', lambda: k6art.torii(160, 1420, 0.6))
    fr.ink(T['red'], VERM, 0.9)
    fr.ink(T['ink'], INK, 0.95)
    big = typeset('働', 600, font=FONT_BRUSH_JP, seed=243, medians=False, sweep=(0.7, 1))
    kit.put(fr, CAM, big.dens, 540, 930, INK, 1.0)
    kit.put(fr, CAM, brush('JAPAN', 170, 1), 540, 330, VERM, 1.0, rot=-0.03)
    kit.put(fr, CAM, text_layer('CREATED ITS OWN', FONT_HEAD, 74)[0], 540, 460, INK, 1.0)
    kit.put(fr, CAM, brush('KANJI?!', 180, 2), 540, 590, VERM, 1.0, rot=-0.03)
    # the build, on a paper plate
    plate = kit.blur(np.pad(np.ones((150, 640), np.float32), 30), 14)
    kit.put(fr, CAM, plate, 540, 1305, IVORY, 0.85, over=True)
    for s_, x, col in (('亻', 300, INK), ('+', 410, VERM), ('動', 520, INK), ('=', 640, VERM), ('働', 770, INK)):
        f = FONT_EN_BOLD if s_ in '+=' else FONT_JP
        kit.put(fr, CAM, text_layer(s_, f, 110)[0], x, 1300, col, 1.0)
    bn = StrokeSet(W, H)
    bn.add(stroke(W, H, [(270, 1596), (540, 1586), (810, 1594)], 112, ink=0.96, dry=0.5,
                  profile=((0, 0.6), (0.06, 1.0), (0.85, 0.95), (1, 0.4)), seed=171), 0, 1)
    fr.ink(bn.render(5), INK, 0.96)
    kit.put(fr, CAM, text_layer('@raj_in_jp', FONT_BRUSH_EN, 86)[0], 540, 1584, IVORY, 1.0, over=True)
    os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
    out = os.path.join(HERE, 'output', 'thumbnail.jpg')
    Image.fromarray((np.clip(fr.img, 0, 1) * 255).astype(np.uint8)).save(out, quality=95)
    print('wrote', out)


if __name__ == '__main__':
    main()

"""thumbnail — Instagram Reel covers (1080x1920), made from the film's own frames with the @raj_in_jp banner added.
The profile grid crops the centre 3:4 (y 240-1680), so the title and motif sit inside it.
  A  "hook":      Is KANJI really necessary? over the red sun with 漢字 in it, the reader below
  B  "confusion": ははははながすき。 and the guesses around the reader's head: "Mother likes… noses?!"
    python thumbnail.py   -> output/thumbnail_A.jpg, output/thumbnail_B.jpg"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compose  # noqa: E402
from engine import FONT_EN_BOLD, H, VERM, W, Frame, StrokeSet, stroke, text_layer  # noqa: E402

OUT = os.path.join(HERE, 'output')


def handle(img, y):
    fr = Frame(img.astype(np.float32) / 255)
    h, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 62, tracking=2)
    bn = StrokeSet(W, H)
    bn.add(stroke(W, H, [(300, y + 6), (540, y - 2), (780, y + 4)], 96, ink=0.95, dry=0.55,
                  profile=((0, 0.7), (0.06, 1.0), (0.85, 0.95), (1, 0.35)), seed=13), 0, 1)
    fr.ink(bn.render(5), VERM, 0.95)
    hh, hw = h.shape
    A = np.zeros((H, W), np.float32)
    A[y - 4 - hh // 2:y - 4 - hh // 2 + hh, 540 - hw // 2:540 - hw // 2 + hw] = h
    fr.over(np.array([0.98, 0.95, 0.88], np.float32), A, 1.0)
    return (np.clip(fr.img, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, t, y in (('A', 1.6, 1590), ('B', 22.6, 1590)):
        img = compose.frame_at(t, subtitles=False)
        Image.fromarray(handle(img, y)).save(os.path.join(OUT, f'thumbnail_{name}.jpg'), quality=95)
        print('wrote', name)

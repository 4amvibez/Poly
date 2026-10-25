"""The Reel cover, painted with the same brush as the film.

    python -m reels.nihongo.thumbnail      # -> drawing_output/reels/nihongo_cover.png (+ crop previews)

1080x1920. Instagram shows a Reel cover full-frame in the Reels tab, as a 3:4 centre crop on the profile grid
and close to a centre square in feeds, so the question, the three scripts and "But why?" all sit inside the
centre square (y 420-1500); the handle stays inside the 3:4 crop.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

import poly_ink  # noqa: F401
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, SWEEP, brush_stroke, splatter, sweep

from .glyph import write
from .ink import INK, PAPER, VERM, H, W, compose, draw_text, noise, over, paper, text_width

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'drawing_output', 'reels')


def _blit(buf, st):
    for s in (st if isinstance(st, list) else [st]):
        if s is None:
            continue
        h, w = s.dens.shape
        b = buf[s.y0:s.y0 + h, s.x0:s.x0 + w]
        np.copyto(b, 1 - (1 - b) * (1 - s.dens))


def cover():
    S = 2                                   # paint at 2x and downsample, for crisp brush edges
    w, h = W * S, H * S
    u = lambda *v: tuple(a * S for a in v) if len(v) > 1 else v[0] * S
    pap = np.asarray(paper(w, h, seed=301), np.float32) / 255
    D = np.zeros((h, w), np.float32)
    R = np.zeros((h, w), np.float32)
    YY, XX = np.mgrid[0:h, 0:w].astype(np.float32)

    # the vermilion sun behind the middle character
    sx, sy, r = u(540), u(862), u(182)
    dist = np.hypot(XX - sx, YY - sy)
    over(R, np.clip((r - dist) / 4, 0, 1) * (0.8 + 0.2 * noise(h, w, u(16), 3, 7)) * 0.85)

    # misty ink mountains at the foot, and the opening dry-brush sweep
    p = Painting(w, h, seed=302)
    far = p.ridge(u(1360), peaks=[(u(150), u(170), u(170)), (u(560), u(90), u(220)), (u(930), u(210), u(170))],
                  rough=u(12), seed=303)
    p.wash(far, dens=0.17, decay=u(120), tex=u(80), edge=0.3, mist_y=u(1470), mist_h=u(50), seed=304)
    near = p.ridge(u(1440), peaks=[(u(60), u(90), u(130)), (u(1040), u(110), u(140))], rough=u(8), seed=305)
    p.wash(near, dens=0.26, decay=u(70), tex=u(60), edge=0.35, mist_y=u(1520), mist_h=u(35), seed=306)
    over(D, p.D)
    _blit(D, sweep(w, h, [u(40, 1318), u(330, 1296), u(700, 1290), u(1050, 1270)], u(52), ink=0.92, dry=0.52,
                   seed=7, wash=0.3))
    _blit(D, splatter(w, h, *u(1010, 1268), (u(50), u(22)), n=16, seed=3, direction=0.0))

    # the three scripts, one of each, brushed in stroke order: 漢 (kanji) あ (hiragana, on the sun) ア (katakana)
    size, y0 = 270, 726
    for k, (ch, x) in enumerate([('漢', 78), ('あ', 405), ('ア', 732)]):
        ws, _ = write(w, h, ch, *u(x, y0), u(size), 0.0, weight=0.112, dry=0.42, ink=0.97, seed=11 + 7 * k)
        for x_ in ws:
            _blit(D, x_.stroke)
    img = compose(pap, D, R)

    # typography
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    ink = tuple(int(c * 255) for c in INK)
    verm = tuple(int(c * 255) for c in VERM)
    spaced = lambda s: ' '.join(s)
    draw_text(d, u(540, 500), spaced('JAPANESE HAS'), u(56), ink, anchor='ms')
    # "3 SCRIPTS": the 3 in vermilion
    big = u(150)
    t3, ts = '3', ' SCRIPTS'
    wa, wb = text_width(t3, big), text_width(ts, big)
    x0 = u(540) - (wa + wb) / 2
    draw_text(d, (x0, u(660)), t3, big, verm, anchor='ls')
    draw_text(d, (x0 + wa, u(660)), ts, big, ink, anchor='ls')
    for x, lab in [(213, 'KANJI'), (540, 'HIRAGANA'), (867, 'KATAKANA')]:
        draw_text(d, u(x, 1062), spaced(lab), u(30), ink, anchor='ms')
    draw_text(d, u(540, 1222), 'But why?', u(132), verm, latin='italic', anchor='ms')
    # the handle on a small ink band, inside the profile-grid crop
    band = np.zeros((h, w), np.float32)
    _blit(band, sweep(w, h, [u(380, 1612), u(540, 1606), u(700, 1610)], u(70), ink=0.9, dry=0.4, seed=240, wash=0.6,
                      profile=((0, 0.6), (0.08, 1.0), (0.9, 0.95), (1, 0.4))))
    arr = np.asarray(im, np.float32) / 255
    arr = arr * (1 - band[..., None] * (1 - INK))
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    draw_text(d, u(540, 1624), '@raj_in_jp', u(44), (242, 233, 210), anchor='ms')
    # a small vermilion seal: 日本語, bottom right above the mist (inside the profile-grid crop)
    sx0, sy0, ss = u(946), u(1512), u(38)
    d.rectangle([sx0, sy0, sx0 + ss * 1.4, sy0 + ss * 1.25 * 3 + ss * 0.3], fill=verm)
    for i, ch in enumerate('日本語'):
        draw_text(d, (sx0 + ss * 0.7, sy0 + ss * 1.05 + i * ss * 1.25), ch, ss, (244, 236, 222), cjk='mincho-bold',
                  anchor='ms')
    im = im.resize((W, H), Image.LANCZOS)
    # vignette
    a = np.asarray(im, np.float32) / 255
    YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.hypot((XX - W / 2) / (W * 0.75), (YY - H / 2) / (H * 0.7))
    a = a * (1 - 0.13 * np.clip(rr - 0.35, 0, 1) ** 1.5)[..., None]
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    im = cover()
    path = os.path.join(OUT, 'nihongo_cover.png')
    im.save(path)
    im.convert('RGB').save(os.path.join(OUT, 'nihongo_cover.jpg'), quality=95)
    print(path)
    return im


if __name__ == '__main__':
    main()

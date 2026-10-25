"""Command line:
    python -m poly_ink cloud-temple out.png [--stages DIR] [--gif out.gif]
    python -m poly_ink landscape out.png [--title 山水] [--seal 山居] [--seed 1] [--size 1920x1080] [--no-sun] [--stages DIR] [--gif out.gif]
"""
import argparse
import os

from . import cloud_temple, find_cjk_font, ink_landscape, make_gif


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m poly_ink', description='Ink wash paintings drawn in code.')
    ap.add_argument('scene', choices=['cloud-temple', 'landscape'])
    ap.add_argument('out', help='output image (.png or .jpg)')
    ap.add_argument('--stages', help='also save each drawing stage as a PNG in this directory')
    ap.add_argument('--gif', help='also write a draw-on animation GIF')
    ap.add_argument('--title', default='山水')
    ap.add_argument('--seal', default='山居')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--size', default='1920x1080', help='WIDTHxHEIGHT (landscape only)')
    ap.add_argument('--no-sun', action='store_true')
    a = ap.parse_args(argv)

    if not find_cjk_font():
        print('warning: no CJK font found, title and seal are skipped (set INKPAINT_FONT to a .ttf/.ttc)')
    if a.scene == 'cloud-temple':
        img, stages = cloud_temple(with_stages=True)
    else:
        w, h = (int(v) for v in a.size.lower().split('x'))
        img, stages = ink_landscape(a.title, a.seal, a.seed, w, h, sun=not a.no_sun, with_stages=True)

    img.convert('RGB').save(a.out, **({'quality': 92} if a.out.lower().endswith(('.jpg', '.jpeg')) else {}))
    print('saved', a.out)
    if a.stages:
        os.makedirs(a.stages, exist_ok=True)
        for i, (name, im) in enumerate(stages):
            im.save(os.path.join(a.stages, f'{i:02d}_{name}.png'))
        print('saved', len(stages), 'stages to', a.stages)
    if a.gif:
        n, secs = make_gif(stages, a.gif)
        print('saved', a.gif, f'({n} frames, {secs:.2f} s)')


if __name__ == '__main__':
    main()

"""contact sheets for checking scenes:  python preview.py hook 0.5 2 4 6 8   (times in video seconds)
or a full-video sheet:  python preview.py all  (one frame every few seconds through compositor)"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get('PREVIEW_DIR', '/tmp/claude-0/preview')
os.makedirs(OUT, exist_ok=True)


def sheet(frames, labels, path, cols=None, scale=0.3):
    cols = cols or min(len(frames), 6)
    rows = (len(frames) + cols - 1) // cols
    w, h = int(1080 * scale), int(1920 * scale)
    im = Image.new('RGB', (cols * w, rows * (h + 22)), (30, 30, 30))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    for i, (fr, lb) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + 22)
        im.paste(Image.fromarray(fr).resize((w, h), Image.LANCZOS), (x, y + 22))
        d.text((x + 4, y + 3), lb, fill=(255, 255, 255), font=f)
    im.save(path)
    return path


if __name__ == '__main__':
    import compose
    name = sys.argv[1]
    times = [float(x) for x in sys.argv[2:]]
    frames, labels = [], []
    for t in times:
        t0 = time.time()
        fr = compose.frame_at(t, only=None if name == 'all' else name)
        frames.append(fr)
        labels.append(f'{t:.2f}s  ({time.time() - t0:.2f}s)')
        print(labels[-1], flush=True)
    p = sheet(frames, labels, os.path.join(OUT, f'{name}.png'))
    if os.environ.get('FULL'):
        for t, fr in zip(times, frames):
            Image.fromarray(fr).save(os.path.join(OUT, f'{name}_{t:06.2f}.png'))
    print(p)

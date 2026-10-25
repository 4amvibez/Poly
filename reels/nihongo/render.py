"""Render the film: every frame is painted in numpy and piped to ffmpeg, in parallel chunks, then muxed with the
synthesised soundtrack.

    python -m reels.nihongo.render                       # the whole film -> drawing_output/reels/nihongo.mp4
    python -m reels.nihongo.render --stills 1:0.5,3,6    # PNG stills of scene 1 at those local times
    python -m reels.nihongo.render --frames 20.5,64      # PNG stills of the finished film at global times
"""
import argparse
import importlib
import multiprocessing as mp
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from .ink import FPS, H, INK, W, draw_text, ramp, smooth, text_width
from .timeline import Timeline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(REPO, 'drawing_output', 'reels')

NAMES = {1: ('scenes_a', 'Title'), 2: ('scenes_a', 'BeforeWriting'), 3: ('scenes_b', 'Map'), 4: ('scenes_b', 'Kanji'),
         5: ('scenes_b', 'Hiragana'), 6: ('scenes_b', 'Katakana'), 7: ('scenes_c', 'ThreeScripts'),
         8: ('scenes_c', 'Sentence'), 9: ('scenes_c', 'Street'), 10: ('scenes_c', 'Finale'), 11: ('scenes_c', 'EndCard')}
CAPTION_Y = 1455          # centre of the caption band: clear of the Reels caption/handle area and side buttons
CREAM = np.array([0.965, 0.935, 0.87], np.float32)


def scene_classes():
    out = {}
    for n, (mod, cls) in NAMES.items():
        try:
            out[n] = getattr(importlib.import_module(f'.{mod}', __package__), cls)
        except (ImportError, AttributeError):
            pass
    return out


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ captions
class Captions:
    """English captions in EB Garamond (Japanese in Shippori Mincho): dark ink text on a soft, semi-transparent
    paper-coloured pad that just fits the words — readable over light paper and dark ink alike, and it hides as
    little of the painting as possible"""

    PAD_ALPHA = 0.8                       # opacity of the paper pad
    PAD_RGB = np.array([0.965, 0.945, 0.9], np.float32)
    TEXT_RGB = np.array([0.11, 0.10, 0.09], np.float32)

    def __init__(self, tl, size=50):
        self.size = size
        self.chunks = tl.captions(size=size)
        self._cache = {}

    def layers(self, lines):
        """(text mask, pad mask) for a caption, both W x 260, centred on CAPTION_Y"""
        from PIL import ImageFilter
        key = tuple(lines)
        if key not in self._cache:
            h = 260
            im = Image.new('L', (W, h), 0)
            d = ImageDraw.Draw(im)
            lh = self.size * 1.28
            y0 = h / 2 - (len(lines) - 1) * lh / 2 + self.size * 0.33
            for i, ln in enumerate(lines):
                draw_text(d, (540, y0 + i * lh), ln, self.size, 255, anchor='ms')
            tx = np.asarray(im, np.float32) / 255
            tw = max(text_width(ln, self.size) for ln in lines)
            th = (len(lines) - 1) * lh + self.size * 1.05
            pad = Image.new('L', (W, h), 0)
            px, py = 34, 22
            ImageDraw.Draw(pad).rounded_rectangle([540 - tw / 2 - px, h / 2 - th / 2 - py, 540 + tw / 2 + px,
                                                   h / 2 + th / 2 + py], radius=14, fill=255)
            pad = np.asarray(pad.filter(ImageFilter.GaussianBlur(7)), np.float32) / 255
            self._cache[key] = (tx, pad)
        return self._cache[key]

    def draw(self, img, T):
        live = [c for c in self.chunks if c[0] - 0.2 <= T <= c[1] + 0.25]
        if not live:
            return img
        for c in live:
            a_txt = ramp(T, c[0] - 0.18, c[0]) * (1 - ramp(T, c[1], c[1] + 0.2))
            tx, pad = self.layers(c[2])
            y0 = int(CAPTION_Y - tx.shape[0] / 2)
            sl = img[y0:y0 + tx.shape[0]]
            k = (pad * self.PAD_ALPHA * a_txt)[..., None]
            sl[:] = sl * (1 - k) + self.PAD_RGB * k
            k = (tx * a_txt)[..., None]
            sl[:] = sl * (1 - k) + self.TEXT_RGB * k
        return img

    def srt(self, path):
        def ts(x):
            h, r = divmod(x, 3600)
            m, s = divmod(r, 60)
            return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'.replace(',1000', ',999')
        with open(path, 'w', encoding='utf-8') as f:
            for i, (a, b, lines) in enumerate(self.chunks):
                f.write(f'{i + 1}\n{ts(a)} --> {ts(b)}\n' + '\n'.join(lines) + '\n\n')


# ------------------------------------------------------------------ the film
class Film:
    def __init__(self, only=None):
        self.tl = Timeline()
        classes = scene_classes()
        self.scenes = {}
        for n in sorted(self.tl.scenes):
            if only and n not in only:
                continue
            t0 = time.time()
            self.scenes[n] = classes[n](self.tl, n)
            print(f'  scene {n:2d} {classes[n].__name__:14s} built in {time.time() - t0:5.1f}s', flush=True)
        self.captions = Captions(self.tl)
        rng = np.random.default_rng(5)
        self.grain = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
        YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((XX - W / 2) / (W * 0.75), (YY - H / 2) / (H * 0.7))
        self.vig = (1 - 0.13 * np.clip(r - 0.35, 0, 1) ** 1.5).astype(np.float32)[..., None]
        self.specks = rng.random((16, 5))

    def blend(self, a, b, T):
        """scene a hands over to scene b around b's start: b's own enter() mask decides where b shows"""
        tl = self.tl
        s = tl.scenes[b]
        p = np.clip((T - (s['start'] - s['trans'] / 2)) / s['trans'], 0, 1)
        A = self.scenes[a].frame(T - tl.scenes[a]['start'])
        B = self.scenes[b].frame(T - s['start'])
        m = self.scenes[b].enter(T - s['start'], p)[..., None]
        out = A * (1 - m) + B * m
        if b != 5:                    # the wet rim where the ink has just arrived (not on the 安 match cut)
            out = out * (1 - 4 * m * (1 - m) * 0.16 * (1 - INK))
        return out

    def frame(self, T):
        tl = self.tl
        nums = sorted(self.scenes)
        cur = max([n for n in nums if tl.scenes[n]['start'] <= T] or [nums[0]])
        nxt, prv = cur + 1, cur - 1
        if nxt in self.scenes and T >= tl.scenes[nxt]['start'] - tl.scenes[nxt]['trans'] / 2:
            A = self.blend(cur, nxt, T)
        elif prv in self.scenes and T < tl.scenes[cur]['start'] + tl.scenes[cur]['trans'] / 2:
            A = self.blend(prv, cur, T)
        else:
            A = self.scenes[cur].frame(T - tl.scenes[cur]['start'])
        img = A
        # drifting paper fibres and the odd ink speck, very sparse
        for k, (a, b, c, d, e) in enumerate(self.specks):
            x = (a * W + T * (8 + 14 * c)) % W
            y = (b * H - T * (5 + 9 * d) + 30 * np.sin(T * 0.3 + k)) % H
            xi, yi = int(x), int(y)
            if 6 < xi < W - 6 and 6 < yi < H - 6:
                if e < 0.6:     # a fibre
                    ang = k + T * 0.2
                    for q in range(-4, 5):
                        px, py = int(x + q * np.cos(ang)), int(y + q * np.sin(ang) + 0.5 * np.sin(q))
                        img[py, px] = img[py, px] * 0.94
                else:           # an ink speck
                    img[yi - 1:yi + 1, xi - 1:xi + 1] *= 0.7
        img = img * (1 + 0.01 * np.sin(T * 0.45))
        img = self.captions.draw(img, T)
        img = img * self.vig
        g = self.grain[int(T * FPS) % len(self.grain)]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
        img = img + g * 0.011
        return img


# ------------------------------------------------------------------ parallel render
_FILM = None


def _work(args):
    k, f0, f1, path = args
    film = _FILM
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p',
           '-x264-params', 'keyint=60:min-keyint=30', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(f0, f1):
        p.stdin.write(to_u8(film.frame(i / FPS)).tobytes())
        if (i - f0) % 60 == 0:
            el = time.time() - t0
            print(f'    worker {k}: frame {i - f0}/{f1 - f0}  {el / max(1, i - f0 + 1):.2f}s/frame', flush=True)
    p.stdin.close()
    p.wait()
    return path


def render(workers=4, out=None):
    global _FILM
    os.makedirs(OUT_DIR, exist_ok=True)
    out = out or os.path.join(OUT_DIR, 'nihongo.mp4')
    tmp = os.path.join(OUT_DIR, '_chunks')
    os.makedirs(tmp, exist_ok=True)
    print('building scenes', flush=True)
    _FILM = Film()
    tl = _FILM.tl
    print('soundtrack', flush=True)
    from . import audio
    wav = audio.build(tl, _FILM.scenes, os.path.join(tmp, 'mix.wav'))
    _FILM.captions.srt(os.path.join(OUT_DIR, 'nihongo.srt'))
    n = int(round(tl.total * FPS))
    # interleave the chunks so each worker gets a mix of cheap and expensive scenes
    k_chunks = workers * 3
    bounds = np.linspace(0, n, k_chunks + 1).astype(int)
    jobs = [(k, bounds[k], bounds[k + 1], os.path.join(tmp, f'c{k:02d}.mp4')) for k in range(k_chunks)]
    print(f'rendering {n} frames with {workers} workers', flush=True)
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        paths = pool.map(_work, jobs, chunksize=1)
    print(f'frames done in {(time.time() - t0) / 60:.1f} min', flush=True)
    lst = os.path.join(tmp, 'list.txt')
    with open(lst, 'w') as f:
        f.writelines(f"file '{os.path.basename(p)}'\n" for p in paths)
    silent = os.path.join(tmp, 'video.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent],
                   check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
                    '-shortest', '-movflags', '+faststart', out], check=True)
    print('wrote', out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stills', help='scene:t1,t2,... local times')
    ap.add_argument('--frames', help='global times, comma separated (finished frames with captions)')
    ap.add_argument('--audio', action='store_true', help='only build the soundtrack')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.stills:
        tl = Timeline()
        n, ts = a.stills.split(':')
        n = int(n)
        t0 = time.time()
        sc = scene_classes()[n](tl, n)
        print(f'scene {n} built in {time.time() - t0:.1f}s, dur {sc.dur:.2f}')
        out = a.out or os.path.join(OUT_DIR, 'stills')
        os.makedirs(out, exist_ok=True)
        for t in [float(x) for x in ts.split(',')]:
            t1 = time.time()
            Image.fromarray(to_u8(sc.frame(t))).save(os.path.join(out, f's{n}_{t:05.2f}.png'))
            print(f'  t={t:.2f} {time.time() - t1:.2f}s')
    elif a.frames or a.audio:
        times = [float(x) for x in a.frames.split(',')] if a.frames else []
        tl = Timeline()
        need = set()
        for T in times:
            for n, s in tl.scenes.items():
                if s['start'] - 1.0 <= T <= s['end'] + 1.0:
                    need.add(n)
        film = Film(only=None if a.audio else need)
        out = a.out or os.path.join(OUT_DIR, 'stills')
        os.makedirs(out, exist_ok=True)
        if a.audio:
            from . import audio
            print(audio.build(film.tl, film.scenes, os.path.join(out, 'mix.wav')))
        for T in times:
            Image.fromarray(to_u8(film.frame(T))).save(os.path.join(out, f'f_{T:06.2f}.png'))
            print('  frame', T)
    else:
        render(a.workers, a.out)


if __name__ == '__main__':
    main()

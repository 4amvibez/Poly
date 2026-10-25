"""Draw-on animation: stage snapshots -> looping GIF, to the claude-drawing skill's spec (720x405, each stage
held 900 ms, 5 crossfade frames of 90 ms, final frame held 2600 ms, one shared 256-colour palette)."""
import glob
import os

import numpy as np
from PIL import Image

_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5


def _load(stages, size):
    """stages: a directory of numbered PNGs, a list of (name, Image), or a list of Images / paths"""
    if isinstance(stages, (str, os.PathLike)):
        stages = sorted(glob.glob(os.path.join(stages, '*.png')))
    out = []
    for s in stages:
        im = s[1] if isinstance(s, tuple) else s
        im = Image.open(im) if isinstance(im, (str, os.PathLike)) else im
        out.append(np.asarray(im.convert('RGB').resize(size, Image.LANCZOS), np.float32))
    return out


def make_gif(stages, out, size=(720, 405), hold=900, fade_frames=5, fade_ms=90, final_hold=2600):
    """Write a looping draw-on GIF to `out`; returns (frame_count, total_seconds)."""
    keys = _load(stages, size)
    if len(keys) < 2:
        raise ValueError('need at least two stages')
    frames, durations = [], []
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        frames.append(k); durations.append(final_hold if last else hold)
        if not last:
            for t in np.linspace(0, 1, fade_frames + 2)[1:-1]:
                frames.append(k * (1 - t) + keys[i + 1] * t); durations.append(fade_ms)

    # shared palette from every keyframe, saturated pixels repeated 5x so the vermilion keeps its colour
    px = np.concatenate([k.reshape(-1, 3)[::7] for k in keys])
    sat = px.max(1) - px.min(1) > 40
    sample = np.concatenate([px] + [px[sat]] * 4).astype(np.uint8)
    side = int(np.ceil(np.sqrt(len(sample))))
    pad = np.zeros((side * side - len(sample), 3), np.uint8) + sample[0]
    pal = Image.fromarray(np.concatenate([sample, pad]).reshape(side, side, 3)).quantize(256, method=Image.MEDIANCUT)

    w, h = size
    dith = np.tile(_BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w, None] * 4.8      # fixed +-2.4 levels against banding
    q = [Image.fromarray(np.clip(f + dith, 0, 255).astype(np.uint8)).quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=False, disposal=1)
    return len(q), sum(durations) / 1000

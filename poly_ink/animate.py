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
    return write_gif(frames, durations, out, palette_from=keys)


def write_gif(frames, durations, out, palette_from=None):
    """frames: float RGB arrays (all the same size); one shared 256-colour palette, saturated pixels weighted x5
    so the vermilion keeps its colour, and a fixed Bayer dither against banding. Returns (frames, seconds)."""
    src = palette_from if palette_from is not None else frames[::max(1, len(frames) // 8)] + [frames[-1]]
    px = np.concatenate([k.reshape(-1, 3)[::7] for k in src])
    sat = px.max(1) - px.min(1) > 40
    sample = np.concatenate([px] + [px[sat]] * 4).astype(np.uint8)
    side = int(np.ceil(np.sqrt(len(sample))))
    pad = np.zeros((side * side - len(sample), 3), np.uint8) + sample[0]
    pal = Image.fromarray(np.concatenate([sample, pad]).reshape(side, side, 3)).quantize(256, method=Image.MEDIANCUT)

    h, w = frames[0].shape[:2]
    dith = np.tile(_BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w, None] * 4.8      # fixed +-2.4 levels against banding
    q = [Image.fromarray(np.clip(f + dith, 0, 255).astype(np.uint8)).quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=list(durations), loop=0, optimize=False, disposal=1)
    return len(q), sum(durations) / 1000


def animate_scene(scene, out, size=(720, 405), frame_ms=60, final_hold=2600, stages_dir=None):
    """Write the brush-by-brush painting of an InkScene (poly_ink.brush) as a looping GIF: the paper is sampled
    every frame_ms while each stroke is painted, unchanged frames are merged, and the finished painting is held
    for final_hold ms. Returns (frames, seconds)."""
    frames, durations, last = [], [], None
    for at in range(0, int(scene.length) + frame_ms, frame_ms):
        f = np.asarray(scene.render(at=at).convert('RGB').resize(size, Image.LANCZOS), np.float32)
        if last is not None and np.abs(f - last).max() < 1:
            durations[-1] += frame_ms
            continue
        frames.append(f); durations.append(frame_ms); last = f
    final = np.asarray(scene.render().convert('RGB').resize(size, Image.LANCZOS), np.float32)
    frames.append(final); durations.append(final_hold)
    if stages_dir:
        os.makedirs(stages_dir, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f.astype(np.uint8)).save(os.path.join(stages_dir, f'{i:03d}.png'))
    return write_gif(frames, durations, out)

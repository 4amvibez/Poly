"""Draw-on animation from --stages snapshots, to the skill's spec: 720x405, each stage held 900 ms, a 5-frame
(90 ms each) crossfade to the next, final frame held 2600 ms, infinite loop, one shared 256-colour palette
(saturated pixels weighted x5 so the red sun / seal / path keep their colour).
python3 make_gif.py STAGES_DIR out.gif
"""
import sys, glob, os
import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
keys = [np.asarray(Image.open(f).convert('RGB').resize((720, 405), Image.LANCZOS), np.float32)
        for f in sorted(glob.glob(os.path.join(src, '*.png')))]

frames, durations = [], []
for i, k in enumerate(keys):
    frames.append(k); durations.append(900 if i < len(keys) - 1 else 2600)
    if i < len(keys) - 1:
        for t in np.linspace(0, 1, 7)[1:-1]:
            frames.append(k * (1 - t) + keys[i + 1] * t); durations.append(90)

# shared palette: sample every keyframe, saturated pixels repeated 5x
px = np.concatenate([k.reshape(-1, 3)[::7] for k in keys])
sat = px.max(1) - px.min(1) > 40
sample = np.concatenate([px] + [px[sat]] * 4).astype(np.uint8)
side = int(np.ceil(np.sqrt(len(sample))))
pad = np.zeros((side * side - len(sample), 3), np.uint8) + sample[0]
pal_img = Image.fromarray(np.concatenate([sample, pad]).reshape(side, side, 3)).quantize(256, method=Image.MEDIANCUT)

bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5
dith = np.tile(bayer, (405 // 4 + 1, 720 // 4 + 1))[:405, :720, None] * 4.8       # fixed +-2.4 levels against banding
q = [Image.fromarray(np.clip(f + dith, 0, 255).astype(np.uint8)).quantize(palette=pal_img, dither=Image.NONE) for f in frames]
q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=False, disposal=1)
print('saved', out, len(q), 'frames', round(sum(durations) / 1000, 2), 's', round(os.path.getsize(out) / 1e6, 2), 'MB')

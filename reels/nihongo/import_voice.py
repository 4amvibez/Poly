"""Use a recorded or externally generated voiceover (one audio file per line of script.py) instead of Kokoro.

    python -m reels.nihongo.import_voice DIR [--voice NAME] [--tempo 1.0]

DIR holds l00.mp3 … l17.mp3 (or .wav), in script order. Each line is decoded, its leading and trailing silence
trimmed, and written to assets/voice/line_XX.wav; lines.json gets the measured durations, so the timeline,
captions and music follow the new voice. --tempo > 1 tightens the read without changing pitch (ffmpeg atempo).
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np

from .script import LINES, parse
from .voice import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--voice', default='recorded')
    ap.add_argument('--tempo', type=float, default=1.0)
    a = ap.parse_args()
    import soundfile as sf
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for i, (scene, line) in enumerate(LINES):
        src = sorted(glob.glob(os.path.join(a.dir, f'l{i:02d}.*')))[0]
        tmp = os.path.join(OUT, f'_tmp_{i:02d}.wav')
        af = ['-af', f'atempo={a.tempo}'] if a.tempo != 1.0 else []
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, *af, '-ac', '1', '-ar', '48000', tmp], check=True)
        x, sr = sf.read(tmp)
        os.remove(tmp)
        env = np.convolve(np.abs(x), np.ones(int(0.01 * sr)) / int(0.01 * sr), mode='same')
        on = np.where(env > 0.012 * env.max() * 4)[0]
        x = x[max(0, on[0] - int(0.04 * sr)): on[-1] + int(0.12 * sr)]
        x = x * np.minimum(1, np.arange(len(x)) / (0.01 * sr))[::1]                       # click-free start
        x = x * np.minimum(1, (len(x) - np.arange(len(x))) / (0.04 * sr))                # and end
        sf.write(os.path.join(OUT, f'line_{i:02d}.wav'), x.astype(np.float32), sr)
        cap, _ = parse(line)
        meta.append({'i': i, 'scene': scene, 'caption': cap, 'dur': len(x) / sr, 'sr': sr})
        print(f'{i:02d} {len(x) / sr:5.2f}s  {cap}')
    with open(os.path.join(OUT, 'lines.json'), 'w') as f:
        json.dump({'voice': a.voice, 'tempo': a.tempo, 'lines': meta}, f, ensure_ascii=False, indent=1)
    print('total speech', round(sum(m['dur'] for m in meta), 2), 's')


if __name__ == '__main__':
    main()

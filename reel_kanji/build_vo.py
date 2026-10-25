"""build_vo — the narration: Raj's take B of the full script (chosen for its breathing pauses between sections)
followed by the channel's Japanese sign-off in the "Raj in Japanese" voice (最後まで見てくれてありがとう。またね！).
Writes audio/narration.wav + .mp3 and audio/words.json (Scribe word timings of take B, plus the sign-off timings)."""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
AT_JA = 89.55            # the English ends at 88.64 s; a breath, then the sign-off


def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(A, name), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def fade(x, ms=12):
    n = int(SR * ms / 1000)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def main():
    en, ja = load('take_b.mp3'), load('ja_signoff.mp3')
    out = np.zeros(int((AT_JA + len(ja) / SR + 0.2) * SR), np.float32)
    for seg, at in ((fade(en), 0.0), (fade(ja), AT_JA)):
        i = int(at * SR)
        out[i:i + len(seg)] += seg
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    words = [dict(w, type='word') for w in json.load(open(os.path.join(A, 'words_take_b.json')))['words']
             if w.get('type', 'word') == 'word']
    # sign-off timings measured on the clip in the first reel (same recording)
    ja = dict(text='最後まで見てくれてありがとう。またね！', start=AT_JA, end=round(AT_JA + 2.8, 3),
              arigatou_end=round(AT_JA + 1.86, 3), matane=round(AT_JA + 2.3, 3))
    json.dump({'words': words, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False)
    print(f'narration {len(out) / SR:.2f}s, {len(words)} English words + Japanese sign-off at {AT_JA}s')


if __name__ == '__main__':
    main()

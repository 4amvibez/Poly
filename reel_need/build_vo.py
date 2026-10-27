"""build_vo — assemble the narration: Raj's English take B with the Japanese lines (Raj in Japanese) spliced into
its breaths:
  after "Look at this sentence."                  母は花が好き。      (haha wa hana ga suki)
  after "...write everything in hiragana."        ははははながすき。  (the same sentence, the same sound)
  after "Here's another example."                 にわにはにわにわとりがいる。 (niwa ni wa, niwa, niwatori ga iru)
  after "Now look at it with kanji."              庭には二羽鶏がいる。 (the same sentence, the same sound)
  after "...master Japanese together!"            またね！
The Japanese voice was given the spoken reading (ははわ、はなが すき / にわにわ、にわ、にわとりが いる) so the particle
は is said "wa"; each take was checked with Scribe before use.  Writes audio/narration.wav + .mp3 and audio/words.json
(English words shifted into place, plus the Japanese segments)."""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
# (cut point in take_en_b, clip, key, breath before, breath after)
SPLICES = [(6.64, 'haha_a.mp3', 'haha_kanji', 0.25, 0.45),
           (13.05, 'haha_a.mp3', 'haha_kana', 0.3, 0.7),
           (32.3, 'niwa_a.mp3', 'niwa_kana', 0.25, 0.45),
           (34.1, 'niwa_a.mp3', 'niwa_kanji', 0.25, 0.45)]
MATANE_GAP = 0.45
TEXT = {'haha_kanji': '母は花が好き。', 'haha_kana': 'ははははながすき。', 'niwa_kana': 'にわにはにわにわとりがいる。',
        'niwa_kanji': '庭には二羽鶏がいる。', 'matane': 'またね！'}


def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(A, name), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def speech(x, thresh=-40):
    """the clip trimmed to its speech (a few ms of air either side), faded"""
    h = SR // 100
    e = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(h) / h, 'same')[::h]) + 1e-9)
    on = np.where(e > thresh)[0]
    a, b = max(0, on[0] * h - SR // 50), min(len(x), (on[-1] + 1) * h + SR // 25)
    y = x[a:b].copy()
    n = SR // 100
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def main():
    en = load('take_en_b.mp3')
    words = [w for w in json.load(open(os.path.join(A, 'words_en_b.json')))['words'] if w.get('type', 'word') == 'word']
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    level = rms(en)
    parts, segs, shift_at = [], {}, []
    t_out, prev = 0.0, 0.0
    for cut, clip, key, pre, post in SPLICES:
        parts.append(en[int(prev * SR):int(cut * SR)])
        t_out += cut - prev
        j = speech(load(clip))
        j *= level / rms(j)
        gap1, gap2 = np.zeros(int(pre * SR), np.float32), np.zeros(int(post * SR), np.float32)
        parts += [gap1, j, gap2]
        segs[key] = dict(text=TEXT[key], start=round(t_out + pre, 3), end=round(t_out + pre + len(j) / SR, 3))
        ins = pre + len(j) / SR + post
        shift_at.append((cut, ins))
        t_out += ins
        prev = cut
    parts.append(en[int(prev * SR):])
    t_out += len(en) / SR - prev
    end_en = words[-1]['end'] + sum(d for _, d in shift_at)
    m = speech(load('matane_a.mp3'))
    m *= level / rms(m)
    out = np.concatenate(parts)
    i = int((end_en + MATANE_GAP) * SR)
    out = np.concatenate([out, np.zeros(max(0, i + len(m) + SR // 5 - len(out)), np.float32)])
    out[i:i + len(m)] += m
    segs['matane'] = dict(text=TEXT['matane'], start=round(end_en + MATANE_GAP, 3), end=round(end_en + MATANE_GAP + len(m) / SR, 3))
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    sh = lambda t: t + sum(d for c, d in shift_at if t >= c)
    ws = [dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)) for w in words]
    json.dump({'words': ws, 'ja': segs}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(out) / SR:.2f}s', {k: (v['start'], v['end']) for k, v in segs.items()})


if __name__ == '__main__':
    main()

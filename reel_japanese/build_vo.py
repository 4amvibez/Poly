"""build_vo — assemble the narration from the Raj takes and write the merged word timings.

  new hook ("Japanese has three scripts. But... why?")
  + the approved full take from "Long before..." through "...made its own."
  + new ending ("And you can still read it on every street in Japan today. Three scripts. One language. Follow ...")
  + the Japanese sign-off in the "Raj in Japanese" voice (最後まで見てくれてありがとう。またね！)
"""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
MID_IN, MID_OUT = 7.95, 100.05          # cut points in the full take (just before "Long", just after "own.")
AT_MID, AT_END, AT_JA = 4.05, 96.55, 105.0


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
    full, hook, end, ja = load('raj_b.mp3'), load('new_hook.mp3'), load('new_end.mp3'), load('new_ja.mp3')
    mid = fade(full[int(MID_IN * SR):int(MID_OUT * SR)])
    out = np.zeros(int((AT_JA + len(ja) / SR + 0.2) * SR), np.float32)
    for seg, at in ((fade(hook), 0.0), (mid, AT_MID), (fade(end), AT_END), (fade(ja), AT_JA)):
        i = int(at * SR)
        out[i:i + len(seg)] += seg
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    full_words = json.load(open(os.path.join(A, 'words_raj_full_take.json')))['words']
    new = json.load(open(os.path.join(A, 'new_lines_words.json')))
    words = [dict(w, type='word') for w in new['hook']]
    for w in full_words:
        if w.get('type', 'word') == 'word' and MID_IN <= w['start'] < MID_OUT:
            words.append(dict(text=w['text'], type='word', start=round(w['start'] - MID_IN + AT_MID, 3),
                              end=round(w['end'] - MID_IN + AT_MID, 3)))
    words += [dict(text=w['text'], type='word', start=w['start'] + AT_END, end=w['end'] + AT_END) for w in new['end']]
    ja = dict(text='最後まで見てくれてありがとう。またね！', start=AT_JA, end=round(AT_JA + 2.8, 3),
              arigatou_end=round(AT_JA + 1.86, 3), matane=round(AT_JA + 2.3, 3))
    json.dump({'words': words, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False)
    print(f'narration {len(out) / SR:.2f}s, {len(words)} English words + Japanese sign-off at {AT_JA}s')


if __name__ == '__main__':
    main()

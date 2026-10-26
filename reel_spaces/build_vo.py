"""build_vo — assemble the narration:
  Raj's English take A from its very first word (the reel opens on his question, no Japanese intro), with
  - the example sentence 私は日本語を勉強します。 (Raj in Japanese) spliced in right after "Look at this sentence."
  - コーヒー (Raj in Japanese) in place of the English-accented "kohee"
  + the channel's Japanese sign-off 最後まで見てくれてありがとう。またね！
Writes audio/narration.wav + .mp3 and audio/words.json (English word timings shifted into place, plus the Japanese
segments: sentence with its three chunks 私は | 日本語を | 勉強します, sign-off)."""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
JA_SENT = (3.98, 6.30)                    # the example sentence
JA_CHUNKS = [(4.02, 4.70), (4.70, 5.19), (5.19, 6.15)]   # 私は | 日本語を | 勉強します (measured from the energy envelope)
EN_CUT = 21.42                            # in take_en_a: after "Look at this sentence." (silence 21.20-21.67)
EN_START = 0.0                            # take_en_a speaks from its first 30 ms ("How")
AT_EN = 0.0                               # so the reel opens on the first word
KOHEE = (49.28, 50.06)                    # in take_en_a: "kohee" (speech 49.33-49.92), replaced by コーヒー
KOHI = ('kohi_a.mp3', 0.02, 0.72, 49.31)  # the Japanese take, its span, and where its speech starts in take_en_a
GAP_BEFORE, GAP_AFTER = 0.2, 0.35         # breaths around the spliced sentence
GAP_JA_END = 0.9                          # before the sign-off


def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(A, name), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def fade(x, ms=12):
    n = int(SR * ms / 1000)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def cut(x, a, b):
    return fade(x[int(a * SR):int(b * SR)].copy())


def main():
    en, ja, bye = load('take_en_a.mp3'), load('take_ja_a.mp3'), load('ja_signoff.mp3')
    # コーヒー in Japanese, level-matched to the English word it replaces
    k = cut(load(KOHI[0]), KOHI[1], KOHI[2])
    a, b = int(KOHEE[0] * SR), int(KOHEE[1] * SR)
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    k *= rms(en[a:b]) / rms(k)
    en[a:b] *= np.concatenate([np.linspace(1, 0, int(0.01 * SR)), np.zeros(b - a - int(0.02 * SR)), np.linspace(0, 1, int(0.01 * SR))])
    i = int((KOHI[3] - 0.03) * SR)
    en[i:i + len(k)] += k
    en_len = len(en) / SR
    at_sent = AT_EN + EN_CUT + GAP_BEFORE
    at_en2 = at_sent + (JA_SENT[1] - JA_SENT[0]) + GAP_AFTER
    shift2 = at_en2 - EN_CUT
    at_bye = shift2 + 78.96 + GAP_JA_END
    segs = [(cut(en, EN_START, EN_CUT), AT_EN + EN_START), (cut(ja, *JA_SENT), at_sent),
            (cut(en, EN_CUT, en_len), at_en2), (fade(bye.copy()), at_bye)]
    out = np.zeros(int((at_bye + len(bye) / SR + 0.2) * SR), np.float32)
    for s, at in segs:
        i = int(at * SR)
        out[i:i + len(s)] += s
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    words = []
    for w in json.load(open(os.path.join(A, 'words_en_a.json')))['words']:
        if w.get('type', 'word') != 'word':
            continue
        sh = AT_EN if w['start'] < EN_CUT else shift2
        words.append(dict(text=w['text'], type='word', start=round(w['start'] + sh, 3), end=round(w['end'] + sh, 3)))
    js = at_sent - JA_SENT[0]
    data = {'words': words,
            'ja_sentence': dict(text='私は日本語を勉強します。', start=round(4.02 + js, 3), end=round(6.15 + js, 3),
                                chunks=[[round(a + js, 3), round(b + js, 3)] for a, b in JA_CHUNKS]),
            'ja': dict(text='最後まで見てくれてありがとう。またね！', start=round(at_bye, 3), end=round(at_bye + 2.8, 3),
                       arigatou_end=round(at_bye + 1.86, 3), matane=round(at_bye + 2.3, 3))}
    json.dump(data, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False)
    print(f'narration {len(out) / SR:.2f}s; JA sentence at {data["ja_sentence"]["start"]}s; sign-off at {at_bye:.2f}s')


if __name__ == '__main__':
    main()

"""build_vo — the narration: Raj's English take with every Japanese word spoken by Raj in Japanese.

    audio/take_en_c.mp3 + words_en_c.json     the English take, romaji stand-ins in it (Scribe word timings)
    audio/words_ja_a.mp3                      こくじ。 うごく。 ひと。 はたらく。 とうげ。 はたけ。 (split on its silences)
    audio/matane_a.mp3                        またね！ (after "follow for more.")

Each stand-in is cut at the silent edge either side of it (so the pauses round it stay) and replaced by its
Japanese clip; kokuji is said twice and both use the same clip.  Writes audio/narration.wav + .mp3 and
audio/words.json: the English words shifted into place (each replaced word keeps its romaji, flagged ja) plus the
Japanese segments {key: text, romaji, start, end} (the second kokuji is 'kokuji2', the goodbye 'matane').
"""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
JA_WORDS = [('kokuji', 'こくじ', 'kokuji'), ('ugoku', 'うごく', 'ugoku'), ('hito', 'ひと', 'hito'),
            ('hataraku', 'はたらく', 'hataraku'), ('touge', 'とうげ', 'tōge'), ('hatake', 'はたけ', 'hatake')]
# stand-ins in the order they are spoken: (word in the take, key in words.json, clip)
STAND_IN = [('kokuji,', 'kokuji', 'kokuji'), ('ugoku,', 'ugoku', 'ugoku'), ('hito,', 'hito', 'hito'),
            ('hataraku,', 'hataraku', 'hataraku'), ('touge,', 'touge', 'touge'), ('hatake,', 'hatake', 'hatake'),
            ('kokuji.', 'kokuji2', 'kokuji')]
MATANE = ('matane', 'またね！', 'matane!')
HOLD = {'this.': 0.5, 'sight!': 0.4, 'move.': 0.4, 'it.': 0.6, 'work.': 0.2, 'work!': 0.7, 'pass.': 0.6,
        'field.': 0.6, 'more!': 3.0}         # silence after these words: each build lands, the eight cards are read
PAD = 0.05


def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(A, name), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def env(x):
    h = SR // 100
    return 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(h) / h, 'same')[::h]) + 1e-9)


def segments(x, thresh=-42, hold=12):
    on = env(x) > thresh
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and (on[j] or on[j:j + hold].any()):
                j += 1
            segs.append((max(0, i - 2) / 100, min(len(on), j + 3) / 100))
            i = j
        else:
            i += 1
    return segs


def faded(y, ms=10):
    y = y.copy()
    n = int(SR * ms / 1000)
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def quietest(e, t0, t1):
    a, b = int(t0 * 100), max(int(t0 * 100) + 1, int(t1 * 100))
    return (a + int(np.argmin(e[a:b]))) / 100


def silent_edge(e, t0, t1, side, thresh=-55):
    a, b = int(t0 * 100), max(int(t0 * 100) + 1, int(t1 * 100))
    q = np.where(e[a:b] < thresh)[0]
    if len(q) == 0:
        return quietest(e, t0, t1)
    return (a + (q[-1] if side == 'before' else q[0])) / 100


def main():
    en = load('take_en_c.mp3')
    e = env(en)
    words = [w for w in json.load(open(os.path.join(A, 'words_en_c.json')))['words'] if w.get('type', 'word') == 'word']
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    level = rms(en)
    jw = load('words_ja_a.mp3')
    segs = segments(jw)
    assert len(segs) == len(JA_WORDS), f'{len(segs)} segments in the Japanese words take'
    clips = {k: faded(jw[int(a * SR):int(b * SR)]) for (k, _, _), (a, b) in zip(JA_WORDS, segs)}
    for k in clips:
        clips[k] *= 0.95 * level / rms(clips[k])
    edits, i0 = [], 0
    for word, key, clip in STAND_IN:
        i = next(i for i in range(i0, len(words)) if words[i]['text'] == word)
        i0 = i + 1
        prev_end, nxt = words[i - 1]['end'], words[i + 1]['start'] if i + 1 < len(words) else words[i]['end'] + 0.3
        a = silent_edge(e, prev_end - 0.04, words[i]['start'] + 0.06, 'before')
        b = silent_edge(e, words[i]['end'] - 0.06, nxt + 0.04, 'after')
        pad = np.zeros(int(PAD * SR), np.float32)
        edits.append([a, b, np.concatenate([pad, clips[clip], pad]), key, i])
    for i, w in enumerate(words[:-1]):
        if w['text'] in HOLD:
            c = quietest(e, w['end'] + 0.02, words[i + 1]['start'] - 0.02)
            edits.append([c, c, np.zeros(int(HOLD[w['text']] * SR), np.float32), None, i])
    end = min(len(en) / SR, words[-1]['end'] + 0.15)
    mt = load('matane_a.mp3')
    ms = segments(mt, hold=40)
    mt = faded(mt[int(ms[0][0] * SR):int(ms[-1][1] * SR)])
    mt *= 0.95 * level / rms(mt)
    out, segs_out, cursor, t, shifts = [], {}, 0.0, 0.0, []
    for a, b, audio, k, i in sorted(edits, key=lambda x: (x[0], x[1])):
        out.append(en[int(cursor * SR):int(a * SR)])
        t += a - cursor
        if k is not None:
            segs_out[k] = dict(start=round(t + PAD, 3), end=round(t + len(audio) / SR - PAD, 3))
        out.append(audio)
        t += len(audio) / SR
        shifts.append((a, b, t))
        cursor = b
    out.append(faded(en[int(cursor * SR):int(end * SR)]))
    t += end - cursor
    t_mt = t + 0.45
    out += [np.zeros(int(0.45 * SR), np.float32), mt, np.zeros(int(0.2 * SR), np.float32)]
    nar = np.concatenate(out)

    def sh(x):
        base_en, base_out = 0.0, 0.0
        for a, b, t_after in shifts:
            if x < a:
                break
            base_en, base_out = b, t_after
        return base_out + (x - base_en)

    replaced = {x[4]: x[3] for x in edits if x[3] is not None}
    lex = {k: (kana, rom) for k, kana, rom in JA_WORDS}
    lex['kokuji2'] = lex['kokuji']
    ws = []
    for i, w in enumerate(words):
        if i in replaced:
            k = replaced[i]
            punct = w['text'][-1] if w['text'][-1] in '.,:!' else ''
            ws.append(dict(text=lex[k][1] + punct, type='word', ja=k, start=segs_out[k]['start'], end=segs_out[k]['end']))
        else:
            ws.append(dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)))
    t_end = round(t_mt + len(mt) / SR, 3)
    ws.append(dict(text=MATANE[2], type='word', ja=MATANE[0], start=round(t_mt, 3), end=t_end))
    ja = {k: dict(text=lex[k][0], romaji=lex[k][1], **segs_out[k]) for k in segs_out}
    ja[MATANE[0]] = dict(text=MATANE[1], romaji=MATANE[2], start=round(t_mt, 3), end=t_end)
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(nar, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    json.dump({'words': ws, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(nar) / SR:.2f}s')
    for k, v in ja.items():
        print(f'  {k:11s} {v["start"]:6.2f}-{v["end"]:6.2f}  {v["text"]}')


if __name__ == '__main__':
    main()

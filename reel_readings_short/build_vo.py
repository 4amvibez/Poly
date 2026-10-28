"""build_vo — the narration of the short cut: Raj's English take (take_short_a.mp3), every Japanese word spoken by Raj
in Japanese, and the quiz pauses.

The English take was recorded with romaji stand-ins (yama, Fujisan, san, hi, nichiyoubi, nichi, bi, mizu, sui,
suiyoubi, hito, nihonjin, onyomi, kunyomi).  Each is cut out at the silent edge on either side of it (so the
pause round it stays) and replaced by the same word from a Japanese take: words_ja_a.mp3 (a calm list of 14 words)
and, for the quiz answers and the び, words_extra_a.mp3 (び。すい！すいようび。にほんじん！).  Both checked with Scribe:
exact match; the English take too.  After "So how do you read this?" and "And this one?" a silence is opened for
the viewer to answer, and a short one after "the last one?".
Writes audio/narration.wav + .mp3 and audio/words.json (English words shifted into place, each replaced word keeping
its romaji and flagged ja, plus the Japanese segments {key: text, romaji, start, end} and the quiz pauses)."""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
CALM = [('yama', 'やま', 'yama'), ('fujisan', 'ふじさん', 'Fujisan'), ('san', 'さん', 'san'), ('hi', 'ひ', 'hi'),
        ('nichiyoubi', 'にちようび', 'nichiyōbi'), ('nichi', 'にち', 'nichi'), ('mizu', 'みず', 'mizu'),
        ('suiyoubi_c', 'すいようび', 'suiyōbi'), ('sui_c', 'すい', 'sui'), ('hito', 'ひと', 'hito'),
        ('nihonjin_c', 'にほんじん', 'nihonjin'), ('jin', 'じん', 'jin'), ('onyomi', 'おんよみ', "on'yomi"),
        ('kunyomi', 'くんよみ', "kun'yomi")]                         # words_ja_a.mp3, in order
EXTRA = [('bi', 'び', 'bi'), ('sui', 'すい', 'sui'), ('suiyoubi', 'すいようび', 'suiyōbi'), ('nihonjin', 'にほんじん', 'nihonjin')]
# (the stand-in as Scribe wrote it, which occurrence, the clip, the key it is filed under in words.json)
REPLACE = [('onyomi,', 0, 'onyomi', 'onyomi'), ('kunyomi.', 0, 'kunyomi', 'kunyomi'), ('yama.', 0, 'yama', 'yama'),
           ('Fujisan,', 0, 'fujisan', 'fujisan'), ('san.', 0, 'san', 'san'), ('hi.', 0, 'hi', 'hi'),
           ('nichiyoubi,', 0, 'nichiyoubi', 'nichiyoubi'), ('nichi...', 0, 'nichi', 'nichi'), ('bi.', 0, 'bi', 'bi'),
           ('mizu.', 0, 'mizu', 'mizu'), ('sui!', 0, 'sui', 'sui'), ('suiyoubi.', 0, 'suiyoubi', 'suiyoubi'),
           ('hito.', 0, 'hito', 'hito'), ('nihonjin!', 0, 'nihonjin', 'nihonjin'),
           ('kunyomi.', 1, 'kunyomi', 'kunyomi_tip'), ('onyomi.', 0, 'onyomi', 'onyomi_tip')]
PAUSES = [('one?', 0, 0.35, 'pause_bi'), ('this?', 0, 1.2, 'quiz_water'), ('one?', 1, 1.2, 'quiz_person')]
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
    """the cut nearest the word on its `side` ('before' / 'after') that is already silent, so the pause between it and
    its neighbour stays in the English audio; falls back to the quietest point"""
    a, b = int(t0 * 100), max(int(t0 * 100) + 1, int(t1 * 100))
    q = np.where(e[a:b] < thresh)[0]
    if len(q) == 0:
        return quietest(e, t0, t1)
    return (a + (q[-1] if side == 'before' else q[0])) / 100


def nth(words, text, n):
    return [i for i, w in enumerate(words) if w['text'] == text][n]


def main():
    en = load('take_short_a.mp3')
    e = env(en)
    words = [w for w in json.load(open(os.path.join(A, 'words_short_a.json')))['words'] if w.get('type', 'word') == 'word']
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    level = rms(en)
    clips, info = {}, {}
    for take, table in (('words_ja_a.mp3', CALM), ('words_extra_a.mp3', EXTRA)):
        x = load(take)
        segs = segments(x)
        assert len(segs) == len(table), f'{take}: {len(segs)} segments'
        for (k, kana, rom), (a, b) in zip(table, segs):
            c = faded(x[int(a * SR):int(b * SR)])
            clips[k] = c * 0.95 * level / rms(c)
            info[k] = (kana, rom)
    edits = []                                     # (cut_a, cut_b, audio, key, word index or None)
    for text, n, clip, key in REPLACE:
        i = nth(words, text, n)
        prev_end, nxt = words[i - 1]['end'], words[i + 1]['start'] if i + 1 < len(words) else words[i]['end'] + 0.3
        a = silent_edge(e, prev_end - 0.04, words[i]['start'] + 0.06, 'before')
        b = silent_edge(e, words[i]['end'] - 0.06, nxt + 0.04, 'after')
        pad = np.zeros(int(PAD * SR), np.float32)
        edits.append([a, b, np.concatenate([pad, clips[clip], pad]), key, i, clip])
    for text, n, dur, key in PAUSES:
        i = nth(words, text, n)
        a = silent_edge(e, words[i]['end'] - 0.06, words[i + 1]['start'] + 0.04, 'after')
        edits.append([a, a, np.zeros(int(dur * SR), np.float32), key, None, None])
    out, segs_out, cursor, t, shifts = [], {}, 0.0, 0.0, []
    for a, b, audio, k, i, clip in sorted(edits, key=lambda x: x[0]):
        out.append(en[int(cursor * SR):int(a * SR)])
        t += a - cursor
        if i is None:
            segs_out[k] = dict(start=round(t, 3), end=round(t + len(audio) / SR, 3))
        else:
            segs_out[k] = dict(start=round(t + PAD, 3), end=round(t + len(audio) / SR - PAD, 3))
        out.append(audio)
        t += len(audio) / SR
        shifts.append((a, b, t))
        cursor = b
    out.append(en[int(cursor * SR):])
    nar = np.concatenate(out)

    def sh(x):
        base_en, base_out = 0.0, 0.0
        for a, b, t_after in shifts:
            if x < a:
                break
            base_en, base_out = b, t_after
        return base_out + (x - base_en)

    replaced = {x[4]: (x[3], x[5]) for x in edits if x[4] is not None}
    ws = []
    for i, w in enumerate(words):
        if i in replaced:
            k, clip = replaced[i]
            rom = info[clip][1]
            punct = w['text'][-1] if w['text'][-1] in '.,!' else ''
            rom = rom[0].upper() + rom[1:] if w['text'][0].isupper() else rom
            ws.append(dict(text=rom + punct, type='word', ja=clip, start=segs_out[k]['start'], end=segs_out[k]['end']))
        else:
            ws.append(dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)))
    ja = {k: dict(text=info[clip][0], romaji=info[clip][1], **segs_out[k]) for _, _, clip, k in REPLACE}
    pauses = {k: segs_out[k] for *_, k in PAUSES}
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(nar, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    json.dump({'words': ws, 'ja': ja, 'pauses': pauses}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(nar) / SR:.2f}s')
    for k, v in ja.items():
        print(f'  {k:12s} {v["start"]:6.2f}-{v["end"]:6.2f}  {v["text"]}')
    print('  pauses', pauses)


if __name__ == '__main__':
    main()

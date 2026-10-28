"""build_vo — assemble the narration from Raj's English take A, with every Japanese word spoken by Raj in Japanese.

The English take was recorded with romaji stand-ins (yama, Fujisan, san, hi, nichiyoubi, ...), so its rhythm has a
natural slot for each word.  Each stand-in is cut out at the quietest point either side of it and replaced by the
same word from the Japanese take (words_ja_a.mp3: a calm list of the 14 words, split on its silences).  The
Japanese line 「山は『やま』とも読みます。でも、富士山は『ふじさん』と読みます。」 is set into the breath after
"...we say san."  Both Japanese takes were checked with Scribe (exact match) before use.
Writes audio/narration.wav + .mp3 and audio/words.json: the English words shifted into place (each replaced word
keeps its romaji, flagged ja) plus the Japanese segments {key: text, romaji, start, end}."""
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
# the 14 words of words_ja_a.mp3, in order, as (key, kana, romaji); their speech spans are found on the silences
JA_WORDS = [('yama', 'やま', 'yama'), ('fujisan', 'ふじさん', 'Fujisan'), ('san', 'さん', 'san'), ('hi', 'ひ', 'hi'),
            ('nichiyoubi', 'にちようび', 'nichiyōbi'), ('nichi', 'にち', 'nichi'), ('mizu', 'みず', 'mizu'),
            ('suiyoubi', 'すいようび', 'suiyōbi'), ('sui', 'すい', 'sui'), ('hito', 'ひと', 'hito'),
            ('nihonjin', 'にほんじん', 'nihonjin'), ('jin', 'じん', 'jin'), ('onyomi', 'おんよみ', "on'yomi"),
            ('kunyomi', 'くんよみ', "kun'yomi")]
# the English stand-in each one replaces (as Scribe wrote it)
STAND_IN = {'yama': 'yama.', 'fujisan': 'Fujisan,', 'san': 'san.', 'hi': 'hi.', 'nichiyoubi': 'nichiyoubi,', 'nichi': 'nichi.',
            'mizu': 'mizu.', 'suiyoubi': 'suiyoubi,', 'sui': 'sui.', 'hito': 'hito.', 'nihonjin': 'nihonjin,', 'jin': 'jin.',
            'onyomi': 'Onyomi', 'kunyomi': 'kunyomi'}
LINE = ('yama_line', 'yama_line_a.mp3', '山は「やま」とも読みます。でも、富士山は「ふじさん」と読みます。',
        'Yama wa "yama" tomo yomimasu. Demo, Fujisan wa "Fujisan" to yomimasu.')
LINE_AFTER = 'san'                     # the line follows the word that replaced "san."
PAD = 0.05                             # air either side of a spliced word
LINE_GAP = (0.45, 0.6)                 # breaths before / after the Japanese line


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


def main():
    en = load('take_en_a.mp3')
    e = env(en)
    words = [w for w in json.load(open(os.path.join(A, 'words_en_a.json')))['words'] if w.get('type', 'word') == 'word']
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    level = rms(en)
    jw = load('words_ja_a.mp3')
    segs = segments(jw)
    assert len(segs) == len(JA_WORDS), f'{len(segs)} segments in the Japanese words take'
    clips = {k: faded(jw[int(a * SR):int(b * SR)]) for (k, _, _), (a, b) in zip(JA_WORDS, segs)}
    for k in clips:
        clips[k] *= 0.95 * level / rms(clips[k])
    line = load(LINE[1])
    ls = segments(line, hold=40)
    line = faded(line[int(ls[0][0] * SR):int(ls[-1][1] * SR)])
    line *= 0.95 * level / rms(line)
    # where each stand-in is cut: the quietest 10 ms on either side, inside the pauses round it
    edits = []                                     # (cut_a, cut_b, audio, key)
    used = set()
    for k, kana, rom in JA_WORDS:
        i = next(i for i, w in enumerate(words) if w['text'] == STAND_IN[k] and i not in used)
        used.add(i)
        prev_end, nxt = words[i - 1]['end'], words[i + 1]['start'] if i + 1 < len(words) else words[i]['end'] + 0.3
        a = quietest(e, prev_end - 0.04, words[i]['start'] + 0.06)
        b = quietest(e, words[i]['end'] - 0.06, nxt + 0.04)
        pad = np.zeros(int(PAD * SR), np.float32)
        edits.append([a, b, np.concatenate([pad, clips[k], pad]), k, i])
    # the Japanese line goes into the pause after "san."
    k_san = next(x for x in edits if x[3] == LINE_AFTER)
    t_line = k_san[1]
    out, segs_out, cursor, t = [], {}, 0.0, 0.0
    shifts = []                                    # (en_time, delta) for mapping English words
    for a, b, audio, k, i in sorted(edits):
        out.append(en[int(cursor * SR):int(a * SR)])
        t += a - cursor
        segs_out[k] = dict(start=round(t + PAD, 3), end=round(t + len(audio) / SR - PAD, 3))
        out.append(audio)
        t += len(audio) / SR
        shifts.append((a, b, t))
        cursor = b
        if k == LINE_AFTER:
            g0, g1 = np.zeros(int(LINE_GAP[0] * SR), np.float32), np.zeros(int(LINE_GAP[1] * SR), np.float32)
            segs_out[LINE[0]] = dict(start=round(t + LINE_GAP[0], 3), end=round(t + LINE_GAP[0] + len(line) / SR, 3))
            out += [g0, line, g1]
            t += (len(g0) + len(line) + len(g1)) / SR
            shifts[-1] = (a, b, t)
    out.append(en[int(cursor * SR):])
    nar = np.concatenate(out)

    def sh(x):
        """English take time -> narration time"""
        base_en, base_out = 0.0, 0.0
        for a, b, t_after in shifts:
            if x < a:
                break
            base_en, base_out = b, t_after
        return base_out + (x - base_en)

    replaced = {x[4]: x[3] for x in edits}
    ws = []
    for i, w in enumerate(words):
        if i in replaced:
            k = replaced[i]
            rom = next(r for kk, _, r in JA_WORDS if kk == k)
            punct = w['text'][-1] if w['text'][-1] in '.,' else ''
            rom = rom[0].upper() + rom[1:] if w['text'][0].isupper() else rom
            ws.append(dict(text=rom + punct, type='word', ja=k, start=segs_out[k]['start'], end=segs_out[k]['end']))
        else:
            ws.append(dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)))
    ja = {k: dict(text=kana, romaji=rom, **segs_out[k]) for k, kana, rom in JA_WORDS}
    ja[LINE[0]] = dict(text=LINE[2], romaji=LINE[3], **segs_out[LINE[0]])
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(nar, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    json.dump({'words': ws, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(nar) / SR:.2f}s')
    for k, v in ja.items():
        print(f'  {k:11s} {v["start"]:6.2f}-{v["end"]:6.2f}  {v["text"]}')


if __name__ == '__main__':
    main()

"""build_vo — assemble the narration from Raj's English take A, with every Japanese word spoken by Raj in Japanese.

The 水 section of take A ("Now, this kanji means water ... the word decides the reading.") is dropped, and after 人
take_mid2_a.mp3 is inserted (Raj, Scribe: exact): にん in さんにん, then the names on'yomi / kun'yomi, and that one
kanji can have more than one of each.  Its Japanese words come from the same list take and from words_ja_n_a.mp3
(さんにん にん, Scribe: exact).

The English take was recorded with romaji stand-ins (yama, Fujisan, san, hi, nichiyoubi, ...), so its rhythm has a
natural slot for each word.  Each stand-in is cut out at the quietest point either side of it and replaced by the
same word from the Japanese take (words_ja_a.mp3: a calm list of the 14 words, split on its silences, checked with
Scribe: exact match).  The take's sign-off is replaced from "Every kanji has a story." on by signoff_a.mp3
("Every kanji has a story. Follow me for more.", Scribe: exact match).
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
JA_N = [('sannin', 'さんにん', 'sannin'), ('nin', 'にん', 'nin')]
# every stand-in in narration order: (key in words.json, clip, the English word(s) it replaces)
SEQ = [('yama', 'yama', 'yama.'), ('fujisan', 'fujisan', 'Fujisan,'), ('san', 'san', 'san.'), ('hi', 'hi', 'hi.'),
       ('nichiyoubi', 'nichiyoubi', 'nichiyoubi,'), ('nichi', 'nichi', 'nichi.'),
       ('hito', 'hito', 'hito.'), ('nihonjin', 'nihonjin', 'nihonjin,'), ('jin', 'jin', 'jin.'),
       ('sannin', 'sannin', 'sannin,'), ('nin', 'nin', 'nin.'),
       ('san_n', 'san', 'san,'), ('nichi_n', 'nichi', 'nichi'), ('jin_n', 'jin', 'jin,'), ('onyomi_n', 'onyomi', 'onyomi.'),
       ('yama_n', 'yama', 'yama,'), ('hi_n', 'hi', 'hi'), ('hito_n', 'hito', 'hito,'), ('kunyomi_n', 'kunyomi', 'kunyomi.'),
       ('onyomi', 'onyomi', 'Onyomi'), ('kunyomi', 'kunyomi', 'kunyomi')]
SIGNOFF = 'signoff_a.mp3'                # "Every kanji has a story. Follow me for more." (Scribe: exact)
PAD = 0.05                             # air either side of a spliced word


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


def english():
    """take A without its 水 section, with the mid take after 人: (audio, words)"""
    en = load('take_en_a.mp3')
    e = env(en)
    words = [w for w in json.load(open(os.path.join(A, 'words_en_a.json')))['words'] if w.get('type', 'word') == 'word']
    idx = lambda txt, after=0: next(i for i, w in enumerate(words) if w['text'] == txt and i > after)
    i_now, i_fin = idx('Now,'), idx('Finally,')
    i_so = idx('So', i_fin)
    cut = lambda i: quietest(e, words[i - 1]['end'] + 0.03, words[i]['start'] - 0.02)
    c_now, c_fin, c_so = cut(i_now), cut(i_fin), cut(i_so)
    mid = load('take_mid2_a.mp3')
    mw = [w for w in json.load(open(os.path.join(A, 'words_mid2_a.json')))['words'] if w.get('type', 'word') == 'word']
    m0, m1 = max(mw[0]['start'] - 0.08, 0), mw[-1]['end'] + 0.12
    gap = np.zeros(int(0.2 * SR), np.float32)
    parts, out_words, t = [], [], 0.0

    def add(audio, ws, src0):
        nonlocal t
        parts.append(audio)
        out_words.extend(dict(w, start=round(t + w['start'] - src0, 3), end=round(t + w['end'] - src0, 3)) for w in ws)
        t += len(audio) / SR

    add(en[:int(c_now * SR)], words[:i_now], 0.0)                      # ... 山, 日
    add(en[int(c_fin * SR):int(c_so * SR)], words[i_fin:i_so], c_fin)  # 人: ひと, じん
    add(gap, [], 0.0)
    add(faded(mid[int(m0 * SR):int(m1 * SR)]), mw, m0)                 # にん, then on'yomi / kun'yomi
    add(gap, [], 0.0)
    add(en[int(c_so * SR):], words[i_so:], c_so)                       # the takeaway
    return np.concatenate(parts), out_words


def main():
    en, words = english()
    e = env(en)
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    level = rms(en)
    clips = {}
    for name, keys in (('words_ja_a.mp3', JA_WORDS), ('words_ja_n_a.mp3', JA_N)):
        jw = load(name)
        segs = segments(jw)
        assert len(segs) == len(keys), f'{len(segs)} segments in {name}'
        for (k, _, _), (a, b) in zip(keys, segs):
            clips[k] = faded(jw[int(a * SR):int(b * SR)])
            clips[k] *= 0.95 * level / rms(clips[k])
    LEX = {k: (kana, rom) for k, kana, rom in JA_WORDS + JA_N}
    # where each stand-in is cut: the silent edge on either side, inside the pauses round it
    edits = []                                     # (cut_a, cut_b, audio, key, first word, last word)
    i0 = 0
    for key, clip, sin in SEQ:
        sin = (sin,) if isinstance(sin, str) else sin
        i = next(i for i in range(i0, len(words)) if all(words[i + k]['text'] == x for k, x in enumerate(sin)))
        j = i + len(sin) - 1
        i0 = j + 1
        prev_end, nxt = words[i - 1]['end'], words[j + 1]['start'] if j + 1 < len(words) else words[j]['end'] + 0.3
        a = silent_edge(e, prev_end - 0.04, words[i]['start'] + 0.06, 'before')
        b = silent_edge(e, words[j]['end'] - 0.06, nxt + 0.04, 'after')
        pad = np.zeros(int(PAD * SR), np.float32)
        edits.append([a, b, np.concatenate([pad, clips[clip], pad]), key, i, j])
    # the sign-off: the take is replaced from "Every kanji has a story." on by SIGNOFF ("... Follow me for more.")
    i_every = next(i for i, w in enumerate(words) if w['text'] == 'Every')
    cut_end = quietest(e, words[i_every - 1]['end'] - 0.04, words[i_every]['start'] + 0.04)
    so = load(SIGNOFF)
    ss = segments(so, hold=40)
    so0 = ss[0][0]
    so = faded(so[int(so0 * SR):int(ss[-1][1] * SR)])
    so *= level / rms(so)
    out, segs_out, cursor, t = [], {}, 0.0, 0.0
    shifts = []                                    # (en_time, delta) for mapping English words
    for a, b, audio, k, i, j in sorted(edits):
        out.append(en[int(cursor * SR):int(a * SR)])
        t += a - cursor
        segs_out[k] = dict(start=round(t + PAD, 3), end=round(t + len(audio) / SR - PAD, 3))
        out.append(audio)
        t += len(audio) / SR
        shifts.append((a, b, t))
        cursor = b
    out.append(en[int(cursor * SR):int(cut_end * SR)])
    t += cut_end - cursor
    t_so = t + 0.05
    out += [np.zeros(int(0.05 * SR), np.float32), so]
    nar = np.concatenate(out)

    def sh(x):
        """English take time -> narration time"""
        base_en, base_out = 0.0, 0.0
        for a, b, t_after in shifts:
            if x < a:
                break
            base_en, base_out = b, t_after
        return base_out + (x - base_en)

    replaced = {x[4]: (x[3], x[5]) for x in edits}
    clip_of = {key: clip for key, clip, _ in SEQ}
    ws = []
    skip = -1
    for i, w in enumerate(words[:i_every]):
        if i <= skip:
            continue
        if i in replaced:
            k, j = replaced[i]
            skip = j
            rom = LEX[clip_of[k]][1]
            last = words[j]['text']
            punct = last[-1] if last[-1] in '.,:' else ''
            rom = rom[0].upper() + rom[1:] if w['text'][0].isupper() else rom
            ws.append(dict(text=rom + punct, type='word', ja=k, start=segs_out[k]['start'], end=segs_out[k]['end']))
        else:
            ws.append(dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)))
    so_words = [w for w in json.load(open(os.path.join(A, 'words_signoff_a.json')))['words'] if w.get('type', 'word') == 'word']
    ws += [dict(text=w['text'], type='word', start=round(t_so + w['start'] - so0, 3), end=round(t_so + w['end'] - so0, 3)) for w in so_words]
    ja = {k: dict(text=LEX[clip_of[k]][0], romaji=LEX[clip_of[k]][1], **segs_out[k]) for k in segs_out}
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(nar, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    json.dump({'words': ws, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(nar) / SR:.2f}s')
    for k, v in ja.items():
        print(f'  {k:11s} {v["start"]:6.2f}-{v["end"]:6.2f}  {v["text"]}')


if __name__ == '__main__':
    main()

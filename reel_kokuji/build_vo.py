"""build_vo — the narration: Raj's English take with every Japanese word spoken by Raj in Japanese.

    audio/take_en_a.mp3 + words_en_a.json     the English take, romaji stand-ins in it (Scribe word timings)
    audio/words_ja_a.mp3                      こくじ。 うごく。 ひと。 はたらく。 とうげ。 はたけ。 (split on its silences)
    audio/line_ja_a.mp3                       一歩ずつ、夢に近づこう。 (after "being written!")
    audio/final_en_a.mp3 + words_final_a.json "Thanks for watching till the end! Follow me for more."

Each stand-in is cut at the silent edge either side of it (so the pauses round it stay) and replaced by its
Japanese clip.  Writes audio/narration.wav + .mp3 and audio/words.json: the English words shifted into place
(each replaced word keeps its romaji, flagged ja) plus the Japanese segments {key: text, romaji, start, end}.

    python build_vo.py --draft   # no audio yet: timings estimated from script.md's text, silent narration
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SR = 48000
JA_WORDS = [('kokuji', 'こくじ', 'kokuji'), ('ugoku', 'うごく', 'ugoku'), ('hito', 'ひと', 'hito'),
            ('hataraku', 'はたらく', 'hataraku'), ('touge', 'とうげ', 'tōge'), ('hatake', 'はたけ', 'hatake')]
STAND_IN = {'kokuji': 'kokuji,', 'ugoku': 'ugoku.', 'hito': 'hito,', 'hataraku': 'hataraku,', 'touge': 'touge,',
            'hatake': 'hatake,'}
LINE = ('dream_line', '一歩ずつ、夢に近づこう。', 'Ippo zutsu, yume ni chikazukō.')
AFTER_LINE = 'written!'                 # the Japanese line follows this word
HOLD = {'more!': 2.6}                   # silence after these words: the eight kokuji cards are read in it
PAD = 0.05

EN_TEXT = ("Not every kanji came from China. Japan created some of its own! "
           "They're called kokuji, Japanese-created kanji. And some of them are hiding in plain sight! "
           "First, look at this character: ugoku. It means to move. Now, add hito, a person. Combine them, "
           "and you get hataraku, meaning to work! "
           "Next: mountain, up, and down. Together they make touge, a mountain pass, a path through the mountains. "
           "And fire, plus rice field, makes hatake, a cultivated field. "
           "Some compiled lists contain over two hundred of these characters. Here are a few more! "
           "Every kanji has a story. Every mistake helps you grow. Keep learning, one step at a time. "
           "Your Japanese journey is still being written!")
FINAL_TEXT = "Thanks for watching till the end! Follow me for more."


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


def draft():
    """estimated timings (2.6 words/s, breaths at punctuation) and a silent narration, to build the scenes on"""
    t, ws, ja = 0.0, [], {}
    lex = {k: (kana, rom) for k, kana, rom in JA_WORDS}
    for text in (EN_TEXT, None, FINAL_TEXT):
        if text is None:
            ja[LINE[0]] = dict(text=LINE[1], romaji=LINE[2], start=round(t + 0.3, 3), end=round(t + 3.1, 3))
            t += 3.6
            continue
        for w in text.split():
            key = next((k for k, s in STAND_IN.items() if s.rstrip(',.:!') == re.sub(r'[,.:!]', '', w)), None)
            d = 0.62 if key else max(0.16, 0.055 * len(w) + 0.1)
            if key:
                ja[key] = dict(text=lex[key][0], romaji=lex[key][1], start=round(t, 3), end=round(t + d, 3))
            ws.append(dict(text=(lex[key][1] + w[-1] if key and w[-1] in ',.:!' else lex[key][1]) if key else w, type='word',
                           start=round(t, 3), end=round(t + d, 3), **({'ja': key} if key else {})))
            t += d + 0.06 + (0.3 if w[-1] in ',:' else 0.55 if w[-1] in '.!?' else 0) + HOLD.get(w, 0)
    n = int((t + 1) * SR)
    wavfile.write(os.path.join(A, 'narration.wav'), SR, np.zeros(n, np.int16))
    json.dump({'words': ws, 'ja': ja, 'draft': True}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'draft narration {t:.2f}s')


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
    edits, i0 = [], 0
    for k, kana, rom in JA_WORDS:
        i = next(i for i in range(i0, len(words)) if words[i]['text'] == STAND_IN[k])
        i0 = i + 1
        prev_end, nxt = words[i - 1]['end'], words[i + 1]['start'] if i + 1 < len(words) else words[i]['end'] + 0.3
        a = silent_edge(e, prev_end - 0.04, words[i]['start'] + 0.06, 'before')
        b = silent_edge(e, words[i]['end'] - 0.06, nxt + 0.04, 'after')
        pad = np.zeros(int(PAD * SR), np.float32)
        edits.append([a, b, np.concatenate([pad, clips[k], pad]), k, i])
    # the Japanese line goes in after "being written!", then the final English take
    i_line = next(i for i, w in enumerate(words) if w['text'] == AFTER_LINE)
    cut_line = words[i_line]['end'] + 0.25
    line = load('line_ja_a.mp3')
    ls = segments(line, hold=40)
    line = faded(line[int(ls[0][0] * SR):int(ls[-1][1] * SR)])
    line *= 0.95 * level / rms(line)
    fin = load('final_en_a.mp3')
    fs = segments(fin, hold=40)
    f0 = fs[0][0]
    fin = faded(fin[int(f0 * SR):int(fs[-1][1] * SR)])
    fin *= level / rms(fin)
    out, segs_out, cursor, t, shifts = [], {}, 0.0, 0.0, []
    for a, b, audio, k, i in sorted(edits):
        out.append(en[int(cursor * SR):int(a * SR)])
        t += a - cursor
        segs_out[k] = dict(start=round(t + PAD, 3), end=round(t + len(audio) / SR - PAD, 3))
        out.append(audio)
        t += len(audio) / SR
        shifts.append((a, b, t))
        cursor = b
    out.append(en[int(cursor * SR):int(cut_line * SR)])
    t += cut_line - cursor
    gap = np.zeros(int(0.35 * SR), np.float32)
    t_line = t + 0.35
    out += [gap, line, gap]
    t = t_line + len(line) / SR + 0.35
    t_fin = t + 0.3
    out += [np.zeros(int(0.3 * SR), np.float32), fin]
    nar = np.concatenate(out)

    def sh(x):
        base_en, base_out = 0.0, 0.0
        for a, b, t_after in shifts:
            if x < a:
                break
            base_en, base_out = b, t_after
        return base_out + (x - base_en)

    replaced = {x[4]: x[3] for x in edits}
    lex = {k: (kana, rom) for k, kana, rom in JA_WORDS}
    ws = []
    for i, w in enumerate(words[:i_line + 1]):
        if i in replaced:
            k = replaced[i]
            punct = w['text'][-1] if w['text'][-1] in '.,:!' else ''
            ws.append(dict(text=lex[k][1] + punct, type='word', ja=k, start=segs_out[k]['start'], end=segs_out[k]['end']))
        else:
            ws.append(dict(text=w['text'], type='word', start=round(sh(w['start']), 3), end=round(sh(w['end']), 3)))
    fw = [w for w in json.load(open(os.path.join(A, 'words_final_a.json')))['words'] if w.get('type', 'word') == 'word']
    ws += [dict(text=w['text'], type='word', start=round(t_fin + w['start'] - f0, 3), end=round(t_fin + w['end'] - f0, 3)) for w in fw]
    ja = {k: dict(text=lex[k][0], romaji=lex[k][1], **segs_out[k]) for k in segs_out}
    ja[LINE[0]] = dict(text=LINE[1], romaji=LINE[2], start=round(t_line, 3), end=round(t_line + len(line) / SR, 3))
    wavfile.write(os.path.join(A, 'narration.wav'), SR, (np.clip(nar, -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', os.path.join(A, 'narration.wav'), '-b:a', '192k',
                    os.path.join(A, 'narration.mp3')], check=True)
    json.dump({'words': ws, 'ja': ja}, open(os.path.join(A, 'words.json'), 'w'), ensure_ascii=False, indent=0)
    print(f'narration {len(nar) / SR:.2f}s')
    for k, v in ja.items():
        print(f'  {k:11s} {v["start"]:6.2f}-{v["end"]:6.2f}  {v["text"]}')


if __name__ == '__main__':
    draft() if '--draft' in sys.argv else main()

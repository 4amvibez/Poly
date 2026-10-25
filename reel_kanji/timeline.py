"""timeline — scene boundaries and transitions, all derived from the narration's word timestamps."""
import json
import os
import re

from engine import HERE

VO_OFFSET = 0.4          # narration starts just after the first brush contact

_W = json.load(open(os.path.join(HERE, 'audio', 'words.json'), encoding='utf8'))
_words = _W['words']
JA = {k: (v + VO_OFFSET if isinstance(v, (int, float)) else v) for k, v in _W['ja'].items()}   # Japanese sign-off
WORDS = [dict(text=w['text'], start=w['start'] + VO_OFFSET, end=w['end'] + VO_OFFSET)
         for w in _words if w.get('type', 'word') == 'word']


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def at(phrase, nth=0):
    """(start, end) in video seconds of a phrase in the narration, e.g. at('Look at this')"""
    target = [_norm(p) for p in phrase.split()]
    hits = 0
    for i in range(len(WORDS) - len(target) + 1):
        if all(_norm(WORDS[i + k]['text']) == target[k] for k in range(len(target))):
            if hits == nth:
                return WORDS[i]['start'], WORDS[i + len(target) - 1]['end']
            hits += 1
    raise KeyError(phrase)


# Scene boundaries come from the narration itself (single source of truth): each scene begins a little before
# the first word of its section, and the outgoing scene overlaps it for the transition.
_CUTS = [  # (name, first words of the section, lead before the words, transition length into this scene)
    ('hook', None, 0.0, 0.0),
    ('tablets', 'Thousands of', 0.15, 0.8),
    ('oracle', 'In ancient', 0.3, 0.9),
    ('objects', 'The sun', 0.2, 0.7),
    ('sun', 'Look at this', 0.35, 0.8),
    ('swirl', 'And the idea', 0.12, 0.6),
    ('nature', 'Yama means', 0.3, 0.9),
    ('compound', 'But heres', 0.15, 0.8),
    ('map', 'Over time', 0.25, 0.8),
    ('scholar', 'These characters became', 0.3, 0.8),
    ('message', 'So the next', 0.35, 0.9),
    ('endcard', 'Follow Raj', 0.1, 0.8),
]
DURATION = round(JA['end'] + 1.6, 2)              # hold the end card ~1.6 s after the Japanese sign-off
_starts = [0.0 if w is None else at(w)[0] - lead for _, w, lead, _ in _CUTS]
SCENES = []
for i, (name, _, _, _) in enumerate(_CUTS):
    end = DURATION if i == len(_CUTS) - 1 else _starts[i + 1] + _CUTS[i + 1][3]
    SCENES.append((name, round(_starts[i], 3), round(end, 3)))


def quiet_gap(t0, t1, thresh_db=-45.0):
    """centre (video seconds) of the longest silence in the narration between t0 and t1 — measured from the audio,
    because word timestamps can be a few hundred ms off at the edges of a breath"""
    import numpy as np
    from scipy.io import wavfile
    sr, x = wavfile.read(os.path.join(HERE, 'audio', 'narration.wav'))
    x = x.astype(np.float32) / 32768
    hop = sr // 100
    a, b = int((t0 - VO_OFFSET) * sr), int((t1 - VO_OFFSET) * sr)
    seg = x[a:b]
    e = 20 * np.log10(np.sqrt(np.convolve(seg ** 2, np.ones(hop) / hop, 'same')[::hop]) + 1e-9)
    q = e < thresh_db
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(q):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    s0, n = best
    return t0 + (s0 + n / 2) / 100, n / 100

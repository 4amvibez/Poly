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
    """(start, end) in video seconds of a phrase in the narration, e.g. at('Japanese does')"""
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
    ('before_writing', 'Long before', 0.64, 1.0),
    ('map', 'Writing came', 0.5, 1.0),
    ('kanji', 'These characters', 0.38, 1.0),
    ('hiragana', 'Writing sounds', 0.32, 0.8),
    ('katakana', 'Meanwhile', 0.42, 0.8),
    ('triptych', 'Today all three', 0.36, 0.9),
    ('sentence', 'Take this', 0.4, 0.8),
    ('street', 'Every line', 0.34, 1.0),
    ('modern', 'And you can', 0.35, 1.0),
    ('endcard', 'Follow for more', 0.3, 0.7),
]
DURATION = round(JA['end'] + 1.6, 2)              # hold the end card ~1.6 s after the Japanese sign-off
_starts = [0.0 if w is None else at(w)[0] - lead for _, w, lead, _ in _CUTS]
SCENES = []
for i, (name, _, _, _) in enumerate(_CUTS):
    end = DURATION if i == len(_CUTS) - 1 else _starts[i + 1] + _CUTS[i + 1][3]
    SCENES.append((name, round(_starts[i], 3), round(end, 3)))

"""Timing for the whole film, derived from the measured voiceover durations (assets/voice/lines.json).

Lines in the same scene are separated by a short breath, scene changes by a longer pause, and a few lines get
an extra beat before a visual reveal. Each scene starts slightly before its first line, so the picture leads
the voice. Everything else (visual beats, captions, music cues) hangs off these times.
"""
import json
import os

from .ink import text_width, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.path.join(HERE, 'assets', 'voice')

LEAD_IN = 1.0           # title brushwork before the first word
BREATH = 0.3            # between lines of one scene
SCENE_GAP = 0.55        # between scenes
EXTRA = {1: 1.5, 3: 0.2, 9: 0.25, 14: 0.2, 18: 0.6}     # an extra beat before these lines (a reveal lands first)
TAIL = 2.5              # final hold after the last word
LEAD = 0.3              # a scene begins this long before its first line
TRANSITION = {2: 1.2, 3: 1.4, 4: 1.2, 5: 1.0, 6: 0.9, 7: 1.0, 8: 1.1, 9: 1.4, 10: 1.3, 11: 1.3}


class Timeline:
    def __init__(self):
        meta = json.load(open(os.path.join(VOICE, 'lines.json'), encoding='utf-8'))
        self.lines = meta['lines']
        t, prev_scene = LEAD_IN, None
        for ln in self.lines:
            if prev_scene is not None:
                t += SCENE_GAP if ln['scene'] != prev_scene else BREATH
            t += EXTRA.get(ln['i'], 0)
            ln['start'], ln['end'] = t, t + ln['dur']
            t = ln['end']
            prev_scene = ln['scene']
        self.total = round((t + TAIL) * 30) / 30
        self.scenes = {}
        nums = sorted({ln['scene'] for ln in self.lines})
        for n in nums:
            first = min(ln['start'] for ln in self.lines if ln['scene'] == n)
            self.scenes[n] = {'start': 0.0 if n == nums[0] else first - LEAD}
        for a, b in zip(nums, nums[1:]):
            self.scenes[a]['end'] = self.scenes[b]['start']
        self.scenes[nums[-1]]['end'] = self.total
        for n in nums:
            s = self.scenes[n]
            s['dur'] = s['end'] - s['start']
            s['trans'] = TRANSITION.get(n, 0.0)
            # line times local to the scene
            s['lines'] = [(ln['start'] - s['start'], ln['end'] - s['start']) for ln in self.lines if ln['scene'] == n]

    def word_time(self, i, word, scene=None, after=0):
        """estimated time a word is spoken in line i (proportional to its position in the caption); local to
        `scene` if given"""
        ln = self.lines[i]
        cap = ln['caption']
        k = cap.find(word, after)
        f = (k + len(word) * 0.5) / len(cap) if k >= 0 else 0.5
        t = ln['start'] + ln['dur'] * f
        return t - self.scenes[scene]['start'] if scene else t

    def captions(self, size=50, max_w=900, max_chars=76):
        """caption chunks [(t0, t1, [line, line])]: each spoken line split at sentence / clause breaks into
        chunks of at most two lines, timed by their share of the characters"""
        out = []
        for ln in self.lines:
            cap = ln['caption'].replace(' 。', '。')
            parts = _split(cap, max_chars)
            total = sum(len(p) for p in parts)
            t = ln['start']
            for p in parts:
                d = ln['dur'] * len(p) / total
                out.append([t, t + d, wrap(p, size, max_w, balance=True)])
                t += d
        # hold each caption through the short breaths, so the band doesn't flicker between lines
        for a, b in zip(out, out[1:]):
            if b[0] - a[1] < 0.5:
                a[1] = b[0]
            else:
                a[1] += 0.25
        out[-1][1] += 0.6
        return out


def _split(text, max_chars):
    """split at sentence ends first, then at the clause break nearest the middle, until every chunk fits"""
    import re
    if len(text) <= max_chars:
        return [text]
    cuts = [m.end() for m in re.finditer(r'[.:;!?。]\s+', text)] or [m.end() for m in re.finditer(r'[,]\s+', text)]
    cuts = [c for c in cuts if 12 <= c <= len(text) - 12]
    if not cuts:
        sp = [m.start() for m in re.finditer(' ', text)]
        cuts = [min(sp, key=lambda c: abs(c - len(text) / 2))]
    c = min(cuts, key=lambda c: abs(c - len(text) / 2))
    return _split(text[:c].strip(), max_chars) + _split(text[c:].strip(), max_chars)


if __name__ == '__main__':
    tl = Timeline()
    print('total', tl.total)
    for n, s in tl.scenes.items():
        print(n, round(s['start'], 2), round(s['end'], 2), round(s['dur'], 2), [tuple(round(v, 2) for v in x) for x in s['lines']])
    for c in tl.captions():
        print(round(c[0], 2), round(c[1], 2), c[2])

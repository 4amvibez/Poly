"""Write score.js: the timeline (scenes, sync markers, captions) in the animation-studio format, so the skill's
tools (levels.js and friends) read the same single source of truth as the Python renderer."""
import json
import os

import numpy as np
from scipy.io import wavfile

import subtitles
from timeline import DURATION, SCENES, at

HERE = os.path.dirname(os.path.abspath(__file__))


def markers():
    import scenes
    from timeline import JA
    s1, s4, s5, s6, s10 = (scenes.get(n) for n in ('hook', 'kanji', 'hiragana', 'katakana', 'modern'))
    return {
        'why': {'t': round(s1.t_why + 0.02, 3), 'sync': 'av'},              # the ？ dot lands + taiko
        'kanji_done': {'t': round(s4.t_write_end + 0.05, 3), 'sync': 'av'},  # 字 completes + bowl
        'hiragana_a': {'t': round(s5.tD + 1.5, 3), 'sync': 'av'},            # あ completes + bowl
        'katakana_i': {'t': round(s6.t_kana + 0.75, 3), 'sync': 'av'},       # イ completes + bowl
        'rings': {'t': round(s10.labels[0]['t'], 3), 'sync': 'av'},         # first sign ringed + taiko
        'matane': {'t': round(JA['matane'], 3), 'sync': 'av'},              # またね！ stamped + bowl
    }


def _write_f32(path, sr, x):
    """plain 44-byte-header IEEE-float stereo WAV (what levels.js expects)"""
    import struct
    data = np.ascontiguousarray(x, '<f4').tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 3, 2, sr, sr * 8, 8, 32)
    open(path, 'wb').write(hdr + b'data' + struct.pack('<I', len(data)) + data)


def main():
    S = {n: [a, b] for n, a, b in SCENES}
    subs = [{'start': round(c['start'], 3), 'end': round(c['end'], 3), 'text': ' '.join(w for w, _ in c['words'])}
            for c in subtitles.CUES]
    data = {'FPS': 30, 'FORMAT': '9:16', 'DURATION': DURATION, 'S': S, 'markers': markers(), 'subtitles': subs}
    js = ('(function (root, f) { if (typeof module === "object") module.exports = f(); else root.SCORE = f(); })'
          '(this, function () {\n  return ' + json.dumps(data, ensure_ascii=False, indent=1) + ';\n});\n')
    open(os.path.join(HERE, 'score.js'), 'w', encoding='utf8').write(js)
    # float32 stereo copies of the stems for levels.js
    out = os.path.join(HERE, 'audio', 'stems_f32')
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(HERE, 'audio', 'stems'))) + ['../mix_raw.wav']:
        sr, x = wavfile.read(os.path.join(HERE, 'audio', 'stems', f))
        x = (x.astype(np.float32) / 32768.0).reshape(len(x), -1)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, 1)
        _write_f32(os.path.join(out, os.path.basename(f)), sr, x)
    print('wrote score.js;', len(subs), 'captions;', json.dumps(data['markers']))


if __name__ == '__main__':
    main()

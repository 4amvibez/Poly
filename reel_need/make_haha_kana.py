"""make_haha_kana — the kana-only line ははははながすき built from Raj's own recordings: one は (ha_user.mp3) said four
times, then ながすき (nagasuki_user.mp3), at the pace of someone reading it out.  (ElevenLabs turns two or more は in a
row into laughter, so this line is not generated.)  The は are spaced a little unevenly and each a touch softer than
the one before, as a person says them, with a short breath before ながすき.  Writes audio/haha_kana.wav."""
import os
import sys

import numpy as np
from scipy.io import wavfile

from build_vo import A, SR, load, speech

HA_CUT = (0.03, 0.40)                 # the syllable inside ha_user.mp3 (s), with its natural tail
STEPS = [0.40, 0.38, 0.41]            # from one は to the next
GAIN = [1.0, 0.95, 0.93, 0.9]
GAP = 0.10                            # the breath before ながすき


def main(pace=1.0, out_name='haha_kana.wav'):
    ha, nag = speech(load('ha_user.mp3')), speech(load('nagasuki_user.mp3'))
    ha = ha[int(HA_CUT[0] * SR):int(HA_CUT[1] * SR)].copy()
    n = SR // 100
    ha[:n] *= np.linspace(0, 1, n)
    ha[-10 * n:] *= np.linspace(1, 0, 10 * n) ** 1.5
    ha *= np.sqrt(np.mean(nag ** 2)) / np.sqrt(np.mean(ha ** 2))
    onsets = np.concatenate([[0], np.cumsum(STEPS)]) * pace
    t_nag = onsets[-1] + len(ha) / SR + GAP * pace
    out = np.zeros(int(t_nag * SR) + len(nag), np.float32)
    for s, g in zip(onsets, GAIN):
        out[int(s * SR):int(s * SR) + len(ha)] += ha * g
    out[int(t_nag * SR):int(t_nag * SR) + len(nag)] += nag
    wavfile.write(os.path.join(A, out_name), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    print(f'は at {np.round(onsets, 2).tolist()}, ながすき at {t_nag:.2f}s, line {len(out) / SR:.2f}s')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)

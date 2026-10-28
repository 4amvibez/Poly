"""make_haha_kana — the kana-only line ははははながすき built from Raj's own recordings: one は (ha_user.mp3) said four
times, then ながすき (nagasuki_user.mp3), in one flowing phrase.  (ElevenLabs turns two or more は in a row into
laughter, so this line is not generated.)  Each は is cut to its syllable (the breathy onset and the long tail
trimmed) and they run straight on into ながすき.  Writes audio/haha_kana.wav."""
import os

import numpy as np
from scipy.io import wavfile

from build_vo import A, SR, load, speech

HA_CUT = (0.05, 0.29)                 # the syllable inside ha_user.mp3 (s)
STEP = 0.25                           # one は every STEP seconds
GAP = 0.02                            # before ながすき


def main():
    ha, nag = speech(load('ha_user.mp3')), speech(load('nagasuki_user.mp3'))
    ha = ha[int(HA_CUT[0] * SR):int(HA_CUT[1] * SR)].copy()
    n = SR // 100
    ha[:n] *= np.linspace(0, 1, n)
    ha[-4 * n:] *= np.linspace(1, 0, 4 * n)
    ha *= np.sqrt(np.mean(nag ** 2)) / np.sqrt(np.mean(ha ** 2))
    t_nag = 3 * STEP + len(ha) / SR + GAP
    out = np.zeros(int(t_nag * SR) + len(nag), np.float32)
    for k in range(4):
        out[int(k * STEP * SR):int(k * STEP * SR) + len(ha)] += ha
    out[int(t_nag * SR):int(t_nag * SR) + len(nag)] += nag
    wavfile.write(os.path.join(A, 'haha_kana.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    print(f'ながすき starts at {t_nag:.2f}s, line {len(out) / SR:.2f}s')


if __name__ == '__main__':
    main()

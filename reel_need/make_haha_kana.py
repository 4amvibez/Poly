"""make_haha_kana — the kana-only line ははははながすき built from Raj's own recordings: one は (ha_user.mp3) said four
times, then ながすき (nagasuki_user.mp3).  (ElevenLabs turns two or more は in a row into laughter, so this line is not
generated.)  The clip is laid out to the same length as before so nothing else in the reel moves.  Writes
audio/haha_kana.wav."""
import os

import numpy as np
from scipy.io import wavfile

from build_vo import A, SR, load, speech

LENGTH = 3.94                         # seconds of speech, as the line it replaces
HA = [0.0, 0.72, 1.44, 2.16]          # where each は starts


def main():
    ha, nag = speech(load('ha_user.mp3')), speech(load('nagasuki_user.mp3'))
    ha *= np.sqrt(np.mean(nag ** 2)) / np.sqrt(np.mean(ha ** 2))
    out = np.zeros(int(LENGTH * SR), np.float32)
    for s in HA:
        out[int(s * SR):int(s * SR) + len(ha)] += ha
    out[-len(nag):] += nag
    wavfile.write(os.path.join(A, 'haha_kana.wav'), SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    print(f'ながすき starts at {LENGTH - len(nag) / SR:.2f}s')


if __name__ == '__main__':
    main()

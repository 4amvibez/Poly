"""Generate the voiceover with Kokoro (local neural TTS, no network at run time), one WAV per line.

    python -m reels.nihongo.voice --model kokoro-v1.0.onnx --voices voices-v1.0.bin

The model files come from https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0).
Writes assets/voice/line_XX.wav and assets/voice/lines.json (durations)."""
import argparse
import json
import os

import numpy as np

from .script import LINES, parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'assets', 'voice')


def phonemes(k, parts):
    out = []
    for kind, text in parts:
        if kind == 'ipa':
            out.append(text)
        elif text.strip():
            lead = ' ' if text[0] == ' ' else ''
            trail = ' ' if text[-1] == ' ' else ''
            out.append(lead + k.tokenizer.phonemize(text.strip(), 'en-us') + trail)
        else:
            out.append(text)
    return ''.join(out).replace('  ', ' ').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--voices', required=True)
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', type=float, default=0.97)
    a = ap.parse_args()

    import soundfile as sf
    from kokoro_onnx import Kokoro
    k = Kokoro(a.model, a.voices)
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for i, (scene, line) in enumerate(LINES):
        cap, parts = parse(line)
        ph = phonemes(k, parts)
        audio, sr = k.create(ph, voice=a.voice, speed=a.speed, is_phonemes=True)
        audio = np.asarray(audio, np.float32)
        # trim the model's leading/trailing silence so the timeline controls the pauses
        on = np.where(np.abs(audio) > 0.01)[0]
        audio = audio[max(0, on[0] - int(0.03 * sr)): on[-1] + int(0.08 * sr)]
        path = os.path.join(OUT, f'line_{i:02d}.wav')
        sf.write(path, audio, sr)
        meta.append({'i': i, 'scene': scene, 'caption': cap, 'phonemes': ph, 'dur': len(audio) / sr, 'sr': sr})
        print(f'{i:02d} {len(audio) / sr:5.2f}s  {cap}')
    with open(os.path.join(OUT, 'lines.json'), 'w') as f:
        json.dump({'voice': a.voice, 'speed': a.speed, 'lines': meta}, f, ensure_ascii=False, indent=1)
    print('total speech', round(sum(m['dur'] for m in meta), 2), 's')


if __name__ == '__main__':
    main()

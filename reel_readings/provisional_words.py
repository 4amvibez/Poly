"""provisional_words — a stand-in audio/words.json timed from the draft script at Raj's measured pace (2.6 words/s,
pauses at punctuation), so scenes can be laid out and previewed for the storyboard before the voice is recorded.
Replaced by build_vo.py once the narration exists."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WPS = 2.6


def main():
    text = open(os.path.join(HERE, 'script_draft.txt'), encoding='utf8').read()
    t, words = 0.0, []
    for tok in text.split():
        d = 1 / WPS * (1.5 if re.search(r'[぀-鿿]', tok) else 1.0)
        words.append(dict(text=tok, type='word', start=round(t, 3), end=round(t + d * 0.85, 3)))
        t += d + (0.45 if tok[-1] in '.?!' else 0.15 if tok[-1] in ',:' else 0)
    os.makedirs(os.path.join(HERE, 'audio'), exist_ok=True)
    json.dump({'words': words, 'ja': {}, 'provisional': True}, open(os.path.join(HERE, 'audio', 'words.json'), 'w'), ensure_ascii=False)
    print(f'{len(words)} words, {t:.1f}s')


if __name__ == '__main__':
    main()

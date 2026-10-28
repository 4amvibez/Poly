"""make_docs — the delivery documents, written from the same timeline the film uses:
output/narration_script_timestamps.md (script, scene/beat table, sync hits, captions, checked Japanese) and the .srt"""
import os

import subtitles
import timeline as T
from audio import hit_times
from timeline import JA, at

HERE = os.path.dirname(os.path.abspath(__file__))
fmt = lambda s: f'{int(s // 60):d}:{s % 60:05.2f}'

BEATS = {
    'hook': ['Frame 1 slams in: "Is KANJI really necessary?" over a red sun with 漢字 inside, a reader below (taiko)',
             ('What if', '漢字 bursts into ink droplets in the breath before "What if…" (bowl); hiragana and katakana swirl into the sun'),
             ('hiragana and katakana', 'A thought cloud over the reader: あいうえお / カタカナ？')],
    'scroll': [('Look at this', 'A hanging scroll unrolls; 母は花が好き。 is written as the Japanese voice says it'),
               ('Mother', 'Arrows to a mother, a heart (likes), a flower; 母 / 好 / 花 take their colours'),
               ('remove the kanji', 'In the breath after "hiragana" the kanji are wiped away in a puff of ink (bowl); the kana slide apart'),
               ('Wait', 'ははははながすき。 is written in as it is read again; then it jolts and a red ？ pops (wood knock)')],
    'confusion': [('Wait', 'A reader\'s head, a confusion spiral, a sweat drop; guesses pop up: 母? 花? 歯? 鼻? 葉? 羽?'),
                  ('confusing', '"Mother likes… noses?!" (母は鼻が好き)')],
    'alike': [('Japanese has many', 'Three cards: はし hashi, はし, はし; sound waves on "sound alike"; "one sound…"'),
              ('Without kanji', 'Each card\'s picture is lost in mist, flickering between bridge / chopsticks / edge; "…three meanings?"'),
              ('Kanji gives', 'A big ink eye opens over the cards (bowl), sight lines to each'),
              ('recognize', 'The cards turn over: 橋 bridge, 箸 chopsticks, 端 edge')],
    'niwa': [('Heres another', '"another example…"; にわにはにわにわとりがいる。 written as it is read; clouds: にわ? (garden) にわ? (feathers) にわとり? (chicken)'),
             ('Now look at it', '庭には二羽鶏がいる。 colour-coded as it is read: 庭 garden, 二羽 two (birds), 鶏 chicken; a garden is painted below'),
             ('There are two', 'Two hens pop into the garden in the breath before (knock + clucks), counted 一羽, 二羽'),
             ('See how', 'Split screen: without kanji (grey, ？) vs with kanji (bright, a lit bulb)')],
    'answer': [('So is kanji', 'Calm view: mountains, sun, pagoda, the reader; "So, is kanji really necessary?"'),
               ('can be written', 'An open picture book in hiragana (むかしむかし、あるところに…): picture books for little children'),
               ('everyday', 'Signboards swing in: 駅 station, 出口 exit, 病院 hospital; they knock in the breath after "read." (wood)'),
               ('Hiragana katakana', 'ひらがな grammar & endings, カタカナ borrowed words, 漢字 meaning; one vermilion stroke joins them')],
    'obstacle': [('So dont look', 'A wall of kanji blocks drops in front of a small traveller; it rumbles on "obstacle" (？！)'),
                 ('Every character', 'The blocks rebuild as a staircase; the traveller climbs, each step turning gold; a red sun rises'),
                 ('Dont fear', '"Don\'t fear kanji." stamps in the breath before (taiko)'),
                 ('Learn to see', '花 brushed large as a cherry tree blooms behind it: "Learn to see the meaning behind it."')],
    'endcard': [('Follow me', 'Sunrise; the traveller with a pack; FOLLOW @raj_in_jp; "Let\'s master Japanese together!"; 一歩ずつ')],
}


def main():
    L = ['# Is kanji really necessary? — narration script & timeline', '',
         'Voices: **Raj** (English) and **Raj in Japanese** (every Japanese line). 母は花が好き。 is read naturally from the '
         'kanji (haha wa hana ga suki); once the kanji are removed, ははははながすき。 is read literally, kana by kana '
         '(ha・ha・ha・ha・na・ga・su・ki), the way it looks without kanji. Every take was checked with ElevenLabs Scribe.', '',
         f'Film length {fmt(T.DURATION)} ({T.DURATION:.1f} s), 1080×1920, 30 fps.', '',
         '## Full narration', '', open(os.path.join(HERE, 'script.txt')).read().strip(), '',
         '## Scenes and beats', '', '| Scene | Start–end | Beat time | Visual beat |', '|---|---|---|---|']
    for n, a, b in T.SCENES:
        first = True
        for bt in BEATS[n]:
            tt, txt = (a, bt) if isinstance(bt, str) else (at(bt[0])[0], bt[1])
            L.append(f"| {n if first else ''} | {fmt(a) + '–' + fmt(b) if first else ''} | {fmt(tt)} | {txt} |")
            first = False
        if n == 'endcard':
            L.append(f"|  |  | {fmt(JA['matane']['start'])} | またね！ stamped in vermilion (bowl); hold |")
    L += ['', '## Sync hits (picture event + sound, each in a breath of the narration)', '', '| Time | Hit |', '|---|---|']
    L += [f'| {fmt(t)} | {n} |' for n, t in hit_times()]
    L += ['', '## Captions (video time)', '', '| Start | End | Caption |', '|---|---|---|']
    for c in subtitles.CUES:
        L.append(f"| {fmt(c['start'])} | {fmt(c['end'])} | {' '.join(w for w, _ in c['words'])} |")
    L += ['', '## Japanese on screen (checked)', '',
          '- 母は花が好き。 (haha wa hana ga suki) "Mother likes flowers." — 母 haha mother, 花 hana flower, 好き suki like; '
          'は is the topic particle, written は and read "wa". In hiragana only: ははははながすき。, read here kana by kana '
          '(ha ha ha ha na ga su ki) to show how it looks without kanji.',
          '- The guesses: 母 haha mother, 歯 ha teeth, 花 hana flower, 鼻 hana nose, 葉 ha leaf, 羽 ha/hane feather; '
          '母は鼻が好き = "Mother likes noses".',
          '- はし hashi: 橋 bridge, 箸 chopsticks, 端 edge (written the same in kana; the pitch accent differs in speech).',
          '- にわにはにわにわとりがいる。 = 庭には二羽鶏がいる。 (niwa ni wa niwa niwatori ga iru) "There are two chickens in the garden": '
          '庭 niwa garden, 二羽 niwa two (birds; 羽 is the counter for birds), 鶏 niwatori chicken. 一羽 ichiwa one bird.',
          '- Everyday signs: 駅 eki station, 出口 deguchi exit, 病院 byōin hospital.',
          '- Picture book: むかしむかし、あるところに、おじいさんとおばあさんが… ("Once upon a time, in a certain place, an old man and '
          'an old woman…"), the classic opening of Japanese folk tales.',
          '- 一歩ずつ ippo zutsu "one step at a time"; またね！ mata ne "see you!"',
          '- Every illustration is painted in code; no image generation and no stock images.']
    out = os.path.join(HERE, 'output')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'narration_script_timestamps.md'), 'w').write('\n'.join(L) + '\n')
    t = lambda s: f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round((s % 1) * 1000)) % 1000:03d}'
    srt = []
    for i, c in enumerate(subtitles.CUES, 1):
        srt += [str(i), f"{t(c['start'])} --> {t(c['end'])}", ' '.join(w for w, _ in c['words']), '']
    open(os.path.join(out, 'kanji_necessary_reel.srt'), 'w').write('\n'.join(srt))
    print('wrote docs:', len(subtitles.CUES), 'caption cues')


if __name__ == '__main__':
    main()

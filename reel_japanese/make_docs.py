"""make_docs — write the delivery documents from the same timeline the film uses:
output/narration_script_timestamps.md (script, scene/beat table, captions, verified Japanese) and output/*.srt"""
import os

import subtitles
import timeline as T
from timeline import JA, at

HERE = os.path.dirname(os.path.abspath(__file__))
fmt = lambda s: f'{int(s // 60):d}:{s % 60:05.2f}'

BEATS = {
    'hook': [('Japanese has', '日本語 written'), ('three scripts', '漢字 · ひらがな · カタカナ stamp in with their names'),
             ('But', 'The scripts fall back into the paper'), ('why', 'Vermilion brushed ？ (taiko on the dot)')],
    'before_writing': ['Ink spreads from the ？ into an ancient settlement', ('living spoken', 'People brushed in; fire lit'),
                       ('conversation', 'Speech lines: conversation'), ('song', 'song'), ('story', 'story'),
                       ('no script', 'Lines dissolve: no written trace')],
    'map': ['Curved line becomes the Chinese coast', ('Through centuries', 'Ship sets out from Shandong'), ('China', 'CHINA label'),
            ('Korean Peninsula', 'KOREAN PENINSULA label'), ('Japanese islands', 'Ship reaches N. Kyushu; JAPAN; faint routes, wide view')],
    'kanji': ['Map parchment becomes manuscript; brush writes 字', ('kanji', '漢字 kanji, "Han characters"'),
              ('some for their meaning', '山 yama, by meaning'), ('others purely', '夜麻 ya + ma, by sound')],
    'hiragana': [('Writing sounds', '安 加 左 太 奈 = a ka sa ta na'), ('As scribes', '安 written; 宀 / 女 highlighted'),
                 ('in flowing cursive', 'Cursive form brushed over ghosted 安'), ('came hiragana', 'あ written; ひらがな label; bowl'),
                 ('The kanji an', 'Row 安 → cursive → あ (simplified illustration)'), ('Hiragana flourished', 'Iroha poem, Heian period')],
    'katakana': ['Heart Sutra page', ('quick notes', 'Vermilion reading notes beside the text'), ('They took', '伊 written'),
                 ('left side', '亻 ringed in vermilion'), ('and created', '亻 isolated → イ; カタカナ label; bowl'),
                 ('Sharp', '伊→イ, 加→カ, 多→タ')],
    'triptych': [('Today all three', 'Three bands: 漢字 / ひらがな / カタカナ'), ('Kanji carry', '山 core meanings'),
                 ('Hiragana handle', 'は・を grammar'), ('Katakana mark', 'コーヒー foreign words')],
    'sentence': ['Bands converge to the sentence line', ('Watashi', '私は日本語を勉強します。 written group by group'),
                 ('I study', '"I study Japanese."'), ('Kanji form', 'Kanji 私・日本語・勉強 underlined'),
                 ('Hiragana link', 'は・を・します turn vermilion; "katakana: none"')],
    'street': ['Ink wash opens a traditional street', ('Every line', 'Track forward: pagoda, lanterns, traveller, falling blossom')],
    'modern': ['A brushstroke passes; your Tokyo footage: rainy Shinjuku, ラーメン shop, BIG ECHO karaoke, yokocho alley, GiGO',
               ('Three scripts', 'GiGO sign freezes into a print: ゲーム・オアシス (katakana), の・に・び・め (hiragana), 飛・込 (kanji) ringed'),
               ('One language', '"Dive into the oasis of games!" — one sign, three scripts, one language')],
    'endcard': [('Follow for more', '"Follow for more Japanese content", @raj_in_jp + seal; vermilion sun, ink mountains, petals')],
}


def main():
    L = ['# The Japanese Writing System — narration script & timeline', '',
         f'Voice: **Raj** (your ElevenLabs cloned voice) and **Raj in Japanese** for the sign-off. '
         f'Narration starts at 0:00.{int(T.VO_OFFSET * 100):02d}. Film length {fmt(T.DURATION)} ({T.DURATION:.1f} s), '
         '1080×1920, 30 fps.', '', '## Full narration', '', open(os.path.join(HERE, 'script.txt')).read().strip(), '',
         '## Scenes and beats', '', '| Scene | Start–end | Beat time | Visual beat |', '|---|---|---|---|']
    for n, a, b in T.SCENES:
        first = True
        for bt in BEATS[n]:
            nth = 1 if (n == 'modern' and bt[0] == 'Three scripts') else 0
            tt, txt = (a, bt) if isinstance(bt, str) else (at(bt[0], nth)[0], bt[1])
            L.append(f"| {n if first else ''} | {fmt(a) + '–' + fmt(b) if first else ''} | {fmt(tt)} | {txt} |")
            first = False
        if n == 'endcard':
            L.append(f"|  |  | {fmt(JA['start'])} | 最後まで見てくれてありがとう written as Raj says it in Japanese |")
            L.append(f"|  |  | {fmt(JA['matane'])} | またね！ stamped in vermilion; English translation; hold to the end |")
    L += ['', '## Captions (video time)', '', '| Start | End | Caption |', '|---|---|---|']
    for c in subtitles.CUES:
        L.append(f"| {fmt(c['start'])} | {fmt(c['end'])} | {' '.join(w for w, _ in c['words'])} |")
    L.append(f"| {fmt(JA['start'])} | {fmt(T.DURATION)} | (on the end card) 最後まで見てくれてありがとう。またね！ — \"Thanks for watching till the end. See you!\" |")
    L += ['', '## Japanese used on screen (verified)', '',
          '- 日本語 (nihongo, "Japanese language"); 漢字 kanji; ひらがな hiragana; カタカナ katakana',
          '- 字 ji, "character"; 山 yama "mountain" written by meaning; 夜麻 ya+ma, man\'yōgana written by sound',
          '- 安 加 左 太 奈 → a ka sa ta na (man\'yōgana sources of あ か さ た な)',
          '- 安 → cursive → あ (shown as a simplified illustration); 伊 → イ (left component 亻); 加 → カ (力); 多 → タ (夕)',
          '- Iroha poem (いろはにほへと ちりぬるを わかよたれそ つねならむ), Heian period (794–1185)',
          "- Heart Sutra opening (Xuanzang's Chinese): 觀自在菩薩 行深般若波羅蜜多時 照見五蘊皆空 度一切苦厄",
          '- 私は日本語を勉強します。 "I study Japanese." Kanji: 私, 日本語, 勉強. Hiragana: は, を, します. Katakana: none.',
          '- GiGO sign in your footage: ゲームのオアシスに飛び込め！ (gēmu no oashisu ni tobikome!, "Dive into the oasis of games!"). '
          'Katakana: ゲーム, オアシス. Hiragana: の, に, び, め. Kanji: 飛, 込.',
          '- End card: 最後まで見てくれてありがとう。またね！ (saigo made mite kurete arigatō. mata ne!)',
          '- Fonts: Shippori Mincho B1 and Noto Serif CJK JP (JP face), with an automatic glyph-coverage check; stroke-order '
          'kanji from Make Me a Hanzi outlines (Arphic Public License). No image generation was used for any character.']
    out = os.path.join(HERE, 'output')
    open(os.path.join(out, 'narration_script_timestamps.md'), 'w').write('\n'.join(L) + '\n')
    t = lambda s: f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round((s % 1) * 1000)) % 1000:03d}'
    srt = []
    cues = [(c['start'], c['end'], ' '.join(w for w, _ in c['words'])) for c in subtitles.CUES]
    cues.append((JA['start'], T.DURATION - 0.2, '最後まで見てくれてありがとう。またね！\n(Thanks for watching till the end. See you!)'))
    for i, (a, b, txt) in enumerate(cues, 1):
        srt += [str(i), f'{t(a)} --> {t(b)}', txt, '']
    open(os.path.join(out, 'japanese_writing_reel.srt'), 'w').write('\n'.join(srt))
    print('wrote docs:', len(cues), 'caption cues')


if __name__ == '__main__':
    main()

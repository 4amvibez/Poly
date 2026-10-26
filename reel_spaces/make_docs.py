"""make_docs — the delivery documents, written from the same timeline the film uses:
output/narration_script_timestamps.md (script, scene/beat table, captions, checked Japanese, sources) and the .srt"""
import os

import subtitles
import timeline as T
from audio import hit_times
from timeline import JA, JA_SENT, at

HERE = os.path.dirname(os.path.abspath(__file__))
fmt = lambda s: f'{int(s // 60):d}:{s % 60:05.2f}'

BEATS = {
    'hook': ['Frame 1: the camera slams into a wall of unbroken Japanese, rows racing left and right; a vermilion brush '
             'banner NO SPACES?! is slapped across it (taiko); Raj\'s question starts at once, with captions; a reader sits '
             'below, staring up at it',
             ('long sentence', 'A red ？ pops over the reader\'s head'),
             ('without spaces', 'The rows grind to a halt; in the breath after the question red ？ marks burst out all over the wall (taiko)')],
    'spaces': [('In English', 'A stream: "I", "love", "learning" set down on stepping stones, with a "space" marked in each gap'),
               ('where one word', 'A little reader hops from stone to stone, word to word'),
               ('But Japanese', 'A river and an arched bridge: 私は日本語を勉強します。 written across it in one unbroken run'),
               ('run together', 'A vermilion stroke runs under the whole sentence; the reader walks out onto the bridge'),
               ('So how', 'Dashed lines try themselves between the characters, each with a small ？'),
               ('another begins', 'A big vermilion ？ is brushed; its dot lands in the breath after "begins?" (bowl)')],
    'kanji': [('The first clue', 'CLUE 1 · 漢字 kanji stamps in and settles as a header; landscape with a seated reader'),
              ('Look at', 'The sentence is written chunk by chunk as Raj says it in Japanese; "I study Japanese." beneath'),
              ('Kanji carries', 'Gold swashes sweep behind 私, 日本語, 勉強 while the hiragana step back'),
              ('nihongo', '日本語 lifts out with にほんご and "Japanese language"; a vermilion frame closes round it (bowl)'),
              ('recognize the word', 'Three more kanji words on cards with ink drawings: 学校 school, 東京 Tokyo, 電車 train')],
    'hiragana': [('Next comes', 'CLUE 2 · ひらがな hiragana; a little bullet train rolls in on its track below'),
                 ('Characters like', 'Kanji step back; は (topic marker), を (object marker) and します (polite verb ending) picked out'),
                 ('structured', 'The sentence parts at the particles into 私は | 日本語を | 勉強します (I · Japanese · study); '
                                'the chunks are written into the cars and the couplers は / を snap on (wood clack)'),
                 ('grammatical', 'Brush arcs ask who? and what? of the verb'),
                 ('work together', 'The couplers pulse and the train pulls away')],
    'katakana': [('And katakana', 'CLUE 3 · カタカナ katakana; three cards dealt face down'),
                 ('foreign-origin', 'Card turns: パソコン pasokon, computer (a borrowed word) with a laptop'),
                 ('names', 'Card turns: ラージ rāji, Raj (a name), waving hi'),
                 ('emphasized', 'Card turns: スゴイ！ sugoi, amazing!, with manga focus lines bursting round it'),
                 ('For example', 'The cards lift away; a cup of coffee drops onto its saucer in the breath (clink), steaming'),
                 ('kohee', 'コーヒー brushed large; kōhī = coffee')],
    'secret': [('But heres the', 'A reader\'s head is brushed in profile; 秘密 THE SECRET stamps in the breath after "secret." (taiko)'),
               ('just one clue', 'A lone clue seal pops up and shakes its head'),
               ('They combine', 'Five badges land one per word, 漢字 kanji, かな kana, 文法 grammar, 語彙 vocabulary, 文脈 context, '
                                'each pouring a coloured stream into the head; the brain warms with each'),
               ('to recognize', 'Below, the sentence is written; its chunks light up pink, gold and blue'),
               ('Their brains', 'The brain flashes in the breath after "sentence." (bowl)'),
               ('boundaries', 'The chunks part at the joins (vermilion ticks; I · Japanese · study)'),
               ('aren\'t written', 'They close up again, the colours still showing where each chunk ends')],
    'endless': [('At first', 'One endless line of Japanese flows past like a river, past a reader on the bank. It is made of '
                             'everything the reel taught: 私は日本語を勉強します。毎朝コーヒーを飲みます。東京の学校まで電車で行きます。'),
                ('once you understand', 'The scripts show their colours as it flows: kanji ink, hiragana rose, katakana blue'),
                ('the patterns', 'The line slows and stops on 毎朝コーヒーを飲みます。'),
                ('starts to make', 'It parts into 毎朝 | コーヒーを | 飲みます with every morning · coffee · drink')],
    'endcard': [('Follow me', 'FOLLOW · @raj_in_jp on a vermilion brush banner; four stepping stones carry 一 歩 ず つ'),
                ('one step', 'The reader hops across them one per word, each character turning vermilion; '
                             'ippo zutsu · one step at a time; the last leap lands in the breath (bowl)')],
}


def main():
    L = ['# How do Japanese people read without spaces? — narration script & timeline', '',
         f'Voice: **Raj** (your ElevenLabs voice) for the English, **Raj in Japanese** for the Japanese lines. '
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
            L.append(f"|  |  | {fmt(JA['start'])} | 最後まで見てくれてありがとう。 written as Raj says it in Japanese |")
            L.append(f"|  |  | {fmt(JA['matane'])} | またね！ in vermilion, the reader turns and waves (bowl); English translation; hold |")
    L += ['', '## Sync hits (picture event + sound, each in a breath of the narration)', '', '| Time | Hit |', '|---|---|']
    L += [f'| {fmt(t)} | {n} |' for n, t in hit_times()]
    L += ['', '## Captions (video time)', '', '| Start | End | Caption |', '|---|---|---|']
    for c in subtitles.CUES:
        L.append(f"| {fmt(c['start'])} | {fmt(c['end'])} | {' '.join(w for w, _ in c['words'])} |")
    L.append(f"| {fmt(JA['start'])} | {fmt(T.DURATION)} | (on the end card) 最後まで見てくれてありがとう。またね！ — "
             '"Thanks for watching till the end. See you next time!" |')
    L += ['', '## Japanese on screen (checked)', '',
          '- 私は日本語を勉強します。 (watashi wa nihongo o benkyō shimasu) "I study Japanese." Chunks 私は | 日本語を | 勉強します。',
          '- は is the topic particle, read "wa"; を is the object particle, read "o"; します is the polite form of する "do"',
          '- 日本語 にほんご nihongo "Japanese language"; 学校 がっこう gakkō "school"; 東京 とうきょう Tōkyō; 電車 でんしゃ densha "train"',
          '- 漢字 kanji, ひらがな hiragana, カタカナ katakana, かな kana; 文法 bunpō "grammar"; 語彙 goi "vocabulary"; 文脈 bunmyaku "context"; '
          '秘密 himitsu "secret"',
          '- パソコン pasokon "computer" (from "personal computer"); ラージ rāji, Raj written in katakana; スゴイ！ sugoi "amazing!" '
          '(すごい, written in katakana for emphasis); コーヒー kōhī "coffee" (spoken by the Japanese voice)',
          '- 毎朝コーヒーを飲みます。 (maiasa kōhī o nomimasu) "I drink coffee every morning."; '
          '東京の学校まで電車で行きます。 (Tōkyō no gakkō made densha de ikimasu) "I go to school in Tokyo by train."',
          '- 一歩ずつ (ippo zutsu) "one step at a time"',
          '- Sign-off: 最後まで見てくれてありがとう。またね！ (saigo made mite kurete arigatō. mata ne!). Your note had 最終まで; '
          '最後まで "to the end" is the natural form, and is what the Japanese voice says.',
          '- The hook\'s handscroll text is the opening of Natsume Sōseki\'s 吾輩は猫である (1905, public domain).',
          '- Every illustration, character and card is painted in code; no image generation and no stock images were used.']
    out = os.path.join(HERE, 'output')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'narration_script_timestamps.md'), 'w').write('\n'.join(L) + '\n')
    t = lambda s: f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round((s % 1) * 1000)) % 1000:03d}'
    cues = [(c['start'], c['end'], ' '.join(w for w, _ in c['words'])) for c in subtitles.CUES]
    cues.append((JA['start'], T.DURATION - 0.2, '最後まで見てくれてありがとう。またね！\n(Thanks for watching till the end. See you next time!)'))
    srt = []
    for i, (a, b, txt) in enumerate(cues, 1):
        srt += [str(i), f'{t(a)} --> {t(b)}', txt, '']
    open(os.path.join(out, 'spaces_reel.srt'), 'w').write('\n'.join(srt))
    print('wrote docs:', len(cues), 'caption cues;', JA_SENT['text'])


if __name__ == '__main__':
    main()

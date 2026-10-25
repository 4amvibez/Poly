"""make_docs — the delivery documents, written from the same timeline the film uses:
output/narration_script_timestamps.md (script, scene/beat table, captions, verified Japanese, sources) and the .srt"""
import os

import subtitles
import timeline as T
from timeline import JA, at

HERE = os.path.dirname(os.path.abspath(__file__))
fmt = lambda s: f'{int(s // 60):d}:{s % 60:05.2f}'

BEATS = {
    'hook': ['"How was KANJI created?", KANJI brushed in vermilion; sun, mountains, pagoda, blossom, a figure from behind',
             ('pictures', 'The oracle-bone sun (a circle with a mark inside) is brushed round the sun, then 日 is written into it (taiko)')],
    'record': [('Thousands of', 'Dawn by a river cliff: two people at a fire, a third paints on the rock'),
               ('record the world', 'Sun, mountains, fish, tree appear in red ochre on the rock, one after another')],
    'oracle': [('In ancient', 'Your oracle-bone photo; 甲骨文 brushed vertically; card "Oracle bones · Shang dynasty, China · c. 1200 BCE"'),
               ('turtle shells', 'The second oracle-bone photo slides in'),
               ('Some of these', 'Eight pictographs carved into bone tiles: sun, moon, mountain, tree, river, rain, person, fish')],
    'objects': [('The sun', 'Panel: vermilion sun'), ('The moon', 'Panel: crescent moon in a night wash'),
                ('Mountains', 'Panel: misty peaks'), ('Trees', 'Panel: broadleaf tree; the camera moves into the sun')],
    'sun': [('Look at this', 'The brush writes 日; "hi", "sun"'),
            ('Early forms', 'A painted sun → the oracle-bone form'), ('mark inside', 'The mark inside is ringed in vermilion'),
            ('Over centuries', 'Column: 甲骨文 oracle bone (Shang) → 金文 bronze (Zhou) → 篆書 seal (Qin) → 隷書 clerical (Han) → 楷書 regular'),
            ('today', 'A vermilion frame closes on today\'s 日 in the breath after "today" (bowl)')],
    'swirl': [('And the idea', 'A big brush sweeps a spiral across a misty peak; the next page opens out of the ink')],
    'nature': [('Yama means', 'Formed in place: a painted mountain → its pictograph traced over it → 山 yama "mountain"'),
               ('Kawa means', 'A river between its banks → three wavy lines → 川 kawa "river"'),
               ('Ki means', 'A tree with branches and roots → its pictograph → 木 ki "tree"; the three line up at the top'),
               ('Look closely', 'A painted sparrow → the early bird pictograph → 鳥 tori "bird"'),
               ('resemble', 'The paintings return faintly behind every character (bowl in the breath)')],
    'compound': [('But heres', 'The brush writes 河 (river)'), ('related to meaning', '氵 turns water-blue, "meaning: water"'),
                 ('clue to pronunciation', '可 turns vermilion, "sound: ka"; the parts click back together: 河 "ka", river (bowl)'),
                 ('This allowed', 'A person resting beside a tree → person + tree → 休 yasumi "rest"; then 日 + 月 → 明 bright, 田 + 力 → 男 man')],
    'map': [('Over time', 'THE JOURNEY OF KANJI map: Anyang, c. 1200 BCE (oracle bones)'),
            ('sophisticated', 'Chang\'an, Han dynasty ("kanji" = "Han characters"); the trail runs east to the coast'),
            ('They later', 'A boat takes over, red trail behind it, across the Yellow Sea'),
            ('Japan through', 'Baekje, Korean Peninsula, 4th–5th century: "Books and scholars cross from Korea to Japan"'),
            ('adapted', 'Kyushu → Inland Sea → Naniwa → Nara, 8th century: the Kojiki (712); wide view of the whole route')],
    'scholar': [('These characters became', 'A scholar from behind over a scroll; 漢字 brushed: kan "Han (China)" + ji "character"; kanji')],
    'message': [('So the next', 'A figure faces the sun over a misty peak and lake'), ('write a kanji', 'The brush writes 日 in the sky'),
                ('behind that', 'Its older forms come out from behind it and recede: 隷書, 篆書, 金文, 甲骨文'),
                ('history', 'The trail warms to vermilion in the breath after "history" (bowl)')],
    'endcard': [('Follow Raj', 'Ensō around 漢字; vermilion brush banner "@raj_in_jp"; "Follow @raj_in_jp for more fascinating stories about Japanese."')],
}


def main():
    L = ['# How was kanji created? — narration script & timeline', '',
         f'Voice: **Raj** (your ElevenLabs cloned voice, take B) and **Raj in Japanese** for the sign-off. '
         f'Narration starts at 0:00.{int(T.VO_OFFSET * 100):02d}. Film length {fmt(T.DURATION)} ({T.DURATION:.1f} s), '
         '1080×1920, 30 fps.', '', '## Full narration', '', open(os.path.join(HERE, 'script.txt')).read().strip(), '',
         '## Scenes and beats', '', '| Scene | Start–end | Beat time | Visual beat |', '|---|---|---|---|']
    for n, a, b in T.SCENES:
        first = True
        for bt in BEATS[n]:
            tt, txt = (a, bt) if isinstance(bt, str) else (at(bt[0])[0], bt[1])
            L.append(f"| {n if first else ''} | {fmt(a) + '–' + fmt(b) if first else ''} | {fmt(tt)} | {txt} |")
            first = False
        if n == 'endcard':
            L.append(f"|  |  | {fmt(JA['start'])} | 最後まで見てくれてありがとう written as Raj says it in Japanese |")
            L.append(f"|  |  | {fmt(JA['matane'])} | またね！ stamped in vermilion; English translation; hold to the end |")
    L += ['', '## Captions (video time)', '', '| Start | End | Caption |', '|---|---|---|']
    for c in subtitles.CUES:
        L.append(f"| {fmt(c['start'])} | {fmt(c['end'])} | {' '.join(w for w, _ in c['words'])} |")
    L.append(f"| {fmt(JA['start'])} | {fmt(T.DURATION)} | (on the end card) 最後まで見てくれてありがとう。またね！ — \"Thanks for watching till the end. See you!\" |")
    L += ['', '## Japanese and history on screen (checked)', '',
          '- 日 hi "sun"; 山 yama "mountain"; 川 kawa "river"; 木 ki "tree"; 鳥 tori "bird" (Japanese form; the simplified '
          'Chinese 鸟 is not used in Japanese)',
          '- 甲骨文 oracle-bone script (Shang dynasty, c. 1250–1050 BCE, shown as "c. 1200 BCE"); 金文 bronze script (Zhou); '
          '篆書 seal script (standardised under the Qin, 221 BCE); 隷書 clerical script (Han); 楷書 regular script (today\'s form)',
          '- The ancient forms are hand-redrawn, simplified illustrations of each period style, following your reference sheets',
          '- 河 = 氵 (water: the meaning) + 可 (ka: the sound); 河 reads ka (on\'yomi) and kawa (kun\'yomi), "river"',
          '- 休 = 亻 person + 木 tree: a person resting by a tree, "rest" (休み yasumi); 明 = 日 sun + 月 moon, "bright"; '
          '男 = 田 field + 力 strength, "man"',
          '- 漢字 kanji = 漢 kan, the Han dynasty (China) + 字 ji, "character"',
          '- Map: Anyang 安陽 (late Shang capital, where the oracle bones were found); Chang\'an 長安 (Han capital); '
          'Baekje 百済 (Korean kingdom; tradition credits its scholars with bringing Chinese books to Japan, 4th–5th c.); '
          'Nara 奈良; the Kojiki (712) is Japan\'s oldest surviving book. The route is an illustration, not one documented voyage.',
          '- End card: 最後まで見てくれてありがとう。またね！ (saigo made mite kurete arigatō. mata ne!)',
          '- No image generation was used for any character or map. The oracle-bone photographs were '
          'supplied by you; see data/README.md about credits.']
    out = os.path.join(HERE, 'output')
    open(os.path.join(out, 'narration_script_timestamps.md'), 'w').write('\n'.join(L) + '\n')
    t = lambda s: f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round((s % 1) * 1000)) % 1000:03d}'
    srt = []
    cues = [(c['start'], c['end'], ' '.join(w for w, _ in c['words'])) for c in subtitles.CUES]
    cues.append((JA['start'], T.DURATION - 0.2, '最後まで見てくれてありがとう。またね！\n(Thanks for watching till the end. See you!)'))
    for i, (a, b, txt) in enumerate(cues, 1):
        srt += [str(i), f'{t(a)} --> {t(b)}', txt, '']
    open(os.path.join(out, 'kanji_reel.srt'), 'w').write('\n'.join(srt))
    print('wrote docs:', len(cues), 'caption cues')


if __name__ == '__main__':
    main()

"""The narration: one entry per spoken line, with the scene it belongs to.

`say` is the caption text (what the viewer reads). Words in {braces} are Japanese: the caption shows the part
before the "|", and the voice speaks the hand-written IPA after it, so the English voice says "kanji" as
[kɑːndʒi] rather than [kændʒi].

Historical notes (checked against the standard accounts of the writing system):
  - spoken Japanese long predates writing; Chinese characters arrived through contact with China and the
    Korean kingdoms (Baekje scribes, Buddhist texts in the 6th c.), not with the spoken language itself
  - kanji were used for meaning and, as man'yōgana, purely for sound (安 for "a", 伊 for "i")
  - hiragana grew out of cursive (草書) man'yōgana: 安 → あ
  - katakana grew out of parts of man'yōgana written by monks and scholars glossing Chinese texts: 伊 → イ
    (the left side, 亻). The storyboard's 以 → イ is not the accepted source: 以 gives hiragana い.
"""

# IPA for the Japanese words, spoken by the English voice
JA = {
    'kanji': 'kˈɑːndʒi', 'Kanji': 'kˈɑːndʒi',
    'hiragana': 'hˌiɾɑɡˈɑːnɑ', 'Hiragana': 'hˌiɾɑɡˈɑːnɑ',
    'katakana': 'kˌɑtɑkˈɑːnɑ', 'Katakana': 'kˌɑtɑkˈɑːnɑ',
    'Nihongo': 'nˌihˈoŋɡo',
    'an': 'ˈɑn', 'a': 'ˈɑː', 'i': 'ˈiː', 'wa': 'wˈɑ', 'o': 'ˈoː', 'shimasu': 'ʃimˈɑs',
    'Watashi wa nihongo o benkyō shimasu': 'wɑtˈɑʃi wɑ nˌihˈoŋɡo o bˈeŋkjoː ʃimˈɑs',
}

LINES = [
    # (scene, caption text with {japanese|spoken-key})
    (1, "Every sentence of modern Japanese weaves together three different writing systems."),
    (2, "To see why, we have to go back to a time before Japan had writing at all."),
    (2, "For centuries, the language of these islands lived only in speech: in conversation, in song, and in memory."),
    (3, "Writing came from the continent."),
    (3, "Through centuries of contact with China and the Korean kingdoms, Chinese characters reached Japan, "
        "carried by scribes, scholars and Buddhist scriptures."),
    (4, "These characters, {kanji}, were created for Chinese, a very different language."),
    (4, "So scribes in Japan adapted them. Some {kanji} were used for their meaning. "
        "Others were borrowed purely for their sound."),
    (5, "Written quickly with a brush, those sound characters became softer and more flowing."),
    (5, "Over generations, the cursive forms were simplified into a new script: {hiragana}."),
    (5, "The {kanji} {安 (an)|an}, in flowing cursive, gradually became {あ (a)|a}."),
    (6, "{Katakana} took a different path."),
    (6, "Buddhist monks and scholars, annotating Chinese texts, needed a quicker shorthand, "
        "so they often wrote just one part of a {kanji}."),
    (6, "The left side of {伊 (i)|i} became the {katakana} {イ (i)|i}."),
    (7, "Today, all three work together. {Kanji} carry meaning. {Hiragana} handle grammar. "
        "And {katakana} mostly write words borrowed from other languages."),
    (8, "{私は日本語を勉強します。|Watashi wa nihongo o benkyō shimasu} I study Japanese."),
    (8, "{Kanji} give us \"I\", \"Japanese\" and \"study\". {Hiragana} supply the particles {wa} and {o}, "
        "and the verb {shimasu}."),
    (9, "From temple manuscripts to the screen in your hand, these scripts still carry Japanese culture and identity."),
    (10, "{Nihongo}. One language, three scripts, and centuries of history in every stroke."),
]


def parse(line):
    """-> (caption text, [(kind, text)]) where kind is 'en' (phonemize as English) or 'ipa' (spoken as given)"""
    import re
    cap, parts, pos = '', [], 0
    for m in re.finditer(r'\{([^}|]+)(?:\|([^}]+))?\}', line):
        if m.start() > pos:
            parts.append(('en', line[pos:m.start()]))
            cap += line[pos:m.start()]
        shown, key = m.group(1), m.group(2) or m.group(1)
        parts.append(('ipa', JA[key]))
        cap += shown
        pos = m.end()
    if pos < len(line):
        parts.append(('en', line[pos:]))
        cap += line[pos:]
    return cap, parts

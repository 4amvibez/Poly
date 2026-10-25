# 日本語 — where Japanese writing came from

A 110-second documentary for Instagram Reels (1080×1920, 30 fps) about the Japanese writing system, painted
entirely in code in sumi-e ink wash with the `poly_ink` bristle brush. Every kanji and kana on screen is
**written by the brush, stroke by stroke, in the correct stroke order** (stroke paths from KanjiVG); the
voiceover, the music and the sound effects are generated locally too. No image or video model is involved.

Output: `drawing_output/reels/nihongo.mp4` (H.264 + AAC, −14 LUFS) and `drawing_output/reels/nihongo.srt`.

## Narration

Voice: Kokoro (`af_heart`, speed 0.97), generated offline. Japanese words are spoken from hand-written IPA
(`script.py`), so the English voice says *kanji* [kɑːndʒi], *hiragana*, *Nihongo* and the example sentence with
Japanese vowels instead of English ones.

| Scene | | Time (s) | Line |
|---|---|---|---|
| 1 | Opening hook — 日本語 | 1.2–6.0 | Every sentence of modern Japanese weaves together three different writing systems. |
| 2 | Before writing | 6.7–11.2 | To see why, we have to go back to a time before Japan had writing at all. |
|  |  | 11.6–18.5 | For centuries, the language of these islands lived only in speech: in conversation, in song, and in memory. |
| 3 | China, Korea and Japan | 19.5–21.2 | Writing came from the continent. |
|  |  | 21.5–30.9 | Through centuries of contact with China and the Korean kingdoms, Chinese characters reached Japan, carried by scribes, scholars and Buddhist scriptures. |
| 4 | The adoption of kanji | 31.6–36.2 | These characters, kanji, were created for Chinese, a very different language. |
|  |  | 36.6–43.7 | So scribes in Japan adapted them. Some kanji were used for their meaning. Others were borrowed purely for their sound. |
| 5 | Hiragana | 44.4–49.2 | Written quickly with a brush, those sound characters became softer and more flowing. |
|  |  | 49.6–54.7 | Over generations, the cursive forms were simplified into a new script: hiragana. |
|  |  | 55.4–58.9 | The kanji 安 (an), in flowing cursive, gradually became あ (a). |
| 6 | Katakana | 59.6–61.5 | Katakana took a different path. |
|  |  | 61.9–70.1 | Buddhist monks and scholars, annotating Chinese texts, needed a quicker shorthand, so they often wrote just one part of a kanji. |
|  |  | 70.7–73.3 | The left side of 伊 (i) became the katakana イ (i). |
| 7 | The three scripts | 74.0–83.3 | Today, all three work together. Kanji carry meaning. Hiragana handle grammar. And katakana mostly write words borrowed from other languages. |
| 8 | Modern Japanese in practice | 84.2–87.7 | 私は日本語を勉強します。 I study Japanese. |
|  |  | 88.1–95.1 | Kanji give us "I", "Japanese" and "study". Hiragana supply the particles wa and o, and the verb shimasu. |
| 9 | Past and present | 95.8–101.9 | From temple manuscripts to the screen in your hand, these scripts still carry Japanese culture and identity. |
| 10 | The final ink composition | 102.6–107.2 | Nihongo. One language, three scripts, and centuries of history in every stroke. |

247 words, 95.8 s of speech (155 words per minute), 110.0 s in all.
Direction: curious and quiet at the open, steady through the history, a beat of silence before each reveal
(the map, あ, イ, the sentence), reflective at the close. Pauses come from the timeline (`timeline.py`), not
from slowing the voice.

## Historical notes

- Spoken Japanese long predates writing. Chinese characters reached Japan through centuries of contact with
  China and the Korean kingdoms (Baekje scribes, Buddhist texts); the language itself did not come from China.
- Kanji were used for meaning (山 *yama*, "mountain") and, as man'yōgana, purely for sound (安 for *a*).
- Hiragana grew out of cursive (草書) forms of those sound characters: 安 → あ. The cursive 安 on screen is an
  illustration of the stage, not a facsimile of one manuscript.
- Katakana grew out of parts of characters, written by monks and scholars glossing Chinese texts: the left side
  of 伊 → イ. **The storyboard showed 以 → イ; that is not the accepted derivation (以 is the source of hiragana
  い), so the film uses 伊.** The sutra page in scene 6 carries real reading glosses (kunten) for the opening of
  the Heart Sutra (行ズル, 多ヲ, 空ナリト, 見シテ …).
- 私は日本語を勉強します。: kanji 私・日本語・勉強; hiragana は・を・します; no katakana.

## How it is made

| File | What |
|---|---|
| `script.py` | narration lines, scene per line, IPA for the Japanese words |
| `voice.py` | Kokoro TTS → `assets/voice/line_XX.wav` + `lines.json` (durations) |
| `timeline.py` | line start times, scene windows, caption chunks — all from the measured voice durations |
| `glyph.py` | KanjiVG stroke paths → `poly_ink.brush_stroke`, with the pressure profile picked from the stroke type (㇐ press-and-stop, ㇒ lift to a point, ㇏ heavy foot, ㇚ hook …) |
| `ink.py` | the motion engine: `Track` (timed strokes with wet ink that blooms and settles), eased `Keys`, camera / parallax, organic `InkFront` masks, pigment-on-paper compositing, typography |
| `common.py` | `Scene` base, the writing brush that follows the stroke tip, sprites, labels |
| `scenes_a/b/c.py` | the ten scenes (title, settlement, map, kanji, hiragana, katakana, triptych, sentence, street, wave) |
| `audio.py` | the soundtrack: koto (Karplus-Strong) in the miyako-bushi scale, drone, shakuhachi-like flute, taiko, bell, hyoshigi, brush sounds from the real stroke timings, ocean / wind / fire, ducking under the voice |
| `render.py` | the film: transitions, captions, grain; parallel rendering to ffmpeg; mux |

```bash
pip install -r reels/nihongo/requirements.txt
# 1. voice (only when the script changes; the WAVs are committed)
pip install kokoro-onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python -m reels.nihongo.voice --model kokoro-v1.0.onnx --voices voices-v1.0.bin
# 2. the film (about 25 minutes on 4 cores)
python -m reels.nihongo.render
# stills of one scene at local times, or of the finished film at global times
python -m reels.nihongo.render --stills 5:2,9,14
python -m reels.nihongo.render --frames 20,64
```

To use a recorded voiceover instead, replace `assets/voice/line_XX.wav` (one file per line of `script.py`) and
run `voice.py`'s duration step again (or edit `lines.json`); the timeline, captions and music follow the new
durations.

## Assets and licences

- Stroke order data: [KanjiVG](https://kanjivg.tagaini.net) © Ulrich Apel, CC BY-SA 3.0 (`assets/kanjivg`).
- Coastlines and rivers: [Natural Earth](https://www.naturalearthdata.com), public domain (`assets/geo`).
- Fonts: Shippori Mincho and EB Garamond, SIL Open Font License (`assets/fonts`).
- Voice: Kokoro-82M (Apache 2.0).

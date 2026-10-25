# Sources

- `hanzi_strokes.json`: stroke outlines and medians for the kanji the reel writes stroke by stroke
  (日 月 山 川 木 河 可 字 男 田 力 分 八 刀 電 雨 人 漢 鳥 休 明). Taken from hanzi-writer-data 2.0.1 (Make Me a Hanzi),
  derived from Arphic PL fonts under the Arphic Public License. 漢 is shown in a Japanese font instead, because the
  data's 漢 is the 14-stroke Chinese traditional form; Japanese uses a 13-stroke form.
- `ref_*.png/jpg`: the reference sheets supplied by @raj_in_jp. They are guides only and are not shown in the film.
  They cover the evolution chart (oracle bone, greater and lesser seal, clerical and later forms), the
  picture → ancient → modern sheet, the compound sheet and the storyboard. The ancient forms in `../ancient.py`
  are redrawn by hand from them as simplified illustrations of each period style.
- `../photos/`: photographs supplied by @raj_in_jp of two Shang-dynasty oracle bones. The photographs may belong to the museums or photographers who took them, so credit
  or clear them before posting if you know the source.
- Fonts (`../fonts/`, all SIL Open Font License, from Google Fonts):
  - Shippori Mincho B1 ExtraBold (subset) for Japanese text.
  - Yuji Syuku (subset), a Japanese brush face, for 漢字.
  - Kaushan Script for the brush "KANJI".
  - Cormorant Garamond and Noto Serif JP for the Latin text.
- Coastlines: global-land-mask (GLOBE 1 km land data). The rivers on the map are simplified by hand.

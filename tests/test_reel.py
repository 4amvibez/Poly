import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reels.nihongo.glyph import strokes_of, write  # noqa: E402
from reels.nihongo.script import LINES, parse  # noqa: E402
from reels.nihongo.timeline import Timeline  # noqa: E402


def test_kanjivg_stroke_counts():
    # stroke counts every learner knows: the brush must write each character with the right number of strokes
    for ch, n in {'日': 4, '本': 5, '語': 14, '漢': 13, '字': 6, '安': 6, '伊': 6, 'あ': 3, 'イ': 2, '私': 7,
                  '勉': 10, '強': 11, 'カ': 2, 'ひ': 1}.items():
        assert len(strokes_of(ch)) == n, ch


def test_write_glyph_strokes_in_order():
    ws, end = write(300, 300, '本', 20, 20, 240, 0.5, seed=1)
    assert len(ws) == 5
    assert all(a.t1 <= b.t0 for a, b in zip(ws, ws[1:]))       # one stroke after another
    assert end > ws[-1].t1 and ws[0].t0 == 0.5
    assert all(np.isfinite(w.stroke.dens).all() and w.stroke.dens.max() > 0.5 for w in ws)


def test_script_parses_japanese_terms():
    cap, parts = parse(LINES[9][1])
    assert '安' in cap and 'あ' in cap and '{' not in cap
    assert any(kind == 'ipa' for kind, _ in parts)


def test_timeline_fits_reels_and_captions_cover_every_line():
    tl = Timeline()
    assert 100 <= tl.total <= 110.5
    assert [n for n in sorted(tl.scenes)] == list(range(1, 11))
    caps = tl.captions()
    for ln in tl.lines:
        assert any(c[0] <= ln['start'] + 0.01 and c[1] >= ln['start'] for c in caps)
    assert all(len(c[2]) <= 2 for c in caps)

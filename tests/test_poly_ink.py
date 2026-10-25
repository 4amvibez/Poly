import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from poly_ink import InkScene, animate_scene, brush_stroke, cloud_temple, ink_landscape, make_gif, splatter, sweep  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def test_cloud_temple_matches_committed_painting():
    img = cloud_temple()
    ref = Image.open(os.path.join(REPO, 'drawing_output', 'images', 'ink_cloud_temple.png')).convert('RGB')
    assert img.size == (1920, 1080)
    assert np.array_equal(np.asarray(img.convert('RGB')), np.asarray(ref))


def test_landscape_size_stages_and_determinism():
    a, stages = ink_landscape('测试', seed=5, width=640, height=360, with_stages=True)
    b = ink_landscape('测试', seed=5, width=640, height=360)
    assert a.size == (640, 360)
    assert [n for n, _ in stages] == ['宣纸', '远山', '主峰', '云雾', 'final']
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(ink_landscape('测试', seed=6, width=640, height=360)))


def test_make_gif(tmp_path):
    _, stages = ink_landscape(seed=2, width=480, height=270, with_stages=True)
    out = tmp_path / 'a.gif'
    n, secs = make_gif(stages, str(out), size=(240, 135))
    g = Image.open(out)
    assert g.size == (240, 135) and g.n_frames == n == 5 + 4 * 5          # 5 keyframes + 5 fades between each pair
    assert abs(secs - (4 * 0.9 + 4 * 5 * 0.09 + 2.6)) < 1e-9


def _small_scene():
    s = InkScene(400, 240, seed=1)
    t = s.add(brush_stroke(400, 240, [(60, 40), (62, 120), (58, 200)], 24, dry=0.4, seed=1), 0, 300)
    t = s.add(sweep(400, 240, [(120, 200), (220, 120), (380, 60)], 50, seed=2), t, 300)
    s.add(splatter(400, 240, 330, 70, (40, 25), n=12, seed=3), t, 100)
    return s


def test_brush_is_deterministic_and_inked():
    a = brush_stroke(400, 240, [(40, 40), (200, 120), (360, 200)], 30, seed=7)
    b = brush_stroke(400, 240, [(40, 40), (200, 120), (360, 200)], 30, seed=7)
    assert np.array_equal(a.dens, b.dens)
    assert a.dens.max() > 0.8 and 0 <= a.time.min() and a.time.max() <= 1
    assert brush_stroke(400, 240, [(5000, 5000), (6000, 6000)], 30) is None      # off the canvas


def test_strokes_are_revealed_in_painting_order():
    st = brush_stroke(400, 240, [(20, 120), (200, 120), (380, 120)], 30, dry=0.2, seed=3)
    half = st.reveal(0.5)
    xs = np.nonzero(half.sum(0) > 0.5)[0] + st.x0
    assert xs.max() < 230                                     # only the first half (left side) is painted
    assert (half <= st.dens + 1e-6).all() and half.sum() < st.dens.sum()


def test_scene_render_over_time(tmp_path):
    s = _small_scene()
    blank, mid, done = (np.asarray(s.render(at=a), float) for a in (0, 450, None))
    ink = lambda img: (255 - img).sum()
    assert ink(blank) < ink(mid) < ink(done)
    n, secs = animate_scene(s, str(tmp_path / 'w.gif'), size=(200, 120), frame_ms=100)
    assert Image.open(tmp_path / 'w.gif').size == (200, 120) and n > 4 and secs > 2.6

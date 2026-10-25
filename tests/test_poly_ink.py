import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from poly_ink import cloud_temple, ink_landscape, make_gif  # noqa: E402

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

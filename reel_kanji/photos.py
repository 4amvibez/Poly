"""photos — the archival photographs of two Shang-dynasty oracle bones (supplied by @raj_in_jp), cut out of their
backgrounds and graded to sit on the washi paper like real objects.

    cutout(name) -> dict(rgb HxWx3 float, alpha HxW float, parts [(x, y, w, h), ...])"""
import os

import cv2
import numpy as np

from engine import HERE, blur, cached

SRC = {
    'bone1': ('oracle_bone_1.jpg', 'dark'),       # inscribed scapula on black
    'bone2': ('oracle_bone_2.jpg', 'saturated'),  # inscribed fragment on a grey gradient
}


def _mask(img, mode):
    if mode == 'dark':
        v = cv2.GaussianBlur(img.mean(2), (0, 0), 1.5)
        m = (v > (38 if img.shape[0] == 1000 else 30)).astype(np.uint8)
    else:
        s = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32), (0, 0), 1.5)
        m = (s > 60).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if st[i, 4] > (3000 if mode == 'dark' else 20000)]
    m = np.isin(lab, keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    parts = [tuple(int(v) for v in st[i, :4]) for i in keep]
    return m, parts


def cutout(name, scale=2.0):
    def build():
        fn, mode = SRC[name]
        img = cv2.imread(os.path.join(HERE, 'photos', fn))
        m, parts = _mask(img, mode)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        a = cv2.resize(m.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        a = np.clip(blur(a, 1.0 * scale) * 1.15 - 0.08, 0, 1)
        a = cv2.erode(a, np.ones((3, 3), np.uint8))                    # lose the dark fringe of the old background
        rgb = img[..., ::-1].astype(np.float32) / 255
        # grade: a touch less saturated, warmed towards the paper, lifted shadows (an archival print look)
        g = rgb.mean(2, keepdims=True)
        rgb = rgb * 0.86 + g * 0.14
        rgb = np.clip(rgb * np.array([1.03, 1.0, 0.93], np.float32) * 0.94 + 0.04, 0, 1)
        return {'rgb': rgb.astype(np.float32), 'alpha': a.astype(np.float32),
                'parts': (np.array(parts, np.float32) * scale).astype(np.float32)}
    return cached(f'photo_{name}_v1', build)


def place_photo(fr, ph, M, opacity=1.0, alpha=None, shadow=0.38, lift=1.0):
    """composite a cutout through affine M (photo px -> screen) with a soft contact shadow on the paper"""
    from engine import H, W
    a = ph['alpha'] if alpha is None else alpha
    A = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR) * opacity
    if shadow > 0:
        Ms = M.copy()
        Ms[:, 2] += (10 * lift, 16 * lift)
        sh = blur(cv2.warpAffine(a, Ms, (W, H), flags=cv2.INTER_LINEAR), 10 * lift) * opacity
        fr.img *= (1 - shadow * sh)[..., None]
    C = cv2.warpAffine(ph['rgb'], M, (W, H), flags=cv2.INTER_LINEAR)
    fr.img = fr.img * (1 - A[..., None]) + C * A[..., None]
    return A

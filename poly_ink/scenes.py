"""Ink wash scenes. Each returns a PIL Image, or (Image, stages) with with_stages=True, where stages is a
list of (name, Image) snapshots in drawing order (paper -> far ranges -> peaks -> ... -> final)."""
import numpy as np
from inkpaint import Painting, fbm1d
from core import spline


def _finish(p, with_stages):
    img = p.composite()
    if not with_stages:
        return img
    return img, list(p.stages) + [('final', img)]


def cloud_temple(with_stages=False):
    """云深不知处 — temples on twin peaks above the mist, a footbridge with a monk, a pine on the
    foreground rock, a red sun, birds, title and seal. 1920x1080, seed 7071 (always the same picture)."""
    p = Painting(1920, 1080, seed=7071)
    p.paper()
    p.stage('宣纸')

    # ---- far ranges, pale, sinking into mist
    far = p.ridge(600, peaks=[(300, 130, 280), (1780, 150, 300)], rough=24, rough_scale=220, seed=3)
    p.wash(far, dens=0.14, decay=160, tex=140, edge=0.25, mist_y=720, mist_h=120, seed=4)
    p.stage('远山')

    # ---- twin main peaks: the taller one right of centre, a lower shoulder peak to its left
    SX, SY = 1180, 230
    main = p.peak(SX, SY, height=600, left_w=330, right_w=480, shoulders=[(SX + 260, 80, 100), (SX - 260, 60, 90)], seed=5)
    p.wash(main, dens=0.44, decay=230, tex=90, edge=0.45, mist_y=900, mist_h=90, seed=6)
    p.cun(main, SX - 560, SX + 620, n=150, shadow_x=SX + 40, max_y=900)
    p.moss(main, SX - 540, SX + 600, n=60)
    p.ridge_line(main, SX - 620, SX + 660)
    TX, TY = 760, 430
    second = p.peak(TX, TY, height=420, left_w=300, right_w=260, shoulders=[(TX - 200, 50, 80)], seed=8)
    p.wash(second, dens=0.36, decay=200, tex=100, edge=0.4, mist_y=880, mist_h=90, seed=9)
    p.cun(second, TX - 380, TX + 300, n=90, shadow_x=TX - 30, max_y=880)
    p.moss(second, TX - 360, TX + 280, n=40)
    p.ridge_line(second, TX - 420, TX + 320)
    p.stage('主峰')

    # ---- mist in the gorge, the footbridge across it, a pavilion on each peak
    p.mist(860, 80, 0.85)
    p.mist(640, 50, 0.5)
    BY = 742
    p.stroke(spline([(905, BY), (960, BY + 12), (1015, BY)], 10), 3, 0.75, dry=0.25, taper=(0.1, 0.1))
    p.stroke(spline([(905, BY - 14), (960, BY - 3), (1015, BY - 14)], 10), 1.4, 0.55, dry=0.3, taper=(0.1, 0.1))
    for x in np.linspace(912, 1008, 7):
        y = BY + 12 * (1 - ((x - 960) / 55) ** 2)
        p.stroke([(x, y - 14), (x, y)], 1.1, 0.5, dry=0.3)
    p.pagoda(TX, int(second[TX]) + 6)
    p.pagoda(SX, int(main[SX]) + 6)
    p.traveller(940, BY + 4)
    p.stage('桥亭')

    # ---- foreground rock and pine, bottom-left
    rock = 820 + 260 * ((p.xs / 560) ** 1.5) + 28 * fbm1d(1920, 70, 4, 17)
    rock[p.xs > 560] = 2000
    p.rock(rock)
    p.pine((220, 1000), (300, 620), bend=60,
           branches=[((270, 770), (450, 720), -30), ((284, 700), (150, 660), 26), ((296, 650), (420, 600), -20)],
           extra_tufts=[(305, 610), (360, 628)])
    p.stage('松石')

    # ---- red sun, birds, title and seal in the open sky at top left
    p.sun(1560, 170, 40)
    p.birds([(1380, 280, 0.9), (1420, 300, 0.7), (1350, 312, 0.6)])
    p.title_vertical('云深不知处', 150, 120, 86)
    p.seal('山居', 258, 150, 40)
    return _finish(p, with_stages)


def ink_landscape(title='山水', seal='山居', seed=1, width=1920, height=1080, sun=True, with_stages=False):
    """A configurable ink landscape: far ranges, a main peak with a summit pavilion, mist, an optional red
    sun, a vertical title and a seal. Layout scales with width/height; seed changes the brushwork."""
    sx, sy = width / 1920, height / 1080
    X = lambda v: v * sx
    Y = lambda v: v * sy
    p = Painting(width, height, seed=seed)
    p.paper()
    p.stage('宣纸')

    far = p.ridge(Y(600), peaks=[(X(400), Y(140), X(300)), (X(1600), Y(120), X(260))], seed=seed + 1)
    p.wash(far, dens=0.16, decay=Y(160), tex=140, mist_y=Y(720), mist_h=Y(110), seed=seed + 2)
    p.stage('远山')

    PX, PY = X(1100), Y(260)
    main = p.peak(PX, PY, height=Y(560), left_w=X(360), right_w=X(460),
                  shoulders=[(PX + X(240), Y(70), X(90))], seed=seed + 3)
    p.wash(main, dens=0.45, decay=Y(230), mist_y=Y(900), mist_h=Y(90), seed=seed + 4)
    p.cun(main, PX - X(600), PX + X(600), n=150, shadow_x=PX, max_y=Y(900))
    p.moss(main, PX - X(580), PX + X(600), n=60)
    p.ridge_line(main, PX - X(650), PX + X(650))
    p.stage('主峰')

    p.pagoda(PX, int(main[int(PX)]) + 6)
    p.mist(Y(860), Y(80), 0.85)
    p.stage('云雾')

    if sun:
        p.sun(X(1560), Y(170), 40 * min(sx, sy))
    p.birds([(X(1380), Y(280), 0.9), (X(1420), Y(300), 0.7)])
    size = int(86 * min(sx, sy))
    if title:
        p.title_vertical(title, X(150), Y(120), size)
    if seal:
        p.seal(seal, X(150) + size * 1.25, Y(150), int(size * 0.47))
    return _finish(p, with_stages)

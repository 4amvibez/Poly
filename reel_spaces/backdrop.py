"""backdrop — the painted surroundings the explanation scenes sit in, so no page is bare text: misty mountains low in
the frame, a blossom branch reaching in from a top corner, a far pagoda, and (optionally) a small reader at a desk.
Cached per variant; returns density layers {'land', 'deco', 'bl'} to ink with INK / INK / SAKURA."""
import numpy as np

from art import blossom_cluster, cherry_branch, figure, pagoda
from engine import H, W, StrokeSet, cached


def backdrop(seed=0, branch='right', pagoda_x=None, reader=False, ground=1330):
    def build():
        from inkpaint import Painting
        p = Painting(W, H, seed=seed)
        r1 = p.ridge(ground - 50, peaks=[(160, 120, 170), (620, 80, 230), (980, 140, 160)], rough=10, seed=seed + 1)
        p.wash(r1, dens=0.13, decay=90, tex=70, edge=0.25, mist_y=ground, mist_h=40, seed=seed + 2)
        r2 = p.ridge(ground + 30, peaks=[(300, 50, 200), (850, 70, 220)], rough=8, seed=seed + 3)
        p.wash(r2, dens=0.2, decay=60, tex=60, edge=0.3, mist_y=ground + 70, mist_h=30, seed=seed + 4)
        land = p.D.copy()
        land[ground + 140:] *= np.linspace(1, 0, H - ground - 140)[:, None] ** 2
        ss = StrokeSet(W, H)
        if pagoda_x is not None:
            pagoda(ss, pagoda_x, ground - 10, 0.75, 0, 0.01, seed=seed + 5, ink=0.6)
        if branch == 'right':
            pts = [(1130, 120), (1020, 172), (920, 158), (836, 222), (760, 210)]
        else:
            pts = [(-50, 130), (60, 180), (160, 166), (246, 228), (320, 214)]
        tips = cherry_branch(ss, pts, 16, 0, 0.01, seed=seed + 6)
        if reader:
            figure(ss, 150, ground + 60, 1.5, 0, seed=seed + 7, pose='sit', facing=1, dur=0.01)
        bl = np.zeros((H, W), np.float32)
        for k, (bx, by) in enumerate(tips):
            bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 10, 36, seed=seed + 10 + k))
        return {'land': land.astype(np.float32), 'deco': ss.render(9).astype(np.float32), 'bl': bl.astype(np.float32)}
    return cached(f'backdrop_{seed}_{branch}_{pagoda_x}_{int(reader)}_{ground}', build)


def draw_backdrop(scene, fr, cam, B, op=1.0):
    from engine import INK, SAKURA
    scene.layer(fr, cam, B['land'], opacity=op, depth=0.92)
    scene.layer(fr, cam, B['deco'], opacity=op * 0.9)
    scene.layer(fr, cam, B['bl'], color=SAKURA, opacity=op * 0.9)

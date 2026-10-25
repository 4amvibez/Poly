"""Brush-writing scenes: calligraphy and big expressive brushwork, written stroke by stroke."""
import numpy as np

from .brush import BLUNT, HOOK, PRESS_DOWN, RISE, InkScene, brush_stroke, splatter, sweep, wash

NA = ((0, 0.4), (0.2, 0.75), (0.85, 1.15), (1, 0.15))   # 捺: thin entry, swelling to a heavy foot, then out


def water_swordsman():
    """水 — the character for water written large, a towering dry-brush wave crashing across the paper with
    flung ink, and a swordsman in a straw hat facing it, a red cord at his wrist. Returns an InkScene:
    .render() for the finished painting, .render(at=ms) for any moment of it being painted."""
    W, H = 1920, 1080
    s = InkScene(W, H, seed=2101)
    p = s.p
    B = lambda pts, width, **k: brush_stroke(W, H, pts, width, **k)

    # ---- faint distant mountains, sinking into mist (faded in, not written)
    far = p.ridge(800, peaks=[(260, 110, 260), (760, 70, 220), (1350, 60, 300), (1800, 90, 240)], rough=18, seed=4)
    p.wash(far, dens=0.1, decay=90, tex=120, edge=0.25, mist_y=860, mist_h=90, seed=5)
    s.add_background(200, 700)

    # ---- 水, stroke by stroke in the proper order: vertical hook, left 横撇, right 撇, right 捺
    C = lambda pts: [(80 + 0.95 * x, 10 + 0.95 * y) for x, y in pts]
    t = 1000
    for pts, width, prof, dry, dur in [
            (C([(470, 95), (468, 250), (463, 430), (460, 540), (415, 505)]), 54, HOOK, 0.45, 520),
            (C([(270, 250), (330, 240), (395, 236), (330, 370), (250, 500)]), 44, PRESS_DOWN, 0.55, 480),
            (C([(640, 190), (590, 260), (515, 330)]), 38, PRESS_DOWN, 0.4, 300),
            (C([(512, 330), (570, 420), (640, 500), (705, 560)]), 48, NA, 0.6, 480)]:
        t = s.add(B(pts, width, dry=dry, profile=prof, seed=t), t, dur) + 140
    s.add(splatter(W, H, 755, 545, (40, 18), n=8, size=(1.2, 3.5), seed=31, direction=0.6), t - 200, 120)

    # ---- the swordsman, bottom left: straw hat, robe, sword arm with a red cord, ribbons in the wind
    t += 200
    t = s.add(wash(W, H, [(300, 688), (166, 770), (300, 762), (440, 776)], ink=0.42, soft=1.5, seed=1), t, 240)
    t = s.add([B([(300, 690), (236, 728), (168, 770)], 9, dry=0.2, profile=BLUNT, seed=2),
               B([(300, 690), (370, 732), (440, 776)], 9, dry=0.2, profile=BLUNT, seed=3)], t, 160)
    t = s.add(B([(160, 772), (230, 764), (300, 760), (372, 765), (446, 780)], 15, dry=0.3, profile=BLUNT, seed=4), t, 200)
    for k, (a, b) in enumerate([((300, 694), (250, 760)), ((300, 694), (350, 762)), ((300, 694), (300, 760))]):
        s.add(B([a, b], 3, ink=0.45, dry=0.3, profile=PRESS_DOWN, seed=10 + k), t, 120)
    t += 120
    t = s.add([B([(226, 788), (160, 806), (110, 800), (60, 784)], 7, dry=0.35, profile=PRESS_DOWN, seed=5),
               B([(232, 796), (170, 830), (110, 846), (52, 842)], 6, dry=0.4, profile=PRESS_DOWN, seed=6)], t, 220)
    robe = [(262, 782), (338, 782), (382, 806), (398, 900), (420, 1080), (176, 1080), (200, 900), (216, 806)]
    t = s.add([wash(W, H, robe, ink=0.28, soft=2.5, seed=7),
               wash(W, H, [(270, 776), (330, 776), (322, 792), (278, 792)], ink=0.5, soft=3, seed=8)], t, 300)
    t = s.add([B([(258, 784), (218, 806), (204, 900), (190, 1000), (178, 1080)], 11, dry=0.35, profile=BLUNT, seed=9),
               B([(340, 784), (382, 808), (398, 900), (412, 1000), (420, 1080)], 9, dry=0.45, profile=BLUNT, seed=10),
               B([(270, 790), (296, 846), (304, 960)], 6, ink=0.7, dry=0.5, profile=PRESS_DOWN, seed=11),
               B([(330, 800), (318, 920), (340, 1080)], 26, ink=0.22, dry=0.35, profile=BLUNT, seed=12),
               B([(240, 980), (236, 1080)], 18, ink=0.2, dry=0.4, profile=BLUNT, seed=22)], t, 300)
    t = s.add(B([(198, 962), (300, 952), (406, 960)], 13, dry=0.3, profile=BLUNT, seed=13), t, 140)
    t = s.add([wash(W, H, [(372, 800), (470, 818), (466, 850), (380, 846)], ink=0.32, soft=1.5, seed=14),
               B([(372, 800), (420, 806), (470, 818)], 6, dry=0.3, profile=BLUNT, seed=15),
               B([(380, 846), (430, 850), (466, 850)], 6, dry=0.35, profile=BLUNT, seed=16)], t, 200)
    t = s.add([B([(478, 816), (476, 834), (480, 852)], 16, ink=0.95, dry=0.2, profile=BLUNT, seed=17, red=True),
               B([(474, 852), (462, 872), (470, 890)], 5, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=18, red=True)], t, 160)
    t = s.add(B([(486, 822), (500, 820)], 14, dry=0.1, profile=BLUNT, seed=19), t, 80)
    t = s.add([B([(500, 806), (508, 838)], 6, dry=0.1, profile=BLUNT, seed=20),
               B([(508, 822), (640, 800), (800, 772)], 5, ink=0.98, dry=0.15, profile=((0, 1), (0.85, 0.9), (1, 0.2)),
                 seed=21)], t, 220)

    # ---- the wave: the sword points, and the brush crashes across the paper
    t += 250
    # the face: heavy strokes surging up from the bottom, running dry as they rise and curl over at the crest
    face = [
        ([(700, 1075), (880, 830), (1040, 500), (1150, 250), (1210, 130), (1170, 70), (1080, 80), (1020, 140)], 130, 0.85, 0.6),
        ([(860, 1080), (1010, 840), (1130, 540), (1210, 290), (1250, 160), (1220, 100)], 105, 0.9, 0.62),
        ([(1050, 1080), (1160, 840), (1240, 570), (1280, 330), (1280, 200)], 85, 0.8, 0.66),
        ([(640, 980), (820, 760), (970, 480), (1080, 270)], 60, 0.6, 0.62),
    ]
    for k, (pts, width, ink, dry) in enumerate(face):
        s.add(sweep(W, H, pts, width, ink=ink, dry=dry, seed=100 + 7 * k, wash=0.3, profile=RISE), t + 110 * k, 420)
    t += 110 * len(face) + 200
    # the back of the wave running down to the right, and the lip curling over
    for k, (pts, width, ink, dry) in enumerate([
            ([(1230, 130), (1400, 300), (1580, 560), (1740, 820), (1870, 1030)], 100, 0.75, 0.6),
            ([(1290, 230), (1470, 440), (1640, 710), (1780, 980)], 66, 0.6, 0.66),
            ([(1310, 220), (1290, 120), (1220, 55), (1120, 45), (1040, 90), (1000, 160), (1020, 210)], 54, 0.9, 0.5)]):
        s.add(sweep(W, H, pts, width, ink=ink, dry=dry, seed=150 + 7 * k, wash=0.25), t + 100 * k, 360)
    s.add(splatter(W, H, 1150, 70, (200, 50), n=55, seed=41), t + 150, 260)
    t += 520
    # the horizontal sweeps cutting through it, with ink thrown off their ends
    for k, (pts, width, ink, dry) in enumerate([
            ([(770, 420), (1000, 450), (1350, 520), (1920, 590)], 115, 0.95, 0.62),
            ([(790, 320), (1100, 292), (1500, 300), (1920, 330)], 58, 0.55, 0.66),
            ([(800, 640), (1200, 660), (1600, 700), (1920, 730)], 44, 0.7, 0.7)]):
        s.add(sweep(W, H, pts, width, ink=ink, dry=dry, seed=200 + 7 * k, wash=0.22), t + 120 * k, 320)
    s.add(splatter(W, H, 1640, 520, (260, 130), n=55, seed=42, direction=0.2), t + 200, 260)
    s.add(splatter(W, H, 1420, 780, (200, 90), n=25, seed=43), t + 300, 200)
    # a curl of spray flung off the left of the crest
    t += 400
    s.add([B([(1000, 212), (950, 196), (900, 200), (870, 222), (884, 240)], 9, ink=0.85, dry=0.55, profile=PRESS_DOWN, seed=300),
           B([(990, 236), (948, 238), (930, 256)], 6, ink=0.75, dry=0.6, profile=PRESS_DOWN, seed=301),
           splatter(W, H, 880, 215, (60, 30), n=12, size=(1.2, 3.5), seed=44, direction=3.3)], t, 240)
    t += 450
    s.seal('水墨', 250, 440, 30, start=t)
    return s

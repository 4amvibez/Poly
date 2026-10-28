"""r5art — the paintings of this reel, each painted once (and cached by the scene that uses it): the straw-hat
traveller, an ink map of East Asia, a junk with vermilion sails, Mount Fuji over a lake, a misty sun landscape, a
river winding down out of the hills, a calendar page, a group of travellers under a red circle, and the final
landscape where all four kanji become one picture.  Everything returns float32 densities (ink unless named)."""
import cv2
import numpy as np

from art import blossom_cluster, cherry_branch, pagoda, sun_disc
from engine import H, W, StrokeSet, blur, noise, spline, stroke
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash


def _f(a):
    return np.clip(np.asarray(a, np.float32), 0, 1)


def ronin(ss, fx, fy, k=1.0, seed=0, hat=True, staff=True, ink=0.92):
    """a traveller standing, seen side-on: robe with sleeves, head, a straw kasa hat and a staff; feet at (fx, fy),
    about 200*k px tall"""
    w, h = ss.w, ss.h
    P = lambda pts: [(fx + x * k, fy + y * k) for x, y in pts]
    ss.add(pwash(w, h, P([(-14, -150), (14, -150), (30, -120), (26, -100), (20, -90), (30, -40), (40, 0), (-38, 0), (-28, -40),
                          (-20, -90), (-27, -104), (-30, -122)]), ink=ink, soft=1.6 * k, seed=seed, grain=0.3), 0, 1)
    ss.add(stroke(w, h, spline(P([(-4, -86), (-6, -40), (-10, -2)]), 6), 1.6 * k, ink=0.35, dry=0.6, profile=SWEEP, seed=seed + 1), 0, 1)
    ss.add(pwash(w, h, [(fx + 10 * k * np.cos(a), fy - 162 * k + 11 * k * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)],
                 ink=min(1, ink + 0.03), soft=k, seed=seed + 2), 0, 1)
    if hat:
        ss.add(pwash(w, h, P([(-40, -164), (0, -196), (40, -164), (0, -170)]), ink=min(1, ink + 0.05), soft=1.2 * k, seed=seed + 3), 0, 1)
    if staff:
        ss.add(stroke(w, h, P([(26, -128), (40, 6)]), 4 * k, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=seed + 4), 0, 1)


# ------------------------------------------------------------------ scene 2: the map and the ship
def east_asia_map():
    """an old ink map on a parchment sheet: the Chinese coast and the Korean peninsula on the left, the Japanese islands
    curving up on the right, the sea between with a few wave marks.  Returns sheet (fill), ink, land (wash), and the
    coordinates the scene needs (ports)."""
    r = np.random.default_rng(3)
    sheet = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = 60, 230, 1020, 880
    sheet[y0:y1, x0:x1] = 1
    sheet = blur(sheet, 3) * (0.85 + 0.15 * noise(H, W, 40, 3, 4))
    ss = StrokeSet(W, H)
    land = np.zeros((H, W), np.float32)

    def coast(pts, seed, width=4.0, fill=True):
        p = np.array(spline(pts, 10))
        p = p + r.normal(0, 2.5, p.shape)
        ss.add(stroke(W, H, [tuple(q) for q in p], width, ink=0.85, dry=0.45, profile=((0, 0.4), (0.05, 1), (0.95, 1), (1, 0.4)), seed=seed), 0, 1)
        if fill:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [p.astype(np.int32)], 1)
            return m.astype(np.float32)
        return None

    # the mainland: from the top-left edge down the coast, round the Shandong bulge, down to the bottom edge
    main = coast([(0, 300), (230, 290), (300, 330), (270, 400), (330, 430), (300, 480), (250, 520), (270, 600), (230, 680),
                  (200, 760), (150, 840), (120, 900), (0, 900), (0, 300)], 10)
    # the Korean peninsula
    kor = coast([(300, 330), (380, 320), (430, 360), (440, 430), (420, 500), (380, 540), (350, 520), (360, 460), (330, 430), (300, 330)], 11)
    # Japan: Hokkaido, Honshu's arc, Shikoku, Kyushu
    hok = coast([(820, 260), (880, 255), (930, 300), (900, 360), (850, 370), (810, 330), (820, 260)], 12)
    hon = coast([(840, 390), (870, 400), (880, 460), (860, 540), (820, 600), (760, 640), (690, 670), (620, 690), (570, 700), (560, 680),
                 (610, 660), (670, 630), (730, 590), (780, 540), (810, 480), (820, 420), (840, 390)], 13)
    shi = coast([(600, 715), (660, 705), (680, 730), (630, 745), (600, 715)], 14)
    kyu = coast([(520, 700), (555, 705), (565, 750), (545, 800), (515, 790), (505, 740), (520, 700)], 15)
    for m, d in ((main, 0.22), (kor, 0.22), (hok, 0.26), (hon, 0.26), (shi, 0.26), (kyu, 0.26)):
        land = np.maximum(land, blur(m, 2) * d)
    land *= 0.7 + 0.3 * noise(H, W, 30, 3, 16)
    # wave marks in the sea
    for i in range(14):
        cx, cy = r.uniform(380, 560), r.uniform(560, 860)
        if cx < 250 or (cx > 500 and cy < 700):
            continue
        ss.add(stroke(W, H, spline([(cx - 16, cy), (cx - 6, cy - 6), (cx + 4, cy), (cx + 14, cy - 6)], 6), 2, ink=0.45, dry=0.5,
                      profile=SWEEP, seed=30 + i), 0, 1)
    # a thin double frame round the sheet
    inside = np.zeros((H, W), np.float32)
    inside[y0 + 26:y1 - 26, x0 + 26:x1 - 26] = 1
    inside = blur(inside, 1.5)
    ink = ss.render(9) * inside
    fr = StrokeSet(W, H)
    for d, wd in ((14, 3), (24, 1.5)):
        corners = [(x0 + d, y0 + d), (x1 - d, y0 + d), (x1 - d, y1 - d), (x0 + d, y1 - d)]
        for j in range(4):
            fr.add(stroke(W, H, [corners[j], corners[(j + 1) % 4]], wd, ink=0.6, dry=0.4, profile=((0, 1), (1, 1)), seed=50 + d + j), 0, 1)
    ink = np.maximum(ink, fr.render(9))
    return {'sheet': _f(sheet), 'ink': _f(ink), 'land': _f(land * inside),
            'china_port': np.array([300, 520], np.float32), 'japan_port': np.array([555, 712], np.float32)}


def junk(w=260, h=240):
    """a junk under sail: a curved dark hull, three masts, ribbed vermilion sails; returns (ink, red)"""
    ss, red = StrokeSet(w, h), StrokeSet(w, h)
    hull = [(20, 170), (60, 196), (200, 196), (245, 160), (230, 175), (200, 182), (60, 182), (30, 165)]
    ss.add(pwash(w, h, [(18, 160), (245, 152), (210, 200), (55, 204)], ink=0.9, soft=1.5, seed=1, grain=0.3), 0, 1)
    ss.add(stroke(w, h, spline([(10, 150), (70, 166), (180, 164), (250, 140)], 8), 4, ink=0.95, dry=0.3, profile=PRESS_DOWN, seed=2), 0, 1)
    for k, (mx, top, sw) in enumerate(((70, 40, 46), (135, 14, 56), (195, 54, 40))):
        ss.add(stroke(w, h, [(mx, 160), (mx + 2, top)], 3, ink=0.9, dry=0.3, profile=PRESS_DOWN, seed=10 + k), 0, 1)
        sail = [(mx - sw, top + 12), (mx + 4, top + 4), (mx + 8, 146), (mx - sw - 10, 140)]
        red.add(pwash(w, h, sail, ink=0.85, soft=1.6, seed=20 + k, grain=0.5), 0, 1)
        for j in range(5):
            y = top + 20 + j * (126 - top) / 5
            ss.add(stroke(w, h, [(mx - sw - 4 - j * 1.5, y), (mx + 6, y - 4)], 1.5, ink=0.55, dry=0.5, profile=SWEEP, seed=30 + k * 5 + j), 0, 1)
    for j in range(4):
        ss.add(stroke(w, h, spline([(0 + j * 60, 214), (25 + j * 60, 208), (50 + j * 60, 214)], 6), 2, ink=0.5, dry=0.5, profile=SWEEP,
                      seed=60 + j), 0, 1)
    return _f(ss.render(9)), _f(red.render(9))


def manuscript(w=980, h=440):
    """an open book seen from above, two pages with a spine and a few faint ruled columns; returns (fill, ink)"""
    fill, ss = StrokeSet(w, h), StrokeSet(w, h)
    left = [(30, 50), (300, 24), (488, 46), (490, 410), (300, 392), (30, 414)]
    right = [(492, 46), (680, 24), (950, 50), (950, 414), (680, 392), (490, 410)]
    for k, poly in enumerate((left, right)):
        fill.add(pwash(w, h, poly, ink=1.0, soft=1.5, seed=k), 0, 1)
        ss.add(pwash(w, h, poly, ink=0.06, soft=2, seed=10 + k, grain=0.6), 0, 1)
    for k, pts in enumerate(([(30, 50), (300, 24), (488, 46)], [(492, 46), (680, 24), (950, 50)],
                             [(30, 414), (300, 392), (490, 412), (680, 392), (950, 414)], [(30, 50), (30, 414)], [(950, 50), (950, 414)])):
        ss.add(stroke(w, h, spline(pts, 10) if len(pts) > 2 else pts, 3.5, ink=0.8, dry=0.4, profile=PRESS_DOWN, seed=20 + k), 0, 1)
    ss.add(stroke(w, h, [(490, 46), (490, 412)], 3, ink=0.55, dry=0.5, profile=SWEEP, seed=30), 0, 1)
    return _f(fill.render(9)), _f(ss.render(9))


# ------------------------------------------------------------------ scene 3: the mountain and Mount Fuji
def mountain_scene():
    """a tall mountain with its neighbours rising out of mist, pines on a near rock (the 山 painting)"""
    p = Painting(W, H, seed=21)
    r0 = p.ridge(820, peaks=[(150, 160, 130), (930, 200, 150)], rough=10, seed=22)
    p.wash(r0, dens=0.16, decay=110, tex=80, edge=0.25, mist_y=860, mist_h=50, seed=23)
    r1 = p.peak(560, 330, height=600, left_w=170, right_w=210, power=1.25, shoulders=[(380, 120, 90), (760, 90, 100)], rough=14, seed=24)
    p.wash(r1, dens=0.55, decay=200, tex=60, edge=0.5, mist_y=900, mist_h=60, seed=25)
    p.cun(r1, 380, 760, n=110, shadow_x=520, spread=60, max_y=860, ink=0.22)
    r2 = p.ridge(1080, peaks=[(160, 130, 160), (980, 100, 150)], rough=10, seed=26)
    p.wash(r2, dens=0.8, decay=90, tex=40, edge=0.5, seed=27)
    p.pine((150, 960), (175, 820), bend=20, branches=[((165, 870), (230, 845), 10)], trunk_w=9)
    p.pine((960, 990), (930, 860), bend=-18, trunk_w=8)
    d = p.D.copy()
    d[1140:] *= np.linspace(1, 0, H - 1140)[:, None] ** 2
    return {'ink': _f(d)}


def fuji_scene():
    """Mount Fuji: a broad snow-capped cone with concave slopes under a vermilion sun, a lake with its reflection,
    a pagoda on the left shore and pines on the right"""
    p = Painting(W, H, seed=31)
    xs = np.arange(W, dtype=np.float32)
    d = np.clip(np.abs(xs - 540) - 46, 0, None)
    ridge = 470 + 520 * (1 - np.exp(-d / 300)) + 4 * (noise(1, W, 40, 2, 32)[0] - 0.5)
    depth = p.YY - ridge[None]
    body = np.where(depth > 0, np.exp(-depth / 420), 0) * (depth > 0)
    fingers = 70 * np.abs(np.sin(xs / 23 + 1.3)) * noise(1, W, 20, 3, 39)[0]
    snow_line = np.clip(170 + 50 * noise(1, W, 30, 3, 33)[0] + fingers - 0.45 * d, 0, None)
    snow = np.clip((snow_line[None] - depth) / 10, 0, 1) * (depth > 0)
    streak = 0.6 + 0.4 * np.abs(np.sin(p.XX / 13 + noise(H, W, 30, 2, 34) * 3))
    mass = (depth > 0) * (0.35 + 0.65 * body) * (1 - 0.97 * snow) * (0.55 + 0.45 * noise(H, W, 60, 4, 35)) * 0.7
    rim = np.clip(blur((depth > 0).astype(np.float32), 2) - blur((depth > 0).astype(np.float32), 7), 0, None) * 0.7
    mass = np.maximum(mass, rim) + snow * (depth > 0) * 0.08 * streak
    mass[1000:] = 0
    p.D = np.maximum(p.D, mass.astype(np.float32))
    r2 = p.ridge(1000, peaks=[(120, 60, 120), (980, 70, 140)], rough=8, seed=36)
    p.wash(r2, dens=0.7, decay=40, tex=40, edge=0.5, seed=37)
    land = p.D.copy()
    land[1010:] = 0
    # the lake: the reflection, broken by ripples
    refl = np.flipud(land[1000 - 520:1000])
    lake = np.zeros((H, W), np.float32)
    lake[1010:1010 + 520] = blur(refl[:520], 2) * 0.22
    rip = 0.65 + 0.35 * np.sin(p.YY / 2.2 + noise(H, W, 40, 2, 38) * 3)
    lake *= rip * np.clip((1500 - p.YY) / 400, 0, 1)
    ss = StrokeSet(W, H)
    for i in range(10):
        y = 1030 + i * 30
        x = 200 + (i * 137) % 600
        ss.add(stroke(W, H, [(x, y), (x + 160 - i * 6, y + 1)], 2.2, ink=0.4, dry=0.6, profile=SWEEP, seed=40 + i), 0, 1)
    pagoda(ss, 150, 1000, 0.75, 0, 0.01, seed=41, ink=0.9)
    pp = Painting(W, H, seed=42)
    pp.pine((975, 1040), (955, 880), bend=-16, trunk_w=8)
    sun = sun_disc(H, W, 790, 360, 118, seed=43)
    sun_refl = np.zeros((H, W), np.float32)
    for i in range(7):
        y = 1040 + i * 24
        wd = 90 - i * 10
        sun_refl[y - 4:y + 4, 790 - wd:790 + wd] = 0.7 * (1 - i / 8)
    sun_refl = blur(sun_refl, 2) * (0.6 + 0.4 * noise(H, W, 8, 2, 44))
    ink = np.maximum(np.maximum(land, lake), np.maximum(ss.render(9), pp.D))
    ink[1500:] *= np.linspace(1, 0, H - 1500)[:, None] ** 2
    return {'ink': _f(ink), 'sun': _f(sun), 'sun_refl': _f(sun_refl)}


# ------------------------------------------------------------------ scene 4: the sun and the calendar
def sun_landscape():
    """low misty hills and a still lake for the sun to rise over (the sun itself is placed by the scene)"""
    p = Painting(W, H, seed=51)
    r0 = p.ridge(930, peaks=[(200, 120, 160), (520, 60, 200), (880, 150, 170)], rough=10, seed=52)
    p.wash(r0, dens=0.2, decay=90, tex=80, edge=0.25, mist_y=960, mist_h=40, seed=53)
    r1 = p.ridge(1050, peaks=[(80, 80, 140), (700, 70, 220)], rough=8, seed=54)
    p.wash(r1, dens=0.5, decay=60, tex=50, edge=0.4, seed=55)
    p.birds([(300, 520, 1.3), (345, 500, 1.0), (380, 530, 0.9)])
    d = p.D.copy()
    ss = StrokeSet(W, H)
    for i in range(8):
        y = 1090 + i * 26
        x = 260 + (i * 173) % 460
        ss.add(stroke(W, H, [(x, y), (x + 200 - i * 10, y + 1)], 2, ink=0.35, dry=0.6, profile=SWEEP, seed=60 + i), 0, 1)
    d = np.maximum(d, ss.render(9))
    d[1320:] *= np.linspace(1, 0, H - 1320)[:, None] ** 2
    return {'ink': _f(d)}


def calendar(w=720, h=560, seed=0):
    """a hanging calendar page: rods top and bottom, the seven day kanji 日 月 火 水 木 金 土 across the top, a grid of
    dates below (Sundays in vermilion).  Returns fill, ink, red, and the centres of the header cells"""
    from engine import FONT_EN_BOLD, FONT_JP, text_layer
    fill, ss = StrokeSet(w, h), StrokeSet(w, h)
    x0, y0, x1, y1 = 40, 40, w - 40, h - 40
    page = [(x0, y0), (x1, y0), (x1 + 3, y1), (x0 - 2, y1)]
    fill.add(pwash(w, h, page, ink=1.0, soft=1.5, seed=seed), 0, 1)
    ss.add(pwash(w, h, page, ink=0.05, soft=2, seed=seed + 1, grain=0.6), 0, 1)
    for k, y in enumerate((y0, y1)):
        ss.add(stroke(w, h, [(x0 - 22, y), (x1 + 22, y)], 14, ink=0.9, dry=0.3, profile=((0, 1), (1, 1)), seed=seed + 2 + k), 0, 1)
    cw = (x1 - x0) / 7
    hy = y0 + 70
    ink = ss.render(9)
    red = np.zeros((h, w), np.float32)
    cells = []
    for i, c in enumerate('日月火水木金土'):
        g, _ = text_layer(c, FONT_JP, 52)
        cx = x0 + cw * (i + 0.5)
        cells.append((cx, hy))
        tgt = red if i == 0 else ink
        gh, gw = g.shape
        ys, xs = int(hy - gh / 2), int(cx - gw / 2)
        tgt[ys:ys + gh, xs:xs + gw] = np.maximum(tgt[ys:ys + gh, xs:xs + gw], g * 0.95)
    grid = StrokeSet(w, h)
    for j in range(6):
        y = hy + 48 + j * (y1 - hy - 70) / 5
        grid.add(stroke(w, h, [(x0 + 12, y), (x1 - 12, y)], 1.6, ink=0.45, dry=0.5, profile=((0, 1), (1, 1)), seed=seed + 20 + j), 0, 1)
    for i in range(1, 7):
        x = x0 + cw * i
        grid.add(stroke(w, h, [(x, hy + 48), (x, y1 - 22)], 1.4, ink=0.35, dry=0.5, profile=((0, 1), (1, 1)), seed=seed + 30 + i), 0, 1)
    ink = np.maximum(ink, grid.render(9))
    day = 1
    for j in range(5):
        for i in range(7):
            if j == 0 and i < 3 or day > 31:
                continue
            g, _ = text_layer(str(day), FONT_EN_BOLD, 26)
            gh, gw = g.shape
            cx, cy = x0 + cw * i + 22, hy + 48 + j * (y1 - hy - 70) / 5 + 22
            tgt = red if i == 0 else ink
            ys, xs = int(cy - gh / 2), int(cx - gw / 2)
            tgt[ys:ys + gh, xs:xs + gw] = np.maximum(tgt[ys:ys + gh, xs:xs + gw], g * 0.7)
            day += 1
    return {'fill': _f(fill.render(9)), 'ink': _f(ink), 'red': _f(red), 'cells': cells, 'size': (w, h)}


# ------------------------------------------------------------------ scene 5: the river
def river_scene():
    """a river winding down out of far hills, widening towards us between rocky banks with pines; the current lines
    are returned separately so the scene can make them flow"""
    p = Painting(W, H, seed=61)
    r0 = p.ridge(560, peaks=[(200, 160, 160), (840, 200, 160)], rough=10, seed=62)
    p.wash(r0, dens=0.22, decay=80, tex=80, edge=0.3, mist_y=600, mist_h=40, seed=63)
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None]
    centre = 540 + 120 * np.sin(ys / 260 + 0.8) * np.clip((ys - 500) / 300, 0, 1)
    half = 14 + 330 * np.clip((ys - 560) / (1500 - 560), 0, 1) ** 1.3
    dist = np.abs(xs - centre) - half
    top = np.clip((ys - 560) / 80, 0, 1)
    bank = np.clip(dist / 8, 0, 1) * top * np.exp(-np.clip(dist, 0, None) / 120) * (0.5 + 0.45 * noise(H, W, 40, 4, 64))
    p.D = np.maximum(p.D * (1 - (dist <= 0) * top), bank.astype(np.float32))
    p.pine((110, 1120), (150, 880), bend=24, branches=[((130, 960), (230, 920), 12)], trunk_w=12)
    p.pine((990, 1180), (950, 960), bend=-20, trunk_w=10)
    rocks = StrokeSet(W, H)
    r = np.random.default_rng(65)
    for j in range(9):
        y = 700 + j * 85 + r.uniform(-20, 20)
        side = 1 if j % 2 else -1
        c = 540 + 120 * np.sin(y / 260 + 0.8) * min(1, (y - 500) / 300) + side * (14 + 330 * min(1, (y - 560) / 940) ** 1.3 + r.uniform(-8, 26))
        rw, rh = r.uniform(26, 60), r.uniform(12, 24)
        poly = [(c + rw * np.cos(a) * (1 + 0.2 * np.sin(3 * a + j)), y + rh * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        rocks.add(pwash(W, H, poly, ink=0.75, soft=2.5, seed=66 + j, grain=0.5), 0, 1)
    ink = np.maximum(p.D, rocks.render(9))
    ink[1500:] *= np.linspace(1, 0, H - 1500)[:, None] ** 2
    flow = StrokeSet(W, H)
    for i in range(14):
        off = (i - 6.5) / 6.5
        pts = [(540 + 120 * np.sin(y / 260 + 0.8) * min(1, max(0, (y - 500) / 300)) + off * (10 + 300 * min(1, max(0, (y - 560) / 940)) ** 1.3), y)
               for y in np.linspace(600, 1560, 22)]
        flow.add(stroke(W, H, spline(pts, 6), 2.4, ink=0.42, dry=0.75, profile=SWEEP, seed=80 + i), 0, 1)
    water = np.clip(-dist / 20, 0, 1) * top * (0.45 + 0.55 * noise(H, W, 60, 3, 90))
    water[1500:] *= np.linspace(1, 0, H - 1500)[:, None] ** 2
    return {'ink': _f(ink), 'flow': _f(flow.render(9)), 'water': _f(water)}


# ------------------------------------------------------------------ scene 6: the people
def person_scene():
    """a lone traveller on a ridge, distant peaks behind"""
    p = Painting(W, H, seed=71)
    r0 = p.ridge(840, peaks=[(760, 220, 170), (300, 120, 150)], rough=10, seed=72)
    p.wash(r0, dens=0.18, decay=90, tex=80, edge=0.25, mist_y=880, mist_h=40, seed=73)
    r1 = p.ridge(1120, peaks=[(380, 170, 260), (980, 60, 140)], rough=10, seed=74)
    p.wash(r1, dens=0.8, decay=80, tex=40, edge=0.5, seed=75)
    d = p.D.copy()
    d[1250:] *= np.linspace(1, 0, H - 1250)[:, None] ** 2
    ss = StrokeSet(W, H)
    ronin(ss, 380, 958, k=1.7, seed=76)
    return {'land': _f(d), 'fig': _f(ss.render(9))}


def group_scene():
    """five travellers standing together on a shore under a large vermilion circle, a pagoda and a cherry tree"""
    p = Painting(W, H, seed=81)
    r0 = p.ridge(1000, peaks=[(150, 90, 140), (900, 120, 160)], rough=8, seed=82)
    p.wash(r0, dens=0.2, decay=80, tex=80, edge=0.25, mist_y=1020, mist_h=30, seed=83)
    r1 = p.ridge(1150, peaks=[(540, 40, 400)], rough=6, seed=84)
    p.wash(r1, dens=0.75, decay=60, tex=40, edge=0.5, seed=85)
    land = p.D.copy()
    land[1260:] *= np.linspace(1, 0, H - 1260)[:, None] ** 2
    ss = StrokeSet(W, H)
    pagoda(ss, 860, 1010, 0.8, 0, 0.01, seed=86, ink=0.85)
    for k, (x, kk, hat) in enumerate(((330, 1.15, True), (430, 1.3, False), (530, 1.4, True), (640, 1.25, False), (735, 1.1, True))):
        ronin(ss, x, 1120 - 6 * abs(k - 2), k=kk, seed=90 + 7 * k, hat=hat, staff=k in (0, 4))
    br = StrokeSet(W, H)
    tips = cherry_branch(br, [(-20, 980), (90, 900), (160, 830), (230, 790)], 18, 0, 0.01, seed=95)
    tips += cherry_branch(br, [(90, 900), (40, 820), (60, 740)], 9, 0, 0.01, seed=96)
    bl = np.zeros((H, W), np.float32)
    for k, (bx, by) in enumerate(tips + [(150, 800), (200, 860), (110, 760)]):
        bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 9, 44, seed=100 + k))
    sun = sun_disc(H, W, 540, 700, 250, seed=97)
    return {'land': _f(land), 'figs': _f(ss.render(9)), 'branch': _f(br.render(9)), 'bloss': _f(bl), 'sun': _f(sun)}


# ------------------------------------------------------------------ scene 7: one landscape
def final_landscape():
    """the four kanji become one picture: a mountain (山) under a sun (日), a river (水) running down from it, and a
    traveller (人) walking beside the water"""
    p = Painting(W, H, seed=111)
    r0 = p.ridge(860, peaks=[(160, 120, 130), (900, 160, 150)], rough=10, seed=112)
    p.wash(r0, dens=0.16, decay=90, tex=80, edge=0.25, mist_y=890, mist_h=40, seed=113)
    r1 = p.peak(560, 470, height=480, left_w=170, right_w=220, power=1.25, shoulders=[(380, 100, 90)], rough=12, seed=114)
    p.wash(r1, dens=0.5, decay=180, tex=60, edge=0.5, mist_y=940, mist_h=50, seed=115)
    p.cun(r1, 400, 740, n=80, shadow_x=520, spread=50, max_y=900, ink=0.2)
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None]
    centre = 560 + 140 * np.sin(ys / 200 - 3.5) * np.clip((ys - 900) / 200, 0, 1)
    half = 8 + 200 * np.clip((ys - 930) / 600, 0, 1) ** 1.2
    dist = np.abs(xs - centre) - half
    top = np.clip((ys - 930) / 60, 0, 1)
    bank = np.clip(dist / 8, 0, 1) * top * np.exp(-np.clip(dist, 0, None) / 90) * (0.45 + 0.45 * noise(H, W, 40, 4, 116))
    p.D = np.maximum(p.D * (1 - (dist <= 0) * top), bank.astype(np.float32) * (ys > 920))
    water = np.clip(-dist / 16, 0, 1) * top * (0.45 + 0.55 * noise(H, W, 50, 3, 117))
    p.pine((900, 1260), (870, 1050), bend=-20, trunk_w=10)
    ink = p.D.copy()
    ss = StrokeSet(W, H)
    ronin(ss, 300, 1240, k=1.0, seed=118)
    ink = np.maximum(ink, ss.render(9))
    ink[1420:] *= np.linspace(1, 0, H - 1420)[:, None] ** 2
    water[1420:] *= np.linspace(1, 0, H - 1420)[:, None] ** 2
    sun = sun_disc(H, W, 760, 520, 105, seed=119)
    return {'ink': _f(ink), 'water': _f(water), 'sun': _f(sun)}


# ------------------------------------------------------------------ scene 2 (v2): a real chart of East Asia
CHART_K = 80.0                         # pixels per degree of longitude
CHART_LON0, CHART_LAT1 = 110.0, 47.0   # the chart's top-left corner


def _merc(lat):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def chart_xy(lon, lat):
    """chart pixel coordinates of a longitude / latitude (Mercator, so the islands keep their shapes)"""
    return np.array([(lon - CHART_LON0) * CHART_K, (_merc(CHART_LAT1) - _merc(lat)) * CHART_K], np.float32)


def east_asia_chart():
    """an old sea chart of East Asia from real coastlines (Natural Earth, public domain): parchment land with a
    shaded coast and an ink coastline, a teal sea with ripple lines following the coast, ink mountain marks, and
    the names CHINA 中国 / KOREA 朝鮮半島 / JAPAN 日本.  Returns the chart as RGB (float16) plus its size."""
    import json
    import os
    from engine import FONT_EN_BOLD, FONT_JP, HERE, INK, text_layer
    rings = json.load(open(os.path.join(HERE, 'data', 'east_asia_land.json')))['rings']
    w, h = int((148 - CHART_LON0) * CHART_K), int(chart_xy(110, 15)[1])     # open sea south of 22°N
    land = np.zeros((h, w), np.uint8)
    for ring in rings:
        pts = np.array([chart_xy(x, y) for x, y in ring], np.float32)
        cv2.fillPoly(land, [np.round(pts).astype(np.int32)], 1, lineType=cv2.LINE_AA)
    L = land.astype(np.float32)
    dsea = cv2.distanceTransform((1 - land).astype(np.uint8), cv2.DIST_L2, 5)
    dland = cv2.distanceTransform(land, cv2.DIST_L2, 5)
    rng = np.random.default_rng(5)
    nz = noise(h, w, 160, 4, 6)
    # sea: teal, lighter in a band along the coast, darker out to sea, with ripple lines following the coast
    sea_c = np.array([0.40, 0.56, 0.58], np.float32)
    deep_c = np.array([0.30, 0.45, 0.49], np.float32)
    coast_c = np.array([0.62, 0.72, 0.70], np.float32)
    s_mix = np.clip(dsea / 260, 0, 1)[..., None]
    sea = sea_c * (1 - s_mix) + deep_c * s_mix
    band = np.exp(-dsea / 14)[..., None]
    sea = sea * (1 - band * 0.6) + coast_c * band * 0.6
    sea *= (0.93 + 0.1 * nz)[..., None]
    rip = np.zeros((h, w), np.float32)
    for d0 in (22, 44):
        rip = np.maximum(rip, np.exp(-((dsea - d0) / 1.3) ** 2) * (dsea > 0))
    sea *= (1 - 0.22 * rip)[..., None]
    # land: parchment, a little darker just inside the coast
    land_c = np.array([0.90, 0.84, 0.70], np.float32)
    lnd = land_c * (0.95 + 0.07 * nz)[..., None] * (1 - 0.1 * np.exp(-dland / 10))[..., None]
    img = sea * (1 - L[..., None]) + lnd * L[..., None]
    # the coastline
    edge = np.clip(cv2.GaussianBlur(L, (0, 0), 0.9) * 2 - 0.0, 0, 1) - np.clip(cv2.GaussianBlur(L, (0, 0), 0.9) * 2 - 1, 0, 1)
    line = np.exp(-(dsea / 1.4) ** 2) * (dsea > 0) + np.exp(-(dland / 1.0) ** 2) * (dland > 0)
    line = np.clip(line, 0, 1) * 0.85
    ink = np.zeros((h, w), np.float32)
    ink = np.maximum(ink, line)
    # mountains: little ink peaks in clusters inland
    peaks = [(118.5, 39.5, 6), (113.5, 37.5, 7), (116.5, 35.6, 4), (117.5, 31.0, 6), (114.5, 27.5, 6), (119.0, 28.0, 5),
             (127.8, 40.2, 6), (128.3, 37.3, 5), (127.6, 35.6, 4), (126.8, 39.0, 3), (137.6, 36.2, 7), (139.5, 37.8, 5),
             (140.5, 39.5, 5), (135.8, 34.4, 3), (131.0, 32.8, 4), (133.3, 33.7, 3), (142.6, 43.6, 4), (132.5, 34.8, 3)]
    pk = StrokeSet(w, h)
    for j, (lo, la, n) in enumerate(peaks):
        cx, cy = chart_xy(lo, la)
        for i in range(n):
            x, y = cx + rng.normal(0, 34), cy + rng.normal(0, 22)
            if land[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] == 0:
                continue
            s = rng.uniform(11, 17)
            pk.add(stroke(w, h, [(x - s, y + s * 0.45), (x, y - s * 0.6), (x + s * 0.9, y + s * 0.45)], 1.5, ink=0.85, dry=0.25,
                          profile=((0, 0.5), (0.5, 1), (1, 0.5)), seed=200 + j * 10 + i), 0, 1)
            pk.add(stroke(w, h, [(x + s * 0.05, y - s * 0.35), (x + s * 0.35, y + s * 0.35)], 1.2, ink=0.45, dry=0.5,
                          profile=((0, 1), (1, 0.4)), seed=300 + j * 10 + i), 0, 1)
    ink = np.maximum(ink, pk.render(9))
    # wave marks out at sea
    wv = StrokeSet(w, h)
    for i in range(70):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        if dsea[int(y), int(x)] < 40:
            continue
        s = rng.uniform(14, 24)
        wv.add(stroke(w, h, spline([(x - s, y), (x - s * 0.4, y - s * 0.3), (x + s * 0.2, y), (x + s * 0.8, y - s * 0.3)], 6), 1.8, ink=0.5,
                      dry=0.5, profile=SWEEP, seed=400 + i), 0, 1)
    ink = np.maximum(ink, wv.render(9) * 0.6)
    img *= (1 - ink[..., None] * (1 - INK))
    # names
    for en, jp, lo, la, sz in (('CHINA', '中国', 115.8, 34.0, 56), ('KOREA', '朝鮮半島', 127.6, 38.6, 46), ('JAPAN', '日本', 136.2, 37.3, 50)):
        cx, cy = chart_xy(lo, la)
        for txt, font, size, dy, op in ((en, FONT_EN_BOLD, sz, 0, 0.8), (jp, FONT_JP, int(sz * 0.62), sz * 0.95, 0.7)):
            g, _ = text_layer(txt, font, size, tracking=8 if font == FONT_EN_BOLD else 0)
            gh, gw = g.shape
            y0, x0 = int(cy + dy - gh / 2), int(cx - gw / 2)
            sl = img[y0:y0 + gh, x0:x0 + gw]
            sl *= 1 - (g * op)[..., None] * (1 - INK)
    return {'rgb': img.astype(np.float16), 'land': land.astype(np.uint8)}

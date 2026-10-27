"""Scene 6 — the answer.  "So, is kanji really necessary?  Japanese can be written without kanji in some situations.
But kanji helps make many everyday sentences easier to distinguish and read.  Hiragana, katakana, and kanji each have
their own important role."
A calm wide view: misty mountains, a red sun, a pagoda, the reader sitting with their back to us.  The question is
brushed at the top.  "In some situations": an open picture book written all in hiragana (むかし むかし…), for little
children.  "Everyday sentences": three wooden signboards swing in, the kind you read at a glance (駅 station,
出口 exit, 病院 hospital); they knock against their posts in the breath after "read." (wood knock).  Then the three
scripts take their places as Raj names them, each with its job: ひらがな grammar & endings, カタカナ borrowed words,
漢字 meaning, joined by one vermilion stroke on "important role"."""
import numpy as np

import kit
from art import figure_back, pagoda, sun_disc
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, GREY, H, INK, SAKURA, VERM, W, Cam, StrokeSet, blur, cached,
                    ease_io, ease_out, keyframes, smooth, spline, stroke, text_layer)
from glyphs import typeset
from inkpaint import Painting
from poly_ink.brush import PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from sentence import BLUE, GOLD, PINK
from timeline import at, quiet_gap

SIGNS = [('駅', 'station', 210), ('出口', 'exit', 540), ('病院', 'hospital', 870)]
SCRIPTS = [('ひらがな', 'grammar & endings', PINK, 200), ('カタカナ', 'borrowed words', BLUE, 540), ('漢字', 'meaning', GOLD, 880)]
WOOD = np.array([0.42, 0.28, 0.16], np.float32)


class Scene(SceneBase):
    paper_seed = 86

    def build(self):
        def paint():
            p = Painting(W, H, seed=30)
            r1 = p.ridge(1210, peaks=[(150, 220, 200), (560, 300, 260), (960, 200, 200)], rough=12, seed=31)
            p.wash(r1, dens=0.15, decay=110, tex=80, edge=0.25, mist_y=1240, mist_h=50, seed=32)
            r2 = p.ridge(1330, peaks=[(320, 90, 260), (820, 120, 240)], rough=9, seed=33)
            p.wash(r2, dens=0.24, decay=70, tex=60, edge=0.3, mist_y=1380, mist_h=40, seed=34)
            ss = StrokeSet(W, H)
            pagoda(ss, 880, 1260, 0.85, 0, 0.01, seed=35, ink=0.75)
            figure_back(ss, 230, 1420, 1.3, 0, seed=36, dur=0.01)
            land = p.D.copy()
            land[1480:] *= np.linspace(1, 0, H - 1480)[:, None] ** 2
            return {'land': land.astype(np.float32), 'deco': ss.render(9).astype(np.float32),
                    'sun': sun_disc(H, W, 780, 1110, 95, seed=37).astype(np.float32), 'book': _book(), 'board': _board(),
                    'board_ink': _board(ink=True)}
        self.A = cached('s06_paint_v2', paint)
        self.q1, _ = text_layer('So, is kanji', FONT_HEAD, 84)
        self.q2, _ = text_layer('really necessary?', FONT_HEAD, 84)
        self.t_q = at('So is kanji')[0]
        self.t_book = at('can be written')[0]
        self.t_but = at('But kanji')[0]
        self.t_sign = [at('everyday')[0] - 0.1 + 0.25 * k for k in range(3)]
        c, n = quiet_gap(at('distinguish and read')[1] - 0.2, at('Hiragana katakana')[0] + 0.1)
        self.t_hit = c - n / 2 + 0.06                                  # sync: the signboards knock (+ wood)
        self.t_scr = [at('Hiragana katakana')[0], at('katakana and')[0], at('and kanji each')[0] + 0.2]
        self.t_role = at('important role')[0]
        self.book_text = [kit.Row(s, 34, seed=400 + k) for k, s in enumerate(('むかしむかし、', 'あるところに', 'おじいさんと', 'おばあさんが'))]
        self.book_lbl, _ = text_layer('picture books for little children', FONT_EN_ITALIC, 42)
        self.sign = [(kit.Row(k, 92 if len(k) == 1 else 78, seed=420 + i), text_layer(e, FONT_EN_ITALIC, 42)[0], x)
                     for i, (k, e, x) in enumerate(SIGNS)]
        self.scr = [(kit.Row(k, 70, font=__import__('engine').FONT_BRUSH_JP, seed=440 + i), text_layer(e, FONT_EN_ITALIC, 40)[0], c, x)
                    for i, (k, e, c, x) in enumerate(SCRIPTS)]
        yy, xx = np.mgrid[0:200, 0:300].astype(np.float32)
        self.pad = np.clip((1 - ((xx - 150) / 140) ** 2 - ((yy - 100) / 90) ** 2) * 6, 0, 1).astype(np.float32)
        tie = StrokeSet(W, H)
        tie.add(stroke(W, H, spline([(110, 960), (360, 990), (720, 985), (970, 955)], 16), 10, ink=0.95, dry=0.45,
                       profile=((0, 0.4), (0.1, 1), (0.9, 0.9), (1, 0.3)), seed=7, red=True), self.t_role, 0.5)
        self.tie = tie

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.04)), (self.t_book, (540, 960, 1.0)), (self.t1, (540, 950, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        A = self.A
        self.layer(fr, cam, A['sun'], color=VERM, opacity=0.85, depth=0.9)
        self.layer(fr, cam, A['land'], depth=0.95)
        self.layer(fr, cam, A['deco'])
        o = smooth((t - self.t_q) / 0.4)
        kit.put(fr, cam, self.q1, 540, 260, INK, o)
        kit.put(fr, cam, self.q2, 540, 360, INK, smooth((t - self.t_q - 0.3) / 0.4))
        # in some situations: a picture book, all hiragana
        fb = smooth((t - self.t_book) / 0.5) * (1 - smooth((t - self.t_but) / 0.4))
        if fb > 0:
            kit.put(fr, cam, A['book'], 540, 740 + 30 * (1 - fb), INK, fb)
            for k, R in enumerate(self.book_text):
                side, row = (k // 2, k % 2)
                R.draw(fr, cam, 365 + 350 * side, 700 + 62 * row, lambda i: 1.0, op=lambda i: fb)
            kit.put(fr, cam, self.book_lbl, 540, 910, GREY, fb)
        # everyday reading: signboards
        so = 1 - smooth((t - self.t_scr[0] + 0.3) / 0.4)
        for k, (R, lbl, x) in enumerate(self.sign):
            f = (t - self.t_sign[k]) / 0.45
            if f <= 0 or so <= 0:
                continue
            swing = 0.25 * np.exp(-max(f, 0) * 2.2) * np.cos(f * 7) + 0.06 * np.exp(-max(t - self.t_hit, 0) * 4) * np.sin((t - self.t_hit) * 30) * (t > self.t_hit)
            y = 700 - 120 * (1 - ease_out(min(f, 1)))
            op = smooth(f * 2) * so
            post = np.zeros((40, 8), np.float32) + 1
            kit.put(fr, cam, blur(post, 1), x, y - 145, WOOD, op, 1, 2.2)
            kit.put(fr, cam, A['board'], x, y, np.array([0.9, 0.82, 0.68], np.float32), op, rot=swing, over=True)
            kit.put(fr, cam, A['board_ink'], x, y, WOOD, op, rot=swing)
            R.draw(fr, cam, x, y, lambda i: 1.0, op=lambda i: op)
            kit.put(fr, cam, lbl, x, y + 150, GREY, op)
        # each script, its own job
        for k, (R, lbl, c, x) in enumerate(self.scr):
            f = (t - self.t_scr[k]) / 0.35
            if f <= 0:
                continue
            sc = 0.7 + 0.3 * ease_out(min(f, 1))
            kit.put(fr, cam, self.pad, x, 780, c, 0.5 * smooth(f), sc)
            R.draw(fr, cam, x, 770, lambda i: 1.0, op=lambda i: smooth(f), scale=sc)
            kit.put(fr, cam, lbl, x, 900, INK, smooth(f - 0.5))
        self.layer(fr, cam, self.tie.render(t, red=True), color=VERM)
        return fr.img


def _book(w=720, h=300):
    ss = StrokeSet(w, h)
    L = lambda pts, wd, ink=0.85, sd=0: ss.add(stroke(w, h, spline(pts, 12) if len(pts) > 2 else pts, wd, ink=ink, dry=0.35, profile=PRESS_DOWN, seed=sd), 0, 1)
    left = [(30, 40), (200, 20), (358, 44), (360, 270), (200, 248), (30, 268)]
    right = [(362, 44), (520, 20), (690, 40), (690, 268), (520, 248), (360, 270)]
    for k, poly in enumerate((left, right)):
        ss.add(pwash(w, h, poly, ink=0.08, soft=2, seed=k, grain=0.5), 0, 1)
    L([(30, 40), (200, 20), (358, 44)], 4, sd=2)
    L([(362, 44), (520, 20), (690, 40)], 4, sd=3)
    L([(30, 268), (200, 248), (360, 272), (520, 248), (690, 268)], 5, sd=4)
    L([(30, 40), (30, 268)], 4, sd=5)
    L([(690, 40), (690, 268)], 4, sd=6)
    L([(360, 44), (360, 272)], 3, 0.6, 7)
    return ss.render(9).astype(np.float32)


def _board(w=280, h=210, ink=False):
    ss = StrokeSet(w, h)
    poly = [(14, 30), (266, 24), (270, 190), (12, 196)]
    if not ink:
        ss.add(pwash(w, h, poly, ink=1.0, soft=1.5, seed=1), 0, 1)
        return np.clip(ss.render(9), 0, 1).astype(np.float32)
    ss.add(stroke(w, h, poly + [poly[0]], 6, ink=0.9, dry=0.4, profile=PRESS_DOWN, seed=2), 0, 1)
    for k, y in enumerate((70, 130)):
        ss.add(stroke(w, h, [(30, y), (250, y + 3)], 2, ink=0.25, dry=0.7, profile=SWEEP, seed=3 + k), 0, 1)        # grain
    for k, x in enumerate((50, 230)):
        ss.add(stroke(w, h, [(x, 26), (140, 0)], 3, ink=0.8, dry=0.3, profile=PRESS_DOWN, seed=6 + k), 0, 1)        # cord
    return ss.render(9).astype(np.float32)

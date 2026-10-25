"""Scenes 7-10: the three scripts together, a sentence that uses them, a street, and the closing wave."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import poly_ink  # noqa: F401
from core import blur, curve, spline
from inkpaint import Painting
from poly_ink.brush import BLUNT, HOOK, PRESS_DOWN, RISE, SWEEP, brush_stroke, splatter, sweep, wash

from .common import BrushTool, Labels, Scene, directional_front, ink_sprite, paste, sprite_from
from .glyph import Written, write, write_text
from .ink import (H, INK, PAPER, VERM, W, InkFront, Keys, Track, compose, ease, glaze, glow, noise, over, paper,
                  parallax, ramp, smooth, to_screen, view, view_rgb)

NA = ((0, 0.4), (0.2, 0.75), (0.85, 1.15), (1, 0.15))


def _blit(buf, st, k=1.0):
    if st is None:
        return
    for s in (st if isinstance(st, list) else [st]):
        if s is None:
            continue
        h, w = s.dens.shape
        b = buf[s.y0:s.y0 + h, s.x0:s.x0 + w]
        np.copyto(b, 1 - (1 - b) * (1 - s.dens * k))


def _up(a, size):
    return np.asarray(Image.fromarray(a.astype(np.float32), 'F').resize(size, Image.BILINEAR), np.float32)


# ===================================================================== 7. the three scripts
class ThreeScripts(Scene):
    """A triptych: 漢字 / ひらがな / カタカナ brushed down three panels, each over its own faint painting
    (a pagoda, a mountain over water, a torii). Each panel comes forward as the narration names its job.
    At the end the panels slide together towards the centre, into the sentence."""
    S = 1.25

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        self.pap = paper(ww, wh, seed=71)
        self.cols = [190, 540, 890]
        self.tracks = [Track(ww, wh) for _ in range(3)]
        sizes = [168, 124, 124]
        for k, (text, x, sz, t0) in enumerate(zip(['漢字', 'ひらがな', 'カタカナ'], self.cols, sizes, [0.15, 0.75, 1.35])):
            ws, _ = write_text(ww, wh, text, *u(x - sz / 2, 215), u(sz), t0, vertical=True, spacing=1.03, weight=0.105,
                               dry=0.36, ink=0.96, speed=4.2, seed=72 + k * 9, pause=0.04)
            self.tracks[k].add(ws)
        # separators: two thin dry vertical strokes
        self.sep = np.zeros((wh, ww), np.float32)
        for k, x in enumerate([365, 715]):
            _blit(self.sep, brush_stroke(ww, wh, U([(x, 190), (x + 3, 700), (x - 2, 1300)]), u(3), ink=0.45, dry=0.6,
                                         profile=SWEEP, seed=80 + k))
        # the paintings, faint, in the lower part of each panel
        self.bg = [np.zeros((wh, ww), np.float32) for _ in range(3)]
        B = lambda pts, w, **kw: brush_stroke(ww, wh, U(pts), u(w), **kw)
        # kanji: a five-storey pagoda on a hill
        p = Painting(ww, wh, seed=81)
        hill = p.ridge(u(1230), peaks=[(u(150), u(80), u(120))], rough=u(6), seed=82)
        p.wash(hill, dens=0.22, decay=u(80), tex=u(40), edge=0.3, seed=83)
        g = self.bg[0]
        over(g, p.D * _colmask(ww, wh, u, 0))
        px, py = 190, 1160
        for j in range(5):
            w_ = 92 - j * 13
            y = py - j * 46
            _blit(g, B([(px - w_, y - 2), (px - w_ * 0.4, y - 12), (px + w_ * 0.4, y - 12), (px + w_, y - 2)], 6,
                       ink=0.7, dry=0.35, profile=BLUNT, seed=84 + j))
            _blit(g, wash(ww, wh, U([(px - w_ * 0.55, y - 8), (px + w_ * 0.55, y - 8), (px + w_ * 0.5, y - 40),
                                     (px - w_ * 0.5, y - 40)]), ink=0.25, soft=u(1.5), seed=90 + j))
        _blit(g, B([(px, py - 230), (px, py - 300)], 3, ink=0.7, dry=0.2, profile=BLUNT, seed=96))
        # hiragana: a mountain over still water
        p = Painting(ww, wh, seed=101)
        mt = p.peak(u(540), u(1010), height=u(170), left_w=u(150), right_w=u(160), rough=u(6), seed=102)
        p.wash(mt, dens=0.3, decay=u(140), tex=u(50), edge=0.4, mist_y=u(1190), mist_h=u(40), seed=103)
        self.bg[1] = (p.D * _colmask(ww, wh, u, 1)).astype(np.float32)
        for j in range(5):
            y = 1220 + j * 18
            _blit(self.bg[1], B([(430 + j * 12, y), (540, y + 2), (650 - j * 10, y)], 2, ink=0.35, dry=0.6,
                                profile=SWEEP, seed=104 + j))
        # katakana: a torii gate
        g = self.bg[2]
        tx, ty = 890, 1290
        _blit(g, B([(tx - 135, 1052), (tx - 60, 1062), (tx + 60, 1062), (tx + 135, 1052)], 16, ink=0.8, dry=0.35,
                   profile=BLUNT, seed=111))
        _blit(g, B([(tx - 105, 1098), (tx + 105, 1098)], 9, ink=0.75, dry=0.35, profile=BLUNT, seed=112))
        for j, x in enumerate([tx - 78, tx + 78]):
            _blit(g, B([(x, 1060), (x + (4 if j else -4), ty)], 13, ink=0.78, dry=0.3, profile=BLUNT, seed=113 + j))
        _blit(g, B([(tx, 1062), (tx, 1098)], 7, ink=0.7, dry=0.3, profile=BLUNT, seed=115))
        p = Painting(ww, wh, seed=116)
        gr = p.ridge(u(1300), rough=u(4), seed=117)
        p.wash(gr, dens=0.15, decay=u(60), tex=u(40), edge=0.2, seed=118)
        over(g, p.D * _colmask(ww, wh, u, 2))
        self.fronts = [InkFront(wh // 4, ww // 4, origin=(u(x) / 4, u(1150) / 4), rough=0.4, scale=40, soft=0.08,
                                seed=120 + k) for k, x in enumerate(self.cols)]
        # names, jobs and examples, revealed as the narration reaches each
        L = self.labels = []
        t_m, t_g, t_b = self.wt(13, 'meaning'), self.wt(13, 'grammar'), self.wt(13, 'borrowed')
        self.focus = [(t_m - 0.4, t_g - 0.5), (t_g - 0.5, t_b - 1.2), (t_b - 1.2, self.L[0][1] + 0.3)]
        for k, (en, role, ex, gl, tr) in enumerate([('Kanji', 'M E A N I N G', '山', '“mountain”', t_m - 0.3),
                                                    ('Hiragana', 'G R A M M A R', 'は · を', 'particles', t_g - 0.4),
                                                    ('Katakana', 'L O A N W O R D S', 'コーヒー', '“coffee”', t_b - 1.1)]):
            lb = Labels(ww, wh)
            x = self.cols[k]
            lb.add(en, *u(x, 818), u(50), 2.1 + 0.25 * k)
            lb.add(role, *u(x, 876), u(27), tr, red=True, ink=0.95)
            lb.add(ex, *u(x, 960), u(52), tr + 0.3, cjk='mincho')
            lb.add(gl, *u(x, 1012), u(36), tr + 0.45, latin='italic')
            L.append(lb)
        self.ref = self.cam(0)

    def cam(self, t):
        u = self.u
        return self.clamp(Keys([(-0.6, (*u(540, 760), 0.83)), (self.dur - 0.3, (*u(540, 790), 0.86)),
                                (self.dur + 0.6, (*u(540, 760), 0.78))])(t))

    def frame(self, t):
        cam = self.cam(t)
        img = view_rgb(self.pap, cam)
        D = view(self.sep, cam) * (1 - ramp(t, self.dur - 0.6, self.dur + 0.2))
        conv = ramp(t, self.dur - 0.6, self.dur + 0.6)
        light = []
        for k in range(3):
            # slide toward the centre at the end (the panels converge into the sentence)
            dx = (self.u(540) - self.u(self.cols[k])) * 0.55 * conv
            c = (cam[0] - dx, cam[1], cam[2])
            bgc = parallax(c, self.ref, 0.88)
            m = _up(self.fronts[k].mask(ramp(t, -0.4 + 0.45 * k, 1.1 + 0.45 * k)), (self.ww, self.wh))
            fade = 1 - conv
            over(D, view(self.bg[k] * m, bgc) * fade)
            Dk, _ = self.tracks[k].density(t)
            over(D, view(Dk, c) * (1 - 0.6 * conv))
            # focus: the panel being talked about stays full strength, the others step back a little
            a0, a1 = self.focus[k]
            others = [self.focus[j] for j in range(3) if j != k]
            dim = max(ramp(t, o0, o0 + 0.4) * (1 - ramp(t, o1, o1 + 0.4)) for o0, o1 in others)
            dim *= 1 - ramp(t, a0, a0 + 0.3) * (1 - ramp(t, a1, a1 + 0.3))
            light.append((dim * 0.45, c))
        img = compose(img, D)
        for k in range(3):
            img = self.labels[k].draw(img, light[k][1], t)
        # step the unfocused panels back by lifting their ink toward the paper
        pap = view_rgb(self.pap, cam)
        for k, (a, c) in enumerate(light):
            if a <= 0.01:
                continue
            x0, _ = to_screen(c, self.u(self.cols[k] - 175), 0)
            x1, _ = to_screen(c, self.u(self.cols[k] + 175), 0)
            xa, xb = max(0, int(x0)), min(W, int(x1))
            if xb > xa:
                img[:, xa:xb] = img[:, xa:xb] * (1 - a) + pap[:, xa:xb] * a
        return img

    def enter(self, t, p):
        """three separate ink movements: each panel washes in from its own painting"""
        if not hasattr(self, '_fr'):
            self._fr = InkFront(H // 4, W // 4, seeds=[(x / 4, 1150 / 4, 1.0) for x in (190, 540, 890)], rough=0.3,
                                scale=70, soft=0.1, seed=129)
        return _up(self._fr.mask(ease(p)), (W, H))


def _colmask(ww, wh, u, k):
    """soft mask that keeps a painting inside its panel"""
    XX = np.arange(ww, dtype=np.float32)[None, :]
    x0, x1 = u(15 + 350 * k), u(345 + 350 * k)
    m = np.clip((XX - x0) / u(30), 0, 1) * np.clip((x1 - XX) / u(30), 0, 1)
    return np.broadcast_to(m, (wh, ww))


# ===================================================================== 8. a sentence
class Sentence(Scene):
    """私は日本語を勉強します。 written in two lines; then kanji groups underlined in ink and hiragana in
    vermilion, each with its job, as the narration names them."""
    S = 1.25

    GROUPS = [  # (line, first, last, kind, gloss, spoken-word)
        (0, 0, 0, 'k', '“I”', '"I"'), (0, 1, 1, 'h', 'wa · topic', 'wa'), (0, 2, 4, 'k', '“Japanese”', '"Japanese"'),
        (0, 5, 5, 'h', 'o · object', ' o,'), (1, 0, 1, 'k', '“study”', '"study"'), (1, 2, 4, 'h', 'shimasu · do', 'shimasu')]

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        self.pap = paper(ww, wh, seed=131)
        T = self.track
        sz, x0 = 150, 90
        self.rows = [250, 610]
        t = -0.35
        for li, text in enumerate(['私は日本語を', '勉強します']):
            for j, ch in enumerate(text):
                ws, t = write(ww, wh, ch, *u(x0 + j * sz, self.rows[li]), u(sz * 0.94), t, weight=0.1, dry=0.3,
                              speed=5.2, seed=140 + li * 10 + j, gap=0.05)
                T.add(ws)
                t += 0.03
        # 。 — a small brush circle
        cx, cy = x0 + 5 * sz + 40, self.rows[1] + 118
        ring = [(cx + np.cos(a) * 15, cy + np.sin(a) * 14) for a in np.linspace(-1.9, 4.2, 14)]
        T.add(Written(brush_stroke(ww, wh, U(ring), u(6), ink=0.9, dry=0.2, profile=BLUNT, seed=160), t, t + 0.15,
                      np.array(U(ring), np.float32)))
        self.write_end = t + 0.15
        L = self.labels = Labels(ww, wh)
        L.add('“I study Japanese.”', *u(540, 975), u(58), self.wt(14, 'I study') - 0.2, latin='italic')
        for k, (li, a, b, kind, gloss, word) in enumerate(self.GROUPS):
            tw = self.wt(15, word)
            y = self.rows[li] + sz + 22
            xa, xb = x0 + a * sz + 12, x0 + (b + 1) * sz - 12
            red = kind == 'h'
            st = brush_stroke(ww, wh, U([(xa, y + 2), ((xa + xb) / 2, y - 3), (xb, y + 1)]), u(7), ink=0.88,
                              dry=0.4, profile=BLUNT, seed=170 + k, red=red)
            T.add(Written(st, tw - 0.35, tw - 0.05, np.array(U([(xa, y), (xb, y)]), np.float32), red=red))
            xm = (xa + xb) / 2
            L.add('漢字' if kind == 'k' else 'ひらがな', *u(xm, y + 42), u(29), tw - 0.15, red=red, ink=0.9)
            L.add('kanji' if kind == 'k' else 'hiragana', *u(xm, y + 75), u(29), tw - 0.05, latin='italic', red=red,
                  ink=0.9)
            L.add(gloss, *u(xm, y + 122), u(38), tw, latin='italic', ink=0.88)
        L.add('カタカナ — none in this sentence', *u(540, 1040), u(33), self.L[1][1] - 0.6, ink=0.7)

    def cam(self, t):
        u = self.u
        return self.clamp(Keys([(-0.6, (*u(540, 660), 0.84)), (self.dur + 1, (*u(540, 680), 0.88))])(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        img = compose(view_rgb(self.pap, cam), view(D, cam), view(R, cam))
        return self.labels.draw(img, cam, t)

    def enter(self, t, p):
        """the panels have converged on the centre; the sentence page spreads out from there"""
        if not hasattr(self, '_fr'):
            self._fr = InkFront(H // 4, W // 4, origin=(W / 8, 700 / 4), rough=0.3, scale=80, soft=0.12, seed=161)
        return _up(self._fr.mask(ease(p)), (W, H))


# ===================================================================== 9. a street, past and present
class Street(Scene):
    """A traditional street running back to a pagoda in the mist: wooden houses with deep eaves, red lanterns
    swaying, a cherry branch overhead dropping petals, a traveller in a straw hat walking away up the street."""
    S = 1.25

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        rng = np.random.default_rng(91)
        self.pap = paper(ww, wh, seed=191)
        B = lambda pts, w, **kw: brush_stroke(ww, wh, U(pts), u(w), **kw)
        VP = (540, 800)
        # ---- far: mountains and the pagoda
        p = Painting(ww, wh, seed=192)
        r = p.ridge(u(640), peaks=[(u(220), u(260), u(200)), (u(840), u(200), u(180))], rough=u(14), seed=193)
        p.wash(r, dens=0.16, decay=u(160), tex=u(90), edge=0.3, mist_y=u(780), mist_h=u(70), seed=194)
        self.far = p.D.copy()
        px, py = 540, 760
        for j in range(5):
            w_ = 62 - j * 9
            y = py - j * 44
            _blit(self.far, B([(px - w_, y), (px - w_ * 0.45, y - 10), (px + w_ * 0.45, y - 10), (px + w_, y)], 5,
                              ink=0.55, dry=0.35, profile=BLUNT, seed=195 + j), 1.0)
            _blit(self.far, wash(ww, wh, U([(px - w_ * 0.5, y - 6), (px + w_ * 0.5, y - 6), (px + w_ * 0.45, y - 40),
                                             (px - w_ * 0.45, y - 40)]), ink=0.22, soft=u(1.2), seed=200 + j))
        _blit(self.far, B([(px, py - 215), (px, py - 290)], 3, ink=0.6, dry=0.2, profile=BLUNT, seed=206))
        # ---- main: houses on both sides, receding to the vanishing point
        self.mid = np.zeros((wh, ww), np.float32)
        M = self.mid
        lanterns = []
        for side in (-1, 1):
            for j in range(5):
                f0 = 1 - j * 0.17          # perspective scale of this house's near edge
                f1 = 1 - (j + 1) * 0.17
                xn = 540 + side * (540 * f0 + 10)
                xf = 540 + side * (540 * f1 + 10)
                eave_n, eave_f = VP[1] - 520 * f0, VP[1] - 520 * f1
                base_n, base_f = VP[1] + 520 * f0, VP[1] + 520 * f1
                # roof: a dark eave band sloping to the vanishing point
                roof = [(xn, eave_n - 80 * f0), (xf, eave_f - 80 * f1), (xf, eave_f + 26 * f1), (xn, eave_n + 26 * f0)]
                _blit(M, wash(ww, wh, U(roof), ink=0.58, soft=u(1.5), seed=210 + j + 10 * side))
                _blit(M, B([(xn, eave_n + 26 * f0), (xf, eave_f + 26 * f1)], 6 * f0, ink=0.85, dry=0.3, profile=BLUNT,
                           seed=230 + j + 10 * side))
                for q in range(6):          # tile lines
                    a = q / 6
                    xa = xn + (xf - xn) * a
                    ea = eave_n + (eave_f - eave_n) * a
                    sa = f0 + (f1 - f0) * a
                    _blit(M, B([(xa, ea - 78 * sa), (xa, ea + 22 * sa)], 2.2 * sa, ink=0.5, dry=0.5, profile=PRESS_DOWN,
                               seed=250 + q + 10 * j + 100 * side))
                # wall: pale wash with lattice and a post at the near edge
                wall = [(xn, eave_n + 26 * f0), (xf, eave_f + 26 * f1), (xf, base_f), (xn, base_n)]
                _blit(M, wash(ww, wh, U(wall), ink=0.13 + 0.05 * (j % 2), soft=u(3), seed=300 + j + 10 * side, grain=0.6))
                _blit(M, B([(xn, eave_n + 26 * f0), (xn, base_n)], 9 * f0, ink=0.8, dry=0.35, profile=BLUNT,
                           seed=320 + j + 10 * side))
                for q in range(1, 4):
                    a = q / 4
                    xa = xn + (xf - xn) * a
                    ea = eave_n + (eave_f - eave_n) * a + 40 * (f0 + (f1 - f0) * a)
                    ba = base_n + (base_f - base_n) * a - 120 * (f0 + (f1 - f0) * a)
                    _blit(M, B([(xa, ea), (xa, ba)], 2.2 * f0, ink=0.32, dry=0.65, profile=BLUNT,
                               seed=340 + q + 10 * j + 100 * side))
                # noren curtain on the nearer houses
                if j < 3:
                    a0, a1 = 0.25, 0.6
                    xa, xb = xn + (xf - xn) * a0, xn + (xf - xn) * a1
                    ya = eave_n + (eave_f - eave_n) * a0 + 34 * f0
                    yb = eave_n + (eave_f - eave_n) * a1 + 34 * f1
                    cur = [(xa, ya), (xb, yb), (xb, yb + 120 * f1), (xa, ya + 120 * f0)]
                    _blit(M, wash(ww, wh, U(cur), ink=0.45, soft=u(1.2), seed=380 + j + 10 * side))
                lanterns.append((xn + (xf - xn) * 0.12, eave_n + 30 * f0 + 4, 0.9 * f0 + 0.1))
        # the street: a pale wash with a few perspective lines
        road = [(0, 1920), (1080, 1920), (560, VP[1] + 10), (520, VP[1] + 10)]
        _blit(M, wash(ww, wh, U(road), ink=0.06, soft=u(4), seed=400, grain=0.5))
        for k, xb in enumerate([-200, 150, 400, 700, 950, 1300]):
            _blit(M, B([(xb, 1920), (540 + (xb - 540) * 0.35, 1180), (540, VP[1] + 30)], 2.2, ink=0.18, dry=0.7,
                       profile=SWEEP, seed=410 + k))
        # mist rising at the far end of the street swallows the bases of the houses
        YY, XX = np.mgrid[0:wh, 0:ww].astype(np.float32)
        mist = np.exp(-((YY - u(VP[1] + 60)) / u(260)) ** 2) * np.exp(-((XX - u(540)) / u(330)) ** 2)
        mist *= 0.55 + 0.45 * noise(wh, ww, u(160), 4, 199)
        M *= 1 - 0.8 * np.clip(mist, 0, 1)
        self.lanterns = lanterns
        self.lantern = _lantern_sprite()
        # ---- the traveller (a sprite, walking away up the street)
        self.trav = _traveller_sprite()
        # ---- near: a cherry branch top-right with blossoms (vermilion, very pale = pink)
        self.near = np.zeros((wh, ww), np.float32)
        self.nearR = np.zeros((wh, ww), np.float32)
        br = [(1120, 120), (960, 190), (820, 230), (700, 300), (610, 330)]
        _blit(self.near, B(br, 22, ink=0.85, dry=0.45, profile=PRESS_DOWN, seed=420))
        _blit(self.near, B([(900, 205), (860, 300), (840, 380)], 9, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=421))
        _blit(self.near, B([(780, 250), (720, 190), (650, 170)], 8, ink=0.8, dry=0.5, profile=PRESS_DOWN, seed=422))
        for j in range(70):
            bx, by = br[rng.integers(1, 5)]
            cx, cy = bx + rng.normal(0, 60), by + rng.normal(10, 45)
            r_ = rng.uniform(8, 15)
            petals = [(cx + np.cos(a) * r_ * (0.8 + 0.3 * rng.random()), cy + np.sin(a) * r_ * (0.8 + 0.3 * rng.random()))
                      for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
            _blit(self.nearR, wash(ww, wh, U(petals), ink=rng.uniform(0.22, 0.4), soft=u(1.2), seed=430 + j, grain=0.3))
            if rng.random() < 0.5:
                _blit(self.near, B([(cx - 2, cy - 2), (cx + 2, cy + 2)], 3, ink=0.7, dry=0.2, profile=BLUNT, seed=520 + j))
        self.petals = rng.random((26, 4))
        self.ref = self.cam(0)

    def cam(self, t):
        u = self.u
        return self.clamp(Keys([(-0.7, (*u(540, 900), 0.82)), (self.dur + 1.3, (*u(560, 860), 0.97))])(t))

    def frame(self, t):
        cam = self.cam(t)
        z = cam[2]
        img = view_rgb(self.pap, cam)
        D = view(self.far, parallax(cam, self.ref, 0.6))
        over(D, view(self.mid, cam))
        img = compose(img, D)
        # lanterns: swaying, glowing
        for k, (lx, ly, s) in enumerate(self.lanterns):
            sx, sy = to_screen(cam, self.u(lx), self.u(ly))
            sw = 3.5 * np.sin(t * 1.6 + k * 0.9)
            sc = s * z * 1.25
            rgb, a = self.lantern
            im = Image.fromarray((np.dstack([rgb, a]) * 255).astype(np.uint8), 'RGBA')
            im = im.resize((max(2, int(im.width * sc)), max(2, int(im.height * sc))), Image.LANCZOS)
            im = im.rotate(sw, resample=Image.BICUBIC, center=(im.width / 2, 0))
            r2, a2 = sprite_from(im)
            g = np.zeros((im.height + 80, im.width + 80), np.float32)
            g[40:-40, 40:-40] = a2
            gl = blur(g, 14 * sc) * 0.6
            x0, y0 = int(sx - im.width / 2) - 40, int(sy) - 40
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + g.shape[1]), min(H, y0 + g.shape[0])
            if xb > xa and yb > ya:
                img[ya:yb, xa:xb] = glow(img[ya:yb, xa:xb], gl[ya - y0:yb - y0, xa - x0:xb - x0], (1.0, 0.72, 0.42))
            img = paste(img, r2, a2, sx - im.width / 2, sy)
        # the traveller walks away up the street
        rgb, a = self.trav
        f = ramp(t, -0.5, self.dur + 1.0)
        wy = 1420 - 260 * f
        sc = (1.0 - 0.42 * f) * z * 1.25
        im = Image.fromarray((np.dstack([rgb, a]) * 255).astype(np.uint8), 'RGBA')
        im = im.resize((max(2, int(im.width * sc)), max(2, int(im.height * sc))), Image.LANCZOS)
        im = im.rotate(1.5 * np.sin(t * 6.0), resample=Image.BICUBIC)
        sx, sy = to_screen(cam, self.u(548), self.u(wy))
        bob = abs(np.sin(t * 6.0)) * 5 * sc
        r2, a2 = sprite_from(im)
        img = paste(img, r2, a2, sx - im.width / 2, sy - im.height - bob)
        # near branch and blossoms
        nc = parallax(cam, self.ref, 1.4)
        sway = (self.u(6) * np.sin(t * 0.9), 0)
        nc = (nc[0] + sway[0], nc[1], nc[2])
        img = img * (1 - view(self.near, nc)[..., None] * (1 - INK))
        img = img * (1 - view(self.nearR, nc)[..., None] * (1 - np.array([0.93, 0.62, 0.62])))
        # falling petals
        for k, (a0, b0, c0, d0) in enumerate(self.petals):
            life = (t * (0.18 + 0.1 * c0) + a0) % 1.0
            x = 560 + b0 * 520 + 120 * np.sin(life * 6 + k) - life * 260
            y = 260 + life * 1150
            r_ = 5 + 4 * d0
            xi, yi = int(x), int(y)
            if 8 < xi < W - 8 and 8 < yi < H - 8:
                YY, XX = np.ogrid[-8:9, -8:9]
                ang = t * 3 + k
                e = np.exp(-(((XX * np.cos(ang) + YY * np.sin(ang)) / r_) ** 2 + ((-XX * np.sin(ang) + YY * np.cos(ang)) / (r_ * 0.55)) ** 2))
                sl = img[yi - 8:yi + 9, xi - 8:xi + 9]
                sl[:] = sl * (1 - (e * 0.55)[..., None] * (1 - np.array([0.93, 0.6, 0.62])))
        return img

    def enter(self, t, p):
        """the sentence washes away in ink and the street spreads out from its centre"""
        if not hasattr(self, '_fr'):
            self._fr = InkFront(H // 4, W // 4, origin=(W / 8, 900 / 4), rough=0.32, scale=80, soft=0.12, seed=98)
        return _up(self._fr.mask(ease(p)), (W, H))


def _lantern_sprite():
    def draw(w, h):
        out = [wash(w, h, [(w / 2 + np.cos(a) * 30, 62 + np.sin(a) * 44) for a in np.linspace(0, 2 * np.pi, 24)],
                    ink=0.9, soft=1.2, seed=1)]
        out[0].red = True
        out.append(brush_stroke(w, h, [(w / 2 - 18, 18), (w / 2 + 18, 18)], 7, ink=0.9, dry=0.2, profile=BLUNT, seed=2))
        out.append(brush_stroke(w, h, [(w / 2 - 18, 106), (w / 2 + 18, 106)], 7, ink=0.9, dry=0.2, profile=BLUNT, seed=3))
        out.append(brush_stroke(w, h, [(w / 2, 0), (w / 2, 16)], 2.5, ink=0.8, dry=0.2, profile=BLUNT, seed=4))
        for k in range(4):
            y = 36 + k * 16
            hw = 30 * np.sqrt(max(0.0, 1 - ((y - 62) / 44) ** 2))
            out.append(brush_stroke(w, h, [(w / 2 - hw, y), (w / 2, y + 3), (w / 2 + hw, y)], 1.6, ink=0.55, dry=0.4,
                                    profile=BLUNT, seed=5 + k))
        return out
    return ink_sprite(80, 116, draw)


def _traveller_sprite():
    def draw(w, h):
        c = w / 2
        out = [wash(w, h, [(c, 10), (c - 70, 58), (c + 70, 58)], ink=0.55, soft=1.5, seed=1),
               brush_stroke(w, h, [(c - 74, 60), (c, 52), (c + 74, 60)], 7, ink=0.9, dry=0.3, profile=BLUNT, seed=2),
               brush_stroke(w, h, [(c - 72, 59), (c - 30, 32), (c, 10)], 4, ink=0.8, dry=0.3, profile=PRESS_DOWN, seed=3),
               brush_stroke(w, h, [(c, 10), (c + 30, 32), (c + 72, 59)], 4, ink=0.8, dry=0.3, profile=PRESS_DOWN, seed=4),
               wash(w, h, [(c - 30, 62), (c + 30, 62), (c + 48, 210), (c + 40, 250), (c - 40, 250), (c - 48, 210)],
                    ink=0.5, soft=2, seed=5),
               brush_stroke(w, h, [(c - 30, 64), (c - 46, 160), (c - 44, 250)], 6, ink=0.8, dry=0.4, profile=BLUNT, seed=6),
               brush_stroke(w, h, [(c + 30, 64), (c + 48, 160), (c + 44, 250)], 5, ink=0.75, dry=0.45, profile=BLUNT, seed=7),
               brush_stroke(w, h, [(c + 52, 40), (c + 58, 160), (c + 62, 290)], 4, ink=0.85, dry=0.3, profile=BLUNT, seed=8),
               brush_stroke(w, h, [(c - 20, 250), (c - 22, 292)], 7, ink=0.7, dry=0.4, profile=BLUNT, seed=9),
               brush_stroke(w, h, [(c + 18, 250), (c + 20, 288)], 7, ink=0.7, dry=0.4, profile=BLUNT, seed=10)]
        return out
    return ink_sprite(200, 300, draw)


# ===================================================================== 10. the closing wave
class Finale(Scene):
    """Paper; a huge wave written in dry brush sweeps across from the right and curls over; a vermilion sun;
    日本語 brushed down the left; a small traveller faces the wave."""
    S = 1.25

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        self.pap = paper(ww, wh, seed=201)
        T = self.track
        # a breaking wave in dry brush (after poly_ink's 水 painting), drawn for a portrait frame: the body rises
        # from the bottom right and curls over to the left at the crest
        t = 0.25
        body = [([(1110, 1400), (1010, 1150), (930, 900), (860, 690), (780, 520), (680, 430), (580, 420), (510, 470),
                  (490, 550), (530, 612)], 118, 0.86, 0.6),
                ([(1010, 1410), (925, 1160), (852, 925), (790, 735), (722, 592), (650, 522), (592, 516), (562, 560),
                  (578, 602)], 86, 0.9, 0.62),
                ([(1100, 1300), (1045, 1050), (992, 822), (930, 640), (860, 512), (770, 452)], 70, 0.78, 0.66),
                ([(905, 1390), (842, 1180), (782, 990), (730, 842), (668, 722), (612, 652)], 48, 0.6, 0.62)]
        for k, (pts, width, ink, dry) in enumerate(body):
            T.stroke(sweep(ww, wh, U(pts), u(width), ink=ink, dry=dry, seed=100 + 7 * k, wash=0.3, profile=RISE),
                     t + 0.13 * k, t + 0.13 * k + 0.55, path=U(pts))
        t += 0.13 * len(body) + 0.3
        for k, (pts, width, ink, dry) in enumerate([
                ([(800, 470), (905, 525), (992, 645), (1060, 790), (1110, 920)], 58, 0.5, 0.66),
                ([(700, 426), (604, 408), (532, 440), (494, 506), (504, 572), (546, 604)], 40, 0.92, 0.5)]):
            T.stroke(sweep(ww, wh, U(pts), u(width), ink=ink, dry=dry, seed=150 + 7 * k, wash=0.25), t + 0.12 * k,
                     t + 0.12 * k + 0.4)
        T.stroke(splatter(ww, wh, *u(585, 418), (u(150), u(70)), n=46, seed=41), t + 0.2, t + 0.5)
        T.stroke(splatter(ww, wh, *u(505, 575), (u(50), u(40)), n=14, size=(1.2, 3.5), seed=44, direction=2.6),
                 t + 0.35, t + 0.55)
        t += 0.45
        for k, (pts, width, ink, dry) in enumerate([
                ([(130, 1332), (440, 1312), (790, 1334), (1100, 1304)], 44, 0.62, 0.7),
                ([(300, 1392), (650, 1380), (1100, 1398)], 28, 0.45, 0.72)]):
            T.stroke(sweep(ww, wh, U(pts), u(width), ink=ink, dry=dry, seed=200 + 7 * k, wash=0.2), t + 0.12 * k,
                     t + 0.12 * k + 0.4)
        T.stroke(splatter(ww, wh, *u(960, 1290), (u(140), u(40)), n=24, seed=42, direction=0.1), t + 0.2, t + 0.45)
        self.wave_end = t + 0.5
        # 日本語 down the left
        ws, self.title_end = write_text(ww, wh, '日本語', *u(95, 600), u(235), 0.45, vertical=True, spacing=1.02,
                                        weight=0.112, dry=0.46, ink=0.97, speed=3.4, seed=4, pause=0.05)
        T.add(ws)
        # a small traveller facing the wave
        tx, ty = 410, 1300
        for st in [wash(ww, wh, U([(tx, ty - 92), (tx - 30, ty - 70), (tx + 30, ty - 70)]), ink=0.5, soft=u(1), seed=301),
                   brush_stroke(ww, wh, U([(tx - 32, ty - 69), (tx + 32, ty - 69)]), u(4), ink=0.9, dry=0.3, profile=BLUNT, seed=302),
                   wash(ww, wh, U([(tx - 12, ty - 66), (tx + 12, ty - 66), (tx + 20, ty), (tx - 20, ty)]), ink=0.55, soft=u(1.5), seed=303),
                   brush_stroke(ww, wh, U([(tx + 18, ty - 50), (tx + 46, ty - 64), (tx + 70, ty - 74)]), u(2.5), ink=0.9, dry=0.2,
                                profile=PRESS_DOWN, seed=304)]:
            T.stroke(st, self.wave_end + 0.1, self.wave_end + 0.5)
        # the sun
        YY, XX = np.mgrid[0:wh, 0:ww].astype(np.float32)
        sx, sy, r = *u(300, 430), u(150)
        dist = np.hypot(XX - sx, YY - sy)
        self.sun = (np.clip((r - dist) / 3, 0, 1) * (0.82 + 0.18 * noise(wh, ww, u(14), 3, 21)) * 0.88).astype(np.float32)
        self.sun_d = (dist / r).astype(np.float32)

    def cam(self, t):
        u = self.u
        return self.clamp(Keys([(-0.6, (*u(600, 900), 0.9)), (self.wave_end, (*u(570, 920), 0.86)),
                                (self.dur, (*u(540, 940), 0.81))])(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        sp = ramp(t, 0.9, 2.0)
        if sp > 0:
            R = 1 - (1 - R) * (1 - self.sun * np.clip((sp * 1.15 - self.sun_d) / 0.15, 0, 1))
        return compose(view_rgb(self.pap, cam), view(D, cam), view(R, cam))

    def enter(self, t, p):
        """one last brushstroke crosses the street, and leaves an open sheet of paper behind it"""
        if not hasattr(self, '_fr'):
            self._fr = directional_front(12, rough=0.4, soft=0.06, seed=219)
        return _up(self._fr.mask(ease(p)), (W, H))


# ===================================================================== 11. the end card
class EndCard(Scene):
    """The sign-off: "Follow for more Japanese" over the @raj_in_jp band, 最後までみてくれてありがとう。 brushed in
    stroke order with its translation, and a vermilion またね！ to close, above a quiet ridge in mist."""
    S = 1.25

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        self.pap = paper(ww, wh, seed=221)
        T = self.track
        # a quiet ridge in mist at the foot of the card, and a traveller walking off into it
        p = Painting(ww, wh, seed=222)
        r = p.ridge(u(1300), peaks=[(u(220), u(150), u(190)), (u(820), u(210), u(200))], rough=u(10), seed=223)
        p.wash(r, dens=0.15, decay=u(130), tex=u(80), edge=0.3, mist_y=u(1400), mist_h=u(60), seed=224)
        self.hills = p.D.copy()
        tx, ty = 735, 1262
        for st in [wash(ww, wh, U([(tx, ty - 46), (tx - 16, ty - 35), (tx + 16, ty - 35)]), ink=0.55, soft=u(0.8), seed=231),
                   wash(ww, wh, U([(tx - 6, ty - 34), (tx + 6, ty - 34), (tx + 10, ty), (tx - 10, ty)]), ink=0.6, soft=u(1), seed=232),
                   brush_stroke(ww, wh, U([(tx + 11, ty - 40), (tx + 13, ty + 2)]), u(1.6), ink=0.8, dry=0.2, profile=BLUNT,
                                seed=233)]:
            _blit(self.hills, st)
        # "Follow for more Japanese" and the handle on a soft ink band
        t_follow = self.L[0][0] - 0.1
        self.band = np.zeros((wh, ww), np.float32)
        _blit(self.band, sweep(ww, wh, U([(250, 408), (540, 400), (830, 406)]), u(96), ink=0.9, dry=0.4, seed=240,
                               wash=0.6, profile=((0, 0.6), (0.08, 1.0), (0.9, 0.95), (1, 0.4))))
        self.band_front = InkFront(wh // 4, ww // 4, direction=(1, 0), rough=0.3, scale=20, soft=0.1, seed=241)
        self.t_band = t_follow
        L = self.labels = Labels(ww, wh)
        L.add('Follow for more Japanese', *u(540, 318), u(46), t_follow, latin='italic', ink=0.85)
        L.add('@raj_in_jp', *u(540, 424), u(58), t_follow + 0.45, color=(0.95, 0.91, 0.82), ink=1.0)
        # 最後までみてくれてありがとう。 brushed in two lines
        t0 = t_follow + 1.0
        sz = 96
        ws, t = write_text(ww, wh, '最後までみてくれて', *u(540 - 9 * sz / 2, 560), u(sz), t0, spacing=1.0, weight=0.1,
                           dry=0.3, ink=0.96, speed=7.0, seed=250, pause=0.01)
        T.add(ws)
        ws, t = write_text(ww, wh, 'ありがとう', *u(540 - 5.6 * sz / 2, 690), u(sz), t + 0.05, spacing=1.0, weight=0.1,
                           dry=0.3, ink=0.96, speed=6.5, seed=260, pause=0.01)
        T.add(ws)
        cx, cy = 540 - 5.6 * sz / 2 + 5 * sz + 22, 690 + sz * 0.8           # 。
        ring = [(cx + np.cos(a) * 10, cy + np.sin(a) * 9.5) for a in np.linspace(-1.9, 4.2, 14)]
        T.add(Written(brush_stroke(ww, wh, U(ring), u(4.5), ink=0.9, dry=0.2, profile=BLUNT, seed=270), t, t + 0.12,
                      np.array(U(ring), np.float32)))
        self.thanks_end = t + 0.12
        L.add('Thank you for watching till the end.', *u(540, 868), u(40), self.thanks_end - 0.1, latin='italic', ink=0.8)
        # またね！ in vermilion
        t_m = self.wt(20, 'Mata ne') - 1.1
        sz = 150
        x0 = 540 - 4 * sz / 2
        ws, t = write_text(ww, wh, 'またね', *u(x0, 960), u(sz), t_m, spacing=1.0, weight=0.105, dry=0.4, ink=0.95,
                           speed=4.0, seed=280, all_red=True, pause=0.03)
        T.add(ws)
        ex = x0 + 3 * sz + sz * 0.45
        T.add(Written(brush_stroke(ww, wh, U([(ex, 980), (ex - 2, 1050), (ex - 4, 1072)]), u(15), ink=0.95, dry=0.35,
                                   profile=PRESS_DOWN, seed=281, red=True), t, t + 0.12,
                      np.array(U([(ex, 980), (ex - 4, 1072)]), np.float32), red=True))
        T.add(Written(brush_stroke(ww, wh, U([(ex - 5, 1092), (ex - 2, 1100)]), u(15), ink=0.95, dry=0.2,
                                   profile=((0, 0.5), (0.4, 1.0), (1, 0.6)), seed=282, red=True), t + 0.15, t + 0.22,
                      np.array(U([(ex - 5, 1092), (ex - 2, 1100)]), np.float32), red=True))
        self.matane_end = t + 0.22
        L.add('mata ne  ·  see you', *u(540, 1180), u(36), self.matane_end - 0.1, latin='italic', red=True, ink=0.9)

    def cam(self, t):
        u = self.u
        return self.clamp(Keys([(-0.6, (*u(540, 780), 0.88)), (self.dur + 1, (*u(540, 760), 0.82))])(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        over(D, self.hills * ramp(t, -0.5, 1.2))
        bp = ramp(t, self.t_band - 0.2, self.t_band + 0.5)
        if bp > 0:
            over(D, self.band * _up(self.band_front.mask(bp), (self.ww, self.wh)))
        img = compose(view_rgb(self.pap, cam), view(D, cam), view(R, cam))
        return self.labels.draw(img, cam, t)

    def enter(self, t, p):
        """the wave settles: a soft ink wash opens onto a clean sheet"""
        if not hasattr(self, '_fr'):
            self._fr = InkFront(H // 4, W // 4, origin=(W / 8, 700 / 4), rough=0.3, scale=80, soft=0.12, seed=229)
        return _up(self._fr.mask(ease(p)), (W, H))

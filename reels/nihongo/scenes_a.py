"""Scenes 1-2: the title, and the islands before writing."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import poly_ink  # noqa: F401
from core import blur, curve, spline
from inkpaint import Painting
from poly_ink.brush import BLUNT, HOOK, PRESS_DOWN, RISE, SWEEP, brush_stroke, splatter, sweep, wash

from .common import BrushTool, Scene, paste
from .glyph import write_text
from .ink import (H, PAPER, VERM, W, InkFront, Keys, compose, ease, glow, noise, over, paper, parallax, ramp, smooth,
                  to_screen, view, view_rgb)


# ===================================================================== 1. the title
class Title(Scene):
    """paper in shallow focus; one bold stroke sweeps across and blooms; 日本語 is written down the page by a
    brush; a vermilion sun rises behind it and ink mountains climb out of the first stroke"""
    S = 1.3

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        self.pap = paper(ww, wh, seed=11)
        # the opening sweep: lands at the left with a bloom, runs dry to the right
        self.sweep_pts = U([(30, 1318), (260, 1290), (620, 1282), (900, 1268), (1075, 1250)])
        self.track.stroke(sweep(ww, wh, self.sweep_pts, u(64), ink=0.92, dry=0.52, seed=7, wash=0.32), 0.15, 0.8)
        self.track.stroke(splatter(ww, wh, u(1040), u(1250), (u(60), u(26)), n=18, seed=3, direction=0.0), 0.7, 0.85)
        # 日本語, written down the page
        ws, self.title_end = write_text(ww, wh, '日本語', u(392), u(262), u(296), 0.95, vertical=True, spacing=1.0,
                                        weight=0.112, dry=0.46, ink=0.97, speed=3.1, seed=4, pause=0.06)
        self.track.add(ws)
        # mountains that grow out of the sweep (revealed by an ink front climbing from the stroke)
        p = Painting(ww, wh, seed=5)
        far = p.ridge(u(1120), peaks=[(u(130), u(230), u(170)), (u(330), u(120), u(150)), (u(860), u(300), u(170)),
                                      (u(1010), u(160), u(130))], rough=u(14), seed=8)
        p.wash(far, dens=0.16, decay=u(110), tex=u(90), edge=0.25, mist_y=u(1240), mist_h=u(45), seed=9)
        near = p.ridge(u(1230), peaks=[(u(90), u(110), u(120)), (u(980), u(140), u(140))], rough=u(10), seed=10)
        p.wash(near, dens=0.26, decay=u(70), tex=u(60), edge=0.35, mist_y=u(1300), mist_h=u(30), seed=11)
        self.mount = p.D.copy()
        # keep the title column clear of mountain ink
        YY, XX = np.mgrid[0:wh, 0:ww].astype(np.float32)
        self.mount *= 1 - 0.85 * np.exp(-((XX - u(540)) / u(170)) ** 2)
        self.mfront = InkFront(wh // 4, ww // 4, direction=(0, -1), rough=0.35, scale=50, soft=0.08, seed=12)
        # the sun: a vermilion disc that soaks into the paper
        sx, sy, r = u(752), u(430), u(150)
        dist = np.hypot(XX - sx, YY - sy)
        tex = 0.82 + 0.18 * noise(wh, ww, u(14), 3, 21)
        self.sun = (np.clip((r - dist) / 3, 0, 1) * tex * 0.88).astype(np.float32)
        self.sun_d = (dist / r).astype(np.float32)
        self.brush = BrushTool(scale=0.95)
        self.impact = self.title_end + 0.1
        self.camk = Keys([(0.0, (u(150), u(1290), 1.25)), (0.8, (u(520), u(1210), 1.05)),
                          (1.15, (u(540), u(500), 0.98)), (2.4, (u(540), u(820), 0.9)),
                          (self.title_end + 0.2, (u(540), u(960), 0.8)), (self.dur + 1, (u(540), u(930), 0.79))])

    def cam(self, t):
        return self.clamp(self.camk(t))

    def frame(self, t):
        cam = self.cam(t)
        D, R = self.track.density(t)
        mp = ramp(t, 2.6, 5.2)
        if mp > 0:
            m = np.asarray(Image.fromarray(self.mfront.mask(mp), 'F').resize((self.ww, self.wh), Image.BILINEAR))
            over(D, self.mount * m)
        sp = ramp(t, 3.0, 4.4)
        if sp > 0:
            # the sun soaks outward from its centre, the rim arriving last
            R = 1 - (1 - R) * (1 - self.sun * np.clip((sp * 1.15 - self.sun_d) / 0.15, 0, 1))
        img = compose(view_rgb(self.pap, cam), view(D, cam), view(R, cam))
        if t < self.title_end + 0.4:
            img = self.brush.draw(img, self.track, t, cam, presence=ramp(t, 0.85, 1.0))
        foc = 6 * (1 - ramp(t, 0.0, 0.75))
        if foc > 0.3:
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(foc))
            img = np.asarray(im, np.float32) / 255
        return img


# ===================================================================== 2. before writing
DOTP = ((0, 0.5), (0.4, 1.0), (1, 0.55))


class BeforeWriting(Scene):
    """a Yayoi-period settlement under misty mountains: pit dwellings, a raised granary, people talking round a
    fire. Speech is shown as fine ink lines that drift up between the speakers and dissolve."""
    S = 1.25

    def build(self):
        u, U, ww, wh = self.u, self.U, self.ww, self.wh
        rng = np.random.default_rng(22)
        self.pap = paper(ww, wh, seed=23)
        YY, XX = np.mgrid[0:wh, 0:ww].astype(np.float32)

        # ---- far layer: mountain ranges in mist
        p = Painting(ww, wh, seed=24)
        r1 = p.ridge(u(560), peaks=[(u(180), u(210), u(160)), (u(520), u(330), u(190)), (u(900), u(250), u(170))],
                     rough=u(16), seed=25)
        p.wash(r1, dens=0.13, decay=u(150), tex=u(110), edge=0.25, mist_y=u(700), mist_h=u(70), seed=26)
        r2 = p.ridge(u(760), peaks=[(u(80), u(150), u(140)), (u(420), u(90), u(180)), (u(1000), u(180), u(150))],
                     rough=u(14), seed=27)
        p.wash(r2, dens=0.22, decay=u(120), tex=u(80), edge=0.35, mist_y=u(860), mist_h=u(60), seed=28)
        p.birds([(u(700), u(380), 1.4), (u(745), u(360), 1.1), (u(790), u(392), 1.2)])
        self.far = p.D.copy()

        # ---- main plane: ground, dwellings, trees, people, fire
        p = Painting(ww, wh, seed=30)
        ground = p.ridge(u(900), peaks=[(u(150), u(40), u(200)), (u(950), u(30), u(220))], rough=u(8), seed=31)
        p.wash(ground, dens=0.1, decay=u(260), tex=u(70), edge=0.2, seed=32)
        self.mid = p.D.copy()
        t = 0.2
        T = self.track
        B = lambda pts, w, **k: brush_stroke(ww, wh, U(pts), u(w), **k)
        # pit dwellings: thatched roofs down to the ground, a smoke hole at the top
        for k, (x, y, s) in enumerate([(790, 1000, 1.35), (990, 1040, 1.0), (610, 955, 0.8)]):
            roof = [(x - 120 * s, y), (x - 70 * s, y - 70 * s), (x - 22 * s, y - 118 * s), (x + 22 * s, y - 118 * s),
                    (x + 70 * s, y - 70 * s), (x + 120 * s, y)]
            T.stroke(wash(ww, wh, U(roof), ink=0.34, soft=u(2), seed=40 + k), t, t + 0.5)
            T.stroke([B([(x - 124 * s, y + 2), (x - 72 * s, y - 70 * s), (x - 26 * s, y - 120 * s)], 5 * s, dry=0.4,
                        profile=PRESS_DOWN, seed=50 + k),
                      B([(x + 26 * s, y - 120 * s), (x + 72 * s, y - 70 * s), (x + 124 * s, y + 2)], 5 * s, dry=0.45,
                        profile=BLUNT, seed=60 + k),
                      B([(x - 30 * s, y - 122 * s), (x, y - 140 * s), (x + 30 * s, y - 122 * s)], 4 * s, dry=0.3,
                        profile=BLUNT, seed=70 + k)], t + 0.3, t + 0.8)
            for j in range(9):        # thatch: dry downward strokes
                a = -0.8 + 1.6 * j / 8
                x0 = x + a * 40 * s
                T.stroke(B([(x0, y - 100 * s + abs(a) * 30 * s), (x0 + a * 70 * s, y - 8)], 1.6 * s + 1, ink=0.5,
                           dry=0.6, profile=PRESS_DOWN, seed=80 + k * 10 + j), t + 0.5, t + 0.9)
            T.stroke(wash(ww, wh, U([(x - 18 * s, y), (x - 16 * s, y - 40 * s), (x + 16 * s, y - 40 * s),
                                     (x + 18 * s, y)]), ink=0.72, soft=u(1.5), seed=90 + k), t + 0.6, t + 0.9)
            t += 0.45
        # raised granary on posts (left)
        gx, gy, g = 225, 985, 1.3
        G = lambda pts: [(gx + a * g, gy + b * g) for a, b in pts]
        T.stroke(wash(ww, wh, U(G([(-95, -100), (-50, -175), (50, -175), (95, -100)])), ink=0.36, soft=u(2), seed=100),
                 0.6, 1.1)
        T.stroke([B(G([(-100, -98), (-50, -178), (50, -178), (100, -98)]), 5 * g, dry=0.4, profile=BLUNT, seed=101),
                  wash(ww, wh, U(G([(-60, -100), (60, -100), (60, -52), (-60, -52)])), ink=0.22, soft=u(1.5),
                       seed=102)], 0.9, 1.3)
        for j in range(9):
            a = -0.8 + 1.6 * j / 8
            T.stroke(B(G([(a * 40, -165 + abs(a) * 20), (a * 95, -104)]), 1.8, ink=0.5, dry=0.6, profile=PRESS_DOWN,
                       seed=109 + j), 1.0, 1.3)
        for j, px in enumerate([-50, -15, 20, 52]):
            T.stroke(B(G([(px, -52), (px + 2, 10)]), 4.5 * g, dry=0.3, profile=BLUNT, seed=103 + j),
                     1.2 + 0.05 * j, 1.45 + 0.05 * j)
        T.stroke(B(G([(-78, -50), (78, -52)]), 4 * g, dry=0.3, profile=BLUNT, seed=108), 1.3, 1.5)
        T.stroke(B(G([(70, 12), (96, -30), (118, -64)]), 3.5 * g, dry=0.4, profile=BLUNT, seed=119), 1.4, 1.6)  # ladder
        for j in range(4):
            T.stroke(B(G([(78 + j * 9, -2 - j * 15), (92 + j * 9, -6 - j * 15)]), 2.5 * g, dry=0.3, profile=BLUNT,
                       seed=118 - j), 1.5, 1.6)
        # broadleaf trees: trunk and branches, then foliage as soft washes with darker dabs on top
        for k, (x, y, s) in enumerate([(400, 975, 1.25), (1030, 950, 1.45), (80, 1040, 1.15)]):
            T.stroke(B([(x, y), (x - 6 * s, y - 80 * s), (x + 4 * s, y - 170 * s)], 11 * s, dry=0.45, profile=BLUNT,
                       seed=120 + k), 0.9 + 0.2 * k, 1.3 + 0.2 * k)
            for j, (bx, by) in enumerate([(-50, -200), (48, -190), (-10, -235)]):
                T.stroke(B([(x - 2 * s, y - 120 * s), (x + bx * 0.5 * s, y + (by + 40) * s * 0.85), (x + bx * s, y + by * s)],
                           4.5 * s, dry=0.45, profile=PRESS_DOWN, seed=124 + k * 5 + j), 1.2 + 0.2 * k, 1.4 + 0.2 * k)
            for j in range(11):
                cx = x + rng.normal(0, 42 * s)
                cy = y - 205 * s + rng.normal(0, 30 * s)
                blob = [(cx + np.cos(a) * 44 * s * (0.8 + 0.3 * rng.random()), cy + np.sin(a) * 26 * s * (0.8 + 0.3 * rng.random()))
                        for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)]
                T.stroke(wash(ww, wh, U(blob), ink=rng.uniform(0.22, 0.34), soft=u(4), seed=140 + k * 13 + j, grain=0.45),
                         1.3 + 0.2 * k + j * 0.04, 1.6 + 0.2 * k + j * 0.04)
            for j in range(50):
                cx = x + rng.normal(0, 44 * s)
                cy = y - 205 * s + rng.normal(0, 28 * s)
                ang = rng.uniform(-0.6, 0.6) + np.pi * rng.integers(0, 2)
                ln = rng.uniform(8, 16) * s
                T.stroke(B([(cx, cy), (cx + np.cos(ang) * ln, cy + np.sin(ang) * ln * 0.4)], rng.uniform(8, 13) * s,
                           ink=rng.uniform(0.45, 0.75), dry=0.35, profile=DOTP, seed=160 + k * 40 + j),
                         1.6 + 0.2 * k + j * 0.015, 1.7 + 0.2 * k + j * 0.015)

        # ---- the people around the fire
        self.fire = (540, 1222)
        people = [(318, 1170, 1, 1.85), (445, 1270, 1, 1.95), (762, 1168, -1, 1.85), (640, 1274, -1, 1.95)]
        self.heads = []
        for k, (x, y, f, s) in enumerate(people):
            t0 = 2.2 + 0.35 * k
            body = [(x - 34 * s, y + 46 * s), (x - 26 * s, y - 2 * s), (x - 8 * s, y - 26 * s),
                    (x + 12 * s, y - 24 * s), (x + 30 * s, y + 4 * s), (x + 44 * s, y + 46 * s)]
            body = [(x + (bx - x) * f, by) for bx, by in body]
            T.stroke(wash(ww, wh, U(body), ink=0.5, soft=u(2), seed=200 + k), t0, t0 + 0.35)
            hx, hy = x + 2 * f * s, y - 40 * s
            self.heads.append((hx, hy, f))
            T.stroke(B([(hx - 9 * s, hy - 4 * s), (hx, hy - 12 * s), (hx + 9 * s, hy - 3 * s), (hx + 6 * s, hy + 9 * s),
                        (hx - 6 * s, hy + 9 * s)], 13 * s, ink=0.85, dry=0.15, profile=BLUNT, seed=210 + k),
                     t0 + 0.2, t0 + 0.45)
            T.stroke(B([(hx - 2 * f * s, hy - 14 * s), (hx - 4 * f * s, hy - 22 * s)], 7 * s, ink=0.9, dry=0.2,
                       profile=BLUNT, seed=220 + k), t0 + 0.4, t0 + 0.5)          # topknot
            T.stroke(B([(x - 30 * f * s, y + 46 * s), (x - 24 * f * s, y + 2 * s), (x - 6 * f * s, y - 24 * s)], 4 * s,
                       ink=0.8, dry=0.35, profile=PRESS_DOWN, seed=230 + k), t0 + 0.3, t0 + 0.55)
            T.stroke(B([(x + 6 * f * s, y - 6 * s), (x + 28 * f * s, y + 10 * s), (x + 46 * f * s, y + 2 * s)], 5 * s,
                       ink=0.75, dry=0.3, profile=PRESS_DOWN, seed=240 + k), t0 + 0.45, t0 + 0.65)   # arm
        fx, fy = self.fire
        for j in range(6):           # firewood
            a = j / 6 * np.pi
            T.stroke(B([(fx - np.cos(a) * 52, fy + 26 - np.sin(a) * 9), (fx + np.cos(a) * 46, fy + 14 + np.sin(a) * 7)],
                       6.5, ink=0.8, dry=0.4, profile=BLUNT, seed=250 + j), 3.4 + 0.04 * j, 3.6 + 0.04 * j)
        T.stroke(wash(ww, wh, U([(fx - 70, fy + 30), (fx + 70, fy + 30), (fx + 52, fy + 44), (fx - 52, fy + 44)]),
                      ink=0.25, soft=u(4), seed=259), 3.3, 3.5)
        # ground strokes in front
        for j in range(18):
            x = rng.uniform(40, 1040)
            y = rng.uniform(1300, 1360)
            T.stroke(B([(x, y), (x + rng.uniform(30, 80), y + rng.normal(0, 3))], rng.uniform(2, 4), ink=0.4,
                       dry=0.65, profile=SWEEP, seed=260 + j), 3.0 + j * 0.03, 3.2 + j * 0.03)

        # ---- near layer: an overhanging branch top-left, a dark rock bottom-right (parallax)
        p = Painting(ww, wh, seed=33)
        self.near = np.zeros((wh, ww), np.float32)
        for st in [brush_stroke(ww, wh, U([(-30, 180), (140, 210), (300, 260), (430, 320), (520, 352)]), u(16),
                                ink=0.85, dry=0.45, profile=PRESS_DOWN, seed=300),
                   brush_stroke(ww, wh, U([(170, 214), (230, 300), (260, 372)]), u(7), ink=0.75, dry=0.5,
                                profile=PRESS_DOWN, seed=301),
                   brush_stroke(ww, wh, U([(330, 268), (380, 220), (450, 200)]), u(6), ink=0.75, dry=0.5,
                                profile=PRESS_DOWN, seed=302)]:
            _blit(self.near, st)
        for j in range(60):
            cx, cy = rng.choice([(220, 250), (300, 300), (420, 220), (480, 360), (120, 220), (250, 370)])
            cx, cy = cx + rng.normal(0, 34), cy + rng.normal(0, 22)
            ang = rng.uniform(0.3, 2.6)
            _blit(self.near, brush_stroke(ww, wh, U([(cx, cy), (cx + np.cos(ang) * 22, cy + np.sin(ang) * 22)]),
                                          u(rng.uniform(7, 12)), ink=rng.uniform(0.45, 0.8), dry=0.3,
                                          profile=PRESS_DOWN, seed=310 + j))
        # foreground: a dark rock bottom-right and grass bottom-left, under the captions and the app's buttons
        rock = [(640, 1920), (690, 1790), (760, 1712), (850, 1690), (905, 1640), (985, 1626), (1050, 1650),
                (1080, 1668), (1080, 1920)]
        rk = wash(ww, wh, U(rock), ink=0.5, soft=u(4), seed=35, grain=0.5)
        yy = np.linspace(0, 1, rk.dens.shape[0])[:, None]
        rk.dens *= (0.35 + 0.65 * yy ** 0.6)          # lit from above: pale top, dark foot
        _blit(self.near, rk)
        _blit(self.near, brush_stroke(ww, wh, U([(650, 1900), (692, 1792), (762, 1714), (850, 1692), (905, 1642),
                                                 (985, 1628), (1080, 1668)]), u(7), ink=0.75, dry=0.55,
                                      profile=PRESS_DOWN, seed=37))
        for j in range(16):
            x0 = rng.uniform(700, 1060)
            y0 = 1700 + rng.uniform(0, 80)
            _blit(self.near, brush_stroke(ww, wh, U([(x0, y0), (x0 - 18 + rng.normal(0, 8), y0 + rng.uniform(60, 140))]),
                                          u(rng.uniform(3, 6)), ink=0.4, dry=0.75, profile=PRESS_DOWN, seed=38 + j))
        for j in range(40):
            x = rng.uniform(0, 420)
            y = 1920 - rng.uniform(0, 330) * (1 - x / 520)
            ang = -np.pi / 2 + rng.normal(0, 0.35)
            ln = rng.uniform(60, 160)
            _blit(self.near, brush_stroke(ww, wh, U([(x, y), (x + np.cos(ang) * ln * 0.5 + 10, y + np.sin(ang) * ln * 0.6),
                                                      (x + np.cos(ang) * ln + 25, y + np.sin(ang) * ln)]),
                                          u(rng.uniform(4, 8)), ink=rng.uniform(0.45, 0.8), dry=0.45,
                                          profile=PRESS_DOWN, seed=60 + j))

        # ---- speech: fine lines drifting up between speakers, then dissolving into the air
        self.speech = []
        L1, L2 = self.L[0], self.L[1]
        plan = [(0, 2, 5.3), (2, 3, 6.4), (1, 0, 7.6), (3, 1, 8.6), (0, 2, 9.9), (2, 0, 10.9), (1, 3, 11.8)]
        for k, (a, b, ts) in enumerate(plan):
            ax, ay, af = self.heads[a]
            bx, by, bf = self.heads[b]
            pts = curve((ax + 34 * af, ay - 10), (bx + 34 * bf, by - 10), -110 - 35 * (k % 3), 10)
            st = brush_stroke(ww, wh, U(pts), u(3.0), ink=0.62, dry=0.35, profile=RISE, seed=400 + k, wet=0.1)
            st2 = brush_stroke(ww, wh, U(curve((ax + 34 * af, ay - 30), (bx + 34 * bf, by - 30), -140 - 35 * (k % 3), 10)),
                               u(1.5), ink=0.4, dry=0.45, profile=RISE, seed=420 + k, wet=0.1)
            for s_ in (st, st2):
                if s_ is not None:
                    s_.soft = blur(s_.dens, u(5))
                    self.speech.append((s_, ts, ts + 0.9))
        # song: one long wavering line rising from the fire circle (on "song")
        ts = self.wt(2, 'song')
        xs = np.linspace(380, 720, 16)
        pts = [(x, 1040 - 22 * np.sin(i * 1.3) - i * 4) for i, x in enumerate(xs)]
        st = brush_stroke(ww, wh, U(pts), u(2.2), ink=0.5, dry=0.4, profile=RISE, seed=440, wet=0.1)
        st.soft = blur(st.dens, u(6))
        self.speech.append((st, ts - 0.2, ts + 1.2))

        # smoke texture: tall noise, scrolled upward
        self.smoke = noise(600, 160, 40, 4, 41)
        self.fire_noise = np.random.default_rng(42).random(4000)
        self.mist = noise(wh // 4, ww // 2, 70, 4, 43)
        self.camk = Keys([(-0.8, (u(540), u(900), 0.81)), (4.5, (u(540), u(960), 0.85)),
                          (self.dur + 1.2, (u(570), u(1080), 1.0))])
        self.ref = tuple(self.camk(0))

    def cam(self, t):
        return self.clamp(self.camk(t))

    def enter(self, t, p):
        return _enter_from_stroke(self, p)

    def frame(self, t):
        cam = self.cam(t)
        u = self.u
        D, R = self.track.density(t)
        # speech lines
        for st, t0, t1 in self.speech:
            if t < t0:
                continue
            f = np.clip((t - t0) / (t1 - t0), 0, 1)
            dens = st.reveal(f, soft=0.08)
            age = t - t1
            if age > 0:
                fade = smooth(age / 1.4)
                dens = (dens * (1 - fade) + st.soft * fade * 1.4) * (1 - fade)
            if dens.max() < 0.004:
                continue
            lift = int(u(18) * max(0, t - t0) / 2.0)
            y0 = st.y0 - lift
            h, w = dens.shape
            if y0 < 0:
                continue
            buf = D[y0:y0 + h, st.x0:st.x0 + w]
            np.copyto(buf, 1 - (1 - buf) * (1 - dens))
        # smoke above the fire
        fx, fy = self.u(*self.fire)
        if t > 3.6:
            k = int((t * 26) % 600)
            sm = np.roll(self.smoke, -k, axis=0)
            hh, w = sm.shape
            env = np.linspace(1, 0, hh)[:, None] ** 1.5 * np.exp(-((np.arange(w) - w / 2) / (w / 5)) ** 2)[None, :]
            sm = np.clip(sm - 0.45, 0, 1) * env * 0.5 * ramp(t, 3.6, 5)
            big = np.asarray(Image.fromarray(sm.astype(np.float32), 'F').resize((int(u(160)), int(u(600)))))
            x0, y0 = int(fx - big.shape[1] / 2 + u(10) * np.sin(t)), int(fy - u(20) - big.shape[0])
            D[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = 1 - (1 - D[y0:y0 + big.shape[0], x0:x0 + big.shape[1]]) * (1 - big)
        bgc = parallax(cam, self.ref, 0.55)
        fgc = parallax(cam, self.ref, 1.35)
        Dv = view(D, cam)
        over(Dv, view(self.mid, cam))
        far = view(self.far, bgc)
        # drifting mist over the far mountains
        mist = np.asarray(Image.fromarray(self.mist, 'F').resize((W * 2, H), Image.BILINEAR))
        off = int((t * 14) % W)
        mv = mist[:, off:off + W]
        far *= 1 - 0.55 * np.clip(mv - 0.35, 0, 1) * 1.6
        Dv = 1 - (1 - Dv) * (1 - far * (1 - Dv))
        over(Dv, view(self.near, fgc))
        img = compose(view_rgb(self.pap, cam), Dv, view(R, cam))
        # fire: a flickering vermilion flame and warm light
        if t > 3.5:
            sx, sy = to_screen(cam, fx, fy)
            n = self.fire_noise
            i = int(t * 12) % 3900
            amp = ramp(t, 3.5, 4.3) * (0.8 + 0.2 * n[i])
            YY, XX = np.ogrid[int(sy) - 220:int(sy) + 140, int(sx) - 220:int(sx) + 220]
            dx, dy = (XX - sx), (YY - sy)
            z = cam[2]
            fl = np.zeros(np.broadcast(dx, dy).shape, np.float32)
            up = np.maximum(-dy, 0).astype(np.float32)
            for j in range(4):
                hgt = (62 + 34 * n[i + j * 7]) * z * 1.25 * (1.0 if j in (1, 2) else 0.7)
                wd = (15 + 5 * n[i + j * 3]) * z * 1.25
                ox = (j - 1.5) * 16 * z + 6 * np.sin(t * 7 + j * 1.7)
                sway = 14 * z * np.sin(t * 5.3 + j * 2.1)
                k = np.clip(up / hgt, 0, 1)
                width = wd * (1 - k) ** 0.75 * (0.75 + 0.6 * np.sin(np.pi * np.clip(k * 1.6, 0, 1)))
                d = np.abs(dx - ox - sway * k ** 2) - width
                tongue = np.clip(0.5 - d / 2.5, 0, 1) * (up < hgt) * np.clip(1 - dy / (8 * z), 0, 1)
                fl = np.maximum(fl, tongue * (0.75 + 0.25 * (1 - k)))
            fl = fl * amp
            g = np.exp(-(dx ** 2 + (dy + 30) ** 2) / (2 * (140 * z) ** 2)) * 0.2 * amp
            y0, x0 = int(sy) - 220, int(sx) - 220
            sl = img[max(0, y0):y0 + 360, max(0, x0):x0 + 440]
            fl, g = fl[max(0, -y0):, max(0, -x0):][:sl.shape[0], :sl.shape[1]], g[max(0, -y0):, max(0, -x0):][:sl.shape[0], :sl.shape[1]]
            sl[:] = glow(sl, g, (1.0, 0.62, 0.32))
            sl[:] = sl * (1 - fl[..., None] * (1 - np.array([0.86, 0.32, 0.16])))
            core = np.clip(fl - 0.6, 0, 1) * 1.8
            sl[:] = glow(sl, core, (1.0, 0.85, 0.55))
        return img


def _enter_from_stroke(scene, p):
    """the opening brushstroke (a line across the frame at y ~ 1265) bleeds out into the next landscape"""
    if not hasattr(scene, '_front'):
        scene._front = InkFront(H // 4, W // 4, seeds=[(x / 4, (1265 - 0.03 * x) / 4, 1.0) for x in range(0, W, 40)],
                                rough=0.3, scale=80, soft=0.1, seed=29)
    return np.asarray(Image.fromarray(scene._front.mask(ease(p)), 'F').resize((W, H), Image.BILINEAR), np.float32)


def _blit(buf, st):
    if st is None:
        return
    h, w = st.dens.shape
    b = buf[st.y0:st.y0 + h, st.x0:st.x0 + w]
    np.copyto(b, 1 - (1 - b) * (1 - st.dens))

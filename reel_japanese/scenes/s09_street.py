"""Scene 9 — past and present.  A traditional street in one-point perspective, painted in the same sumi-e hand
with a little more detail: machiya facades with lattices and deep eaves, a five-storey pagoda in the mist at the
end of the street, red paper lanterns swaying, a traveller in a straw hat walking away, a cherry branch in the
foreground shedding petals.  The camera tracks slowly forward; a final brushstroke passes across the frame."""
import cv2
import numpy as np

from art import blossom_cluster, mist_band
from engine import (GREY, H, INK, PAPER, SAKURA, SEPIA, VERM, W, Cam, Frame, StrokeSet, blur, cached, ease_io, keyframes,
                    noise, paste, ramp, reveal_time, smooth, spline, sstep, stroke, warp, wash_map, wipe_mix)
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash
from scenes.base import SceneBase
from timeline import at

VP = (540.0, 960.0)
REF = (540, 960)


def _ytop(x):
    return 380 + (x + 40) / 580 * (VP[1] - 380)


def _ybot(x):
    return 1640 + (x + 40) / 580 * (VP[1] - 1640)


class Scene(SceneBase):
    paper_seed = 19

    def build(self):
        A = cached('s09_street_v2', self._paint)
        self.far, self.mid, self.fg, self.road = A['far'], A['mid'], A['fg'], A['road']
        self.blossom = A['blossom']
        self.walker = A['walker']
        self.T = wash_map(H, W, (540, 1100), noise_amp=0.5, noise_scale=130, seed=7)
        self.mist = mist_band(H, W, 930, 60, seed=9, strength=0.9)
        # lanterns: (world x, y of the eave, size), on both sides
        self.lanterns = []
        for x in (150, 300, 395):
            for side in (-1, 1):
                xx = x if side < 0 else 1080 - x
                yt, yb = _ytop(x), _ybot(x)
                self.lanterns.append((xx, yt + 0.40 * (yb - yt) + 4, (yb - yt) * 0.075, x * 0.01 + side))
        r = np.random.default_rng(3)
        self.petals = [(r.uniform(-50, 700), r.uniform(-200, 900), r.uniform(0.6, 1.2), r.uniform(0, 6)) for _ in range(16)]

    # ------------------------------------------------------------ painting
    def _paint(self):
        from inkpaint import Painting
        p = Painting(W, H, seed=41)
        far = p.ridge(860, peaks=[(160, 140, 200), (900, 170, 180)], rough=12, seed=2)
        p.wash(far, dens=0.12, decay=110, tex=90, edge=0.25, mist_y=930, mist_h=60, seed=3)
        ss = StrokeSet(W, H)
        self._pagoda(ss, 540, 925, 1.0)
        D_far = np.clip(1 - (1 - p.D) * (1 - ss.render(9)), 0, 1)

        mid = StrokeSet(W, H)
        r = np.random.default_rng(5)
        xs = [-40, 130, 255, 345, 408, 448, 474, 492]
        k = 0
        for side in (-1, 1):
            X = (lambda x: x) if side < 0 else (lambda x: 1080 - x)
            for a, b in zip(xs, xs[1:]):
                ya_t, ya_b, yb_t, yb_b = _ytop(a), _ybot(a), _ytop(b), _ybot(b)
                ha = ya_b - ya_t
                hb = yb_b - yb_t
                # roof: a dark band leaning out over the street, with tile hatching
                roof = [(X(a), ya_t + 2), (X(b), yb_t + 2), (X(b), yb_t - 0.13 * hb), (X(a), ya_t - 0.13 * ha)]
                mid.add(pwash(W, H, roof, ink=0.55, soft=1.5, seed=k), 0, 1)
                for u in np.linspace(0.05, 0.95, max(3, int((b - a) / 9))):
                    x = a + (b - a) * u
                    yt, hh = _ytop(x), _ybot(x) - _ytop(x)
                    mid.add(stroke(W, H, [(X(x), yt - 0.12 * hh), (X(x), yt)], max(1.2, hh * 0.006), ink=0.75, dry=0.4,
                                   profile=BLUNT, seed=1000 + k * 50 + int(u * 40)), 0, 1)
                mid.add(stroke(W, H, [(X(a), ya_t + 1), (X(b), yb_t + 1)], max(2, ha * 0.02), ink=0.9, dry=0.3,
                               profile=BLUNT, seed=2000 + k), 0, 1)
                # first-floor eave
                ma, mb = ya_t + 0.40 * ha, yb_t + 0.40 * hb
                mid.add(pwash(W, H, [(X(a), ma), (X(b), mb), (X(b), mb - 0.06 * hb), (X(a), ma - 0.06 * ha)], ink=0.45, soft=1.2,
                              seed=3000 + k), 0, 1)
                mid.add(stroke(W, H, [(X(a), ma), (X(b), mb)], max(1.5, ha * 0.012), ink=0.85, dry=0.35, profile=BLUNT, seed=3100 + k), 0, 1)
                # posts
                for x, yt, yb in ((a, ya_t, ya_b), (b, yb_t, yb_b)):
                    mid.add(stroke(W, H, [(X(x), yt), (X(x), yb)], max(1.5, (yb - yt) * 0.016), ink=0.8, dry=0.35,
                                   profile=BLUNT, seed=4000 + k + int(x)), 0, 1)
                # lattice on the ground floor, slatted window above
                n = max(4, int((b - a) / 11))
                for u in np.linspace(0.06, 0.94, n):
                    x = a + (b - a) * u
                    yt, yb = _ytop(x), _ybot(x)
                    hh = yb - yt
                    mid.add(stroke(W, H, [(X(x), yt + 0.47 * hh), (X(x), yb - 0.04 * hh)], max(0.8, hh * 0.0035), ink=0.3,
                                   dry=0.45, profile=BLUNT, seed=5000 + k * 97 + int(u * 90)), 0, 1)
                    if 0.25 < u < 0.75:
                        mid.add(stroke(W, H, [(X(x), yt + 0.14 * hh), (X(x), yt + 0.30 * hh)], max(0.8, hh * 0.005), ink=0.55,
                                       dry=0.4, profile=BLUNT, seed=6000 + k * 97 + int(u * 90)), 0, 1)
                # a noren curtain in some doorways
                if r.random() < 0.6 and b - a > 25:
                    u0, u1 = 0.3, 0.62
                    x0, x1 = a + (b - a) * u0, a + (b - a) * u1
                    t0, t1 = _ytop(x0), _ytop(x1)
                    h0, h1 = _ybot(x0) - t0, _ybot(x1) - t1
                    mid.add(pwash(W, H, [(X(x0), t0 + 0.47 * h0), (X(x1), t1 + 0.47 * h1), (X(x1), t1 + 0.66 * h1),
                                         (X(x0), t0 + 0.66 * h0)], ink=0.62, soft=1.2, seed=7000 + k), 0, 1)
                k += 1
        D_mid = mid.render(9)
        # road: converging joints of a stone path
        road = np.zeros((H, W), np.float32)
        for xb in np.linspace(-300, 1380, 11):
            paste(road, stroke(W, H, [(xb, 1920), (VP[0] + (xb - VP[0]) * 0.18, 1080)], 3, ink=0.18, dry=0.6, profile=SWEEP, seed=int(xb) + 900))
        for y in (1180, 1290, 1430, 1620, 1850):
            hw = (y - VP[1]) * 0.75
            paste(road, stroke(W, H, [(540 - hw, y), (540 + hw, y + 3)], 2.5, ink=0.08, dry=0.65, profile=SWEEP, seed=y))
        # foreground cherry branch from the top left
        fg = np.zeros((H, W), np.float32)
        br = spline([(-30, 330), (120, 360), (260, 330), (390, 380), (470, 360)], 12)
        paste(fg, stroke(W, H, br, 22, ink=0.85, dry=0.45, profile=((0, 1), (0.8, 0.6), (1, 0.15)), seed=11))
        for (a, b2) in [((180, 350), (240, 450)), ((300, 345), (330, 270)), ((390, 378), (440, 450)), ((120, 358), (90, 440))]:
            paste(fg, stroke(W, H, [a, b2], 8, ink=0.8, dry=0.45, profile=PRESS_DOWN, seed=a[0] + 3))
        bl = np.zeros((H, W), np.float32)
        for cx, cy, n, rad in [(240, 450, 30, 40), (330, 280, 22, 34), (440, 450, 26, 36), (90, 440, 24, 36), (470, 360, 18, 30),
                               (180, 330, 16, 30)]:
            bl = np.maximum(bl, blossom_cluster(H, W, cx, cy, n, rad, seed=cx))
        # the traveller (seen from behind): straw hat, robe, staff, on a small canvas
        wk = StrokeSet(240, 360)
        circ = lambda cx, cy, rr: [(cx + rr * np.cos(q), cy + rr * np.sin(q)) for q in np.linspace(0, 2 * np.pi, 18, endpoint=False)]
        wk.add(pwash(240, 360, [(104, 112), (136, 112), (160, 290), (80, 290)], ink=0.5, soft=2, seed=3, grain=0.4), 0, 1)   # robe
        wk.add(pwash(240, 360, circ(120, 100, 13), ink=0.85, soft=1.2, seed=8), 0, 1)                                     # head
        wk.add(pwash(240, 360, [(120, 50), (182, 86), (174, 92), (66, 92), (58, 86)], ink=0.7, soft=1.4, seed=1), 0, 1)   # kasa
        wk.add(stroke(240, 360, [(56, 88), (120, 80), (184, 88)], 5, ink=0.9, dry=0.3, profile=BLUNT, seed=2), 0, 1)
        wk.add(pwash(240, 360, [(98, 118), (142, 118), (148, 168), (92, 168)], ink=0.62, soft=1.5, seed=9), 0, 1)          # bundle
        wk.add(stroke(240, 360, [(104, 114), (88, 200), (80, 290)], 6, ink=0.85, dry=0.35, profile=PRESS_DOWN, seed=4), 0, 1)
        wk.add(stroke(240, 360, [(136, 114), (152, 200), (160, 290)], 5, ink=0.8, dry=0.4, profile=PRESS_DOWN, seed=5), 0, 1)
        wk.add(stroke(240, 360, [(104, 288), (100, 336)], 7, ink=0.8, dry=0.3, profile=BLUNT, seed=10), 0, 1)              # legs
        wk.add(stroke(240, 360, [(136, 288), (142, 330)], 7, ink=0.8, dry=0.3, profile=BLUNT, seed=11), 0, 1)
        wk.add(stroke(240, 360, [(172, 130), (180, 240), (186, 348)], 4, ink=0.85, dry=0.3, profile=BLUNT, seed=7), 0, 1)  # staff
        walker = wk.render(9)
        f32 = lambda a: a.astype(np.float32)
        return {'far': f32(D_far), 'mid': f32(D_mid), 'fg': f32(fg), 'road': f32(road), 'blossom': f32(bl), 'walker': f32(walker)}

    def _pagoda(self, ss, x, y, s):
        """five-storey pagoda, base at (x, y)"""
        W_, H_ = ss.w, ss.h
        h = 0
        for k in range(5):
            wr = (78 - 9 * k) * s
            yb = y - h
            body_h = (26 - 2 * k) * s
            ss.add(pwash(W_, H_, [(x - wr * 0.42, yb), (x + wr * 0.42, yb), (x + wr * 0.4, yb - body_h), (x - wr * 0.4, yb - body_h)],
                         ink=0.22, soft=1.2, seed=k), 0, 1)
            ry = yb - body_h
            roof = spline([(x - wr - 8 * s, ry - 8 * s), (x - wr * 0.6, ry - 2 * s), (x, ry - 12 * s), (x + wr * 0.6, ry - 2 * s),
                           (x + wr + 8 * s, ry - 8 * s)], 8)
            ss.add(stroke(W_, H_, roof, 7 * s, ink=0.55, dry=0.35, profile=BLUNT, seed=10 + k), 0, 1)
            ss.add(pwash(W_, H_, [(x - wr, ry - 6 * s), (x + wr, ry - 6 * s), (x + wr * 0.45, ry - 22 * s), (x - wr * 0.45, ry - 22 * s)],
                         ink=0.3, soft=1.2, seed=20 + k), 0, 1)
            h += body_h + 22 * s
        ss.add(stroke(W_, H_, [(x, y - h), (x, y - h - 70 * s)], 3 * s, ink=0.55, dry=0.2, profile=BLUNT, seed=40), 0, 1)
        for j in range(6):
            yy = y - h - 12 * s - j * 9 * s
            ss.add(stroke(W_, H_, [(x - 7 * s, yy), (x + 7 * s, yy)], 2.2 * s, ink=0.5, dry=0.2, profile=BLUNT, seed=50 + j), 0, 1)

    # ------------------------------------------------------------ motion
    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.0)), (self.t1, (540, 960, 1.2))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam, ref=REF)
        fw = ramp(t, self.t0 - 0.3, self.t0 + 1.4)
        rv = lambda D: reveal_time(D, self.T, fw * 1.08, soft=0.07, wet=0.3) if fw < 1 else D
        self.layer(fr, cam, rv(self.far), depth=0.35, ref=REF)
        m = self.project(cam, np.roll(self.mist, int(t * 12) % W, axis=1), depth=0.5, ref=REF)
        fr.img = fr.img * (1 - m[..., None] * 0.85) + PAPER * m[..., None] * 0.85
        self.layer(fr, cam, rv(self.road), depth=1.0, ref=REF)
        self.layer(fr, cam, rv(self.mid), depth=1.0, ref=REF)
        self._lanterns(fr, cam, t, fw)
        self._walker(fr, cam, t, fw)
        # foreground branch, nearest the lens, swaying a little
        k = 0.012 * np.sin(t * 0.9)
        M = cam.matrix(1.45, REF)
        S = np.array([[1, 0, 0], [k, 1, -k * 0]], np.float32)
        Mt = np.concatenate([M[:, :2] @ S[:, :2], (M[:, :2] @ S[:, 2] + M[:, 2])[:, None]], 1)
        fo = smooth((t - self.t0 - 0.4) / 1.0)
        fr.ink(warp(self.blossom, Mt), SAKURA, fo)
        fr.ink(warp(self.fg, Mt), INK, fo)
        self._petals(fr, t, fo)
        return fr.img

    def _lanterns(self, fr, cam, t, fw):
        M = cam.matrix(1.0, REF)
        R = np.zeros((H, W), np.float32)
        D = np.zeros((H, W), np.float32)
        G = np.zeros((H, W), np.float32)
        for k, (x, y, s, ph) in enumerate(self.lanterns):
            sx, sy = M[:, :2] @ np.array([x, y]) + M[:, 2]
            z = M[0, 0]
            s = s * z
            ang = 0.08 * np.sin(t * 1.3 + ph * 2.1)
            cord = s * 0.5
            cx, cy = sx + np.sin(ang) * (cord + s), sy + np.cos(ang) * (cord + s)
            cv2.line(D, (int(sx), int(sy)), (int(cx - np.sin(ang) * s), int(cy - np.cos(ang) * s)), 0.8, max(1, int(s * 0.06)), cv2.LINE_AA)
            cv2.ellipse(R, (int(cx), int(cy)), (int(s * 0.62), int(s * 0.9)), np.degrees(-ang), 0, 360, 0.85, -1, cv2.LINE_AA)
            cv2.ellipse(G, (int(cx), int(cy)), (int(s * 1.6), int(s * 1.9)), 0, 0, 360, 0.35, -1, cv2.LINE_AA)
            for cap in (-1, 1):
                px, py = cx - cap * np.sin(ang) * s * 0.9, cy - cap * np.cos(ang) * s * 0.9
                cv2.ellipse(D, (int(px), int(py)), (int(s * 0.36), int(s * 0.13)), np.degrees(-ang), 0, 360, 0.9, -1, cv2.LINE_AA)
            for rib in (-0.45, 0, 0.45):
                cv2.ellipse(D, (int(cx), int(cy + rib * s)), (int(s * 0.6 * np.sqrt(1 - rib ** 2)), int(s * 0.05) + 1),
                            np.degrees(-ang), 0, 180, 0.35, 1, cv2.LINE_AA)
        op = smooth(fw * 1.5 - 0.5)
        fr.img = fr.img * (1 - blur(G, 10)[..., None] * 0.25 * op) + np.array([1.0, 0.86, 0.66]) * blur(G, 10)[..., None] * 0.25 * op
        fr.ink(blur(R, 0.7), VERM, op)
        fr.ink(blur(D, 0.6), INK, op)

    def _walker(self, fr, cam, t, fw):
        u = (t - self.t0) / (self.t1 - self.t0)
        y = 1560 - 230 * u
        s = 0.62 - 0.22 * u
        bob = 3 * abs(np.sin(t * 4.2))
        sway = 2 * np.sin(t * 2.1)
        h, w = self.walker.shape
        M = cam.matrix(1.0, REF)
        A = np.array([[s, 0, 540 + sway - s * w / 2], [0, s, y - bob - s * h]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(self.walker, Mt), INK, smooth(fw * 1.4 - 0.4))

    def _petals(self, fr, t, op):
        P = np.zeros((H, W), np.float32)
        for x0, y0, sp, ph in self.petals:
            y = (y0 + (t - self.t0) * 70 * sp) % 1500
            x = x0 + (t - self.t0) * 22 * sp + 25 * np.sin(t * 1.2 + ph)
            ang = (t * 2 + ph) * 57
            cv2.ellipse(P, (int(x), int(y)), (7, 4), ang, 0, 360, 0.55, -1, cv2.LINE_AA)
        fr.ink(blur(P, 0.7), SAKURA, op)

    def enter(self, prev_img, cur_img, f, prev_scene):
        """the sentence fades through a spreading ink wash that opens onto the street"""
        T = wash_map(H, W, (540, 1000), noise_amp=0.45, noise_scale=120, seed=99)
        return wipe_mix(prev_img, cur_img, T, ease_io(f) * 1.15 - 0.05, soft=0.045, edge=0.4, edge_color=GREY)

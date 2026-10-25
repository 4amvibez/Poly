"""Scene 10 — Japan today.  A last brushstroke passes across the old street and opens onto photographs of modern
Japan (the user's own photos): an Ueno alley, a road sign, Shibuya's signboards, a Tokyo night street.  Each gets a
slow push-in, joined by ink wipes.  On "Three scripts" the Tokyo street's signs are ringed in vermilion and named:
カラオケ (katakana), 居酒屋 (kanji), やきとり (hiragana) — the history of the film, still written on the street."""
import cv2
import numpy as np

from engine import (FONT_EN_ITALIC, FONT_JP, H, INK, PAPER, W, Cam, Frame, blur, ease_io, ease_out, noise, reveal_time,
                    smooth, spline, sstep, stroke, text_layer, wipe_mix)
from glyphs import text_glyphs
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

PHOTOS = [  # file, (zoom start, zoom end), (pan start, pan end) as fractions of the spare margin
    ('ueno_alley.jpg', (1.0, 1.10), ((0.5, 0.55), (0.5, 0.45))),
    ('road_signs.jpg', (1.04, 1.12), ((0.5, 0.35), (0.5, 0.45))),
    ('shibuya_signs.jpg', (1.0, 1.08), ((0.35, 0.5), (0.6, 0.5))),
    ('tokyo_night.jpg', (1.0, 1.035), ((0.5, 0.42), (0.5, 0.48))),
]
# signs on the Tokyo night photo, in its own pixels (736x1307): box, label, script name
SIGNS = [((90, 4, 194, 336), 'カタカナ', 'katakana'),
         ((100, 440, 167, 606), '漢字', 'kanji'),
         ((33, 660, 103, 790), 'ひらがな', 'hiragana')]
RING = np.array([0.98, 0.95, 0.87], np.float32)        # ivory brush ring: reads against red and blue neon


def _cover(img, w, h):
    s = max(w / img.shape[1], h / img.shape[0])
    big = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_LANCZOS4)
    big = cv2.addWeighted(big, 1.35, cv2.GaussianBlur(big, (0, 0), 1.2), -0.35, 0)        # recover some upscale softness
    return big, s


class Scene(SceneBase):
    paper_seed = 21

    def build(self):
        import os
        from engine import HERE
        self.ph = []
        for f, zz, pp in PHOTOS:
            img = cv2.imread(os.path.join(HERE, 'photos', f))[:, :, ::-1].astype(np.float32) / 255
            big, s = _cover(img, W, H)
            # a gentle grade so the photos sit with the paper: slightly warm, a touch less saturated
            g = big.mean(2, keepdims=True)
            big = np.clip((big * 0.88 + g * 0.12) * np.array([1.03, 1.0, 0.95]), 0, 1).astype(np.float32)
            self.ph.append(dict(img=big, s=s, zoom=zz, pan=pp))
        a0 = at('And you can')[0]
        three = at('Three scripts', 1)[0]
        self.starts = [self.t0, a0 + 1.15, a0 + 2.2, three - 0.3]
        self.t_three = three
        self.wipe_T = [None] + [self._wipe(k) for k in range(1, 4)]
        self.labels = []
        for k, (box, jp, en) in enumerate(SIGNS):
            D, T, _ = text_glyphs(jp, 46, font=FONT_JP, seed=70 + k)
            lab, _ = text_layer(en, FONT_EN_ITALIC, 38)
            self.labels.append(dict(box=box, D=D, T=T, en=lab, t=three - 0.05 + 0.32 * k))

    def _wipe(self, k):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ang = [0.0, 0.6, -0.5, 0.25][k]
        d = (xx * np.cos(ang) + yy * np.sin(ang))
        d = (d - d.min()) / (d.max() - d.min())
        return np.clip(d * 0.8 + (noise(H, W, 60, 4, 300 + k) - 0.5) * 0.3, 0, 1)

    def _matrix(self, k, t):
        p = self.ph[k]
        t0 = self.starts[k]
        t1 = self.starts[k + 1] + 0.4 if k + 1 < len(self.starts) else self.t1
        u = ease_io((t - t0) / max(t1 - t0, 0.1))
        z = p['zoom'][0] + (p['zoom'][1] - p['zoom'][0]) * u
        (px0, py0), (px1, py1) = p['pan']
        px, py = px0 + (px1 - px0) * u, py0 + (py1 - py0) * u
        ih, iw = p['img'].shape[:2]
        vw, vh = W / z, H / z                                  # visible window in image pixels
        cx = vw / 2 + (iw - vw) * px
        cy = vh / 2 + (ih - vh) * py
        return np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)

    def _photo(self, k, t):
        M = self._matrix(k, t)
        return cv2.warpAffine(self.ph[k]['img'], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def render(self, t):
        k = max(i for i, s in enumerate(self.starts) if t >= s or i == 0)
        img = self._photo(k, t)
        if k > 0 and t < self.starts[k] + 0.35:                # ink wipe from the previous photo
            prev = self._photo(k - 1, t)
            img = wipe_mix(prev, img, self.wipe_T[k], ease_io((t - self.starts[k]) / 0.35) * 1.15 - 0.05, soft=0.03, edge=0.35)
        # a whisper of paper over the photographs keeps them in the film's world
        fr = Frame(img)
        fr.img = fr.img * (0.9 + 0.1 * (self.paper / self.paper.mean()))
        if k == 3:
            self._rings(fr, t)
        return fr.img

    def _rings(self, fr, t):
        M = self._matrix(3, t)
        s = self.ph[3]['s']
        for L in self.labels:
            f = (t - L['t']) / 0.45
            if f <= 0:
                continue
            x0, y0, x1, y1 = L['box']
            p0 = M[:, :2] @ (np.array([x0, y0]) * s) + M[:, 2]
            p1 = M[:, :2] @ (np.array([x1, y1]) * s) + M[:, 2]
            cx, cy = (p0 + p1) / 2
            rx, ry = (p1 - p0) / 2 + 16
            n = 64
            a = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * 1.06 * min(f, 1), max(2, int(n * min(f, 1))))
            # a soft rounded rectangle (superellipse), brushed on clockwise
            e = 4
            pts = np.stack([cx + rx * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** (2 / e),
                            cy + ry * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** (2 / e)], 1)
            ring = np.zeros((H, W), np.float32)
            cv2.polylines(ring, [np.round(pts * 4).astype(np.int32)], False, 1.0, 9, cv2.LINE_AA, shift=2)
            dark = blur(ring, 4) * 0.65
            fr.img *= (1 - dark)[..., None]
            ring = blur(ring, 0.8)
            fr.img = fr.img * (1 - ring[..., None]) + RING * ring[..., None]
            # label on a small paper plate to the right of the ring
            op = smooth((f - 0.6) / 0.6)
            if op <= 0:
                continue
            D, en = L['D'], L['en']
            lx = int(p1[0] + 40)
            ly = int(cy)
            w = max(D.shape[1], en.shape[1]) + 40
            h = D.shape[0] + en.shape[0] + 26
            plate = np.zeros((H, W), np.float32)
            plate[max(0, ly - h // 2):ly + h // 2, lx:lx + w] = 1
            plate = blur(plate, 5) * 0.88 * op
            fr.img = fr.img * (1 - plate[..., None]) + PAPER * plate[..., None]
            d = reveal_time(D, L['T'], (f - 0.6) * 1.3, soft=0.05, wet=0.3)
            self._put(fr, d, lx + 20, ly - h // 2 + 10, INK, op)
            self._put(fr, en, lx + 22, ly - h // 2 + 14 + D.shape[0], np.array([0.42, 0.39, 0.36]), op)

    @staticmethod
    def _put(fr, a, x, y, color, op):
        hh, ww = a.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + ww), min(H, y + hh)
        if x1 <= x0 or y1 <= y0:
            return
        sub = a[y0 - y:y1 - y, x0 - x:x1 - x][..., None] * op
        fr.img[y0:y1, x0:x1] *= 1 - sub * (1 - np.asarray(color, np.float32))

    def enter(self, prev_img, cur_img, f, prev_scene):
        """a last brushstroke passes across the old street; behind it, today's Japan"""
        if not hasattr(self, '_pass'):
            st = stroke(W, H, spline([(-80, 1180), (300, 1020), (700, 960), (1160, 820)], 14), 300, ink=0.95, dry=0.6,
                        profile=SWEEP, seed=301)
            from engine import paste
            c = np.zeros((H, W), np.float32)
            paste(c, st)
            T = np.ones((H, W), np.float32) * 2
            hh, ww = st.dens.shape
            T[st.y0:st.y0 + hh, st.x0:st.x0 + ww] = np.where(st.dens > 0.01, st.time, 2)
            xx = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
            sweep_T = np.clip(xx * 0.85 + (noise(H, W, 100, 4, 61) - 0.5) * 0.3, 0, 1)
            self._pass = (c, T, sweep_T)
        c, T, sweep_T = self._pass
        head = f * 1.5
        vis = c * np.clip((head - T) / 0.04 + 0.5, 0, 1) * np.clip((T - (head - 0.55)) / 0.15, 0, 1)
        out = wipe_mix(prev_img, cur_img, sweep_T, f * 1.25 - 0.15, soft=0.08, edge=0.3)
        fr = Frame(out)
        fr.ink(vis, INK)
        return fr.img

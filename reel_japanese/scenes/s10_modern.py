"""Scene 10 — Japan today, in real footage (the user's own clip of Tokyo street life).
A last brushstroke passes across the old street and opens onto rainy Shinjuku; quick cuts follow — a ramen shop,
the BIG ECHO karaoke sign, a lantern-lit yokocho alley — while Raj says "you can still read it on every street in
Japan today".  On "Three scripts" the GiGO sign freezes and settles onto the washi as a print: one sentence,
ゲームのオアシスに飛び込め！ ("Dive into the oasis of games!"), written in all three scripts at once —
katakana ゲーム・オアシス, hiragana の・に・び・め, kanji 飛・込 — each ringed by the brush and named."""
import os
import subprocess

import cv2
import numpy as np

from engine import (CACHE, FONT_EN, FONT_EN_ITALIC, FONT_JP, H, HERE, INK, PAPER, W, Cam, Frame, blur, ease_io, noise,
                    reveal_time, smooth, spline, stroke, text_layer, wipe_mix)
from glyphs import text_glyphs
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at

CLIP = os.path.join(HERE, 'footage', 'tokyo_signs.mp4')
FRAMES = os.path.join(CACHE, 'footage')
CW, CH = 720, 1280
FREEZE_AT = 6.93                      # the GiGO sign, fully readable, just before its LED animation wipes it
# clip time at the start of each slot
SHOTS = [0.03,                        # rainy Shinjuku street
         4.15,                        # ramen shop, ラーメン
         10.48,                       # BIG ECHO karaoke
         11.82,                       # lantern-lit yokocho alley
         6.70]                        # GiGO: ゲームのオアシスに飛び込め！
# glyph boxes on the freeze frame (clip pixels): script, name, ring colour, boxes
GROUPS = [('カタカナ', 'katakana', np.array([0.98, 0.95, 0.87], np.float32), [(196, 60, 436, 202), (196, 200, 484, 302)]),
          ('ひらがな', 'hiragana', np.array([1.0, 0.80, 0.30], np.float32), [(432, 104, 510, 202), (480, 218, 536, 296),
                                                                           (352, 318, 428, 430), (506, 326, 562, 416)]),
          ('漢字', 'kanji', np.array([0.07, 0.06, 0.06], np.float32), [(250, 316, 354, 436), (426, 318, 508, 430)])]
GREY_TXT = np.array([0.38, 0.35, 0.32], np.float32)


def _extract():
    if os.path.isdir(FRAMES) and len(os.listdir(FRAMES)) > 400:
        return
    os.makedirs(FRAMES, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', CLIP, '-q:v', '2', os.path.join(FRAMES, '%04d.jpg')], check=True)


class Scene(SceneBase):
    paper_seed = 21

    def build(self):
        _extract()
        a0 = at('And you can')[0]
        three = at('Three scripts', 1)[0]
        self.starts = [self.t0, a0 + 0.8, a0 + 1.6, a0 + 2.4, a0 + 3.1]          # a cut roughly every 0.8 s
        self.t_freeze = self.starts[4] + (FREEZE_AT - SHOTS[4])
        self.t_print = (self.t_freeze, self.t_freeze + 0.55)
        self.t_three = three
        self.t_one = at('One language')[0]
        self.labels = []
        for k, (jp, en, col, boxes) in enumerate(GROUPS):
            D, T, _ = text_glyphs(jp, 44, font=FONT_JP, seed=80 + k)
            lab, _ = text_layer(en, FONT_EN_ITALIC, 38)
            self.labels.append(dict(jp=jp, D=D, T=T, en=lab, col=col, boxes=boxes, t=three - 0.18 + 0.32 * k))
        self.tr1, _ = text_layer('"Dive into the oasis of games!"', FONT_EN_ITALIC, 44)
        self.tr2, _ = text_layer('one sign  ·  three scripts  ·  one language', FONT_EN, 34)
        self._cache = {}

    # ------------------------------------------------------------ footage
    def _clip_frame(self, ct):
        i = int(round(ct * 30)) + 1
        if i not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            im = cv2.imread(os.path.join(FRAMES, f'{min(i, 461):04d}.jpg'))[:, :, ::-1].astype(np.float32) / 255
            g = im.mean(2, keepdims=True)
            im = np.clip((im * 0.9 + g * 0.1) * np.array([1.03, 1.0, 0.96]), 0, 1)       # the film's gentle grade
            self._cache[i] = im.astype(np.float32)
        return self._cache[i]

    def _shot(self, t):
        k = max(i for i, s in enumerate(self.starts) if t >= s or i == 0)
        ct = SHOTS[k] + (t - self.starts[k])
        if k == 4:
            ct = min(ct, FREEZE_AT)
        return k, ct

    def _print_matrix(self, t):
        """clip pixels -> screen: a full-frame cover that settles into a slightly tilted print on the paper"""
        u = ease_io((t - self.t_print[0]) / (self.t_print[1] - self.t_print[0]))
        k = 1.5 + (1.29 - 1.5) * u
        r = np.radians(-1.2) * u
        cx, cy = 540, 960 + (1075 - 960) * u
        c, s = np.cos(r), np.sin(r)
        A = np.array([[k * c, -k * s], [k * s, k * c]], np.float32)
        t_ = np.array([cx, cy], np.float32) - A @ np.array([CW / 2, CH / 2], np.float32)
        return np.concatenate([A, t_[:, None]], 1), u

    def render(self, t):
        k, ct = self._shot(t)
        im = self._clip_frame(ct)
        if k < 4 or t < self.t_print[0]:
            M = np.array([[1.5, 0, 0], [0, 1.5, 0]], np.float32)
            img = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            fr = Frame(img)
            fr.img = fr.img * (0.92 + 0.08 * (self.paper / self.paper.mean()))
            return fr.img
        # the frozen frame becomes a print laid on the washi
        M, u = self._print_matrix(t)
        fr = self.frame(Cam(540, 960, 1.0))
        corners = np.array([[0, 0], [CW, 0], [CW, CH], [0, CH]], np.float32)
        quad = corners @ M[:, :2].T + M[:, 2]
        cen = quad.mean(0)
        border = cen + (quad - cen) * (1 + 0.022 * u)
        sh = np.zeros((H, W), np.float32)
        cv2.fillPoly(sh, [np.round((border + [12, 18]) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
        fr.img *= (1 - blur(sh, 14) * 0.35 * u)[..., None]
        bd = np.zeros((H, W), np.float32)
        cv2.fillPoly(bd, [np.round(border * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
        fr.over(np.array([0.97, 0.95, 0.9], np.float32), bd, u)
        photo = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC)
        mask = cv2.warpAffine(np.ones((CH, CW), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        fr.img = fr.img * (1 - mask[..., None]) + photo * mask[..., None]
        self._rings(fr, t, M)
        return fr.img

    # ------------------------------------------------------------ the payoff
    def _rings(self, fr, t, M):
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        ang = float(np.arctan2(M[1, 0], M[0, 0]))
        for gi, L in enumerate(self.labels):
            f = (t - L['t']) / 0.35
            if f <= 0:
                continue
            for bi, (x0, y0, x1, y1) in enumerate(L['boxes']):
                fb = float(np.clip(f * 1.3 - bi * 0.25, 0, 1))
                if fb <= 0:
                    continue
                c = M[:, :2] @ np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + M[:, 2]
                rx, ry = (x1 - x0) / 2 * sc + 10, (y1 - y0) / 2 * sc + 10
                a = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * 1.05 * fb, max(2, int(60 * fb)))
                e = 4 if (x1 - x0) > 120 else 2.4
                px = rx * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** (2 / e)
                py = ry * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** (2 / e)
                pts = np.stack([c[0] + px * np.cos(ang) - py * np.sin(ang), c[1] + px * np.sin(ang) + py * np.cos(ang)], 1)
                ring = np.zeros((H, W), np.float32)
                cv2.polylines(ring, [np.round(pts * 4).astype(np.int32)], False, 1.0, 7, cv2.LINE_AA, shift=2)
                halo = blur(ring, 3.5) * 0.55
                halo_col = np.array([0.97, 0.94, 0.86], np.float32) if gi == 2 else np.array([0.05, 0.04, 0.04], np.float32)
                fr.over(halo_col, halo)
                fr.over(L['col'], blur(ring, 0.7))
            # legend line under the sign: a swatch ring, the script in Japanese, its name
            op = smooth((f - 0.5) / 0.5)
            if op <= 0:
                continue
            y = 900 + 80 * gi
            plate = np.zeros((H, W), np.float32)
            plate[y - 34:y + 34, 120:560] = 1
            fr.over(PAPER, blur(plate, 6) * 0.9 * op)
            sw = np.zeros((H, W), np.float32)
            cv2.circle(sw, (165, y), 20, 1.0, 6, cv2.LINE_AA)
            if gi != 2:
                fr.over(np.array([0.1, 0.09, 0.08], np.float32), blur(sw, 2.5) * 0.6 * op)
            fr.over(L['col'], blur(sw, 0.7), op)
            d = reveal_time(L['D'], L['T'], (f - 0.5) * 1.4, soft=0.05, wet=0.3)
            self._put(fr, d, 205, y - d.shape[0] // 2, INK, op)
            self._put(fr, L['en'], 210 + d.shape[1] + 18, y - L['en'].shape[0] // 2 + 4, GREY_TXT, op)
        # "One language": the whole sentence, translated
        ot = smooth((t - self.t_one) / 0.5)
        if ot > 0:
            plate = np.zeros((H, W), np.float32)
            plate[1150:1290, 100:980] = 1
            fr.over(PAPER, blur(plate, 8) * 0.9 * ot)
            self._put(fr, self.tr1, 540 - self.tr1.shape[1] // 2, 1188 - self.tr1.shape[0] // 2, INK, ot)
            self._put(fr, self.tr2, 540 - self.tr2.shape[1] // 2, 1252 - self.tr2.shape[0] // 2, GREY_TXT, ot)

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

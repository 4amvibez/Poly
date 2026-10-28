"""Scene 1 — the hook, a puzzle: "This word uses the same kanji twice... but reads it two different ways.  Why?"
Frame 1: the screen is a field of wet black ink with 日曜日 burst out of it in ivory, droplets flying, the camera
jolting.  In half a second the ink peels back from the centre like a wave drawing off the paper, the kanji turn to
black ink on washi, a vermilion sun blooms behind 曜 and misty hills rise along the bottom.  On "the same kanji
twice" both 日 are ringed, the first in vermilion and the last in indigo, and SAME KANJI. is set above; on "two
different ways" their readings drop in under them, にち (vermilion) and び (indigo), with よう in grey between, and
DIFFERENT READINGS. appears; in the breath after "Why?" a vermilion WHY? is stamped across the bottom (taiko)."""
import numpy as np

import kit
from cards import INDIGO
from engine import FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_HEAD, FONT_JP, GREY, H, INK, VERM, W, Cam, blur, cached, ease_out, noise, smooth, text_layer
from art import sun_disc
import r5art
from glyphs import typeset
from scenes.kscene import KanjiScene
from timeline import at, quiet_gap

XS = [205, 540, 875]
KY = 760
READ = [('にち', VERM), ('よう', GREY), ('び', INDIGO)]


class Scene(KanjiScene):
    paper_seed = 101
    paper_stains = 0.12

    def build(self):
        self.kj = [typeset(c, 280, font=FONT_BRUSH_JP, seed=20 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('日曜日')]
        self.rd = [text_layer(r, FONT_JP, 92 if k != 1 else 64)[0] for k, (r, _) in enumerate(READ)]
        self.splat = cached('s01s_splat_v1', lambda: {'d': _splat()})['d']
        self.l1, _ = text_layer('SAME KANJI.', FONT_HEAD, 76)
        self.l2, _ = text_layer('DIFFERENT READINGS.', FONT_HEAD, 64)
        self.why, _ = text_layer('WHY?', FONT_BRUSH_EN, 230)
        self.why = self.why * (0.85 + 0.15 * noise(*self.why.shape, 6, 2, 4))
        self.land = cached('s04_land_v1', r5art.sun_landscape)['ink']
        self.sun = sun_disc(H, W, 540, KY + 10, 270, seed=55).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - 540) / 1.0, (yy - KY) / 1.15)
        self.peel = (r / r.max() + 0.12 * (noise(H, W, 180, 4, 9) - 0.5)).astype(np.float32)   # where the ink lets go first
        self.inkfield = (0.96 * (0.9 + 0.1 * noise(H, W, 60, 4, 10))).astype(np.float32)
        self.IVORY = np.array([0.97, 0.94, 0.86], np.float32)
        self.t_same = at('same kanji')[0]
        self.t_two = at('two different')[0]
        c, n = quiet_gap(at('Why?')[1] - 0.15, at('Japan already')[0] + 0.05)
        self.t_hit = c - min(n / 2, 0.1)                               # sync: WHY? stamped (+ taiko, stamp)

    def cam(self, t):
        jolt = 0.05 * np.exp(-t * 9)
        punch = 0.035 * np.exp(-max(t - self.t_hit, 0) * 6) * (t > self.t_hit)
        return Cam(540, 960, 1.0 + jolt + punch + 0.02 * smooth(t / 6))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        # the page underneath: hills and a vermilion sun behind 曜
        reveal = ease_out((t - 0.3) / 0.55)
        self.layer(fr, cam, self.land, 0, 380 - 140 * smooth((t - 0.4) / 1.4), opacity=0.8 * smooth((t - 0.45) / 0.8))
        self.layer(fr, cam, self.sun, color=VERM, opacity=0.62 * smooth((t - 0.5) / 0.7))
        self.layer(fr, cam, self.splat, opacity=0.9 * smooth((t - 0.3) / 0.2) * (1 - smooth((t - 0.9) / 0.8)))
        for k, g in enumerate(self.kj):
            kit.put(fr, cam, g.at((t + 0.2 - 0.04 * k) / 0.2, soft=0.05), XS[k], KY, INK, 1.0, 1 + 0.12 * np.exp(-t * 10))
        # frame 1: a field of wet ink with the word burst out of it in ivory; it peels back from the centre
        if reveal < 1:
            m = np.clip((self.peel - reveal * 1.15 + 0.08) / 0.08, 0, 1)
            fr.ink(self.inkfield * m, INK, 1.0)
            for k, g in enumerate(self.kj):
                kit.put(fr, cam, g.dens, XS[k], KY, self.IVORY, 1.0 - smooth((t - 0.25) / 0.25), 1 + 0.12 * np.exp(-t * 10), over=True)
        # the same kanji, twice: ringed
        for k, col in ((0, VERM), (2, INDIGO)):
            self.ring(fr, cam, t, self.t_same + 0.25 * (k == 2), XS[k], KY, col, r=3.1)
        kit.put(fr, cam, self.l1, 540, 330, INK, smooth((t - self.t_same) / 0.25))
        # ... read two different ways
        for k, (g, (_, col)) in enumerate(zip(self.rd, READ)):
            f = (t - self.t_two - 0.2 * (k != 1) - 0.25 * (k == 2)) / 0.25
            if f > 0:
                kit.put(fr, cam, g, XS[k], KY + 250, col, smooth(f) * (0.7 if k == 1 else 1), 1 + 0.35 * np.exp(-f * 2.5))
        kit.put(fr, cam, self.l2, 540, 430, VERM, smooth((t - self.t_two - 0.4) / 0.25))
        fw = (t - self.t_hit) / 0.15
        if fw > 0:
            kit.put(fr, cam, self.why, 540, 1240, VERM, smooth(fw), 1 + 0.35 * np.exp(-fw * 2.2), rot=-0.05)
        return fr.img


def _splat():
    """ink flung off the brush as the word slams in"""
    import cv2
    r = np.random.default_rng(31)
    d = np.zeros((H, W), np.float32)
    for _ in range(200):
        a = r.uniform(0, 2 * np.pi)
        rr = r.uniform(260, 520)
        x, y = 540 + rr * np.cos(a) * 1.05, KY + rr * np.sin(a) * 0.75
        cv2.circle(d, (int(x), int(y)), int(r.uniform(2, 13)), float(r.uniform(0.6, 1)), -1, cv2.LINE_AA)
    for _ in range(14):
        a = r.uniform(0, 2 * np.pi)
        x0, y0 = 540 + 400 * np.cos(a), KY + 300 * np.sin(a)
        x1, y1 = x0 + 150 * np.cos(a), y0 + 150 * np.sin(a)
        cv2.line(d, (int(x0), int(y0)), (int(x1), int(y1)), 0.85, int(r.uniform(3, 8)), cv2.LINE_AA)
    return blur(d, 1.0)

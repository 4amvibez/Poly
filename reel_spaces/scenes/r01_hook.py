"""Scene 1 — the hook: "How do Japanese people read such a long sentence, without spaces?"
Raj asks it from the very first frame.  The page is already a wall of unbroken Japanese, rows racing past in opposite
directions (the opening of Natsume Sōseki's 吾輩は猫である, 1905, public domain, which mixes kanji, hiragana and
katakana), the camera slams in, and a vermilion brush banner cuts across it: NO SPACES?!  At the bottom a reader sits
with their back to us, staring up at it, a question mark over their head.  On "without spaces" the rows grind to a
halt, and in the breath after the question red ？ marks burst out all over the wall (taiko): where does one word end?"""
import cv2
import numpy as np

from art import figure_back
from engine import (FONT_BRUSH_EN, FONT_JP, FONT_JP_FALLBACK, H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_out,
                    noise, smooth, stroke, text_layer)
from scenes.base import SceneBase
from timeline import at, quiet_gap

LINES = ['吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。',
         '吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。',
         'しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。']
ROWS = [150 + 84 * k for k in range(13)]                       # the wall: 13 rows of text, alternately left / right
SPEED = [170, -230, 200, -150, 260, -190, 160, -240, 210, -170, 250, -200, 180]
BANNER_Y = 640
IVORY = np.array([0.97, 0.94, 0.86], np.float32)
PAPER_C = np.array([0.95, 0.92, 0.85], np.float32)


class Scene(SceneBase):
    paper_seed = 61
    paper_stains = 0.1

    def build(self):
        self.rows = []
        for k in range(len(ROWS)):
            a_, _ = text_layer(LINES[k % 3][(k * 17) % 40:] + LINES[(k + 1) % 3], FONT_JP_FALLBACK, 62, index=0)
            a_ = np.clip(a_ * (0.85 + 0.15 * noise(a_.shape[0], a_.shape[1], 8, 2, k)), 0, 1)
            self.rows.append(np.concatenate([a_, a_], 1))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        band = np.clip(1 - np.abs(yy - BANNER_Y) / 118, 0, 1) ** 0.6                     # cleared strip for the banner
        pit = np.clip(1.25 - np.hypot((xx - 540) / 560, (yy - 1420) / 360), 0, 1)     # cleared pool round the reader
        self.clear = np.clip(np.maximum(band * 1.4, pit * 2.2), 0, 1)
        self.fade = np.clip((1260 - yy) / 260, 0, 1) * np.clip((yy - 60) / 80, 0, 1)

        def paint():
            ss = StrokeSet(W, H)
            figure_back(ss, 540, 1385, 1.5, 0, seed=5, dur=0.01)
            bn = StrokeSet(W, H)
            bn.add(stroke(W, H, [(60, BANNER_Y + 12), (540, BANNER_Y - 6), (1020, BANNER_Y + 6)], 180, ink=0.96, dry=0.45,
                          profile=((0, 0.55), (0.05, 1.0), (0.88, 0.95), (1, 0.45)), seed=11, red=True), 0, 1)
            return {'fig': ss.render(9).astype(np.float32), 'banner': bn.render(9).astype(np.float32)}
        self.A = cached('r01_hook_v3', paint)
        self.title, _ = text_layer('NO SPACES?!', FONT_BRUSH_EN, 124)
        self.title = self.title * (0.88 + 0.12 * noise(*self.title.shape, 6, 2, 3))
        self.qsmall, _ = text_layer('？', FONT_JP, 96)
        self.qbig, _ = text_layer('？', FONT_JP, 92)
        self.t_slow = at('without spaces')[0]
        c, n = quiet_gap(at('without spaces')[1] - 0.2, at('In English')[0])
        self.t_hit = c - n / 2 + 0.06                                  # sync: the ？ marks burst out (+ taiko)
        r = np.random.default_rng(8)
        self.qpos = [(r.uniform(90, 990), ROWS[k] + 2) for k in r.choice([0, 1, 2, 3, 4, 7, 8, 9, 10], 20)]
        self.qdel = r.uniform(0, 0.12, len(self.qpos))

    def pos(self, k, t):
        """how far row k has travelled by time t: full speed, then grinding to a stop on "without spaces" """
        v, D = SPEED[k], 0.75
        if t < self.t_slow:
            return v * t
        u = min((t - self.t_slow) / D, 1.0)
        return v * self.t_slow + v * D * (u - u * u / 2)

    def cam(self, t):
        slam = 1.0 + 0.22 * (1 - ease_out(t / 0.45))                     # the first frames slam in
        punch = 0.05 * np.exp(-max(t - self.t_hit, 0) * 5) * (t > self.t_hit)
        return Cam(540, 940 - 10 * smooth(t / 4), slam + punch + 0.015 * t)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        # the wall of text
        T = np.zeros((H, W), np.float32)
        for k, (L, y) in enumerate(zip(self.rows, ROWS)):
            half = L.shape[1] // 2
            off = int((self.pos(k, t) + 337 * k) % half)
            seg = L[:, off:off + W]
            h = min(seg.shape[0], H - y)
            T[y:y + h, :seg.shape[1]] = np.maximum(T[y:y + h, :seg.shape[1]], seg[:h])
        T *= self.fade * (1 - self.clear)
        self.layer(fr, cam, T, color=INK, opacity=0.9)
        # the reader, looking up at it
        self.layer(fr, cam, self.A['fig'], opacity=0.95)
        q = smooth((t - at('long sentence')[0]) / 0.25)
        bob = 6 * np.sin(t * 5)
        self.place(fr, cam, self.qbig, 540, 1080 + bob, color=VERM, opacity=q, scale=1 + 0.3 * np.exp(-max(t - at('long sentence')[0], 0) * 4))
        # NO SPACES?!
        m = np.zeros_like(self.A['banner'])
        m[:, :int(W * min(1, t / 0.12 + 0.25))] = 1
        self.layer(fr, cam, self.A['banner'] * blur(m, 12), color=VERM, opacity=0.95)
        pop = 1 + 0.3 * (1 - ease_out(t / 0.35))
        self._over(fr, cam, self.title, 540, BANNER_Y - 4, IVORY, smooth((t + 0.05) / 0.12), pop)
        # where does one word end?  ？ marks burst out all over the wall
        for (qx, qy), d in zip(self.qpos, self.qdel):
            g = (t - self.t_hit - d) / 0.2
            if g > 0:
                self._over(fr, cam, blur(cv2.dilate(self.qsmall, np.ones((9, 9), np.uint8)), 3), qx, qy + 31, PAPER_C, smooth(g), 1 + 0.6 * np.exp(-g * 2))
                self.place(fr, cam, self.qsmall, qx, qy + 31, color=VERM, opacity=smooth(g), scale=1 + 0.6 * np.exp(-g * 2))
                self.place(fr, cam, self.qsmall, qx, qy + 31, color=VERM, opacity=0.6 * smooth(g), scale=1 + 0.6 * np.exp(-g * 2))
        return fr.img

    def _over(self, fr, cam, a, x, y, color, op, scale=1.0):
        if op <= 0:
            return
        h, w = a.shape
        M = cam.matrix()
        A = np.array([[scale, 0, x - scale * w / 2], [0, scale, y - scale * h / 2]], np.float32)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.over(color, cv2.warpAffine(a.astype(np.float32), Mt, (W, H)), op)

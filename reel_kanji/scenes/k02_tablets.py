"""Scene 2 — "Thousands of years ago, people needed a way to record the world around them."
The photograph of Mesopotamian clay tablets (supplied by @raj_in_jp): each tablet is laid onto the washi one after
another, settling with a soft contact shadow, while the camera glides down the table.  A label keeps it honest:
these are among the world's earliest writing, from Mesopotamia, not China."""
import numpy as np

from engine import FONT_EN_ITALIC, GREY, Cam, ease_io, ease_out, keyframes, smooth, text_layer
from photos import cutout, place_photo
from scenes.base import SceneBase, museum_card

S = 0.7                         # cutout px -> world px


class Scene(SceneBase):
    paper_seed = 42
    paper_stains = 0.12

    def build(self):
        self.ph = cutout('tablets')
        a = self.ph['alpha']
        hh, ww = a.shape
        self.ox = 540 - ww * S / 2
        self.parts = []
        order = sorted(self.ph['parts'].tolist(), key=lambda b: (b[1] + b[3] / 2) * 3 + b[0])
        for k, (x, y, w, h) in enumerate(order):
            m = np.zeros_like(a)
            x0, y0, x1, y1 = int(x) - 6, int(y) - 6, int(x + w) + 6, int(y + h) + 6
            m[max(0, y0):y1, max(0, x0):x1] = a[max(0, y0):y1, max(0, x0):x1]
            self.parts.append(dict(alpha=m, cx=x + w / 2, cy=y + h / 2, t=self.t0 + 0.05 + 0.2 * k))
        self.lbl1, _ = text_layer('Clay tablets · Mesopotamia', FONT_EN_ITALIC, 46)
        self.lbl2, _ = text_layer("among the world's earliest writing", FONT_EN_ITALIC, 38)
        self.world_h = hh * S

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 820, 1.1)), (self.t1, (540, self.world_h - 860, 1.02))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        M = cam.matrix()
        for p in self.parts:
            f = (t - p['t']) / 0.45
            if f <= 0:
                continue
            e = ease_out(f)
            sc = S * (1.06 - 0.06 * e)
            wx, wy = self.ox + p['cx'] * S, p['cy'] * S - 18 * (1 - e)
            A = np.array([[sc, 0, wx - sc * p['cx']], [0, sc, wy - sc * p['cy']]], np.float32)
            Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
            place_photo(fr, self.ph, Mt, opacity=smooth(f * 1.6), alpha=p['alpha'], shadow=0.42, lift=1 + 1.5 * (1 - e))
        # the label: a museum card on screen, above the captions
        op = smooth((t - self.t0 - 1.0) / 0.45)
        museum_card(fr, [(self.lbl1, GREY * 0.55), (self.lbl2, GREY)], 540, 1300, op)
        return fr.img

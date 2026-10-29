"""Scene 1 — the hook.  "Not every kanji came from China.  Japan created some of its own!"
Frame 1 is already the picture: aged washi, a vermilion sun rising behind a pagoda on misty hills, and JAPAN /
CREATED ITS OWN / KANJI?! across the top.  The title settles as the camera pushes in slowly.  Kanji made in Japan (働 峠 畑 凪 雫 辻) drift up through the
mist, faint.  On "Japan created some of its own!" the title pulses and a vermilion 国字 seal is stamped (sync)."""
import numpy as np

import k6art
import kit
from art import pagoda, sun_disc
from engine import (FONT_BRUSH_EN, FONT_HEAD, FONT_JP, H, INK, VERM, W, Cam, StrokeSet, cached, ease_out, noise, smooth,
                    text_layer)
from scenes.kscene import KanjiScene
from timeline import at

DRIFT = [('働', 170, 1180, 0.0), ('峠', 900, 1260, 0.6), ('畑', 300, 1420, 1.1), ('凪', 820, 1500, 1.7), ('雫', 560, 1330, 2.2),
         ('辻', 980, 1120, 2.8)]


class Scene(KanjiScene):
    paper_seed = 201
    paper_stains = 0.1

    def build(self):
        self.sun = sun_disc(H, W, 700, 1010, 250, seed=5).astype(np.float32)
        self.hills = cached('k6_hills_v1', lambda: k6art.mist_hills(1180, seed=3))['ink']
        ss = StrokeSet(W, H)
        pagoda(ss, 250, 1250, 1.25, seed=7)
        self.pag = ss.render(9)
        self.japan = self.brush('JAPAN', 200, 1)
        self.mid, _ = text_layer('CREATED ITS OWN', FONT_HEAD, 74)
        self.kanji = self.brush('KANJI?!', 210, 2)
        self.drift = [(text_layer(c, FONT_JP, 120)[0], x, y, d) for c, x, y, d in DRIFT]
        self.seal = _seal('国字')
        self.t_japan = at('Japan created')[0]
        self.t_hit = at('of its own')[1] + 0.05                       # sync: the 国字 seal lands as the line ends

    @staticmethod
    def brush(s, size, seed):
        d, _ = text_layer(s, FONT_BRUSH_EN, size)
        return d * (0.86 + 0.14 * noise(*d.shape, 6, 2, seed))

    def cam(self, t):
        punch = 0.03 * np.exp(-max(t - self.t_hit, 0) * 6) * (t > self.t_hit)
        return Cam(540, 960 + 20 * smooth(t / 6), 1.0 + 0.05 * smooth(t / 6.5) + punch + 0.04 * np.exp(-t * 7))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.sun, 0, 60 * (1 - ease_out(t / 2.5)), color=VERM, opacity=0.85)
        self.layer(fr, cam, self.hills, opacity=0.9)
        self.layer(fr, cam, self.pag, opacity=0.95)
        for g, x, y, d in self.drift:                                  # Japan's own kanji, drifting up in the mist
            u = (t + 3.0 - d) / 6.0
            o = 0.22 * np.sin(np.clip(u, 0, 1) * np.pi)
            kit.put(fr, cam, g, x, y - 260 * u, INK, o)
        # the opening sweep: the title shows through as the brush drags across
        rev = 1.0                                                       # the title is there from frame 1
        p = 1 + 0.06 * np.exp(-max(t - self.t_japan, 0) * 4) * (t > self.t_japan)
        kit.put(fr, cam, self.japan, 540, 360, VERM, rev, p * (1.08 - 0.08 * rev), rot=-0.03)
        kit.put(fr, cam, self.mid, 540, 505, INK, 1.0, p * (1.06 - 0.06 * smooth(t / 0.5)))
        kit.put(fr, cam, self.kanji, 540, 650, VERM, 1.0, p * (1.1 - 0.1 * smooth(t / 0.6)), rot=-0.03)
        fs = (t - self.t_hit) / 0.15
        if fs > 0:
            kit.put(fr, cam, self.seal, 905, 760, VERM, smooth(fs) * 0.95, 1 + 0.5 * np.exp(-fs * 2), rot=0.08)
        return fr.img


def _seal(text):
    """a square vermilion seal with the characters cut out in white (density: 1 = vermilion)"""
    from engine import blur
    s = 160
    d = np.zeros((s, s), np.float32)
    d[8:-8, 8:-8] = 1
    d = blur(d, 1.2) * (0.82 + 0.18 * noise(s, s, 10, 3, 140))
    for k, c in enumerate(text):
        g, _ = text_layer(c, FONT_JP, 64)
        gh, gw = g.shape
        cx, cy = 80, 46 + 68 * k
        d[int(cy - gh / 2):int(cy - gh / 2) + gh, int(cx - gw / 2):int(cx - gw / 2) + gw] *= 1 - 0.95 * g
    return d

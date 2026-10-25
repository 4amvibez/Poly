"""Scene 7 — the three writing systems.  A triptych stacked for the vertical frame: three ink-washed bands, each
with a motif matching its script's character (a solid mountain for kanji, flowing water for hiragana, sharp bamboo
for katakana), a large label, and an example that appears as its role is spoken:
  漢字 kanji — 山 "mountain" (core meanings);  ひらがな hiragana — は・を (grammar);
  カタカナ katakana — コーヒー "coffee" (foreign words, emphasis)."""
import numpy as np

from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP_FALLBACK, GREY, H, INK, PAPER, SEPIA, VERM, W, Cam, Frame, StrokeSet,
                    blur, ease_io, keyframes, noise, ramp, reveal_time, smooth, spline, sstep, stroke, text_layer, warp,
                    wash_map)
from glyphs import text_glyphs, typeset
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, sweep
from scenes.base import SceneBase
from timeline import at

BANDS = [  # label, english, example, example font, role, narration cue
    ('漢字', 'Kanji', '山', FONT_JP_FALLBACK, 'core meanings  ·  "mountain"', 'Kanji carry'),
    ('ひらがな', 'Hiragana', 'は・を', None, 'grammar  ·  particles', 'Hiragana handle'),
    ('カタカナ', 'Katakana', 'コーヒー', FONT_JP_FALLBACK, 'foreign words  ·  "coffee"', 'Katakana mark'),
]


class Scene(SceneBase):
    paper_seed = 17

    def build(self):
        self.y0 = [300, 680, 1060]
        self.bh = 340
        t_in = at('Today all three')[0]
        self.t_in = [t_in - 0.1 + 0.45 * k for k in range(3)]
        bg = []
        for k, y in enumerate(self.y0):
            ss = StrokeSet(W, H)
            cy = y + self.bh / 2
            # the band itself: a broad, pale dry-brush wash across the page
            ss.add(sweep(W, H, [(40, cy + 30), (400, cy - 10), (760, cy + 15), (1060, cy - 20)], 300, ink=0.12, dry=0.6,
                         seed=10 + k, wash=0.12), 0, 1)
            if k == 0:   # a solid mountain
                from inkpaint import Painting
                p = Painting(W, H, seed=3)
                r = p.ridge(y + self.bh - 40, peaks=[(800, 170, 120), (950, 90, 90), (640, 60, 110)], rough=6, seed=5)
                p.wash(r, dens=0.22, decay=70, tex=60, edge=0.5, seed=6)
                fade = np.clip((y + self.bh - 6 - np.arange(H, dtype=np.float32)) / 60, 0, 1)[:, None]
                extra = p.D * fade
                extra[:y + 10] = 0
            elif k == 1:  # flowing water
                extra = np.zeros((H, W), np.float32)
                for j in range(5):
                    yy = cy - 70 + j * 34
                    pts = [(x, yy + 14 * np.sin(x / 90 + j)) for x in np.linspace(560, 1040, 12)]
                    st = stroke(W, H, spline(pts, 8), 5 - j * 0.5, ink=0.28, dry=0.4, profile=SWEEP, seed=40 + j)
                    from engine import paste
                    paste(extra, st)
            else:         # sharp bamboo
                extra = np.zeros((H, W), np.float32)
                from engine import paste
                for j, x in enumerate((930, 990, 1045)):
                    for seg in range(3):
                        a, b = y + 20 + seg * 100, y + 112 + seg * 100
                        paste(extra, stroke(W, H, [(x + j * 2, a), (x + j * 2, b)], 18 - j * 3, ink=0.18, dry=0.35, profile=BLUNT, seed=60 + 3 * j + seg))
                    paste(extra, stroke(W, H, [(x, y + 140), (x + 70, y + 110)], 10, ink=0.25, dry=0.3, profile=PRESS_DOWN, seed=70 + j))
            bg.append((ss.render(5), extra))
        self.bg = bg
        self.bg_T = [wash_map(H, W, None, direction=(1, 0.15), noise_amp=0.35, noise_scale=90, seed=80 + k) for k in range(3)]

        self.labels = []
        for k, (jp, en, ex, exf, role, cue) in enumerate(BANDS):
            D, T, _ = text_glyphs(jp, 112 if len(jp) == 2 else 96, seed=100 + 10 * k, tracking=0.02)
            en_a, _ = text_layer(en, FONT_EN_ITALIC, 50)
            exD, exT, _ = text_glyphs(ex, 120 if len(ex) == 1 else (84 if len(ex) == 3 else 64), font=exf or __import__('engine').FONT_JP,
                                      seed=150 + k, tracking=0.0)
            role_a, _ = text_layer(role, FONT_EN, 34)
            ul = StrokeSet(W, H)
            tc = at(cue)[0]
            ul.add(stroke(W, H, [(110, self.y0[k] + 250), (300, self.y0[k] + 246), (470, self.y0[k] + 251)], 7, ink=0.9, dry=0.45,
                          profile=SWEEP, seed=200 + k, red=True), tc - 0.05, 0.4)
            self.labels.append(dict(D=D, T=T, en=en_a, exD=exD, exT=exT, role=role_a, ul=ul, tc=tc))

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 1000, 1.0)), (self.t1, (540, 940, 1.045))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        for k, (y, (band, extra), L) in enumerate(zip(self.y0, self.bg, self.labels)):
            f = (t - self.t_in[k]) / 0.9
            if f <= 0:
                continue
            T = self.bg_T[k]
            self.layer(fr, cam, reveal_time(band, T, f * 1.1, soft=0.08, wet=0.2), depth=0.85, ref=(540, 960))
            act = smooth((t - L['tc']) / 0.5)
            self.layer(fr, cam, reveal_time(extra, T, (f - 0.3) * 1.1, soft=0.08, wet=0.2), depth=0.8, ref=(540, 960),
                       opacity=0.6 + 0.4 * act)
            cy = y + self.bh / 2
            D = L['D']
            self.layer(fr, cam, reveal_time(D, L['T'], f * 1.15, soft=0.05, wet=0.4), 110, int(cy - 40 - D.shape[0] / 2))
            en = L['en']
            self.layer(fr, cam, en, 118, int(cy + 52 - en.shape[0] / 2), color=GREY, opacity=smooth(f - 0.6))
            self.layer(fr, cam, L['ul'].render(t, red=True), color=VERM)
            # the example and its role, when the narration reaches this script
            fe = (t - L['tc'] - 0.2) / 0.6
            if fe > 0:
                exD = L['exD']
                ex_x = 800 - exD.shape[1] // 2
                self.layer(fr, cam, reveal_time(exD, L['exT'], fe * 1.1, soft=0.05, wet=0.4), ex_x, int(cy - 40 - exD.shape[0] / 2))
                ro = L['role']
                self.layer(fr, cam, ro, 800 - ro.shape[1] // 2, int(cy + 62 - ro.shape[0] / 2), color=GREY, opacity=smooth(fe - 0.5))
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        """three bands of ink sweep across in turn, opening the three-part composition"""
        from engine import wipe_mix
        if not hasattr(self, '_T'):
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            band = np.clip((yy - 120) // 560, 0, 2)
            T = band * 0.22 + (xx / W) * 0.56 + (noise(H, W, 70, 4, 5) - 0.5) * 0.12
            self._T = np.clip(T, 0, 1)
        return wipe_mix(prev_img, cur_img, self._T, ease_io(f) * 1.15 - 0.05, soft=0.028, edge=0.42)

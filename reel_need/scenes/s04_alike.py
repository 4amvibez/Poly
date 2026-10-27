"""Scene 4 — "Japanese has many words that sound alike.  Without kanji, readers may need more context to figure out
which words and meanings are intended.  Kanji gives your eyes an important clue, helping you recognize meaningful
words more easily."
Three cards deal in, all reading はし hashi; sound waves ripple over them on "sound alike".  Without kanji, each card's
picture is lost in mist and flickers between the three things hashi could be: a bridge, chopsticks, an edge.  Then a
big ink eye opens over them in the breath before "Kanji gives your eyes…" (bowl), and the cards turn over one by one
on "recognize": 橋 bridge, 箸 chopsticks, 端 edge, each with its own picture, sharp."""
import numpy as np

import kit
import pics
from backdrop import backdrop, draw_backdrop
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet, blur, cached, ease_io,
                    ease_out, keyframes, smooth, stroke, text_layer)
from glyphs import typeset
from poly_ink.brush import SWEEP
from scenes.base import SceneBase
from timeline import at, quiet_gap

CX = [190, 540, 890]
CY = 1000
BH, BW = 470, 300
WORDS = [('橋', 'bridge', 'bridge'), ('箸', 'chopsticks', 'chopsticks'), ('端', 'edge', 'edge')]


class Scene(SceneBase):
    paper_seed = 84

    def build(self):
        self.B = backdrop(seed=230, branch='left', pagoda_x=920, ground=1480)
        self.P = cached('s04_pics_v1', lambda: {k: getattr(pics, k)() for k in ('bridge', 'chopsticks', 'edge')} |
                        dict(zip(('lid', 'iris', 'pupil'), pics.eye())))
        self.box, self.rim = kit.box(BH, BW), kit.rim(BH, BW)
        self.kana = typeset('は', 92, font=FONT_JP, seed=1, pad=0.04, medians=False), typeset('し', 92, font=FONT_JP, seed=2, pad=0.04, medians=False)
        self.ro, _ = text_layer('hashi', FONT_EN_ITALIC, 44)
        self.kj = [typeset(k, 128, font=FONT_JP, seed=10 + i, pad=0.04, medians=False) for i, (k, _, _) in enumerate(WORDS)]
        self.en = [text_layer(e, FONT_EN, 46)[0] for _, e, _ in WORDS]
        self.q, _ = text_layer('？', FONT_JP, 110)
        self.t_deal = [at('Japanese has many')[0] + 0.1 + 0.18 * k for k in range(3)]
        self.t_sound = at('sound alike')[0]
        self.t_mist = at('Without kanji')[0]
        self.t_flick = at('which words')[0] - 0.3
        c, n = quiet_gap(at('are intended')[1] - 0.2, at('Kanji gives')[0] + 0.1)
        self.t_hit = c - n / 2 + 0.06                              # sync: the eye opens (+ bowl)
        self.t_flip = [at('recognize')[0] + 0.05 + 0.4 * k for k in range(3)]
        waves = StrokeSet(W, H)
        for k, x in enumerate(CX):
            for j, r in enumerate((36, 58, 80)):
                a = np.linspace(-0.7, 0.7, 12)
                waves.add(stroke(W, H, [(x + r * np.sin(u), CY - BH / 2 - 40 - r * np.cos(u) * 0.6 + 20) for u in a], 4, ink=0.8, dry=0.4,
                                 profile=SWEEP, seed=10 * k + j), self.t_sound + 0.08 * k + 0.1 * j, 0.2)
        self.waves = waves
        from engine import FONT_BRUSH_EN
        self.one, _ = text_layer('one sound…', FONT_BRUSH_EN, 96)
        self.three, _ = text_layer('…three meanings?', FONT_BRUSH_EN, 96)

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 975, 1.02)), (self.t_hit, (540, 960, 1.0)), (self.t1, (540, 965, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        P = self.P
        draw_backdrop(self, fr, cam, self.B, smooth((t - self.t0) / 0.8))
        # one sound, three meanings?  (making way for the eye)
        gone = 1 - smooth((t - self.t_hit + 0.25) / 0.25)
        kit.put(fr, cam, self.one, 540, 380, INK, smooth((t - self.t_sound) / 0.35) * gone)
        kit.put(fr, cam, self.three, 540, 500, VERM, smooth((t - self.t_flick) / 0.35) * gone)
        # the eye that opens over the cards
        fe = (t - self.t_hit) / 0.35
        if fe > 0:
            sy = 0.06 + 0.94 * ease_out(min(fe, 1))
            look = 14 * np.sin(t * 1.3)
            kit.put(fr, cam, P['iris'], 540 + look, 400, np.array([0.42, 0.3, 0.22], np.float32), 0.85 * smooth(fe), 1.0, sy)
            kit.put(fr, cam, P['pupil'], 540 + look, 400, INK, smooth(fe), 1.0, sy)
            kit.put(fr, cam, P['lid'], 540, 400, INK, smooth(fe * 2), 1.0, sy)
            if fe > 1:                                           # sight lines down to the cards
                rays = np.zeros((H, W), np.float32)
                import cv2
                for x in CX:
                    cv2.line(rays, (540, 520), (int(x), int(CY - BH / 2 - 10)), 1.0, 3, cv2.LINE_AA)
                self.layer(fr, cam, blur(rays, 1.2), color=VERM, opacity=0.45 * smooth((fe - 1) / 0.6))
        self.layer(fr, cam, self.waves.render(t), color=GREY, opacity=1 - smooth((t - self.t_mist) / 0.6))
        # the cards
        for k, (x, (kj, en, pic)) in enumerate(zip(CX, WORDS)):
            f = (t - self.t_deal[k]) / 0.4
            if f <= 0:
                continue
            e = ease_out(min(f, 1))
            y = CY + 120 * (1 - e)
            u = np.clip((t - self.t_flip[k]) / 0.4, 0, 1)
            sx = max(abs(np.cos(np.pi * u)), 0.02)
            sy = 1 + 0.06 * np.sin(np.pi * u)
            up = u >= 0.5
            op = smooth(f * 1.5)
            wob = 0.03 * np.sin(t * 6 + k) * smooth((t - self.t_mist) / 0.4) * (1 - u)
            kit.card(fr, cam, self.box, self.rim, x, y, op, sx, sy)
            Q = lambda d, dy, col=INK, o=1.0, s=1.0: kit.put(fr, cam, d, x, y + dy * sy, col, op * o, sx * s, sy * s, rot=wob)
            if not up:
                kit.put(fr, cam, self.kana[0].dens, x - 44 * sx, y - 160 * sy, INK, op, sx, sy, rot=wob)
                kit.put(fr, cam, self.kana[1].dens, x + 44 * sx, y - 160 * sy, INK, op, sx, sy, rot=wob)
                Q(self.ro, -86, GREY)
                # the picture: plain at first, then lost in mist and flickering between what it might be
                m = smooth((t - self.t_mist) / 0.5)
                if m > 0:
                    names = ['bridge', 'chopsticks', 'edge']
                    fk = (t - self.t_flick) * 2.2 + k
                    j = int(np.floor(fk)) % 3 if t > self.t_flick else k
                    ph = (fk % 1) if t > self.t_flick else 1.0
                    Q(P[names[j]], 90, INK, 0.35 * m * np.sin(np.pi * ph) ** 0.5, 1.05)
                    Q(blur(kit.box(200, 240, 6), 14), 90, kit.PAPER_C, 0.55 * m)
                    Q(self.q, 90, VERM, 0.85 * m, 0.9 + 0.08 * np.sin(t * 4 + k))
            else:
                Q(self.kj[k].dens, -150)
                Q(self.ro, -66, GREY, s=0.9)
                Q(P[pic], 80, INK, 1.0, 1.08)
                Q(self.en[k], 190, VERM * 0.85)
        return fr.img

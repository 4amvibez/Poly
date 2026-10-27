"""Scene 1 — the hook: "Is kanji really necessary?  What if Japanese used only hiragana and katakana?"
Frame 1 slams in on the question: "Is KANJI really necessary?" over a huge vermilion sun with 漢字 brushed inside it,
blossom and a far pagoda, a reader sitting below with their back to us (taiko).  In the breath after the question
漢字 bursts apart into ink droplets (bowl), and on "What if…" a ring of hiragana and katakana swirls into the sun
where it was, a thought cloud rising over the reader: あいうえお / カタカナ … ?"""
import numpy as np

import kit
from art import figure_back, sun_disc
from backdrop import backdrop, draw_backdrop
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_HEAD, FONT_JP, H, INK, VERM, W, Cam, StrokeSet, cached, ease_io, ease_out,
                    noise, smooth, text_layer)
from glyphs import typeset
from scenes.base import SceneBase
from timeline import at, quiet_gap

SUN = (540, 930)
KANA = 'あいうえおカタカナひらがな'
IVORY = np.array([0.98, 0.95, 0.88], np.float32)


class Scene(SceneBase):
    paper_seed = 81
    paper_stains = 0.1

    def build(self):
        self.B = backdrop(seed=200, branch='right', pagoda_x=900, ground=1330)

        def paint():
            ss = StrokeSet(W, H)
            figure_back(ss, 300, 1395, 1.35, 0, seed=4, dur=0.01)
            return {'sun': sun_disc(H, W, SUN[0], SUN[1], 250, seed=8).astype(np.float32), 'fig': ss.render(9).astype(np.float32)}
        self.A = cached('s01_paint_v1', paint)
        self.kj = [typeset(c, 250, font=FONT_BRUSH_JP, seed=10 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('漢字')]
        self.is_, _ = text_layer('Is', FONT_HEAD, 104)
        self.kanji, _ = text_layer('KANJI', FONT_BRUSH_EN, 260)
        self.kanji = self.kanji * (0.88 + 0.12 * noise(*self.kanji.shape, 6, 2, 4))
        self.nec, _ = text_layer('really necessary?', FONT_HEAD, 84)
        c, n = quiet_gap(at('necessary?')[1] - 0.2, at('What if')[0] + 0.1)
        self.t_hit = c - n / 2 + 0.08                              # sync: 漢字 bursts (+ bowl)
        self.t_what = at('What if')[0]
        self.t_kana = at('hiragana and katakana')[0]
        r = np.random.default_rng(3)
        # the burst: the two characters broken into droplets that fly out
        self.drops = []
        for k, g in enumerate(self.kj):
            ys, xs = np.nonzero(g.dens > 0.4)
            pick = r.choice(len(xs), 70, replace=False)
            for i in pick:
                x0 = SUN[0] + (k - 0.5) * 255 + xs[i] - g.dens.shape[1] / 2
                y0 = SUN[1] + ys[i] - g.dens.shape[0] / 2
                ang = np.arctan2(y0 - SUN[1], x0 - SUN[0]) + r.normal(0, 0.4)
                self.drops.append((x0, y0, ang, r.uniform(250, 900), r.uniform(5, 16)))
        self.kana = [typeset(c, 92, font=FONT_JP, seed=30 + k, pad=0.04, medians=False) for k, c in enumerate(KANA)]
        self.cloud, self.cfill = cached('s01_cloud_v1', lambda: dict(zip(('l', 'f'), __import__('pics').cloud(460, 280, seed=5, tail=(-1, 1))))).values()
        self.thought = [typeset(c, 60, font=FONT_JP, seed=60 + k, pad=0.04, medians=False) for k, c in enumerate('あいうえお')]
        self.thought2 = [typeset(c, 60, font=FONT_JP, seed=70 + k, pad=0.04, medians=False) for k, c in enumerate('カタカナ？')]
        # frame 1: 漢字 fills the screen, ink still flying, and a vermilion "no" stamp slams across it
        self.T_STAMP = 0.24
        sign = StrokeSet(W, H)
        from art import enso
        enso(sign, SUN[0], SUN[1], 300, 46, self.T_STAMP, 0.16, seed=6, start=-2.2, sweep=6.1, red=True)
        from engine import stroke
        sign.add(stroke(W, H, [(SUN[0] - 205, SUN[1] - 205), (SUN[0], SUN[1]), (SUN[0] + 205, SUN[1] + 205)], 44, ink=0.97, dry=0.35,
                        profile=((0, 0.6), (0.1, 1), (0.9, 0.95), (1, 0.5)), seed=7, red=True), self.T_STAMP + 0.14, 0.1)
        self.sign = sign
        self.splat = cached('s01_splat_v1', lambda: {'d': _splat()})['d']
        self.t_name = at('kanji')[0]
        self.t_nec = at('necessary?')[0]

    def cam(self, t):
        pull = ease_io((t - 0.45) / 0.65)                        # frame 1 is tight on 漢字; the camera pulls back to the scene
        jolt = 0.06 * np.exp(-max(t - self.T_STAMP - 0.14, 0) * 9) * (t > self.T_STAMP + 0.14)
        punch = 0.04 * np.exp(-max(t - self.t_hit, 0) * 5) * (t > self.t_hit)
        z = 1.0 + 0.8 * (1 - pull) + jolt + punch + 0.01 * t
        return Cam(540, SUN[1] + (960 - SUN[1]) * pull - 15 * smooth(t / 6), z)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        draw_backdrop(self, fr, cam, self.B, 1.0)
        sun = smooth((t - 0.55) / 0.6)                          # the sun blooms in behind 漢字 as the camera pulls back
        self.layer(fr, cam, self.A['sun'], color=VERM, opacity=0.9 * sun, depth=0.95)
        self.layer(fr, cam, self.splat, opacity=0.9 * (1 - smooth((t - 0.5) / 0.6)))
        # 漢字 in the sun, until it bursts
        burst = t - self.t_hit
        if burst < 0.05:
            for k, g in enumerate(self.kj):
                kit.put(fr, cam, g.at((t + 0.22 - 0.06 * k) / 0.22, soft=0.06), SUN[0] + (k - 0.5) * 255, SUN[1], INK, 1.0)
            self.layer(fr, cam, self.sign.render(t, red=True), color=VERM, opacity=1 - smooth((t - 0.75) / 0.5))
        if burst > 0:
            d = np.zeros((H, W), np.float32)
            u = ease_out(min(burst / 0.9, 1))
            fade = 1 - smooth((burst - 0.4) / 0.6)
            import cv2
            for x0, y0, ang, dist, sz in self.drops:
                x, y = x0 + np.cos(ang) * dist * u, y0 + np.sin(ang) * dist * u + 260 * u * u
                cv2.circle(d, (int(x), int(y)), int(sz * (1 - 0.5 * u)), 1.0, -1, cv2.LINE_AA)
            self.layer(fr, cam, kit.blur(d, 1.2) * fade, opacity=0.9)
        # the kana that take its place, swirling in
        f = (t - self.t_what) / 1.2
        if f > 0:
            for k, g in enumerate(self.kana):
                a = 2 * np.pi * k / len(self.kana) + 0.5 * t
                rr = 170 + 60 * (1 - ease_out(min(f, 1))) + 12 * np.sin(t * 2 + k)
                o = smooth((t - self.t_what - 0.06 * k) / 0.3)
                kit.put(fr, cam, g.dens, SUN[0] + rr * np.cos(a), SUN[1] + rr * 0.8 * np.sin(a), IVORY, o, over=True)
        # the reader, wondering
        self.layer(fr, cam, self.A['fig'], opacity=0.95)
        fc = smooth((t - self.t_kana + 0.2) / 0.35)
        if fc > 0:
            cx, cy = 700, 1150
            kit.put(fr, cam, self.cfill, cx, cy, kit.PAPER_C, 0.95 * fc, 0.7 + 0.3 * ease_out(fc), over=True)
            kit.put(fr, cam, self.cloud, cx, cy, INK, fc, 0.7 + 0.3 * ease_out(fc))
            for row, (gl, y) in enumerate(((self.thought, cy - 50), (self.thought2, cy + 30))):
                for k, g in enumerate(gl):
                    o = smooth((t - self.t_kana - 0.05 * k - 0.4 * row) / 0.25)
                    kit.put(fr, cam, g.dens, cx + (k - (len(gl) - 1) / 2) * 64, y, VERM if row and k >= 4 else INK, o)
        # the question, word by word as Raj asks it, as the camera pulls back
        kit.put(fr, cam, self.is_, 540, 230, INK, smooth((t - 0.8) / 0.15))
        fk = (t - max(self.t_name, 0.95)) / 0.25
        kit.put(fr, cam, self.kanji, 540, 410, VERM, smooth(fk), 1 + 0.25 * np.exp(-max(fk, 0) * 3) * (fk > 0))
        kit.put(fr, cam, self.nec, 540, 590, INK, smooth((t - self.t_nec + 0.4) / 0.25))
        return fr.img


def _splat():
    """ink flung off the brush round 漢字: droplets and a few flicks (density)"""
    import cv2
    r = np.random.default_rng(11)
    d = np.zeros((H, W), np.float32)
    for _ in range(140):
        a = r.uniform(0, 2 * np.pi)
        rr = r.uniform(250, 420)
        x, y = SUN[0] + rr * np.cos(a) * 1.15, SUN[1] + rr * np.sin(a) * 0.9
        cv2.circle(d, (int(x), int(y)), int(r.uniform(2, 11)), float(r.uniform(0.6, 1)), -1, cv2.LINE_AA)
    for _ in range(10):
        a = r.uniform(0, 2 * np.pi)
        x0, y0 = SUN[0] + 260 * np.cos(a), SUN[1] + 220 * np.sin(a)
        x1, y1 = x0 + 120 * np.cos(a), y0 + 120 * np.sin(a)
        cv2.line(d, (int(x0), int(y0)), (int(x1), int(y1)), 0.8, int(r.uniform(3, 7)), cv2.LINE_AA)
    return __import__('engine').blur(d, 1.0)

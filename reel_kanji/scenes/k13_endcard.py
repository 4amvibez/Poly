"""Scene 13 — the end card.  An ensō is brushed around 漢字 (Yuji Syuku), a vermilion brush banner carries
@raj_in_jp, "Follow @raj_in_jp for more fascinating stories about Japanese."; a blossom branch, a far pagoda and
petals.  Then, as Raj says it in Japanese, 最後まで見てくれてありがとう is written and またね！ stamped in vermilion."""
import cv2
import numpy as np

from art import blossom_cluster, cherry_branch, enso, pagoda
from engine import (FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, SAKURA, VERM, W, Cam, StrokeSet,
                    blur, bloom, cached, ease_io, ease_out, keyframes, reveal_time, smooth, stroke, text_layer, wash_map, wipe_mix)
from glyphs import text_glyphs, typeset
from scenes.base import SceneBase
from timeline import JA, at

EC = (540, 610)
IVORY = np.array([0.98, 0.95, 0.88], np.float32)


class Scene(SceneBase):
    paper_seed = 55

    def build(self):
        def paint():
            from inkpaint import Painting
            p = Painting(W, H, seed=56)
            r1 = p.ridge(930, peaks=[(160, 120, 170), (620, 60, 220), (980, 140, 160)], rough=10, seed=1)
            p.wash(r1, dens=0.12, decay=90, tex=70, edge=0.25, mist_y=990, mist_h=40, seed=2)
            ss = StrokeSet(W, H)
            pagoda(ss, 905, 960, 0.75, 0, 0.01, seed=3, ink=0.7)
            tips = cherry_branch(ss, [(1130, 150), (1020, 200), (930, 186), (850, 244), (780, 232)], 18, 0, 0.01, seed=4)
            bl = np.zeros((H, W), np.float32)
            for k, (bx, by) in enumerate(tips):
                bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 12, 40, seed=10 + k))
            return {'land': p.D.astype(np.float32), 'deco': ss.render(9).astype(np.float32), 'bl': bl}
        self.A = cached('k13_scene_v1', paint)
        self.wash_T = wash_map(H, W, (540, 700), noise_amp=0.5, noise_scale=120, seed=7)
        self.t_f = at('Follow Raj')[0]
        es = StrokeSet(W, H)
        enso(es, EC[0], EC[1], 262, 40, self.t0 + 0.15, 0.85, seed=8)
        self.es = es
        self.kan = typeset('漢', 225, font=FONT_BRUSH_JP, seed=11, medians=False, sweep=(0.6, 1))
        self.ji = typeset('字', 225, font=FONT_BRUSH_JP, seed=12, medians=False, sweep=(0.6, 1))
        # the banner: one loaded vermilion sweep, the handle in ivory over it
        bn = StrokeSet(W, H)
        bn.add(stroke(W, H, [(215, 985), (420, 975), (650, 978), (870, 968)], 118, ink=0.95, dry=0.55,
                      profile=((0, 0.7), (0.06, 1.0), (0.85, 0.95), (1, 0.35)), seed=13, red=True), self.t_f + 0.1, 0.45)
        self.banner = bn
        self.handle, _ = text_layer('@raj_in_jp', FONT_EN_BOLD, 80, tracking=2)
        self.follow = [text_layer('Follow @raj_in_jp for more', FONT_EN_ITALIC, 46)[0],
                       text_layer('fascinating stories about Japanese.', FONT_EN_ITALIC, 46)[0]]
        # the Japanese sign-off, in time with the voice
        a, b, m = JA['start'], JA['arigatou_end'], JA['matane']
        split = a + (b - a) * 0.6
        self.ja = []
        for text, size, y, t0, t1, col in (('最後まで見てくれて', 74, 1135, a - 0.05, split, INK),
                                            ('ありがとう', 96, 1242, split - 0.05, b, INK),
                                            ('またね！', 112, 1370, m - 0.08, m + 0.35, VERM)):
            D, T, _ = text_glyphs(text, size, font=FONT_JP, seed=hash(text) % 997, tracking=0.03)
            self.ja.append(dict(D=D, T=T, halo=blur(D, 4), y=y, t0=t0, t1=t1, col=col))
        self.tr, _ = text_layer('"Thanks for watching till the end. See you!"', FONT_EN_ITALIC, 38)
        self.t_tr = m + 0.4
        r = np.random.default_rng(14)
        self.petals = [(r.uniform(-50, 1100), r.uniform(-300, 1700), r.uniform(0.6, 1.2), r.uniform(0, 6)) for _ in range(14)]

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 980, 1.05)), (self.t1, (540, 950, 1.0))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        fw = (t - self.t0) / 1.4
        self.layer(fr, cam, reveal_time(self.A['land'], self.wash_T, fw, soft=0.1, wet=0.2), depth=0.9)
        self.layer(fr, cam, reveal_time(self.A['deco'], self.wash_T, fw * 1.1, soft=0.1, wet=0.25))
        self.layer(fr, cam, self.A['bl'], color=SAKURA, opacity=0.9 * smooth(fw))
        self.layer(fr, cam, self.es.render(t))
        for g, x, t0 in ((self.kan, EC[0] - 112, self.t0 + 0.6), (self.ji, EC[0] + 112, self.t0 + 1.0)):
            f = (t - t0) / 0.5
            if f > 0:
                self.place(fr, cam, bloom(g.at(min(f, 1.05), soft=0.05, wet=0.45), t - t0 - 0.5, amount=0.14), x, EC[1] + 6)
        # banner + handle
        self.layer(fr, cam, self.banner.render(t, red=True), color=VERM, opacity=0.95)
        oh = smooth((t - self.t_f - 0.45) / 0.35)
        if oh > 0:
            M = cam.matrix()
            h, w = self.handle.shape
            A = np.array([[1, 0, 540 - w / 2], [0, 1, 975 - h / 2]], np.float32)
            Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
            fr.over(IVORY, cv2.warpAffine(self.handle, Mt, (W, H)), oh)
        jo = 1 - smooth((t - JA['start'] + 0.4) / 0.4)                   # the follow line makes way for the Japanese
        for k, a in enumerate(self.follow):
            self.place(fr, cam, a, 540, 1092 + 54 * k, color=GREY, opacity=smooth((t - self.t_f - 0.8 - 0.25 * k) / 0.4) * jo)
        for L in self.ja:
            f = (t - L['t0']) / (L['t1'] - L['t0'])
            if f > 0:
                d = bloom(reveal_time(L['D'], L['T'], f * 1.08, soft=0.05, wet=0.45), t - L['t1'], amount=0.15, halo=L['halo'])
                punch = 1 + 0.1 * np.exp(-max(t - L['t0'], 0) * 5) if L['col'] is VERM else 1
                self.place(fr, cam, d, 540, L['y'], color=L['col'], scale=punch)
        self.place(fr, cam, self.tr, 540, 1462, color=GREY, opacity=smooth((t - self.t_tr) / 0.5))
        # drifting petals
        P = np.zeros((H, W), np.float32)
        for x0, y0, sp, ph in self.petals:
            y = (y0 + (t - self.t0) * 60 * sp) % 1900
            x = x0 + (t - self.t0) * 18 * sp + 22 * np.sin(t * 1.1 + ph)
            cv2.ellipse(P, (int(x), int(y)), (7, 4), (t * 2 + ph) * 57, 0, 360, 0.5, -1, cv2.LINE_AA)
        fr.ink(blur(P, 0.7), SAKURA, smooth(fw))
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        from scenes.base import ink_bloom
        return ink_bloom(prev_img, cur_img, f, EC, seed=self.paper_seed)

"""Scene 8 — "But here's the fascinating part: not every kanji is a picture.  Many characters combine a component
related to meaning, with another that gives a clue to pronunciation.  This allowed the writing system to express
far more than simple objects."
The brush writes 河 (river).  On "meaning" its water radical 氵 turns blue and steps out, tagged "meaning: water";
on "pronunciation" 可 turns vermilion and steps the other way, tagged "sound: ka"; they rejoin as 河 "ka", river.
Then the idea grows: three compounds build up one after another (after the reference sheet @raj_in_jp sent):
田 field + 力 strength → 男 man,  八 split + 刀 knife → 分 divide,  雨 rain + lightning → 電 electricity."""
import numpy as np

from ancient import ancient
from art import brush_sprite, draw_sprite
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, PAPER, SEA, VERM, W, Cam, StrokeSet, blur, bloom,
                    ease_io, ease_out, keyframes, lerp, smooth, stroke, text_layer)
from glyphs import calligraphy
from poly_ink.brush import PRESS_DOWN
from scenes.base import SceneBase
from timeline import at

WATER = np.asarray(SEA, np.float32) * 0.85
GC = (540, 800)                         # 河 while it is explained
COMPOUNDS = [(('田', 'field'), ('力', 'strength'), ('男', 'man')),
             (('八', 'split'), ('刀', 'knife'), ('分', 'divide')),
             (('雨', 'rain'), ('申', 'lightning'), ('電', 'electricity'))]
ROW_Y = [720, 960, 1200]


def _tag(text, sub, tint):
    """a small tinted label plate: bold line + a smaller second line"""
    a1, _ = text_layer(text, FONT_EN, 44)
    a2, _ = text_layer(sub, FONT_EN_ITALIC, 36)
    w = max(a1.shape[1], a2.shape[1]) + 56
    h = a1.shape[0] + a2.shape[0] + 34
    plate = np.zeros((h, w), np.float32)
    plate[4:-4, 4:-4] = 1
    plate = blur(plate, 3)
    txt = np.zeros((h, w), np.float32)
    txt[14:14 + a1.shape[0], (w - a1.shape[1]) // 2:(w - a1.shape[1]) // 2 + a1.shape[1]] = a1
    y = 14 + a1.shape[0] + 4
    txt2 = np.zeros((h, w), np.float32)
    txt2[y:y + a2.shape[0], (w - a2.shape[1]) // 2:(w - a2.shape[1]) // 2 + a2.shape[1]] = a2
    return dict(plate=plate, txt=txt, txt2=txt2, tint=np.asarray(tint, np.float32))


class Scene(SceneBase):
    paper_seed = 48

    def build(self):
        self.g = calligraphy('河', 500, seed=5)
        # every inked pixel belongs to exactly one part: 氵 (strokes 0-2) or 可 (3-7), by its nearest stroke
        import cv2
        sw = np.isin(self.g.sidx, [0, 1, 2]).astype(np.uint8)
        sk = np.isin(self.g.sidx, [3, 4, 5, 6, 7]).astype(np.uint8)
        dw = cv2.distanceTransform(1 - sw, cv2.DIST_L2, 5)
        dk = cv2.distanceTransform(1 - sk, cv2.DIST_L2, 5)
        self.m_w = (dw < dk).astype(np.float32)
        self.m_k = 1 - self.m_w
        self.t_write = at('But heres')[0] + 0.1
        self.t_write_end = at('fascinating part')[1] - 0.1
        self.brush = brush_sprite(seed=6)
        self.t_mean = at('related to meaning')[0] - 0.2
        self.t_sound = at('clue to pronunciation')[0] - 0.2
        from timeline import quiet_gap
        self.t_hit, _ = quiet_gap(at('pronunciation')[1] - 0.15, at('This allowed')[0] + 0.05)   # the parts click together
        self.t_join = self.t_hit - 0.35
        self.t_more = at('This allowed')[0] - 0.1
        self.tag_w = _tag('meaning: water', 'the water radical', WATER)
        self.tag_k = _tag('sound: ka', 'a clue to the reading', VERM)
        self.ka, _ = text_layer('"ka"', FONT_EN_ITALIC, 64)
        self.river, _ = text_layer('river', FONT_EN, 48)
        self.kanji_k = text_layer('可', FONT_JP, 72)[0]
        arrows = StrokeSet(W, H)
        arrows.add([stroke(W, H, [(330, 1010), (285, 1060), (262, 1100)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=1),
                    stroke(W, H, [(258, 1078), (262, 1100), (284, 1092)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=2)],
                   self.t_mean + 0.35, 0.3)
        arrows.add([stroke(W, H, [(700, 1010), (760, 1060), (790, 1100)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=3),
                    stroke(W, H, [(768, 1094), (790, 1100), (792, 1078)], 4, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=4)],
                   self.t_sound + 0.35, 0.3)
        self.arrows = arrows
        # the compounds
        self.rows = []
        ops = StrokeSet(W, H)
        for k, ((c1, e1), (c2, e2), (cr, er)) in enumerate(COMPOUNDS):
            t = self.t_more + 0.25 + 1.05 * k
            y = ROW_Y[k]
            g1 = calligraphy(c1, 130, seed=30 + k)
            g2 = ancient('申', 'oracle', 130, seed=40) if c2 == '申' else calligraphy(c2, 130, seed=33 + k)
            gr = calligraphy(cr, 175, seed=36 + k)
            self.rows.append(dict(t=t, y=y, g1=g1, g2=g2, gr=gr, l1=text_layer(e1, FONT_EN_ITALIC, 34)[0],
                                  l2=text_layer(e2, FONT_EN_ITALIC, 34)[0], lr=text_layer(er, FONT_EN, 44)[0]))
            ops.add([stroke(W, H, [(285, y - 2), (325, y - 2)], 5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=50 + k),
                     stroke(W, H, [(305, y - 22), (305, y + 18)], 5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=60 + k)], t + 0.2, 0.2)
            ops.add([stroke(W, H, [(530, y), (600, y - 2)], 5, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=70 + k),
                     stroke(W, H, [(582, y - 17), (602, y - 2), (582, y + 14)], 5, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=80 + k)],
                    t + 0.5, 0.2)
        self.ops = ops

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 900, 1.05)), (self.t_write_end, (540, 930, 1.0)),
                                  (self.t_more, (540, 940, 1.0)), (self.t1, (540, 930, 1.03))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        g = self.g
        # 河: written, split into its two parts, rejoined, then lifted to the top of the page
        f = (t - self.t_write) / (self.t_write_end - self.t_write)
        if f > 0:
            d = bloom(g.at(min(f, 1.05), soft=0.012, wet=0.45), t - self.t_write_end, amount=0.12)
            sm = smooth((t - self.t_mean) / 0.5) * (1 - smooth((t - self.t_join) / 0.5))
            ss = smooth((t - self.t_sound) / 0.5) * (1 - smooth((t - self.t_join) / 0.5))
            up = ease_io((t - self.t_more) / 0.8)
            cx, cy = GC[0], lerp(GC[1], 380, up)
            sc = lerp(1.0, 0.46, up)
            cw = smooth((t - self.t_mean) / 0.4)                     # colours stay once given
            ck = smooth((t - self.t_sound) / 0.4)
            dw, dk = d * self.m_w, d * self.m_k
            for part, dx, c, amt in ((dw, -60 * sm, WATER, cw), (dk, 50 * ss, VERM, ck)):
                col = np.asarray(INK) * (1 - amt) + c * amt
                self.place(fr, cam, part, cx + dx * sc, cy, color=col, scale=sc)
            if t < self.t_write_end + 0.9:
                self._brush(fr, cam, t, f)
            # tags and arrows while it is explained
            out = 1 - smooth((t - self.t_more) / 0.4)
            for tag, x, t0 in ((self.tag_w, 250, self.t_mean + 0.45), (self.tag_k, 820, self.t_sound + 0.45)):
                op = smooth((t - t0) / 0.35) * out
                if op > 0:
                    self.place(fr, cam, tag['plate'], x, 1160, color=tag['tint'] * 0.22 + np.asarray(PAPER) * 0.78, opacity=0.95 * op)
                    self.place(fr, cam, tag['txt'], x, 1160, color=tag['tint'] * 0.75, opacity=op)
                    self.place(fr, cam, tag['txt2'], x, 1160, color=GREY, opacity=op)
            self.layer(fr, cam, self.arrows.render(t), color=GREY, opacity=out)
            # 河 = "ka", river (after the parts rejoin, and carried up with the glyph)
            oj = smooth((t - self.t_join) / 0.4)
            if oj > 0:
                self.place(fr, cam, self.ka, cx + lerp(330, 210, up), cy - lerp(40, 20, up), color=VERM, opacity=oj, scale=lerp(1, 0.8, up))
                self.place(fr, cam, self.river, cx + lerp(330, 210, up), cy + lerp(40, 26, up), color=GREY, opacity=oj, scale=lerp(1, 0.85, up))
        # the compounds
        for R in self.rows:
            fr_ = (t - R['t']) / 0.3
            if fr_ <= 0:
                continue
            y = R['y']
            self.place(fr, cam, R['g1'].at(fr_, soft=0.03), 200, y)
            self.place(fr, cam, R['g2'].at(fr_ - 1.0, soft=0.03, wet=0.3), 415, y)
            fm = (t - R['t'] - 0.6) / 0.45
            if fm > 0:
                self.place(fr, cam, bloom(R['gr'].at(min(fm, 1.05), soft=0.015, wet=0.45), t - R['t'] - 1.05, amount=0.12), 760, y - 8)
            for a, x, dy, col, dt in ((R['l1'], 200, 92, GREY, 0.2), (R['l2'], 415, 92, GREY, 0.5), (R['lr'], 760, 112, VERM, 0.95)):
                self.place(fr, cam, a, x, y + dy, color=col, opacity=smooth((t - R['t'] - dt) / 0.3))
        self.layer(fr, cam, self.ops.render(t), color=GREY)
        return fr.img

    def _brush(self, fr, cam, t, f):
        g = self.g
        x, y, k, u, moving = g.brush_pos(np.clip(f, 0, 1))
        wx, wy = GC[0] - g.w / 2 + x, GC[1] - g.h / 2 + y
        M = cam.matrix()
        sx, sy = M[:, :2] @ np.array([wx, wy]) + M[:, 2]
        lift = 0 if moving else 30
        if f <= 0:
            lift = 60
        if f > 1:
            lift = 30 + 700 * ease_out((t - self.t_write_end) / 0.9)
        draw_sprite(fr, *self.brush, sx, sy - lift * 0.5, -0.3, scale=0.62 * cam.zoom, shadow=0.3, lift=lift)

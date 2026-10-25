"""Scene 8 — "But here's the fascinating part: not every kanji is a picture.  Many characters combine a component
related to meaning, with another that gives a clue to pronunciation.  This allowed the writing system to express
far more than simple objects."
The brush writes 河 (river).  On "meaning" its water radical 氵 turns blue and steps out, tagged "meaning: water";
on "pronunciation" 可 turns vermilion and steps the other way, tagged "sound: ka"; they rejoin as 河 "ka", river.
Then the idea grows beyond things: a painting of a person resting beside a tree becomes its pictograph, person +
tree, and the brush turns it into 休 yasumi, "rest"; then 日 sun + 月 moon → 明 bright, and 田 field + 力 strength →
男 man (from the reference sheet @raj_in_jp sent)."""
import numpy as np

from ancient import ancient
from art import brush_sprite, draw_sprite
from engine import (FONT_EN, FONT_EN_ITALIC, FONT_JP, GREY, H, INK, PAPER, SEA, VERM, W, Cam, StrokeSet, blur, bloom,
                    ease_io, ease_out, keyframes, lerp, reveal_time, smooth, stroke, text_layer)
from glyphs import calligraphy
from poly_ink.brush import PRESS_DOWN
from scenes.base import SceneBase
from timeline import at

WATER = np.asarray(SEA, np.float32) * 0.85
GC = (540, 800)                         # 河 while it is explained
RC = (540, 700)                         # 休 while it forms
COMPOUNDS = [(('日', 'sun'), ('月', 'moon'), ('明', 'bright')), (('田', 'field'), ('力', 'strength'), ('男', 'man'))]
ROW_Y = [1112, 1250]


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
        # 休: picture → pictograph → kanji, in place
        from pictures import picture
        from engine import wash_map
        pc = picture('休', 400)
        D = pc['ink'].shape[0]
        self.rest = dict(t=self.t_more + 0.5, ink=pc['ink'], T=wash_map(D, D, (D / 2, D / 2), noise_amp=0.4, noise_scale=60, seed=3),
                         anc=ancient('休', 'oracle', 400, seed=41), kan=calligraphy('休', 360, seed=42),
                         rd=text_layer('yasumi', FONT_EN_ITALIC, 62)[0], en=text_layer('"rest"', FONT_EN, 48)[0],
                         fm=text_layer('a person + a tree', FONT_EN_ITALIC, 38)[0])
        # then two quick compounds
        self.rows = []
        ops = StrokeSet(W, H)
        for k, ((c1, e1), (c2, e2), (cr, er)) in enumerate(COMPOUNDS):
            t = self.t_more + 2.05 + 0.8 * k
            y = ROW_Y[k]
            self.rows.append(dict(t=t, y=y, g1=calligraphy(c1, 105, seed=30 + k), g2=calligraphy(c2, 105, seed=33 + k),
                                  gr=calligraphy(cr, 140, seed=36 + k), l1=text_layer(e1, FONT_EN_ITALIC, 32)[0],
                                  l2=text_layer(e2, FONT_EN_ITALIC, 32)[0], lr=text_layer(er, FONT_EN, 46)[0]))
            ops.add([stroke(W, H, [(296, y - 2), (334, y - 2)], 5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=50 + k),
                     stroke(W, H, [(315, y - 20), (315, y + 16)], 5, ink=0.7, dry=0.3, profile=PRESS_DOWN, seed=60 + k)], t + 0.15, 0.15)
            ops.add([stroke(W, H, [(500, y), (566, y - 2)], 5, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=70 + k),
                     stroke(W, H, [(548, y - 16), (568, y - 2), (548, y + 13)], 5, ink=0.7, dry=0.4, profile=PRESS_DOWN, seed=80 + k)],
                    t + 0.35, 0.15)
        self.ops = ops

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 900, 1.05)), (self.t_write_end, (540, 930, 1.0)),
                                  (self.t_more, (540, 940, 1.0)), (self.t1, (540, 935, 1.0))], ease=ease_io))

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
            cx, cy = lerp(GC[0], 215, up), lerp(GC[1], 330, up)
            sc = lerp(1.0, 0.36, up)
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
                self.place(fr, cam, self.ka, cx + lerp(330, 170, up), cy - lerp(40, 20, up), color=VERM, opacity=oj, scale=lerp(1, 0.8, up))
                self.place(fr, cam, self.river, cx + lerp(330, 170, up), cy + lerp(40, 26, up), color=GREY, opacity=oj, scale=lerp(1, 0.85, up))
        # 休: the painting, the pictograph traced over it, then the kanji
        R = self.rest
        u = t - R['t']
        if u > 0:
            p_op = 1 - smooth((u - 0.8) / 0.3)
            if p_op > 0:
                self.place(fr, cam, reveal_time(R['ink'], R['T'], u / 0.45 * 1.1, soft=0.08, wet=0.3), *RC, opacity=p_op)
            fo, fm = (u - 0.35) / 0.45, (u - 1.05) / 0.35
            if fo > 0 and fm < 1:
                self.place(fr, cam, R['anc'].at(fo, soft=0.03, wet=0.35), *RC, color=VERM, opacity=0.95 * (1 - smooth(fm)),
                           scale=1 - 0.06 * smooth(fm))
            if fm > 0:
                self.place(fr, cam, bloom(R['kan'].at(min(fm * 1.15, 1.05), soft=0.02, wet=0.45), u - 1.45, amount=0.12), RC[0], RC[1] - 6)
            lo = smooth((u - 1.25) / 0.3)
            rw, ew = R['rd'].shape[1], R['en'].shape[1]
            tot = rw + ew + 26
            self.place(fr, cam, R['rd'], RC[0] - tot / 2 + rw / 2, RC[1] + 232, color=VERM, opacity=lo)
            self.place(fr, cam, R['en'], RC[0] + tot / 2 - ew / 2, RC[1] + 236, color=GREY, opacity=lo)
            self.place(fr, cam, R['fm'], RC[0], RC[1] + 290, color=GREY, opacity=smooth((u - 1.45) / 0.3))
        # two quick compounds
        for R in self.rows:
            fr_ = (t - R['t']) / 0.25
            if fr_ <= 0:
                continue
            y = R['y']
            self.place(fr, cam, R['g1'].at(fr_, soft=0.03), 215, y)
            self.place(fr, cam, R['g2'].at(fr_ - 0.8, soft=0.03), 410, y)
            fm = (t - R['t'] - 0.45) / 0.35
            if fm > 0:
                self.place(fr, cam, bloom(R['gr'].at(min(fm, 1.05), soft=0.015, wet=0.45), t - R['t'] - 0.8, amount=0.12), 650, y - 6)
            for a, x, dy, col, dt in ((R['l1'], 215, 70, GREY, 0.15), (R['l2'], 410, 70, GREY, 0.35)):
                self.place(fr, cam, a, x, y + dy, color=col, opacity=smooth((t - R['t'] - dt) / 0.25))
            self.place(fr, cam, R['lr'], 760 + R['lr'].shape[1] / 2, y, color=VERM, opacity=smooth((t - R['t'] - 0.7) / 0.25))
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

"""Scene 7 — "'Yama' means mountain. 'Kawa' means river. 'Ki' means tree.  Look closely, and you can see how some
of these characters resemble the things they represent."
Each character is formed in place, so the step from picture to kanji is plain: a painting fills the page (a
mountain with three peaks, a river between its banks, a tree with branches and roots), its early pictograph is
traced over it in vermilion, the painting fades and leaves the lines, and the brush turns them into the kanji with
its reading.  Each finished kanji moves up to a row at the top.  On "Look closely" the same happens for the bird
(after the reference sheets @raj_in_jp sent): sparrow → pictograph → 鳥 tori.  On "resemble the things they
represent" the paintings come back faintly behind every character."""
import numpy as np

from ancient import ancient
from engine import (FONT_EN, FONT_EN_ITALIC, GREY, INK, SEA, VERM, Cam, bloom, ease_io, keyframes, lerp, reveal_time,
                    smooth, text_layer, wash_map, wipe_mix)
from glyphs import calligraphy
from pictures import picture
from scenes.base import SceneBase
from timeline import at

C = (540, 840)
S = 500
ITEMS = [('山', 'yama', 'mountain', 'Yama'), ('川', 'kawa', 'river', 'Kawa'), ('木', 'ki', 'tree', 'Ki'),
         ('鳥', 'tori', 'bird', 'Look closely')]
SHELF = [(250, 335), (540, 335), (830, 335)]
SH = 0.36


class Scene(SceneBase):
    paper_seed = 47

    def build(self):
        from timeline import quiet_gap
        self.items = []
        for k, (ch, rd, en, word) in enumerate(ITEMS):
            pc = picture(ch, S)
            D = pc['ink'].shape[0]
            self.items.append(dict(
                k=k, ch=ch, t=at(word)[0] - (0.2 if ch == '鳥' else 0.35), slow=1.5 if ch == '鳥' else 1.0,
                ink=pc['ink'], water=pc.get('water'), T=wash_map(D, D, (D / 2, D / 2), noise_amp=0.4, noise_scale=60, seed=k),
                anc=ancient(ch, 'oracle', S, seed=70 + k), kan=calligraphy(ch, 440, seed=80 + k),
                rd=text_layer(rd, FONT_EN_ITALIC, 66)[0], en=text_layer(en, FONT_EN, 50)[0]))
        for k in range(3):
            self.items[k]['t_shelf'] = self.items[k + 1]['t'] - 0.15
        self.t_hit, _ = quiet_gap(at('represent')[1] - 0.15, at('But heres')[0] + 0.05)   # the paintings return in the breath
        self.t_ghost = at('resemble')[0] - 0.3

    def cam(self, t):
        return Cam(*keyframes(t, [(self.t0, (540, 930, 1.03)), (self.items[3]['t'], (540, 920, 1.0)),
                                  (self.t1, (540, 900, 1.02))], ease=ease_io))

    def _pose(self, I, t):
        """(x, y, scale) of an item: in the centre while it forms, then on the shelf"""
        if 't_shelf' not in I:
            return C[0], C[1], 1.0
        m = ease_io((t - I['t_shelf']) / 0.5)
        x, y = SHELF[I['k']]
        return lerp(C[0], x, m), lerp(C[1], y, m), lerp(1.0, SH, m)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        ghost = smooth((t - self.t_ghost) / 0.8)
        glow = float(np.exp(-max(t - self.t_hit, 0) * 3) * (t > self.t_hit))
        for I in self.items:
            u = (t - I['t']) / I['slow']
            if u <= 0:
                continue
            x, y, sc = self._pose(I, t)
            # 1. the painting washes in; 4. it fades away, and returns faintly on "resemble"
            fp = u / 0.45
            p_op = (1 - smooth((u - 0.8) / 0.3)) + 0.3 * ghost * (1 + glow) * (u >= 1.1)
            if p_op > 0.01:
                if I['water'] is not None:
                    self.place(fr, cam, reveal_time(I['water'], I['T'], fp * 1.1, soft=0.08, wet=0.2), x, y,
                               color=SEA, opacity=0.4 * min(p_op, 1), scale=sc)
                self.place(fr, cam, reveal_time(I['ink'], I['T'], fp * 1.1, soft=0.08, wet=0.3), x, y, opacity=min(p_op, 1), scale=sc)
            # 2. the pictograph is traced over it in vermilion, then 5. the brush turns it into the kanji
            fo = (u - 0.35) / 0.45
            fm = (u - 1.05) / 0.35
            if fo > 0:
                op = 1 - smooth(fm)
                if op > 0:
                    self.place(fr, cam, I['anc'].at(fo, soft=0.03, wet=0.35), x, y, color=VERM, opacity=0.95 * op,
                               scale=sc * (1 - 0.06 * smooth(fm)))
            if fm > 0:
                d = bloom(I['kan'].at(min(fm * 1.15, 1.05), soft=0.02, wet=0.45), u - 1.45, amount=0.12)
                self.place(fr, cam, d, x, y - 6 * sc, scale=sc * (1 + 0.04 * glow))
            # 6. the reading, under the kanji
            lo = smooth((u - 1.25) / 0.3)
            if lo > 0:
                ls = lerp(0.62, 1.0, (sc - SH) / (1 - SH))
                rw, ew = I['rd'].shape[1] * ls, I['en'].shape[1] * ls
                tot = rw + ew + 26 * ls
                ly = y + (262 * sc if sc >= 1 else 262 * sc + 40 * (1 - sc))
                self.place(fr, cam, I['rd'], x - tot / 2 + rw / 2, ly, color=VERM, opacity=lo, scale=ls)
                self.place(fr, cam, I['en'], x + tot / 2 - ew / 2, ly + 4 * ls, color=GREY, opacity=lo, scale=ls)
        return fr.img

    def enter(self, prev_img, cur_img, f, prev_scene):
        if not hasattr(self, '_T'):
            self._T = prev_scene.exit_map()
        return wipe_mix(prev_img, cur_img, self._T, ease_io(f) * 1.15 - 0.05, soft=0.05, edge=0.5)

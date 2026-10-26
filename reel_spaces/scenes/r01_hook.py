"""Scene 1 — the hook: こんなに長い日本語の文章を、どうやって読むの？ / "How do Japanese people read such a long sentence,
without spaces?"
The question is brushed in, line by line, as Raj asks it in Japanese (どうやって読むの？ in vermilion).  Below, a long
handscroll of continuous text runs past — the opening of Natsume Sōseki's 吾輩は猫である (1905, public domain), which
mixes kanji, hiragana and katakana — with a reader looking up at it from the bottom of the frame.  On "without
spaces" small vermilion question marks pop up between the characters: where does one word end?"""
import cv2
import numpy as np

from art import blossom_cluster, cherry_branch, figure_back, pagoda, sun_disc
from engine import (FONT_JP, FONT_JP_FALLBACK, H, INK, SAKURA, SEPIA, VERM, W, Cam, StrokeSet, blur, bloom, cached,
                    ease_io, ease_out, keyframes, noise, reveal_time, smooth, stroke, text_layer)
from glyphs import text_glyphs
from scenes.base import SceneBase, dof
from timeline import JA_HOOK, at, quiet_gap

LINES = ['吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。',
         '吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。',
         'しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。']
SCROLL = (60, 1020, 790, 1100)        # x0, x1, y0, y1 of the paper
SPEED = (46, 38, 52)


class Scene(SceneBase):
    paper_seed = 61
    paper_stains = 0.1

    def build(self):
        a, b = JA_HOOK['start'], JA_HOOK['end']
        d = (b - a) / 3
        self.title = []
        for k, (txt, y, col) in enumerate((('こんなに長い', 300, INK), ('日本語の文章を', 418, INK), ('どうやって読むの？', 548, VERM))):
            D, T, _ = text_glyphs(txt, 92, font=FONT_JP, seed=10 + k, tracking=0.03)
            self.title.append(dict(D=D, T=T, halo=blur(D, 4), y=y, col=col, t=a - 0.55 + d * k, dur=d * 1.05))   # line 1 is half written in frame 1
        # the scroll's text: three long lines, each tiled twice so it can run past endlessly
        self.lines = []
        for k, txt in enumerate(LINES):
            a_, _ = text_layer(txt, FONT_JP_FALLBACK, 54, index=0)
            a_ = np.clip(a_ * (0.85 + 0.15 * noise(a_.shape[0], a_.shape[1], 8, 2, k)), 0, 1)
            self.lines.append(np.concatenate([a_, a_], 1))
        def deco():
            from inkpaint import Painting
            p = Painting(W, H, seed=62)
            r1 = p.ridge(1290, peaks=[(160, 120, 170), (620, 70, 220), (980, 140, 160)], rough=10, seed=1)
            p.wash(r1, dens=0.2, decay=90, tex=70, edge=0.3, mist_y=1350, mist_h=40, seed=2)
            ss = StrokeSet(W, H)
            pagoda(ss, 905, 1310, 0.9, 0, 0.01, seed=3, ink=0.8)
            tips = cherry_branch(ss, [(1130, 120), (1015, 175), (905, 160), (820, 228), (740, 214)], 18, 0, 0.01, seed=4)
            figure_back(ss, 330, 1500, 1.5, 0, seed=5, dur=0.01)
            bl = np.zeros((H, W), np.float32)
            for k, (bx, by) in enumerate(tips):
                bl = np.maximum(bl, blossom_cluster(H, W, bx, by, 12, 40, seed=10 + k))
            return {'land': p.D.astype(np.float32), 'deco': ss.render(9).astype(np.float32), 'bl': bl}
        self.A = cached('r01_deco_v1', deco)
        self.sun = sun_disc(H, W, 650, 1235, 115, seed=63)
        # the scroll paper with rollers
        x0, x1, y0, y1 = SCROLL
        pm = np.zeros((H, W), np.float32)
        pm[y0:y1, x0:x1] = 1
        self.scroll_mask = blur(pm, 1.2)
        rol = np.zeros((H, W), np.float32)
        for x in (x0 - 18, x1 + 2):
            rol[y0 - 26:y1 + 26, x:x + 16] = 1
        self.rollers = blur(rol, 1.0)
        # "without spaces": question marks between characters
        self.t_q = at('without spaces')[0]
        q, _ = text_layer('？', FONT_JP, 64)
        self.qmark = q
        r = np.random.default_rng(3)
        self.qpos = [(r.uniform(140, 940), 832 + 108 * (k % 3) - 24) for k in range(7)]
        # sync hit: in the breath after the Japanese question, a vermilion flick underlines it (+ taiko)
        c, n = quiet_gap(JA_HOOK['end'], JA_HOOK['end'] + 1.2)
        self.t_hit = c - n / 2 + 0.3
        fl = StrokeSet(W, H)
        fl.add(stroke(W, H, [(150, 616), (540, 606), (930, 614)], 12, ink=0.95, dry=0.45,
                      profile=((0, 0.4), (0.08, 1.0), (0.8, 0.8), (1, 0.2)), seed=21, red=True), self.t_hit - 0.03, 0.22)
        self.flick = fl

    def cam(self, t):
        return Cam(*keyframes(t, [(0.0, (540, 900, 1.03)), (self.t_q, (540, 930, 1.06)), (self.t1, (540, 960, 1.1))], ease=ease_io))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.sun, color=VERM, opacity=0.85, depth=0.85)
        self.layer(fr, cam, self.A['land'], depth=0.9)
        self.layer(fr, cam, self.A['deco'])
        self.layer(fr, cam, self.A['bl'], color=SAKURA, opacity=0.9)
        # the scroll: shadow, paper, rollers, running text
        x0, x1, y0, y1 = SCROLL
        sm = self.project(cam, self.scroll_mask)
        sh = self.project(cam, blur(np.roll(np.roll(self.scroll_mask, 10, 0), 6, 1), 10))
        fr.img *= (1 - 0.25 * sh)[..., None]
        fr.over(np.array([0.97, 0.94, 0.86], np.float32), sm, 0.96)
        self.layer(fr, cam, self.rollers, color=np.array([0.32, 0.2, 0.12], np.float32), opacity=0.95)
        T = np.zeros((H, W), np.float32)
        for k, (L, sp) in enumerate(zip(self.lines, SPEED)):
            half = L.shape[1] // 2
            off = int((t * sp + 300 * k) % half)
            y = y0 + 22 + 104 * k
            seg = L[:, off:off + (x1 - x0)]
            h = min(seg.shape[0], H - y)
            T[y:y + h, x0:x0 + seg.shape[1]] = np.maximum(T[y:y + h, x0:x0 + seg.shape[1]], seg[:h])
        T *= self.scroll_mask
        self.layer(fr, cam, T, color=INK, opacity=0.9)
        # where are the spaces?
        for k, (qx, qy) in enumerate(self.qpos):
            f = (t - self.t_q - 0.1 * k) / 0.25
            if f > 0:
                pop = 1 + 0.4 * np.exp(-f * 3)
                self.place(fr, cam, self.qmark, qx, qy, color=VERM, opacity=smooth(f), scale=pop)
        # the question, brushed in as it is spoken
        for L in self.title:
            f = (t - L['t']) / L['dur']
            if f > 0:
                d = bloom(reveal_time(L['D'], L['T'], f * 1.1, soft=0.05, wet=0.45), t - L['t'] - L['dur'], amount=0.14, halo=L['halo'])
                halo = np.clip(blur(cv2.dilate(d, np.ones((13, 13), np.uint8)), 6) * 1.4, 0, 1)
                fr.over(np.array([0.95, 0.92, 0.85], np.float32), self.project(cam, halo, 540 - d.shape[1] // 2, L['y'] - d.shape[0] // 2), 0.5)
                self.place(fr, cam, d, 540, L['y'], color=L['col'])
        self.layer(fr, cam, self.flick.render(t, red=True), color=VERM)
        return dof(fr.img, 0.3 * (1 - smooth(t / 0.7)), focus_y=0.25, band=0.2)

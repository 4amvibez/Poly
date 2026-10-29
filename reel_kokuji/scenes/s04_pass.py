"""Scene 4 — 峠, built.  "Next: mountain, up, and down.  Together they make とうげ, a mountain pass, a path through the
mountains."
The camera rises slowly over tall ink mountains in mist.  On "mountain", "up" and "down" the parts come in one by
one, 山 MOUNTAIN + 上 UP + 下 DOWN, and a dotted vermilion path climbs over the ridge behind them and runs down the far
side as Raj says "up" and "down".  On "make" they slide together and 峠 is brushed over them; card 峠 / とうげ /
mountain pass; mountain wind under it all."""

import r5art
from engine import H, VERM, W, Cam, StrokeSet, cached, smooth, spline, stroke
from scenes.compound import CompoundScene
from timeline import JA, at


class Scene(CompoundScene):
    paper_seed = 204
    chapter = 1

    def build(self):
        self.Mt = cached('s03_mountain_v1', r5art.mountain_scene)['ink']
        t_m = at('Next mountain')[0] + 0.25
        t_up = at('up', after=t_m)[0]
        t_down = at('down', after=t_up)[0]
        self.parts('山上下', ('MOUNTAIN', 'UP', 'DOWN'), (t_m, t_up, t_down), (200, 540, 880))
        self.c_pass = self.card('峠', ['とうげ'], ['kun'], 'tōge', 'mountain pass', seed=250)
        self.t_read = JA['touge']['start']
        self.result('峠', at('make', after=t_down)[0], self.c_pass, self.t_read, size=460, seed=251)
        self.t_hit = self.t_read
        ss = StrokeSet(W, H)
        pts = spline([(60, 1010), (250, 930), (420, 830), (540, 790), (660, 830), (830, 930), (1020, 1010)], 46)
        half = len(pts) // 2
        for k in range(0, len(pts) - 3, 3):
            t0 = t_up + 0.1 + k / half * 0.9 if k < half else t_down + 0.1 + (k - half) / half * 0.9
            ss.add(stroke(W, H, pts[k:k + 2], 6, ink=0.85, dry=0.5, seed=300 + k, red=True), t0, 0.05)
        self.path = ss

    def cam(self, t):
        base = super().cam(t)
        return Cam(540, 960 + 60 * (1 - smooth((t - self.t0) / 4)), base.zoom)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.Mt, 0, 160, opacity=0.6 * smooth((t - self.t0) / 0.8), depth=0.8)
        self.layer(fr, cam, self.path.render(t, red=True), color=VERM, opacity=0.85 * (1 - 0.5 * smooth((t - self.t_merge) / 0.5)))
        self.draw_build(fr, cam, t)
        self.header(fr, t)
        return fr.img

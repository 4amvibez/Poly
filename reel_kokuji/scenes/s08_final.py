"""Scene 7 — the final frame.  "Thanks for watching, till the end, and follow for more."  then, in Japanese, またね！
The red sun sinks softly into the ivory paper behind a last misty ridge.  THANK YOU FOR WATCHING / TILL THE END! comes
in on "Thanks", on "follow" the black banner FOLLOW @raj_in_jp sweeps in and holds, and as Raj says またね！ it is
brushed large in vermilion over the sun with "matane! · see you!" under it, and the 国字 seal is stamped."""
import numpy as np

import k6art
import kit
from art import sun_disc
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, GREY, FONT_HEAD, H, INK, VERM, W, Cam, StrokeSet, cached, noise, smooth, stroke,
                    text_layer)
from scenes.kscene import KanjiScene
from scenes.s01_hook import _seal
from timeline import JA, at

IVORY = np.array([0.98, 0.95, 0.88], np.float32)


class Scene(KanjiScene):
    paper_seed = 208

    def build(self):
        self.sun = sun_disc(H, W, 540, 880, 230, seed=81).astype(np.float32)
        self.hills = cached('k6_hills_v3', lambda: k6art.mist_hills(1080, seed=12))['ink']
        mt = text_layer('またね！', FONT_BRUSH_JP, 170)[0]
        self.matane = mt * (0.86 + 0.14 * noise(*mt.shape, 6, 2, 4))
        self.mt_sub, _ = text_layer('matane!  ·  see you!', FONT_EN_ITALIC, 44)
        self.ty, _ = text_layer('THANK YOU FOR WATCHING', FONT_HEAD, 58)
        self.te, _ = text_layer('TILL THE END!', FONT_HEAD, 58)
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 40, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 84)
        self.seal = _seal('国字')
        self.t_thanks = at('Thanks for watching')[0]
        self.t_follow = at('and follow')[1]
        self.t_mt = JA['matane']['start']
        self.t_hit = self.t_mt
        bn = StrokeSet(W, H)
        bn.add(stroke(W, H, [(250, 1336), (540, 1324), (830, 1332)], 118, ink=0.96, dry=0.45,
                      profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=171), self.t_follow - 0.05, 0.4)
        self.banner = bn

    def cam(self, t):
        return Cam(540, 960, 1.0 + 0.02 * smooth((t - self.t0) / 6))

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        sink = smooth((t - self.t0) / 6)
        self.layer(fr, cam, self.sun, 0, 80 * sink, color=VERM, opacity=0.8 * (1 - 0.55 * sink))
        self.layer(fr, cam, self.hills, opacity=0.8)
        ft = smooth((t - self.t_thanks + 0.1) / 0.35)
        kit.put(fr, cam, self.ty, 540, 330 + 20 * (1 - ft), INK, ft)
        fe = smooth((t - self.t_thanks - 0.35) / 0.35)
        kit.put(fr, cam, self.te, 540, 410 + 20 * (1 - fe), VERM, fe)
        fm = (t - self.t_mt + 0.1) / 0.4
        if fm > 0:
            kit.put(fr, cam, self.matane, 540, 860, VERM, smooth(fm), 1 + 0.25 * np.exp(-fm * 2.5), rot=-0.04)
            kit.put(fr, cam, self.mt_sub, 540, 1030, GREY, smooth(fm - 1.2))
        fb = (t - self.t_follow + 0.05) / 0.4
        if fb > 0:
            self.layer(fr, cam, self.banner.render(t), opacity=0.96)
            kit.put(fr, cam, self.handle, 540, 1324, IVORY, smooth((t - self.t_follow - 0.15) / 0.3), 1 + 0.1 * np.exp(-fb * 2.5), over=True)
            kit.put(fr, cam, self.follow, 540, 1238, INK, smooth(fb))
        fs = (t - self.t_hit) / 0.15
        if fs > 0:
            kit.put(fr, cam, self.seal, 905, 1180, VERM, smooth(fs) * 0.95, 0.7 * (1 + 0.5 * np.exp(-fs * 2)), rot=0.08)
        return fr.img

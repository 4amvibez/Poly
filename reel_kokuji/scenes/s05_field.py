"""Scene 5 — 畑, built.  "Fire, plus rice field, makes はたけ, a cultivated field."
The mountains dissolve into a wide field: rows of crops running away to a farmhouse on a low horizon.  On "fire"
火 comes in with a flickering brushed flame beside it (FIRE); on "rice field" 田 (RICE FIELD); on "makes" they slide together and 畑 is brushed over them; card 畑 / はたけ /
cultivated field."""
import numpy as np

import k6art
import kit
from engine import INK, VERM, cached, smooth
from scenes.compound import CompoundScene
from timeline import JA, at


class Scene(CompoundScene):
    paper_seed = 205
    chapter = 2

    def build(self):
        self.F = cached('k6_fields_v1', k6art.fields)['ink']
        self.fl = cached('k6_flame_v1', k6art.flame)
        t_fire = at('Fire plus')[0]
        t_rice = at('rice field')[0]
        self.parts('火田', ('FIRE', 'RICE FIELD'), (t_fire, t_rice), (300, 780), color=(VERM, INK))
        self.c_field = self.card('畑', ['はたけ'], ['kun'], 'hatake', 'cultivated field', seed=260)
        self.t_read = JA['hatake']['start']
        self.result('畑', at('makes', after=t_rice)[0], self.c_field, self.t_read, size=460, seed=261)
        self.t_hit = self.t_read
        self.t_fire = t_fire
        self.T_in = self.wipe_map((540, 1300), seed=31)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        m = self.wipe(self.T_in, smooth((t - self.t0) / 1.0))
        self.layer(fr, cam, self.F * m, 0, -60, opacity=0.55)
        # the flame beside 火, flickering until the merge
        fo = smooth((t - self.t_fire) / 0.3) * (1 - smooth((t - self.t_merge) / 0.4))
        if fo > 0:
            fl = 1 + 0.06 * np.sin(t * 17) + 0.04 * np.sin(t * 29)
            kit.put(fr, cam, self.fl['red'], 300, 455, VERM, 0.8 * fo, 0.85, 0.85 * fl)
            kit.put(fr, cam, self.fl['ink'], 300, 455, INK, 0.6 * fo, 0.85, 0.85 * fl)
        self.draw_build(fr, cam, t)
        self.header(fr, t)
        return fr.img

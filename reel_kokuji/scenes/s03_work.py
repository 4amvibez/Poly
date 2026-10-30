"""Scene 3 — 働, built by a push.  "This part is うごく, to move.  Add ひと, a person, who pushes it.  Put them
together, and you get はたらく, to work.  A person moving something: that's work!"
動 is brushed in large and drifts forward, a fading ink trail behind it; card 動 / うごく / to move.  On "Add" it
steps aside to the right and 人 is brushed on the left; card 人 / ひと / person.  On "pushes" 人 leans in and
shoves: it slides right with vermilion speed strokes behind it and 動 is knocked further right with a little
recoil.  On "Put them together" 人 slims into 亻 and the two glide together; on "you get" 働 is brushed over them
with a camera punch; card 働 / はたらく / to work.  On "A person moving something" the sum 人 + 動 = 働 / PERSON +
MOVE = WORK is set above, and on "that's work!" 働 gives a last pulse.  The header seals (働 峠 畑) show where we are."""
import numpy as np

import k6art
import kit
from engine import FONT_EN_BOLD, FONT_EN_ITALIC, FONT_JP, GREY, INK, VERM, cached, ease_io, ease_out, lerp, smooth, text_layer
from scenes.compound import BIG_Y, CompoundScene
from timeline import JA, at


class Scene(CompoundScene):
    paper_seed = 203
    chapter = 0

    def build(self):
        self.hills = cached('k6_hills_v2', lambda: k6art.mist_hills(1240, seed=8))['ink']
        self.walker = cached('k6_walker_v1', lambda: {'d': k6art.walker(0 + 140, 1290, 1.1, seed=5)})['d']
        self.dou = self.big_kanji('動', size=400, seed=231)
        self.hito = self.big_kanji('人', size=300, seed=232)
        self.nin = text_layer('亻', FONT_JP, 240)[0]
        self.dou_s = text_layer('動', FONT_JP, 240)[0]
        self.lab = {k: text_layer(v, FONT_EN_BOLD, 34, tracking=8)[0] for k, v in (('p', 'PERSON'), ('m', 'TO MOVE'))}
        self.plus, _ = text_layer('+', FONT_EN_BOLD, 96)
        self.c_dou = self.card('動', ['うごく'], ['kun'], 'ugoku', 'to move', seed=240)
        self.c_hito = self.card('人', ['ひと'], ['kun'], 'hito', 'person', seed=241)
        self.c_work = self.card('働', ['はたらく'], ['kun'], 'hataraku', 'to work', seed=242)
        self.t_first = at('This part is')[0]
        self.t_ugoku = JA['ugoku']['start']
        self.t_add = at('Add')[0]
        self.t_hito = JA['hito']['start']
        self.t_push = at('pushes')[0]
        self.t_comb = at('Put them together')[0]
        self.t_get = at('you get')[0]
        self.t_read = JA['hataraku']['start']
        self.t_sum = at('A person moving')[0]
        self.t_work = at("that's work")[0]
        self.t_hit = self.t_read
        self.result('働', self.t_get, self.c_work, self.t_read, size=460, seed=243)
        self.sum = [(text_layer(c, FONT_EN_BOLD if c in '+=' else FONT_JP, 84)[0], text_layer(l, FONT_EN_BOLD, 26, tracking=6)[0], x, col)
                    for c, l, x, col in (('人', 'PERSON', 300, INK), ('+', '', 420, VERM), ('動', 'MOVE', 540, INK),
                                         ('=', '', 660, VERM), ('働', 'WORK', 780, VERM))]
        self._note, _ = text_layer('a memory aid, not the full history', FONT_EN_ITALIC, 30)
        self.dash = kit.blur(np.pad(np.ones((10, 120), np.float32), 12), 3)

    def cam(self, t):
        cam = super().cam(t)
        u = t - self.t_push - 0.3
        cam.cx += 10 * np.exp(-u * 8) * np.sin(u * 40) * (u > 0)              # the shove lands
        return cam

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.hills, opacity=0.8)
        side = ease_io((t - self.t_add) / 0.6)                          # 動 steps aside for 人
        push = ease_io((t - self.t_push) / 0.55)                        # 人 shoves 動 to the right
        knock = ease_out(np.clip((t - self.t_push - 0.25) / 0.45, 0, 1)) # 動 lags a beat, then is knocked on
        comb = ease_io((t - self.t_comb) / 0.6)                         # 亻 and 動 glide together
        merge = ease_io((t - self.t_get) / 0.5)
        fd = (t - self.t_first + 0.15) / 0.5
        if fd > 0:
            drift = 30 * ease_io((t - self.t_first) / 1.6)
            x = lerp(540 + drift, 720, side)
            x = x + 90 * knock - 14 * np.exp(-max(t - self.t_push - 0.7, 0) * 6) * (t > self.t_push + 0.7)
            x = lerp(x, 640, comb)
            x = lerp(x, 540, merge)
            s_ = lerp(1.0, 0.62, side)
            o = 1 - smooth((comb - 0.4) / 0.4)
            for k in range(3, 0, -1):                                   # the trail
                kit.put(fr, cam, self.dou.dens, x - 40 * k * (1 - side), 700, INK, 0.08 * k / 3 * smooth(fd) * (1 - side), s_)
            kit.put(fr, cam, self.dou.at(fd, soft=0.05), x, 700, INK, o, s_, rot=0.05 * knock * (1 - comb))
            if comb > 0:
                kit.put(fr, cam, self.dou_s, x, 700, INK, smooth((comb - 0.3) / 0.4) * (1 - smooth((merge - 0.55) / 0.45)), 1.0)
            kit.put(fr, cam, self.lab['m'], x, 880, GREY, smooth((t - self.t_ugoku) / 0.3) * (1 - comb))
        # 人, then the shove, then 亻
        fh = (t - self.t_add - 0.2) / 0.45
        if fh > 0:
            x = lerp(260, 470, push)
            x = lerp(x, 440, comb)
            x = lerp(x, 540, merge)
            lean = -0.16 * push * (1 - comb)
            kit.put(fr, cam, self.hito.at(fh, soft=0.05), x, 700, INK, 1 - smooth((comb - 0.1) / 0.5), lerp(1.0, 0.7, comb), rot=lean)
            fn = smooth((comb - 0.2) / 0.5) * (1 - smooth((merge - 0.55) / 0.45))
            kit.put(fr, cam, self.nin, x, 700, INK, fn, lerp(1.4, 1.0, smooth((comb - 0.2) / 0.5)), 1.0)
            kit.put(fr, cam, self.lab['p'], x, 880, GREY, smooth(fh - 1) * (1 - comb))
            sp = np.sin(np.clip((t - self.t_push) / 0.8, 0, 1) * np.pi)  # speed strokes behind the shove
            for k, dy in enumerate((-70, 0, 70)):
                kit.put(fr, cam, self.dash, x - 150 - 30 * k, 700 + dy, VERM, 0.7 * sp, 0.6 + 0.6 * push)
        # the traveller walking in on "a person"
        fw = (t - self.t_hito) / 2.5
        if 0 < fw:
            self.layer(fr, cam, self.walker, -60 + 200 * ease_io(min(fw, 1)), 2 * np.sin(t * 9), opacity=smooth(fw * 4) * (1 - merge))
        # 働, brushed over the merge
        fb = (t - self.t_get - 0.3) / 0.55
        if fb > 0:
            kit.put(fr, cam, self._big.at(fb, soft=0.05), 540, BIG_Y, INK, 1.0, 1 + 0.1 * np.exp(-fb * 3))
        # the cards
        self.draw_card(fr, cam, t, self.c_dou, self.t_ugoku - 0.35, op=1 - smooth((t - self.t_add) / 0.3), pulse=self.t_ugoku)
        if t > self.t_add:
            self.draw_card(fr, cam, t, self.c_hito, self.t_hito - 0.35, op=1 - smooth((t - self.t_comb) / 0.3), pulse=self.t_hito)
        if t > self.t_get:
            self.draw_card(fr, cam, t, self.c_work, self.t_read - 0.35, pulse=self.t_read)
        # the sum, set above on "A person moving something"
        for k, (g, l, x, col) in enumerate(self.sum):
            f = (t - self.t_sum - 0.15 * k) / 0.3
            if f > 0:
                p = 1 + 0.35 * np.exp(-max(t - self.t_work, 0) * 5) * (t > self.t_work) * (k == 4)
                kit.put(fr, cam, g, x, 392, col, smooth(f), (1 + 0.25 * np.exp(-f * 3)) * p)
                kit.put(fr, cam, l, x, 448, GREY, smooth(f - 0.5))
        kit.put(fr, cam, self._note, 540, 1500, GREY, 0.75 * smooth((t - self.t_sum - 0.8) / 0.4))
        self.header(fr, t)
        return fr.img

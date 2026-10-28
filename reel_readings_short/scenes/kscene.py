"""kscene — what the four kanji scenes (山 日 水 人) share: the big brushed kanji, the word card at the bottom on a soft
paper plate, ink-sweep wipes between two paintings, and the hanging calendar."""
import numpy as np

import kit
from cards import WordCard
from engine import FONT_BRUSH_JP, H, INK, VERM, W, blur, smooth, wash_map
from glyphs import typeset
from scenes.base import SceneBase

CARD_Y = 1150
CARD_SIZE = 140
PLATE_RGB = np.array([0.95, 0.92, 0.85], np.float32)
CAL_RGB = np.array([0.97, 0.95, 0.89], np.float32)


class KanjiScene(SceneBase):
    def big_kanji(self, ch, size=520, seed=0):
        return typeset(ch, size, font=FONT_BRUSH_JP, seed=seed, medians=False, sweep=(0.7, 1))

    def card(self, chars, readings, kinds, romaji, meaning, focus=None, seed=0):
        return WordCard(chars, readings, kinds, romaji, meaning, size=CARD_SIZE, seed=seed, focus=focus)

    def plate(self, fr, cam, x, y, w, h, op):
        """a soft paper plate so a card reads over a painting"""
        key = (w, h)
        if not hasattr(self, '_plates'):
            self._plates = {}
        if key not in self._plates:
            p = np.zeros((h + 80, w + 80), np.float32)
            p[40:-40, 40:-40] = 1
            self._plates[key] = blur(p, 18)
        kit.put(fr, cam, self._plates[key], x, y, PLATE_RGB, 0.82 * op, over=True)

    def draw_card(self, fr, cam, t, card, t0, x=540, y=CARD_Y, op=1.0, pulse=None, scale=1.0, dim=1.0, pulses=None):
        if op <= 0 or t < t0:
            return
        w = int(max(card.n, 2) * CARD_SIZE * 1.1 * scale + 120)
        self.plate(fr, cam, x, y + 0.7 * CARD_SIZE * scale, w, int(CARD_SIZE * 2.2 * scale), op * smooth((t - t0) / 0.3))
        card.at(t).draw(fr, cam, x, y, t0, op=op, pulse=pulse, scale=scale, dim=dim, pulses=pulses)

    def wipe_map(self, origin, seed=0):
        if not hasattr(self, '_wipes'):
            self._wipes = {}
        if origin not in self._wipes:
            self._wipes[origin] = wash_map(H, W, origin, noise_amp=0.25, noise_scale=220, seed=seed)
        return self._wipes[origin]

    @staticmethod
    def wipe(T, f, soft=0.06):
        """0..1 mask of how far an ink sweep with time map T has got at progress f"""
        return np.clip((f * 1.12 - 0.04 - T) / soft, 0, 1).astype(np.float32)

    def calendar(self, fr, cam, t, C, x, y, t0, mark, op=1.0, t_mark=None):
        """the hanging calendar centred at (x, y), swinging in from t0; mark = index of the day circled in vermilion"""
        f = smooth((t - t0) / 0.45)
        if f <= 0 or op <= 0:
            return
        w, h = C['size']
        sw = 0.05 * np.exp(-max(t - t0, 0) * 3) * np.sin((t - t0) * 9)
        yy = y - 40 * (1 - f)
        kit.put(fr, cam, blur(C['fill'], 10), x + 10, yy + 14, INK, 0.18 * f * op, rot=sw)
        kit.put(fr, cam, C['fill'], x, yy, CAL_RGB, f * op, rot=sw, over=True)
        kit.put(fr, cam, C['ink'], x, yy, INK, f * op, rot=sw)
        kit.put(fr, cam, C['red'], x, yy, VERM, f * op, rot=sw)
        cx, cy = C['cells'][mark]
        circ = self._circle()
        fc = (t - (t0 + 0.5 if t_mark is None else t_mark - 0.4)) / 0.4
        if fc > 0:
            kit.put(fr, cam, circ.render(min(fc, 1.3), red=True), x + cx - w / 2, yy + cy - h / 2, VERM, op, 1 + 0.15 * np.exp(-fc * 3))

    def ring(self, fr, cam, t, t0, x, y, color, r=1.0, dur=0.4):
        """a brushed ring drawn round (x, y) from t0"""
        f = (t - t0) / dur
        if f > 0:
            kit.put(fr, cam, self._circle().render(min(f, 1.3), red=True), x, y, color, 1.0, r * (1 + 0.15 * np.exp(-f * 3)))

    def _circle(self):
        if not hasattr(self, '_circ'):
            from engine import StrokeSet, spline, stroke
            s = 130
            ss = StrokeSet(s, s)
            pts = [(s / 2 + 46 * np.cos(a), s / 2 + 42 * np.sin(a)) for a in np.linspace(-2.0, -2.0 + 2 * np.pi * 1.08, 30)]
            st = stroke(s, s, spline(pts, 4), 6, ink=0.95, dry=0.4, profile=((0, 0.4), (0.1, 1), (0.85, 0.9), (1, 0.2)), seed=7, red=True)
            ss.add(st, 0, 1)
            self._circ = ss
        return self._circ

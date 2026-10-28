"""kscene — what the four kanji scenes (山 日 水 人) share: the big brushed kanji (first scratched in its 3,000-year-old
oracle-bone form, which then turns into the modern character), the word card at the bottom on a soft paper plate,
the 訓 / 音 seal stamped beside each card, the progress header (four small seals 山 日 水 人 along the top, the one
being taught in vermilion, the ones done ticked, with its meaning under them), ink-sweep wipes between two
paintings, and the hanging calendar."""
import numpy as np

import kit
from cards import WordCard
from ancient import ancient
from cards import INDIGO
from engine import FONT_BRUSH_JP, FONT_EN_BOLD, FONT_EN_ITALIC, FONT_HEAD, FONT_JP, GREY, H, INK, VERM, W, Cam, blur, ease_io, lerp, smooth, text_layer, wash_map
from glyphs import typeset
from scenes.base import SceneBase

CARD_Y = 1150
CARD_SIZE = 140
PLATE_RGB = np.array([0.95, 0.92, 0.85], np.float32)
CAL_RGB = np.array([0.97, 0.95, 0.89], np.float32)
IVORY = np.array([0.98, 0.95, 0.88], np.float32)
CHAPTERS = [('山', 'MOUNTAIN'), ('日', 'SUN · DAY'), ('人', 'PERSON'), ('読', "ON'YOMI · KUN'YOMI")]
SEAL_Y, SEAL_X, SEAL = 246, [405, 495, 585, 675], 62
SCREEN = Cam(540, 960, 1.0)


class KanjiScene(SceneBase):
    chapter = None                      # 0..3: which of 山 日 水 人 this scene teaches

    # ---------------------------------------------------------------- the 3,000-year-old form turning into the kanji
    def origin(self, ch, size, t0, seed=0):
        """set up the oracle-bone form of `ch`, scratched from t0 and turning into the modern kanji; returns when
        the modern kanji starts to show"""
        self._anc = ancient(ch, 'oracle', int(size * 0.92), seed=seed)
        self._anc_t0 = t0
        self._ago = text_layer('3,000 YEARS AGO', FONT_EN_BOLD, 38, tracking=8)[0]
        self._now = text_layer('TODAY', FONT_EN_BOLD, 38, tracking=8)[0]
        self.t_modern = t0 + 0.75
        return self.t_modern

    def draw_origin(self, fr, cam, t, x, y, label_y=None):
        """the scratched pictograph (fading as the modern kanji is brushed over it) and its label"""
        t0 = self._anc_t0
        f = (t - t0) / 0.45
        if f <= 0:
            return
        m = smooth((t - self.t_modern) / 0.35)
        kit.put(fr, cam, self._anc.at(f, soft=0.05), x, y, INK, 0.85 * (1 - m), lerp(1.0, 1.12, m))
        ly = (y + 300) if label_y is None else label_y
        out = 1 - smooth((t - self.t_modern - 0.7) / 0.3)
        kit.put(fr, cam, self._ago, x, ly, GREY, smooth((t - t0) / 0.2) * (1 - m))
        kit.put(fr, cam, self._now, x, ly, VERM, m * out)

    def modern(self, t, dur=0.45):
        """progress for brushing the modern kanji in over the pictograph"""
        return (t - self.t_modern) / dur

    # ---------------------------------------------------------------- 訓 / 音 seals
    def reading_seal(self, fr, cam, t, t0, kind, x=905, y=1075, op=1.0, full=True):
        """stamped at t0 for 訓 (indigo, kun'yomi) or 音 (vermilion, on'yomi).  full = a labelled tag centred above
        the card (the seal, then KUN'YOMI / native Japanese reading spelled out, so a beginner can read it);
        otherwise just the square seal at (x, y)"""
        f = (t - t0) / 0.12
        if f <= 0 or op <= 0:
            return
        if not hasattr(self, '_seals'):
            from engine import noise
            self._seals = {k: text_layer(k, FONT_JP, 70)[0] for k in '訓音'}
            self._sq = kit.box(112, 112, 3.0) * (0.88 + 0.12 * np.asarray(noise(112, 112, 8, 3, 21)))
            self._tag = {k: (text_layer(n, FONT_EN_BOLD, 50, tracking=6)[0], text_layer(g, FONT_EN_ITALIC, 36)[0])
                         for k, n, g in (('訓', "KUN'YOMI", 'native Japanese reading'), ('音', "ON'YOMI", 'Chinese-based reading'))}
        col = INDIGO if kind == '訓' else VERM
        sc = 1 + 0.45 * np.exp(-f * 2.6)
        o = smooth(f) * op
        if not full:
            kit.put(fr, cam, self._sq, x, y, col, 0.95 * o, sc, rot=-0.06)
            kit.put(fr, cam, self._seals[kind], x, y + 2, IVORY, o, sc, rot=-0.06, over=True)
            return
        name, gloss = self._tag[kind]
        w = 0.7 * 112 + 24 + max(name.shape[1], gloss.shape[1])
        x0, y = 540 - w / 2, 1005
        self.plate(fr, cam, 540, y, int(w + 40), 110, o)
        sx = x0 + 0.7 * 56
        kit.put(fr, cam, self._sq, sx, y, col, 0.95 * o, 0.7 * sc, rot=-0.06)
        kit.put(fr, cam, self._seals[kind], sx, y + 2, IVORY, o, 0.7 * sc, rot=-0.06, over=True)
        tx = x0 + 0.7 * 112 + 24
        kit.put(fr, cam, name, tx + name.shape[1] / 2, y - 22, col, smooth(f - 0.5))
        kit.put(fr, cam, gloss, tx + gloss.shape[1] / 2, y + 27, INK, smooth(f - 1.5))

    # ---------------------------------------------------------------- the progress header
    def header(self, fr, t):
        """four small seals along the top; drawn in screen space so it holds still while the page moves"""
        if self.chapter is None:
            return
        if not hasattr(self, '_hdr'):
            self._hdr = dict(ch=[text_layer(c, FONT_JP, 40)[0] for c, _ in CHAPTERS], sq=kit.box(SEAL, SEAL, 2.0),
                             rim=kit.rim(SEAL, SEAL, inset=6, t=3), word=text_layer(CHAPTERS[self.chapter][1], FONT_HEAD, 40,
                                                                                    tracking=10)[0])
        h = self._hdr
        a = smooth((t - self.t0 - 0.1) / 0.4)
        for k, x in enumerate(SEAL_X):
            if k == self.chapter:
                f = (t - self.t0 - 0.15) / 0.15
                sc = 1 + 0.35 * np.exp(-max(f, 0) * 2.6)
                kit.put(fr, SCREEN, h['sq'], x, SEAL_Y, VERM, smooth(f), sc)
                kit.put(fr, SCREEN, h['ch'][k], x, SEAL_Y, IVORY, smooth(f), sc, over=True)
            elif k < self.chapter:
                kit.put(fr, SCREEN, h['sq'], x, SEAL_Y, VERM, 0.35 * a)
                kit.put(fr, SCREEN, h['ch'][k], x, SEAL_Y, IVORY, a, over=True)
            else:
                kit.put(fr, SCREEN, h['rim'], x, SEAL_Y, GREY, 0.6 * a)
                kit.put(fr, SCREEN, h['ch'][k], x, SEAL_Y, GREY, 0.55 * a)
        kit.put(fr, SCREEN, h['word'], 540, SEAL_Y + 64, INK, smooth((t - self.t0 - 0.3) / 0.3))

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

    def draw_card(self, fr, cam, t, card, t0, x=540, y=CARD_Y, op=1.0, pulse=None, scale=1.0, dim=1.0):
        if op <= 0 or t < t0:
            return
        w = int(max(card.n, 2) * CARD_SIZE * 1.1 * scale + 120)
        self.plate(fr, cam, x, y + 0.7 * CARD_SIZE * scale, w, int(CARD_SIZE * 2.2 * scale), op * smooth((t - t0) / 0.3))
        card.at(t).draw(fr, cam, x, y, t0, op=op, pulse=pulse, scale=scale, dim=dim)

    def scroll(self, fr, cam, t, t0, x, y, w, h, edge, vertical=False, op=1.0):
        """a hanging scroll centred at (x, y), unrolling from t0: washi with a coloured mount (edge), dark rods at
        its ends (top and bottom when vertical, left and right when not)"""
        f = ease_io((t - t0) / 0.45)
        if f <= 0 or op <= 0:
            return
        key = (w, h)
        if not hasattr(self, '_scrolls'):
            self._scrolls = {}
        if key not in self._scrolls:
            from engine import noise
            b = kit.box(h, w, 2.0)
            self._scrolls[key] = dict(B=b, mount=kit.rim(h, w, inset=4, t=14), line=kit.rim(h, w, inset=24, t=2),
                                      grain=(b * (0.9 + 0.1 * noise(h, w, 30, 3, 31))).astype(np.float32),
                                      rod=kit.box(26, w + 60, 2.0) if vertical else kit.box(h + 60, 26, 2.0))
        S = self._scrolls[key]
        sx, sy = (1.0, f) if vertical else (f, 1.0)
        yy = y - (1 - f) * h / 2 if vertical else y                 # a vertical scroll unrolls downwards from its top rod
        kit.put(fr, cam, blur(S['B'], 10), x + 8, yy + 12, INK, 0.22 * op, sx, sy)
        kit.put(fr, cam, S['grain'], x, yy, PLATE_RGB, 0.96 * op, sx, sy, over=True)
        kit.put(fr, cam, S['mount'], x, yy, edge, 0.85 * op, sx, sy)
        kit.put(fr, cam, S['line'], x, yy, edge, 0.5 * op, sx, sy)
        for s_ in (-1, 1):
            if vertical:
                kit.put(fr, cam, S['rod'], x, yy + s_ * h * f / 2, INK, 0.92 * op)
            else:
                kit.put(fr, cam, S['rod'], x + s_ * w * f / 2, y, INK, 0.92 * op)

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

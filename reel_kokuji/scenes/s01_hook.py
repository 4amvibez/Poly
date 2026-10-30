"""Scene 1 — the hook.  "Two parts.  Not China.  Japan made this."
Frame 1 is already the picture: aged washi, a vermilion sun behind misty hills, WHAT MAKES THIS? brushed across the
top.  亻 and 動 slam onto the paper far apart (one on "Two", one on "parts"), each with a dust ring and a camera
kick.  On "Not China." a small FROM CHINA? appears under them and a vermilion brush slash strikes it out.  On
"Japan made this." the two parts glide together and lock into a large brushed 働 (camera punch), and a vermilion
MADE IN JAPAN stamp slams onto it as the line ends (sync)."""
import numpy as np

import k6art
import kit
from art import sun_disc
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN_BOLD, FONT_HEAD, FONT_JP, GREY, H, INK, VERM, W, Cam, StrokeSet,
                    blur, cached, ease_io, ease_out, lerp, noise, smooth, stroke, text_layer)
from glyphs import typeset
from scenes.kscene import KanjiScene
from timeline import at

X_NIN, X_DOU, Y = 250, 735, 860             # where the parts land, far apart
X_NIN_IN, X_DOU_IN = 418, 610               # where they sit inside 働


class Scene(KanjiScene):
    paper_seed = 201
    paper_stains = 0.1

    def build(self):
        self.sun = sun_disc(H, W, 540, 880, 300, seed=5).astype(np.float32)
        self.hills = cached('k6_hills_v1', lambda: k6art.mist_hills(1180, seed=3))['ink']
        self.title1 = self.brush('WHAT MAKES', 150, 1)
        self.title2 = self.brush('THIS?', 190, 2)
        self.nin = text_layer('亻', FONT_JP, 330)[0]
        self.dou = text_layer('動', FONT_JP, 330)[0]
        self.big = typeset('働', 520, font=FONT_BRUSH_JP, seed=243, medians=False, sweep=(0.7, 1))
        self.china, _ = text_layer('FROM CHINA?', FONT_HEAD, 64)
        self.ring = _ring()
        self.stamp = _stamp(('MADE IN', 'JAPAN'))
        w0, w1 = at('Two')[0], at('parts')[0]
        self.t_nin, self.t_dou = min(0.05, w0 - 0.05), w1 - 0.02     # 亻 hits on frame 2: motion from the start
        self.t_not = at('Not China')[0]
        self.t_cross = at('Not China')[1] - 0.15
        self.t_join = at('Japan made')[0] - 0.05
        self.t_lock = self.t_join + 0.55
        self.t_hit = at('this', after=self.t_join)[1] - 0.1        # sync: the stamp lands as the line ends
        ss = StrokeSet(W, H)
        ss.add(stroke(W, H, [(300, 1150), (540, 1120), (790, 1088)], 20, ink=0.95, dry=0.5,
                      profile=((0, 0.5), (0.1, 1), (0.85, 0.9), (1, 0.3)), seed=19, red=True), self.t_cross, 0.25)
        self.slash = ss

    @staticmethod
    def brush(s, size, seed):
        d, _ = text_layer(s, FONT_BRUSH_EN, size)
        return d * (0.86 + 0.14 * noise(*d.shape, 6, 2, seed))

    def kick(self, t, t0, amp):
        u = t - t0
        return amp * np.exp(-u * 9) * np.sin(u * 55) * (u > 0)

    def cam(self, t):
        shake = self.kick(t, self.t_nin, 6) + self.kick(t, self.t_dou, 7) + self.kick(t, self.t_hit, 9)
        punch = 0.04 * np.exp(-max(t - self.t_lock, 0) * 5) * (t > self.t_lock) + 0.03 * np.exp(-max(t - self.t_hit, 0) * 6) * (t > self.t_hit)
        return Cam(540 + shake, 960 + 0.6 * shake, 1.0 + 0.05 * smooth(t / 4.5) + punch)

    def render(self, t):
        cam = self.cam(t)
        fr = self.frame(cam)
        self.layer(fr, cam, self.sun, 0, 50 * (1 - ease_out(t / 3)), color=VERM, opacity=0.55)
        self.layer(fr, cam, self.hills, opacity=0.85)
        # the question, there from frame 1, settling in
        kit.put(fr, cam, self.title1, 540, 330, INK, 1.0, 1.06 - 0.06 * ease_out(t / 0.5), rot=-0.02)
        kit.put(fr, cam, self.title2, 540, 480, VERM, 1.0, 1.1 - 0.1 * ease_out(t / 0.6), rot=-0.03)
        # the two parts slam in apart, then glide together into 働
        join = ease_io((t - self.t_join) / 0.55)
        fade = 1 - smooth((t - self.t_lock + 0.1) / 0.25)
        for g, t0, x0, x1 in ((self.nin, self.t_nin, X_NIN, X_NIN_IN), (self.dou, self.t_dou, X_DOU, X_DOU_IN)):
            f = (t - t0) / 0.16
            if f <= 0:
                continue
            s = lerp(1.0, 0.82, join) * (1 + 0.6 * (1 - ease_out(min(f, 1))))
            kit.put(fr, cam, g, lerp(x0, x1, join), Y, INK, smooth(f * 1.5) * fade, s)
            fr_ = (t - t0) / 0.45                                            # the dust ring as it hits the paper
            if 0 < fr_ < 1:
                kit.put(fr, cam, self.ring, lerp(x0, x1, join), Y + 165, INK, 0.4 * (1 - fr_), 0.6 + 0.9 * ease_out(fr_))
        kit.put(fr, cam, self._plus(), 492, Y, VERM, smooth((t - self.t_dou - 0.2) / 0.25) * (1 - join))
        # FROM CHINA?  struck out
        fc = smooth((t - self.t_not) / 0.25) * (1 - smooth((t - self.t_lock) / 0.3))
        kit.put(fr, cam, self.china, 540, 1110, GREY, fc)
        if fc > 0:
            self.layer(fr, cam, self.slash.render(t, red=True), color=VERM, opacity=fc)
        # 働, brushed fast as they lock
        fb = (t - self.t_lock + 0.15) / 0.35
        if fb > 0:
            kit.put(fr, cam, blur(self.big.dens, 14), 540, Y + 10, VERM, 0.25 * np.exp(-max(fb - 1, 0) * 3), 1.08)
            kit.put(fr, cam, self.big.at(fb, soft=0.05), 540, Y, INK, 1.0, 1 + 0.12 * np.exp(-fb * 3))
        fs = (t - self.t_hit) / 0.12
        if fs > 0:
            kit.put(fr, cam, self.stamp, 780, 1120, VERM, smooth(fs) * 0.95, 1.3 + 0.9 * np.exp(-fs * 2.4), rot=-0.12)
        return fr.img

    def _plus(self):
        if not hasattr(self, '_pl'):
            self._pl = text_layer('+', FONT_EN_BOLD, 110)[0]
        return self._pl


def _ring():
    """a soft ink ring: the puff of dust as a part hits the paper"""
    s = 400
    yy, xx = np.mgrid[:s, :s] - s / 2
    r = np.hypot(xx, yy * 5) / (s / 2)
    d = np.exp(-((r - 0.75) / 0.2) ** 2) * (0.6 + 0.4 * noise(s, s, 12, 3, 77))
    return d.astype(np.float32)


def _stamp(lines):
    """a rectangular vermilion hanko with the words cut out in white (density: 1 = vermilion)"""
    w, h = 300, 190
    d = np.zeros((h, w), np.float32)
    d[10:-10, 10:-10] = 1
    d[24:-24, 24:-24] *= 0.0
    d[30:-30, 30:-30] = 1
    d = blur(d, 1.2) * (0.78 + 0.22 * noise(h, w, 10, 3, 141))
    for k, s in enumerate(lines):
        g, _ = text_layer(s, FONT_EN_BOLD, 46 if k == 0 else 60, tracking=4)
        gh, gw = g.shape
        cx, cy = w // 2, 72 + 56 * k
        y0, x0 = int(cy - gh / 2), int(cx - gw / 2)
        d[y0:y0 + gh, x0:x0 + gw] *= 1 - 0.95 * g
    return d


def _seal(text):
    """a square vermilion seal with the characters cut out in white (density: 1 = vermilion)"""
    s = 160
    d = np.zeros((s, s), np.float32)
    d[8:-8, 8:-8] = 1
    d = blur(d, 1.2) * (0.82 + 0.18 * noise(s, s, 10, 3, 140))
    for k, c in enumerate(text):
        g, _ = text_layer(c, FONT_JP, 64)
        gh, gw = g.shape
        cx, cy = 80, 46 + 68 * k
        d[int(cy - gh / 2):int(cy - gh / 2) + gh, int(cx - gw / 2):int(cx - gw / 2) + gw] *= 1 - 0.95 * g
    return d

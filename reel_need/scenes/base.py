"""shared scene machinery: paper, layer projection, depth of field, transitions"""
import numpy as np

from engine import H, INK, W, Cam, Frame, blur, make_paper, shift_matrix, warp


class SceneBase:
    paper_size = (W, H)
    paper_seed = 0
    paper_stains = 0.06

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1
        pw, ph = self.paper_size
        self.paper = make_paper(pw, ph, self.paper_seed, stains=self.paper_stains)
        self.build()

    def build(self):
        pass

    # --- per-frame helpers
    def frame(self, cam, depth=1.0, ref=None, paper=None):
        p = self.paper if paper is None else paper
        return Frame(warp(p, cam.matrix(depth, ref), border='reflect'))

    def layer(self, fr, cam, dens, x0=0, y0=0, depth=1.0, color=INK, opacity=1.0, ref=None):
        if opacity <= 0 or dens is None:
            return
        M = shift_matrix(cam.matrix(depth, ref), x0, y0)
        fr.ink(warp(dens, M), color, opacity)

    def place(self, fr, cam, dens, cx, cy, color=INK, opacity=1.0, scale=1.0, depth=1.0, rot=0.0):
        """ink a patch centred on world point (cx, cy), optionally scaled / rotated about its centre"""
        if opacity <= 0 or dens is None:
            return
        h, w = dens.shape
        c, s = np.cos(rot) * scale, np.sin(rot) * scale
        A = np.array([[c, -s, cx - (c * w / 2 - s * h / 2)], [s, c, cy - (s * w / 2 + c * h / 2)]], np.float32)
        M = cam.matrix(depth)
        Mt = np.concatenate([M[:, :2] @ A[:, :2], (M[:, :2] @ A[:, 2] + M[:, 2])[:, None]], 1)
        fr.ink(warp(dens, Mt), color, opacity)

    def project(self, cam, arr, x0=0, y0=0, depth=1.0, ref=None, border=0.0):
        return warp(arr, shift_matrix(cam.matrix(depth, ref), x0, y0), border=border)

    def render(self, t):
        raise NotImplementedError

    # --- transitions: the incoming scene decides how it replaces the outgoing one
    enter_origin = (540, 900)

    def enter(self, prev_img, cur_img, f, prev_scene):
        """default: the new page blooms open from one point, a single organic ink front with a thin wet edge"""
        return ink_bloom(prev_img, cur_img, f, self.enter_origin, seed=self.paper_seed)


_BLOOM = {}


def ink_bloom(prev_img, cur_img, f, origin=(540, 900), seed=0, edge=0.3):
    from engine import ease_io, wash_map, wipe_mix
    key = (origin, seed)
    if key not in _BLOOM:
        _BLOOM[key] = wash_map(H, W, origin, noise_amp=0.22, noise_scale=260, seed=seed)
    return wipe_mix(prev_img, cur_img, _BLOOM[key], ease_io(f) * 1.12 - 0.04, soft=0.055, edge=edge)


def dof(img, amount, focus_y=0.5, band=0.22):
    """shallow depth of field: soft blur above and below a focus band (macro look)"""
    if amount <= 0.01:
        return img
    yy = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    m = np.clip((np.abs(yy - focus_y) - band) / 0.25, 0, 1) * amount
    b = blur(img, 7 * amount + 1)
    return img * (1 - m) + b * m


def museum_card(fr, lines, cx, cy, op=1.0, pad=(46, 26), rule=None):
    """a small paper label card (like a museum caption) on screen: lines = [(alpha patch, colour), ...] stacked,
    centred at (cx, cy); a crisp edge, a soft shadow and an optional thin vermilion rule"""
    from engine import PAPER, SEPIA, VERM
    if op <= 0:
        return
    gap = 10
    w = max(a.shape[1] for a, _ in lines) + 2 * pad[0]
    h = sum(a.shape[0] for a, _ in lines) + gap * (len(lines) - 1) + 2 * pad[1]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    box = np.zeros((H, W), np.float32)
    box[y0:y0 + h, x0:x0 + w] = 1
    sh = blur(np.roll(np.roll(box, 8, 0), 6, 1), 9)
    fr.img *= (1 - 0.28 * op * sh)[..., None]
    fr.over(np.asarray(PAPER, np.float32) * 1.02, blur(box, 0.8), op)
    inner = np.zeros_like(box)
    inner[y0 + 7:y0 + h - 7, x0 + 7:x0 + w - 7] = 1
    inner2 = np.zeros_like(box)
    inner2[y0 + 9:y0 + h - 9, x0 + 9:x0 + w - 9] = 1
    ring = np.clip(blur(inner, 0.6) - blur(inner2, 0.6), 0, 1)                 # a thin printed border
    fr.ink(ring, SEPIA, 0.55 * op)
    y = y0 + pad[1]
    for a, col in lines:
        hh, ww = a.shape
        D = np.zeros((H, W), np.float32)
        D[y:y + hh, int(cx - ww / 2):int(cx - ww / 2) + ww] = a
        fr.ink(D, col, op)
        y += hh + gap
    if rule is not None:
        fr.ink(rule, VERM, op)

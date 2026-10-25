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

    def project(self, cam, arr, x0=0, y0=0, depth=1.0, ref=None, border=0.0):
        return warp(arr, shift_matrix(cam.matrix(depth, ref), x0, y0), border=border)

    def render(self, t):
        raise NotImplementedError

    # --- transitions: the incoming scene decides how it replaces the outgoing one
    def enter(self, prev_img, cur_img, f, prev_scene):
        """default: a soft ink dissolve through the paper"""
        from engine import noise, wipe_mix
        T = noise(H, W, 90, 4, 77)
        T = (T - T.min()) / (T.max() - T.min())
        return wipe_mix(prev_img, cur_img, T, f * 1.15 - 0.05, soft=0.08, edge=0.25)


def dof(img, amount, focus_y=0.5, band=0.22):
    """shallow depth of field: soft blur above and below a focus band (macro look)"""
    if amount <= 0.01:
        return img
    yy = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    m = np.clip((np.abs(yy - focus_y) - band) / 0.25, 0, 1) * amount
    b = blur(img, 7 * amount + 1)
    return img * (1 - m) + b * m

"""head — a reader's head in profile, facing right, painted like the reel's figures: one long outline stroke from the
nape over the crown and down the face, a pale wash, a dark knot of hair, an eye and an ear.  The brain sits where
the scene can make it glow (BRAIN, in sprite coordinates)."""
import numpy as np

from engine import StrokeSet, spline, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash

HW, HH = 440, 470
BRAIN = (200, 175)
OUTLINE = [(128, 452), (118, 380), (104, 320), (82, 252), (86, 170), (120, 104), (176, 58), (246, 44), (306, 66), (338, 112),
           (348, 160), (350, 184), (378, 236), (380, 246), (352, 254), (358, 270), (348, 280), (356, 294), (344, 322), (318, 340),
           (290, 344), (282, 372), (290, 452)]


def build():
    ss = StrokeSet(HW, HH)
    ss.add(pwash(HW, HH, OUTLINE + [(210, 456)], ink=0.12, soft=2.5, seed=1, grain=0.45), 0, 1)
    hair = [(84, 250), (82, 168), (118, 102), (176, 56), (246, 42), (306, 64), (330, 96), (292, 92), (238, 96), (196, 124),
            (170, 170), (150, 236), (118, 300)]
    ss.add(pwash(HW, HH, hair, ink=0.55, soft=2.0, seed=2, grain=0.35), 0, 1)
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ss.add(pwash(HW, HH, [(92 + 34 * np.cos(t), 120 + 30 * np.sin(t)) for t in a], ink=0.85, soft=1.5, seed=3, grain=0.3), 0, 1)
    ss.add(stroke(HW, HH, spline(OUTLINE[:12], 10), 9, ink=0.9, dry=0.35, profile=PRESS_DOWN, seed=4), 0, 1)
    ss.add(stroke(HW, HH, spline(OUTLINE[10:], 10), 7, ink=0.9, dry=0.3, profile=((0, 0.7), (0.2, 1), (0.8, 0.9), (1, 0.4)), seed=5), 0, 1)
    ss.add(stroke(HW, HH, [(292, 168), (308, 164), (320, 170)], 5, ink=0.9, dry=0.2, profile=BLUNT, seed=6), 0, 1)        # eye
    ss.add(stroke(HW, HH, [(296, 146), (316, 142), (332, 150)], 4, ink=0.7, dry=0.4, profile=SWEEP, seed=7), 0, 1)        # brow
    ss.add(stroke(HW, HH, spline([(190, 196), (178, 206), (178, 232), (192, 244)], 8), 5, ink=0.75, dry=0.3, profile=PRESS_DOWN, seed=8), 0, 1)
    return ss.render(9).astype(np.float32)

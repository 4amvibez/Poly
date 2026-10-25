"""Ink wash (水墨): 云深不知处 — a temple pavilion on a misty spur below twin peaks, a footbridge over
the gorge, a pine on the foreground rock, a red sun and a travelling monk. Pure code, no image model.
python3 ink_cloud_temple.py [out.png] [--stages DIR]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.claude', 'skills', 'claude-drawing', 'lib'))
import numpy as np
from inkpaint import Painting, fbm1d
from core import spline

out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'ink_cloud_temple.png'
stages = sys.argv[sys.argv.index('--stages') + 1] if '--stages' in sys.argv else None

p = Painting(1920, 1080, seed=7071)
p.paper()
p.stage('宣纸')

# ---- far ranges, pale, sinking into mist
far = p.ridge(600, peaks=[(300, 130, 280), (1780, 150, 300)], rough=24, rough_scale=220, seed=3)
p.wash(far, dens=0.14, decay=160, tex=140, edge=0.25, mist_y=720, mist_h=120, seed=4)
p.stage('远山')

# ---- twin main peaks: the taller one right of centre, a lower shoulder peak to its left
SX, SY = 1180, 230
main = p.peak(SX, SY, height=600, left_w=330, right_w=480, shoulders=[(SX + 260, 80, 100), (SX - 260, 60, 90)], seed=5)
p.wash(main, dens=0.44, decay=230, tex=90, edge=0.45, mist_y=900, mist_h=90, seed=6)
p.cun(main, SX - 560, SX + 620, n=150, shadow_x=SX + 40, max_y=900)
p.moss(main, SX - 540, SX + 600, n=60)
p.ridge_line(main, SX - 620, SX + 660)
TX, TY = 760, 430
second = p.peak(TX, TY, height=420, left_w=300, right_w=260, shoulders=[(TX - 200, 50, 80)], seed=8)
p.wash(second, dens=0.36, decay=200, tex=100, edge=0.4, mist_y=880, mist_h=90, seed=9)
p.cun(second, TX - 380, TX + 300, n=90, shadow_x=TX - 30, max_y=880)
p.moss(second, TX - 360, TX + 280, n=40)
p.ridge_line(second, TX - 420, TX + 320)
p.stage('主峰')

# ---- pilgrim paths climbing out of the mist to the summit temple, pavilions on both peaks
paths = p.trails(main, (SX, SY), n=12, x_range=(SX - 120, SX + 420), start_y=(820, 860), red_index=6, seed=12)
p.mist(860, 90, 0.9)
p.mist(640, 50, 0.5)
for (i, t) in [(3, 0.4), (6, 0.55), (9, 0.35)]:
    q = paths[i][int(t * (len(paths[i]) - 1))]
    p.traveller(q[0], q[1])
p.pagoda(TX, int(second[TX]) + 6)
p.pagoda(SX, int(main[SX]) + 6)
p.stage('山径')

# ---- foreground rock and pine, bottom-left
rock = 820 + 260 * ((p.xs / 560) ** 1.5) + 28 * fbm1d(1920, 70, 4, 17)
rock[p.xs > 560] = 2000
p.rock(rock)
p.pine((240, 1010), (330, 700), bend=60,
       branches=[((290, 830), (470, 780), -30), ((306, 770), (190, 735), 26), ((322, 725), (450, 680), -20)],
       extra_tufts=[(335, 690), (390, 706)])
p.stage('松石')

# ---- red sun, birds, title and seal in the open sky at top left
p.sun(1560, 170, 40)
p.birds([(1380, 280, 0.9), (1420, 300, 0.7), (1350, 312, 0.6)])
p.title_vertical('云深不知处', 150, 110, 80)
p.seal('山居', 258, 150, 40)
p.save(out, stages_dir=stages)
print('saved', out)

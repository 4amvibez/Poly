"""poly_ink — Chinese ink wash (水墨) paintings drawn in code, built on the claude-drawing skill.

    from poly_ink import cloud_temple, ink_landscape, make_gif

    img = cloud_temple()                                  # 云深不知处, a PIL Image (1920x1080)
    img = ink_landscape("山水", seed=3, width=1280, height=720)
    stages = cloud_temple(with_stages=True)[1]            # [(name, Image), ...] for a draw-on animation
    make_gif(stages, "out.gif")

    s = water_swordsman()                                 # 水: brush calligraphy + a dry-brush wave (an InkScene)
    s.render().save("water.png")                          # finished, or s.render(at=2500) mid-painting
    animate_scene(s, "water.gif")                         # written stroke by stroke

No image model is involved: every wash and brush stroke is computed with numpy + Pillow, and a fixed
seed always gives the same picture.
"""
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_LIB = os.path.join(_REPO, '.claude', 'skills', 'claude-drawing', 'lib')

if not os.path.isfile(os.path.join(SKILL_LIB, 'inkpaint.py')):
    raise ImportError(f'claude-drawing skill not found at {SKILL_LIB}; run: git submodule update --init')
if SKILL_LIB not in sys.path:
    sys.path.insert(0, SKILL_LIB)

# titles and seals need a CJK font; the skill looks for Kaiti / Noto CJK, so also offer common Linux fonts
_FONT_GLOBS = ['/usr/share/fonts/**/wqy-zenhei.ttc', '/usr/share/fonts/**/wqy-microhei.ttc',
               '/usr/share/fonts/**/NotoSerifCJK*.tt[cf]', '/usr/share/fonts/**/NotoSansCJK*.tt[cf]']


def find_cjk_font():
    """path of the font used for titles and seals: $INKPAINT_FONT if set, else the first known CJK font found"""
    env = os.environ.get('INKPAINT_FONT')
    if env and os.path.exists(env):
        return env
    for pat in _FONT_GLOBS:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


if not os.environ.get('INKPAINT_FONT') and find_cjk_font():
    os.environ['INKPAINT_FONT'] = find_cjk_font()

from .scenes import cloud_temple, ink_landscape                                     # noqa: E402
from .animate import animate_scene, make_gif                                          # noqa: E402
from .brush import InkScene, brush_stroke, splatter, sweep, wash                      # noqa: E402
from .calligraphy import water_swordsman                                              # noqa: E402

__all__ = ['cloud_temple', 'ink_landscape', 'water_swordsman', 'make_gif', 'animate_scene', 'InkScene',
           'brush_stroke', 'sweep', 'splatter', 'wash', 'find_cjk_font', 'SKILL_LIB']

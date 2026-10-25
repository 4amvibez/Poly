# Poly

Chinese ink wash (水墨) paintings drawn entirely in code — no image model. Every wash, brush stroke and
mist band is computed with numpy + Pillow, using the [claude-drawing](https://github.com/cloveric/claude-drawing-skill)
skill (a git submodule in `.claude/skills/claude-drawing`). The same seed always gives the same picture.

![云深不知处](drawing_output/gifs/ink_cloud_temple.gif)

## Setup

```bash
git clone --recurse-submodules https://github.com/4amvibez/Poly.git
cd Poly
pip install -r requirements.txt
# already cloned without the submodule?  git submodule update --init
```

Titles and seals need a Chinese font. `poly_ink` finds Kaiti/Songti on macOS, KaiTi/SimSun on Windows, and
WenQuanYi or Noto CJK on Linux (`sudo apt install fonts-wqy-zenhei`). To use another font, set
`INKPAINT_FONT=/path/to/font.ttf`. Without a font the painting still renders, just without the title and seal.

## Use it from Python

Run from the repo root, or put the repo root on `PYTHONPATH` / `sys.path`.

```python
from poly_ink import cloud_temple, ink_landscape, make_gif

# the finished painting 云深不知处 (1920x1080 PIL Image)
cloud_temple().save("cloud_temple.png")

# your own landscape: title, seal, seed (changes the brushwork) and size are all configurable
img = ink_landscape("江山如画", seal="墨韵", seed=9, width=1280, height=720)
img.save("landscape.jpg", quality=92)

# draw-on animation: get the stage snapshots, then write a looping GIF
img, stages = ink_landscape("山水", with_stages=True)   # stages = [(name, Image), ...]
make_gif(stages, "landscape.gif")
```

Each call returns a Pillow `Image`, so you can also resize it, paste it into another image, or return it from a
web handler. A 1920x1080 painting takes about 4 seconds. In a web app, render it once and cache the result.

## Use it from the command line

```bash
python -m poly_ink cloud-temple out.png --gif out.gif
python -m poly_ink landscape out.jpg --title 江山如画 --seal 墨韵 --seed 9 --size 1280x720 --stages stages/ --gif out.gif
python -m poly_ink --help
```

## Layout

| Path | What |
|---|---|
| `poly_ink/scenes.py` | `cloud_temple()` and `ink_landscape()` — the scenes, written with the skill's `Painting` API |
| `poly_ink/animate.py` | `make_gif()` — stages → GIF (720x405, 900 ms per stage, 5-frame crossfades, 2.6 s final hold) |
| `poly_ink/__main__.py` | command line |
| `drawing_output/` | the rendered 云深不知处: PNG/JPG, stage snapshots, GIF, and the original scripts |
| `tests/` | `python -m pytest tests` |

To draw something new, copy `cloud_temple()` in `poly_ink/scenes.py` and change it. The available drawing calls
(`peak`, `wash`, `cun`, `pine`, `rock`, `pagoda`, `traveller`, `birds`, `sun`, `mist`, `trails`, `stroke`, …) are
listed in the "水墨 `Painting`" section of `.claude/skills/claude-drawing/SKILL.md`.

#!/usr/bin/env python3
"""Render the ないで vs ず Instagram Reel (1080x1920, 30fps)."""
import math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "reel.mp4")
PREVIEW = os.environ.get("PREVIEW")  # comma list of output seconds -> PNGs only

FPS, W, H = 30, 1080, 1920
SW, SH = 720, 1280
SR = 48000
SPF = SR // FPS  # audio samples per frame

FB = os.path.join(HERE, "fonts/MPLUSRounded1c-Black.ttf")
FX = os.path.join(HERE, "fonts/MPLUSRounded1c-ExtraBold.ttf")
FBo = os.path.join(HERE, "fonts/MPLUSRounded1c-Bold.ttf")
_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

# palette
NAVY = (19, 32, 58)
WHITE = (255, 255, 255)
GOLD = (255, 210, 63)
CORAL = (255, 107, 107)
STRIKE = (140, 150, 170)
LAV = (165, 185, 255)
MINT = (80, 220, 170)

# ---------------------------------------------------------------- audio in
def load_audio():
    tmp = os.path.join(HERE, "src48.wav")
    if not os.path.exists(tmp):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-ac", "2", "-ar", str(SR), tmp], check=True)
    w = wave.open(tmp)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return a.reshape(-1, 2)
AUD = load_audio()
_mono = np.abs(AUD.mean(1))
def rms_at(t):
    i = int(t * SR); h = SR // 100
    seg = _mono[max(0, i - h): i + h]
    return float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0
def snap(t, win=0.06):
    cands = np.arange(t - win, t + win + 1e-9, 0.01)
    return float(min(cands, key=rms_at))

# ---------------------------------------------------------------- edit list
# (src_start, src_end, zoom) ; ("freeze", src_t, dur, zoom)
EDL = [
    (0.08, 2.30, "hook"),
    (4.10, 6.00, 1.06),
    (6.70, 9.06, 1.14),
    (9.38, 10.80, 1.06),
    (11.74, 14.68, 1.14),
    (14.68, 16.60, 1.06),
    (17.00, 20.78, 1.14),
    (20.86, 22.20, 1.26),
    (24.50, 28.14, 1.06),
    (30.82, 32.12, 1.14),
    (32.26, 33.98, 1.06),
    (35.30, 38.00, 1.14),
    (38.00, 41.02, 1.06),
    (41.02, 44.10, 1.14),
    (44.10, 47.28, 1.06),
    (47.28, 49.98, 1.16),
    (50.26, 51.94, 1.06),
    ("freeze", 51.94, 1.6, 1.10),
    (52.28, 54.20, 1.22),
    (57.38, 62.28, 1.06),
    (62.28, 63.58, 1.16),
    (63.68, 67.88, 1.06),
    (71.50, 75.30, 1.12),
]
SEGS = []  # dicts with out frames
of = 0
for e in EDL:
    if e[0] == "freeze":
        n = round(e[2] * FPS)
        SEGS.append(dict(kind="freeze", s=e[1], e=e[1], f0=of, n=n, zoom=e[3], sf=round(e[1] * FPS)))
    else:
        s, en = snap(e[0]), snap(e[1])
        sf, ef = round(s * FPS), round(en * FPS)
        n = ef - sf
        SEGS.append(dict(kind="clip", s=sf / FPS, e=ef / FPS, f0=of, n=n, zoom=e[2], sf=sf))
    of += n
NFRAMES = of
DUR = NFRAMES / FPS
FREEZE = next(g for g in SEGS if g["kind"] == "freeze")
FZ = FREEZE["f0"] / FPS

def o(t):
    """source time -> output time (string 'F+x' = freeze start + x)."""
    if isinstance(t, str):
        return FZ + float(t[2:])
    for g in SEGS:
        if g["kind"] == "clip" and g["s"] - 1e-6 <= t < g["e"] - 1e-6:
            return (g["f0"] + (t - g["s"]) * FPS) / FPS
    for g in SEGS:  # in a gap -> next clip start
        if g["kind"] == "clip" and g["s"] >= t:
            return g["f0"] / FPS
    return DUR

# ---------------------------------------------------------------- content
# markup: {gold}  <coral>  [strike]
# kind "jp": JP big / romaji / EN   kind "en": EN big / JP small
CAPS = [
    dict(t=(0.08, 2.30), kind="en", en="How do you say {“without doing”}\nsomething in Japanese?", jp="「〜しないで」って日本語でどう言う？"),
    dict(t=(4.10, 6.00), kind="en", en="“I went to school\n{without having breakfast}.”", jp="朝ごはんを<食べないで>、学校へ行った。"),
    dict(t=(6.70, 9.06), kind="en", en="You can use the {ない form} here", jp="ここでは「<ない形>」を使います"),
    dict(t=(9.38, 10.80), kind="jp", stages=[(9.38, "<ない>形 ＋ <で>")], ro="nai-form + de", en="the ない form + で"),
    dict(t=(11.74, 14.68), kind="jp", stages=[(11.74, "朝ごはんを<食べないで>、\n学校へ行った。")],
         ro="asagohan o <tabenaide>, gakkō e itta.", en="I went to school {without eating breakfast}."),
    dict(t=(14.68, 16.60), kind="jp", stages=[(14.68, "…学校へ<行きました>。")],
         ro="…gakkō e <ikimashita>.", en="(polite) …I went to school."),
    dict(t=(17.00, 20.78), kind="en", en="But there's actually\n{another way} to say it…", jp="でも実は、{もう一つ}の言い方があります…"),
    dict(t=(20.86, 22.20), kind="jp", stages=[(20.86, "{ず}形")], ro="zu-form", en="The {ず form} of the verb!"),
    dict(t=(24.50, 28.14), kind="en", en="The only difference:\nit's more {formal}", jp="ず形は{フォーマル}な場面で使われます"),
    dict(t=(30.82, 32.12), kind="en", en="Take the {ない form}…", jp="まず「<ない形>」にして…"),
    dict(t=(32.26, 33.98), kind="jp", stages=[(32.26, "行か<ない>"), (32.62, "行か[ない]"), (33.28, "行か[ない]{ず}")],
         ro="remove ない → add ず", en="Remove {ない}, add {ず}"),
    dict(t=(35.30, 38.00), kind="jp", stages=[(35.36, "行きます"), (35.76, "行きます → 行か<ない>"), (36.80, "行きます → 行か<ない> → 行か{ず}")],
         ro="ikimasu → ika<nai> → ika{zu}", en="行く  to go  ·  Group 1"),
    dict(t=(38.00, 41.02), kind="jp", stages=[(38.78, "食べます"), (39.36, "食べます → 食べ<ない>"), (39.92, "食べます → 食べ<ない> → 食べ{ず}")],
         ro="tabemasu → tabe<nai> → tabe{zu}", en="食べる  to eat  ·  Group 2"),
    dict(t=(41.02, 44.10), kind="jp", stages=[(41.04, "話します"), (42.04, "話します → 話さ<ない>"), (43.22, "話します → 話さ<ない> → 話さ{ず}")],
         ro="hanashimasu → hanasa<nai> → hanasa{zu}", en="話す  to speak  ·  Group 1"),
    dict(t=(44.10, 45.30), kind="en", en="Now the {3rd group}", jp="次は{グループ3}"),
    dict(t=(45.30, 47.28), kind="jp", stages=[(45.34, "来ます"), (45.88, "来ます → 来<ない>"), (46.68, "来ます → 来<ない> → 来{ず}")],
         ro="kimasu → ko<nai> → ko{zu}", en="来る  to come  ·  Group 3"),
    dict(t=(47.28, 49.98), kind="en", en="The ONLY exception:\n{します}", jp="唯一の例外は「{します}」！"),
    dict(t=(50.26, 52.28), kind="jp", stages=[(50.36, "します"), (50.88, "します → し<ない>"), ("F+0.0", "します → し<ない> → ？")],
         ro="shimasu → shi<nai> → ?", en="する  to do"),
    dict(t=(52.28, 54.20), kind="jp", stages=[(52.28, "します → し<ない> → {せず}")],
         ro="shimasu → shi<nai> → {sezu}", en="{せず}  —  NOT [しず]!"),
    dict(t=(57.38, 62.28), kind="jp", stages=[(57.38, "妹は何も<言わないで>、\n部屋を出ていった。")],
         ro="imōto wa nani mo <iwanaide>, heya o dete itta.", en="My little sister left the room\n{without saying anything}."),
    dict(t=(62.28, 63.58), kind="en", en="Now in the {ず form}…", jp="{ず形}にすると…"),
    dict(t=(63.68, 67.88), kind="jp", stages=[(63.68, "妹は何も{言わず}、\n部屋を出ていった。")],
         ro="imōto wa nani mo {iwazu}, heya o dete itta.", en="My little sister left the room\n{without saying anything}."),
    dict(t=(71.50, 73.40), kind="jp", stages=[(71.50, "見てくれて\nありがとうございます！")],
         ro="mite kurete arigatō gozaimasu!", en="Thank you for watching!"),
    dict(t=(73.40, 75.30), kind="jp", stages=[(73.40, "またね！")], ro="mata ne!", en="See you next time!"),
]

# top chips: (src t0, src t1, [(text, bg, fg)])
CHIPS = [
    (9.38, 16.60, [("WAY 1 :  ないで", CORAL, WHITE)]),
    (20.86, 22.20, [("WAY 2 :  ず", GOLD, NAVY)]),
    (24.50, 28.14, [("ないで = casual", CORAL, WHITE), ("ず = formal", GOLD, NAVY)]),
    (30.82, 47.28, [("RULE :  ない → ず", GOLD, NAVY)]),
    (47.28, 52.28, [("EXCEPTION !", CORAL, WHITE)]),
    (52.28, 54.20, [("する → せず", GOLD, NAVY)]),
    (57.38, 62.28, [("ないで  (casual)", CORAL, WHITE)]),
    (62.28, 67.88, [("ず  (formal)", GOLD, NAVY)]),
    (71.50, 75.30, [("SAVE  +  FOLLOW", GOLD, NAVY), ("for daily Japanese", WHITE, NAVY)]),
]

# ---------------------------------------------------------------- text render
def parse(markup):
    """-> list of lines, each list of (text, style)."""
    lines = []
    for raw in markup.split("\n"):
        spans, cur, style = [], "", "n"
        closers = {"{": ("}", "hi"), "<": (">", "co"), "[": ("]", "st")}
        i = 0
        while i < len(raw):
            c = raw[i]
            if style == "n" and c in closers:
                if cur: spans.append((cur, "n")); cur = ""
                style = closers[c][1]; end = closers[c][0]
            elif style != "n" and c == end:
                spans.append((cur, style)); cur = ""; style = "n"
            else:
                cur += c
            i += 1
        if cur: spans.append((cur, style))
        lines.append(spans)
    return lines

def line_w(spans, f):
    return sum(f.getlength(t) for t, _ in spans)

def fit(lines, path, size, maxw):
    while size > 20:
        f = font(path, size)
        if max(line_w(l, f) for l in lines) <= maxw:
            return f
        size -= 2
    return font(path, size)

def draw_lines(d, lines, f, cx, y, base, gap=1.18, stroke=0, stroke_fill=(0, 0, 0)):
    asc, desc = f.getmetrics()
    lh = int((asc + desc) * gap)
    for spans in lines:
        x = cx - line_w(spans, f) / 2
        for t, st in spans:
            col = {"n": base, "hi": GOLD, "co": CORAL, "st": STRIKE}[st]
            d.text((x, y), t, font=f, fill=col, stroke_width=stroke, stroke_fill=stroke_fill)
            wdt = f.getlength(t)
            if st == "st":
                my = y + asc * 0.62
                d.line([(x - 4, my), (x + wdt + 4, my)], fill=(255, 70, 70), width=max(4, f.size // 11))
            x += wdt
        y += lh
    return y

def block_height(lines, f, gap=1.18):
    asc, desc = f.getmetrics()
    return int((asc + desc) * gap) * len(lines)

_bc = {}
def caption_block(cap, stage_txt):
    """Rendered RGBA (transparent bg) for the panel text."""
    k = (id(cap), stage_txt)
    if k in _bc: return _bc[k]
    maxw = 1000
    if cap["kind"] == "en":
        rows = [(parse(cap["en"]), fit(parse(cap["en"]), FB, 62, maxw), WHITE),
                (parse(cap["jp"]), fit(parse(cap["jp"]), FX, 44, maxw), (255, 228, 190))]
    else:
        big = 104 if "\n" not in stage_txt and len(stage_txt) <= 4 else 70
        rows = [(parse(stage_txt), fit(parse(stage_txt), FB, big, maxw), WHITE),
                (parse(cap["ro"]), fit(parse(cap["ro"]), FX, 40, maxw), LAV),
                (parse(cap["en"]), fit(parse(cap["en"]), FX, 46, maxw), WHITE)]
    gaps = 14
    hh = sum(block_height(l, f) for l, f, _ in rows) + gaps * (len(rows) - 1) + 20
    im = Image.new("RGBA", (W, hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = 6
    for l, f, col in rows:
        y = draw_lines(d, l, f, W / 2, y, col) + gaps
    _bc[k] = im
    return im

def pill(text, bg, fg, size=46):
    f = font(FB, size)
    tw = f.getlength(text)
    asc, desc = f.getmetrics()
    pw, ph = int(tw + 64), int(asc + desc + 22)
    im = Image.new("RGBA", (pw + 16, ph + 16), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([8, 12, pw + 8, ph + 12], ph // 2, fill=(0, 0, 0, 110))
    sh = sh.filter(ImageFilter.GaussianBlur(5))
    im.alpha_composite(sh)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 8, pw + 8, ph + 8], ph // 2, fill=bg + (255,))
    d.text((8 + 32, 8 + 11), text, font=f, fill=fg)
    return im

_chip_cache = {}
def chip_row(items):
    k = tuple(items)
    if k in _chip_cache: return _chip_cache[k]
    ims = [pill(*it) for it in items]
    wtot = sum(i.width for i in ims) + 10 * (len(ims) - 1)
    row = Image.new("RGBA", (wtot, max(i.height for i in ims)), (0, 0, 0, 0))
    x = 0
    for i in ims:
        row.alpha_composite(i, (x, 0)); x += i.width + 10
    _chip_cache[k] = row
    return row

def outlined(markup, size, path=FB, maxw=1000, stroke=10, base=WHITE):
    lines = parse(markup)
    f = fit(lines, path, size, maxw)
    hh = block_height(lines, f, 1.12) + 40
    im = Image.new("RGBA", (W, hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    draw_lines(d, lines, f, W / 2, 14, base, gap=1.12, stroke=stroke, stroke_fill=(15, 20, 35))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 140), (0, 0), im.split()[3])
    sh = sh.filter(ImageFilter.GaussianBlur(8))
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (0, 8)); out.alpha_composite(im)
    return out

HOOK1 = outlined("「<ないで>」is fine…", 84)
HOOK2 = outlined("but「{ず}」sounds\nWAY more formal", 92)
GUESS = outlined("GUESS!", 120, base=GOLD)
DIGITS = {n: outlined(str(n), 260) for n in (3, 2, 1)}

# ---------------------------------------------------------------- helpers
def ease_out_back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))

def paste_scaled(frame, im, cx, cy, scale=1.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.01: return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        a = im.split()[3].point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

BAND_TOP, BAND_BOT, BAND_C = 872, 1138, 1007  # burned-caption zone in source px
BASE = W / SW

# timeline objects in output seconds
cap_tl = []
for c in CAPS:
    t0, t1 = o(c["t"][0]), o(c["t"][1])
    # end in gap -> end of previous clip
    for g in SEGS:
        if g["kind"] == "clip" and abs(g["e"] - c["t"][1]) < 0.05:
            t1 = (g["f0"] + g["n"]) / FPS
    stages = [(o(s), txt) for s, txt in c.get("stages", [])] or [(t0, None)]
    cap_tl.append(dict(c=c, t0=t0, t1=t1, stages=stages))
chip_tl = []
for a, b, items in CHIPS:
    t1 = o(b)
    for g in SEGS:
        if g["kind"] == "clip" and abs(g["e"] - b) < 0.05:
            t1 = (g["f0"] + g["n"]) / FPS
    chip_tl.append((o(a), t1, items))
SEZU_T = o(52.28)

# ---------------------------------------------------------------- SFX
rng = np.random.default_rng(7)
def env(n, a=0.005, r=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / r)
def sfx_whoosh(d=0.38):
    n = int(d * SR); x = rng.standard_normal(n)
    y = np.zeros(n); k = 0.0
    cut = np.linspace(0.02, 0.25, n) * np.sin(np.linspace(0, math.pi, n)) + 0.01
    for i in range(n):
        k += cut[i] * (x[i] - k); y[i] = k
    return y * np.sin(np.linspace(0, math.pi, n)) ** 2 * 1.6
def sfx_pop():
    n = int(0.09 * SR); t = np.arange(n) / SR
    f = np.linspace(500, 1100, n)
    return np.sin(2 * math.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.03) * 0.8
def sfx_ding():
    n = int(1.0 * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * math.pi * f * t) for f, a in ((1568, 1), (2352, .45), (3136, .25), (4704, .1)))
    return y * env(n, 0.002, 0.28) * 0.45
def sfx_tick():
    n = int(0.05 * SR); t = np.arange(n) / SR
    return np.sin(2 * math.pi * 1900 * t) * env(n, 0.001, 0.012) * 0.9
def sfx_impact():
    n = int(0.7 * SR); t = np.arange(n) / SR
    f = np.linspace(120, 38, n)
    y = np.sin(2 * math.pi * np.cumsum(f) / SR) * env(n, 0.003, 0.22)
    return y * 0.9 + rng.standard_normal(n) * env(n, 0.001, 0.03) * 0.25
def sfx_buzz():
    n = int(0.22 * SR); t = np.arange(n) / SR
    return np.sign(np.sin(2 * math.pi * 150 * t)) * env(n, 0.003, 0.08) * 0.18

# ---------------------------------------------------------------- audio build
def build_audio():
    out = np.zeros((NFRAMES * SPF + SR, 2), np.float32)
    fade = int(0.008 * SR)
    for g in SEGS:
        if g["kind"] != "clip": continue
        a0 = g["sf"] * SPF; n = g["n"] * SPF
        chunk = AUD[a0:a0 + n].copy()
        r = np.ones(len(chunk)); r[:fade] = np.linspace(0, 1, fade); r[-fade:] = np.linspace(1, 0, fade)
        chunk *= r[:, None]
        out[g["f0"] * SPF: g["f0"] * SPF + len(chunk)] += chunk
    peak = np.abs(out).max(); out *= 0.89 / peak
    def add(t, s, gain):
        i = int(t * SR); s = s[: max(0, len(out) - i)]
        out[i:i + len(s)] += (s * gain)[:, None]
    add(0.0, sfx_impact(), 0.55)
    add(0.55, sfx_whoosh(0.3), 0.30)
    for t0, t1, items in chip_tl:
        add(max(0, t0 - 0.05), sfx_whoosh(), 0.22)
    for c in cap_tl:
        for i, (st, txt) in enumerate(c["stages"]):
            if txt and i > 0: add(st, sfx_pop(), 0.22)
    add(o(20.86), sfx_ding(), 0.45)
    add(o(33.28), sfx_ding(), 0.30)
    for k in range(3): add(FZ + 0.1 + k * 0.5, sfx_tick(), 0.45)
    add(SEZU_T, sfx_ding(), 0.5)
    add(o(53.0), sfx_buzz(), 0.6)
    add(o(63.68), sfx_ding(), 0.3)
    out = out[: NFRAMES * SPF]
    path = os.path.join(HERE, "reel_audio.wav")
    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes()); w.close()
    return path

# ---------------------------------------------------------------- video
class Reader:
    def __init__(self):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-vf",
                                   "eq=contrast=1.05:saturation=1.12:brightness=0.01,unsharp=5:5:0.5",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.idx = -1; self.img = None
    def get(self, fi):
        while self.idx < fi:
            b = self.p.stdout.read(SW * SH * 3)
            if len(b) < SW * SH * 3: break
            self.idx += 1
            if self.idx >= fi - 0:
                self.img = Image.frombuffer("RGB", (SW, SH), b)
        return self.img

def seg_at(fi):
    for g in SEGS:
        if g["f0"] <= fi < g["f0"] + g["n"]: return g
    return SEGS[-1]

def zoom_for(g, k):
    t = k / FPS
    if g["zoom"] == "hook":
        z = 1.16 + 0.22 * (1 - clamp(t / 0.45)) ** 3
    else:
        z = g["zoom"] + 0.012 * t  # slow push
    return z

def render_frame(fi, src):
    g = seg_at(fi); k = fi - g["f0"]; t = fi / FPS
    z = zoom_for(g, k)
    s = BASE * z
    # anchor: source (360, BAND_C) -> output (540, 1511)
    ax, ay = 360, BAND_C; ox, oy = W / 2, BASE * BAND_C
    x0 = ax - ox / s; y0 = ay - oy / s
    frame = src.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + W / s, y0 + H / s)).convert("RGBA")
    if g["kind"] == "freeze":
        frame = Image.blend(frame, Image.new("RGBA", (W, H), (10, 14, 30, 255)), 0.45)
    d = ImageDraw.Draw(frame)

    # subtitle panel covering burned captions
    py0 = oy + s * (BAND_TOP - ay); py1 = oy + s * (BAND_BOT - ay)
    area_bot = min(py1, 1660)
    need = 0
    for c in cap_tl:
        if c["t0"] <= t < c["t1"]:
            need = max(caption_block(c["c"], st[1]).height for st in c["stages"]) if c["stages"][0][1] else caption_block(c["c"], None).height
    py0 = min(py0, area_bot - need - 30)
    d.rectangle([0, py0, W, py1], fill=NAVY + (255,))
    d.rectangle([0, py0, W, py0 + 6], fill=GOLD + (255,))

    # captions
    for c in cap_tl:
        if c["t0"] <= t < c["t1"]:
            st_t, txt = c["stages"][0]
            for st, tx in c["stages"]:
                if t >= st: st_t, txt = st, tx
            im = caption_block(c["c"], txt)
            age = t - (st_t if txt else c["t0"])
            first = t - c["t0"]
            sc = 0.82 + 0.18 * ease_out_back(clamp(age / 0.22)) if age < 0.22 else 1.0
            al = clamp(first / 0.12)
            cy = (py0 + 12 + area_bot) / 2
            paste_scaled(frame, im, W / 2, cy, sc, al)

    # top chips
    for t0, t1, items in chip_tl:
        if t0 <= t < t1:
            row = chip_row(items)
            a = clamp((t - t0) / 0.25)
            yy = 250 - 120 * (1 - ease_out_back(a)) if a < 1 else 250
            paste_scaled(frame, row, W / 2, yy, 1.0, clamp((t - t0) / 0.12) * clamp((t1 - t) / 0.1))

    # hook
    if t < o(2.30):
        a1 = clamp(t / 0.18); a2 = clamp((t - 0.45) / 0.2)
        paste_scaled(frame, HOOK1, W / 2, 240, 0.7 + 0.3 * ease_out_back(a1) if a1 < 1 else 1, a1)
        paste_scaled(frame, HOOK2, W / 2, 470, 0.6 + 0.4 * ease_out_back(a2) if a2 < 1 else 1, a2)

    # freeze quiz
    if g["kind"] == "freeze":
        paste_scaled(frame, GUESS, W / 2, 400, 1.0 + 0.05 * math.sin(k / 3), 1)
        n = 3 - min(2, int((k / FPS - 0.1) / 0.5)) if k / FPS >= 0.1 else 3
        age = ((k / FPS - 0.1) % 0.5) if k / FPS >= 0.1 else 0
        paste_scaled(frame, DIGITS[n], W / 2, 1040, 1.4 - 0.4 * ease_out_back(clamp(age / 0.18)), 1)

    # flash on せず
    if 0 <= t - SEZU_T < 0.2:
        fl = Image.new("RGBA", (W, H), (255, 255, 255, int(170 * (1 - (t - SEZU_T) / 0.2))))
        frame.alpha_composite(fl)

    # progress bar
    d = ImageDraw.Draw(frame)
    d.rectangle([0, 0, W, 9], fill=(0, 0, 0, 90))
    d.rectangle([0, 0, int(W * (fi + 1) / NFRAMES), 9], fill=GOLD + (255,))
    return frame.convert("RGB")

def main():
    print(f"duration {DUR:.2f}s frames {NFRAMES}")
    if PREVIEW:
        want = sorted(int(float(x) * FPS) for x in PREVIEW.split(","))
        rd = Reader()
        for fi in want:
            g = seg_at(fi)
            sfi = g["sf"] + (fi - g["f0"] if g["kind"] == "clip" else 0)
            render_frame(fi, rd.get(sfi).copy()).save(os.path.join(HERE, f"prev_{fi/FPS:05.2f}.png"))
        return
    apath = build_audio()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-i", apath,
                            "-af", "loudnorm=I=-14:TP=-1.0:LRA=11",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-shortest", OUT],
                           stdin=subprocess.PIPE)
    rd = Reader()
    for fi in range(NFRAMES):
        g = seg_at(fi)
        sfi = g["sf"] + (fi - g["f0"] if g["kind"] == "clip" else 0)
        enc.stdin.write(render_frame(fi, rd.get(sfi)).tobytes())
        if fi % 150 == 0: print(f"frame {fi}/{NFRAMES}", flush=True)
    enc.stdin.close(); enc.wait()
    print("done", OUT)

if __name__ == "__main__":
    main()

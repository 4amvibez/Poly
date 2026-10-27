"""art — painted motifs shared by the scenes: the calligraphy brush, ink figures, dwellings, trees, a ship,
lanterns and blossom.  All drawn in code with poly_ink's bristle brush and claude-drawing's Painting."""
import cv2
import numpy as np

from engine import SAKURA, StrokeSet, blur, noise, spline, sstep, stroke
from poly_ink.brush import BLUNT, PRESS_DOWN, SWEEP, wash as pwash


# ------------------------------------------------------------------ the calligraphy brush
def brush_sprite(length=700, seed=0):
    """a fude (calligraphy brush) pointing straight down; returns (rgb, alpha, tip_xy).
    Bamboo handle with nodes, lacquered ferrule, a full belly of hair tapering to a wet black point."""
    r = np.random.default_rng(seed)
    w = 110
    h = length
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2
    hair_len, fer_len = 150, 46
    handle_h = h - hair_len - fer_len
    # handle: a cylinder, lit from the upper left
    hw = 17
    u = (xx - cx) / hw
    inside = np.abs(u) <= 1
    shade = np.clip(0.55 + 0.5 * np.sqrt(np.clip(1 - u ** 2, 0, 1)) - 0.25 * u, 0, 1.1)
    bamboo = np.array([0.66, 0.52, 0.33], np.float32)
    grain = 0.92 + 0.08 * noise(h, w, 3, 2, seed)[..., None]
    region = inside & (yy < handle_h)
    rgb[region] = (bamboo * shade[..., None] * grain)[region]
    a[region] = 1
    for ny in np.arange(70, handle_h - 20, 165):                          # bamboo nodes
        band = region & (np.abs(yy - ny) < 4)
        rgb[band] *= 0.62
    # ferrule: dark lacquer with a thin vermilion ring
    fer = (np.abs(xx - cx) <= hw + 2) & (yy >= handle_h) & (yy < handle_h + fer_len)
    rgb[fer] = (np.array([0.10, 0.08, 0.07]) * (0.7 + 0.6 * shade[..., None]))[fer]
    ring = fer & (np.abs(yy - handle_h - 8) < 3)
    rgb[ring] = [0.55, 0.16, 0.10]
    a[fer] = 1
    # hair: full belly then a fine point
    y0 = handle_h + fer_len
    t = np.clip((yy - y0) / hair_len, 0, 1)
    prof = (hw + 4) * np.where(t < 0.35, 1 + 0.35 * np.sin(t / 0.35 * np.pi / 2), 1.35 * (1 - (t - 0.35) / 0.65) ** 1.3)
    hair = (yy >= y0) & (np.abs(xx - cx) <= prof)
    streak = 0.85 + 0.3 * noise(h, w, 2, 2, seed + 3)
    hair_col = np.array([0.07, 0.065, 0.06]) * streak[..., None]
    upper = np.clip(1 - t / 0.3, 0, 1)[..., None]                          # dry, paler hair near the ferrule
    hair_col = hair_col * (1 - upper) + np.array([0.42, 0.36, 0.30]) * streak[..., None] * upper
    rgb[hair] = hair_col[hair]
    a[hair] = 1
    a = blur(a, 0.7)
    tip = (cx, h - 2)
    return rgb, a, tip


def draw_sprite(frame, rgb, alpha, tip, x, y, angle, scale=1.0, shadow=0.35, lift=0.0):
    """composite a sprite into a Frame with its `tip` at screen (x, y), rotated by `angle` (radians,
    0 = pointing down) and scaled.  lift > 0 raises the brush off the paper (shadow drifts, softens)."""
    from engine import H, W
    c, s = np.cos(angle), np.sin(angle)
    tx, ty = tip
    # sprite -> screen: rotate+scale about tip, then move tip to (x, y)
    M = np.array([[scale * c, -scale * s, 0], [scale * s, scale * c, 0]], np.float32)
    M[:, 2] = np.array([x, y - lift]) - M[:, :2] @ np.array([tx, ty])
    if shadow > 0:
        Ms = M.copy()
        Ms[:, 2] += (18 + lift * 0.6, 26 + lift * 0.8)
        sh = cv2.warpAffine(alpha, Ms, (W, H), flags=cv2.INTER_LINEAR)
        sh = blur(sh, 9 + lift * 0.1)
        frame.img *= (1 - shadow * sh)[..., None]
    A = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)
    C = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR)
    frame.img = frame.img * (1 - A[..., None]) + C * A[..., None]


# ------------------------------------------------------------------ ink figures and dwellings (strokes with timing)
def figure(ss, x, y, s, t0, seed, pose='stand', facing=1, dur=0.6, ink=0.8):
    """a robed figure (~100*s px tall) painted in a few economical strokes, feet at (x, y).
    pose: stand / speak (arm raised towards the listener) / sit / listen (head inclined)"""
    W, H = ss.w, ss.h
    f = facing
    B = lambda pts, w, **k: stroke(W, H, spline(pts, 8) if len(pts) > 2 else pts, w, seed=seed + len(ss.items) + 7 * len(k), **k)
    if pose == 'sit':
        hx, hy = x + f * 4 * s, y - 58 * s
        sh = (hx - f * 2 * s, hy + 12 * s)
        robe = [(sh[0] - 13 * s, sh[1]), (sh[0] + 12 * s, sh[1]), (x + f * 30 * s, y - 4 * s), (x + f * 34 * s, y),
                (x - f * 26 * s, y), (x - f * 22 * s, y - 14 * s)]
        lines = [B([(sh[0] - f * 12 * s, sh[1] + 2), (x - f * 18 * s, y - 24 * s), (x - f * 24 * s, y - 2)], 4.2 * s, ink=ink, dry=0.35, profile=PRESS_DOWN),
                 B([(x - f * 20 * s, y - 2), (x + f * 4 * s, y + 1), (x + f * 32 * s, y - 1)], 3.6 * s, ink=ink * 0.85, dry=0.45, profile=SWEEP),
                 B([(sh[0] + f * 8 * s, sh[1] + 6 * s), (hx + f * 18 * s, hy + 30 * s), (hx + f * 30 * s, hy + 36 * s)], 3.4 * s, ink=ink, dry=0.3, profile=PRESS_DOWN)]
    else:
        lean = {'stand': 0, 'speak': 2, 'listen': -2}.get(pose, 0)
        hx, hy = x + f * (3 + lean) * s, y - 92 * s
        sh = (hx - f * 1 * s, hy + 13 * s)
        robe = [(sh[0] - 11 * s, sh[1]), (sh[0] + 11 * s, sh[1]), (x + 17 * s, y - 4 * s), (x + 19 * s, y),
                (x - 18 * s, y), (x - 16 * s, y - 4 * s)]
        lines = [B([(sh[0] - f * 10 * s, sh[1] + 2), (x - f * 14 * s, y - 40 * s), (x - f * 18 * s, y - 2)], 4.0 * s, ink=ink, dry=0.35, profile=PRESS_DOWN),
                 B([(sh[0] + f * 8 * s, sh[1] + 8 * s), (x + f * 15 * s, y - 2)], 2.6 * s, ink=ink * 0.6, dry=0.55, profile=PRESS_DOWN),
                 B([(x - 18 * s, y - 1), (x + 19 * s, y - 1)], 3.2 * s, ink=ink * 0.8, dry=0.5, profile=SWEEP)]
        if pose == 'speak':
            lines.append(B([(sh[0] + f * 6 * s, sh[1] + 10 * s), (sh[0] + f * 20 * s, sh[1] + 14 * s), (sh[0] + f * 32 * s, sh[1] + 2 * s),
                            (sh[0] + f * 38 * s, sh[1] - 6 * s)], 3.6 * s, ink=ink, dry=0.3, profile=PRESS_DOWN))
        else:
            lines.append(B([(sh[0] + f * 6 * s, sh[1] + 10 * s), (sh[0] + f * 14 * s, sh[1] + 34 * s), (sh[0] + f * 10 * s, sh[1] + 46 * s)],
                           3.4 * s, ink=ink, dry=0.3, profile=PRESS_DOWN))
    seq = [pwash(W, H, robe, ink=ink * 0.34, soft=2.2 * s, seed=seed, grain=0.45)] + lines
    circ = lambda cx, cy, r, sq=1.0: [(cx + r * np.cos(a), cy + r * sq * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
    seq.append([pwash(W, H, circ(hx, hy, 6.8 * s, 1.12), ink=0.86, soft=0.9 * s, seed=seed + 3, grain=0.1),       # head
                pwash(W, H, circ(hx - f * 2.5 * s, hy - 8.5 * s, 3.6 * s), ink=0.9, soft=0.7 * s, seed=seed + 4)])  # hair knot
    t = t0
    step = dur / len(seq)
    for st in seq:
        t = ss.add(st, t, step * 1.4) - step * 0.4
    return t


def granary(ss, x, y, s, t0, seed, dur=1.0):
    """a Yayoi raised-floor storehouse: four posts, a platform, a steep gabled thatch roof, a notched ladder"""
    W, H = ss.w, ss.h
    r = np.random.default_rng(seed)
    B = lambda pts, w, **k: stroke(W, H, pts, w, seed=seed + len(ss.items), **k)
    posts = [B([(x + dx * s, y - 70 * s), (x + dx * s + r.normal(0, 1), y)], 5 * s, ink=0.75, dry=0.4, profile=BLUNT) for dx in (-48, -16, 16, 48)]
    t = ss.add(posts, t0, dur * 0.25)
    t = ss.add([B([(x - 62 * s, y - 72 * s), (x + 62 * s, y - 74 * s)], 9 * s, ink=0.8, dry=0.3, profile=BLUNT),
                pwash(W, H, [(x - 52 * s, y - 74 * s), (x + 52 * s, y - 74 * s), (x + 50 * s, y - 112 * s), (x - 50 * s, y - 112 * s)],
                      ink=0.25, soft=1.5, seed=seed + 1)], t - 0.05, dur * 0.2)
    apex_l, apex_r = (x - 40 * s, y - 196 * s), (x + 40 * s, y - 198 * s)
    roof = [apex_l, apex_r, (x + 82 * s, y - 106 * s), (x - 82 * s, y - 104 * s)]
    t = ss.add(pwash(W, H, roof, ink=0.3, soft=2, seed=seed + 2), t - 0.05, dur * 0.2)
    th = [B([(apex_l[0] + (apex_r[0] - apex_l[0]) * u + r.normal(0, 2), apex_l[1] + 4),
             (x - 82 * s + 164 * s * u + r.normal(0, 3), y - 106 * s)], r.uniform(1.8, 3) * s, ink=r.uniform(0.35, 0.6), dry=0.6,
            profile=PRESS_DOWN) for u in np.linspace(0, 1, 22)]
    t = ss.add(th, t - 0.05, dur * 0.25)
    ss.add([B([(apex_l[0] - 14 * s, apex_l[1] - 12 * s), (apex_r[0] + 14 * s, apex_r[1] - 12 * s)], 6 * s, ink=0.8, dry=0.3, profile=BLUNT),
            B([(x + 70 * s, y), (x + 40 * s, y - 72 * s)], 5 * s, ink=0.65, dry=0.4, profile=BLUNT)], t - 0.05, dur * 0.15)
    return t


def pit_dwelling(ss, x, y, s, t0, seed, dur=0.9):
    """a Jomon/Yayoi pit dwelling: low conical thatched roof with a dark doorway"""
    W, H = ss.w, ss.h
    r = np.random.default_rng(seed)
    apex = (x + r.normal(0, 3) * s, y - 120 * s)
    roof = [apex, (x + 105 * s, y - 8 * s), (x + 96 * s, y), (x - 96 * s, y), (x - 105 * s, y - 8 * s)]
    t = ss.add(pwash(W, H, roof, ink=0.22, soft=2.5, seed=seed), t0, dur * 0.35)
    thatch = []
    for k in range(26):
        u = (k + r.uniform(-0.3, 0.3)) / 25
        bx = x + (u - 0.5) * 205 * s
        topx = apex[0] + (bx - apex[0]) * 0.1
        thatch.append(stroke(W, H, [(topx, apex[1] + 6 * s), ((topx + bx) / 2 + r.normal(0, 2), (apex[1] + y) / 2),
                                    (bx, y - 4 * s)], r.uniform(2.0, 3.6) * s, ink=r.uniform(0.35, 0.6), dry=0.55,
                             profile=PRESS_DOWN, seed=seed + 10 + k))
    t = ss.add(thatch, t - dur * 0.1, dur * 0.45)
    # ridge poles crossing at the apex, doorway
    ss.add([stroke(W, H, [(apex[0] - 12 * s, apex[1] - 14 * s), (apex[0] + 6 * s, apex[1] + 6 * s)], 4 * s, ink=0.8, dry=0.2,
                   profile=BLUNT, seed=seed + 50),
            stroke(W, H, [(apex[0] + 12 * s, apex[1] - 14 * s), (apex[0] - 6 * s, apex[1] + 6 * s)], 4 * s, ink=0.8, dry=0.2,
                   profile=BLUNT, seed=seed + 51),
            pwash(W, H, [(x - 16 * s, y), (x - 12 * s, y - 36 * s), (x + 12 * s, y - 36 * s), (x + 16 * s, y)], ink=0.75,
                  soft=1.5, seed=seed + 52),
            stroke(W, H, [(x - 110 * s, y + 2), (x, y + 6 * s), (x + 110 * s, y + 2)], 5 * s, ink=0.55, dry=0.5,
                   profile=SWEEP, seed=seed + 53)], t - 0.1, dur * 0.3)
    return t + dur * 0.2


def ink_tree(ss, x, y, s, t0, seed, dur=1.0, ink=0.7):
    """a broadleaf tree: a bent trunk and clusters of wet foliage dabs (点葉)"""
    W, H = ss.w, ss.h
    r = np.random.default_rng(seed)
    top = (x + r.normal(0, 18) * s, y - 260 * s)
    trunk = spline([(x, y), (x + 14 * s, y - 90 * s), (top[0] - 8 * s, y - 180 * s), top], 10)
    t = ss.add(stroke(W, H, trunk, 16 * s, ink=ink, dry=0.45, profile=BLUNT, seed=seed), t0, dur * 0.25)
    branches = []
    for k in range(4):
        b0 = trunk[int(len(trunk) * r.uniform(0.45, 0.85))]
        ang = r.uniform(-2.6, -0.5)
        ln = r.uniform(50, 100) * s
        b1 = (b0[0] + np.cos(ang) * ln, b0[1] + np.sin(ang) * ln)
        branches.append(stroke(W, H, [tuple(b0), b1], 6 * s, ink=ink, dry=0.45, profile=PRESS_DOWN, seed=seed + 1 + k))
    t = ss.add(branches, t - 0.05, dur * 0.2)
    dabs = []
    centres = [top] + [tuple(b0 + 0) for b0 in [trunk[int(len(trunk) * u)] for u in (0.6, 0.75)]]
    centres += [(top[0] + r.normal(0, 40) * s, top[1] + r.normal(30, 20) * s) for _ in range(3)]
    for k in range(110):
        c0 = centres[k % len(centres)]
        ang, rad = r.uniform(0, 2 * np.pi), abs(r.normal(0, 1)) * 34 * s
        cx, cy = c0[0] + np.cos(ang) * rad * 1.5, c0[1] - 6 * s + np.sin(ang) * rad * 0.75
        a2 = r.uniform(0, np.pi)
        ln = r.uniform(7, 13) * s
        dabs.append(stroke(W, H, [(cx - np.cos(a2) * ln, cy - np.sin(a2) * ln * 0.4), (cx + np.cos(a2) * ln, cy + np.sin(a2) * ln * 0.4)],
                           r.uniform(8, 13) * s, ink=r.uniform(0.3, 0.7) * ink / 0.7, dry=0.25, profile=BLUNT, seed=seed + 20 + k))
    order = np.argsort([d.y0 if d is not None else 0 for d in dabs])
    dabs = [dabs[i] for i in order]
    for k in range(0, len(dabs), 10):
        ss.add(dabs[k:k + 10], t + k / len(dabs) * dur * 0.5, dur * 0.12)
    return t + dur * 0.6


# ------------------------------------------------------------------ painted density helpers
def soft_disc(h, w, cx, cy, r, edge=2.5, rough=0.0, seed=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    if rough:
        d = d + (noise(h, w, 18, 3, seed) - 0.5) * rough
    return np.clip((r - d) / edge + 0.5, 0, 1).astype(np.float32)


def sun_disc(h, w, cx, cy, r, seed=0):
    """vermilion sun painted with a loaded brush: uneven body, wet darker rim, slight bleed"""
    body = soft_disc(h, w, cx, cy, r, edge=3.0, rough=5, seed=seed)
    tex = 0.78 + 0.22 * noise(h, w, 60, 4, seed + 1)
    rim = np.clip(blur(body, 2) - blur(body, 10), 0, None) * 0.9
    return np.clip(body * tex * 0.86 + rim * 0.35 + blur(body, 14) * 0.05, 0, 1).astype(np.float32)


def blossom_cluster(h, w, cx, cy, n, rad, seed=0):
    """cherry blossom dabs (pale sakura with darker centres), density for SAKURA colour"""
    r = np.random.default_rng(seed)
    d = np.zeros((h, w), np.float32)
    for _ in range(n):
        ang, rr = r.uniform(0, 2 * np.pi), abs(r.normal(0, 0.55)) * rad
        x, y = cx + np.cos(ang) * rr, cy + np.sin(ang) * rr * 0.7
        pr = r.uniform(5, 10)
        for k in range(5):
            a = k / 5 * 2 * np.pi + r.uniform(0, 1)
            cv2.circle(d, (int(x + np.cos(a) * pr * 0.7), int(y + np.sin(a) * pr * 0.7)), int(pr * 0.62), float(r.uniform(0.35, 0.6)), -1, cv2.LINE_AA)
    return np.clip(blur(d, 1.2), 0, 0.75)


def mist_band(h, w, y, thick, seed=0, strength=0.85):
    yy = np.arange(h, dtype=np.float32)[:, None]
    n = noise(h, w, 260, 3, seed)
    m = np.exp(-((yy - y - 60 * (n - 0.5)) / thick) ** 2) * (0.55 + 0.45 * noise(h, w, 120, 3, seed + 1))
    return np.clip(m * strength, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ motifs added for the kanji reel
def pagoda(ss, x, y, s, t0=0.0, dur=0.0, seed=0, ink=1.0):
    """five-storey pagoda, base at (x, y); tiers are painted bottom-up over `dur` seconds"""
    W_, H_ = ss.w, ss.h
    h = 0
    step = dur / 6
    for k in range(5):
        wr = (78 - 9 * k) * s
        yb = y - h
        body_h = (26 - 2 * k) * s
        ry = yb - body_h
        roof = spline([(x - wr - 8 * s, ry - 8 * s), (x - wr * 0.6, ry - 2 * s), (x, ry - 12 * s), (x + wr * 0.6, ry - 2 * s),
                       (x + wr + 8 * s, ry - 8 * s)], 8)
        ss.add([pwash(W_, H_, [(x - wr * 0.42, yb), (x + wr * 0.42, yb), (x + wr * 0.4, ry), (x - wr * 0.4, ry)],
                      ink=0.22 * ink, soft=1.2, seed=seed + k),
                stroke(W_, H_, roof, 7 * s, ink=0.55 * ink, dry=0.35, profile=BLUNT, seed=seed + 10 + k),
                pwash(W_, H_, [(x - wr, ry - 6 * s), (x + wr, ry - 6 * s), (x + wr * 0.45, ry - 22 * s), (x - wr * 0.45, ry - 22 * s)],
                      ink=0.3 * ink, soft=1.2, seed=seed + 20 + k)], t0 + k * step, step * 1.3)
        h += body_h + 22 * s
    ss.add([stroke(W_, H_, [(x, y - h), (x, y - h - 70 * s)], 3 * s, ink=0.55 * ink, dry=0.2, profile=BLUNT, seed=seed + 40)] +
           [stroke(W_, H_, [(x - 7 * s, y - h - 12 * s - j * 9 * s), (x + 7 * s, y - h - 12 * s - j * 9 * s)], 2.2 * s,
                   ink=0.5 * ink, dry=0.2, profile=BLUNT, seed=seed + 50 + j) for j in range(6)], t0 + 5 * step, step)
    return y - h - 70 * s


def figure_back(ss, x, y, s, t0, seed, dur=1.0, ink=0.85, pose='sit'):
    """a person seen from behind (~175*s px tall), seated (pose='sit') or bowed over a scroll ('read'), hair tied in
    a topknot; base of the robe at (x, y).  One dark wash for the robe, a few contour strokes, a wet head."""
    W_, H_ = ss.w, ss.h
    r = np.random.default_rng(seed)
    bow = 14 * s if pose == 'read' else 0
    hx, hy = x + bow * 0.2, y - 158 * s + bow
    circ = lambda cx, cy, rx, ry: [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 26, endpoint=False)]
    sh = hy + 30 * s
    robe = [(hx - 9 * s, hy + 16 * s), (hx + 9 * s, hy + 16 * s), (x + 36 * s, sh), (x + 54 * s, sh + 22 * s),
            (x + 62 * s, y - 70 * s), (x + 74 * s, y - 24 * s), (x + 82 * s, y - 4 * s), (x + 76 * s, y), (x - 76 * s, y),
            (x - 82 * s, y - 4 * s), (x - 74 * s, y - 24 * s), (x - 62 * s, y - 70 * s), (x - 54 * s, sh + 22 * s), (x - 36 * s, sh)]
    t = ss.add([pwash(W_, H_, robe, ink=0.62 * ink, soft=3 * s, seed=seed, grain=0.35),
                pwash(W_, H_, [(x - 30 * s, sh + 30 * s), (x + 30 * s, sh + 30 * s), (x + 50 * s, y - 6 * s), (x - 50 * s, y - 6 * s)],
                      ink=0.18 * ink, soft=6 * s, seed=seed + 1, grain=0.5)], t0, dur * 0.35)
    lines = [stroke(W_, H_, spline([(x - 36 * s, sh), (x - 56 * s, sh + 24 * s), (x - 64 * s, y - 70 * s), (x - 80 * s, y - 6 * s)], 8),
                    6 * s, ink=ink, dry=0.4, profile=PRESS_DOWN, seed=seed + 2),
             stroke(W_, H_, spline([(x + 36 * s, sh), (x + 56 * s, sh + 24 * s), (x + 64 * s, y - 70 * s), (x + 80 * s, y - 6 * s)], 8),
                    6 * s, ink=ink, dry=0.4, profile=PRESS_DOWN, seed=seed + 3),
             stroke(W_, H_, spline([(hx + r.normal(0, 1), sh + 6 * s), (x + 3 * s, y - 70 * s), (x + 6 * s, y - 8 * s)], 8),
                    3.2 * s, ink=ink * 0.55, dry=0.55, profile=PRESS_DOWN, seed=seed + 4),
             stroke(W_, H_, spline([(x - 52 * s, y - 90 * s), (x - 30 * s, y - 40 * s), (x - 34 * s, y - 6 * s)], 8),
                    3.0 * s, ink=ink * 0.45, dry=0.6, profile=PRESS_DOWN, seed=seed + 5),
             stroke(W_, H_, spline([(x + 52 * s, y - 90 * s), (x + 32 * s, y - 40 * s), (x + 36 * s, y - 6 * s)], 8),
                    3.0 * s, ink=ink * 0.45, dry=0.6, profile=PRESS_DOWN, seed=seed + 6),
             stroke(W_, H_, [(hx - 14 * s, hy + 22 * s), (hx, hy + 27 * s), (hx + 14 * s, hy + 22 * s)], 5 * s, ink=ink * 0.9, dry=0.3,
                    profile=BLUNT, seed=seed + 7)]                                                     # collar
    t = ss.add(lines, t - dur * 0.1, dur * 0.35)
    ss.add([pwash(W_, H_, circ(hx, hy, 16 * s, 19 * s), ink=ink * 0.95, soft=1.5 * s, seed=seed + 30, grain=0.15),
            pwash(W_, H_, circ(hx, hy - 21 * s, 8 * s, 7 * s), ink=ink, soft=1.0 * s, seed=seed + 32),
            stroke(W_, H_, [(hx - 7 * s, hy - 15 * s), (hx + 7 * s, hy - 15 * s)], 3 * s, ink=ink, dry=0.2, profile=BLUNT, seed=seed + 33)],
           t - dur * 0.1, dur * 0.3)
    return hx, hy


def cherry_branch(ss, pts, width, t0, dur, seed=0):
    """a gnarled blossom branch along pts (the trunk end first); returns the blossom centres"""
    W_, H_ = ss.w, ss.h
    r = np.random.default_rng(seed)
    main = spline(pts, 12)
    t = ss.add(stroke(W_, H_, main, width, ink=0.8, dry=0.5, profile=BLUNT, seed=seed), t0, dur * 0.4)
    twigs, tips = [], []
    for k in range(7):
        b0 = main[int(len(main) * r.uniform(0.25, 0.95))]
        ang = r.uniform(-2.8, 0.4)
        ln = r.uniform(40, 110) * width / 14
        b1 = (b0[0] + np.cos(ang) * ln, b0[1] + np.sin(ang) * ln)
        twigs.append(stroke(W_, H_, [tuple(b0), b1], width * 0.35, ink=0.75, dry=0.5, profile=PRESS_DOWN, seed=seed + 1 + k))
        tips.append(b1)
    ss.add(twigs, t - dur * 0.1, dur * 0.3)
    return tips + [tuple(main[-1]), tuple(main[len(main) // 2])]


def sparrow(ss, x, y, s, t0, dur, seed=0, ink=0.9):
    """a small bird perched side-on facing left (head at upper left, tail down-right), ~150*s px wide, painted in
    the 没骨 manner: a wash body, wet wing strokes, a dark cap and eye, thin legs"""
    W_, H_ = ss.w, ss.h
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    body = P([(-50, -30), (-30, -48), (0, -50), (28, -36), (60, -8), (84, 22), (70, 26), (30, 18), (-10, 10), (-38, -6)])
    t = ss.add(pwash(W_, H_, body, ink=0.32 * ink, soft=2.5 * s, seed=seed, grain=0.35), t0, dur * 0.25)
    wing = [stroke(W_, H_, spline(P([(-8, -38), (24, -26), (52, -4), (78, 18)]), 8), 15 * s, ink=0.85 * ink, dry=0.35, profile=PRESS_DOWN, seed=seed + 1),
            stroke(W_, H_, spline(P([(-4, -24), (26, -12), (54, 8)]), 8), 11 * s, ink=0.7 * ink, dry=0.45, profile=PRESS_DOWN, seed=seed + 2),
            stroke(W_, H_, spline(P([(60, 6), (92, 34), (120, 58)]), 8), 9 * s, ink=0.85 * ink, dry=0.4, profile=PRESS_DOWN, seed=seed + 3),
            stroke(W_, H_, spline(P([(56, 12), (84, 42), (106, 66)]), 8), 7 * s, ink=0.7 * ink, dry=0.5, profile=PRESS_DOWN, seed=seed + 4)]
    t = ss.add(wing, t - dur * 0.05, dur * 0.35)
    circ = lambda cx, cy, rr: [(x + (cx + rr * np.cos(a)) * s, y + (cy + rr * np.sin(a)) * s) for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
    head = [pwash(W_, H_, circ(-34, -36, 17), ink=0.75 * ink, soft=1.4 * s, seed=seed + 5, grain=0.2),
            pwash(W_, H_, circ(-40, -38, 3.2), ink=1.0, soft=0.5 * s, seed=seed + 6),
            stroke(W_, H_, P([(-50, -34), (-66, -30), (-50, -26)]), 4 * s, ink=0.95 * ink, dry=0.1, profile=BLUNT, seed=seed + 7),
            stroke(W_, H_, spline(P([(-38, -6), (-24, 8), (0, 14), (24, 16)]), 8), 4 * s, ink=0.55 * ink, dry=0.4, profile=SWEEP, seed=seed + 8)]
    t = ss.add(head, t - dur * 0.05, dur * 0.25)
    legs = [stroke(W_, H_, P([(-6, 12), (-12, 36), (-24, 44)]), 2.6 * s, ink=ink, dry=0.2, profile=BLUNT, seed=seed + 9),
            stroke(W_, H_, P([(-12, 36), (-4, 46)]), 2.2 * s, ink=ink, dry=0.2, profile=BLUNT, seed=seed + 10),
            stroke(W_, H_, P([(10, 16), (8, 38), (-2, 46)]), 2.6 * s, ink=ink, dry=0.2, profile=BLUNT, seed=seed + 11)]
    return ss.add(legs, t - dur * 0.05, dur * 0.2)


def enso(ss, cx, cy, r, width, t0, dur, seed=0, start=-1.9, sweep=5.75, red=False):
    """an ensō: one breath of the brush around a circle, heavy at the start, dry and open at the end"""
    W_, H_ = ss.w, ss.h
    a = np.linspace(start, start + sweep, 60)
    rr = r * (1 + 0.025 * np.sin(a * 2.3 + seed))
    pts = [(cx + q * np.cos(b), cy + q * np.sin(b)) for q, b in zip(rr, a)]
    st = stroke(W_, H_, pts, width, ink=0.95, dry=0.62, profile=((0, 0.7), (0.06, 1.0), (0.7, 0.85), (1, 0.3)), seed=seed, red=red)
    return ss.add(st, t0, dur, bloom_amt=0.12)

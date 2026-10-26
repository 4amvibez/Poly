"""audio — the soundtrack, synthesized in code and synced to the animation's own events.

Score: a quiet drone in the miyako-bushi scale (D Eb G A Bb) with Karplus-Strong koto plucks at scene beats and on
the things that land (cards turning, clue badges, stepping stones), a low taiko when the question is underlined and
when the secret is stamped, singing-bowl chimes on the reveals: the ？ dot, the frame round 日本語, the brain lighting
up, the last stepping stone, またね！  Every sync hit sits in a measured breath of the narration, never on a word.
Sound design: brush-on-paper texture generated from every stroke and glyph the scenes write (timed from the scene
objects themselves), seal stamps, card flips, the little bullet train's wheels clacking over the rail joints as it
rolls in and pulls away, wooden clacks as its couplers snap shut, a cup set down on its saucer, running water under
the stepping stones, the bridge and the endless line, paper whooshes at the transitions.  The music ducks under the
narration.

    python audio.py            -> audio/mix_raw.wav (48 kHz stereo) + audio/stems/*.wav
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SR = 48000

from timeline import DURATION, JA, SCENES, VO_OFFSET

N = int(DURATION * SR)
rng = np.random.default_rng(7)
D4 = 293.66
SCALE = {'D': 0, 'Eb': 1, 'G': 5, 'A': 7, 'Bb': 8}


def note(name, octave):
    return D4 * 2 ** ((SCALE[name] + 12 * (octave - 4)) / 12)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype='low', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype='high', fs=SR, output='sos'), x)


def place(bus, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= bus.shape[0] or i + len(sig) <= 0:
        return
    s0 = max(0, -i)
    sig = sig[s0:]
    i = max(0, i)
    n = min(len(sig), bus.shape[0] - i)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    bus[i:i + n, 0] += sig[:n] * gain * l * 1.414
    bus[i:i + n, 1] += sig[:n] * gain * r * 1.414


def env_adsr(n, a, d, s, r):
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n)
    e[:na] = np.linspace(0, 1, na)
    if na + nd < n:
        e[na:na + nd] = np.linspace(1, s, nd)
    if nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ------------------------------------------------------------------ instruments
def koto(freq, dur=2.6, bright=0.5):
    """Karplus-Strong plucked string with a koto-like bright attack"""
    n = int(dur * SR)
    period = int(SR / freq)
    buf = rng.uniform(-1, 1, period) * np.hanning(period) ** 0.3
    out = np.zeros(n)
    decay = 0.996 - (1 - bright) * 0.004
    b = buf.copy()
    for i in range(0, n, period):
        k = min(period, n - i)
        out[i:i + k] = b[:k]
        b = decay * 0.5 * (b + np.roll(b, 1))
    pluck = hp(rng.normal(0, 1, int(0.012 * SR)), 2000) * np.linspace(1, 0, int(0.012 * SR)) * 0.4
    out[:len(pluck)] += pluck
    return out * np.exp(-np.linspace(0, dur, n) * 1.2) * 0.5


def bowl(freq, dur=6.0):
    """singing bowl: inharmonic partials with slow beating, long decay"""
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for ratio, amp, dec in ((1, 1, 0.6), (2.71, 0.5, 0.9), (5.12, 0.25, 1.4), (8.2, 0.12, 2.2)):
        out += amp * np.sin(2 * np.pi * freq * ratio * t + 0.3 * np.sin(2 * np.pi * 1.7 * t)) * np.exp(-t * dec)
    out *= np.minimum(1, t / 0.004)
    return out * 0.18


def taiko(dur=2.0, pitch=58):
    """a low drum: pitched body falling in frequency, a felt-soft attack"""
    t = np.arange(int(dur * SR)) / SR
    f = pitch * (1 + 0.6 * np.exp(-t * 18))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    skin = lp(rng.normal(0, 1, len(t)), 900) * np.exp(-t * 30) * 0.5
    return np.tanh((body + skin) * 1.4) * 0.6


def flute(freq, dur, vib=5.0):
    """breathy bamboo-flute tone (sparingly)"""
    t = np.arange(int(dur * SR)) / SR
    v = 1 + 0.006 * np.sin(2 * np.pi * vib * t) * np.minimum(1, t / 0.8)
    ph = 2 * np.pi * np.cumsum(freq * v) / SR
    tone = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    breath = bp(rng.normal(0, 1, len(t)), freq * 0.9, freq * 3.5) * 0.35
    e = env_adsr(len(t), 0.5, 0.3, 0.8, 1.2)
    return (tone + breath) * e * 0.12


def pad_voice(freq, dur, bright=1200):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for det in (-0.12, 0.0, 0.11):
        f = freq * 2 ** (det / 12)
        for h, a in ((1, 1), (2, 0.4), (3, 0.18), (4, 0.08)):
            out += a * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6))
    return lp(out, bright) / 6


def brush(dur, intensity=1.0, heavy=False):
    """bristles dragged over washi: band-passed noise with a grainy, pressured envelope"""
    dur = max(dur, 0.06)
    n = int(dur * SR)
    noise_ = rng.normal(0, 1, n + int(0.15 * SR))
    tex = bp(noise_, 1400, 7000)
    low = bp(noise_, 250, 1200) * (0.6 if heavy else 0.25)
    sig = tex + low
    grain = np.interp(np.arange(len(sig)), np.linspace(0, len(sig), max(4, int(len(sig) / SR * 45))),
                      rng.uniform(0.55, 1.0, max(4, int(len(sig) / SR * 45))))
    e = np.ones(len(sig))
    a = int(0.012 * SR)
    e[:a] = np.linspace(0, 1, a)
    pressure = np.interp(np.arange(len(sig)), [0, n * 0.15, n * 0.8, len(sig)], [1.0, 0.85, 0.55, 0.0])
    out = sig * grain * e * pressure
    if heavy:
        thump = np.sin(2 * np.pi * 140 * np.arange(int(0.05 * SR)) / SR) * np.exp(-np.arange(int(0.05 * SR)) / SR * 60)
        out[:len(thump)] += thump * 2.5
    return out * 0.05 * intensity


def whoosh(dur=1.2, lo=300, hi=3000):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    out = np.zeros(n)
    seg = 512
    for i in range(0, n, seg):
        u = i / n
        fc = lo + (hi - lo) * np.sin(np.pi * u) ** 1.5
        out[i:i + seg] = bp(x[max(0, i - 2048):i + seg], max(60, fc * 0.5), min(SR / 2 - 100, fc * 1.6))[-min(seg, n - i):]
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * 0.12


def stamp():
    """a seal pressed onto paper: a soft low thump with a papery crush on top"""
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * 95 * t * (1 + 0.5 * np.exp(-t * 40))) * np.exp(-t * 22)
    crush = bp(rng.normal(0, 1, n), 600, 4000) * np.exp(-t * 35) * 0.5
    return (thump + crush) * 0.22


def flip():
    """a stiff card turned over: a quick papery flick"""
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    return bp(rng.normal(0, 1, n), 1200, 6500) * np.sin(np.pi * t / t[-1]) ** 0.5 * np.exp(-t * 18) * 0.12


def clack(pitch=1.0):
    """hyoshigi: two hardwood blocks struck together (a bright, short, pitched knock)"""
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    body = sum(a * np.sin(2 * np.pi * f * pitch * t) * np.exp(-t * d) for f, a, d in ((1180, 1, 38), (2630, 0.5, 55), (4100, 0.25, 80)))
    click = hp(rng.normal(0, 1, n), 3000) * np.exp(-t * 300) * 0.6
    return (body + click) * 0.25


def clink():
    """a ceramic cup set down on its saucer: a bright inharmonic ping with a tiny rattle"""
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    out = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((2210, 1, 6), (3480, 0.6, 9), (5170, 0.35, 14), (6890, 0.2, 20)))
    out += hp(rng.normal(0, 1, n), 2500) * np.exp(-t * 120) * 0.5
    k = int(0.07 * SR)
    out[k:] += out[:-k] * 0.25 * np.exp(-t[k:] * 8)                     # it rocks once on the saucer
    return out * 0.07


def wheels(t0, t1, speed, gain=1.0):
    """train wheels over rail joints: pairs of clicks (ta-tan) whose spacing follows speed(t) in 0..1, over a rumble"""
    n = int((t1 - t0) * SR)
    out = np.zeros(n)
    tt = 0.0
    while tt < t1 - t0:
        v = max(speed(t0 + tt), 0.05)
        for dt, a in ((0.0, 1.0), (0.11 / (0.4 + v), 0.8)):
            i = int((tt + dt) * SR)
            k = int(0.03 * SR)
            if i + k < n:
                out[i:i + k] += bp(rng.normal(0, 1, k), 1500, 6000) * np.exp(-np.arange(k) / SR * 140) * a * v ** 0.5
        tt += 0.55 / (0.35 + v)
    sp = np.array([speed(t0 + i / SR) for i in range(0, n, 480)])
    env = np.interp(np.arange(n), np.arange(len(sp)) * 480, sp)
    rumble = lp(rng.normal(0, 1, n), 220) * env * 0.5 + bp(rng.normal(0, 1, n), 300, 1200) * env ** 2 * 0.15
    return (out * 0.18 + rumble * 0.2) * gain


def water(n, seed=0, bright=1.0):
    """running water: band-passed noise with slow gurgling modulation and small droplets"""
    r = np.random.default_rng(seed)
    base = bp(r.normal(0, 1, n), 400, 2600) * (0.6 + 0.4 * lp(r.normal(0, 1, n), 3) * 30)
    drops = np.zeros(n)
    for _ in range(int(n / SR * 9)):
        i = int(r.uniform(0, n - 4000))
        k = int(r.uniform(0.01, 0.03) * SR)
        f = r.uniform(900, 2400) * bright
        drops[i:i + k] += np.sin(2 * np.pi * np.cumsum(np.linspace(f, f * 1.6, k)) / SR) * np.exp(-np.arange(k) / k * 4) * r.uniform(0.2, 0.6)
    return base * 0.03 + drops * 0.025


# ------------------------------------------------------------------ events from the scenes
def hit_times():
    """the sync hits: (name, time) — each in a breath of the narration, each with a picture event on it"""
    import scenes
    from timeline import JA
    g = scenes.get
    return [('hook_flick', g('hook').t_hit), ('question', g('spaces').t_hit), ('nihongo', g('kanji').t_hit),
            ('couplers', g('hiragana').t_hit), ('cup', g('katakana').t_hit), ('secret', g('secret').t_in),
            ('brain', g('secret').t_brain), ('last_step', g('endcard').t_hit), ('matane', JA['matane'])]


def scene_events():
    """(time, duration, intensity, heavy) for every brush event the animation shows"""
    import scenes
    from timeline import JA_SENT
    ev = []

    def strokes(ss, gain=1.0):
        for st, s, d, *_ in ss.items:
            area = float(st.dens.sum())
            ev.append((s, d, gain * np.clip(np.sqrt(area) / 220, 0.15, 1.6), area > 60 * 900))

    def clue(c):
        for k in range(len(c.jp)):
            ev.append((c.t_in + 0.15 + 0.12 * k, 0.35, 0.8, False))

    s = scenes.get('hook')
    for L in s.title:
        ev.append((max(L['t'], 0.0), L['dur'] + min(L['t'], 0), 1.0, False))
    strokes(s.flick, 1.2)
    s = scenes.get('spaces')
    ev += [(s.t_en - 0.2, 0.4, 0.8, True), (s.t_jp - 0.2, 0.4, 0.8, True), (s.t_jp + 0.3, 0.9, 0.9, False)]
    strokes(s.line, 1.0)
    strokes(s.q, 1.3)
    for i, ts in s.splits:
        ev.append((ts, 0.12, 0.3, False))
    s = scenes.get('kanji')
    clue(s.clue)
    for a, b in JA_SENT['chunks']:
        ev.append((a, b - a, 0.9, False))
    for _, ts in s.gold:
        ev.append((ts, 0.35, 0.6, False))
    strokes(s.frame_red, 1.0)
    s = scenes.get('hiragana')
    clue(s.clue)
    ev.append((s.t_write, 0.8, 0.8, False))
    for tm in s.t_mark:
        ev.append((tm, 0.3, 0.6, False))
    strokes(s.ticks, 0.8)
    strokes(s.arcs, 0.7)
    for gl, xs, ts in s.car_words:
        ev.append((ts, 0.07 * len(gl) + 0.3, 0.6, False))
    s = scenes.get('katakana')
    clue(s.clue)
    for k in range(4):
        ev.append((s.t_hit + 0.08 + 0.14 * k, 0.35, 0.9, k == 0))
    s = scenes.get('secret')
    clue(s.clue)
    ev.append((s.t_head, 1.0, 1.0, True))
    for ss in s.streams:
        strokes(ss, 0.6)
    ev.append((s.t_write, 0.9, 0.8, False))
    for k in range(3):
        ev.append((s.t_chunk + 0.28 * k, 0.35, 0.6, False))
    strokes(s.ticks, 0.8)
    s = scenes.get('endless')
    strokes(s.under, 0.8)
    s = scenes.get('endcard')
    ev.append((s.t_follow - 0.1, 0.36, 1.3, True))
    for k in range(4):
        ev.append((s.t0 + 0.5 + 0.15 * k, 0.4, 0.7, False))
    for line in s.lines:
        for g, gx, gy, ts, d in line:
            ev.append((ts, d, 0.7, False))
    # keep each sync hit the only onset in its moment: a brush that would start right on top of a hit starts
    # just before it instead (the picture leads by a frame or two, which reads as the same moment)
    hits = [h for _, h in hit_times()]
    out = []
    for t, d, i, hv in ev:
        for h in hits:
            if abs(t - h) < 0.13:
                d, t = d + (t - (h - 0.14)), h - 0.14
        out.append((t, max(d, 0.06), i, hv))
    return out


# ------------------------------------------------------------------ build the mix
def build(out_dir=os.path.join(HERE, 'audio')):
    import scenes
    g = scenes.get
    music = np.zeros((N, 2))
    sfx = np.zeros((N, 2))
    amb = np.zeros((N, 2))
    T0 = {nm: a0 for nm, a0, _ in SCENES}
    H_ = dict(hit_times())

    # drone pad: chord colour per scene, swelling with the story
    chords = {
        'hook': ([('D', 2), ('A', 2), ('D', 3)], 0.6), 'spaces': ([('G', 2), ('D', 3), ('A', 3)], 0.55),
        'kanji': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.6), 'hiragana': ([('Bb', 1), ('D', 3), ('A', 3)], 0.55),
        'katakana': ([('G', 2), ('D', 3), ('Bb', 3)], 0.6), 'secret': ([('D', 2), ('A', 2), ('Eb', 3), ('Bb', 3)], 0.8),
        'endless': ([('G', 2), ('D', 3), ('A', 3)], 0.65), 'endcard': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.75)}
    for nm, a, b in SCENES:
        notes, lvl = chords[nm]
        dur = b - a + 1.5
        sig = sum(pad_voice(note(n, o), dur, bright=900 + 400 * lvl) for n, o in notes) / len(notes) ** 0.5
        e = np.minimum(1, np.minimum(np.arange(len(sig)) / (1.4 * SR), (len(sig) - np.arange(len(sig))) / (1.6 * SR)))
        e = np.clip(e, 0, 1) ** 1.5
        place(music, sig * e * lvl * 0.5, a, pan=-0.15)
        place(music, np.roll(sig, 900) * e * lvl * 0.5, a, pan=0.15)

    def phrase(t0, notes, gap=0.32, gain=0.9, pan=0.2):
        for k, (n, o) in enumerate(notes):
            place(music, koto(note(n, o)), t0 + k * gap, gain, pan=pan + 0.1 * np.sin(k))

    phrase(0.05, [('D', 4), ('A', 4)], gap=0.3, gain=0.7)
    sp = g('spaces')
    phrase(T0['spaces'] + 0.3, [('D', 4), ('Eb', 4), ('A', 4)], gap=0.45, gain=0.5)
    for tt, n in zip(sp.t_stone, [('G', 4), ('A', 4), ('D', 5)]):
        phrase(tt, [n], gain=0.45)
    phrase(sp.t_jp, [('A', 3), ('D', 4), ('Eb', 4)], gap=0.4, gain=0.5, pan=-0.2)
    kj = g('kanji')
    phrase(kj.clue.t_in, [('A', 3)], gain=0.6)
    for tt, n in zip(kj.t_cards, [('D', 5), ('A', 4), ('G', 4)]):
        phrase(tt, [n], gain=0.45)
    hg = g('hiragana')
    phrase(hg.clue.t_in, [('D', 4)], gain=0.6)
    for tt, n in zip(hg.t_mark, [('A', 4), ('Bb', 4), ('D', 5)]):
        phrase(tt, [n], gain=0.4)
    phrase(hg.t_rel, [('G', 4), ('A', 4)], gap=0.45, gain=0.4)
    kt = g('katakana')
    phrase(kt.clue.t_in, [('G', 3)], gain=0.6)
    for tt, n in zip(kt.t_card, [('D', 5), ('A', 4), ('G', 4)]):
        phrase(tt, [n], gain=0.45)
    phrase(kt.t_coffee, [('A', 4), ('D', 5)], gap=0.25, gain=0.4)
    sc = g('secret')
    for tt, n in zip(sc.t_badge, [('D', 4), ('Eb', 4), ('G', 4), ('A', 4), ('Bb', 4)]):
        phrase(tt, [n], gain=0.5, pan=-0.3)
    phrase(sc.t_bound, [('A', 4), ('D', 5)], gap=0.3, gain=0.4)
    place(music, flute(note('A', 4), 3.6), sc.t0 + 0.8, 0.4, pan=-0.2)
    en = g('endless')
    phrase(en.t_pat, [('D', 4), ('G', 4), ('A', 4), ('D', 5)], gap=0.28, gain=0.45)
    phrase(en.t_part + 0.25, [('A', 4), ('G', 4), ('D', 5)], gap=0.15, gain=0.4)
    ec = g('endcard')
    phrase(ec.t_follow, [('D', 4), ('A', 4)], gap=0.2, gain=0.5)
    for k, n in enumerate([('D', 4), ('G', 4), ('A', 4)]):
        phrase(ec.t_hop[k] + ec.hop_d[k], [n], gain=0.45)
    place(music, flute(note('D', 5), 3.2), JA['start'] - 0.2, 0.35, pan=0.2)
    phrase(JA['matane'] + 0.25, [('D', 3), ('A', 3), ('D', 4), ('A', 4)], gap=0.06, gain=0.45)

    # accents, each in a breath of the narration
    place(music, taiko(2.4, 52), H_['hook_flick'], 0.9)
    place(music, bowl(note('A', 4)), H_['question'], 0.75, pan=0.2)
    place(music, bowl(note('D', 5)), H_['nihongo'], 0.6, pan=-0.2)
    place(music, koto(note('A', 4)), H_['nihongo'], 0.6, pan=-0.2)          # a short breath: give the strike a sharp edge
    place(music, bowl(note('A', 5), 4.0), H_['cup'], 0.3, pan=0.1)
    place(music, taiko(2.2, 60), H_['secret'], 0.7)
    place(music, bowl(note('D', 5), 5.0), H_['brain'], 0.75)
    place(music, bowl(note('A', 4), 5.0), H_['last_step'], 0.6, pan=0.2)
    place(music, koto(note('D', 5)), H_['last_step'], 0.5)
    place(music, bowl(note('D', 5), 5.0), H_['matane'], 0.45)
    place(sfx, clack(1.0), H_['couplers'], 1.0, pan=-0.2)
    place(sfx, clack(1.12), H_['couplers'] + 0.09, 0.7, pan=0.2)
    place(sfx, clink(), H_['cup'], 1.0, pan=0.05)
    place(sfx, stamp(), H_['secret'], 1.0)
    # every hit is struck with a soft mallet, and the bed breathes in for it: a short dip just before the strike,
    # so the strike is the one onset in the breath
    dips = np.ones(N)
    for _, th in hit_times():
        n = int(0.025 * SR)
        place(music, hp(rng.normal(0, 1, n), 2500) * np.exp(-np.arange(n) / SR * 180) * 0.5, th, 2.0)
        i0, i1, i2 = int((th - 0.2) * SR), int((th - 0.015) * SR), int((th + 0.6) * SR)   # slow recovery: no swell right after
        gg = np.ones(i2 - i0)
        gg[:i1 - i0] = 1 - 0.7 * np.sin(np.linspace(0, np.pi / 2, i1 - i0)) ** 2
        gg[i1 - i0:] = np.minimum(1, 0.3 + 0.7 * np.linspace(0, 1, i2 - i1) ** 0.5)
        dips[i0:i2] = np.minimum(dips[i0:i2], gg)

    # brush on paper, seals, cards
    for t, d, inten, heavy in scene_events():
        place(sfx, brush(d, inten, heavy), t, 1.0, pan=rng.uniform(-0.25, 0.25))
    for name, a, b in SCENES[1:]:
        place(sfx, whoosh(1.1, 250, 2600), a - 0.1, 0.8, pan=rng.uniform(-0.3, 0.3))
    for c in (kj.clue, hg.clue, kt.clue):
        place(sfx, stamp(), c.t_in, 0.9)
    place(sfx, stamp() * 0.6, sc.t_one, 0.8)
    for tt in sp.t_stone:
        place(sfx, stamp() * 0.5, tt + 0.1, 0.7, pan=-0.1)
    for tt in kj.t_cards:
        place(sfx, flip(), tt, 0.8, pan=rng.uniform(-0.3, 0.3))
    for tt in kt.t_deal:
        place(sfx, flip(), tt, 0.6, pan=rng.uniform(-0.3, 0.3))
    for tt in kt.t_card:
        place(sfx, flip(), tt + 0.18, 1.0, pan=rng.uniform(-0.3, 0.3))
    for k in range(3):
        place(sfx, stamp() * 0.4, ec.t_hop[k] + ec.hop_d[k], 0.6)
    # the train: wheels over the joints as it rolls in, and again as it pulls away
    a0 = hg.t_arrive
    place(amb, wheels(a0, a0 + 1.9, lambda t: hg.train_x(t)[1] if t < a0 + 1.7 else 0.0), a0, 1.0, pan=-0.3)
    d0 = hg.t_depart
    place(amb, wheels(d0, hg.t1 + 0.3, lambda t: hg.train_x(t)[1]), d0, 1.0, pan=0.3)
    # running water: the stream and the river, the endless line, the stepping stones
    for a, b, sd in ((sp.t_en - 0.3, sp.t1, 1), (en.t0, en.t1, 2), (ec.t0, ec.t1, 3)):
        n = int((b - a) * SR)
        e = np.clip(np.minimum(np.arange(n) / SR / 0.8, (n - np.arange(n)) / SR / 0.8), 0, 1)
        place(amb, water(n, sd) * e, a, pan=0.1 * sd - 0.2)

    # narration
    sr, vo = wavfile.read(os.path.join(out_dir, 'narration.wav'))
    vo = vo.astype(np.float64)
    if vo.ndim > 1:
        vo = vo.mean(1)
    vo /= 32768.0 if np.abs(vo).max() > 2 else 1.0
    if sr != SR:
        raise SystemExit('narration.wav must be 48 kHz')
    voice = np.zeros((N, 2))
    place(voice, vo, VO_OFFSET, 1.0)
    # duck music and ambience under the voice: fast attack, slow release (~0.8 s), so the music eases back between
    # sentences instead of jumping up, and the only onset in a breath is the hit placed there
    envv = np.abs(voice[:, 0])
    k = int(0.25 * SR)
    envv = np.convolve(envv, np.ones(k) / k, mode='same')
    envv = envv / (envv.max() + 1e-9)
    hop = SR // 100
    e = np.clip(envv[::hop] * 4, 0, 1)
    out = np.zeros_like(e)
    for i in range(1, len(e)):
        out[i] = e[i] if e[i] > out[i - 1] else out[i - 1] * 0.975 + e[i] * 0.025
    envv = np.interp(np.arange(N), np.arange(len(out)) * hop, out) / 4
    music *= (1 - 0.5 * np.clip(envv * 4, 0, 1))[:, None]
    music *= dips[:, None]
    amb *= (1 - 0.3 * np.clip(envv * 4, 0, 1))[:, None]
    amb *= dips[:, None]

    stems = {'voice': voice, 'music': music * 0.21, 'sfx': sfx * 0.65, 'amb': amb * 0.45}
    mix = sum(stems.values())
    fade = np.ones(N)
    fade[-int(1.2 * SR):] = np.linspace(1, 0, int(1.2 * SR)) ** 1.5
    mix *= fade[:, None]
    mix = np.tanh(mix * 1.1) / 1.1
    os.makedirs(os.path.join(out_dir, 'stems'), exist_ok=True)
    for k, v in stems.items():
        wavfile.write(os.path.join(out_dir, 'stems', f'{k}.wav'), SR, (np.clip(v, -1, 1) * 32767).astype(np.int16))
    wavfile.write(os.path.join(out_dir, 'mix_raw.wav'), SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print('wrote', os.path.join(out_dir, 'mix_raw.wav'))


if __name__ == '__main__':
    build()

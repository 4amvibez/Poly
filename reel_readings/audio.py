"""audio — the soundtrack, synthesized in code and synced to the animation's own events.

Score: a quiet drone in the miyako-bushi scale (D Eb G A Bb) with Karplus-Strong koto plucks on the things that land
(each reading as Raj says it, the readings rows of the hook, the words on the book), a taiko under each kanji that
slams in at the start and on the WHY? stamp, a breathy flute while the junk sails and over the final landscape,
singing-bowl chimes on the reveals.  Every sync hit sits in a measured breath of the narration, never on a word:
WHY? stamped, the junk docking, さん and the sun pulsing after "we say san.", the circle closing round 日 and
round 水 on the calendar, the red circle swelling behind the travellers, the seal.
Sound design: brush-on-paper texture from every stroke and glyph the scenes write, sea and river water, a wood knock
as the junk docks, paper flicks as the calendars swing in, paper whooshes at the transitions.
The music ducks under the narration.

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


def splash():
    """ink flicked off a brush: a wet burst of noise with droplets"""
    n = int(0.7 * SR)
    t = np.arange(n) / SR
    out = bp(rng.normal(0, 1, n), 800, 7000) * np.exp(-t * 9) * 0.5
    for _ in range(18):
        i = int(rng.uniform(0.02, 0.5) * SR)
        k = int(0.012 * SR)
        out[i:i + k] += bp(rng.normal(0, 1, k), 2000, 8000) * np.exp(-np.arange(k) / k * 5) * rng.uniform(0.2, 0.6)
    return out * 0.2


def cluck(pitch=1.0):
    """a hen's short cluck: a pitched, nasal noise burst"""
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = 620 * pitch * (1 + 0.4 * np.exp(-t * 30))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    buzz = bp(rng.normal(0, 1, n), 800, 3000) * 0.4
    return (tone + buzz) * np.sin(np.pi * t / t[-1]) ** 0.5 * np.exp(-t * 12) * 0.12


def thud():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 70 * t * (1 + 0.6 * np.exp(-t * 30))) * np.exp(-t * 16) + lp(rng.normal(0, 1, n), 500) * np.exp(-t * 25) * 0.6) * 0.25


# ------------------------------------------------------------------ events from the scenes
NAMES = ['hook', 'history', 'yama', 'hi', 'mizu', 'hito', 'takeaway']
HIT_NAMES = {'hook': 'why', 'history': 'dock', 'yama': 'san', 'hi': 'sunday', 'mizu': 'wednesday', 'hito': 'nihon',
             'takeaway': 'seal'}


def hit_times():
    """the sync hits: (name, time) — each in a breath of the narration, each with a picture event on it"""
    import scenes
    return [(HIT_NAMES[n], scenes.get(n).t_hit) for n in NAMES]


def scene_events():
    """(time, duration, intensity, heavy) for every brush event the animation shows"""
    import scenes
    from timeline import JA
    ev = []
    g = scenes.get

    def strokes(ss, gain=1.0):
        for st, s0, d, *_ in ss.items:
            area = float(st.dens.sum())
            ev.append((s0, d, gain * np.clip(np.sqrt(area) / 220, 0.15, 1.6), area > 60 * 900))

    def card(t0, n=1):
        ev.append((t0, 0.3 + 0.08 * n, 0.55, False))                  # the card writing itself

    hk = g('hook')
    from scenes.s01_hook import T_SLAM
    for T in T_SLAM:
        ev.append((max(T - 0.02, 0), 0.2, 1.6, True))                 # each kanji slapped on
    strokes(hk.strokes, 0.8)
    hs = g('history')
    strokes(hs.tick, 0.5)
    strokes(hs.split, 1.0)
    ys = g('yama')
    ev.append((ys.t_first, 0.75, 1.3, True))
    card(ys.t_own)
    card(ys.t_but + 0.25, 3)
    hi = g('hi')
    ev.append((hi.t_next, 0.6, 1.3, True))
    card(hi.t_own)
    card(hi.t_but + 0.3, 3)
    mz = g('mizu')
    ev.append((mz.t_now, 0.6, 1.3, True))
    card(mz.t_own)
    strokes(mz.curl, 1.0)
    card(mz.t_but + 0.35, 3)
    strokes(mz.arrows, 0.6)
    ht = g('hito')
    ev.append((ht.t_fin, 0.55, 1.3, True))
    card(ht.t_alone)
    card(ht.t_but + 0.3, 3)
    tk = g('takeaway')
    for k in range(4):
        ev.append((tk.t_so + 0.12 * k, 0.35, 0.9, False))
    strokes(tk.bands, 0.7)
    strokes(tk.ul, 0.8)
    strokes(tk.banner, 1.0)
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
    from timeline import JA
    g = scenes.get
    music = np.zeros((N, 2))
    sfx = np.zeros((N, 2))
    amb = np.zeros((N, 2))
    H_ = dict(hit_times())

    chords = {
        'hook': ([('D', 2), ('A', 2), ('D', 3)], 0.65), 'history': ([('G', 2), ('D', 3), ('A', 3)], 0.6),
        'yama': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.6), 'hi': ([('Bb', 1), ('D', 3), ('A', 3)], 0.6),
        'mizu': ([('G', 2), ('D', 3), ('Bb', 3)], 0.6), 'hito': ([('D', 2), ('A', 2), ('Eb', 3), ('Bb', 3)], 0.65),
        'takeaway': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.8)}
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

    # the hook: a taiko under each kanji slam, koto as the readings appear
    from scenes.s01_hook import T_SLAM
    hk = g('hook')
    for k, T in enumerate(T_SLAM):
        place(music, taiko(1.6, [50, 56, 62, 68][k]), T, 1.0 if k == 0 else 0.75)
        place(sfx, whoosh(0.22, 900, 5000), max(T - 0.12, 0), 0.5)
    phrase(hk.t_read, [('D', 5), ('A', 4), ('D', 5), ('Eb', 5)], gap=0.07, gain=0.35, pan=-0.3)
    phrase(hk.t_read + 0.4, [('A', 4), ('G', 4), ('A', 4), ('D', 4)], gap=0.07, gain=0.35, pan=0.3)
    # the history: a breathy flute as the junk sails, a wood knock as it docks
    hs = g('history')
    place(music, flute(note('A', 4), 2.4), hs.t_ship, 0.35, pan=0.2)
    place(music, flute(note('D', 5), 1.6), hs.t_ship + 1.2, 0.3, pan=0.2)
    place(amb, water(int((hs.t_split - hs.t0) * SR), seed=3, bright=0.6), hs.t0, 0.6)
    for k in range(4):
        phrase(hs.t_own + 0.25 * k, [[('D', 4), ('G', 4), ('A', 4), ('D', 5)][k]], gain=0.35, pan=-0.3)
        phrase(hs.t_pres + 0.35 + 0.25 * k, [[('A', 4), ('Bb', 4), ('D', 5), ('Eb', 5)][k]], gain=0.35, pan=0.3)
    phrase(hs.t_split + 0.7, [('D', 4), ('A', 4)], gap=0.15, gain=0.5)
    # the four kanji scenes: a koto note on each reading as Raj says it
    for key, n in (('yama', ('A', 4)), ('fujisan', ('D', 5)), ('hi', ('A', 4)), ('nichiyoubi', ('D', 5)),
                   ('nichi', ('Eb', 5)), ('mizu', ('G', 4)), ('suiyoubi', ('Bb', 4)), ('sui', ('D', 5)), ('hito', ('A', 4)),
                   ('nihonjin', ('D', 5)), ('jin', ('Eb', 5)), ('onyomi', ('D', 5)), ('kunyomi', ('A', 4))):
        phrase(JA[key]['end'] + 0.02, [n], gain=0.28, pan=0.25)
    mz = g('mizu')
    place(amb, water(int((mz.t1 - mz.t_means) * SR), seed=5, bright=1.0), mz.t_means, 1.0)
    place(sfx, splash() * 0.5, mz.t_means + 0.05, 0.6)
    tk = g('takeaway')
    place(music, flute(note('D', 5), 3.0), tk.t_story - 0.1, 0.35, pan=0.2)
    phrase(tk.t_follow, [('D', 4), ('A', 4), ('D', 5)], gap=0.2, gain=0.45)

    # accents, each in a breath of the narration
    place(music, taiko(2.2, 58), H_['why'], 0.9)
    place(sfx, stamp(), H_['why'], 1.0)
    place(sfx, clack(0.9), H_['dock'], 1.0)
    place(music, bowl(note('A', 4)), H_['dock'], 0.6, pan=0.1)
    place(music, bowl(note('D', 5), 5.0), H_['san'], 0.6, pan=-0.2)
    place(music, koto(note('A', 5)), H_['san'], 0.45)
    place(music, bowl(note('A', 5), 4.0), H_['sunday'], 0.6)
    place(sfx, brush(0.12, 1.0), H_['sunday'] - 0.1, 0.8)
    place(music, bowl(note('G', 5), 4.0), H_['wednesday'], 0.6)
    place(sfx, brush(0.12, 1.0), H_['wednesday'] - 0.1, 0.8)
    place(music, taiko(2.0, 54), H_['nihon'], 0.7)
    place(music, bowl(note('D', 5), 5.0), H_['nihon'], 0.5)
    place(sfx, stamp(), H_['seal'], 1.2)
    place(music, taiko(2.2, 58), H_['seal'], 0.8)
    place(music, bowl(note('D', 5), 5.0), H_['seal'] + 0.05, 0.5)
    dips = np.ones(N)
    for _, th in hit_times():
        n = int(0.025 * SR)
        place(music, hp(rng.normal(0, 1, n), 2500) * np.exp(-np.arange(n) / SR * 180) * 0.5, th, 2.0)
        i0, i1, i2 = int((th - 0.2) * SR), int((th - 0.015) * SR), int((th + 0.6) * SR)
        gg = np.ones(i2 - i0)
        gg[:i1 - i0] = 1 - 0.7 * np.sin(np.linspace(0, np.pi / 2, i1 - i0)) ** 2
        gg[i1 - i0:] = np.minimum(1, 0.3 + 0.7 * np.linspace(0, 1, i2 - i1) ** 0.5)
        dips[i0:i2] = np.minimum(dips[i0:i2], gg)

    # brush on paper, and the page turning between scenes
    for t, d, inten, heavy in scene_events():
        place(sfx, brush(d, inten, heavy), t, 1.0, pan=rng.uniform(-0.25, 0.25))
    for name, a, b in SCENES[1:]:
        place(sfx, whoosh(1.1, 250, 2600), a - 0.1, 0.8, pan=rng.uniform(-0.3, 0.3))
    for tt in (g('hi').t_but + 0.35, g('mizu').t_but + 0.45):
        place(sfx, flip(), tt, 0.9)                                     # the calendar swinging in

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

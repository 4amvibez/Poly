"""audio — the soundtrack, synthesized in code and synced to the animation's own events.

Score: a quiet drone in the miyako-bushi scale (D Eb G A Bb) with Karplus-Strong koto plucks at scene beats, a low
taiko when 日 lands in the sun, singing-bowl chimes on the reveals that end a thought (today's 日, the bird, 河 "ka",
the trail of history, またね！) — every hit placed in a measured breath of the narration, never on a word.
Sound design: brush-on-paper texture generated from every stroke and glyph the scenes write (timed from the scene
objects themselves), a low fire by the painted cliff, a knife scratching bone for the carved pictographs over a
low fire (oracle bones were cracked with heat), ocean swell under the boat, paper whooshes at the
transitions.  The music ducks under the narration.

    python audio.py            -> audio/mix.wav (48 kHz stereo) + audio/stems/*.wav
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SR = 48000

from timeline import DURATION, JA, SCENES, VO_OFFSET, at  # noqa: E402

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


def carve(dur):
    """a blade scratching bone: bright band-passed noise in short strokes"""
    n = int(max(dur, 0.08) * SR)
    x = bp(rng.normal(0, 1, n), 2500, 9000)
    k = max(3, int(dur * 14))
    env = np.zeros(n)
    for _ in range(k):
        a = int(rng.uniform(0, n * 0.9))
        b = min(n, a + int(rng.uniform(0.03, 0.09) * SR))
        env[a:b] = np.maximum(env[a:b], np.sin(np.linspace(0, np.pi, b - a)) * rng.uniform(0.5, 1))
    return x * env * 0.035


# ------------------------------------------------------------------ events from the scenes
def scene_events():
    """(time, duration, intensity, heavy) for every brush event the animation shows"""
    import scenes
    ev = []

    def from_strokes(ss, gain=1.0, t_min=-1, heavy_w=60):
        for st, s, d, *_ in ss.items:
            if s < t_min:
                continue
            area = float(st.dens.sum())
            inten = gain * np.clip(np.sqrt(area) / 220, 0.15, 1.6)
            ev.append((s, d, inten, area > heavy_w * 900))

    def from_glyph(g, t0, t1, gain=0.9):
        for a, b in g.spans:
            ev.append((t0 + a * (t1 - t0), (b - a) * (t1 - t0), gain, False))

    s = scenes.get('hook')
    ev.append((0.0, 0.5, 1.2, True))                                  # KANJI, brushed in the first frame
    from_strokes(s.ring, 1.2)
    from_glyph(s.g, s.t_ji, s.t_ji_end, 1.0)
    s = scenes.get('objects')
    for k, tt in s.times.items():
        ev.append((tt, 0.55, 0.8, True))
    from_strokes(s.div, 0.4)
    s = scenes.get('sun')
    from_glyph(s.big, s.t_write, s.t_write_end, 1.0)
    ev.append((s.t_early + 0.15, 0.8, 0.8, True))                     # the painted sun
    for tt in s.t_rows[1:]:
        ev.append((tt, 0.6, 0.7, False))
    from_strokes(s.arrows, 0.3)
    from_strokes(s.frame_red, 1.0)
    s = scenes.get('swirl')
    from_strokes(s.ss, 1.3)
    s = scenes.get('nature')
    for I in s.items:                                                 # painting → pictograph → kanji, in place
        ev.append((I['t'], 0.45 * I['slow'], 0.8, True))
        from_glyph(I['kan'], I['t'] + 1.05 * I['slow'], I['t'] + 1.4 * I['slow'], 0.9)
    s = scenes.get('compound')
    from_glyph(s.g, s.t_write, s.t_write_end, 1.0)
    from_strokes(s.arrows, 0.4)
    R = s.rest
    ev.append((R['t'], 0.45, 0.8, True))
    from_glyph(R['kan'], R['t'] + 1.05, R['t'] + 1.4, 0.9)
    for R in s.rows:
        from_glyph(R['gr'], R['t'] + 0.45, R['t'] + 0.8, 0.8)
        ev.append((R['t'], 0.25, 0.5, False))
    from_strokes(s.ops, 0.3)
    s = scenes.get('scholar')
    ev.append((s.t_kan, 0.5, 1.0, True))
    ev.append((s.t_ji, 0.5, 1.0, True))
    s = scenes.get('message')
    from_glyph(s.g, s.t_write, s.t_write_end, 1.0)
    s = scenes.get('endcard')
    from_strokes(s.es, 1.2)
    from_strokes(s.banner, 1.3)
    ev.append((s.t0 + 0.6, 0.5, 0.9, True))
    ev.append((s.t0 + 1.0, 0.5, 0.9, True))
    for L in s.ja:
        ev.append((L['t0'], L['t1'] - L['t0'], 0.8, False))
    return ev


# ------------------------------------------------------------------ build the mix
def build(out_dir=os.path.join(HERE, 'audio')):
    import scenes
    music = np.zeros((N, 2))
    music_dips = []
    sfx = np.zeros((N, 2))
    amb = np.zeros((N, 2))
    T0 = {nm: a0 for nm, a0, _ in SCENES}

    # drone pad: chord colour per scene, swelling with the story
    chords = {
        'hook': ([('D', 2), ('A', 2), ('D', 3)], 0.6), 'record': ([('D', 2), ('A', 2)], 0.5),
        'oracle': ([('D', 2), ('Eb', 3), ('A', 3)], 0.6), 'objects': ([('G', 2), ('D', 3), ('A', 3)], 0.55),
        'sun': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.65), 'swirl': ([('Bb', 1), ('D', 3), ('A', 3)], 0.6),
        'nature': ([('G', 2), ('D', 3), ('Bb', 3)], 0.6), 'compound': ([('D', 2), ('A', 2), ('G', 3)], 0.55),
        'map': ([('D', 2), ('A', 2), ('Eb', 3), ('Bb', 3)], 0.85), 'scholar': ([('G', 2), ('D', 3), ('A', 3)], 0.6),
        'message': ([('D', 2), ('A', 2), ('Bb', 3), ('D', 4)], 0.85), 'endcard': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.7)}
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
    phrase(T0['record'] + 0.4, [('D', 4), ('Eb', 4), ('A', 4)], gap=0.45, gain=0.55)
    phrase(T0['oracle'] + 0.5, [('A', 3), ('D', 4), ('Eb', 4), ('D', 4)], gap=0.5, gain=0.6, pan=-0.2)
    so = scenes.get('objects')
    for (k, tt), n in zip(so.times.items(), [('D', 5), ('A', 4), ('G', 4), ('D', 4)]):
        phrase(tt, [n], gain=0.5)
    phrase(T0['sun'] + 0.4, [('A', 4), ('D', 5)], gap=0.4, gain=0.55)
    ss_ = scenes.get('sun')
    for tt, n in zip(ss_.t_rows, [('D', 4), ('Eb', 4), ('G', 4), ('A', 4), ('D', 5)]):
        phrase(tt, [n], gain=0.45)
    phrase(T0['swirl'] + 0.3, [('Bb', 3), ('D', 4), ('Eb', 4), ('G', 4), ('A', 4)], gap=0.16, gain=0.55, pan=-0.3)
    sn = scenes.get('nature')
    for I, n in zip(sn.items, [('A', 4), ('G', 4), ('D', 4), ('D', 5)]):
        phrase(I['t'] + 1.05 * I['slow'], [n], gain=0.5)
    sc = scenes.get('compound')
    phrase(T0['compound'] + 0.4, [('G', 4), ('A', 4)], gap=0.5, gain=0.5)
    phrase(sc.rest['t'] + 1.05, [('A', 4)], gain=0.5)
    for R, n in zip(sc.rows, [('D', 5), ('A', 4)]):
        phrase(R['t'] + 0.45, [n], gain=0.45)
    phrase(T0['map'] + 0.4, [('D', 4), ('Eb', 4), ('G', 4), ('A', 4), ('Bb', 4), ('D', 5)], gap=0.26, gain=0.6)
    sm = scenes.get('map')
    phrase(sm.leg_t[2][0], [('A', 4), ('D', 5)], gap=0.35, gain=0.5)
    phrase(sm.leg_t[4][1], [('D', 5), ('A', 5)], gap=0.2, gain=0.5)
    phrase(T0['scholar'] + 0.4, [('G', 4), ('D', 5)], gap=0.5, gain=0.5)
    phrase(T0['message'] + 0.4, [('D', 4), ('A', 4), ('Bb', 4), ('D', 5), ('Eb', 5), ('D', 5)], gap=0.5, gain=0.55)
    phrase(T0['endcard'] + 0.5, [('D', 5), ('A', 4), ('G', 4), ('D', 4)], gap=0.3, gain=0.5)
    phrase(JA['matane'] - 0.05, [('D', 3), ('A', 3), ('D', 4), ('A', 4)], gap=0.06, gain=0.5)
    place(music, flute(note('A', 4), 4.0), T0['oracle'] + 1.2, 0.45, pan=-0.2)
    place(music, flute(note('D', 5), 3.4), T0['message'] + 1.0, 0.45, pan=0.2)

    # accents, each in a breath of the narration
    place(music, taiko(2.4, 52), scenes.get('hook').t_hit, 0.9)
    place(music, bowl(note('A', 4)), ss_.t_hit, 0.75, pan=0.2)
    place(music, bowl(note('D', 5)), sn.t_hit, 0.5, pan=-0.2)
    place(music, bowl(note('A', 5), 4.0), sc.t_hit, 0.5, pan=0.25)
    place(music, taiko(2.0, 70), T0['map'] + 0.15, 0.3)
    place(music, bowl(note('D', 5), 5.0), scenes.get('message').t_hit, 0.75)
    place(music, taiko(2.2, 60), scenes.get('message').t_hit, 0.45)        # the emotional peak gets a soft low drum
    place(music, bowl(note('D', 5), 5.0), JA['matane'], 0.45)

    # each bowl hit is struck with a soft mallet, and the bed breathes in for it: a short dip just before the
    # strike, so the strike is the one onset in the breath
    hits = [ss_.t_hit, sn.t_hit, sc.t_hit, scenes.get('message').t_hit, JA['matane']]
    for th in hits:
        n = int(0.025 * SR)
        place(music, hp(rng.normal(0, 1, n), 2500) * np.exp(-np.arange(n) / SR * 180) * 0.5, th, 0.8)
        i0, i1, i2 = int((th - 0.2) * SR), int((th - 0.015) * SR), int((th + 0.25) * SR)
        g = np.ones(i2 - i0)
        g[:i1 - i0] = 1 - 0.7 * np.sin(np.linspace(0, np.pi / 2, i1 - i0)) ** 2
        g[i1 - i0:] = np.minimum(1, 0.3 + 0.7 * np.linspace(0, 1, i2 - i1) ** 0.5)
        dip = np.ones(N)
        dip[i0:i2] = g
        music_dips.append(dip)

    # brush on paper
    for t, d, inten, heavy in scene_events():
        place(sfx, brush(d, inten, heavy), t, 1.0, pan=rng.uniform(-0.25, 0.25))
    for name, a, b in SCENES[1:]:
        place(sfx, whoosh(1.1, 250, 2600), a - 0.1, 0.8, pan=rng.uniform(-0.3, 0.3))
    # the painted cliff: a little fire, and the ochre brush on stone for each picture
    sr_ = scenes.get('record')
    a, b = sr_.t0, sr_.t1
    n = int((b - a) * SR)
    crack = np.zeros(n)
    for _ in range(int((b - a) * 6)):
        i = int(rng.uniform(0, b - a) * SR)
        k = int(rng.uniform(0.002, 0.008) * SR)
        if i + k < n:
            crack[i:i + k] += rng.normal(0, 1, k) * np.exp(-np.arange(k) / k * 5) * rng.uniform(0.2, 1)
    e = np.clip(np.minimum(np.arange(n) / SR / 1.0, (n - np.arange(n)) / SR / 0.8), 0, 1)
    place(amb, (hp(crack, 1500) * 0.05 + lp(rng.normal(0, 1, n), 400) * 0.03) * e, a)
    for Dw in sr_.drawings:
        place(sfx, brush(0.45, 0.7, False), Dw['t'], 1.0, pan=0.2)
    # oracle bones: low fire, and a blade on bone as each pictograph is carved
    so3 = scenes.get('oracle')
    a, b = so3.t0, so3.t1
    n = int((b - a) * SR)
    crack = np.zeros(n)
    for _ in range(int((b - a) * 7)):
        i = int(rng.uniform(0, b - a) * SR)
        k = int(rng.uniform(0.002, 0.008) * SR)
        if i + k < n:
            crack[i:i + k] += rng.normal(0, 1, k) * np.exp(-np.arange(k) / k * 5) * rng.uniform(0.2, 1)
    fire = hp(crack, 1500) * 0.05 + lp(rng.normal(0, 1, n), 300) * 0.02
    e = np.clip(np.minimum(np.arange(n) / SR / 1.2, (n - np.arange(n)) / SR / 1.0), 0, 1)
    place(amb, fire * e, a)
    for T in so3.tiles:
        place(sfx, carve(0.45), T['t'] + 0.2, 1.0, pan=rng.uniform(-0.3, 0.3))
    for I in sn.items:                                                # the pictograph traced over each painting
        place(sfx, carve(0.4 * I['slow']) * 0.6, I['t'] + 0.35 * I['slow'], 0.8)
    place(sfx, carve(0.4) * 0.6, sc.rest['t'] + 0.35, 0.8)
    # the voyage: swell and wash under the boat legs
    for k in (2, 3):
        a, b = sm.leg_t[k][0] - 0.3, sm.leg_t[k][1] + 0.4
        n = int((b - a) * SR)
        br = np.cumsum(rng.normal(0, 1, n))
        br = hp(br - np.convolve(br, np.ones(4000) / 4000, mode='same'), 40)
        br = br / (np.abs(br).max() + 1e-9)
        wash = bp(rng.normal(0, 1, n), 300, 2500) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * np.arange(n) / SR)) ** 3
        e = np.clip(np.minimum(np.arange(n) / SR / 0.6, (n - np.arange(n)) / SR / 0.6), 0, 1)
        place(amb, (lp(br, 600) * 0.45 + wash * 0.035) * e, a)
    # the message: a soft wind and a far wind chime
    a, b = T0['message'], T0['endcard'] + 1
    n = int((b - a) * SR)
    e = np.clip(np.minimum(np.arange(n) / SR / 1.2, (n - np.arange(n)) / SR / 1.0), 0, 1)
    place(amb, lp(rng.normal(0, 1, n), 700) * 0.03 * e, a)
    for tt, f in ((a + 1.0, note('A', 5) * 2), (a + 3.4, note('D', 6) * 2)):
        place(amb, bowl(f, 3.0) * 0.5, tt, 0.6, pan=0.4)

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
    # duck music and ambience under the voice
    envv = np.abs(voice[:, 0])
    k = int(0.25 * SR)
    envv = np.convolve(envv, np.ones(k) / k, mode='same')
    envv = envv / (envv.max() + 1e-9)
    # fast attack, slow release (~0.8 s): the music eases back between sentences instead of jumping up, so the
    # only onset in a breath is the hit placed there
    hop = SR // 100
    e = np.clip(envv[::hop] * 4, 0, 1)
    out = np.zeros_like(e)
    for i in range(1, len(e)):
        out[i] = e[i] if e[i] > out[i - 1] else out[i - 1] * 0.975 + e[i] * 0.025
    envv = np.interp(np.arange(N), np.arange(len(out)) * hop, out) / 4
    duck = 1 - 0.5 * np.clip(envv * 4, 0, 1)
    music *= duck[:, None]
    for dip in music_dips:
        music *= dip[:, None]
    amb *= (1 - 0.3 * np.clip(envv * 4, 0, 1))[:, None]

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

"""audio — the soundtrack, synthesized in code and synced to the animation's own events.

Score: a quiet drone in the miyako-bushi scale (D Eb G A Bb) with Karplus-Strong koto plucks at scene beats,
singing-bowl chimes for the hiragana / katakana reveals and low taiko impacts for the two title reveals.
Sound design: brush-on-paper texture generated from every stroke and glyph the scenes write (timed from the scene
objects themselves), fire and wind in the village, ocean swell and water for the voyage, paper whooshes at
transitions, a rainy night street under the photographs of today's Japan.  The music ducks under the narration.

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

    s1 = scenes.get('hook')
    ev.append((s1.t_title, 0.55, 0.7, False))
    for r in s1.rows:
        ev.append((r['t'], 0.26, 1.0, True))                     # each script stamps onto the page
    from_strokes(s1.q, 1.4)
    s2 = scenes.get('before_writing')
    from_strokes(s2.ss, 0.45, t_min=s2.t0 + 0.5)
    from_strokes(s2.lines, 0.35)
    s4 = scenes.get('kanji')
    from_glyph(s4.g, s4.t_write, s4.t_write_end, 1.0)
    for ts in (s4.t_lbl, s4.t_mean, s4.t_sound):
        ev.append((ts, 0.5, 0.45, False))
    s5 = scenes.get('hiragana')
    for i in range(5):
        ev.append((s5.tA + 0.32 * i, 0.4, 0.5, False))
    from_glyph(s5.an, s5.tB, s5.tB_end, 1.0)
    from_strokes(s5.cursive, 1.1)
    from_glyph(s5.a, s5.tD, s5.tD + 1.5, 0.9)
    from_strokes(s5.arrows, 0.5)
    s6 = scenes.get('katakana')
    for k in range(6):
        ev.append((s6.t_gloss + 0.32 * k, 0.25, 0.3, False))
    from_glyph(s6.i_k, s6.tB, s6.tB_end, 1.0)
    from_strokes(s6.ring, 0.7)
    ev.append((s6.t_kana, 0.6, 0.8, False))
    for g, m, kn, ts in s6.pairs:
        ev.append((ts, 0.35, 0.6, False))
    s7 = scenes.get('triptych')
    for ts in s7.t_in:
        ev.append((ts, 0.9, 0.9, True))
    for L in s7.labels:
        ev.append((L['tc'], 0.4, 0.6, False))
    s8 = scenes.get('sentence')
    for g in s8.groups:
        ev.append((g['ts'], g['dur'], 0.7, False))
    from_strokes(s8.ul_k, 0.6)
    from_strokes(s8.ul_h, 0.6)
    s11 = scenes.get('endcard')
    for L in s11.ja:
        ev.append((L['t0'], L['t1'] - L['t0'], 0.8, False))
    from_strokes(s11.ul, 0.6)
    return ev


# ------------------------------------------------------------------ build the mix
def build(out_dir=os.path.join(HERE, 'audio')):
    music = np.zeros((N, 2))
    sfx = np.zeros((N, 2))
    amb = np.zeros((N, 2))

    # drone pad: chord colour per section, swelling with the story
    chords = {  # scene -> drone notes, level (the colour follows the story)
        'hook': ([('D', 2), ('A', 2)], 0.5), 'before_writing': ([('D', 2), ('A', 2), ('D', 3)], 0.6),
        'map': ([('D', 2), ('A', 2), ('Eb', 3), ('Bb', 3)], 0.85), 'kanji': ([('G', 2), ('D', 3), ('Bb', 3)], 0.6),
        'hiragana': ([('D', 2), ('A', 2), ('G', 3), ('A', 3)], 0.65), 'katakana': ([('Bb', 1), ('D', 3), ('A', 3)], 0.6),
        'triptych': ([('D', 2), ('A', 2), ('D', 3), ('Eb', 4)], 0.65), 'sentence': ([('G', 2), ('D', 3)], 0.45),
        'street': ([('D', 2), ('A', 2), ('Bb', 3), ('D', 4)], 0.85),
        'modern': ([('D', 2), ('A', 2), ('D', 3), ('A', 3), ('D', 4)], 0.9),
        'endcard': ([('D', 2), ('A', 2), ('D', 3), ('A', 3)], 0.7)}
    sections = [(a0, b0, *chords[nm]) for nm, a0, b0 in SCENES]
    T0 = {nm: a0 for nm, a0, _ in SCENES}
    for a, b, notes, lvl in sections:
        dur = b - a + 1.5
        sig = sum(pad_voice(note(n, o), dur, bright=900 + 400 * lvl) for n, o in notes) / len(notes) ** 0.5
        e = np.minimum(1, np.minimum(np.arange(len(sig)) / (1.4 * SR), (len(sig) - np.arange(len(sig))) / (1.6 * SR)))
        e = np.clip(e, 0, 1) ** 1.5
        place(music, sig * e * lvl * 0.5, a, pan=-0.15)
        place(music, np.roll(sig, 900) * e * lvl * 0.5, a, pan=0.15)

    # koto phrases at the beats
    def phrase(t0, notes, gap=0.32, gain=0.9, pan=0.2):
        for k, (n, o) in enumerate(notes):
            place(music, koto(note(n, o)), t0 + k * gap, gain, pan=pan + 0.1 * np.sin(k))

    phrase(0.25, [('D', 4)], gain=0.7)
    import scenes
    s1_ = scenes.get('hook')
    for k, (r, n) in enumerate(zip(s1_.rows, [('A', 4), ('D', 5), ('A', 5)])):
        phrase(r['t'], [n], gain=0.6)
    phrase(T0['before_writing'] + 0.6, [('D', 4), ('A', 4), ('Bb', 4), ('A', 4)], gap=0.45, gain=0.7)
    phrase(at('conversation')[0] - 0.1, [('G', 4), ('A', 4)], gap=0.6, gain=0.45, pan=-0.3)
    phrase(T0['map'] + 0.5, [('D', 4), ('Eb', 4), ('G', 4), ('A', 4), ('Bb', 4), ('D', 5)], gap=0.24, gain=0.7)
    phrase(at('Japanese islands')[0], [('A', 4), ('D', 5), ('A', 5)], gap=0.3, gain=0.6)
    phrase(T0['kanji'] + 0.4, [('G', 3), ('D', 4)], gap=0.5, gain=0.6)
    phrase(T0['hiragana'] + 0.4, [('A', 4), ('G', 4), ('Eb', 4), ('D', 4)], gap=0.38, gain=0.55)
    phrase(at('came hiragana')[0], [('D', 5), ('A', 5)], gap=0.25, gain=0.55)
    phrase(T0['katakana'] + 0.4, [('Bb', 3), ('D', 4), ('Eb', 4)], gap=0.4, gain=0.55, pan=-0.2)
    phrase(at('Sharp')[0], [('A', 4)], gain=0.5)
    phrase(at('compact')[0], [('D', 5)], gain=0.5)
    phrase(at('precise')[0], [('A', 5)], gain=0.5)
    phrase(T0['triptych'] + 0.3, [('D', 4), ('G', 4), ('A', 4), ('D', 5)], gap=0.45, gain=0.6)
    phrase(T0['sentence'] + 0.5, [('A', 4), ('D', 5)], gap=0.5, gain=0.45)
    phrase(T0['street'] + 0.4, [('D', 4), ('A', 4), ('Bb', 4), ('D', 5), ('Eb', 5), ('D', 5)], gap=0.5, gain=0.6)
    phrase(T0['modern'] + 0.3, [('D', 4), ('A', 4), ('D', 5), ('Eb', 5), ('D', 5), ('A', 4)], gap=0.4, gain=0.55)
    phrase(T0['endcard'] + 0.6, [('D', 5), ('A', 4), ('G', 4), ('D', 4)], gap=0.3, gain=0.5)
    phrase(JA['matane'] - 0.05, [('D', 3), ('A', 3), ('D', 4), ('A', 4)], gap=0.06, gain=0.5)
    place(music, flute(note('A', 4), 4.0), T0['before_writing'] + 1.9, 0.5, pan=-0.2)
    place(music, flute(note('D', 5), 3.4), T0['street'] + 1.6, 0.45, pan=0.2)

    # accents
    import scenes
    s1 = scenes.get('hook')
    place(music, taiko(2.4, 52), s1.t_why + 0.02, 0.9)
    place(music, taiko(2.0, 70), T0['map'] + 0.15, 0.35)
    s4 = scenes.get('kanji')
    place(music, bowl(note('A', 4)), s4.t_write_end + 0.05, 0.5, pan=0.2)
    s5 = scenes.get('hiragana')
    place(music, bowl(note('D', 5)), s5.tD + 1.5, 0.6, pan=-0.2)
    s6 = scenes.get('katakana')
    place(music, bowl(note('A', 5), 4.0), s6.t_kana + 0.75, 0.5, pan=0.25)
    s10 = scenes.get('modern')
    place(music, taiko(2.6, 50), s10.labels[0]['t'], 0.75)
    place(music, bowl(note('D', 5), 5.0), JA['matane'], 0.45)

    # brush on paper
    for t, d, inten, heavy in scene_events():
        place(sfx, brush(d, inten, heavy), t, 1.0, pan=rng.uniform(-0.25, 0.25))
    # transitions: paper whooshes
    for name, a, b in SCENES[1:]:
        place(sfx, whoosh(1.1, 250, 2600), a - 0.1, 0.8, pan=rng.uniform(-0.3, 0.3))

    # village: wind + fire crackle
    s2 = scenes.get('before_writing')
    a, b = s2.t0, s2.t1
    n = int((b - a) * SR)
    wind = lp(rng.normal(0, 1, n), 500) * (0.6 + 0.4 * np.sin(np.linspace(0, 7, n)))
    crack = np.zeros(n)
    t_fire = at('living spoken')[0] - a
    for _ in range(int((b - a) * 9)):
        i = int(rng.uniform(t_fire, b - a) * SR)
        k = int(rng.uniform(0.002, 0.008) * SR)
        if i + k < n:
            crack[i:i + k] += rng.normal(0, 1, k) * np.exp(-np.arange(k) / k * 5) * rng.uniform(0.2, 1)
    crack = hp(crack, 1500) + lp(rng.normal(0, 1, n), 300) * 0.25 * (np.arange(n) > t_fire * SR)
    e = np.clip(np.minimum(np.arange(n) / SR / 1.5, (n - np.arange(n)) / SR / 1.5), 0, 1)
    place(amb, (wind * 0.05 + crack * 0.06) * e, a)
    # voyage: ocean swell, wash, water at the hull
    s3 = scenes.get('map')
    a, b = s3.t0, s3.t1
    n = int((b - a) * SR)
    br = np.cumsum(rng.normal(0, 1, n))
    br = hp(br - np.convolve(br, np.ones(4000) / 4000, mode='same'), 40)
    br = br / (np.abs(br).max() + 1e-9)
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.12 * np.arange(n) / SR)
    wash = bp(rng.normal(0, 1, n), 300, 2500) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.21 * np.arange(n) / SR)) ** 3
    e = np.clip(np.minimum(np.arange(n) / SR / 1.2, (n - np.arange(n)) / SR / 1.0), 0, 1)
    place(amb, (lp(br, 600) * 0.5 * swell + wash * 0.035) * e, a)
    for k in range(10):
        place(amb, bp(rng.normal(0, 1, int(0.4 * SR)), 500, 3000) * np.hanning(int(0.4 * SR)) * 0.03,
              s3.t_ship0 + 0.4 + k * 0.6, 1.0, pan=0.3)
    # street: soft wind and a wind chime
    s9 = scenes.get('street')
    a, b = s9.t0, s9.t1
    n = int((b - a) * SR)
    e = np.clip(np.minimum(np.arange(n) / SR / 1.2, (n - np.arange(n)) / SR / 1.0), 0, 1)
    place(amb, lp(rng.normal(0, 1, n), 700) * 0.035 * e, a)
    for tt, f in ((a + 1.2, note('A', 5) * 2), (a + 3.6, note('D', 6) * 2)):
        place(amb, bowl(f, 3.0) * 0.6, tt, 0.6, pan=0.4)
    # today's Japan: a soft rainy night street under the photographs
    a, b = s10.t0, s10.t1
    n = int((b - a) * SR)
    e = np.clip(np.minimum(np.arange(n) / SR / 0.6, (n - np.arange(n)) / SR / 0.5), 0, 1)
    rain = hp(rng.normal(0, 1, n), 2500) * 0.02 + lp(rng.normal(0, 1, n), 400) * 0.06
    place(amb, rain * e, a)
    place(sfx, whoosh(1.4, 150, 2200) * 1.5, s10.t0 - 0.05, 1.0)

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
    duck = 1 - 0.5 * np.clip(envv * 4, 0, 1)
    music *= duck[:, None]
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

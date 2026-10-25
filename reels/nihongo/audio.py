"""The soundtrack, synthesised in numpy: the voiceover placed on the timeline, a score in the miyako-bushi
scale (koto plucks by Karplus-Strong, a drone pad, a breathy shakuhachi-like flute, taiko, a temple bell),
brush-on-paper sounds generated from the real stroke timings of every scene, ocean / wind / fire ambience,
and the music ducked under the voice."""
import json
import os

import numpy as np
from scipy import signal

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.path.join(HERE, 'assets', 'voice')

# D miyako-bushi (in) scale: D Eb G A Bb, over a few octaves
def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


D3, EB3, G3, A3, BB3 = 50, 51, 55, 57, 58
SCALE = [D3, EB3, G3, A3, BB3]


def note(i, octave=0):
    """scale degree i (can exceed 5) -> midi"""
    return SCALE[i % 5] + 12 * (i // 5 + octave)


# ------------------------------------------------------------------ instruments
def karplus(freq, dur, decay=0.996, bright=0.6, seed=0):
    """plucked string (koto-like): Karplus-Strong, computed a period at a time"""
    rng = np.random.default_rng(seed)
    N = max(2, int(SR / freq))
    n = int(dur * SR)
    burst = rng.uniform(-1, 1, N)
    burst = signal.lfilter([bright, 1 - bright], [1], burst)             # pluck position / brightness
    blocks = int(np.ceil(n / N)) + 1
    y = np.zeros(blocks * N)
    y[:N] = burst
    for k in range(1, blocks):
        prev = y[(k - 1) * N:k * N]
        prev1 = np.concatenate([[y[(k - 1) * N - 1] if k > 1 else 0.0], prev[:-1]])
        y[k * N:(k + 1) * N] = decay * 0.5 * (prev + prev1)
    y = y[:n]
    env = np.minimum(1, np.arange(n) / (0.002 * SR))
    body = signal.lfilter(*signal.butter(2, [180 / (SR / 2), 3200 / (SR / 2)], 'band'), y)
    return (0.6 * y + 0.8 * body) * env


def flute(freq, dur, seed=0, vib=5.2):
    """breathy end-blown flute: a sine with a meri slide into the note, growing vibrato and breath noise"""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    slide = -0.03 * np.exp(-t / 0.18)                                   # starts a little flat
    vdepth = 0.006 * np.clip((t - 0.4) / 1.0, 0, 1)
    f = freq * (1 + slide + vdepth * np.sin(2 * np.pi * vib * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = signal.lfilter(*signal.butter(2, [freq * 0.8 / (SR / 2), min(0.95, freq * 5 / (SR / 2))], 'band'),
                            rng.normal(0, 1, n))
    env = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.6, 0, 1)
    return (tone * 0.5 + breath * 0.35 * (0.6 + 0.4 * np.exp(-t / 0.3))) * env


def pad(freqs, dur, seed=0):
    """a soft drone: detuned saws, low-passed, slowly breathing"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        for d in (-0.15, 0.0, 0.17):
            ph = (f * (1 + d / 100) * t + 0.13 * i) % 1.0
            out += (2 * ph - 1) / (len(freqs) * 3)
    out = signal.lfilter(*signal.butter(2, 700 / (SR / 2)), out)
    return out * (0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t + seed))


def taiko(dur=1.4, f0=118, f1=52, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.045)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.42)
    hit = signal.lfilter(*signal.butter(2, 900 / (SR / 2)), rng.normal(0, 1, n)) * np.exp(-t / 0.02)
    return body + 0.5 * hit


def impact(dur=3.5, seed=0):
    """low cinematic boom with a long dark tail"""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 34 + 40 * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    rumble = signal.lfilter(*signal.butter(2, 180 / (SR / 2)), rng.normal(0, 1, n)) * np.exp(-t / 1.2) * 0.8
    return boom + rumble


def bell(freq=659.0, dur=7.0):
    """a small temple bowl / rin: inharmonic partials with long decays and a slight beat"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for ratio, amp, dec in ((1, 1.0, 5.5), (2.71, 0.5, 3.0), (5.15, 0.25, 1.4), (8.5, 0.12, 0.7)):
        out += amp * np.sin(2 * np.pi * freq * ratio * t) * (1 + 0.15 * np.sin(2 * np.pi * 1.3 * t)) * np.exp(-t / dec)
    return out * np.clip(t / 0.004, 0, 1)


def clap(seed=0):
    """hyoshigi: two hardwood sticks struck together — a crisp, ringing click"""
    rng = np.random.default_rng(seed)
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    click = rng.normal(0, 1, n) * np.exp(-t / 0.004)
    ring = np.sin(2 * np.pi * 1850 * t) * np.exp(-t / 0.05) + 0.6 * np.sin(2 * np.pi * 2730 * t) * np.exp(-t / 0.03)
    return signal.lfilter(*signal.butter(2, [900 / (SR / 2), 6000 / (SR / 2)], 'band'), click) + 0.5 * ring


def brush_sound(dur, width, seed=0):
    """bristles dragged over washi: band-passed noise with a rough, grainy texture, louder for wider strokes"""
    rng = np.random.default_rng(seed)
    dur = max(0.06, dur)
    n = int((dur + 0.08) * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n)
    x = signal.lfilter(*signal.butter(2, [1200 / (SR / 2), 7500 / (SR / 2)], 'band'), x)
    grain = signal.lfilter(*signal.butter(1, 40 / (SR / 2)), rng.normal(0, 1, n))
    grain = 0.55 + 0.45 * np.tanh(grain * 6)
    env = np.clip(t / 0.012, 0, 1) * np.clip((dur + 0.06 - t) / 0.06, 0, 1)
    env *= 0.75 + 0.25 * np.sin(np.pi * np.clip(t / dur, 0, 1))
    return x * grain * env * np.clip(width / 30, 0.25, 1.6)


def swish(dur=0.7, seed=0):
    """paper / ink wash moving across the frame"""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n)
    out = np.zeros(n)
    for k in range(8):
        a, b = k * n // 8, (k + 1) * n // 8
        f = 600 + 3000 * (k / 8)
        out[a:b] = signal.lfilter(*signal.butter(2, [f / (SR / 2), min(0.9, f * 2.5 / (SR / 2))], 'band'), x[a:b])
    return out * np.sin(np.pi * t / dur) ** 2


def brown(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n))
    x = signal.lfilter(*signal.butter(1, 20 / (SR / 2), 'high'), x)
    return x / (np.abs(x).max() + 1e-9)


# ------------------------------------------------------------------ mixing helpers
class Bus:
    def __init__(self, total):
        self.x = np.zeros(int(total * SR) + SR)

    def add(self, t, sig, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i < 0:
            sig, i = sig[-i:], 0
        j = min(len(self.x), i + len(sig))
        if j > i:
            self.x[i:j] += sig[:j - i] * gain
        return self


def reverb(x, secs=2.2, wet=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, n) * np.exp(-t / (secs / 5))
    ir = signal.lfilter(*signal.butter(1, 4000 / (SR / 2)), ir)
    ir /= np.sqrt((ir ** 2).sum())
    return x * (1 - wet) + wet * signal.fftconvolve(x, ir)[:len(x)]


def db(v):
    return 10 ** (v / 20)


# ------------------------------------------------------------------ the soundtrack
def build(tl, scenes, out_path):
    import soundfile as sf
    total = tl.total
    rng = np.random.default_rng(1)
    st = {n: tl.scenes[n]['start'] for n in tl.scenes}

    # ---- voice
    voice = Bus(total)
    for ln in tl.lines:
        x, sr = sf.read(os.path.join(VOICE, f'line_{ln["i"]:02d}.wav'))
        x = signal.resample_poly(x, SR, sr)
        x = signal.lfilter(*signal.butter(2, 75 / (SR / 2), 'high'), x)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * db(-19)
        voice.add(ln['start'], x)
    v = reverb(voice.x, 0.8, 0.07, seed=2)
    v = np.tanh(v * 1.6) / 1.6                       # gentle limiting
    env = np.abs(voice.x)
    env = signal.lfilter(*signal.butter(1, 4 / (SR / 2)), env)
    env = np.clip(env / (env.max() + 1e-9) * 3.0, 0, 1)
    duck = 1 - 0.72 * env          # the bed drops well under the voice whenever it speaks

    # ---- music
    mus = Bus(total)
    # drone, section by section (root D, colour notes from the scale)
    sections = [(0.0, st[3], [hz(38), hz(45), hz(50)], -30), (st[3], st[4], [hz(38), hz(45), hz(50), hz(57)], -27),
                (st[4], st[7], [hz(38), hz(45), hz(51)], -27), (st[7], st[8], [hz(38), hz(45), hz(50), hz(55)], -29),
                (st[8], st[9], [hz(38), hz(45), hz(50)], -29), (st[9], total, [hz(38), hz(45), hz(50), hz(57)], -27)]
    for k, (a, b, fr, g) in enumerate(sections):
        p = pad(fr, b - a + 2.5, seed=k)
        n = len(p)
        fade = np.minimum(1, np.minimum(np.arange(n) / (1.5 * SR), (n - np.arange(n)) / (2.0 * SR)))
        mus.add(a - 0.8, p * fade, db(g))

    def phrase(t0, degrees, step, octave=1, gain=-20, seed=0, jitter=0.02, dur=2.6):
        for i, d in enumerate(degrees):
            if d is None:
                continue
            tt = t0 + i * step + rng.normal(0, jitter)
            mus.add(tt, karplus(hz(note(d, octave)), dur, decay=0.9965, bright=0.55, seed=seed + i),
                    db(gain + rng.normal(0, 1.5)))

    # 1: title — a few plucks as the brush writes, low boom when it is done
    s1 = scenes[1]
    phrase(0.95, [0, None, 3, None, 4, 3], 0.42, octave=1, gain=-21, seed=10)
    mus.add(s1.impact, impact(), db(-12))
    mus.add(s1.impact, taiko(), db(-16))
    # 2: before writing — the flute, sparse koto
    for k, (d, t, ln) in enumerate([(3, 1.0, 2.6), (4, 3.8, 1.8), (3, 5.8, 3.2), (2, 9.6, 1.6), (3, 11.4, 2.4)]):
        mus.add(st[2] + t, flute(hz(note(d, 2)), ln, seed=20 + k), db(-27))
    phrase(st[2] + 2.0, [0, None, None, 2, None, None, 1, None, None, 0], 0.8, gain=-25, seed=30)
    # 3: the crossing — a pulse and a swell
    beat = 60 / 76
    pat = [0, 3, 4, 3, 5, 4, 3, 1]
    for b in range(int((st[4] - st[3]) / beat)):
        t = st[3] + 0.4 + b * beat
        g = -26 + 6 * min(1, b / 12)
        mus.add(t, karplus(hz(note(pat[b % 8], 1)), 1.8, decay=0.995, bright=0.6, seed=40 + b), db(g))
        if b % 4 == 0:
            mus.add(t, taiko(1.2, 100, 50, seed=b), db(-27 + 4 * min(1, b / 12)))
        if b % 2 == 1 and b > 6:
            mus.add(t + beat / 2, karplus(hz(note(pat[(b + 3) % 8], 2)), 1.2, decay=0.993, bright=0.7, seed=60 + b),
                    db(-31))
    mus.add(st[4] - 1.8, swish(2.4, seed=3), db(-30))
    # 4: kanji — quiet; a chord when 漢字 is finished
    s4 = scenes[4]
    for i, d in enumerate([0, 3, 5]):
        mus.add(st[4] + s4.kanji_end + 0.05 + 0.03 * i, karplus(hz(note(d, 1)), 3.0, seed=70 + i), db(-21))
    phrase(st[4] + 6.0, [3, None, None, None, 4, None, None, None, 3], 0.6, gain=-27, seed=80)
    # 5: hiragana — a flowing arpeggio, a high accent on あ
    s5 = scenes[5]
    phrase(st[5] + 2.2, [5, 6, 7, 8, 7, 6], 0.32, octave=0, gain=-25, seed=90)
    phrase(st[5] + 7.6, [5, 7, 8, 9], 0.38, octave=0, gain=-26, seed=95)
    mus.add(st[5] + s5.a_end, karplus(hz(note(10, 1)), 3.2, decay=0.997, bright=0.75, seed=99), db(-19))
    # 6: katakana — short, dry plucks; hyoshigi when イ appears
    s6 = scenes[6]
    for b in range(10):
        mus.add(st[6] + 1.0 + b * 1.1, karplus(hz(note([0, 3, 4, 3, 1][b % 5], 1)), 0.6, decay=0.985, bright=0.8,
                                               seed=110 + b), db(-27))
    mus.add(st[6] + s6.k_end - 0.2, clap(seed=1), db(-16))
    mus.add(st[6] + s6.i_end, karplus(hz(note(5, 1)), 2.2, seed=120), db(-23))
    # 7: three scripts — three notes, one for each, then a light pulse
    s7 = scenes[7]
    for k, (a, b) in enumerate(s7.focus):
        mus.add(st[7] + a + 0.3, karplus(hz(note([0, 2, 4][k], 1)), 2.8, seed=130 + k), db(-21))
    for b in range(10):
        mus.add(st[7] + 0.4 + b * beat, taiko(0.9, 140, 70, seed=200 + b), db(-34 + (3 if b % 4 == 0 else 0)))
    # 8: the sentence — the music relaxes
    phrase(st[8] + 1.5, [0, None, None, None, None, 3, None, None, None, None, 2], 0.7, gain=-29, seed=140)
    # 9: past and present — the flute rises, the koto climbs
    for k, (d, t, ln) in enumerate([(4, 0.4, 2.2), (5, 2.7, 1.6), (6, 4.4, 2.6)]):
        mus.add(st[9] + t, flute(hz(note(d, 2)), ln, seed=150 + k), db(-25))
    phrase(st[9] + 0.6, [0, 2, 3, 4, 5, 7, 8, 9], 0.7, gain=-25, seed=160)
    # 10: the wave — taiko and a boom as it breaks, the bell with the handle, the drone resolves
    s10 = scenes[10]
    for k, t in enumerate([0.25, 0.85, 1.25, 1.55]):
        mus.add(st[10] + t, taiko(1.6, 120, 48, seed=300 + k), db(-14 + (2 if k == 0 else 0)))
    mus.add(st[10] + 1.0, impact(4.5, seed=4), db(-13))
    mus.add(st[10] + s10.t_handle + 0.2, bell(659.0, 7.0), db(-20))
    mus.add(st[10] + s10.t_handle + 0.2, bell(329.6, 7.0), db(-28))
    phrase(st[10] + 2.6, [0, 3, 5], 0.5, octave=1, gain=-22, seed=170, dur=4.0)

    music = reverb(mus.x, 3.0, 0.3, seed=5)

    # ---- effects and ambience
    fx = Bus(total)
    for n, sc in scenes.items():
        for x in getattr(sc, 'track', None).items if getattr(sc, 'track', None) else []:
            dur = x.t1 - x.t0
            if dur < 0.03 or getattr(x, 'red', False) and dur < 0.1:
                continue
            w = float(np.mean(x.stroke.dens.shape)) / 6
            if n == 3:            # the route dashes are drawn by the ship, not a brush
                continue
            fx.add(st[n] + x.t0, brush_sound(dur, w, seed=int(x.t0 * 1000) % 100000), db(-31))
        for tr in getattr(sc, 'tracks', []):
            for x in tr.items:
                fx.add(st[n] + x.t0, brush_sound(x.t1 - x.t0, 20, seed=int(x.t0 * 997) % 99991), db(-33))
    # transitions: a soft swish at each scene change
    for n in range(2, 11):
        fx.add(st[n] - tl.scenes[n]['trans'] / 2, swish(max(0.6, tl.scenes[n]['trans']), seed=n), db(-29))
    # ocean on the map (and faintly under the wave)
    for a, b, g in [(st[3] - 0.6, st[4] + 0.6, -25), (st[10], total, -31)]:
        n = int((b - a) * SR)
        t = np.arange(n) / SR
        o = brown(n, seed=int(a))
        o = signal.lfilter(*signal.butter(2, 650 / (SR / 2)), o)
        surf = signal.lfilter(*signal.butter(2, [400 / (SR / 2), 2500 / (SR / 2)], 'band'), rng.normal(0, 1, n))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t) ** 2
        sig = (o * 2.2 + surf * 0.12 * swell) * np.minimum(1, np.minimum(t / 1.0, (t[-1] - t) / 1.0))
        fx.add(a, sig / (np.abs(sig).max() + 1e-9), db(g))
    # wind over the settlement and the street
    for a, b in [(st[2] - 0.6, st[3] + 0.5), (st[9] - 0.5, st[10] + 0.5)]:
        n = int((b - a) * SR)
        t = np.arange(n) / SR
        wnd = signal.lfilter(*signal.butter(2, [250 / (SR / 2), 1100 / (SR / 2)], 'band'), rng.normal(0, 1, n))
        wnd *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.09 * t + 1)) * np.minimum(1, np.minimum(t / 1.2, (t[-1] - t) / 1.2))
        fx.add(a, wnd / (np.abs(wnd).max() + 1e-9), db(-33))
    # the fire crackles
    t0, t1 = st[2] + 3.6, st[3] + 0.3
    for _ in range(int((t1 - t0) * 9)):
        tt = rng.uniform(t0, t1)
        n = int(rng.uniform(0.003, 0.02) * SR)
        c = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (n / 4))
        c = signal.lfilter(*signal.butter(2, [800 / (SR / 2), 6000 / (SR / 2)], 'band'), c)
        fx.add(tt, c, db(-30 + rng.normal(0, 3)))

    fxs = reverb(fx.x, 1.2, 0.12, seed=7)
    bed = music * duck * 1.0 + fxs * (0.45 + 0.55 * duck) * 0.9
    mix = v + bed
    speaking = env > 0.25
    if speaking.any():
        rms = lambda a: 20 * np.log10(np.sqrt(np.mean(a[speaking[:len(a)]] ** 2)) + 1e-12)
        print(f'voice {rms(v):.1f} dB, music+fx under the voice {rms(bed):.1f} dB')
    # fade in / out
    n = len(mix)
    t = np.arange(n) / SR
    mix *= np.clip(t / 0.3, 0, 1) * np.clip((total + 0.2 - t) / 1.6, 0, 1)
    mix = mix[:int(total * SR)]
    mix = mix / (np.abs(mix).max() + 1e-9) * db(-1.5)
    # a little stereo width for the music/ambience (voice stays centred)
    side = signal.lfilter([1], [1], (music * duck * 0.9 + fxs * 0.5)[:len(mix)])
    side = np.roll(side, int(0.011 * SR)) - side
    side = side / (np.abs(side).max() + 1e-9) * np.abs(mix).max() * 0.12
    stereo = np.stack([mix + side, mix - side], 1)
    stereo = stereo / (np.abs(stereo).max() + 1e-9) * db(-1.0)
    sf.write(out_path, stereo.astype(np.float32), SR, subtype='FLOAT')
    return out_path

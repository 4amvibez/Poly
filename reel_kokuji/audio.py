"""audio — the soundtrack of the kokuji reel: the user's own track (audio/bgm.mp3) as the music bed, with light sound
design synced to the animation's own events.

Music: the track's opening runs under the reel and crossfades, beat-matched, into the song's real ending, so the
last bar lands as the reel ends.
Sound design: brush-on-paper under every kanji the scenes write, an ink-stamp thud (and a soft taiko) as each part
slams in and on each build, a shove on "pushes", mountain wind under 峠, a low fire crackle under 火, paper whooshes
at the transitions, paper ticks as the eight kokuji cards are laid down.
The voice always leads: the music ducks deeply under every word and the effects duck with it, so nothing is ever as
loud as Raj.

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

from timeline import DURATION, SCENES, VO_OFFSET

N = int(DURATION * SR)
MUSIC_GAIN = 0.27                 # the user's track is mastered loud (-14.5 LUFS): this sits it well under Raj
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


def wind(n, seed=0):
    """mountain wind: band-passed noise with a slow swell"""
    r = np.random.default_rng(seed)
    x = bp(r.normal(0, 1, n), 200, 1400)
    t = np.arange(n) / SR
    return x * (0.5 + 0.5 * np.sin(2 * np.pi * t / 3.1) ** 2) * 0.25


def crackle(n, seed=0):
    """a low wood-fire crackle: sparse clicks over a soft roar"""
    r = np.random.default_rng(seed)
    x = lp(r.normal(0, 1, n), 600) * 0.15
    for _ in range(int(n / SR * 14)):
        i = int(r.uniform(0, n - 400))
        x[i:i + 300] += hp(r.normal(0, 1, 300), 2000) * np.exp(-np.arange(300) / 40) * r.uniform(0.2, 0.7)
    return x


# ------------------------------------------------------------------ events from the scenes
def bgm(join=43.0, end_at=136.4, xf=1.2):
    """the user's track (audio/bgm.mp3) cut to the reel: its own opening up to `join` seconds, then a beat-matched
    crossfade into its real ending, placed so the song's last bar ends with the reel"""
    import subprocess
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(HERE, 'audio', 'bgm.mp3'), '-ac', '2', '-ar', str(SR),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    tail0 = end_at - (DURATION - join)                       # where the ending section starts in the song

    def flux(a, n):
        m = x[int(a * SR):int(a * SR) + n, 0]
        h = SR // 200
        e = np.sqrt(np.convolve(m ** 2, np.ones(h) / h, 'same')[::h])
        return np.maximum(np.diff(e), 0)
    ref = flux(join - 4, 4 * SR)                             # line the beats up: the tail's onsets on the head's
    best = max(np.arange(-0.5, 0.5, 0.005), key=lambda d: np.dot(ref, flux(tail0 + d - 4, 4 * SR)))
    tail0 += best
    out = np.zeros((N, 2))
    nh = int((join + xf / 2) * SR)
    out[:nh] = x[:nh]
    seg = x[int((tail0 - xf / 2) * SR):int((tail0 - xf / 2) * SR) + N - int((join - xf / 2) * SR)]
    out[int((join - xf / 2) * SR):int((join - xf / 2) * SR) + len(seg)] += seg * np.clip(
        np.arange(len(seg)) / (xf * SR), 0, 1)[:, None] ** 0.5
    a = int((join - xf / 2) * SR)
    out[a:nh] *= 1.0                                         # head fades out over the crossfade (equal power)
    out[a:nh] = x[a:nh] * np.cos(np.linspace(0, np.pi / 2, nh - a))[:, None] + out[a:nh] - x[a:nh]
    return out


def build(out_dir=os.path.join(HERE, 'audio')):
    import scenes
    g = scenes.get
    sfx, amb = np.zeros((N, 2)), np.zeros((N, 2))
    music = bgm()

    def hit(t, gain=0.6, drum=True):
        place(sfx, stamp(), t, gain)
        if drum:
            place(sfx, taiko(1.4, 54), t, 0.35)

    hk = g('hook')
    place(sfx, brush(0.3, 1.4, True), hk.t_nin, 0.6)
    hit(hk.t_nin, 0.5)
    hit(hk.t_dou, 0.55)
    place(sfx, brush(0.3, 1.0, True), hk.t_cross, 0.45)
    place(sfx, whoosh(0.5, 500, 4000), hk.t_join, 0.45)
    place(sfx, brush(0.4, 1.4, True), hk.t_lock - 0.1, 0.6)
    hit(hk.t_lock, 0.4, drum=False)
    hit(hk.t_hit, 0.75)
    sc = g('secret')
    for k in range(2):
        place(sfx, brush(0.45, 1.3, True), sc.t_kokuji - 0.55 + 0.25 * k, 0.55)
    hit(sc.t_kokuji + 0.1, 0.45, drum=False)
    wk = g('work')
    place(sfx, brush(0.5, 1.4, True), wk.t_first - 0.15, 0.6)
    place(sfx, whoosh(0.8, 300, 2500), wk.t_first + 0.2, 0.3)
    place(sfx, brush(0.45, 1.2, True), wk.t_add + 0.2, 0.55)
    place(sfx, whoosh(0.5, 300, 3000), wk.t_push, 0.5)
    place(sfx, thud(), wk.t_push + 0.3, 0.5)
    place(sfx, whoosh(0.45, 500, 4000), wk.t_comb, 0.4)
    place(sfx, whoosh(0.45, 500, 4000), wk.t_get - 0.05, 0.45)
    place(sfx, brush(0.55, 1.4, True), wk.t_get + 0.3, 0.6)
    hit(wk.t_get + 0.3, 0.5)
    for nm in ('pass', 'field'):
        s = g(nm)
        for p in s._parts:
            place(sfx, brush(0.3, 1.0, True), p['t'] - 0.1, 0.45)
        place(sfx, whoosh(0.45, 500, 4000), s.t_merge - 0.05, 0.45)
        place(sfx, brush(0.55, 1.4, True), s.t_merge + 0.3, 0.6)
        hit(s.t_merge + 0.3, 0.45)
    ps = g('pass')
    place(amb, wind(int((ps.t1 - ps.t0) * SR), seed=3), ps.t0, 0.9)
    fd = g('field')
    place(amb, crackle(int((fd.t_merge - fd.t_fire + 0.5) * SR), seed=4), fd.t_fire, 0.7)
    mo = g('more')
    hit(mo.t_over, 0.45, drum=False)
    for k in range(8):
        place(sfx, flip() * 0.6, mo.t_few + 0.3 * k, 0.5, pan=0.12 * (k % 4 - 1.5))
    fn = g('final')
    place(sfx, brush(0.5, 1.2, True), fn.t_follow - 0.05, 0.5)
    hit(fn.t_hit, 0.6)
    for name, a, b in SCENES[1:]:
        place(sfx, whoosh(1.1, 250, 2600), a - 0.1, 0.5, pan=rng.uniform(-0.3, 0.3))
    # narration
    sr, vo = wavfile.read(os.path.join(out_dir, 'narration.wav'))
    vo = vo.astype(np.float64)
    if vo.ndim > 1:
        vo = vo.mean(1)
    vo /= 32768.0 if np.abs(vo).max() > 2 else 1.0
    voice = np.zeros((N, 2))
    place(voice, vo, VO_OFFSET, 1.0)
    # duck everything under the voice: fast attack, slow release, deep enough that Raj always leads
    envv = np.abs(voice[:, 0])
    k = int(0.2 * SR)
    envv = np.convolve(envv, np.ones(k) / k, mode='same')
    envv = envv / (envv.max() + 1e-9)
    hop = SR // 100
    e = np.clip(envv[::hop] * 5, 0, 1)
    out = np.zeros_like(e)
    for i in range(1, len(e)):
        out[i] = e[i] if e[i] > out[i - 1] else out[i - 1] * 0.97 + e[i] * 0.03
    duck = np.interp(np.arange(N), np.arange(len(out)) * hop, out)
    music *= (1 - 0.8 * duck)[:, None]
    amb *= (1 - 0.55 * duck)[:, None]
    sfx *= (1 - 0.65 * duck)[:, None]
    stems = {'voice': voice, 'music': music * MUSIC_GAIN, 'sfx': sfx * 0.4, 'amb': amb * 0.3}
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

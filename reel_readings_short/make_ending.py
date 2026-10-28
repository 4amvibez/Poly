"""make_ending — the closing shot of the merged reel, rendered on its own (it is not on this cut's timeline):
"Thank you for watching till the end!" (Raj) then 「またね！」 (Raj in Japanese).
The ink of the page washes into the final landscape: a mountain under a rising vermilion sun, a river running down
from it, the straw-hat traveller beside the water.  THANK YOU FOR WATCHING / TILL THE END! is set at the top as Raj
says it, word by word; on またね！ a large vermilion またね！ is stamped across the sky (taiko, bowl); the black banner
FOLLOW @raj_in_jp sweeps in under it and holds.  Writes output/ending.mp4 (video + its own mix) and ending.wav."""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kit  # noqa: E402
import r5art  # noqa: E402
from engine import (FONT_BRUSH_EN, FONT_BRUSH_JP, FONT_EN, FONT_EN_BOLD, FONT_HEAD, FPS, H, INK, SEA, VERM, W, Cam, Frame,  # noqa: E402
                    StrokeSet, cached, ease_io, make_paper, noise, smooth, stroke, text_layer, wash_map)
from glyphs import typeset  # noqa: E402

FINAL = os.path.join(HERE, '..', 'reel_readings_final')
DUR = 6.2
T_THANKS, T_MATANE = 0.35, 2.85
IVORY = np.array([0.98, 0.95, 0.88], np.float32)
WORDS = [('THANK', 0.0), ('YOU', 0.36), ('FOR', 0.48), ('WATCHING', 0.64)]


class Ending:
    def __init__(self):
        self.paper = make_paper(W, H, 109, stains=0.08)
        self.F = cached('s07_final_v1', r5art.final_landscape)
        self.T = wash_map(H, W, (540, 300), noise_amp=0.25, noise_scale=220, seed=12)
        self.top = [text_layer(w, FONT_HEAD, 52)[0] for w, _ in WORDS]
        self.till, _ = text_layer('TILL THE END!', FONT_BRUSH_EN, 110)
        self.till = self.till * (0.85 + 0.15 * noise(*self.till.shape, 6, 2, 11))
        self.mata = [typeset(c, 190, font=FONT_BRUSH_JP, seed=180 + k, medians=False, sweep=(0.7, 1)) for k, c in enumerate('またね！')]
        self.follow, _ = text_layer('FOLLOW', FONT_EN_BOLD, 40, tracking=10)
        self.handle, _ = text_layer('@raj_in_jp', FONT_BRUSH_EN, 84)
        bn = StrokeSet(W, H)
        bn.add(stroke(W, H, [(250, 1336), (540, 1324), (830, 1332)], 118, ink=0.96, dry=0.45,
                      profile=((0, 0.55), (0.06, 1.0), (0.85, 0.95), (1, 0.5)), seed=171), T_MATANE + 0.5, 0.4)
        self.banner = bn
        ws = [w for w, _ in WORDS]
        widths = [t.shape[1] for t in self.top]
        gap = 18
        x = 540 - (sum(widths) + gap * (len(ws) - 1)) / 2
        self.xs = []
        for wd in widths:
            self.xs.append(x + wd / 2)
            x += wd + gap

    def render(self, t):
        z = 1.0 + 0.03 * smooth(t / DUR) + 0.035 * np.exp(-max(t - T_MATANE, 0) * 6) * (t > T_MATANE)
        cam = Cam(540, 960 - 15 * smooth(t / DUR), z)
        from engine import warp
        fr = Frame(warp(self.paper, cam.matrix(), border='reflect'))
        m = np.clip((ease_io(t / 1.0) * 1.12 - 0.04 - self.T) / 0.06, 0, 1).astype(np.float32)
        rise = 120 * (1 - ease_io(t / 2.5))
        kit.put(fr, cam, self.F['sun'] * m, 540, 960 + rise, VERM, 0.85)
        fr.ink(warp(self.F['water'] * m, cam.matrix()), SEA, 0.45)
        fr.ink(warp(self.F['ink'] * m, cam.matrix()), INK, 1.0)
        for k, (g, (_, tw)) in enumerate(zip(self.top, WORDS)):
            kit.put(fr, cam, g, self.xs[k], 230, INK, smooth((t - T_THANKS - tw + 0.05) / 0.15))
        ft = (t - T_THANKS - 1.04) / 0.2
        if ft > 0:
            kit.put(fr, cam, self.till, 540, 345, VERM, smooth(ft), 1 + 0.25 * np.exp(-ft * 2.5), rot=-0.03)
        for k, g in enumerate(self.mata):
            fk = (t - T_MATANE - 0.07 * k) / 0.2
            if fk > 0:
                kit.put(fr, cam, g.at(min(fk, 1.05), soft=0.06), 540 + (k - 1.5) * 190, 720, VERM, 1.0, 1 + 0.25 * np.exp(-fk * 2.5))
        fb = (t - T_MATANE - 0.5) / 0.4
        if fb > 0:
            fr.ink(warp(self.banner.render(t), cam.matrix()), INK, 0.96)
            kit.put(fr, cam, self.handle, 540, 1324, IVORY, smooth((t - T_MATANE - 0.65) / 0.3), 1 + 0.1 * np.exp(-fb * 2.5), over=True)
            kit.put(fr, cam, self.follow, 540, 1238, INK, smooth(fb))
        import compose
        return Frame(compose.drift(fr.img, t + 100)).finish(t)


def audio():
    """the ending's own mix: the two voice lines, a pad, a koto phrase, taiko and a bowl on またね！"""
    import audio as A
    SR = A.SR
    n = int(DUR * SR)
    music, sfx, voice = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))

    def load(name):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(FINAL, 'audio', name), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).astype(np.float64)

    th, ma = load('thanks_a.mp3'), load('matane_a.mp3')
    lv = lambda x: np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
    ref = 0.11
    A.place(voice, th / lv(th) * ref, T_THANKS, 1.0)
    A.place(voice, ma / lv(ma) * ref, T_MATANE, 1.0)
    sig = sum(A.pad_voice(A.note(nn, o), DUR + 1, bright=1300) for nn, o in [('D', 2), ('A', 2), ('D', 3), ('A', 3)]) / 2
    e = np.clip(np.minimum(np.arange(len(sig)) / (0.6 * SR), (len(sig) - np.arange(len(sig))) / (1.6 * SR)), 0, 1) ** 1.5
    A.place(music, sig * e * 0.4, 0.0)
    for k, (nn, o) in enumerate([('D', 4), ('A', 4), ('D', 5), ('Eb', 5), ('D', 5)]):
        A.place(music, A.koto(A.note(nn, o)), 0.2 + 0.32 * k, 0.5, pan=0.1 * np.sin(k))
    A.place(music, A.taiko(2.2, 56), T_MATANE, 0.9)
    A.place(sfx, A.stamp(), T_MATANE, 1.0)
    A.place(music, A.bowl(A.note('D', 5), 5.0), T_MATANE + 0.05, 0.6)
    A.place(sfx, A.brush(0.4, 1.0, True), T_MATANE + 0.45, 1.0)
    A.place(sfx, A.whoosh(1.0, 250, 2600), 0.0, 0.7)
    mix = voice + music * 0.21 + sfx * 0.65
    fade = np.ones(n)
    fade[-int(1.2 * SR):] = np.linspace(1, 0, int(1.2 * SR)) ** 1.5
    mix = np.tanh(mix * fade[:, None] * 1.1) / 1.1
    from scipy.io import wavfile
    path = os.path.join(FINAL, 'audio', 'ending.wav')
    wavfile.write(path, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    return path


def main():
    end = Ending()
    wav = audio()
    out = os.path.join(FINAL, 'segments', 'D_ending.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-i', wav, '-c:v', 'libx264', '-crf', '14', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
                          '-shortest', out.replace('.mp4', '.mov')], stdin=subprocess.PIPE)
    for k in range(int(DUR * FPS)):
        p.stdin.write(end.render(k / FPS).tobytes())
    p.stdin.close()
    p.wait()
    print('wrote', out.replace('.mp4', '.mov'))


if __name__ == '__main__':
    main()

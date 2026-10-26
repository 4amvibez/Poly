"""qa — checks on the finished MP4 (the animation-studio check / verify / qa steps, for this pipeline):
  * container: 1080x1920, 30 fps, H.264 + AAC, duration
  * every frame decoded and compared with the one before: unexplained jumps (pops, broken transitions) are listed
  * loudness (integrated LUFS, true peak) of the delivered audio
  * sync: at each marker in score.js the audio must have an onset within +-40 ms
  * subtitle plates and on-screen captions inside the Reels safe area (top 11%, bottom 22%, right buttons)
  * a contact sheet of the delivered file
    python qa.py output/japanese_writing_reel.mp4"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'stream=codec_type,codec_name,width,height,r_frame_rate:format=duration', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def frame_diffs(path, w=135, h=240):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                         stdout=subprocess.PIPE)
    prev, diffs = None, []
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            break
        f = np.frombuffer(b, np.uint8).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(f - prev).mean()))
        prev = f
    return np.array(diffs)


def audio_onsets(path, sr=16000):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = 160                                            # 10 ms
    e = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, 'same')[::hop] + 1e-12)
    flux = np.maximum(0, np.diff(20 * np.log10(e + 1e-6)))
    return flux, hop / sr


def main(path):
    import subtitles
    from engine import SAFE_BOTTOM, SAFE_LEFT, SAFE_RIGHT, SAFE_TOP
    ok = True
    info = probe(path)
    v = next(s for s in info['streams'] if s['codec_type'] == 'video')
    a = next(s for s in info['streams'] if s['codec_type'] == 'audio')
    dur = float(info['format']['duration'])
    print(f"container: {v['codec_name']} {v['width']}x{v['height']} @ {v['r_frame_rate']}, audio {a['codec_name']}, {dur:.2f}s")
    if (v['width'], v['height'], v['r_frame_rate']) != (1080, 1920, '30/1') or not 85 <= dur <= 100:
        print('  ✗ spec mismatch'); ok = False
    else:
        print('  ✓ 9:16 1080x1920, 30 fps, 85–100 s')

    score = json.loads(open(os.path.join(HERE, 'score.js')).read().split('return ', 1)[1].rsplit(';', 2)[0])
    cuts = score.get('cuts') or [[a0, 1.2] for _, (a0, b) in list(score['S'].items())[1:]]
    d = frame_diffs(path)
    t = (np.arange(len(d)) + 1) / 30
    med = np.median(d)
    spikes = [(float(tt), float(dd)) for tt, dd in zip(t, d) if dd > max(12, med * 8)]
    print(f'frames: {len(d) + 1} decoded, median change {med:.2f}/255 per frame')
    bad = [s for s in spikes if not any(c - 0.05 <= s[0] <= c + w for c, w in cuts)]
    if bad:
        ok = False
        print('  ✗ unexplained jumps:', ', '.join(f'{s[0]:.2f}s ({s[1]:.0f})' for s in bad))
    else:
        print(f'  ✓ no unexplained jumps ({len(spikes)} large changes, all inside scene transitions)')

    ld = subprocess.run(['ffmpeg', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
    I = [l for l in ld.splitlines() if l.strip().startswith('I:')][-1].split()[1]
    P = [l for l in ld.splitlines() if l.strip().startswith('Peak:')][-1].split()[1]
    print(f'loudness: {I} LUFS integrated, true peak {P} dBFS')
    if not (-15.5 <= float(I) <= -12.5 and float(P) <= -1.0):
        print('  ✗ loudness outside -14 ±1.5 LUFS / -1 dBTP'); ok = False
    else:
        print('  ✓ -14 LUFS target, peak under -1 dBTP')

    flux, step = audio_onsets(path)
    print('sync markers (audio onset vs picture event):')
    for name, m in score['markers'].items():
        i0, i1 = int((m['t'] - 0.1) / step), int((m['t'] + 0.1) / step)
        k = i0 + int(np.argmax(flux[i0:i1]))
        off = (k * step - m['t']) * 1000
        good = abs(off) <= 40
        ok &= good
        print(f"  {'✓' if good else '✗'} {name:12s} @ {m['t']:7.3f}s  onset {off:+5.0f} ms")

    from engine import H
    bad_subs = []
    for k, c in enumerate(subtitles.CUES):
        al, _ = subtitles._render(k)
        h, w = al.shape
        x0, x1 = subtitles.CX - w / 2, subtitles.CX + w / 2
        y0, y1 = subtitles.CY - h / 2, subtitles.CY + h / 2
        if x0 < SAFE_LEFT or x1 > SAFE_RIGHT or y0 < SAFE_TOP or y1 > SAFE_BOTTOM:
            bad_subs.append(' '.join(wd for wd, _ in c['words']))
    if bad_subs:
        ok = False
        print('  ✗ captions outside the safe area:', bad_subs)
    else:
        print(f'captions: ✓ all {len(subtitles.CUES)} inside the safe area (x {SAFE_LEFT}–{SAFE_RIGHT}, y {SAFE_TOP}–{SAFE_BOTTOM})')

    sheet = os.path.join(os.path.dirname(path), 'contact_sheet.jpg')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', path, '-vf', f'fps=30/{dur / 30 * 30:.3f},scale=216:384,tile=10x3',
                    '-frames:v', '1', '-q:v', '3', sheet], check=True)
    print('contact sheet:', sheet)
    print('RESULT:', 'PASS' if ok else 'ISSUES FOUND')
    return ok


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'output', 'raj_in_jp_spaces_reel.mp4'))

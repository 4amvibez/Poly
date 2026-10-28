"""assemble — the merged reel: the short's hook, the full reel's body, the short's tip + comment question, the ending.

    A  reel_readings_short   0.00 ->  6.40   ink-burst hook (日曜日 puzzle, WHY?)
    B  reel_readings         8.50 -> 90.35   history map, 山 日, on'yomi / kun'yomi, kokuji ... "Every kanji has a story."
    C  reel_readings_short  42.95 -> 57.00   QUICK TIP, 今日？ COMMENT BELOW, FOLLOW
    D  ending (make_ending.py)  6.20 s       THANK YOU FOR WATCHING TILL THE END!  またね！

Each join is a 0.3 s dissolve.  The audio crossfades keep whichever side is speaking at full level (the side in
silence is the one faded), then the whole mix is normalised to -14 LUFS.

    python assemble.py   # segments/*.mp4 + the projects' mixes -> output/raj_in_jp_readings_final.mp4
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, 'segments')
SHORT = os.path.join(HERE, '..', 'reel_readings_short', 'audio', 'mix_raw.wav')
FULL = os.path.join(HERE, '..', 'reel_readings', 'audio', 'mix_raw.wav')
XF = 0.3
# (video, audio source, audio start, duration, crossfade curves into the next part: (out, in))
PARTS = [('A_hook.mp4', SHORT, 0.0, 6.40, ('tri', 'nofade')),
         ('B_main.mp4', FULL, 8.50, 81.85, ('nofade', 'tri')),
         ('C_tip_cta.mp4', SHORT, 42.95, 14.05, ('tri', 'nofade')),
         ('D_ending.mov', os.path.join(HERE, 'audio', 'ending.wav'), 0.0, 6.20, None)]


def main():
    os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
    out = os.path.join(HERE, 'output', 'raj_in_jp_readings_final.mp4')
    args, fv, fa = [], [], []
    for k, (v, a, a0, d, _) in enumerate(PARTS):
        args += ['-i', os.path.join(SEG, v)]
    for k, (v, a, a0, d, _) in enumerate(PARTS):
        args += ['-ss', str(a0), '-t', str(d), '-i', a]
    n = len(PARTS)
    for k, (_, _, _, d, _) in enumerate(PARTS):
        fv.append(f'[{k}:v]trim=duration={d},setpts=PTS-STARTPTS,fps=30,format=yuv420p[v{k}]')
        fa.append(f'[{n + k}:a]atrim=duration={d},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[a{k}]')
    length, vl, al = PARTS[0][3], 'v0', 'a0'
    for k in range(1, n):
        c1, c2 = PARTS[k - 1][4]
        fv.append(f'[{vl}][v{k}]xfade=transition=fade:duration={XF}:offset={length - XF:.3f}[vx{k}]')
        fa.append(f'[{al}][a{k}]acrossfade=d={XF}:c1={c1}:c2={c2}[ax{k}]')
        length += PARTS[k][3] - XF
        vl, al = f'vx{k}', f'ax{k}'
    fa.append(f'[{al}]loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[aout]')
    graph = ';'.join(fv + fa)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *args, '-filter_complex', graph, '-map', f'[{vl}]', '-map', '[aout]',
                    '-c:v', 'libx264', '-crf', '19', '-preset', 'slow', '-profile:v', 'high', '-level', '4.2', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', out], check=True)
    print(f'wrote {out} ({length:.2f} s)')


if __name__ == '__main__':
    main()

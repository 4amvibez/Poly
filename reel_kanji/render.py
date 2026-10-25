"""render — the full reel: frames in parallel worker processes -> chunks -> concatenated -> muxed with the mix.

    python render.py                 # full film -> output/kanji_reel.mp4
    python render.py --start 20 --end 34 --out /tmp/x.mp4   # a section (for checking)
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import FPS, H, W  # noqa: E402
from timeline import DURATION  # noqa: E402


def worker(f0, f1, path):
    import compose
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k in range(f0, f1):
        p.stdin.write(compose.frame_at(k / FPS).tobytes())
        if (k - f0) % 60 == 0:
            done = k - f0 + 1
            print(f'[{os.path.basename(path)}] {done}/{f1 - f0} frames, {(time.time() - t0) / done:.2f}s/frame', flush=True)
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, default=DURATION)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'kanji_reel.mp4'))
    ap.add_argument('--no-audio', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = os.path.join(HERE, 'cache', 'chunks')
    os.makedirs(tmp, exist_ok=True)
    F0, F1 = int(round(a.start * FPS)), int(round(a.end * FPS))
    # split by scene boundaries where possible so each worker builds few scenes; balance by frame count
    n = a.jobs * 3
    edges = [F0 + (F1 - F0) * i // n for i in range(n + 1)]
    jobs = [(edges[i], edges[i + 1], os.path.join(tmp, f'chunk_{i:03d}.mp4')) for i in range(n)]
    t0 = time.time()
    running = []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < a.jobs:
            j = queue.pop(0)
            pr = Process(target=worker, args=j)
            pr.start()
            running.append(pr)
        for pr in running[:]:
            if not pr.is_alive():
                pr.join()
                if pr.exitcode != 0:
                    raise SystemExit(f'worker failed ({pr.exitcode})')
                running.remove(pr)
        time.sleep(0.5)
    lst = os.path.join(tmp, 'list.txt')
    with open(lst, 'w') as f:
        for _, _, p in jobs:
            f.write(f"file '{p}'\n")
    video = os.path.join(tmp, 'video.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', video], check=True)
    if a.no_audio:
        os.replace(video, a.out)
    else:
        mix = os.path.join(HERE, 'audio', 'mix_raw.wav')
        ss = ['-ss', str(a.start), '-t', str(a.end - a.start)]
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video, *ss, '-i', mix,
                        '-af', 'loudnorm=I=-14:TP=-1.5:LRA=9', '-ar', '48000',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', a.out], check=True)
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {a.out}')


if __name__ == '__main__':
    main()

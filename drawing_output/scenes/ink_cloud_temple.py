"""Ink wash (水墨): 云深不知处 — renders poly_ink.cloud_temple().
python3 ink_cloud_temple.py [out.png] [--stages DIR]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from poly_ink.__main__ import main

out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'ink_cloud_temple.png'
args = ['cloud-temple', out]
if '--stages' in sys.argv:
    args += ['--stages', sys.argv[sys.argv.index('--stages') + 1]]
main(args)

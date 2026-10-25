"""Draw-on animation from a --stages directory — wraps poly_ink.make_gif.
python3 make_gif.py STAGES_DIR out.gif
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from poly_ink import make_gif

n, secs = make_gif(sys.argv[1], sys.argv[2])
print('saved', sys.argv[2], n, 'frames', round(secs, 2), 's', round(os.path.getsize(sys.argv[2]) / 1e6, 2), 'MB')

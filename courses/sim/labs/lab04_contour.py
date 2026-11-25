"""Lab 4: contour lines with marching squares,
laid on the terrain inside Isaac Sim.

Run (env isaac):  python labs/lab04_contour.py --stream
Output:           out/contours.npz, out/lab04.usda
"""
from simlab import finish, new_stage, parse_args, save, start, THEME

args = parse_args(n=65)
app = start(args.stream)

import numpy as np

import look
import scenes
import vizcore as vc

stage = new_stage()
prims, view = scenes.lab04(args.n)     # terrain + contour rods
look.usd_scene(stage, prims, view, THEME)

saved = {}
for level, (segs, ambiguous) in scenes.lab04_segments(args.n).items():
    saved[f"level_{level:.1f}"] = segs
    print(f"level {level:.1f} m: {len(segs):4d} segments,"
          f" {ambiguous} ambiguous cells")
np.savez("out/contours.npz", **saved)
print("wrote out/contours.npz")

# A saddle, where the ambiguous cases live, has a known answer.
ambiguous, correct, *_ = vc.saddle_check()
print(f"saddle: {ambiguous} ambiguous cell,"
      f" resolved correctly: {correct}")
save("lab04.usda")
finish(app, args.stream)

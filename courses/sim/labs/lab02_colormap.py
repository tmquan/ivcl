"""Lab 2 (handbook 1.1.2, 1.2.1): a source -> filter -> sink
pipeline that colour-maps elevation, with two lookup tables.

Run (env isaac):  python labs/lab02_colormap.py --stream
Output:           out/lab02.usda
"""
from simlab import finish, new_stage, parse_args, save, show, start

args = parse_args(n=65)
app = start(args.stream)

import numpy as np

import scenes
import vizcore as vc

stage = new_stage()

# SOURCE and FILTER live in scenes.lab02(): a terrain, and its
# height pushed through two lookup tables.
prims, view = scenes.lab02(args.n)

# SINK: the coloured meshes go on the stage; the renderer draws.
show(stage, prims, view)

for p in prims:
    low = p["points"][:, 2].argmin()
    high = p["points"][:, 2].argmax()
    print(f"{p['name']:8s} colour at the lowest point"
          f" {p['colors'][low].round(2)}, at the highest"
          f" {p['colors'][high].round(2)}")
# How bright is each table from its low end to its high end?
s = np.linspace(0, 1, 101)
for name, lut in (("rainbow", vc.lut_rainbow),
                  ("viridis", vc.lut_viridis)):
    lum = vc.luminance(lut(s))
    turns = int((np.diff(np.sign(np.diff(lum))) != 0).sum())
    print(f"{name:8s} brightness changes direction {turns} times")
save("lab02.usda")
finish(app, args.stream)

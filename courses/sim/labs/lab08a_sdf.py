"""Lab 8, part A: implicit functions. Build a
part from three distance functions and contour it. Plain NumPy;
the simulator is not started.

Run (env isaac):  python labs/lab08a_sdf.py
"""
import numpy as np

import vizcore as vc

xs = np.linspace(-1.5, 1.5, 121)
part = vc.demo_part(xs)              # disc - hole - slot

for level, name in ((0.0, "surface"), (0.1, "10 cm offset")):
    segs, _ = vc.marching_squares(part, level, xs, xs)
    length = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1).sum()
    print(f"F = {level}: {len(segs)} contour segments,"
          f" {length:.2f} m long  ({name})")

cell = (xs[1] - xs[0]) ** 2
print(f"area inside (F < 0): {(part < 0).sum() * cell:.3f} m^2")
print("is the origin inside the part?", bool(part[60, 60] < 0),
      f"(F = {part[60, 60]:+.2f})")

# the part as text: # inside, . outside
for row in part[::3, ::6].T[::-1]:
    print("".join("#" if f < 0 else "." for f in row))

"""Step 6: check the simulated fall against the formula. Plain
NumPy; the simulator is not started.

Run (env isaac):  python labs/s6_check_fall.py
Needs:            out/fall.csv  (from Step 5)
"""
import os
import sys

import numpy as np

import scene

G, Z0, HALF = 9.81, scene.START[2], scene.SIZE / 2

if not os.path.exists("out/fall.csv"):
    sys.exit("missing out/fall.csv: run labs/s5_falling_cube.py first")
rows = np.loadtxt("out/fall.csv", delimiter=",", skiprows=1)
t, z = rows[:, 0], rows[:, 1]

exact = np.maximum(Z0 - 0.5 * G * t**2, HALF)
t_exact = np.sqrt(2 * (Z0 - HALF) / G)
landed = t[z <= HALF + 1e-3]             # within 1 mm of rest
falling = t < t_exact - 0.02             # samples before impact
dts = np.diff(t)

print(f"samples                 {len(t)}")
print(f"time step               mean {dts.mean():.5f} s,"
      f" min {dts.min():.5f}, max {dts.max():.5f}")
print(f"impact, formula         {t_exact:.4f} s")
print("impact, simulation      "
      + (f"{landed[0]:.4f} s" if len(landed) else "never landed"))
print(f"largest gap to formula  "
      f"{np.abs(z - exact)[falling].max() * 100:.2f} cm while falling")
print(f"rest height             {z[-1]:.4f} m"
      f" (half the cube = {HALF})")

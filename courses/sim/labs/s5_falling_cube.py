"""Step 5: make the cube fall. You only declare what is a rigid
body and what collides; the physics engine does the rest.

Run (env isaac):  python labs/s5_falling_cube.py
Output:           out/fall.csv, out/falling_cube.usda
"""
import os

import look
import scene

THEME = look.pick()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager

DT = 1 / 60     # physics time step, s
STEPS = 120     # 2 simulated seconds

stage_utils.create_new_stage()
stage = stage_utils.get_current_stage()
paths = look.usd_scene(stage, scene.build(), scene.VIEW, THEME)

# Physics is declared, not programmed:
body = RigidPrim(paths["cube"])                     # mass, gravity
shape = GeomPrim(paths["cube"], apply_collision_apis=True)
shape.set_collision_approximations("convexHull")    # cube collides
GeomPrim(paths["ground"], apply_collision_apis=True)  # ground too

os.makedirs("out", exist_ok=True)
stage_utils.save_stage(os.path.abspath("out/falling_cube.usda"))

SimulationManager.set_physics_dt(DT)   # set before pressing play
app_utils.play()
app.update()

pos, _ = body.get_world_poses()
rows = [(SimulationManager.get_simulation_time(),
         float(pos.numpy()[0, 2]))]
for _ in range(STEPS):
    SimulationManager.step()           # one physics step, no render
    pos, _ = body.get_world_poses()
    rows.append((SimulationManager.get_simulation_time(),
                 float(pos.numpy()[0, 2])))
rows = np.array(rows)
rows[:, 0] -= rows[0, 0]               # start the clock at zero

np.savetxt("out/fall.csv", rows, delimiter=",", header="t,z",
           comments="")
print(f"steps={STEPS} dt={np.diff(rows[:, 0]).mean():.4f}s")
print(f"first z={rows[0, 1]:.3f} m  final z={rows[-1, 1]:.3f} m"
      f"  (half the cube = {scene.SIZE/2})")
print("wrote out/fall.csv, out/falling_cube.usda")
app_utils.stop()
app.close()

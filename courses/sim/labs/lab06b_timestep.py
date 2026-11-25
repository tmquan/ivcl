"""Lab 6, part B: a physics engine is a numerical
integrator. Drop the same cube with three different time steps
and compare each run with the exact free-fall curve.

Run (env isaac):  python labs/lab06b_timestep.py
Output:           out/timestep_240.csv, _60.csv, _15.csv
"""
from simlab import (collide, finish, new_stage, parse_args, start,
                    THEME, to_np)

args = parse_args()
app = start(args.stream)

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager

import look
import scenes

Z0, SIZE, DURATION = (scenes.LAB06_Z0, scenes.LAB06_SIZE,
                      scenes.LAB06_DURATION)
table = []
for hz in scenes.LAB06_RATES:
    stage = new_stage()
    prims, view = scenes.lab06()           # ground + cube at 2 m
    paths = look.usd_scene(stage, prims, view, THEME)
    body = RigidPrim(paths["cube"])
    collide(paths["cube"], "convexHull")
    collide(paths["ground"])

    SimulationManager.set_physics_dt(1.0 / hz)   # before play
    app_utils.play()
    app.update()

    rows = []
    for _ in range(int(DURATION * hz)):
        SimulationManager.step()
        p, _quat = body.get_world_poses()
        rows.append((SimulationManager.get_simulation_time(),
                     float(to_np(p)[0, 2])))
    rows = np.array(rows)
    rows[:, 0] -= rows[0, 0] - 1.0 / hz   # first sample is t = dt
    np.savetxt(f"out/timestep_{hz}.csv", rows, delimiter=",",
               header="t,z", comments="")
    table.append((hz, np.diff(rows[:, 0]).mean(),
                  scenes.lab06_error(rows), rows[:, 1].min(),
                  rows[-1, 1]))
    app_utils.stop()
    app.update()

exact = np.sqrt(2 * (Z0 - SIZE / 2) / 9.81)
print(f"exact impact time {exact:.4f} s; rest height {SIZE / 2} m")
print("  Hz   mean dt    error at t=0.4 s   lowest z   final z")
for hz, dt, error, lowest, final in table:
    print(f" {hz:4d}  {dt:.5f}    {error:+.4f} m"
          f"          {lowest:.4f} m   {final:.4f} m")
finish(app, args.stream)

"""Lab 5: hedgehogs and pathlines.
Throw 25 cubes, record position and velocity at every step.

Run (env isaac):  python labs/lab05_vectors.py --stream
Output:           out/vectors.npz, out/lab05.usda
"""
from simlab import (collide, finish, new_stage, parse_args, save, start,
                    THEME, to_np)

args = parse_args()
app = start(args.stream)

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager

import look
import scenes

DT, STEPS, SNAPSHOT = (scenes.LAB05_DT, scenes.LAB05_STEPS,
                       scenes.LAB05_SNAPSHOT)

stage = new_stage()
prims, view = scenes.lab05_base()          # ground + 25 cubes
paths = look.usd_scene(stage, prims, view, THEME)
cubes = [paths[p["name"]] for p in prims if p["name"] != "ground"]
bodies = RigidPrim(cubes)                  # the cubes move ...
collide(cubes, "convexHull")               # ... and collide
collide(paths["ground"])                   # the ground only collides

SimulationManager.set_physics_dt(DT)
app_utils.play()
app.update()

# initial velocity field: a swirl around the Z axis, plus "up"
_start, v0 = scenes.lab05_start()
bodies.set_velocities(linear_velocities=v0,
                      angular_velocities=np.zeros_like(v0))

t, pos, vel = [], [], []
for _ in range(STEPS):
    SimulationManager.step()
    p, _quat = bodies.get_world_poses()
    v, _omega = bodies.get_velocities()
    t.append(SimulationManager.get_simulation_time())
    pos.append(to_np(p).copy())
    vel.append(to_np(v).copy())
t, pos, vel = np.array(t), np.array(pos), np.array(vel)
np.savez("out/vectors.npz", t=t, pos=pos, vel=vel,
         snapshot=SNAPSHOT)
speed = np.linalg.norm(vel, axis=2)
print(f"recorded {pos.shape[0]} steps x {pos.shape[1]} bodies")
print(f"speed range {speed.min():.2f} .. {speed.max():.2f} m/s")
print("wrote out/vectors.npz")
app_utils.stop()
app.update()

# PATHLINES and HEDGEHOG: thin rods added to the scene, so they
# appear in the stream and in the saved file.
look.usd_author(stage, scenes.lab05_overlay(pos, vel, SNAPSHOT))
save("lab05.usda")
finish(app, args.stream)

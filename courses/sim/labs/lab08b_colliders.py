"""Lab 8, part B: the collision
shape is a model of the mesh. Drop a ball onto a ring and change
the model.

Run (env isaac):  python labs/lab08b_colliders.py
Output:           out/colliders.csv, out/lab08.usda
"""
from simlab import (collide, finish, new_stage, parse_args, save, show,
                    start, to_np)

args = parse_args()
app = start(args.stream)

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager

import scenes

DT = 1 / 120
rows = []
for approximation in ("convexHull", "convexDecomposition", "sdf",
                      "none"):
    stage = new_stage()
    prims, view = scenes.lab08()             # ground, ring, ball
    paths = show(stage, prims, view)
    if not rows:
        save("lab08.usda")                   # the start of every run
    collide(paths["ring"], approximation)    # <- the model under test
    ball = RigidPrim(paths["ball"])
    collide(paths["ball"], "boundingSphere")
    collide(paths["ground"])

    SimulationManager.set_physics_dt(DT)
    app_utils.play()
    app.update()
    SimulationManager.step(steps=240)            # 2 seconds
    p, _quat = ball.get_world_poses()
    z = float(to_np(p)[0, 2])
    where = ("ON the ring" if z > scenes.LAB08_RING_Z
             else "through the hole")
    rows.append((approximation, z))
    print(f"{approximation:20s} ball rests at z = {z:.3f} m  ({where})")
    app_utils.stop()
    app.update()

with open("out/colliders.csv", "w") as f:
    f.write("approximation,ball_z\n")
    f.writelines(f"{a},{z:.4f}\n" for a, z in rows)
print("on the ground the ball centre would be at",
      scenes.LAB08_BALL_R, "m")
print("wrote out/colliders.csv")
finish(app, args.stream)

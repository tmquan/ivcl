"""Lab 10 (handbook 1.5.6): probing. A sensor samples the scene
along rays; how many rays decides what the robot can see.

Run (env isaac):  python labs/lab10_probe.py --stream
Output:           out/probe.npz, out/lab10.usda
"""
from simlab import (collide, finish, new_stage, parse_args, save, show,
                    start)

args = parse_args()
app = start(args.stream)

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import omni.physics.core
from isaacsim.core.simulation_manager import SimulationManager

import look
import scenes
import vizcore as vc

stage = new_stage()
prims, view = scenes.lab10_base()            # ground + obstacles
paths = show(stage, prims, view)
collide([paths[p["name"]] for p in prims if p["name"] != "ground"])

app_utils.play()
app.update()
SimulationManager.step()

query = omni.physics.core.get_physics_scene_query_interface()
measured = {}
for n, _rgb, _width, _lift in scenes.LAB10_FANS:
    directions = vc.ray_fan(n, scenes.LAB10_FOV)
    dist = np.full(n, np.inf)
    for i, d in enumerate(directions):
        ok, hit = query.raycast_closest(
            scenes.LAB10_ORIGIN, tuple(map(float, d)),
            scenes.LAB10_MAX, True)
        if ok:
            dist[i] = hit.distance
    measured[n] = dist
    reference = scenes.lab10_reference(n)
    both = np.isfinite(dist) & np.isfinite(reference)
    worst = np.abs(dist[both] - reference[both]).max() \
        if both.any() else float("nan")
    leg = vc.raycast_boxes_2d(scenes.LAB10_ORIGIN, directions,
                              scenes.LAB10_BOXES[2:])
    print(f"{n:3d} rays: {int(np.isfinite(dist).sum()):3d} hits,"
          f" thin leg seen by {int(np.isfinite(leg).sum())} rays,"
          f" max difference to reference {worst:.4f} m")

np.savez("out/probe.npz", **{f"dist_{n}": d
                             for n, d in measured.items()})
print("wrote out/probe.npz")
app_utils.stop()
app.update()

# The rays, as rods from the sensor to what each one hit.
look.usd_author(stage, scenes.lab10_rays(measured))
save("lab10.usda")
finish(app, args.stream)

"""Lab 7: the inertia of a rigid body is a tensor.
Read it from the physics engine and find its principal axes.

Run (env isaac):  python labs/lab07_inertia.py --stream
Output:           out/inertia.npz, out/lab07.usda
"""
from simlab import (collide, finish, new_stage, parse_args, save, start,
                    THEME, to_np)

args = parse_args()
app = start(args.stream)

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import RigidPrim

import look
import scenes
import vizcore as vc

stage = new_stage()
prims, view = scenes.lab07_base()            # ground + box
paths = look.usd_scene(stage, prims, view, THEME)
body = RigidPrim(paths["box"], masses=[scenes.LAB07_MASS])
collide(paths["box"], "convexHull")
collide(paths["ground"])

app_utils.play()
app.update()


def read_tensor():
    return to_np(body.get_inertias()).reshape(3, 3)


mass = float(to_np(body.get_masses()).ravel()[0])
measured = read_tensor()
expected = vc.box_inertia(mass, *scenes.LAB07_EDGES)
vals, vecs = vc.principal_axes(measured)
np.set_printoptions(precision=4, suppress=True)
print("mass (kg):", mass)
print("inertia tensor from the engine (kg m^2):\n", measured)
print("formula m/12 (b^2+c^2, a^2+c^2, a^2+b^2):\n", expected)
print("principal moments:", vals)
print("principal axes (columns):\n", vecs)

# Turn the box 30 degrees about Z and read the tensor again.
half = np.radians(30) / 2
body.set_world_poses(orientations=[np.cos(half), 0, 0, np.sin(half)])
app.update()
turned = read_tensor()
print("after a 30 degree turn about Z:\n", turned)
print("same numbers as before:", bool(np.allclose(measured, turned,
                                                   atol=1e-5)))
np.savez("out/inertia.npz", tensor=measured, turned=turned, mass=mass)
print("wrote out/inertia.npz")
app_utils.stop()
app.update()

# The principal axes, as three rods above the box.
look.usd_author(stage, scenes.lab07_axes(measured))
save("lab07.usda")
finish(app, args.stream)

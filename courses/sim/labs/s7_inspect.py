"""Step 7: what is an Isaac Sim scene made of?

Run (env isaac):  python labs/s7_inspect.py
Needs:            out/hello_cube.usda  (from Step 3)
"""
import os

import look

look.pick()   # --theme is accepted; the saved scene carries its theme

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim

ok, stage = stage_utils.open_stage(
    os.path.abspath("out/hello_cube.usda"))
assert ok, "run labs/s3_hello_cube.py first"

print(stage_utils.generate_stage_representation(mode="tree"))

cube = stage.GetPrimAtPath("/World/Look/cube")
print("type:", cube.GetTypeName())
print("applied schemas before:", list(cube.GetAppliedSchemas()))
for attr in cube.GetAttributes():
    value = str(attr.Get())
    if len(value) > 44:
        value = value[:44] + " ..."
    print(f"   {attr.GetName():26s} = {value}")

# Physics is extra data attached to the same prim.
RigidPrim("/World/Look/cube")
GeomPrim("/World/Look/cube", apply_collision_apis=True)
print("applied schemas after: ", list(cube.GetAppliedSchemas()))
print("anything about mass, velocity or collision?",
      any("physics" in a.GetName().lower()
          for a in cube.GetAttributes()))
app.close()

"""Step 3: build the scene on Isaac Sim's stage and save it.

Run (env isaac):  python labs/s3_hello_cube.py
                  python labs/s3_hello_cube.py --theme dark
Output:           out/hello_cube.usda
"""
import os

import look
import scene

THEME = look.pick()              # "light" or "dark"

from isaacsim import SimulationApp

# Start the simulator first. Every other isaacsim / pxr import
# must come after this line.
app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.stage as stage_utils

stage_utils.create_new_stage()
stage = stage_utils.get_current_stage()

prims = scene.build()            # ground + cube, from scene.py
look.usd_scene(stage, prims, scene.VIEW, THEME)

os.makedirs("out", exist_ok=True)
path = os.path.abspath("out/hello_cube.usda")
stage_utils.save_stage(path)
print(stage_utils.generate_stage_representation(mode="tree"))
print("wrote", path)
app.close()

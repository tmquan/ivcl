"""Step 4: save what the scene's camera sees as a PNG of
1280 x 720 pixels.

Run (env isaac):
  python labs/s4_shot.py out/hello_cube.usda out/hello_cube.png
"""
import os
import sys

import look

look.pick()
if len(sys.argv) != 3:
    sys.exit("usage: s4_shot.py <scene.usda> <picture.png>")
usd, png = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
if not os.path.exists(usd):
    sys.exit(f"no such file: {usd}")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "open_usd": usd,
                     "width": look.WIDTH, "height": look.HEIGHT})

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
app.update()

from isaacsim.sensors.experimental.rtx import CameraSensor

sensor = CameraSensor("/World/Camera",
                      resolution=(look.HEIGHT, look.WIDTH),
                      annotators=["rgb"])
app_utils.play(commit=True)
image = None
for frame in range(240):          # give the renderer time to settle
    app_utils.update_app()
    data, _info = sensor.get_data("rgb")
    if data is not None and frame >= 60:
        image = data.numpy()
        break
assert image is not None, "the camera returned no picture"
if image.dtype != np.uint8:
    image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

try:
    from PIL import Image
    Image.fromarray(image[..., :3]).save(png)
    print("wrote", png, image.shape)
except ImportError:
    np.save(png + ".npy", image)
    print("Pillow is not installed; wrote", png + ".npy")
app_utils.stop()
app.close()

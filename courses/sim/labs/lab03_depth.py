"""Lab 3: a simulated depth camera produces a
scalar per pixel. Capture it, then colour-map it four ways: two
tables, two ranges.

Run (env isaac):  python labs/lab03_depth.py
Output:           out/lab03.usda, out/depth.npy (metres, H x W),
                  out/depth_rainbow.png, out/depth_viridis.png,
                  out/depth_rainbow_auto.png,
                  out/depth_viridis_auto.png
"""
from simlab import (finish, new_stage, parse_args, save, show, start,
                    to_np)

args = parse_args()
app = start(args.stream)

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
app.update()

from isaacsim.sensors.experimental.rtx import CameraSensor

import look
import scenes
import vizcore as vc

stage = new_stage()
prims, view = scenes.lab03()
show(stage, prims, view)
save("lab03.usda")

# The depth sensor looks through the same camera as the pictures.
H, W = look.HEIGHT // 2, look.WIDTH // 2      # (height, width)
sensor = CameraSensor("/World/Camera", resolution=(H, W),
                      annotators=["distance_to_image_plane"])

app_utils.play(commit=True)
depth = None
for frame in range(120):             # the sensor needs a few frames
    app_utils.update_app()
    data, _info = sensor.get_data("distance_to_image_plane")
    if data is not None:
        depth = to_np(data).reshape(H, W)
        if frame > 10:
            break
assert depth is not None, "no depth data after 120 frames"

np.save("out/depth.npy", depth)
finite = depth[np.isfinite(depth)]
print(f"depth image {depth.shape}, {depth.dtype}")
print(f"nearest {finite.min():.2f} m, farthest {finite.max():.2f} m,"
      f" pixels with no hit: {int((~np.isfinite(depth)).sum())}")
print("wrote out/depth.npy")

# Colour mapping: two tables, each with one fixed range and with
# the range taken from this frame.
NEAR, FAR = 0.0, 8.0
lo, hi = float(finite.min()), float(finite.max())
pictures = {
    "depth_rainbow.png": vc.depth_to_rgb(depth, NEAR, FAR,
                                         vc.lut_rainbow),
    "depth_viridis.png": vc.depth_to_rgb(depth, NEAR, FAR),
    "depth_rainbow_auto.png": vc.depth_to_rgb(depth, lo, hi,
                                              vc.lut_rainbow),
    "depth_viridis_auto.png": vc.depth_to_rgb(depth, lo, hi),
}
try:
    from PIL import Image
    for name, rgb in pictures.items():
        Image.fromarray(rgb).save(f"out/{name}")
        print("wrote", f"out/{name}")
except ImportError:
    print("Pillow is not installed: no pictures written")
app_utils.stop()
finish(app, args.stream)

"""Step 4: show a scene to the MacBook over WebRTC, through the
scene's own camera.

Run (env isaac):  python labs/s4_stream.py out/hello_cube.usda
Then connect the Isaac Sim WebRTC Streaming Client on the Mac
to this machine's IP address. Stop with Ctrl-C.
(Adapted from NVIDIA's standalone example livestream.py.)
"""
import os
import sys

import look

look.pick()   # --theme is accepted; the saved scene carries its theme

from isaacsim import SimulationApp

usd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
if usd and not os.path.exists(usd):
    sys.exit(f"no such file: {usd}")

CONFIG = {
    "headless": True,          # no window on the Spark itself
    "hide_ui": False,          # but keep the full GUI in the stream
    "width": look.WIDTH, "height": look.HEIGHT,
    "window_width": 1920, "window_height": 1080,
    "renderer": "RealTimePathTracing",
    "display_options": 3286,   # show the default grid
    "open_usd": usd,
}
app = SimulationApp(launch_config=CONFIG)

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.utils.app import enable_extension
from isaacsim.core.rendering_manager import ViewportManager

app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.kit.livestream.app")   # WebRTC server

for _ in range(10):
    app.update()
stage = stage_utils.get_current_stage()
if stage.GetPrimAtPath("/World/Camera"):
    ViewportManager.set_camera("/World/Camera")   # the scene's camera
print("Streaming. Signal port TCP 49100, media port UDP 47998.")

while app.is_running() and not app.is_exiting():
    app.update()                # real-time loop, renders each frame
app.close()

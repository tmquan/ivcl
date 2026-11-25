"""simlab: the Isaac Sim boilerplate shared by the Chapter 1 labs.

Import this FIRST in a lab and call start(); only then import
anything else from isaacsim or pxr.

    from simlab import start
    app = start()              # headless, no picture
    app = start(stream=True)   # headless + WebRTC stream to the Mac

Every lab accepts --stream and --theme light|dark. What a lab shows
comes from scenes.py; the lab puts it on the stage itself, with
look.usd_scene(stage, prims, view, THEME).

Written against Isaac Sim 6.2 (isaacsim.core.experimental.*).
The Omniverse licence is accepted in Chapter 0, Step 1; without
that, Isaac Sim asks the first time it starts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import look

THEME = look.pick()      # --theme light (default) | dark
OUT = os.path.abspath("out")


def parse_args(**extra):
    """Every lab accepts --stream. extra = {name: default}."""
    p = argparse.ArgumentParser()
    p.add_argument("--stream", action="store_true",
                   help="keep running and stream to the WebRTC client")
    for name, default in extra.items():
        p.add_argument(f"--{name}", type=type(default),
                       default=default)
    args, _ = p.parse_known_args()
    return args


def start(stream=False):
    """Launch Isaac Sim. Returns the SimulationApp."""
    from isaacsim import SimulationApp

    config = {"headless": True, "width": look.WIDTH,
              "height": look.HEIGHT}
    if stream:   # same settings as NVIDIA's livestream.py example
        config.update({"hide_ui": False, "window_width": 1920,
                       "window_height": 1080,
                       "renderer": "RealTimePathTracing",
                       "display_options": 3286})
    app = SimulationApp(launch_config=config)
    if stream:
        from isaacsim.core.experimental.utils.app import (
            enable_extension)
        app.set_setting("/app/window/drawMouse", True)
        enable_extension("omni.kit.livestream.app")
    os.makedirs(OUT, exist_ok=True)
    return app


def new_stage():
    """An empty stage. Returns the Usd.Stage."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    stage_utils.create_new_stage()
    return stage_utils.get_current_stage()


def collide(paths, approximation=None):
    """Make prims collide. approximation is the collision shape,
    for example "convexHull"; None keeps the triangles (allowed
    only for things that do not move)."""
    from isaacsim.core.experimental.prims import GeomPrim
    shape = GeomPrim(paths, apply_collision_apis=True)
    if approximation is not None:
        shape.set_collision_approximations(approximation)
    return shape


def to_np(array):
    """Isaac Sim returns Warp arrays; turn them into NumPy."""
    import numpy as np
    return array.numpy() if hasattr(array, "numpy") \
        else np.asarray(array)


def save(name):
    """Save the current stage as out/<name> and report it."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    path = os.path.join(OUT, name)
    stage_utils.save_stage(path)
    print("wrote", path)
    return path


def finish(app, stream=False):
    """Close, or keep streaming until Ctrl-C / File > Exit."""
    if stream:
        from isaacsim.core.rendering_manager import ViewportManager
        for _ in range(10):
            app.update()
        ViewportManager.set_camera("/World/Camera")   # shared camera
        print("Streaming on TCP 49100 / UDP 47998."
              " Connect the WebRTC client; Ctrl-C to stop.")
        try:
            while app.is_running() and not app.is_exiting():
                app.update()
        except KeyboardInterrupt:
            pass
    app.close()

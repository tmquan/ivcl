"""Build a USD stage from a scene package and render it in Isaac Sim.

Runs with the Python of Isaac Sim, without a window:

    $ISAACSIM_PATH/python.sh build_scene.py out/crop313 --views 4

It writes ``<package>/scene.usd`` and ``<package>/render_<n>.png``. The stage
itself is made by ``scene.py``.
"""

import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("package")
ap.add_argument("--views", type=int, default=1, help="number of pictures around the block")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=1280)
ap.add_argument("--warmup", type=int, default=100, help="frames rendered before a picture is kept")
ap.add_argument("--no-render", action="store_true", help="write the USD file only")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height,
                     "renderer": "RayTracedLighting"})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402

package = Path(args.package).resolve()
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
center, radius = scene.build(stage, package, args.height / args.width)
usd_path = str(package / "scene.usd")
stage.GetRootLayer().Export(usd_path)
print(f"[connectome] wrote {usd_path}", flush=True)

if not args.no_render:
    import carb
    import omni.replicator.core as rep
    from PIL import Image

    # Render the file that was written, so that the picture shows what a
    # viewer of scene.usd will see.
    app.context.open_stage(usd_path)
    stage = app.context.get_stage()
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", 3)
    settings.set("/rtx/post/tonemap/enabled", True)

    product = rep.create.render_product(scene.CAMERA, (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for n in range(args.views):
        scene.set_view(stage, center, radius, -60.0 + 360.0 * n / args.views)
        for _ in range(args.warmup):
            app.update()
        rep.orchestrator.step()
        image = np.asarray(rgb.get_data())
        if image.size == 0:
            raise RuntimeError("the renderer returned an empty frame")
        out = package / f"render_{n}.png"
        Image.fromarray(image[..., :3]).save(out)
        print(f"[connectome] wrote {out} {image.shape} mean {image[..., :3].mean():.1f}",
              flush=True)

app.close()

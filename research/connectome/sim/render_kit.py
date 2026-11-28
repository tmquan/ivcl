"""Build the USD stage of a scene package and render it, in the Kit runtime of Isaac Sim.

Runs inside Kit, from the release folder of Isaac Sim:

    kit/kit ~/connectome/sim/connectome.viewer.kit --no-window \
        --ext-folder extscache --ext-folder exts \
        --exec "~/connectome/sim/render_kit.py <package> --views 2 --width 1024 --height 1024"

Writes <package>/scene.usd and <package>/render_<n>.png, one picture per view
around the block, and quits. Lines for the caller start with "[connectome]".
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Usd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402


def say(text):
    print(f"[connectome] {text}", flush=True)


async def render(a):
    app = omni.kit.app.get_app()
    package = Path(a.package).expanduser().resolve()
    stage = Usd.Stage.CreateInMemory()
    center, radius = scene.build(stage, package, a.height / a.width)
    usd = package / "scene.usd"
    stage.GetRootLayer().Export(str(usd))
    say(f"wrote {usd}")

    context = omni.usd.get_context()
    await context.open_stage_async(str(usd))
    stage = context.get_stage()
    product = rep.create.render_product(scene.CAMERA, (a.width, a.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for n in range(a.views):
        scene.set_view(stage, center, radius, -60.0 + 360.0 * n / a.views)
        for _ in range(a.warmup):
            await app.next_update_async()
        await rep.orchestrator.step_async()
        image = np.asarray(rgb.get_data())
        if image.size == 0 or image[..., :3].std() < 1:
            raise RuntimeError(f"the renderer returned an empty picture for view {n}")
        target = package / f"render_{n}.png"
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(target)
        say(f"wrote {target} {image.shape} mean {image[..., :3].mean():.1f}")
    rgb.detach()
    product.destroy()


async def main(args):
    code = 0
    try:
        await render(args)
    except Exception:
        say("failed:\n" + traceback.format_exc())
        code = 1
    omni.kit.app.get_app().post_quit(code)


# Kit gives the arguments of --exec in sys.argv only while this file runs, not in the update loop.
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("package")
ap.add_argument("--views", type=int, default=2)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=100, help="frames before each picture")
asyncio.ensure_future(main(ap.parse_args(sys.argv[1:])))

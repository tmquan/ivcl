"""Turn a crop (raw.npy, seg.npy, classes.npy, meta.json) into a scene package.

A crop is a folder written by render/data/fetch_*.py. The package is read by
scene.py: one surface mesh per object (the largest --top objects of seg), one
mesh for the cleft class if the crop has one, three image slices as textures,
and manifest.json. Lengths are in micrometres; axis order of the crop is z, y, x.

Needs numpy, scipy, scikit-image and pillow. Usage:

    python mesh_crop.py ~/connectome/data/crops/snemi3d out/snemi3d --top 60
"""

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure


def surface(mask, spacing, offset_zyx, sigma):
    """Marching cubes on a smoothed mask. Returns points as x, y, z and faces."""
    field = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma)
    if field.max() <= 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5, spacing=spacing)
    verts += (np.array(offset_zyx) - 2) * np.array(spacing)
    return verts[:, ::-1].astype(np.float32), faces[:, ::-1].astype(np.int32)


def colour(index):
    return list(colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.65, 0.95))


# Crops with instances.json (CellMap) give a class per id: those are coloured per class,
# as in the model viewer of https://cellmapchallenge.janelia.org/, and the classes that
# only fill the space between the organelles are not meshed.
CLASS_COLOURS = {"mito": [0.503, 1.0, 0.012], "er": [0.153, 0.15, 0.15], "golgi": [0.863, 0.078, 0.235],
                 "ves": [0.392, 0.584, 0.929], "endo": [0.4, 0.2, 0.6], "lyso": [0.180, 0.545, 0.341],
                 "perox": [1.0, 0.647, 0.0], "mt": [1.0, 0.078, 0.576], "ld": [1.0, 0.388, 0.278]}
CONTEXT_CLASSES = {"cyto", "ecs", "cell"}


def build(crop, out, top=60, xy_step=None):
    crop, out = Path(crop).expanduser(), Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((crop / "meta.json").read_text())
    vz, vy, vx = [float(v) for v in meta["voxel_nm"]]
    raw = np.load(crop / "raw.npy")
    seg = np.load(crop / "seg.npy")
    # Take every step-th voxel in the plane, so that the meshing voxel is near isotropic.
    step = xy_step or max(1, round(vz / (2.5 * vy)))
    spacing = (vz / 1000, vy * step / 1000, vx * step / 1000)
    size = [raw.shape[2] * vx / 1000, raw.shape[1] * vy / 1000, raw.shape[0] * vz / 1000]  # x, y, z

    ids = seg[:, ::step, ::step]
    values, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    dense = inverse.reshape(ids.shape).astype(np.int32) + 1
    boxes = ndimage.find_objects(dense)
    kinds = {}
    if (crop / "instances.json").exists():
        kinds = {int(i["id"]): i["class"] for i in json.loads((crop / "instances.json").read_text())["instances"]}
    order = [k for k in np.argsort(-counts)
             if values[k] != 0 and kinds.get(int(values[k])) not in CONTEXT_CLASSES][:top or None]
    objects = []
    for rank, k in enumerate(order):
        res = surface(dense[boxes[k]] == k + 1, spacing, [s.start for s in boxes[k]], (0.6, 1.0, 1.0))
        if res is None:
            continue
        kind = kinds.get(int(values[k]))
        name = f"{kind or 'object'}_{int(values[k])}"
        np.savez_compressed(out / f"{name}.npz", points=res[0], faces=res[1])
        objects.append({"name": name, "class": kind or "object", "file": f"{name}.npz",
                        "color": CLASS_COLOURS.get(kind) or colour(rank),
                        "opacity": 1.0, "metallic": 0.0, "roughness": 0.55,
                        "triangles": int(len(res[1])), "voxels": int(counts[k])})

    names = {v: int(k) for k, v in meta.get("class_names", {}).items()}
    cleft = next((i for n, i in names.items() if "cleft" in n), None)
    if cleft is not None and (crop / "classes.npy").exists():
        res = surface(np.load(crop / "classes.npy")[:, ::step, ::step] == cleft, spacing, (0, 0, 0), (0.5, 0.7, 0.7))
        if res is not None:
            np.savez_compressed(out / "clefts.npz", points=res[0], faces=res[1])
            objects.append({"name": "clefts", "class": "cleft", "file": "clefts.npz", "color": [1.0, 0.85, 0.0],
                            "opacity": 1.0, "metallic": 0.0, "roughness": 0.3, "triangles": int(len(res[1]))})

    X, Y, Z = size
    slices = []
    for name, image, corners in [
            ("slice_xy", raw[0][::-1], [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]]),
            ("slice_xz", raw[:, -1, :][::-1], [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]]),
            ("slice_yz", raw[:, :, 0][::-1], [[0, 0, 0], [0, Y, 0], [0, Y, Z], [0, 0, Z]])]:
        Image.fromarray(np.ascontiguousarray(image)).save(out / f"{name}.png")
        slices.append({"name": name, "file": f"{name}.png", "corners": corners})

    manifest = {"name": out.name, "source": f"{meta.get('dataset', crop.name)}, crop {crop.name}",
                "unit": "micrometre", "up_axis": "Z", "size": size, "voxel_nm": [vz, vy, vx],
                "mesh_voxel_um": list(spacing), "objects": objects, "slices": slices}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("crop")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=60, help="number of largest objects to mesh (0 = all)")
    ap.add_argument("--xy-step", type=int, default=None)
    a = ap.parse_args()
    m = build(a.crop, a.out, a.top, a.xy_step)
    print(f"{m['name']}: {len(m['objects'])} meshes, {sum(o['triangles'] for o in m['objects'])} triangles, "
          f"block {m['size'][0]:.2f} x {m['size'][1]:.2f} x {m['size'][2]:.2f} um")

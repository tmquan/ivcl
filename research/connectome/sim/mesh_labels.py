"""Turn the labels of a CREMI sample into a scene package of surface meshes.

Reads the HDF5 file of one CREMI sample (image, neuron labels, cleft labels),
makes one surface mesh per neurite and one for all clefts, and writes three
image slices as textures. The output folder is read by ``build_scene.py``.

Needs numpy, h5py, scipy, scikit-image and pillow. Usage:

    python mesh_labels.py sample_A_20160501.hdf out/cremi_A --top 40
"""

import argparse
import colorsys
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

CLEFT_BACKGROUND = np.uint64(0xFFFFFFFFFFFFFFFF)


def surface(mask, spacing, offset_zyx, sigma):
    """Marching cubes on a smoothed mask. Returns points as x, y, z and faces."""
    field = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma)
    if field.max() <= 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5, spacing=spacing)
    verts += (np.array(offset_zyx) - 2) * np.array(spacing)
    points = verts[:, ::-1].astype(np.float32)          # z, y, x  ->  x, y, z
    faces = faces[:, ::-1].astype(np.int32)             # keep outward winding
    return points, faces


def colour(index):
    """Well separated hues, by the golden ratio."""
    hue = (index * 0.61803398875) % 1.0
    return list(colorsys.hsv_to_rgb(hue, 0.65, 0.95))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hdf")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=40,
                    help="number of largest neurites to mesh (0 = all)")
    ap.add_argument("--xy-step", type=int, default=4,
                    help="take every n-th voxel in x and y before meshing")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf, "r") as f:
        raw = f["volumes/raw"][:]
        res_nm = f["volumes/raw"].attrs.get("resolution", (40.0, 4.0, 4.0))
        ids = f["volumes/labels/neuron_ids"][:]
        clefts = f["volumes/labels/clefts"][:] if "volumes/labels/clefts" in f else None

    step = args.xy_step
    # Micrometres per voxel, in z, y, x, after taking every step-th voxel.
    spacing = (res_nm[0] / 1000, res_nm[1] * step / 1000, res_nm[2] * step / 1000)
    size_um = [raw.shape[2] * res_nm[2] / 1000, raw.shape[1] * res_nm[1] / 1000,
               raw.shape[0] * res_nm[0] / 1000]
    print(f"volume {raw.shape} (z, y, x), {size_um[0]:.2f} x {size_um[1]:.2f} x "
          f"{size_um[2]:.2f} um, meshing voxel {spacing} um")

    ids = ids[:, ::step, ::step]
    values, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    dense = inverse.reshape(ids.shape).astype(np.int32) + 1
    order = np.argsort(-counts)
    if args.top:
        order = order[: args.top]
    boxes = ndimage.find_objects(dense)

    objects = []
    for rank, k in enumerate(order):
        box = boxes[k]
        mask = dense[box] == k + 1
        result = surface(mask, spacing, [s.start for s in box], (0.6, 1.0, 1.0))
        if result is None:
            continue
        points, faces = result
        name = f"neurite_{int(values[k])}"
        np.savez_compressed(out / f"{name}.npz", points=points, faces=faces)
        objects.append({
            "name": name, "class": "neurite", "file": f"{name}.npz",
            "color": colour(rank), "opacity": 1.0, "metallic": 0.0,
            "roughness": 0.55, "triangles": int(len(faces)),
            "voxels": int(counts[k]),
        })
    print(f"meshed {len(objects)} of {len(values)} neurites")

    if clefts is not None:
        mask = clefts[:, ::step, ::step] != CLEFT_BACKGROUND
        result = surface(mask, spacing, (0, 0, 0), (0.5, 0.7, 0.7))
        if result is not None:
            points, faces = result
            np.savez_compressed(out / "clefts.npz", points=points, faces=faces)
            objects.append({
                "name": "clefts", "class": "cleft", "file": "clefts.npz",
                "color": [1.0, 0.85, 0.0], "opacity": 1.0, "metallic": 0.0,
                "roughness": 0.3, "triangles": int(len(faces)),
            })
            print(f"meshed clefts: {len(faces)} triangles")

    # Three image slices on the faces of the cube that are away from the camera.
    X, Y, Z = size_um
    slices = []

    def save(name, image, corners):
        Image.fromarray(image).save(out / f"{name}.png")
        slices.append({"name": name, "file": f"{name}.png", "corners": corners})

    # Corners run with the texture: (u, v) = (0, 0), (1, 0), (1, 1), (0, 1),
    # with v = 0 at the bottom row of the image.
    save("slice_xy", raw[0][::-1], [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]])
    save("slice_xz", raw[:, -1, :][::-1], [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]])
    save("slice_yz", raw[:, :, 0][::-1], [[0, 0, 0], [0, Y, 0], [0, Y, Z], [0, 0, Z]])

    manifest = {
        "name": out.name,
        "source": f"CREMI, {Path(args.hdf).name}",
        "unit": "micrometre",
        "up_axis": "Z",
        "size": size_um,
        "objects": objects,
        "slices": slices,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json: "
          f"{sum(o['triangles'] for o in objects)} triangles in all")


if __name__ == "__main__":
    main()

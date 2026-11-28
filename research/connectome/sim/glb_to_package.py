"""Convert a CellMap challenge GLB (for example crop313) to a scene package.

A scene package is a folder with ``manifest.json`` and one ``.npz`` per object
(``points`` N x 3 float32, ``faces`` M x 3 int32). ``build_scene.py`` turns a
package into a USD stage inside Isaac Sim.

Needs only numpy. Usage:

    python glb_to_package.py crop313.glb out/crop313
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

# Labels of the model viewer on https://cellmapchallenge.janelia.org/ for
# crop313: name and position of each hotspot. Each object of the GLB takes the
# name of the hotspot nearest to its surface.
HOTSPOTS = {
    "mitochondria": (109.18, 178.02, 6.48),
    "microtubule": (175.10, 167.30, 144.35),
    "golgi": (189.52, 5.63, 119.32),
    "endosome": (110.96, 58.38, 5.41),
    "vesicle": (122.04, 32.12, 15.68),
    "lipid_droplet": (132.99, 12.27, 179.32),
    "endoplasmic_reticulum": (20.66, 194.99, 148.98),
    "lysosome": (176.04, 24.16, 33.08),
}

DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC3": 3}


def read_glb(path):
    blob = Path(path).read_bytes()
    magic, version, _ = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2 binary file")
    json_len = struct.unpack("<I", blob[12:16])[0]
    doc = json.loads(blob[20 : 20 + json_len])
    start = 20 + json_len
    bin_len = struct.unpack("<I", blob[start : start + 4])[0]
    return doc, blob[start + 8 : start + 8 + bin_len]


def accessor(doc, data, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(DTYPES[acc["componentType"]])
    width = WIDTHS[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    out = np.ndarray(
        (acc["count"], width), dtype, data, offset, (stride, dtype.itemsize)
    )
    return np.ascontiguousarray(out)


def main(src, dst):
    doc, data = read_glb(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    parts = []
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            points = accessor(doc, data, prim["attributes"]["POSITION"])
            if "indices" in prim:
                faces = accessor(doc, data, prim["indices"]).reshape(-1, 3)
            else:
                faces = np.arange(len(points)).reshape(-1, 3)
            parts.append((points.astype(np.float32), faces.astype(np.int32),
                          doc["materials"][prim["material"]]))

    # Name each part after its nearest hotspot; the rest keep the colour name.
    names = [None] * len(parts)
    for label, pos in HOTSPOTS.items():
        dist = [np.linalg.norm(p - np.array(pos), axis=1).min() for p, _, _ in parts]
        order = np.argsort(dist)
        best = next(i for i in order if names[i] is None)
        names[best] = label
        print(f"{label:24s} -> part {best} ({parts[best][2]['name']}), "
              f"distance {dist[best]:.2f}")

    objects = []
    for i, (points, faces, mat) in enumerate(parts):
        name = names[i] or f"other_{mat['name']}"
        pbr = mat.get("pbrMetallicRoughness", {})
        color = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
        np.savez_compressed(dst / f"{name}.npz", points=points, faces=faces)
        objects.append({
            "name": name,
            "class": name,
            "file": f"{name}.npz",
            "color": color[:3],
            "opacity": color[3],
            "metallic": pbr.get("metallicFactor", 1.0),
            "roughness": pbr.get("roughnessFactor", 1.0),
            "triangles": int(len(faces)),
        })

    manifest = {
        "name": Path(src).stem,
        "source": "CellMap segmentation challenge, https://cellmapchallenge.janelia.org/",
        "unit": "as in the GLB file",
        "up_axis": "Y",
        "objects": objects,
        "slices": [],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(objects)} objects to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

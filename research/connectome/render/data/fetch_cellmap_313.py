#!/usr/bin/env python3
"""Fetch crop 313 of the CellMap segmentation challenge and write it in the crop format.

SOURCE
  CellMap 2024 Segmentation Challenge, CellMap Project Team, HHMI Janelia.
    https://cellmapchallenge.janelia.org/
    https://github.com/janelia-cellmap/cellmap-segmentation-challenge   (manifests, class table)
    collection DOI https://doi.org/10.25378/janelia.c.7456966
  Crop 313 lies in the FIB-SEM volume jrc_mus-liver-zon-1 (mouse liver). Data are read
  from the public bucket s3://janelia-cosem-datasets (anonymous access, zarr version 2):
    image   jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/<level>
    labels  jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/groundtruth/crop313/<class>/<level>
  Records of the two data sets (licence, citation, annotation method):
    labels  https://doi.org/10.25378/janelia.24261094   image  https://doi.org/10.25378/janelia.24128700

LICENCE
  Data: CC BY 4.0, as stated in the two records above (read again at run time and
  written to meta.json). Code of the challenge repository: BSD 3-Clause.

WHAT IT WRITES  (--out names the folder of the first crop; the second goes beside it)
  <out>/                     the crop "cellmap_313", on the grid of the image level whose
                             voxel is closest to the voxel of the labels
    raw.npy        uint8  [z, y, x]  FIB-SEM image over the box of crop 313
    classes.npy    uint8  [z, y, x]  atomic class id of the challenge per voxel, 0 = none.
                                     The labels are finer than the image; each image voxel takes
                                     the most frequent label of its block of label voxels (of a
                                     tie, the first in z, y, x order). This equals the level the
                                     challenge itself stores at the voxel size of the image
    seg.npy        uint64 [z, y, x]  one id per object. Classes for which the challenge gives
                                     instance ids keep those instances; the other classes are
                                     cut into 26-connected components. Ids are unique across classes
    instances.json                   class, voxel count, box, centre of every id of seg.npy
    classes_4nm.npy  (extra)         the atomic class ids on the grid of the labels, unresampled
                                     (the name carries the voxel size of the labels)
    glb_check.json   (extra)         what the mesh file crop313.glb contains, compared with the labels
    meta.json, preview.png, preview_ortho.png
  <out>_context/             the crop "cellmap_313_context": image only, a box about twice
                             as large in each direction around the crop
    raw.npy, meta.json, preview.png

USAGE (on the Spark)
  ~/connectome/render-venv/bin/python fetch_cellmap_313.py --out ~/connectome/data/crops/cellmap_313
  Downloads are cached under <out>/../../src/cellmap_313/ (or --src); running again reads
  the cache and writes the same files. --refresh downloads again.

NEEDS  numpy, scipy, s3fs, numcodecs, connected-components-3d (cc3d), matplotlib.
"""

import argparse
import csv
import itertools
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

NAME = "cellmap_313"
CROP = 313
FETCHED = "2026-11-23"

GITHUB = "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/main/"
HTTP_SOURCES = {
    # file in the cache : URL
    "manifest.csv": GITHUB + "src/cellmap_segmentation_challenge/utils/manifest.csv",
    "train_crop_manifest.csv": GITHUB + "src/cellmap_segmentation_challenge/utils/train_crop_manifest.csv",
    "classes.csv": GITHUB + "docs/source/_static/classes.csv",
    "figshare_labels.json": "https://api.figshare.com/v2/articles/24261094",
    "figshare_image.json": "https://api.figshare.com/v2/articles/24128700",
    "figshare_collection.json": "https://api.figshare.com/v2/collections/7456966",
    "challenge_front_page.html": "https://cellmapchallenge.janelia.org/",    # labels of the 3D model of crop 313
}

# Organelle-level classes of seg.npy. Each atomic class id belongs to the first group of
# this list that contains it (groups as in the class table of the challenge); an atomic
# class in none of them is its own object class. "er" of the challenge also contains the
# nuclear envelope and pores; here these go to "nuc", so that the object classes do not overlap.
OBJECT_GROUPS = ["mito", "golgi", "ves", "endo", "lyso", "ld", "perox", "mt", "nuc", "isg", "er"]
# Object classes that are not organelles: the space between them.
NOT_ORGANELLE = {"cyto": "cytosol", "ecs": "extracellular space"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_npy(path, arr):
    """Write through a temporary file, so that a killed run leaves no half file."""
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- sources

class Sources:
    """Cached access to the HTTP files and to the zarr arrays on S3. Records what was read."""

    def __init__(self, cache, refresh=False):
        self.cache = Path(cache)
        self.refresh = refresh
        self.read = []            # URLs actually read (also when served from the cache)
        self._fs = None
        for sub in ("http", "meta", "arrays"):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)

    def note(self, url):
        if url not in self.read:
            self.read.append(url)

    # -- HTTP
    def http(self, name, required=True):
        url = HTTP_SOURCES[name]
        path = self.cache / "http" / name
        if self.refresh or not path.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "fetch-cellmap-crop/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if not data:
                    raise IOError("empty answer")
                tmp = Path(str(path) + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            except Exception as e:                      # noqa: BLE001
                if required or not path.exists():
                    if required:
                        raise
                    log(f"could not read {url}: {e!r}")
                    return None
        self.note(url)
        return path.read_text()

    # -- S3
    @property
    def fs(self):
        if self._fs is None:
            import s3fs
            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def s3_json(self, key, missing_ok=False, note=True):
        """A small JSON object of the bucket (key without s3://), cached."""
        path = self.cache / "meta" / (key.replace("/", "__") + ".json")
        if self.refresh or not path.exists():
            try:
                data = self.fs.cat_file(key)
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise
            tmp = Path(str(path) + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        if note:
            self.note("s3://" + key)
        return json.loads(path.read_text())

    def s3_json_many(self, keys):
        """Fetch the JSON objects that are not in the cache yet, together. Missing ones are skipped."""
        paths = {k: self.cache / "meta" / (k.replace("/", "__") + ".json") for k in keys}
        need = [k for k, p in paths.items() if self.refresh or not p.exists()]
        if need:
            got = self.fs.cat(need, on_error="return")
            for k in need:
                if isinstance(got[k], bytes):
                    tmp = Path(str(paths[k]) + ".tmp")
                    tmp.write_bytes(got[k])
                    os.replace(tmp, paths[k])
                elif not isinstance(got[k], FileNotFoundError):
                    raise got[k]

    def s3_box(self, array, lo, hi, tag):
        """Voxels lo <= index < hi (z, y, x) of the zarr-2 array at `array`, cached (compressed .npz)."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        name = f"{tag}_z{lo[0]}-{hi[0]}_y{lo[1]}-{hi[1]}_x{lo[2]}-{hi[2]}.npz"
        path = self.cache / "arrays" / name
        self.note("s3://" + array)
        if path.exists() and not self.refresh:
            with np.load(path) as z:
                return z["a"]
        import numcodecs
        za = self.s3_json(array + "/.zarray", note=False)
        if za["zarr_format"] != 2 or za["order"] != "C" or za.get("filters"):
            raise ValueError(f"{array}: layout not handled: {za}")
        shape, chunks = za["shape"], za["chunks"]
        if any(l < 0 or h > s or h <= l for l, h, s in zip(lo, hi, shape)):
            raise ValueError(f"{array}: box {lo}..{hi} outside shape {shape}")
        dtype = np.dtype(za["dtype"])
        codec = numcodecs.get_codec(za["compressor"]) if za.get("compressor") else None
        sep = za.get("dimension_separator", ".")
        out = np.full([h - l for l, h in zip(lo, hi)], za.get("fill_value") or 0, dtype)
        ranges = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, chunks)]
        index = list(itertools.product(*ranges))
        keys = {array + "/" + sep.join(str(i) for i in idx): idx for idx in index}
        got = {}
        if int(np.prod([-(-s // c) for s, c in zip(shape, chunks)])) <= 20000:
            # a small array: one listing tells which chunks are stored (an empty chunk is not)
            there = set(self.fs.find(array))
            got = {k: FileNotFoundError(k) for k in keys if k not in there}
        todo = [k for k in keys if k not in got]
        for attempt in range(4):
            for start in range(0, len(todo), 2000):
                got.update(self.fs.cat(todo[start:start + 2000], on_error="return"))
            todo = [k for k in todo if isinstance(got[k], Exception) and not isinstance(got[k], FileNotFoundError)]
            if not todo:
                break
            log(f"{array}: {len(todo)} chunks failed, attempt {attempt + 1}")
            time.sleep(2 + 3 * attempt)
        if todo:
            raise IOError(f"{array}: could not read {len(todo)} chunks, e.g. {got[todo[0]]!r}")
        missing = 0
        for key, idx in keys.items():
            buf = got[key]
            if isinstance(buf, FileNotFoundError):       # chunk not stored: all fill value
                missing += 1
                continue
            if codec is not None:
                buf = codec.decode(buf)
            chunk = np.frombuffer(buf, dtype).reshape(chunks)
            c0 = [i * c for i, c in zip(idx, chunks)]
            src = tuple(slice(max(l - o, 0), min(h - o, c)) for l, h, o, c in zip(lo, hi, c0, chunks))
            dst = tuple(slice(max(o - l, 0), max(o - l, 0) + (s.stop - s.start)) for l, o, s in zip(lo, c0, src))
            out[dst] = chunk[src]
        log(f"read s3://{array} box {lo}..{hi}: {len(index)} chunks, {missing} not stored")
        tmp = Path(str(path) + ".tmp.npz")
        np.savez_compressed(tmp, a=out)
        os.replace(tmp, path)
        return out


def multiscale_levels(zattrs):
    """[(path, scale[z,y,x], translation[z,y,x])] of an OME-NGFF 0.4 multiscale group."""
    ms = zattrs["multiscales"][0]
    axes = [a["name"] for a in ms["axes"]]
    units = {a.get("unit") for a in ms["axes"]}
    if axes != ["z", "y", "x"] or units != {"nanometer"}:
        raise ValueError(f"axes {axes} with units {units}: expected z, y, x in nanometer")
    out = []
    for d in ms["datasets"]:
        scale = trans = None
        for t in d["coordinateTransformations"]:
            if t["type"] == "scale":
                scale = [float(v) for v in t["scale"]]
            elif t["type"] == "translation":
                trans = [float(v) for v in t["translation"]]
        out.append((d["path"], scale, trans or [0.0, 0.0, 0.0]))
    return out


# ----------------------------------------------------------------------------- resampling

def block_mode(a, f, ignore_zero=False):
    """Most frequent value of every f x f x f block of `a`.

    Of several values with the same count, the one that comes first in the block in
    z, y, x scan order wins. This is the rule that reproduces the coarser levels the
    challenge stores for its labels (checked in main). With ignore_zero, zeros do not vote.
    Returns (array, share of blocks with one value only, share of blocks with a tie).
    """
    if f == 1:
        return a.copy(), 1.0, 0.0
    Z, Y, X = (s // f for s in a.shape)
    out = np.zeros((Z, Y, X), a.dtype)
    pure = tied = 0
    step = max(1, 4_000_000 // (Y * X))
    for z0 in range(0, Z, step):
        n = min(step, Z - z0)
        blk = a[z0 * f:(z0 + n) * f].reshape(n, f, Y, f, X, f).transpose(0, 2, 4, 1, 3, 5).reshape(n, Y, X, f ** 3)
        best = np.zeros((n, Y, X), a.dtype)
        top = np.zeros((n, Y, X), np.int16)
        counts = []
        for k in range(f ** 3):
            v = blk[..., k]
            c = (blk == v[..., None]).sum(-1, dtype=np.int16)
            if ignore_zero:
                c[v == 0] = 0
            take = c > top                               # strictly more: the first one keeps a tie
            best[take] = v[take]
            top[take] = c[take]
            counts.append(c)
        out[z0:z0 + n] = best
        holders = sum((c == top) for c in counts)        # block voxels whose value has the top count
        pure += int((top == f ** 3).sum())
        tied += int(((holders > top) & (top > 0)).sum())
    return out, pure / (Z * Y * X), tied / (Z * Y * X)


def eta2(raw, lab, nlab):
    """Share of the variance of `raw` that the label of each voxel explains (0..1)."""
    r = raw.ravel().astype(np.float64)
    l = lab.ravel()
    n = np.bincount(l, minlength=nlab).astype(np.float64)
    s = np.bincount(l, weights=r, minlength=nlab)
    ok = n > 0
    between = (s[ok] ** 2 / n[ok]).sum() - r.sum() ** 2 / r.size
    total = (r ** 2).sum() - r.sum() ** 2 / r.size
    return float(between / total)


# ----------------------------------------------------------------------------- the mesh file

def read_glb(path):
    blob = Path(path).read_bytes()
    magic, version, _ = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2 binary file")
    json_len = struct.unpack("<I", blob[12:16])[0]
    doc = json.loads(blob[20:20 + json_len])
    start = 20 + json_len
    bin_len = struct.unpack("<I", blob[start:start + 4])[0]
    return doc, blob[start + 8:start + 8 + bin_len]


def glb_accessor(doc, data, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = np.dtype({5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[acc["componentType"]])
    width = {"SCALAR": 1, "VEC3": 3}[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    out = np.ndarray((acc["count"], width), dtype, data, offset, (stride, dtype.itemsize))
    return np.ascontiguousarray(out)


def sample_triangles(pts, faces, per_area, rng):
    """Points spread over the triangles of a mesh, about `per_area` per unit of area."""
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = np.repeat(np.arange(len(faces)), np.ceil(area * per_area).astype(int) + 1)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    over = u + v > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx]), float(area.sum())


def parse_hotspots(html):
    """Labels of the 3D model on the front page of the challenge: [(text, [x, y, z])]."""
    import re
    spots = []
    for m in re.finditer(r'data-position="([^"]+)".*?<div class="HotspotAnnotation">([^<]+)</div>', html, flags=re.S):
        xyz = [float(v.rstrip("m")) for v in m.group(1).split()]
        spots.append((m.group(2).strip(), xyz))
    return spots


def glb_check(glb_path, classes, object_class, atomic_names, object_names, object_of, voxel_nm, hotspots,
              class_of_text):
    """Compare the meshes of the GLB with the labels.

    The GLB carries no class names (see "node_names" in the result), so each mesh is
    matched by position. For every class: precision = share of the mesh surface that lies
    on the surface of the class; recall = share of the class surface that the mesh covers;
    both within 1.5 voxels and away from the faces of the crop. The axis order and
    direction of the GLB are found as the one of the 48 that puts most of the mesh
    surface on label surfaces. The front page of the challenge labels points of this
    model with class names ("hotspots"); each is given to the mesh nearest to it.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    try:
        import edt

        def distance(m):
            return edt.edt(m, parallel=4).astype(np.float32)
    except ImportError:
        def distance(m):
            return ndimage.distance_transform_edt(m).astype(np.float32)

    tol = 1.5
    rng = np.random.default_rng(0)
    doc, data = read_glb(glb_path)
    out = {
        "file": Path(glb_path).name,
        "generator": doc.get("asset", {}).get("generator"),
        "node_names": [n.get("name") for n in doc["nodes"]],
        "mesh_names": [m.get("name") for m in doc["meshes"]],
        "material_names": [m.get("name") for m in doc.get("materials", [])],
        "names_note": "the file holds no class name: nodes are numbered, meshes are unnamed, materials are colour names",
    }
    parts = []
    for node in doc["nodes"]:
        if "mesh" not in node:
            continue
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            pts = glb_accessor(doc, data, prim["attributes"]["POSITION"]).astype(np.float64)
            if "indices" in prim:
                faces = glb_accessor(doc, data, prim["indices"]).reshape(-1, 3).astype(np.int64)
            else:
                faces = np.arange(len(pts)).reshape(-1, 3)
            mat = doc["materials"][prim["material"]]["name"] if "material" in prim else None
            surface, area = sample_triangles(pts, faces, 2.0, rng)
            parts.append({"node": node.get("name"), "material": mat, "triangles": int(len(faces)),
                          "points": pts, "surface": surface, "area": area})
    allpts = np.concatenate([p["points"] for p in parts])
    out["bounds_min_max_xyz"] = [np.round(allpts.min(0), 2).tolist(), np.round(allpts.max(0), 2).tolist()]
    shape = np.array(classes.shape)
    out["unit_note"] = (f"the meshes span {out['bounds_min_max_xyz']} and the crop is {shape.tolist()} voxels of "
                        f"{voxel_nm[0]:g} nm: one mesh unit is taken as one voxel of the crop")

    # Candidate classes: every atomic class and every object class present in the crop.
    masks = {}
    for i in np.unique(classes):
        if i:
            masks[atomic_names[int(i)]] = classes == i
    for i in np.unique(object_class):
        if i and object_names[int(i)] not in masks:
            masks[object_names[int(i)]] = object_class == i
    inner = np.zeros(classes.shape, bool)
    inner[3:-3, 3:-3, 3:-3] = True                       # away from the faces of the crop
    surf, dist = {}, {}
    for name, m in masks.items():
        sm = m & ~ndimage.binary_erosion(m, border_value=1)
        surf[name] = np.argwhere(sm & inner).astype(np.float64)
        dist[name] = distance(~sm)

    def to_index(pts, perm, flip):
        """Mesh point -> continuous voxel index (z, y, x); voxel i spans [i, i + 1) in mesh units."""
        q = pts[:, perm].copy()
        for ax in range(3):
            if flip[ax]:
                q[:, ax] = shape[ax] - q[:, ax]
        return q - 0.5

    def lookup(vol, q):
        i = np.clip(np.rint(q).astype(int), 0, shape - 1)
        return vol[i[:, 0], i[:, 1], i[:, 2]]

    total = sum(len(p["surface"]) for p in parts)
    trials = []
    for perm in itertools.permutations(range(3)):
        for flip in itertools.product((0, 1), repeat=3):
            score = 0.0
            for p in parts:
                q = to_index(p["surface"], list(perm), flip)
                score += max(float((lookup(d, q) <= tol).mean()) for d in dist.values()) * len(q)
            trials.append((score / total, perm, flip))
    trials.sort(reverse=True)
    score, perm, flip = trials[0]
    out["axes"] = {
        "mesh_axis_of_array_axis": {"zyx"[i]: "xyz"[perm[i]] for i in range(3)},
        "flipped_array_axes": ["zyx"[i] for i in range(3) if flip[i]],
        "share_of_mesh_surface_on_a_label_surface": round(score, 4),
        "next_best_axis_choice": round(trials[1][0], 4),
        "note": "found by search over 6 axis orders x 8 directions",
    }
    out["meshes"] = []
    trees = []
    for p in parts:
        q = to_index(p["surface"], list(perm), flip)
        qk = q[np.all((q > 2.5) & (q < shape - 3.5), axis=1)]
        trees.append(cKDTree(to_index(p["points"], list(perm), flip)))
        rows = []
        if len(qk):
            tree = cKDTree(qk)
            for name in masks:
                if len(surf[name]) == 0:
                    continue
                prec = float((lookup(dist[name], qk) <= tol).mean())
                d, _ = tree.query(surf[name], distance_upper_bound=tol)
                rec = float(np.isfinite(d).mean())
                rows.append((2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec, name))
        rows.sort(reverse=True)
        out["meshes"].append({
            "node": p["node"], "material": p["material"], "triangles": p["triangles"],
            "area_voxel_faces": round(p["area"], 1),
            "best_class": rows[0][3] if rows else None,
            "organelle": object_of.get(rows[0][3], rows[0][3]) if rows else None,
            "matches": [{"class": n, "f1": round(f, 3), "precision": round(pr, 3), "recall": round(rc, 3)}
                        for f, pr, rc, n in rows[:4]],
            "website_labels": [],
        })
    # the labels of the front page
    out["website_labels"] = []
    for text, xyz in hotspots:
        q = to_index(np.array([xyz]), list(perm), flip)
        d = [float(t.query(q)[0][0]) for t in trees]
        k = int(np.argmin(d))
        cls = class_of_text.get(text.lower())
        mesh = out["meshes"][k]
        mesh["website_labels"].append(text)
        n_vox = int(masks[cls].sum()) if cls in masks else 0
        out["website_labels"].append({
            "text": text, "class_of_the_challenge": cls, "voxels_of_that_class_in_the_crop": n_vox,
            "nearest_mesh": f"{mesh['node']} ({mesh['material']})", "distance_voxels": round(d[k], 2),
            "class_of_that_mesh_by_position": mesh["best_class"],
            "organelle_of_that_mesh": mesh["organelle"],
            "agree": bool(cls is not None and cls == mesh["organelle"]),
        })
    wrong = [w for w in out["website_labels"] if not w["agree"]]
    out["summary"] = (
        f"{len(out['website_labels']) - len(wrong)} of {len(out['website_labels'])} labels of the front page name the "
        "organelle that the label arrays give at that place"
        + ("" if not wrong else "; not: " + "; ".join(
            f"'{w['text']}' sits on the mesh of {w['class_of_that_mesh_by_position']}"
            + (f" (the crop has no voxel of {w['class_of_the_challenge']})"
               if not w["voxels_of_that_class_in_the_crop"] else "") for w in wrong))
        + ". Meshes without a label of the front page: "
        + (", ".join(f"{m['material']} = {m['best_class']} (F1 {m['matches'][0]['f1']})"
                     for m in out["meshes"] if not m["website_labels"] and m["matches"]) or "none") + ".")
    return out


# ----------------------------------------------------------------------------- previews

def upscale(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def boundaries(lab):
    b = np.zeros(lab.shape, bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return b


def id_colours(n):
    """One colour per id; ids next to each other in number get distant hues."""
    import colorsys
    i = np.arange(n)
    return np.array([colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.6 + 0.4 * ((k * 7) % 3) / 2, 0.75 + 0.25 * ((k * 5) % 4) / 3)
                     for k in i])


def overlay(raw2d, lab2d, colours, alpha, k, clear=(), line=(1.0, 1.0, 0.2)):
    """raw as grey, labels as colours (index = label), label boundaries as lines; k = zoom."""
    g = upscale(raw2d, k).astype(np.float32) / 255.0
    lab = upscale(lab2d, k)
    rgb = np.repeat(g[..., None], 3, axis=2)
    show = lab > 0
    for c in clear:
        show &= lab != c
    rgb[show] = (1 - alpha) * rgb[show] + alpha * colours[lab[show]]
    rgb[boundaries(lab)] = line
    return rgb


def write_previews(out, raw, seg, classes, class_names, cyto_ids, voxel_nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    k = max(1, 800 // max(raw.shape))
    nid = int(seg.max()) + 1
    seg_cols = id_colours(nid)
    present = [int(c) for c in np.unique(classes) if c]
    cmap = plt.get_cmap("tab20")
    cls_cols = np.zeros((256, 3))
    for i, c in enumerate(present):
        cls_cols[c] = cmap(i % 20)[:3]
    seg32 = seg.astype(np.int64)

    def panels(axis, index):
        r = np.take(raw, index, axis)
        s = np.take(seg32, index, axis)
        c = np.take(classes, index, axis).astype(np.int64)
        return (np.repeat(upscale(r, k)[..., None], 3, 2).astype(np.float32) / 255.0,
                overlay(r, s, seg_cols, 0.45, k, clear=cyto_ids),
                overlay(r, c, cls_cols, 0.5, k, line=(1.0, 1.0, 1.0)))

    legend = [Patch(color=cls_cols[c], label=f"{c} {class_names[c]}") for c in present]
    # preview.png: the middle z section, three panels side by side.
    z = raw.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.9))
    titles = [f"raw, z = {z} (y down, x right)",
              "seg.npy: one colour per id (cytosol left grey), label boundaries in yellow",
              "classes.npy: one colour per atomic class, boundaries in white"]
    for ax, im, t in zip(axs, panels(0, z), titles):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=legend, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.suptitle(f"{NAME}: {raw.shape[0]} x {raw.shape[1]} x {raw.shape[2]} voxels (z, y, x) of "
                 f"{voxel_nm[0]:g} x {voxel_nm[1]:g} x {voxel_nm[2]:g} nm", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out / "preview.png", dpi=110)
    plt.close(fig)
    # preview_ortho.png: the three orthogonal middle sections, to catch a swap of axes.
    fig, axs = plt.subplots(3, 3, figsize=(18, 18.6))
    names = [("z", "y down, x right"), ("y", "z down, x right"), ("x", "z down, y right")]
    for row, (axis, (nm, how)) in enumerate(zip(range(3), names)):
        idx = raw.shape[axis] // 2
        for ax, im, t in zip(axs[row], panels(axis, idx), ["raw", "seg.npy", "classes.npy"]):
            ax.imshow(im, interpolation="nearest")
            ax.set_title(f"{t}, {nm} = {idx} ({how})", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.legend(handles=legend, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "preview_ortho.png", dpi=90)
    plt.close(fig)


def write_context_preview(out, raw, off, shape, voxel_nm, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6))
    views = [(0, "z", (2, 1), "y down, x right"), (1, "y", (2, 0), "z down, x right"), (2, "x", (1, 0), "z down, y right")]
    for ax, (axis, nm, (h, v), how) in zip(axs, views):
        idx = off[axis] + shape[axis] // 2
        ax.imshow(np.take(raw, idx, axis), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.add_patch(Rectangle((off[h] - 0.5, off[v] - 0.5), shape[h], shape[v], fill=False, edgecolor="yellow", linewidth=1.2))
        ax.set_title(f"raw, {nm} = {idx} ({how}); yellow: the box of {NAME}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: {raw.shape[0]} x {raw.shape[1]} x {raw.shape[2]} voxels (z, y, x) of "
                 f"{voxel_nm[0]:g} x {voxel_nm[1]:g} x {voxel_nm[2]:g} nm, image only", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "preview.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder of the crop cellmap_313; cellmap_313_context goes beside it")
    ap.add_argument("--src", default=None, help="cache folder (default: <out>/../../src/cellmap_313)")
    ap.add_argument("--glb", default=None, help="mesh file of the crop (default: <out>/../../crop313.glb, skipped if absent)")
    ap.add_argument("--context", type=float, default=2.0, help="size of the context box relative to the crop")
    ap.add_argument("--refresh", action="store_true", help="download again instead of using the cache")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out_ctx = out.with_name(out.name + "_context")
    data_root = out.parent.parent
    cache = Path(args.src).expanduser() if args.src else data_root / "src" / NAME
    glb = Path(args.glb).expanduser() if args.glb else data_root / "crop313.glb"
    out.mkdir(parents=True, exist_ok=True)
    out_ctx.mkdir(parents=True, exist_ok=True)
    src = Sources(cache, args.refresh)
    notes = []

    # ---- 1. the manifests of the challenge: where crop 313 is
    row = next(r for r in csv.DictReader(src.http("manifest.csv").splitlines()) if int(r["crop_number"]) == CROP)
    dataset = row["collection"]
    em_key = row["em_url"].replace("s3://", "")
    crop_key = row["crop_url"].replace("s3://", "")
    listed = [r for r in csv.DictReader(src.http("train_crop_manifest.csv").splitlines())
              if r["crop_name"] == str(CROP)]
    listed_classes = sorted(r["class_label"] for r in listed)

    def vec(text):
        return [float(v) for v in text.strip("[]").split(";")]

    man_voxel = {tuple(vec(r["voxel_size"])) for r in listed}
    man_trans = {tuple(vec(r["translation"])) for r in listed}
    man_shape = {tuple(int(v) for v in vec(r["shape"])) for r in listed}
    log(f"crop {CROP}: dataset {dataset}; manifest lists {len(listed_classes)} classes, voxel {man_voxel}, "
        f"translation {man_trans}, shape {man_shape}")

    # ---- 2. the class table of the challenge
    table = list(csv.DictReader(src.http("classes.csv").splitlines()))
    cid = {r["field_name"]: int(r["class_id"]) for r in table}
    long_name = {r["field_name"]: r["long_name"] for r in table}
    members = {r["field_name"]: [int(v) for v in r["ids included"].split("-")] if r["ids included"] else []
               for r in table}
    atomic = {cid[n]: n for n in cid if not members[n]}                    # id -> name, atomic classes
    in_instance_challenge = {r["field_name"] for r in table if r["instance_challenge"] == "True"}

    # ---- 3. metadata of image and labels on S3
    em_levels = multiscale_levels(src.s3_json(em_key + "/.zattrs"))
    crop_attrs = src.s3_json(crop_key + "/.zattrs")
    listing = cache / "meta" / "listing.json"
    if args.refresh or not listing.exists():
        names = sorted(p.rstrip("/").split("/")[-1] for p in src.fs.ls(crop_key))
        save_json(listing, [n for n in names if not n.startswith(".")])
    stored = json.loads(listing.read_text())
    src.note(f"s3://{crop_key}/  (listing; .zattrs, <level>/.zattrs and <level>/.zarray of each of its {len(stored)} arrays)")
    info = {}
    src.s3_json_many([f"{crop_key}/{c}/.zattrs" for c in stored])
    level0 = {c: multiscale_levels(src.s3_json(f"{crop_key}/{c}/.zattrs", note=False))[0][0] for c in stored}
    src.s3_json_many([f"{crop_key}/{c}/{level0[c]}/{k}" for c in stored for k in (".zattrs", ".zarray")])
    for c in stored:
        za = src.s3_json(f"{crop_key}/{c}/.zattrs", note=False)
        lv = multiscale_levels(za)
        a0 = src.s3_json(f"{crop_key}/{c}/{lv[0][0]}/.zattrs", missing_ok=True, note=False) or {}
        ann = a0.get("cellmap", {}).get("annotation", {}) or za.get("cellmap", {}).get("annotation", {})
        arr = src.s3_json(f"{crop_key}/{c}/{lv[0][0]}/.zarray", note=False)
        info[c] = {"levels": lv, "type": ann.get("annotation_type", {}).get("type"),
                   "encoding": ann.get("annotation_type", {}).get("encoding"),
                   "counts": ann.get("complement_counts"), "shape": arr["shape"], "dtype": arr["dtype"]}
    lab_path, lab_scale, lab_trans = info["all"]["levels"][0]
    lab_shape = info["all"]["shape"]
    for c, i in info.items():                           # every class array lies on the same grid
        if i["levels"][0][1:] != (lab_scale, lab_trans) or i["shape"] != lab_shape:
            raise ValueError(f"class {c} is on another grid: {i['levels'][0]} {i['shape']}")
    if man_voxel != {tuple(lab_scale)} or man_trans != {tuple(lab_trans)} or man_shape != {tuple(lab_shape)}:
        notes.append(f"manifest of the challenge and zarr metadata disagree: manifest voxel {man_voxel}, "
                     f"translation {man_trans}, shape {man_shape}; zarr {lab_scale}, {lab_trans}, {lab_shape}")
    log(f"labels: level {lab_path}, voxel {lab_scale} nm, translation {lab_trans} nm, shape {lab_shape}; "
        f"{len(stored)} arrays stored")

    # ---- 4. the grid: the image level whose voxel is closest to the voxel of the labels
    def gap(level):
        return float(np.abs(np.log(np.array(level[1]) / np.array(lab_scale))).mean())

    em_path, em_scale, em_trans = min(em_levels, key=lambda lv: (round(gap(lv), 9), lv[1][0]))
    em_shape = src.s3_json(f"{em_key}/{em_path}/.zarray", note=False)["shape"]
    # A voxel index i has its centre at scale * i + translation (the translations of the
    # coarser levels of both pyramids are those of voxel centres). Edges of the label box:
    lo_nm = np.array(lab_trans) - np.array(lab_scale) / 2
    hi_nm = lo_nm + np.array(lab_shape) * np.array(lab_scale)
    em_edge0 = np.array(em_trans) - np.array(em_scale) / 2
    lo_f = (lo_nm - em_edge0) / np.array(em_scale)
    hi_f = (hi_nm - em_edge0) / np.array(em_scale)
    off = np.rint(lo_f).astype(int)
    shape = np.rint(hi_f).astype(int) - off
    factor = np.array(em_scale) / np.array(lab_scale)
    shift_vox = float(np.abs(lo_f - off).max())
    if shift_vox > 1e-6 or np.abs(hi_f - np.rint(hi_f)).max() > 1e-6:
        notes.append(f"the label box does not end on image voxel edges (up to {shift_vox:.3f} voxel); "
                     "the box was rounded to whole image voxels")
    if not np.allclose(factor, np.rint(factor)) or len(set(np.rint(factor).astype(int))) != 1:
        raise ValueError(f"image voxel {em_scale} is not a whole multiple of label voxel {lab_scale}")
    f = int(np.rint(factor[0]))
    log(f"image: level {em_path}, voxel {em_scale} nm, shape {em_shape}; crop offset {off.tolist()} "
        f"shape {shape.tolist()}; {f}x{f}x{f} label voxels per image voxel")
    if int(np.prod(shape)) > 300_000_000:
        raise ValueError("crop above 300 million voxels: a central sub-box is needed (not written, crop 313 is small)")

    # ---- 5. the image: one read of the context box, the crop is cut from it
    margin = np.rint(shape * (args.context - 1) / 2).astype(int)
    c_lo = np.maximum(off - margin, 0)
    c_hi = np.minimum(off + shape + margin, em_shape)
    ctx = src.s3_box(f"{em_key}/{em_path}", c_lo, c_hi, f"em_{em_path}")
    in_ctx = off - c_lo
    raw = np.ascontiguousarray(ctx[tuple(slice(o, o + s) for o, s in zip(in_ctx, shape))])
    if raw.dtype != np.uint8:
        raise ValueError(f"image dtype {raw.dtype}, expected uint8")

    # ---- 6. the labels at their own resolution
    lab = {}
    for c in stored:
        lab[c] = src.s3_box(f"{crop_key}/{c}/{lab_path}", [0, 0, 0], lab_shape, f"label_{c}_{lab_path}")
    allc = lab["all"]
    counts_all = np.bincount(allc.ravel(), minlength=256)
    checks = {}
    # 6a. stored voxel counts against the arrays
    bad = []
    for c in stored:
        if c == "all":
            continue
        want = info[c]["counts"] or {}
        n_present = int((lab[c] != 0).sum())
        n_unknown = int((lab[c] == 255).sum()) if info[c]["encoding"] and "unknown" in info[c]["encoding"] else 0
        if want.get("absent") is not None and want["absent"] != lab[c].size - n_present:
            bad.append(c)
        if n_unknown:
            notes.append(f"class {c}: {n_unknown} voxels marked unknown (255)")
    checks["complement_counts_equal_arrays"] = {"classes_checked": len(stored) - 1, "disagree": bad}
    # 6b. the array "all" holds the atomic class ids of the class table
    bad = []
    for i, n in atomic.items():
        if n in lab:
            if not np.array_equal(allc == i, lab[n] != 0):
                bad.append(n)
        elif counts_all[i]:
            bad.append(n + " (in 'all' but no array)")
    unknown_ids = [int(i) for i in np.flatnonzero(counts_all) if i and i not in atomic]
    checks["all_equals_atomic_arrays"] = {"atomic_arrays_checked": sum(n in lab for n in atomic.values()),
                                          "disagree": bad, "ids_not_in_class_table": unknown_ids}
    # 6c. every group array is the union of its atomic classes
    bad = []
    for n, m in members.items():
        if m and n in lab:
            if not np.array_equal(np.isin(allc, m), lab[n] != 0):
                bad.append(n)
    checks["group_arrays_equal_union_of_atomic"] = {"groups_checked": sum(bool(m) and n in lab for n, m in members.items()),
                                                    "disagree": bad}
    log("checks on the labels:", json.dumps(checks))
    annotated_present = sorted(c for c in stored if c != "all" and (lab[c] != 0).any())

    # ---- 7. classes.npy on the image grid
    classes, pure, tied = block_mode(allc, f)
    # The challenge stores coarser levels of its labels. The one at the voxel size of the
    # image must be the same array as the one computed here from the finest level.
    own = None
    for path, sc, tr in info["all"]["levels"]:
        if sc == em_scale:
            o2 = np.rint((np.array(tr) - np.array(em_trans)) / np.array(em_scale)).astype(int)
            own_arr = src.s3_box(f"{crop_key}/all/{path}", [0, 0, 0], shape, f"label_all_{path}")
            own = {"level": path, "translation_nm": tr, "offset_in_image_voxels": o2.tolist(),
                   "same_offset_as_computed": bool((o2 == off).all()),
                   "share_of_voxels_equal_to_classes_npy": float((own_arr == classes).mean())}
            if own["share_of_voxels_equal_to_classes_npy"] < 1 or not own["same_offset_as_computed"]:
                notes.append(f"classes.npy differs from the level {path} that the challenge stores: {own}")
    counts = np.bincount(classes.ravel(), minlength=256)
    present = {int(i): int(counts[i]) for i in np.flatnonzero(counts) if i}
    if counts[0]:
        notes.append(f"{int(counts[0])} voxels of classes.npy are 0 (no label)")
    # grey value of the image in each class (for the user of the crop; also shows that membranes are dark)
    r64 = raw.ravel().astype(np.float64)
    s1 = np.bincount(classes.ravel(), weights=r64, minlength=256)
    s2 = np.bincount(classes.ravel(), weights=r64 ** 2, minlength=256)
    grey = {str(i): {"name": atomic[i], "voxels": n, "raw_mean": round(s1[i] / n, 1),
                     "raw_std": round(float(np.sqrt(max(s2[i] / n - (s1[i] / n) ** 2, 0.0))), 1)}
            for i, n in present.items()}
    # the cells: the challenge has a "cell" array with one id per cell
    cells = None
    if "cell" in lab:
        ids, n = np.unique(lab["cell"], return_counts=True)
        cells = {str(int(i)): int(k) for i, k in zip(ids, n)}

    # ---- 8. seg.npy: objects
    import cc3d
    from scipy import ndimage
    group_of = {}                                        # atomic id -> object class name
    for i, n in atomic.items():
        group_of[i] = next((g for g in OBJECT_GROUPS if i in members.get(g, [])), n)
    order = [g for g in OBJECT_GROUPS] + sorted(set(group_of.values()) - set(OBJECT_GROUPS) - set(NOT_ORGANELLE)) \
        + list(NOT_ORGANELLE)
    obj_id = {g: (cid[g] if g in cid else 0) for g in order}
    lut = np.zeros(256, np.uint8)
    for i, g in group_of.items():
        lut[i] = obj_id[g]
    object_class = lut[classes]                          # class id of the object class per voxel
    seg = np.zeros(classes.shape, np.uint64)
    instances, class_rows, next_id = [], {}, 1
    for g in order:
        mask = object_class == obj_id[g]
        if not mask.any():
            continue
        atoms = sorted(i for i, gg in group_of.items() if gg == g)
        has_inst = g in lab and info[g]["type"] == "instance_segmentation"
        if has_inst:
            fine = lab[g]
            if not np.array_equal(fine != 0, np.isin(allc, atoms)):
                raise ValueError(f"instance array {g} does not cover the voxels of its atomic classes {atoms}")
            local = block_mode(fine, f, ignore_zero=True)[0].astype(np.int64)
            local[~mask] = 0
            if (local[mask] == 0).any():
                raise ValueError(f"{g}: voxels of the class without an instance id")
            how = "challenge"
            # Are the instances of the challenge the connected components of its mask?
            fine_ids = np.unique(fine[fine != 0])
            same = {}
            for conn in (6, 18, 26):
                cc = cc3d.connected_components(fine != 0, connectivity=conn)
                pairs = np.unique(cc[fine != 0].astype(np.int64) * 1000 + fine[fine != 0])
                same[str(conn)] = bool(len(pairs) == len(fine_ids) == int(cc.max()))
                same[f"components_{conn}"] = int(cc.max())
            extra = {"instances_at_label_resolution": int(len(fine_ids)),
                     "source_ids_lost_on_the_image_grid": [int(i) for i in fine_ids if not (local == i).any()],
                     "challenge_ids_are_the_connected_components_of_the_mask": same}
        else:
            local = cc3d.connected_components(mask, connectivity=26).astype(np.int64)
            how = "connected_components_26"
            extra = {}
        ids = np.unique(local[local > 0])
        boxes = ndimage.find_objects(local)
        first = next_id
        for sid in ids:
            box = boxes[int(sid) - 1]
            m = local[box] == sid
            zi, yi, xi = np.nonzero(m)
            lo_i = [int(b.start) for b in box]
            hi_i = [int(b.stop) for b in box]
            seg[box][m] = next_id
            instances.append({
                "id": next_id, "class": g, "source_id": int(sid), "voxels": int(m.sum()),
                "box_vox_zyx": [lo_i, hi_i],
                "centre_nm_zyx": [round(float((v.mean() + o + 0.5) * sc), 1)
                                  for v, o, sc in zip((zi, yi, xi), lo_i, em_scale)],
                "components_26": int(cc3d.connected_components(m, connectivity=26).max()),
                "touches_crop_face": bool(any(l == 0 for l in lo_i) or any(h == n for h, n in zip(hi_i, classes.shape))),
            })
            next_id += 1
        class_rows[g] = {
            "class_id": obj_id[g], "long_name": long_name.get(g, g),
            "atomic_class_ids": atoms, "ids": how,
            "challenge_array_type": info[g]["type"] if g in lab else None,
            "scored_as_instances_in_the_challenge": g in in_instance_challenge,
            "kind": NOT_ORGANELLE.get(g, "organelle"),
            "instances": len(ids), "voxels": int(mask.sum()), "seg_id_first": first, "seg_id_last": next_id - 1,
            **extra,
        }
        log(f"seg: {g:6s} {how:24s} {len(ids):5d} ids, {int(mask.sum()):9d} voxels")
    if (seg == 0).any():
        notes.append(f"{int((seg == 0).sum())} voxels of seg.npy are 0")
    cyto_ids = [r["id"] for r in instances if r["class"] in NOT_ORGANELLE]
    by_challenge = [g for g, r in class_rows.items() if r["ids"] == "challenge"]
    by_components = [g for g, r in class_rows.items() if r["ids"] != "challenge"]
    conn6 = [g for g in by_challenge if class_rows[g]["challenge_ids_are_the_connected_components_of_the_mask"]["6"]]
    conn26 = [g for g in by_challenge if class_rows[g]["challenge_ids_are_the_connected_components_of_the_mask"]["26"]]
    pieces = [r["id"] for r in instances if r["components_26"] > 1]

    # ---- 9. is the crop where the labels are? (shift and axis tests against the image)
    nlab = 256
    span = 4
    ok_shift = all(in_ctx[a] >= span and in_ctx[a] + shape[a] + span <= ctx.shape[a] for a in range(3))
    align = {"measure": "share of the variance of raw explained by the class of each voxel (eta squared)",
             "at_zero_shift": round(eta2(raw, classes, nlab), 4)}
    if ok_shift:
        best = (None, -1.0)
        along = {}
        for a, an in enumerate("zyx"):
            vals = {}
            for d in range(-span, span + 1):
                o = in_ctx.copy()
                o[a] += d
                r = ctx[tuple(slice(v, v + s) for v, s in zip(o, shape))]
                vals[str(d)] = round(eta2(r, classes, nlab), 4)
            along[an] = vals
        for d in itertools.product(range(-2, 3), repeat=3):
            o = in_ctx + np.array(d)
            e = eta2(ctx[tuple(slice(v, v + s) for v, s in zip(o, shape))], classes, nlab)
            if e > best[1]:
                best = (list(d), e)
        align["raw_shifted_by_voxels_along_one_axis"] = along
        align["best_shift_zyx_in_5x5x5"] = best[0]
    perms = {}
    for p in itertools.permutations(range(3)):
        if len({shape[i] for i in range(3)}) == 1:
            perms["".join("zyx"[i] for i in p)] = round(eta2(raw, classes.transpose(p), nlab), 4)
    align["labels_with_axes_permuted"] = perms
    align["labels_flipped"] = {an: round(eta2(raw, np.flip(classes, a), nlab), 4) for a, an in enumerate("zyx")}
    log("alignment:", json.dumps(align))

    # ---- 10. the mesh file
    glb_result = None
    if glb.exists():
        obj_names = {v: k for k, v in obj_id.items() if v}
        page = src.http("challenge_front_page.html", required=False)
        hotspots = parse_hotspots(page) if page else []
        if not hotspots:
            notes.append("the labels of the 3D model on the front page of the challenge could not be read")
        class_of_text = {v.lower(): k for k, v in long_name.items()}
        object_of = {atomic[i]: g for i, g in group_of.items()}
        glb_result = glb_check(glb, classes, object_class, atomic, obj_names, object_of, em_scale, hotspots,
                               class_of_text)
        save_json(out / "glb_check.json", glb_result)
        notes.append("Mesh file crop313.glb (see glb_check.json): it holds no class names; " + glb_result["summary"])
        for m in glb_result["meshes"]:
            log(f"glb: {m['node']:12s} {m['material']:14s} {m['triangles']:7d} triangles -> {m['best_class']}  "
                f"website: {m['website_labels']}  {m['matches'][:2]}")
    else:
        notes.append(f"mesh file {glb} not found: no comparison with the meshes")

    # ---- 11. licence and citation, from the records of the two data sets
    lic, cite = {}, {}
    record_classes = None                               # classes the record of the crop lists as annotated
    for key, what in (("figshare_labels.json", "labels"), ("figshare_image.json", "image")):
        text = src.http(key, required=False)
        if text:
            rec = json.loads(text)
            lic[what] = f"{rec['license']['name']} ({rec['license']['url']})"
            cite[what] = rec["citation"]
            if what == "labels":
                import html
                import re
                plain = html.unescape(re.sub(r"<[^>]+>", " ", rec.get("description", "")))
                m = re.search(r"Classes annotated\s*:\s*(.*?)\s*Dataset URL", plain, flags=re.S)
                if m:
                    by_long = {v.lower(): k for k, v in long_name.items()}
                    names = [re.sub(r"\s+", " ", t).strip().lower() for t in m.group(1).split(",")]
                    record_classes = sorted(by_long.get(t, "?" + t) for t in names if t)
    text = src.http("figshare_collection.json", required=False)
    if text:
        cite["collection"] = json.loads(text)["citation"]
    for what, doi in (("labels", "https://doi.org/10.25378/janelia.24261094"),
                      ("image", "https://doi.org/10.25378/janelia.24128700")):
        if what not in lic:
            lic[what] = (f"CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), as read from {doi} on "
                         "2026-11-23; the record could not be read again in this run")
            cite[what] = doi
    cite.setdefault("collection", "CellMap Project Team et al. (2024). CellMap 2024 Segmentation Challenge. "
                                  "Janelia Research Campus. Collection. https://doi.org/10.25378/janelia.c.7456966")

    # ---- 12. write the crop
    class_names = {str(i): atomic[i] for i in sorted(atomic)}
    level_text = (f"image: level {em_path} of {dataset} recon-1/em/fibsem-uint8, voxel "
                  f"{em_scale[0]:g} x {em_scale[1]:g} x {em_scale[2]:g} nm (z, y, x), the finest level stored. "
                  f"labels: level {lab_path} of crop{CROP}, voxel {lab_scale[0]:g} x {lab_scale[1]:g} x {lab_scale[2]:g} nm, "
                  f"brought to the image grid ({f} x {f} x {f} label voxels per image voxel)")
    n_org = sum(1 for r in instances if r["class"] not in NOT_ORGANELLE)
    notes = [
        f"Axis order of every array and of every coordinate is z, y, x. Voxel [0, 0, 0] is voxel {off.tolist()} of "
        f"the image level {em_path}. The centre of voxel i is at (offset_vox + i) * voxel_nm in the nanometre frame "
        "of the source, and at (i + 0.5) * voxel_nm from the corner of the crop (instances.json uses the latter).",
        f"The labels of the challenge have a {lab_scale[0]:g} nm voxel, the image a {em_scale[0]:g} nm voxel "
        f"(the image has no finer level). The crop is on the image grid. The label box "
        f"({lab_shape} voxels at translation {lab_trans} nm) ends exactly on image voxel edges"
        + ("" if shift_vox <= 1e-6 else " after rounding") + f", so each image voxel holds {f ** 3} label voxels.",
        f"classes.npy: most frequent atomic class of the {f ** 3} label voxels of each image voxel; of a tie, the "
        f"class that comes first in the block in z, y, x order. {pure * 100:.1f} % of the image voxels hold one "
        f"class only; {tied * 100:.2f} % are ties. "
        + (f"The result is voxel for voxel the level {own['level']} that the challenge itself stores for the labels "
           f"at {em_scale[0]:g} nm. " if own and own["share_of_voxels_equal_to_classes_npy"] == 1 else "")
        + f"classes_{lab_scale[0]:g}nm.npy holds the labels unresampled: shape {lab_shape}, voxel "
        f"{lab_scale[0]:g} nm; image voxel i holds label voxels {f} * i .. {f} * i + {f - 1} on each axis.",
        "seg.npy: for the classes " + ", ".join(by_challenge)
        + " the instance ids of the challenge (a voxel of such a class takes the most frequent instance id of that "
        "class among its label voxels); for " + ", ".join(by_components)
        + " the challenge gives a mask only, and the ids are the 26-connected components of the mask on the image "
        "grid. Ids run from 1 without gaps, class after class; instances.json gives class and size of each.",
        "The instance ids of the challenge are, in this crop, exactly the 6-connected components of its class mask "
        f"at {lab_scale[0]:g} nm for " + (", ".join(conn6) or "no class")
        + (f" (for {', '.join(g for g in conn6 if g not in conn26)} the 26-connected components are fewer: objects "
           "that touch by an edge or a corner have separate ids)" if set(conn6) - set(conn26) else "")
        + ". So both kinds of ids are components of a manual mask, and no two instances of a class touch face to face. "
        + (f"On the image grid every id is one 26-connected piece." if not pieces else
           f"On the image grid the ids {pieces} are in more than one 26-connected piece (components_26 in instances.json)."),
        f"Ids 1 to {n_org} are organelles. The ids of class cyto ({len(cyto_ids)} of them) are the cytosol between "
        "the organelles, not objects; leave them out for work on organelles.",
        f"Every voxel of the crop has a class: the annotation is dense. Of the {len(stored) - 1} class arrays stored "
        f"for the crop, {len(annotated_present)} have voxels. The others are empty; the metadata of the source count "
        "their voxels as 'absent', not as 'unknown', and the manifest of the challenge lists "
        f"{len(set(listed_classes) - set(annotated_present))} of them for the crop. "
        + ("The record of the crop (labels DOI) names as annotated exactly the classes with voxels. "
           if record_classes == annotated_present else
           f"The record of the crop (labels DOI) names as annotated: {record_classes}. " if record_classes else "")
        + "So the source does not settle whether an empty array means 'looked for and not there' or 'not annotated'.",
        "A structure without a class of its own carries the label of what is around it. The dark granules in the "
        "cytosol (preview_ortho.png) are labelled cyto: the challenge has a glycogen class but stores no array of it "
        "for this crop, and the source does not say what the granules are.",
        "The er object class is the union of er_mem, er_lum and the exit sites; the challenge's own er group also "
        "counts nuclear envelope and pores, which this crop does not contain.",
    ] + notes
    meta = {
        "name": NAME,
        "dataset": f"CellMap segmentation challenge, crop {CROP} of {dataset} (FIB-SEM, mouse liver)",
        "voxel_nm": em_scale,
        "shape": [int(v) for v in shape],
        "offset_vox": [int(v) for v in off],
        "level": level_text,
        "sources": list(src.read),
        "seg_kind": "manual",
        "seg_meaning": ("One id per organelle, from the manual class masks of the challenge: for "
                        + ", ".join(by_challenge) + " the instance ids of the challenge, for "
                        + ", ".join(g for g in by_components if g not in NOT_ORGANELLE)
                        + " the connected components of the mask; the last ids are the pieces of cytosol between them."),
        "class_names": class_names,
        "licence": f"labels: {lic['labels']}; image: {lic['image']}",
        "citation": " | ".join(cite[k] for k in ("labels", "image", "collection")),
        "fetched": FETCHED,
        "notes": notes,
        # ---- beyond the common fields
        "axis_order": "z, y, x",
        "class_long_names": {str(i): long_name[atomic[i]] for i in sorted(atomic)},
        "class_voxels": {str(i): n for i, n in present.items()},
        "classes_present": grey,
        "cell_array_ids_and_voxels_at_label_resolution": cells,
        "class_voxels_at_label_resolution": {str(int(i)): int(counts_all[i]) for i in np.flatnonzero(counts_all) if i},
        "object_classes": class_rows,
        "label_resolution_nm": lab_scale,
        "label_translation_nm": lab_trans,
        "label_shape": lab_shape,
        "classes_listed_for_the_crop_in_the_challenge_manifest": listed_classes,
        "arrays_stored_for_the_crop": [c for c in stored if c != "all"],
        "arrays_with_voxels": annotated_present,
        "classes_annotated_according_to_the_record_of_the_crop": record_classes,
        "label_checks": checks,
        "challenge_own_level_at_image_voxel": own,
        "alignment_check": align,
        "extra_files": {
            "instances.json": "class, voxel count, box and centre (nm, z y x, from the crop corner) of every id of seg.npy",
            f"classes_{lab_scale[0]:g}nm.npy": f"uint8 {lab_shape}: atomic class ids at the resolution of the labels",
            "glb_check.json": "content of the mesh file crop313.glb compared with the labels",
            "preview_ortho.png": "the three orthogonal middle sections",
        },
        "context_crop": {"name": NAME + "_context", "offset_of_this_crop_in_it_vox": [int(v) for v in in_ctx]},
    }
    save_npy(out / "raw.npy", raw)
    save_npy(out / "classes.npy", classes)
    save_npy(out / "seg.npy", seg)
    save_npy(out / f"classes_{lab_scale[0]:g}nm.npy", allc)
    save_json(out / "instances.json", {
        "crop": NAME,
        "coordinates": "nanometres from the corner of the crop, axis order z, y, x; boxes in voxels, upper end excluded",
        "classes": class_rows,
        "instances": instances,
    })
    save_json(out / "meta.json", meta)
    write_previews(out, raw, seg, classes, atomic, cyto_ids, em_scale)

    # ---- 13. write the context crop
    meta_ctx = {
        "name": NAME + "_context",
        "dataset": f"CellMap segmentation challenge, surroundings of crop {CROP} in {dataset} (FIB-SEM, mouse liver)",
        "voxel_nm": em_scale,
        "shape": [int(v) for v in ctx.shape],
        "offset_vox": [int(v) for v in c_lo],
        "level": f"level {em_path} of {dataset} recon-1/em/fibsem-uint8, voxel {em_scale[0]:g} x {em_scale[1]:g} x "
                 f"{em_scale[2]:g} nm (z, y, x), the finest level stored",
        "sources": [u for u in src.read if not any(k in u for k in (
            "labels/groundtruth", "articles/24261094", "train_crop_manifest", "classes.csv", "cellmapchallenge.janelia.org"))],
        "seg_kind": "none",
        "seg_meaning": "No labels: the challenge annotated only the box of the crop inside this image.",
        "class_names": {},
        "licence": lic["image"],
        "citation": " | ".join(cite[k] for k in ("image", "collection")),
        "fetched": FETCHED,
        "notes": [
            "Image only. Axis order z, y, x.",
            f"The labelled crop {NAME} is the box that starts at voxel {[int(v) for v in in_ctx]} of this array and "
            f"has shape {[int(v) for v in shape]}; raw.npy of {NAME} is exactly that part of this array.",
            f"The box is {args.context:g} times the crop in each direction, centred on it"
            + ("" if (c_lo == off - margin).all() and (c_hi == off + shape + margin).all()
               else ", cut at the end of the volume") + ".",
        ],
        "axis_order": "z, y, x",
        "crop": {"name": NAME, "offset_vox": [int(v) for v in in_ctx], "shape": [int(v) for v in shape]},
    }
    save_npy(out_ctx / "raw.npy", ctx)
    save_json(out_ctx / "meta.json", meta_ctx)
    write_context_preview(out_ctx, ctx, [int(v) for v in in_ctx], [int(v) for v in shape], em_scale, NAME + "_context")

    log(f"wrote {out} : raw {raw.shape} {raw.dtype}, classes {len(present)} classes, seg {len(instances)} ids")
    log(f"wrote {out_ctx} : raw {ctx.shape} {ctx.dtype}")
    for n in notes:
        log("note:", n)


if __name__ == "__main__":
    sys.exit(main())

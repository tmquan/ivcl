#!/usr/bin/env python3
"""fetch_cremi3d.py -- CREMI sample A, written as two crops in the common crop format.

SOURCE
    CREMI, the MICCAI 2016 Challenge on Circuit Reconstruction from Electron
    Microscopy Images, https://cremi.org/data/ , training "Dataset A", cropped
    version:
        https://cremi.org/static/data/sample_A_20160501.hdf   (175 159 177 bytes)
    Serial-section transmission EM of adult Drosophila melanogaster brain.
    One HDF5 file ("file_format" 0.2) with
        volumes/raw                 uint8   [125, 1250, 1250]  (z, y, x)
        volumes/labels/neuron_ids   uint64  same shape, neuron segmentation
        volumes/labels/clefts       uint64  same shape, synaptic cleft segmentation
                                    (0xffffffffffffffff = no cleft)
        annotations/{ids, types, locations, presynaptic_site/partners, comments}
                                    pre- and postsynaptic partner sites, locations
                                    as (z, y, x) in nm
    Every volume carries the attribute resolution = [40, 4, 4] nm (z, y, x). This
    script reads the voxel size from that attribute; it is not assumed.
    Who made the labels, in the words of https://cremi.org/data/ :
        Neuron segmentation: "Initial segmentations by Peter Li and Jan Funke,
            proof-read and fixed by Ariadne Service GmbH and Chris Ordish,
            partially reviewed by Stephan Saalfeld and Jan Funke."
        Synaptic cleft segmentation: "Provided by Julia Buhmann and Chong Zhang,
            reviewed by Jan Funke."
        Synaptic partner annotations: "Provided by Jan Funke, reviewed by Julia
            Buhmann."
    Conventions taken from the reference code https://github.com/cremi/cremi_python :
        voxel index of a location = round(location / resolution)  (cremi/Volume.py)
        clefts: 0xffffffffffffffff = background, 0xfffffffffffffffe = invalid
        (cremi/evaluation/Clefts.py)

LICENCE
    The source states none. The pages https://cremi.org/ , /data/ , /metrics/ ,
    /leaderboard/ and /register/ carry no licence or terms-of-use text, and the
    cremi_python repository has no licence file (read on 2026-11-23). The terms
    of redistribution are therefore unknown; the organisers' contact given on
    the site is challenge@cremi.org.

CITATION
    The source states none (the home page says only that the results "will be
    summarized in a review paper"). Organisers named on https://cremi.org/ :
    Jan Funke, Stephan Saalfeld, Davi Bock, Srini Turaga, Eric Perlman.

WRITES  (the common crop format; all arrays [z, y, x], C order)
    <out>/                 crop "cremi3d": 64 x 512 x 512 voxels at the native
                           resolution. Its position is chosen by a fixed rule,
                           see choose_origin(): the window that holds the most
                           whole synaptic clefts, then the most complete
                           pre/post partner pairs.
    <out>_full/            crop "cremi3d_full": the whole sample, 125 x 1250 x 1250.
    In each folder:
        raw.npy            uint8   the EM image
        seg.npy            uint64  neuron ids (the ids of the source, unchanged)
        classes.npy        uint8   1 = synaptic cleft, 0 = none
        cleft_ids.npy      uint64  cleft id of the source, 0 = no cleft
        synapses.json      the annotated pre/post partner pairs inside the crop,
                           in nm relative to the crop origin, axis order z, y, x,
                           with the neuron id found in seg at each site
        meta.json          voxel size, offset, sources, licence, notes; also "stats"
                           (sizes of the ids, connected components, clefts, pairs)
                           and "checks" (label boundaries on membranes, axis order
                           of the annotations, shifts between neighbouring sections)
        preview.png        middle section: raw | raw + seg colours + boundaries |
                           raw + clefts
        preview_synapses.png   the section with the most cleft voxels, with the
                           partner sites drawn (a check of the annotation axes)
    The source file is cached under <out>/../../src/<name>/ . If a copy already
    lies at <out>/../../sample_A_20160501.hdf it is linked, not downloaded again.

USAGE  (on the Spark; safe to run again, every file is replaced atomically)
    ~/connectome/render-venv/bin/python fetch_cremi3d.py \
        --out ~/connectome/data/crops/cremi3d
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://cremi.org/static/data/sample_A_20160501.hdf"
DOC_URL = "https://cremi.org/data/"
CODE_URL = "https://github.com/cremi/cremi_python"
FILE = "sample_A_20160501.hdf"
# SIZE = the Content-Length that URL returned on 2026-11-23. The source
# publishes no checksum; SHA256 is that of the copy the crops were first made
# from, kept here so that a different file cannot be used unnoticed. On
# 2026-11-23 five ranges of 1 MiB of the URL (at the start, at 40, 90 and
# 140 MB, and at the end) were compared with that copy and were equal.
SIZE = 175159177
SHA256 = "4c563d1b78acb2bcfb3ea958b6fe1533422f7f4a19f3e05b600bfa11430b510d"

CLEFT_BACKGROUND = np.uint64(0xFFFFFFFFFFFFFFFF)
CLEFT_INVALID = np.uint64(0xFFFFFFFFFFFFFFFE)

LICENCE = (
    "Not stated by the source: https://cremi.org/ and its data, metrics, "
    "leaderboard and register pages carry no licence or terms-of-use text, and "
    "https://github.com/cremi/cremi_python has no licence file (read 2026-11-23). "
    "Terms of redistribution unknown; contact given by the source: challenge@cremi.org."
)
CITATION = (
    "Not stated by the source. Name of the source as it gives it: CREMI, MICCAI "
    "Challenge on Circuit Reconstruction from Electron Microscopy Images (2016), "
    "https://cremi.org/ ; organisers Jan Funke, Stephan Saalfeld, Davi Bock, "
    "Srini Turaga, Eric Perlman. Credits for the image data and the labels: "
    "'Acknowledgements' on https://cremi.org/data/ ."
)
LABEL_CREDITS = {
    "neuron segmentation": "Initial segmentations by Peter Li and Jan Funke, proof-read and fixed "
                           "by Ariadne Service GmbH and Chris Ordish, partially reviewed by Stephan "
                           "Saalfeld and Jan Funke.",
    "synaptic cleft segmentation": "Provided by Julia Buhmann and Chong Zhang, reviewed by Jan Funke.",
    "synaptic partner annotations": "Provided by Jan Funke, reviewed by Julia Buhmann.",
}

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


# --------------------------------------------------------------------------- source

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def find_source(src_dir, existing):
    """Return (path, how). Cache first, then an existing local copy (linked), then download."""
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / FILE
    if path.is_symlink() and not path.exists():      # a link whose target is gone
        path.unlink()
    if path.exists():
        how = "found in the cache folder (symlink to an existing local copy)" if path.is_symlink() \
            else "found in the cache folder"
    elif existing is not None and existing.exists():
        os.symlink(existing.resolve(), path)
        how = "symlink to an existing local copy; not downloaded by this script"
    else:
        tmp = src_dir / (FILE + ".part")
        log("downloading", URL)
        req = urllib.request.Request(URL, headers={"User-Agent": "fetch_cremi3d.py (research script)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            while True:
                block = r.read(1 << 22)
                if not block:
                    break
                fh.write(block)
        os.replace(tmp, path)
        how = "downloaded by this script from " + URL
    size = path.stat().st_size
    digest = sha256_of(path)
    if size != SIZE or digest != SHA256:
        raise SystemExit("source file %s is not the expected one: size %d (expected %d), sha256 %s (expected %s)"
                         % (path, size, SIZE, digest, SHA256))
    return path, how


def load_source(path):
    import h5py
    src = {}
    with h5py.File(path, "r") as f:
        res = None
        for key in ("volumes/raw", "volumes/labels/neuron_ids", "volumes/labels/clefts"):
            d = f[key]
            r = tuple(float(v) for v in d.attrs["resolution"])
            if res is None:
                res = r
            if r != res:
                raise SystemExit("resolution of %s is %s, of volumes/raw %s" % (key, r, res))
            off = d.attrs.get("offset")
            if off is not None and any(float(v) != 0.0 for v in off):
                raise SystemExit("%s has an offset %s: this is a padded file, not handled here" % (key, off))
        off = f["annotations"].attrs.get("offset")
        if off is not None and any(float(v) != 0.0 for v in off):
            raise SystemExit("annotations have an offset %s: not handled here" % (off,))
        src["file_format"] = str(f.attrs.get("file_format"))
        src["resolution"] = res
        src["raw"] = f["volumes/raw"][:]
        src["neuron_ids"] = f["volumes/labels/neuron_ids"][:]
        src["clefts"] = f["volumes/labels/clefts"][:]
        a = f["annotations"]
        src["ann_ids"] = a["ids"][:].astype(np.int64)
        src["ann_types"] = [t.decode() if isinstance(t, bytes) else str(t) for t in a["types"][:]]
        src["ann_locations"] = a["locations"][:].astype(np.float64)
        src["partners"] = a["presynaptic_site/partners"][:].astype(np.int64)
        src["comments"] = {}
        if "comments" in a:
            for i, c in zip(a["comments/target_ids"][:], a["comments/comments"][:]):
                src["comments"][int(i)] = c.decode() if isinstance(c, bytes) else str(c)
    if src["raw"].dtype != np.uint8 or src["neuron_ids"].dtype != np.uint64 or src["clefts"].dtype != np.uint64:
        raise SystemExit("unexpected dtypes in the source file")
    if not (src["raw"].shape == src["neuron_ids"].shape == src["clefts"].shape):
        raise SystemExit("the three volumes do not have one shape")
    return src


def to_voxel(loc_nm, res):
    """Voxel index of a location in nm: round(location / resolution), as cremi/Volume.py does."""
    return np.floor(np.asarray(loc_nm, np.float64) / np.asarray(res, np.float64) + 0.5).astype(np.int64)


# --------------------------------------------------------------------------- choice of the crop

def choose_origin(cleft_ids, pre_vox, post_vox, crop_shape, stride_yx=16):
    """The rule that places the small crop.

    Candidates: every window of crop_shape with z origin at every section and
    y, x origins on a grid of stride_yx voxels (plus the last possible origin).
    A cleft is "whole" in a window if the bounding box of its id lies inside
    the window; a partner pair is "complete" if both sites do.
    The chosen window has the most whole clefts; ties go to the most complete
    pairs, then to the smallest (z, y, x) origin.
    """
    shape = cleft_ids.shape
    if any(c > s for c, s in zip(crop_shape, shape)):
        raise SystemExit("crop %s does not fit in %s" % (crop_shape, shape))
    co = np.stack(np.nonzero(cleft_ids), 1)
    ids, inv = np.unique(cleft_ids[co[:, 0], co[:, 1], co[:, 2]], return_inverse=True)
    lo = np.full((len(ids), 3), np.iinfo(np.int64).max, np.int64)
    hi = np.full((len(ids), 3), -1, np.int64)
    np.minimum.at(lo, inv, co)
    np.maximum.at(hi, inv, co)
    origins = []
    for d in range(3):
        last = shape[d] - crop_shape[d]
        origins.append(np.unique(np.r_[np.arange(0, last + 1, 1 if d == 0 else stride_yx), last]))

    def inside(lo_d, hi_d, o, n):
        return ((lo_d[None, :] >= o[:, None]) & (hi_d[None, :] < o[:, None] + n)).astype(np.int32)

    c = [inside(lo[:, d], hi[:, d], origins[d], crop_shape[d]) for d in range(3)]
    n_cleft = np.einsum("zn,yn,xn->zyx", *c)
    plo, phi = np.minimum(pre_vox, post_vox), np.maximum(pre_vox, post_vox)
    p = [inside(plo[:, d], phi[:, d], origins[d], crop_shape[d]) for d in range(3)]
    n_pair = np.einsum("zn,yn,xn->zyx", *p)
    key = n_cleft.astype(np.int64) * 1000000 + n_pair
    k = np.unravel_index(int(np.argmax(key)), key.shape)   # first maximum = smallest (z, y, x)
    origin = tuple(int(origins[d][k[d]]) for d in range(3))
    info = {
        "n_candidates": int(key.size),
        "whole_clefts": int(n_cleft[k]),
        "complete_pairs": int(n_pair[k]),
        "mean_whole_clefts_over_candidates": round(float(n_cleft.mean()), 2),
        "mean_complete_pairs_over_candidates": round(float(n_pair.mean()), 2),
        "n_candidates_with_the_same_score": int((key == key[k]).sum()),
    }
    return origin, info


# --------------------------------------------------------------------------- measurements

def seg_stats(seg, raw):
    """Counts that a user of seg must know: sizes of the ids, sections they span, connected components."""
    import cc3d
    try:
        import fastremap
        dense, _ = fastremap.renumber(seg, in_place=False)       # ids 1..N, in order of first appearance
        dense = np.asarray(dense)
    except ImportError:
        _, inv = np.unique(seg, return_inverse=True)
        dense = (inv.reshape(seg.shape) + 1).astype(np.uint32)
    n = int(dense.max())
    counts = np.bincount(dense.ravel(), minlength=n + 1)
    nz = seg.shape[0]
    first = np.full(n + 1, nz, np.int64)
    last = np.full(n + 1, -1, np.int64)
    for z in range(nz):
        u = np.unique(dense[z])
        first[u] = np.minimum(first[u], z)
        last[u] = z
    extent = last - first + 1
    cc, n_cc = cc3d.connected_components(dense, connectivity=26, return_N=True)
    cc_size = np.bincount(cc.ravel(), minlength=n_cc + 1)
    id_of_cc = np.zeros(n_cc + 1, np.int64)
    id_of_cc[cc.ravel()] = dense.ravel()                         # every voxel of a component has one id
    n_comp = np.bincount(id_of_cc[1:], minlength=n + 1)
    largest = np.zeros(n + 1, np.int64)
    np.maximum.at(largest, id_of_cc[1:], cc_size[1:])
    real = np.arange(n + 1) > 0
    big = real & (counts >= 1000)
    one = real & (extent == 1)
    small_vox = ~big[dense]
    return {
        "n_ids": n,
        "n_voxels_with_id_0": int((seg == 0).sum()),
        "n_ids_with_1000_voxels_or_more": int(big.sum()),
        "voxel_share_of_ids_with_1000_voxels_or_more": round(float(counts[big].sum() / seg.size), 5),
        "n_ids_present_in_one_section_only": int(one.sum()),
        "median_voxels_of_ids_present_in_one_section_only": float(np.median(counts[one])) if one.any() else None,
        "max_voxels_of_ids_present_in_one_section_only": int(counts[one].max()) if one.any() else None,
        "n_ids_present_in_every_section": int((real & (extent == nz)).sum()),
        "largest_id_voxel_share": round(float(counts[real].max() / seg.size), 4),
        "mean_raw_in_ids_below_1000_voxels": round(float(raw[small_vox].mean()), 2) if small_vox.any() else None,
        "n_connected_components_26": int(n_cc),
        "n_ids_with_1000_voxels_or_more_in_more_than_one_component_26": int((big & (n_comp > 1)).sum()),
        "n_ids_with_1000_voxels_or_more_whose_largest_component_26_holds_under_99_percent":
            int((big & (largest < 0.99 * counts)).sum()),
        "smallest_share_of_the_largest_component_among_ids_with_1000_voxels_or_more":
            round(float((largest[big] / counts[big]).min()), 3) if big.any() else None,
    }


def boundary_mask(seg):
    """In-plane label boundaries: a voxel whose left, right, upper or lower neighbour has another id."""
    b = np.zeros(seg.shape, bool)
    d = seg[:, :, 1:] != seg[:, :, :-1]
    b[:, :, 1:] |= d
    b[:, :, :-1] |= d
    d = seg[:, 1:, :] != seg[:, :-1, :]
    b[:, 1:, :] |= d
    b[:, :-1, :] |= d
    return b


def alignment_check(raw, seg, margin=16):
    """Mean raw value under the label boundaries, for the labels as stored and
    for shifted, flipped and transposed labels. Membranes are dark in this
    data, so the stored position must give the lowest value."""
    nz, h, w = raw.shape
    zs = np.arange(1, nz - 1, max(1, nz // 16))
    b = boundary_mask(seg[zs])
    m = margin
    core = b[:, m:h - m, m:w - m]

    def mean_under(mask, dz=0, dy=0, dx=0):
        r = raw[zs + dz][:, m + dy:h - m + dy, m + dx:w - m + dx]
        return float(r[mask].mean())

    shifts = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    grid = {(dy, dx): mean_under(core, 0, dy, dx) for dy in shifts for dx in shifts}
    best = min(grid, key=grid.get)
    zero = grid[(0, 0)]
    r0 = raw[zs][:, m:h - m, m:w - m]
    out = {
        "what": "mean raw value (0..255) under the in-plane label boundaries; membranes are dark, "
                "so the lowest value marks the best fit",
        "sections_used": [int(z) for z in zs],
        "labels_as_stored": round(zero, 2),
        "all_voxels": round(float(r0.mean()), 2),
        "voxels_off_the_boundaries": round(float(r0[~core].mean()), 2),
        "best_in_plane_shift_yx_voxels": [int(best[0]), int(best[1])],
        "value_at_best_shift": round(grid[best], 2),
        "shift_1_voxel_yx": {"%+d,%+d" % k: round(grid[k], 2) for k in [(-1, 0), (1, 0), (0, -1), (0, 1)]},
        "shift_8_voxels_yx": {"%+d,%+d" % k: round(grid[k], 2) for k in [(-8, 0), (8, 0), (0, -8), (0, 8)]},
        "labels_one_section_up": round(mean_under(core, +1), 2),
        "labels_one_section_down": round(mean_under(core, -1), 2),
        "labels_flipped_in_y": round(mean_under(b[:, ::-1, :][:, m:h - m, m:w - m]), 2),
        "labels_flipped_in_x": round(mean_under(b[:, :, ::-1][:, m:h - m, m:w - m]), 2),
    }
    if h == w:
        out["labels_transposed_yx"] = round(mean_under(b.transpose(0, 2, 1)[:, m:h - m, m:w - m]), 2)
    others = [v for k, v in out.items() if k.startswith("labels_") and k != "labels_as_stored"]
    out["stored_position_is_the_minimum"] = bool(best == (0, 0) and all(zero < v for v in others))
    return out


def section_shifts(raw):
    """In-plane translation between neighbouring sections, estimated by the peak of the
    cross-correlation of the two whole sections (whole voxels). (dy, dx) is the move of
    section z+1 that fits it best to section z; (0, 0) means the sections are registered."""
    nz, h, w = raw.shape
    shifts, corr = [], []
    for z in range(nz - 1):
        a = raw[z].astype(np.float32)
        b = raw[z + 1].astype(np.float32)
        a -= a.mean()
        b -= b.mean()
        cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
        k = np.unravel_index(int(np.argmax(cc)), cc.shape)
        shifts.append([int(k[0]) if k[0] <= h // 2 else int(k[0]) - h,
                       int(k[1]) if k[1] <= w // 2 else int(k[1]) - w])
        corr.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())))
    mag = np.abs(np.array(shifts)).max(1)
    worst = int(np.argmax(mag))
    return {
        "what": "estimated in-plane shift (dy, dx), in voxels, of section z+1 against section z: the peak "
                "of the cross-correlation of the two whole sections; index z of the list is the pair "
                "(z, z+1). An estimate of the registration of the sections, not a correction.",
        "n_pairs": nz - 1,
        "n_pairs_with_shift_of_3_voxels_or_more": int((mag >= 3).sum()),
        "largest_shift": {"z": worst, "shift_yx": shifts[worst],
                          "correlation_as_stored": round(corr[worst], 3)},
        "median_correlation_of_neighbouring_sections": round(float(np.median(corr)), 3),
        "shifts_yx": shifts,
    }


def nearest_cleft(points_nm, cleft_co_nm, cleft_co_ids):
    """For each point: (id, distance in nm) of the nearest cleft voxel centre."""
    out = []
    for p in points_nm:
        d2 = ((cleft_co_nm - p[None, :]) ** 2).sum(1)
        k = int(np.argmin(d2))
        out.append((int(cleft_co_ids[k]), float(np.sqrt(d2[k]))))
    return out


# --------------------------------------------------------------------------- output

def save_npy(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def seg_colours(ids2d):
    """A fixed pseudo-random colour for every id (0 stays black)."""
    h = ids2d.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    rgb = np.stack([(h >> np.uint64(s)) & np.uint64(255) for s in (40, 48, 56)], -1).astype(np.float32)
    rgb = 60.0 + rgb * (195.0 / 255.0)
    rgb[ids2d == 0] = 0.0
    return rgb


def one_sided_boundary(ids2d):
    """One-voxel lines: a voxel whose right or lower neighbour has another id."""
    b = np.zeros(ids2d.shape, bool)
    b[:, :-1] |= ids2d[:, 1:] != ids2d[:, :-1]
    b[:-1, :] |= ids2d[1:, :] != ids2d[:-1, :]
    return b


def panel_seg(raw2d, seg2d):
    g = np.repeat(raw2d[:, :, None].astype(np.float32), 3, 2)
    out = 0.55 * g + 0.45 * seg_colours(seg2d)
    out[seg2d == 0] = g[seg2d == 0]
    out[one_sided_boundary(seg2d)] = (255, 255, 0)
    return out.clip(0, 255).astype(np.uint8)


def panel_class(raw2d, cleft2d):
    g = np.repeat(raw2d[:, :, None].astype(np.float32), 3, 2)
    m = cleft2d != 0
    g[m] = 0.4 * g[m] + 0.6 * np.array([255, 0, 0], np.float32)
    return g.clip(0, 255).astype(np.uint8)


def compose(panels, titles, scale, path, draw=None):
    """Panels side by side, one image pixel per voxel times `scale`, with a title strip."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    head, gap = 26, 8
    ims = []
    for p in panels:
        if p.ndim == 2:
            p = np.repeat(p[:, :, None], 3, 2)
        ims.append(np.repeat(np.repeat(p, scale, 0), scale, 1))
    h, w = ims[0].shape[:2]
    canvas = Image.new("RGB", (len(ims) * w + (len(ims) - 1) * gap, h + head), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (im, t) in enumerate(zip(ims, titles)):
        x0 = k * (w + gap)
        canvas.paste(Image.fromarray(im), (x0, head))
        d.text((x0 + 4, 4), t, fill=(0, 0, 0), font=font)
        if draw is not None:
            draw(d, x0, head, scale, k)
    tmp = path.with_name(path.name + ".tmp.png")
    canvas.save(tmp)
    os.replace(tmp, path)


def write_crop(src, name, out_dir, origin, shape, how_chosen, source_note, cleft_ids_full, table, axis_check):
    out_dir.mkdir(parents=True, exist_ok=True)
    res = src["resolution"]
    sl = tuple(slice(o, o + n) for o, n in zip(origin, shape))
    raw = np.ascontiguousarray(src["raw"][sl])
    seg = np.ascontiguousarray(src["neuron_ids"][sl])
    cleft_ids = np.ascontiguousarray(cleft_ids_full[sl])
    classes = (cleft_ids != 0).astype(np.uint8)
    assert raw.shape == seg.shape == cleft_ids.shape == classes.shape == tuple(shape)
    log(name, "origin", origin, "shape", shape)

    # ---- synapses: pairs with both sites inside the crop; pairs cut by the crop listed apart
    o = np.array(origin)
    n = np.array(shape)
    o_nm = o * np.array(res)
    cl_ids, cl_n = np.unique(cleft_ids[cleft_ids != 0], return_counts=True)
    cleft_ids_in_crop = set(cl_ids.tolist())
    pairs, cut = [], []
    for row in table:
        ins = [bool(np.all(row[k + "_vox_full"] >= o) and np.all(row[k + "_vox_full"] < o + n)) for k in ("pre", "post")]
        if not any(ins):
            continue
        e = {"pre_id": row["pre_id"], "post_id": row["post_id"]}
        for k in ("pre", "post"):
            e[k + "_nm"] = [round(float(v), 3) for v in (row[k + "_nm_full"] - o_nm)]
            e[k + "_vox"] = [int(v) for v in (row[k + "_vox_full"] - o)]
            e[k + "_neuron"] = row[k + "_neuron"]
            e[k + "_comment"] = row[k + "_comment"]
        e["nearest_cleft_id"] = row["nearest_cleft_id"]
        e["nearest_cleft_dist_nm"] = round(row["nearest_cleft_dist_nm"], 1)
        e["nearest_cleft_in_crop"] = row["nearest_cleft_id"] in cleft_ids_in_crop
        if all(ins):
            for k in ("pre", "post"):               # the id in the crop must be the id recorded
                assert int(seg[tuple(e[k + "_vox"])]) == e[k + "_neuron"]
            pairs.append(e)
        else:
            e["pre_inside"], e["post_inside"] = ins
            cut.append(e)
    synapses = {
        "what": "Annotated synaptic partner pairs of CREMI sample A (annotations/presynaptic_site/partners: "
                "rows of (pre id, post id)). 'pairs' has both sites inside the crop; 'pairs_cut' has one "
                "site inside and one outside (its coordinates then lie outside the crop, and its neuron id "
                "was read from the whole sample).",
        "axis_order": "z, y, x",
        "units": "nm, relative to the crop origin = the centre of voxel [0, 0, 0] of the crop; "
                 "voxel index = round(nm / voxel_nm), the convention of cremi_python (cremi/Volume.py)",
        "voxel_nm": list(res),
        "offset_vox": list(origin),
        "fields": {
            "pre_id, post_id": "annotation ids of the source",
            "pre_nm, post_nm": "location of the site, nm, z y x, relative to the crop origin",
            "pre_vox, post_vox": "voxel index of the site in the crop, z y x",
            "pre_neuron, post_neuron": "value of seg at that voxel",
            "pre_comment, post_comment": "comment of the source on that annotation (only 'unsure' occurs) or null",
            "nearest_cleft_id, nearest_cleft_dist_nm": "DERIVED here, not in the source: id in cleft_ids of the "
                "cleft voxel nearest to the midpoint between the two sites (searched in the whole sample), and "
                "its distance; the source does not link partner pairs to clefts",
            "nearest_cleft_in_crop": "whether that cleft id has any voxel in this crop",
        },
        "n_pairs": len(pairs),
        "n_pairs_cut": len(cut),
        "pairs": pairs,
        "pairs_cut": cut,
    }

    # ---- measurements
    st = seg_stats(seg, raw)
    full_ids, full_n = np.unique(cleft_ids_full[cleft_ids_full != 0], return_counts=True)
    full_count = dict(zip(full_ids.tolist(), full_n.tolist()))
    whole = int(sum(1 for i, c in zip(cl_ids.tolist(), cl_n.tolist()) if full_count[i] == c))
    same = [(e["pre_id"], e["post_id"], e["pre_neuron"]) for e in pairs if e["pre_neuron"] == e["post_neuron"]]
    unsure = int(sum((e["pre_comment"] is not None) + (e["post_comment"] is not None) for e in pairs))
    stats = {
        "seg": st,
        "clefts": {
            "n_cleft_ids": int(len(cl_ids)),
            "n_cleft_ids_with_all_their_voxels_in_the_crop": whole,
            "n_cleft_voxels": int(cl_n.sum()),
            "cleft_voxel_fraction": round(float(cl_n.sum() / classes.size), 6),
        },
        "synapses": {
            "n_pairs": len(pairs),
            "n_pairs_cut_by_the_crop": len(cut),
            "n_distinct_pre_neurons": len({e["pre_neuron"] for e in pairs}),
            "n_distinct_post_neurons": len({e["post_neuron"] for e in pairs}),
            "n_pairs_with_one_neuron_id_at_both_sites": len(same),
            "n_sites_commented_unsure": unsure,
            "median_nearest_cleft_dist_nm": round(float(np.median([e["nearest_cleft_dist_nm"] for e in pairs])), 1)
            if pairs else None,
            "max_nearest_cleft_dist_nm": round(float(max(e["nearest_cleft_dist_nm"] for e in pairs)), 1)
            if pairs else None,
        },
        "raw": {"min": int(raw.min()), "max": int(raw.max()), "mean": round(float(raw.mean()), 2),
                "std": round(float(raw.std()), 2)},
    }
    align = alignment_check(raw, seg)
    m = classes.astype(bool)
    shifts = section_shifts(raw)
    checks = {
        "annotation_axis_order": axis_check,
        "boundaries_on_membranes": align,
        "section_to_section_shift": shifts,
        "mean_raw_in_cleft_voxels": round(float(raw[m].mean()), 2) if m.any() else None,
        "mean_raw_outside_cleft_voxels": round(float(raw[~m].mean()), 2),
    }
    log(name, "alignment:", json.dumps(align))

    # ---- arrays
    save_npy(out_dir / "raw.npy", raw)
    save_npy(out_dir / "seg.npy", seg)
    save_npy(out_dir / "classes.npy", classes)
    save_npy(out_dir / "cleft_ids.npy", cleft_ids)
    save_json(out_dir / "synapses.json", synapses)

    # ---- previews
    scale = 2 if max(shape[1:]) <= 640 else 1
    zm = shape[0] // 2
    compose(
        [raw[zm], panel_seg(raw[zm], seg[zm]), panel_class(raw[zm], cleft_ids[zm])],
        ["%s  raw, section z=%d of %d (source section %d)" % (name, zm, shape[0], zm + origin[0]),
         "seg: one colour per neuron id, boundaries yellow",
         "classes: 1 = synaptic cleft (red), %d voxels here" % int(classes[zm].sum())],
        scale, out_dir / "preview.png")
    zc = int(np.argmax(classes.reshape(shape[0], -1).sum(1)))

    def draw_sites(d, x0, y0, s, k):
        for e in pairs + cut:
            if zc not in (e["pre_vox"][0], e["post_vox"][0]):
                continue
            a = (x0 + e["pre_vox"][2] * s + s // 2, y0 + e["pre_vox"][1] * s + s // 2)
            b = (x0 + e["post_vox"][2] * s + s // 2, y0 + e["post_vox"][1] * s + s // 2)
            d.line([a, b], fill=(255, 255, 255), width=1)
            r = 4
            d.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], outline=(0, 255, 255), width=2)
            d.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], outline=(255, 0, 255), width=2)

    compose(
        [panel_class(raw[zc], cleft_ids[zc]), panel_seg(raw[zc], seg[zc])],
        ["%s  section z=%d (most cleft voxels): clefts red; pre site cyan, post site magenta" % (name, zc),
         "the same section with seg; a pair is drawn if one of its sites is in this section"],
        scale, out_dir / "preview_synapses.png", draw=draw_sites)

    # ---- meta
    notes = [
        "Axis order of every array and of every coordinate: z, y, x (the source stores volumes as "
        "(depth, height, width) and locations as (z, y, x) in nm). z is the section number.",
        "voxel_nm is read from the 'resolution' attribute of the volumes in the source file: "
        "%s nm (z, y, x). Sections are %g nm apart; a pixel is %g x %g nm." % (list(res), res[0], res[1], res[2]),
        "Coordinates in synapses.json: nm relative to the crop origin, which is the centre of voxel "
        "[0, 0, 0] of the crop; voxel index = round(nm / voxel_nm), as in the reference code cremi_python.",
        "seg holds the neuron ids of the source unchanged (the same ids in cremi3d and cremi3d_full). "
        "Every voxel has an id: seg has no 0, and membranes and extracellular space are not a label of "
        "their own. A boundary between two neurites is the place where the id changes.",
        "seg has many tiny ids: %d of the %d ids are present in one section only (median size %s voxels, "
        "largest %s); %d ids have 1000 voxels or more and hold %.2f %% of the voxels. The ids below 1000 "
        "voxels are small specks inside or between the neurites (mean raw value there %s, in the whole crop "
        "%.2f). Filter by size before treating an id as a neurite." % (
            st["n_ids_present_in_one_section_only"], st["n_ids"],
            st["median_voxels_of_ids_present_in_one_section_only"],
            st["max_voxels_of_ids_present_in_one_section_only"],
            st["n_ids_with_1000_voxels_or_more"], 100 * st["voxel_share_of_ids_with_1000_voxels_or_more"],
            st["mean_raw_in_ids_below_1000_voxels"], float(raw.mean())),
        "An id is not always one connected component inside the crop: seg has %d ids and %d connected "
        "components (26-connectivity). Of the %d ids with 1000 voxels or more, %d are in more than one "
        "component; for %d of them the largest component holds less than 99 %% of the voxels (smallest "
        "share: %s). A Frame is defined per connected component, so split ids into components first." % (
            st["n_ids"], st["n_connected_components_26"], st["n_ids_with_1000_voxels_or_more"],
            st["n_ids_with_1000_voxels_or_more_in_more_than_one_component_26"],
            st["n_ids_with_1000_voxels_or_more_whose_largest_component_26_holds_under_99_percent"],
            st["smallest_share_of_the_largest_component_among_ids_with_1000_voxels_or_more"]),
        "Neighbouring sections are not perfectly registered. Estimated by cross-correlation of whole "
        "sections, %d of the %d pairs of neighbouring sections are shifted by 3 voxels or more in y or x; "
        "the largest shift is (dy, dx) = %s voxels between z = %d and z = %d (see checks."
        "section_to_section_shift in this file). The labels follow the image in every section, so the "
        "neurites jump with it. These are the original volumes, not the re-aligned ones that the CREMI "
        "home page mentions (release of 18 April 2017)." % (
            shifts["n_pairs_with_shift_of_3_voxels_or_more"], shifts["n_pairs"],
            shifts["largest_shift"]["shift_yx"], shifts["largest_shift"]["z"], shifts["largest_shift"]["z"] + 1),
        "Who made the labels, as https://cremi.org/data/ states it: " +
        " ".join("%s: '%s'" % kv for kv in LABEL_CREDITS.items()),
        "classes.npy is 1 where volumes/labels/clefts has a label. cleft_ids.npy keeps the cleft ids "
        "(uint64); the background value of the source, 0xffffffffffffffff, is written as 0. The value "
        "0xfffffffffffffffe (invalid, per cremi_python) does not occur in this file.",
        "Clefts in this crop: %d cleft ids, %d voxels, %.4f %% of the crop. %d of the ids have all "
        "their voxels inside the crop." % (
            len(cl_ids), int(cl_n.sum()), 100 * cl_n.sum() / classes.size, whole),
        "synapses.json: %d partner pairs with both sites in the crop, %d more cut by the crop border. "
        "The source does not link a pair to a cleft id; nearest_cleft_id is derived here." % (len(pairs), len(cut)),
        how_chosen,
        source_note,
        "The data page names the dataset 'annotations/type'; in the file it is 'annotations/types'.",
    ]
    if same:
        notes.append("Pairs whose two sites carry one and the same neuron id (pre id, post id, neuron id): %s. "
                     "Kept as the source gives them." % same)
    if unsure:
        notes.append("%d sites of the pairs in this crop carry the comment 'unsure' in the source." % unsure)
    meta = {
        "name": name,
        "dataset": "CREMI sample A (training Dataset A, cropped version 20160501): serial-section TEM, "
                   "adult Drosophila melanogaster brain",
        "voxel_nm": list(res),
        "shape": list(shape),
        "offset_vox": list(origin),
        "level": "level 0, the native and only resolution of the source file: %g x %g x %g nm (z, y, x)" % res,
        "sources": [URL, DOC_URL, CODE_URL],
        "seg_kind": "proofread",
        "seg_meaning": "Neuron segmentation of CREMI (volumes/labels/neuron_ids): one id per neuron "
                       "within the sample; every voxel carries an id, there is no background label.",
        "class_names": {"1": "synaptic cleft"},
        "licence": LICENCE,
        "citation": CITATION,
        "fetched": datetime.date.today().isoformat(),
        "notes": notes,
        "axis_order": "z, y, x",
        "files": {
            "raw.npy": "uint8 [z, y, x], volumes/raw",
            "seg.npy": "uint64 [z, y, x], volumes/labels/neuron_ids",
            "classes.npy": "uint8 [z, y, x], 1 = synaptic cleft, 0 = none",
            "cleft_ids.npy": "uint64 [z, y, x], volumes/labels/clefts with the background written as 0",
            "synapses.json": "pre/post partner pairs, nm relative to the crop origin, axis order z, y, x",
            "preview.png": "middle section: raw | seg overlay with boundaries | cleft overlay",
            "preview_synapses.png": "section with the most cleft voxels, partner sites drawn",
        },
        "stats": stats,
        "checks": checks,
    }
    save_json(out_dir / "meta.json", meta)

    # ---- read back what was written and compare with the source
    for fn, ref in (("raw.npy", src["raw"][sl]), ("seg.npy", src["neuron_ids"][sl]),
                    ("cleft_ids.npy", cleft_ids_full[sl])):
        back = np.load(out_dir / fn, mmap_mode="r")
        assert back.shape == tuple(shape) and back.dtype == ref.dtype and back.flags["C_CONTIGUOUS"], fn
        assert np.array_equal(back, ref), fn
    back = np.load(out_dir / "classes.npy", mmap_mode="r")
    assert back.dtype == np.uint8 and np.array_equal(back != 0, src["clefts"][sl] < CLEFT_INVALID)
    log(name, "written and read back:", out_dir)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder of the crop cremi3d; cremi3d_full goes beside it")
    ap.add_argument("--src", default=None, help="cache folder of the source file "
                                                "(default: <out>/../../src/<name of out>)")
    ap.add_argument("--shape", type=int, nargs=3, default=[64, 512, 512], metavar=("Z", "Y", "X"))
    ap.add_argument("--origin", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                    help="place the small crop here instead of applying the rule")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out)).resolve()
    name = out.name
    data_dir = out.parent.parent
    src_dir = Path(os.path.expanduser(args.src)).resolve() if args.src else data_dir / "src" / name
    path, how = find_source(src_dir, data_dir / FILE)
    log("source", path, "-", how)
    source_note = ("Source file %s, %d bytes, sha256 %s; in this run: %s. Its size equals the Content-Length "
                   "that %s returned on 2026-11-23, and five ranges of 1 MiB of that URL were equal to the "
                   "file on that day; the source publishes no checksum."
                   % (FILE, SIZE, SHA256, how, URL))

    src = load_source(path)
    res = src["resolution"]
    shape_full = src["raw"].shape
    log("volumes", shape_full, "resolution", res, "file_format", src["file_format"])

    # labels
    if (src["neuron_ids"] == 0).any() or (src["neuron_ids"] >= CLEFT_INVALID).any():
        raise SystemExit("neuron_ids holds 0 or a reserved value; decide how to map it before going on")
    cleft_ids_full = src["clefts"].copy()
    if (cleft_ids_full == 0).any():
        raise SystemExit("a cleft has the id 0, which is needed for 'no cleft'")
    n_invalid = int((cleft_ids_full == CLEFT_INVALID).sum())
    cleft_ids_full[cleft_ids_full >= CLEFT_INVALID] = 0
    log("clefts: %d labelled voxels, %d voxels with the 'invalid' value" % (int((cleft_ids_full != 0).sum()), n_invalid))

    # annotations
    index = {int(i): k for k, i in enumerate(src["ann_ids"])}
    loc = src["ann_locations"]
    vox = to_voxel(loc, res)
    if (vox < 0).any() or (vox >= np.array(shape_full)).any():
        raise SystemExit("an annotation lies outside the volume")
    for a, b in src["partners"]:
        if src["ann_types"][index[int(a)]] != "presynaptic_site" or src["ann_types"][index[int(b)]] != "postsynaptic_site":
            raise SystemExit("partner row (%d, %d) is not (presynaptic_site, postsynaptic_site)" % (a, b))
    pre_k = np.array([index[int(a)] for a, _ in src["partners"]])
    post_k = np.array([index[int(b)] for _, b in src["partners"]])
    log("annotations: %d sites, %d pairs, all rows are (pre, post)" % (len(loc), len(pre_k)))

    # nearest cleft to the midpoint of each pair; the same with y and x swapped as a check of the axis order
    co = np.stack(np.nonzero(cleft_ids_full), 1)
    co_ids = cleft_ids_full[co[:, 0], co[:, 1], co[:, 2]]
    co_nm = co * np.array(res)
    mid = 0.5 * (loc[pre_k] + loc[post_k])
    near = nearest_cleft(mid, co_nm, co_ids)
    near_swapped = nearest_cleft(mid[:, [0, 2, 1]], co_nm, co_ids)
    d_ok = np.array([d for _, d in near])
    d_sw = np.array([d for _, d in near_swapped])
    axis_check = {
        "what": "distance from the midpoint of each partner pair to the nearest cleft voxel, nm, whole sample",
        "locations_read_as_z_y_x": {"median": round(float(np.median(d_ok)), 1), "max": round(float(d_ok.max()), 1)},
        "locations_read_as_z_x_y": {"median": round(float(np.median(d_sw)), 1), "max": round(float(d_sw.max()), 1)},
    }
    log("axis order check:", json.dumps(axis_check))

    table = []
    for n_, (kp, kq) in enumerate(zip(pre_k, post_k)):
        row = {"pre_id": int(src["ann_ids"][kp]), "post_id": int(src["ann_ids"][kq]),
               "nearest_cleft_id": near[n_][0], "nearest_cleft_dist_nm": near[n_][1]}
        for k, kk in (("pre", kp), ("post", kq)):
            row[k + "_nm_full"] = loc[kk]
            row[k + "_vox_full"] = vox[kk]
            row[k + "_neuron"] = int(src["neuron_ids"][tuple(vox[kk])])
            row[k + "_comment"] = src["comments"].get(int(src["ann_ids"][kk]))
        table.append(row)

    # the small crop
    crop_shape = tuple(args.shape)
    if args.origin is not None:
        origin = tuple(args.origin)
        how_chosen = "Position of the crop: given by hand with --origin %s." % (list(origin),)
    else:
        origin, info = choose_origin(cleft_ids_full, vox[pre_k], vox[post_k], crop_shape)
        faces = [t for t, hit in (
            ("z = 0", origin[0] == 0), ("z = %d" % (shape_full[0] - 1), origin[0] + crop_shape[0] == shape_full[0]),
            ("y = 0", origin[1] == 0), ("y = %d" % (shape_full[1] - 1), origin[1] + crop_shape[1] == shape_full[1]),
            ("x = 0", origin[2] == 0), ("x = %d" % (shape_full[2] - 1), origin[2] + crop_shape[2] == shape_full[2]),
        ) if hit]
        how_chosen = (
            "Position of the crop: chosen by a fixed rule over %d candidate windows of %s voxels (every z "
            "origin; y and x origins every 16 voxels and at the far border). The window with the most whole "
            "synaptic clefts (bounding box of the cleft id inside the window) wins; ties go to the most "
            "partner pairs with both sites inside, then to the smallest origin. Result: origin %s with %d "
            "whole clefts and %d complete pairs (mean over all candidates: %.2f clefts, %.2f pairs; %d "
            "candidate(s) share the top score). " % (
                info["n_candidates"], list(crop_shape), list(origin), info["whole_clefts"],
                info["complete_pairs"], info["mean_whole_clefts_over_candidates"],
                info["mean_complete_pairs_over_candidates"], info["n_candidates_with_the_same_score"])
            + ("The crop touches the border of the sample at %s." % ", ".join(faces) if faces
               else "The crop does not touch the border of the sample."))
    log(how_chosen)

    full_name = name + "_full"
    metas = [
        write_crop(src, name, out, origin, crop_shape, how_chosen, source_note, cleft_ids_full, table, axis_check),
        write_crop(src, full_name, out.parent / full_name, (0, 0, 0), tuple(shape_full),
                   "Position of the crop: the whole sample; the crop %s is its part at offset %s, shape %s."
                   % (name, list(origin), list(crop_shape)),
                   source_note, cleft_ids_full, table, axis_check),
    ]
    for m in metas:
        s = m["stats"]
        log("%s: shape %s, %d ids (%d with >= 1000 voxels), %d cleft ids, %d cleft voxels, %d pairs" % (
            m["name"], m["shape"], s["seg"]["n_ids"], s["seg"]["n_ids_with_1000_voxels_or_more"],
            s["clefts"]["n_cleft_ids"], s["clefts"]["n_cleft_voxels"], s["synapses"]["n_pairs"]))
    log("FETCH-DONE")


if __name__ == "__main__":
    sys.exit(main())

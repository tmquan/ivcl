#!/usr/bin/env python3
"""Fetch two nested crops of MICrONS "minnie65" and write them in the common crop format.

SOURCE
    MICrONS cubic millimetre of mouse visual cortex, portion "minnie65", public release
    (https://www.microns-explorer.org/cortical-mm3). Everything is read without credentials:
      EM image            precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em
      flat segmentation   .../iarpa_microns/minnie/minnie65/seg_m1300   (materialization v1300, 13 Jan 2025;
                          the newest of the four flat snapshots seg, seg_m343, seg_m943, seg_m1300)
      PSD ("cleft") seg.  .../iarpa_microns/minnie/minnie65/clefts
      nucleus seg.        .../iarpa_microns/minnie/minnie65/nuclei
      tables (static archive of the annotation database, CSV):
                          https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1300/
                              aibs_cell_info.csv.gz                      cell types per nucleus / root id
                              synapses_pni_2_v1_filtered_view.csv.gz     synapse table, 20 GB, read as a stream
                          https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1412/
                              proofreading_status_and_strategy_merged.csv.gz
    The image and the label volumes are Neuroglancer "precomputed" volumes (sharded). Their info
    files give, per scale, the voxel size in nm in the order x, y, z. This script reads the info
    files itself, checks the voxel sizes it expects, and reads the voxels with cloud-volume.

LICENCE
    Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on
    https://www.microns-explorer.org/terms-and-conditions .
    Citation asked for by https://www.microns-explorer.org/citation-policy :
    The MICrONS Consortium et al. 2025. "Functional connectomics spanning multiple areas of mouse
    visual cortex." Nature 640 (8058): 435-447. https://doi.org/10.1038/s41586-025-08790-w

WHAT IT WRITES
    Two crop folders, the first inside the second:
      <out>                 "microns"         64 x 512 x 512 voxels (z, y, x) at 8 x 8 x 40 nm (x, y, z),
                                              the finest public level of image and segmentation; neuropil.
      <out>/../microns_coarse  "microns_coarse"  256 x 512 x 512 voxels at 32 x 32 x 40 nm, the bucket's own
                                              third level (no resampling here); holds the cell body of one
                                              proofread layer-2/3 pyramidal neuron and a capillary.
    Each folder holds
      raw.npy          uint8  [z, y, x]  EM image
      seg.npy          uint64 [z, y, x]  root ids of the flat segmentation v1300, 0 = no label
      classes.npy      uint8  [z, y, x]  0 none, 1 synaptic cleft (PSD), 2 nucleus, 3 blood vessel lumen
      cleft_ids.npy    uint32 [z, y, x]  PSD segmentation; the id is the synapse id of the synapse table
      nucleus_ids.npy  uint32 [z, y, x]  nucleus segmentation (64 nm source voxels, repeated)
      cells.json       cells of the public cell table whose root id occurs in the crop
      synapses.json    rows of the public synapse table whose centre point lies in the crop
      meta.json, preview.png
    Downloads are cached under <src> (default: <out>/../../src/microns). The script is safe to run
    again: cached cutouts and tables are reused, outputs are rewritten.

    Run:  python fetch_microns.py --out ~/connectome/data/crops/microns
    The synapse table is one 20 GB gzip stream with no spatial index; it is read once from start to
    end and only the rows near the crops are kept (cached). On the lab's Spark that took more than
    an hour at 4 to 5 MB/s. Use --skip-synapses to leave it out (an existing synapses.json is then
    removed). --search writes nothing to the crop folders; it prints how the place was chosen.

HOW THE PLACE WAS CHOSEN (23 Nov 2026; the result is the constants CROPS below; --search repeats it)
    1. From aibs_cell_info (v1300) take neurons with dendrite_cleaned and axon_cleaned true.
    2. Keep those whose nucleus is 4 to 8 um from the nearest voxel of the public vessel mask
       (precomputed://https://bossdb-open-data.s3.amazonaws.com/wei2024/minnie/bv, 2048 nm level).
    3. Look at the image around the first candidates (by eye). The block around nucleus 291111 (a
       layer-2/3 pyramidal cell, dendrite "extended", axon "fully_extended") has no missing section
       in 256 sections; the block around nucleus 294562 has an all-black section (z = 21559) and
       was dropped.
    4. Slide a 512 x 512 window over a 768 x 768 x 256 block at 32 nm: keep windows that hold at
       least 90 % of the voxels that the cell's segment and the vessel segment have in that block.
    5. Inside such a window, score every 128 x 128 x 64 sub-window (at 32 nm; 512 x 512 x 64 at
       8 nm) that is at least 32 voxels from the window's faces in x and y and 16 in z and has less
       than 10 % of its voxels in the largest objects of the block (each above 0.4 % of the block):
       the score is the number of distinct segment ids. From the best-scoring ones the place was
       picked by eye, for a window with the cell body near its middle: 430 ids at 32 nm, window
       with 96 % of the cell's voxels and 93 % of the vessel's.
"""
from __future__ import annotations

import argparse
import datetime
import http.client
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

GCS = "https://storage.googleapis.com/"
LAYERS = {
    "em": GCS + "iarpa_microns/minnie/minnie65/em",
    "seg": GCS + "iarpa_microns/minnie/minnie65/seg_m1300",
    "clefts": GCS + "iarpa_microns/minnie/minnie65/clefts",
    "nuclei": GCS + "iarpa_microns/minnie/minnie65/nuclei",
}
TABLES_V1300 = GCS + "mat_dbs/public/minnie65_phase3_v1/v1300/"
TABLES_V1412 = GCS + "mat_dbs/public/minnie65_phase3_v1/v1412/"
SYNAPSE_TABLE = "synapses_pni_2_v1_filtered_view"
CELL_TABLE = "aibs_cell_info"
PROOF_TABLE = "proofreading_status_and_strategy_merged"
DOC_URLS = [
    "https://www.microns-explorer.org/cortical-mm3",
    "https://www.microns-explorer.org/terms-and-conditions",
    "https://www.microns-explorer.org/citation-policy",
    "https://tutorial.microns-explorer.org/static-repositories.html",
    "https://tutorial.microns-explorer.org/release_manifests/version-1300.html",
    "https://tutorial.microns-explorer.org/annotation-tables.html",
    GCS + "microns-static-links/mm3/blood_vessels.json",
    "https://bossdb-open-data.s3.amazonaws.com/wei2024/minnie/bv/info",
]

# Used by --search only: the public vessel mask, the neuron and the block that were examined.
VESSEL_MASK = "https://bossdb-open-data.s3.amazonaws.com/wei2024/minnie/bv"
SEARCH_NUCLEUS = 291111
SEARCH_BLOCK = {"res_xyz": (32, 32, 40), "offset_xyz": (22710, 13422, 20806), "size_xyz": (768, 768, 256)}

# The segment that the MICrONS Explorer page shows under "blood vessels"
# (state https://storage.googleapis.com/microns-static-links/mm3/blood_vessels.json selects it in seg_m1300).
VESSEL_ROOT_ID = 864691134966128927
# The annotation tables give positions in voxels of 4 x 4 x 40 nm (x, y, z), although the finest
# public image level is 8 x 8 x 40 nm (tutorial.microns-explorer.org, pages named above).
TABLE_VOXEL_NM_XYZ = (4.0, 4.0, 40.0)

# Crops. Offsets and sizes are in voxels of the level named by res_xyz, order x, y, z (the order
# of the source). The fine crop is voxels [48:112, 32:160, 352:480] (z, y, x) of the coarse crop.
CROPS = {
    "microns": {
        "res_xyz": (8, 8, 40),
        "offset_xyz": (92632, 54456, 20854),
        "size_xyz": (512, 512, 64),
        "purpose": "neuropil with many thin neurites at the finest public level",
    },
    "microns_coarse": {
        "res_xyz": (32, 32, 40),
        "offset_xyz": (22806, 13582, 20806),
        "size_xyz": (512, 512, 256),
        "purpose": "near-isotropic block with one cell body (nucleus 291111) and a capillary",
    },
}
FINE, COARSE = "microns", "microns_coarse"
CLASS_NAMES = {
    "1": "synaptic cleft (voxels of the automatic PSD segmentation 'clefts')",
    "2": "nucleus (voxels of the automatic nucleus segmentation 'nuclei')",
    "3": "blood vessel lumen (voxels of segment %d of the flat segmentation)" % VESSEL_ROOT_ID,
}
# The synapse stream keeps every row whose centre lies in the coarse block grown by this margin,
# so that a small change of the crops does not need the 20 GB stream again.
SYN_MARGIN_NM_XYZ = (10000.0, 10000.0, 4000.0)

LICENCE = ("CC BY 4.0 (Creative Commons Attribution 4.0 International), "
           "https://www.microns-explorer.org/terms-and-conditions")
CITATION = ("The MICrONS Consortium et al. 2025. Functional connectomics spanning multiple areas of "
            "mouse visual cortex. Nature 640 (8058): 435-447. https://doi.org/10.1038/s41586-025-08790-w")
UA = "connectome-render-fetch/1.0 (python-urllib)"
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------- small downloads

def http_get(url, dest: Path, tries=6):
    """Download url to dest unless dest exists. Returns dest."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            tmp.replace(dest)
            return dest
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            log("  retry", k + 1, url, repr(e)[:120])
            time.sleep(2 ** k)
    raise RuntimeError("could not download " + url)


def read_info(layer, src: Path):
    """The info file of a precomputed layer, as the source serves it (cached)."""
    url = LAYERS[layer] + "/info"
    p = http_get(url, src / "info" / (layer + ".json"))
    return json.loads(p.read_text()), url


def find_scale(info, res_xyz):
    """Index and entry of the scale whose voxel size is res_xyz (nm; x, y, z)."""
    for i, s in enumerate(info["scales"]):
        if all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(s["resolution"], res_xyz)):
            return i, s
    raise KeyError("no scale with resolution %s; have %s"
                   % (res_xyz, [s["resolution"] for s in info["scales"]]))


# ----------------------------------------------------------------------------- voxel cutouts

def cutout(layer, res_xyz, offset_xyz, size_xyz, src: Path, infos, notes):
    """Voxels [offset, offset+size) of a layer at the scale res_xyz, as an array [z, y, x].

    offset is in voxels of that scale, counted from the origin of the volume's physical frame
    (the 'voxel_offset' of the scale is the position of the first stored voxel in the same
    units, which is how Neuroglancer and cloud-volume count). Cached as .npy."""
    info = infos[layer][0]
    idx, sc = find_scale(info, res_xyz)
    lo = [int(round(float(v))) for v in sc.get("voxel_offset", [0, 0, 0])]
    hi = [a + int(b) for a, b in zip(lo, sc["size"])]
    for a, n, l, h, ax in zip(offset_xyz, size_xyz, lo, hi, "xyz"):
        if a < l or a + n > h:
            raise ValueError("%s: crop %s [%d, %d) outside the stored range [%d, %d)"
                             % (layer, ax, a, a + n, l, h))
    x0, y0, z0 = offset_xyz
    sx, sy, sz = size_xyz
    name = "%s_%s_x%d-%d_y%d-%d_z%d-%d.npy" % (layer, sc["key"], x0, x0 + sx, y0, y0 + sy, z0, z0 + sz)
    p = src / "cutouts" / name
    if p.exists():
        a = np.load(p)
        log("cached", name, a.shape, a.dtype)
        return a
    from cloudvolume import CloudVolume
    from cloudvolume.exceptions import EmptyVolumeException
    path = "precomputed://" + LAYERS[layer]
    # Label layers leave out chunks that are all background; image and segmentation should not.
    fill_first = layer in ("clefts", "nuclei", "bv")
    t = time.time()
    try:
        vol = CloudVolume(path, mip=idx, use_https=True, progress=False, fill_missing=fill_first,
                          parallel=1, cache=False)
        a = vol[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz]
    except EmptyVolumeException:
        notes.append("%s: at least one chunk of the cutout is absent in the bucket; "
                     "its voxels were set to 0." % layer)
        vol = CloudVolume(path, mip=idx, use_https=True, progress=False, fill_missing=True,
                          parallel=1, cache=False)
        a = vol[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz]
    got = [float(v) for v in vol.resolution]
    if any(abs(g - float(r)) > 1e-6 for g, r in zip(got, res_xyz)):
        raise RuntimeError("%s: cloud-volume reports voxel size %s, wanted %s" % (layer, got, res_xyz))
    a = np.asarray(a)
    assert a.shape == (sx, sy, sz, 1), a.shape
    a = np.ascontiguousarray(a[..., 0].transpose(2, 1, 0))      # [x, y, z] -> [z, y, x]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part.npy")
    np.save(tmp, a)
    tmp.replace(p)
    log("fetched", name, a.shape, a.dtype, "in %.1fs" % (time.time() - t))
    return a


def repeat_to(a, shape):
    """Repeat the voxels of a coarser label array so that it has the given shape."""
    f = [s // n for s, n in zip(shape, a.shape)]
    assert all(n * k == s for n, k, s in zip(a.shape, f, shape)), (a.shape, shape)
    for ax, k in enumerate(f):
        if k > 1:
            a = np.repeat(a, k, axis=ax)
    return a


def fetch_crop(name, src, infos, notes):
    """raw, seg, cleft ids, nucleus ids of one crop, each [z, y, x]."""
    c = CROPS[name]
    res, off, size = c["res_xyz"], c["offset_xyz"], c["size_xyz"]
    raw = cutout("em", res, off, size, src, infos, notes)
    seg = cutout("seg", res, off, size, src, infos, notes)
    cle = cutout("clefts", res, off, size, src, infos, notes)
    # The nucleus layer starts at 64 x 64 x 40 nm. The crop corners are multiples of 64 nm in x
    # and y, so the crop maps to whole voxels of that level.
    nres = (64, 64, 40)
    k = [n // r for n, r in zip(nres, res)]
    assert all(o % kk == 0 and s % kk == 0 for o, s, kk in zip(off, size, k)), "crop not on the 64 nm grid"
    nuc = cutout("nuclei", nres, [o // kk for o, kk in zip(off, k)], [s // kk for s, kk in zip(size, k)],
                 src, infos, notes)
    nuc = repeat_to(nuc, raw.shape)
    assert raw.dtype == np.uint8 and raw.shape == (size[2], size[1], size[0])
    assert seg.shape == raw.shape and cle.shape == raw.shape and nuc.shape == raw.shape
    return raw, seg.astype(np.uint64), cle.astype(np.uint32), nuc.astype(np.uint32)


def make_classes(seg, cle, nuc):
    cls = np.zeros(seg.shape, np.uint8)
    cls[seg == np.uint64(VESSEL_ROOT_ID)] = 3
    cls[nuc > 0] = 2
    cls[cle > 0] = 1
    return cls


# ----------------------------------------------------------------------------- tables

def read_table(base, table, src: Path):
    """A table of the static archive as a pandas DataFrame (header file + gzip CSV)."""
    import pandas as pd
    hp = http_get(base + table + "_header.csv", src / "tables" / (base.rstrip("/").split("/")[-1] + "_" + table + "_header.csv"))
    dp = http_get(base + table + ".csv.gz", src / "tables" / (base.rstrip("/").split("/")[-1] + "_" + table + ".csv.gz"))
    cols = [ln.split(",")[0] for ln in hp.read_text().splitlines() if ln.strip()]
    df = pd.read_csv(dp, header=None, names=cols, compression="gzip", low_memory=False)
    return df, [base + table + "_header.csv", base + table + ".csv.gz"]


def wkb_pointz(hexstr):
    """x, y, z of a PostGIS 'POINT Z' written as hex EWKB (little endian)."""
    b = bytes.fromhex(hexstr)
    assert b[0] == 1 and struct.unpack("<I", b[1:5])[0] & 0xFF == 1, hexstr
    return struct.unpack("<ddd", b[5:29])


def table_pos_to_crop(pos_xyz_tablevox, name):
    """Table position (voxels of 4 x 4 x 40 nm; x, y, z) -> (nm relative to the crop origin, voxel
    index in the crop), both ordered z, y, x. The nm value is the centre of the table voxel."""
    c = CROPS[name]
    p = np.asarray(pos_xyz_tablevox, np.float64)
    nm_xyz = (np.floor(p) + 0.5) * np.array(TABLE_VOXEL_NM_XYZ) - np.array(c["offset_xyz"], np.float64) * np.array(c["res_xyz"], np.float64)
    vox_xyz = np.floor(nm_xyz / np.array(c["res_xyz"], np.float64)).astype(np.int64)
    return nm_xyz[..., ::-1], vox_xyz[..., ::-1]


def inside(vox_zyx, shape):
    v = np.asarray(vox_zyx)
    return np.all((v >= 0) & (v < np.array(shape)), axis=-1)


def jbool(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).lower() in ("t", "true", "1")


def jstr(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def build_cells(name, seg, src, sources):
    """Rows of the cell table (v1300) whose root id occurs in the crop or whose nucleus point is in it,
    joined with the documented proofreading table (v1412 archive) by identical root id."""
    cells, urls = read_table(TABLES_V1300, CELL_TABLE, src)
    sources += urls
    proof, urls = read_table(TABLES_V1412, PROOF_TABLE, src)
    sources += urls
    ids, counts = np.unique(seg, return_counts=True)
    nvox = dict(zip(ids.tolist(), counts.tolist()))
    xyz = np.array([wkb_pointz(h) for h in cells["pt_position"]])
    nm, vox = table_pos_to_crop(xyz, name)
    ins = inside(vox, seg.shape)
    keep = np.where(ins | cells["pt_root_id"].isin(ids[ids != 0].astype(np.int64)).to_numpy())[0]
    proof_by_root = {int(r.pt_root_id): r for r in proof.itertuples()}
    rows = []
    for i in keep:
        r = cells.iloc[i]
        rid = int(r["pt_root_id"])
        e = {
            "nucleus_id": int(r["id"]),
            "root_id": rid,
            "voxels_in_crop": int(nvox.get(rid, 0)),
            "nucleus_nm": [round(float(v), 1) for v in nm[i]],
            "nucleus_in_crop": bool(ins[i]),
            "seg_at_nucleus_point": int(seg[tuple(vox[i])]) if ins[i] else None,
            "nucleus_volume_um3": round(float(r["nuc_volume"]), 3),
            "broad_type": jstr(r["broad_type"]), "broad_type_source": jstr(r["broad_type_source"]),
            "cell_type": jstr(r["cell_type"]), "cell_type_source": jstr(r["cell_type_source"]),
            "mtype": jstr(r["mtype"]), "mtype_source": jstr(r["mtype_source"]),
            "visual_area": jstr(r["visual_area"]),
            "dendrite_cleaned": jbool(r["dendrite_cleaned"]), "axon_cleaned": jbool(r["axon_cleaned"]),
            "dendrite_strategy": jstr(r["dendrite_strategy"]), "axon_strategy": jstr(r["axon_strategy"]),
            "proofreading_status_and_strategy_v1412": None,
        }
        p = proof_by_root.get(rid)
        if p is not None:
            e["proofreading_status_and_strategy_v1412"] = {
                "valid_id_equals_root_id": bool(int(p.valid_id) == rid),
                "status_dendrite": jbool(p.status_dendrite), "status_axon": jbool(p.status_axon),
                "strategy_dendrite": jstr(p.strategy_dendrite), "strategy_axon": jstr(p.strategy_axon),
            }
        rows.append(e)
    rows.sort(key=lambda e: -e["voxels_in_crop"])
    c = CROPS[name]
    return {
        "axis_order": "z, y, x",
        "units": "nm relative to the crop origin (the corner of voxel [0, 0, 0]); voxel index = floor(nm / voxel_nm)",
        "voxel_nm": [c["res_xyz"][2], c["res_xyz"][1], c["res_xyz"][0]],
        "source": [TABLES_V1300 + CELL_TABLE + ".csv.gz", TABLES_V1412 + PROOF_TABLE + ".csv.gz"],
        "fields": {
            "nucleus_id": "id of the row = id of the nucleus detection (column 'id')",
            "root_id": "column pt_root_id of aibs_cell_info at v1300; the same number as in seg.npy. "
                       "Root ids are larger than 2**53: read them as integers, never as floats.",
            "voxels_in_crop": "number of voxels of seg.npy with that id",
            "nucleus_nm": "column pt_position (given in 4 x 4 x 40 nm voxels), as nm relative to the crop "
                          "origin, z y x; it can lie outside the crop",
            "seg_at_nucleus_point": "value of seg.npy at that point if it is inside the crop",
            "broad_type, cell_type, mtype, visual_area, *_source": "columns of aibs_cell_info, copied. The "
                          "table is in the public archive but is not described on the documentation pages "
                          "read for this crop; the *_source columns name the annotation tables the values "
                          "come from. Cell types of most cells are predictions of classifiers.",
            "dendrite_cleaned, axon_cleaned, dendrite_strategy, axon_strategy": "columns of aibs_cell_info, copied",
            "proofreading_status_and_strategy_v1412": "the row with the same root id in the documented table "
                          "proofreading_status_and_strategy (archive of v1412), or null. A root id changes with "
                          "every edit of an object, so an identical root id means the object of v1300 is the "
                          "object that table describes. status true = 'false merges have been comprehensively "
                          "removed' for that compartment (release manifest v1300).",
        },
        "cells": rows,
    }


# ----------------------------------------------------------------------------- synapse stream

class HTTPStream(io.RawIOBase):
    """Sequential reader of one HTTPS object. After a dropped connection it asks for the rest
    with a Range header, so the bytes handed on are always the object from start to end."""

    def __init__(self, url, timeout=180):
        super().__init__()
        self.url, self.timeout, self.pos, self.size, self.resp, self.reopened = url, timeout, 0, None, None, 0

    def readable(self):
        return True

    def _open(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-" % self.pos, "User-Agent": UA,
                                                        "Accept-Encoding": "identity"})
        self.resp = urllib.request.urlopen(req, timeout=self.timeout)
        cr = self.resp.headers.get("Content-Range")
        if self.resp.status != 206 or not cr:
            raise IOError("server did not honour the Range request (status %s)" % self.resp.status)
        total = int(cr.split("/")[1])
        if self.size is None:
            self.size = total
        elif total != self.size:
            raise IOError("object changed size during the download")

    def readinto(self, b):
        if self.size is not None and self.pos >= self.size:
            return 0
        for attempt in range(40):
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.readinto(b)
                if n == 0 and self.pos < self.size:
                    raise IOError("connection closed at byte %d of %d" % (self.pos, self.size))
                self.pos += n
                return n
            except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
                self.resp = None
                self.reopened += 1
                log("  synapse stream: reconnect %d at byte %d (%s)" % (self.reopened, self.pos, repr(e)[:100]))
                time.sleep(min(60, 2 ** min(attempt, 5)))
        raise IOError("synapse stream: gave up at byte %d" % self.pos)


def synapse_bbox_tablevox():
    """Box (lo, hi; x, y, z; voxels of 4 x 4 x 40 nm) that the stream keeps: coarse block + margin."""
    c = CROPS[COARSE]
    res, off, size = (np.array(c[k], np.float64) for k in ("res_xyz", "offset_xyz", "size_xyz"))
    m, t = np.array(SYN_MARGIN_NM_XYZ), np.array(TABLE_VOXEL_NM_XYZ)
    return np.floor((off * res - m) / t), np.ceil(((off + size) * res + m) / t)


def stream_synapses(src: Path, sources):
    """Rows of the synapse table (v1300) whose ctr_pt lies in the kept box. Cached as CSV."""
    import pandas as pd
    lo, hi = synapse_bbox_tablevox()
    tag = "x%d-%d_y%d-%d_z%d-%d" % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    out = src / "tables" / ("v1300_%s_ctr_in_%s.csv" % (SYNAPSE_TABLE, tag))
    stat = out.with_suffix(".stats.json")
    url = TABLES_V1300 + SYNAPSE_TABLE + ".csv.gz"
    hurl = TABLES_V1300 + SYNAPSE_TABLE + "_header.csv"
    sources += [hurl, url]
    hp = http_get(hurl, src / "tables" / ("v1300_" + SYNAPSE_TABLE + "_header.csv"))
    header = [ln.split(",") for ln in hp.read_text().splitlines() if ln.strip()]
    cols = [h[0] for h in header]
    need = ["id", "pre_pt_position_x", "ctr_pt_position_x", "post_pt_position_x", "size",
            "pre_pt_root_id", "post_pt_root_id"]
    assert all(n in cols for n in need), cols
    if out.exists() and stat.exists():
        df = pd.read_csv(out)
        log("cached synapse rows:", len(df), out.name)
        return df, json.loads(stat.read_text())
    import pyarrow as pa
    import pyarrow.csv as pacsv
    types = {h[0]: (pa.int64() if h[1].strip() == "bigint" else pa.float64()) for h in header}
    http = HTTPStream(url)
    comp = pa.CompressedInputStream(pa.PythonFile(io.BufferedReader(http, buffer_size=1 << 22), mode="r"), "gzip")
    reader = pacsv.open_csv(comp, read_options=pacsv.ReadOptions(column_names=cols, block_size=1 << 26),
                            convert_options=pacsv.ConvertOptions(column_types=types))
    kept, n_rows, n_batches, t0 = [], 0, 0, time.time()
    log("synapse stream: start; keeping ctr_pt in", tag, "(4 x 4 x 40 nm voxels)")
    for batch in reader:
        n_rows += batch.num_rows
        n_batches += 1
        x, y, z = (batch.column(cols.index("ctr_pt_position_" + a)).to_numpy(zero_copy_only=False) for a in "xyz")
        m = (x >= lo[0]) & (x < hi[0]) & (y >= lo[1]) & (y < hi[1]) & (z >= lo[2]) & (z < hi[2])
        if m.any():
            kept.append(batch.filter(pa.array(m)).to_pandas())
        if n_batches % 20 == 0:
            done = http.pos / max(1, http.size or 1)
            el = time.time() - t0
            log("  synapse stream: %5.1f %% of %.1f GB, %d rows read, %d kept, %.1f MB/s, about %.0f min left"
                % (100 * done, (http.size or 0) / 1e9, n_rows, sum(len(k) for k in kept),
                   http.pos / 1e6 / max(el, 1e-6), el * (1 - done) / max(done, 1e-9) / 60))
    if http.size is None or http.pos != http.size:
        raise RuntimeError("synapse stream ended at byte %s of %s" % (http.pos, http.size))
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=cols)
    stats = {"url": url, "bytes_read": int(http.pos), "rows_read": int(n_rows), "rows_kept": int(len(df)),
             "box_lo_xyz_4x4x40": [int(v) for v in lo], "box_hi_xyz_4x4x40": [int(v) for v in hi],
             "reconnects": int(http.reopened), "seconds": round(time.time() - t0, 1),
             "read_on": datetime.date.today().isoformat()}
    tmp = out.with_name(out.name + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    stat.write_text(json.dumps(stats, indent=1))
    log("synapse stream: done.", json.dumps(stats))
    return df, stats


def build_synapses(name, df, stats, seg, cle):
    """Rows of the kept synapse table whose centre lies in the crop, with checks against the voxels."""
    c = CROPS[name]

    def pos(prefix):
        return np.stack([df[prefix + "_position_" + a].to_numpy(np.float64) for a in "xyz"], 1)

    out = {}
    for k in ("pre_pt", "ctr_pt", "post_pt"):
        out[k] = table_pos_to_crop(pos(k), name)
    sel = np.where(inside(out["ctr_pt"][1], seg.shape))[0]
    rows, chk = [], {"n": int(len(sel)), "ctr_on_cleft_with_same_id": 0, "ctr_on_any_cleft": 0,
                     "ctr_on_cleft_with_same_id_if_y_and_x_are_swapped": 0,
                     "pre_in_crop": 0, "pre_seg_equals_pre_root_id": 0,
                     "post_in_crop": 0, "post_seg_equals_post_root_id": 0}
    for i in sel:
        e = {"id": int(df["id"].iloc[i]), "size": float(df["size"].iloc[i]),
             "pre_root_id": int(df["pre_pt_root_id"].iloc[i]), "post_root_id": int(df["post_pt_root_id"].iloc[i])}
        for k, short in (("pre_pt", "pre"), ("ctr_pt", "ctr"), ("post_pt", "post")):
            nm, vox = out[k][0][i], out[k][1][i]
            e[short + "_nm"] = [round(float(v), 1) for v in nm]
            e[short + "_vox"] = [int(v) for v in vox]
            e[short + "_inside"] = bool(inside(vox, seg.shape))
        cv = int(cle[tuple(e["ctr_vox"])])
        e["cleft_id_at_ctr"] = cv
        chk["ctr_on_any_cleft"] += cv != 0
        chk["ctr_on_cleft_with_same_id"] += cv == e["id"]
        zz, yy, xx = e["ctr_vox"]
        if xx < cle.shape[1] and yy < cle.shape[2]:
            chk["ctr_on_cleft_with_same_id_if_y_and_x_are_swapped"] += int(cle[zz, xx, yy]) == e["id"]
        for short in ("pre", "post"):
            if e[short + "_inside"]:
                s = int(seg[tuple(e[short + "_vox"])])
                e[short + "_neuron"] = s
                chk[short + "_in_crop"] += 1
                chk[short + "_seg_equals_" + short + "_root_id"] += s == e[short + "_root_id"]
            else:
                e[short + "_neuron"] = None
        rows.append(e)
    ids_in_volume = set(np.unique(cle).tolist()) - {0}
    ids_in_table = {e["id"] for e in rows}
    chk = {k: int(v) for k, v in chk.items()}
    chk["cleft_ids_in_crop"] = len(ids_in_volume)
    chk["cleft_ids_in_crop_with_a_row_in_the_kept_table"] = len(ids_in_volume & set(df["id"].astype(np.int64).tolist()))
    chk["rows_whose_id_is_a_cleft_id_in_crop"] = len(ids_in_table & ids_in_volume)
    return {
        "axis_order": "z, y, x",
        "units": "nm relative to the crop origin (the corner of voxel [0, 0, 0]); voxel index = floor(nm / voxel_nm). "
                 "A point of the table is a voxel of 4 x 4 x 40 nm; its centre is given.",
        "voxel_nm": [c["res_xyz"][2], c["res_xyz"][1], c["res_xyz"][0]],
        "source": stats["url"],
        "source_table": "view synapses_pni_2_v1_filtered_view of the public annotation database at materialization "
                        "v1300; automatic synapse detection and automatic partner assignment, not manual",
        "selection": "rows whose ctr_pt (centre of the detected junction) lies inside the crop",
        "stream": stats,
        "fields": {
            "id": "synapse id of the table; the same number labels the PSD in cleft_ids.npy",
            "size": "column 'size', copied (voxel count of the PSD object as the source states it)",
            "pre_root_id, post_root_id": "columns pre_pt_root_id, post_pt_root_id (v1300); the same numbers as in "
                                         "seg.npy. They are larger than 2**53: read as integers, never as floats.",
            "pre_nm, ctr_nm, post_nm": "the three points of the row, nm, z y x, relative to the crop origin",
            "pre_vox, ctr_vox, post_vox": "voxel index of each point in the crop, z y x (can be outside)",
            "pre_inside, ctr_inside, post_inside": "whether that voxel is inside the crop",
            "pre_neuron, post_neuron": "value of seg.npy at pre_vox / post_vox, null if outside the crop",
            "cleft_id_at_ctr": "value of cleft_ids.npy at ctr_vox",
        },
        "checks": chk,
        "synapses": rows,
    }


# ----------------------------------------------------------------------------- output

def save_json(path: Path, obj):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    tmp.replace(path)


def save_npy(path: Path, a):
    tmp = path.with_name(path.stem + ".part.npy")
    np.save(tmp, np.ascontiguousarray(a))
    tmp.replace(path)


def preview(path: Path, name, raw, seg, cls, synapses=None):
    """Middle section: raw | raw + seg colours + label boundaries | raw + classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    z = raw.shape[0] // 2
    p1, p99 = np.percentile(raw, [1, 99])
    r = np.clip((raw[z].astype(np.float32) - p1) / max(1.0, p99 - p1), 0, 1)     # display only
    g = np.dstack([r, r, r])
    ids, inv = np.unique(seg[z], return_inverse=True)
    rng = np.random.default_rng(7)
    col = rng.uniform(0.15, 1.0, (len(ids), 3))
    lab = col[inv.reshape(seg[z].shape)]
    a = np.where(seg[z] != 0, 0.38, 0.0)[..., None]
    ov = g * (1 - a) + lab * a
    b = find_boundaries(seg[z], mode="inner")
    ov[b] = (1.0, 0.95, 0.2)
    cc = np.array([[0, 0, 0], [1.0, 0.1, 0.1], [0.2, 0.5, 1.0], [0.1, 0.9, 0.3]])
    ca = np.where(cls[z] != 0, 0.55, 0.0)[..., None]
    co = g * (1 - ca) + cc[cls[z]] * ca
    c = CROPS[name]
    h, w = raw.shape[1:]
    fig, ax = plt.subplots(1, 3, figsize=(3 * 6.0 * w / max(h, w), 6.4 * h / max(h, w) + 0.2), dpi=110)
    ax[0].imshow(g, interpolation="nearest")
    ax[0].set_title("%s raw, section %d of %d, voxel %d x %d x %d nm (x y z); grey scale shown from %d to %d"
                    % ((name, z, raw.shape[0]) + tuple(c["res_xyz"]) + (p1, p99)), fontsize=8)
    ax[1].imshow(ov, interpolation="nearest")
    ax[1].set_title("seg (flat segmentation v1300): colours, label boundaries in yellow", fontsize=8)
    ax[2].imshow(co, interpolation="nearest")
    ax[2].set_title("classes: red cleft (PSD), blue nucleus, green blood vessel lumen", fontsize=8)
    if synapses is not None:
        for e in synapses["synapses"]:
            if abs(e["ctr_vox"][0] - z) <= 1:
                ax[2].plot(e["ctr_vox"][2], e["ctr_vox"][1], "o", ms=5 if name == FINE else 2, mfc="none", mec="yellow", mew=0.7)
                if e["pre_inside"] and e["post_inside"]:
                    ax[2].plot([e["pre_vox"][2], e["post_vox"][2]], [e["pre_vox"][1], e["post_vox"][1]], "-", color="cyan", lw=0.7)
        ax[2].set_title("classes: red cleft (PSD), blue nucleus, green vessel lumen; circles: synapse rows within 1 section, line pre-post", fontsize=7)
    if name == COARSE:
        f = CROPS[FINE]
        x0 = (f["offset_xyz"][0] * f["res_xyz"][0]) / c["res_xyz"][0] - c["offset_xyz"][0]
        y0 = (f["offset_xyz"][1] * f["res_xyz"][1]) / c["res_xyz"][1] - c["offset_xyz"][1]
        wx = f["size_xyz"][0] * f["res_xyz"][0] / c["res_xyz"][0]
        wy = f["size_xyz"][1] * f["res_xyz"][1] / c["res_xyz"][1]
        zf0 = f["offset_xyz"][2] - c["offset_xyz"][2]
        for a_ in ax:
            a_.plot([x0, x0 + wx, x0 + wx, x0, x0], [y0, y0, y0 + wy, y0 + wy, y0], "-", color="white", lw=1.0)
        ax[0].set_xlabel("white box: x-y outline of the crop 'microns' (it spans sections %d to %d)" % (zf0, zf0 + f["size_xyz"][2] - 1), fontsize=8)
    for a_ in ax:
        a_.set_xticks([])
        a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nested_check(fine, coarse):
    """Compare the fine crop, reduced 4 x 4 in y and x, with its place in the coarse crop."""
    f, c = CROPS[FINE], CROPS[COARSE]
    k = [a // b for a, b in zip(c["res_xyz"], f["res_xyz"])]            # (4, 4, 1) in x, y, z
    o = [fo // kk - co for fo, kk, co in zip(f["offset_xyz"], k, c["offset_xyz"])]
    n = [s // kk for s, kk in zip(f["size_xyz"], k)]
    assert all(fo % kk == 0 for fo, kk in zip(f["offset_xyz"], k))
    sl = (slice(o[2], o[2] + n[2]), slice(o[1], o[1] + n[1]), slice(o[0], o[0] + n[0]))
    raw_f, seg_f = fine
    raw_c, seg_c = coarse[0][sl], coarse[1][sl]
    z, y, x = raw_f.shape
    mean = raw_f.reshape(z, y // k[1], k[1], x // k[0], k[0]).astype(np.float64).mean((2, 4))
    d = mean - raw_c
    corr = float(np.corrcoef(mean.ravel(), raw_c.ravel())[0, 1])
    # the same test with the coarse block shifted by one voxel: a wrong offset would do as well
    sh = (slice(o[2], o[2] + n[2]), slice(o[1] + 1, o[1] + 1 + n[1]), slice(o[0] + 1, o[0] + 1 + n[0]))
    corr_shift = float(np.corrcoef(mean.ravel(), coarse[0][sh].ravel())[0, 1])
    agree = [float((seg_f[:, i::k[1], j::k[0]] == seg_c).mean()) for i in range(k[1]) for j in range(k[0])]
    return {
        "fine_in_coarse_vox_zyx": [[o[2], o[2] + n[2]], [o[1], o[1] + n[1]], [o[0], o[0] + n[0]]],
        "scale_zyx": [k[2], k[1], k[0]],
        "raw_4x4_mean_vs_coarse": {"pearson_r": round(corr, 5), "max_abs_diff": round(float(np.abs(d).max()), 2),
                                   "mean_abs_diff": round(float(np.abs(d).mean()), 3),
                                   "pearson_r_if_coarse_shifted_by_1_voxel_in_y_and_x": round(corr_shift, 5)},
        "seg_every_4th_voxel_equals_coarse": {"best_of_16_phases": round(max(agree), 5),
                                              "worst_of_16_phases": round(min(agree), 5),
                                              "phase_0_0": round(agree[0], 5)},
    }


def caliber_stats(seg, res_xyz):
    """How thin the objects are: for each object, the median over the sections it occurs in of the
    diameter of the circle with the area the object has in that section (x-y plane)."""
    ids, inv = np.unique(seg, return_inverse=True)
    inv = inv.reshape(seg.shape)
    n = len(ids)
    area = np.stack([np.bincount(inv[z].ravel(), minlength=n) for z in range(seg.shape[0])])
    d = 2.0 * np.sqrt(area * float(res_xyz[0] * res_xyz[1]) / np.pi)
    med = np.array([np.median(d[area[:, j] > 0, j]) for j in range(n) if ids[j] != 0])
    return {"objects": int(len(med)), "median_nm": round(float(np.median(med)), 1),
            "share_below_200_nm": round(float((med < 200).mean()), 3),
            "share_below_400_nm": round(float((med < 400).mean()), 3),
            "share_above_1000_nm": round(float((med > 1000).mean()), 3)}


def section_stats(raw):
    z = raw.shape[0]
    flat = raw.reshape(z, -1)
    zero = (flat == 0).mean(1)
    d = raw[:, ::2, ::2].astype(np.float32)
    d -= d.mean((1, 2), keepdims=True)
    d /= d.std((1, 2), keepdims=True) + 1e-6
    cc = (d[1:] * d[:-1]).mean((1, 2))
    return {"sections_with_more_than_1_percent_zero_voxels": [int(i) for i in np.where(zero > 0.01)[0]],
            "zero_voxel_fraction": round(float((raw == 0).mean()), 6),
            "mean": round(float(raw.mean()), 2), "std": round(float(raw.std()), 2),
            "min": int(raw.min()), "max": int(raw.max()),
            "correlation_of_neighbouring_sections": {"median": round(float(np.median(cc)), 3),
                                                     "min": round(float(cc.min()), 3),
                                                     "pair_with_min": [int(cc.argmin()), int(cc.argmin()) + 1]}}


def search(src: Path):
    """Print how the place of the crops was chosen (steps 1, 2, 4, 5 of the docstring). Reads the
    vessel mask at 2048 nm and one 768 x 768 x 256 block of image and segmentation at 32 nm."""
    import edt
    LAYERS["bv"] = VESSEL_MASK
    infos = {k: read_info(k, src) for k in ("em", "seg", "bv")}
    notes = []
    # 1, 2: proofread neurons near a vessel
    cells, _ = read_table(TABLES_V1300, CELL_TABLE, src)
    pos_nm = np.array([wkb_pointz(h) for h in cells["pt_position"]]) * np.array(TABLE_VOXEL_NM_XYZ)
    res = (2048, 2048, 1280)
    _, sc = find_scale(infos["bv"][0], res)
    off = [int(round(float(v))) for v in sc["voxel_offset"]]
    mask = cutout("bv", res, off, [int(v) for v in sc["size"]], src, infos, notes) > 0          # [z, y, x]
    dist = edt.edt(~mask, anisotropy=(res[2], res[1], res[0]), black_border=False, parallel=4)   # nm
    idx = np.floor(pos_nm / np.array(res)).astype(np.int64) - np.array(off)                      # x, y, z
    ok = np.all((idx >= 0) & (idx < np.array(mask.shape)[::-1]), axis=1)
    d = np.full(len(cells), np.nan)
    d[ok] = dist[idx[ok, 2], idx[ok, 1], idx[ok, 0]]
    flag = lambda col: cells[col].astype(str).str.lower().isin(["t", "true"]).to_numpy()
    both = flag("dendrite_cleaned") & flag("axon_cleaned")
    cand = np.where(both & (d >= 4000) & (d <= 8000))[0]
    cand = cand[np.argsort(d[cand], kind="stable")]
    log("vessel mask: %.4f of %s voxels; neurons with both flags: %d; of those 4 to 8 um from a vessel voxel: %d"
        % (mask.mean(), mask.shape, both.sum(), len(cand)))
    for i in cand[:12]:
        r = cells.iloc[i]
        log("  nucleus %d root %d %s %s/%s at 32 nm voxel x,y,z = %s, %.1f um from the vessel mask"
            % (r["id"], r["pt_root_id"], r["cell_type"], r["dendrite_strategy"], r["axon_strategy"],
               [int(v) for v in pos_nm[i] // np.array([32, 32, 40])], d[i] / 1000))
    # 4, 5: windows in the block around the chosen neuron
    b = SEARCH_BLOCK
    em = cutout("em", b["res_xyz"], b["offset_xyz"], b["size_xyz"], src, infos, notes)
    seg = cutout("seg", b["res_xyz"], b["offset_xyz"], b["size_xyz"], src, infos, notes)
    log("block %s: %s" % (b, json.dumps(section_stats(em))))
    root = int(cells.loc[cells["id"] == SEARCH_NUCLEUS, "pt_root_id"].iloc[0])
    soma, ves = seg == np.uint64(root), seg == np.uint64(VESSEL_ROOT_ID)

    def box2(a, w):
        c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]

    def box3(c, dz, dy, dx):
        return (c[dz:, dy:, dx:] - c[:-dz, dy:, dx:] - c[dz:, :-dy, dx:] - c[dz:, dy:, :-dx]
                + c[:-dz, :-dy, dx:] + c[:-dz, dy:, :-dx] + c[dz:, :-dy, :-dx] - c[:-dz, :-dy, :-dx])

    fs = box2(soma.sum(0).astype(np.int64), 512) / soma.sum()       # share of the cell's voxels per window
    fv = box2(ves.sum(0).astype(np.int64), 512) / ves.sum()
    ids, cnt = np.unique(seg, return_counts=True)
    big = ids[cnt > 0.004 * seg.size]
    c3 = np.pad(np.isin(seg, big).astype(np.int32), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    bf = box3(c3, 64, 128, 128) / float(64 * 128 * 128)
    log("cell segment %d: %d voxels; vessel segment: %d voxels; %d objects above 0.4 %% of the block"
        % (root, soma.sum(), ves.sum(), len(big)))
    rows, m, mz = [], 32, 16
    for z in range(mz, seg.shape[0] - 64 - mz + 1, 16):
        for y in range(0, seg.shape[1] - 128 + 1, 16):
            for x in range(0, seg.shape[2] - 128 + 1, 16):
                if bf[z, y, x] >= 0.10:
                    continue
                ys = np.arange(max(0, y + 128 + m - 512), min(seg.shape[1] - 512, y - m) + 1)
                xs = np.arange(max(0, x + 128 + m - 512), min(seg.shape[2] - 512, x - m) + 1)
                if len(ys) == 0 or len(xs) == 0:
                    continue
                a, v = fs[np.ix_(ys, xs)], fv[np.ix_(ys, xs)]
                good = (a >= 0.9) & (v >= 0.9)
                if not good.any():
                    continue
                k = np.unravel_index(np.where(good, a + v, -1).argmax(), a.shape)
                sub = seg[z:z + 64, y:y + 128, x:x + 128]
                u, n = np.unique(sub, return_counts=True)
                rows.append((len(u), float(bf[z, y, x]), float(n.max()) / sub.size, z, y, x,
                             int(ys[k[0]]), int(xs[k[1]]), float(a[k]), float(v[k])))
    rows.sort(reverse=True)
    ox, oy, oz = b["offset_xyz"]
    f, c = CROPS[FINE], CROPS[COARSE]
    log("%d sub-windows pass; the best 12 (x, y, z in 32 nm voxels of the volume):" % len(rows))
    for n_ids, bfr, mx, z, y, x, y0, x0, a, v in rows[:12]:
        fine = (ox + x, oy + y, oz + z)
        coarse = (ox + x0, oy + y0, oz)
        chosen = (fine == (f["offset_xyz"][0] // 4, f["offset_xyz"][1] // 4, f["offset_xyz"][2])
                  and coarse == tuple(c["offset_xyz"]))
        log("  ids %d, in large objects %.3f, largest object %.3f, sub-window at %s, window at %s "
            "(cell %.3f, vessel %.3f)%s" % (n_ids, bfr, mx, fine, coarse, a, v, "   <-- the crops" if chosen else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder of the crop 'microns'; 'microns_coarse' is written beside it")
    ap.add_argument("--src", default=None, help="cache folder (default: <out>/../../src/microns)")
    ap.add_argument("--skip-synapses", action="store_true", help="do not read the 20 GB synapse table")
    ap.add_argument("--search", action="store_true", help="only print how the place of the crops was chosen")
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out)).resolve()
    src = Path(os.path.expanduser(args.src)).resolve() if args.src else out.parent.parent / "src" / "microns"
    dirs = {FINE: out, COARSE: out.parent / COARSE}
    if args.search:
        src.mkdir(parents=True, exist_ok=True)
        search(src)
        return 0
    for d in list(dirs.values()) + [src]:
        d.mkdir(parents=True, exist_ok=True)
    log("out:", {k: str(v) for k, v in dirs.items()}, "cache:", str(src))

    # 1. info files: what the source says about scales and voxel sizes
    infos, sources = {}, []
    for layer in LAYERS:
        infos[layer] = read_info(layer, src)
        sources.append(infos[layer][1])
        info = infos[layer][0]
        log("info %-7s type=%s dtype=%s scales=%s" % (layer, info["type"], info["data_type"],
            [s["key"] for s in info["scales"]][:4]))
    em_res = [s["resolution"] for s in infos["em"][0]["scales"]]
    seg_res = [s["resolution"] for s in infos["seg"][0]["scales"]]
    assert [float(v) for v in em_res[0]] == [8.0, 8.0, 40.0], em_res[0]
    assert [float(v) for v in seg_res[0]] == [8.0, 8.0, 40.0], seg_res[0]
    assert [float(v) for v in em_res[2]] == [32.0, 32.0, 40.0] and [float(v) for v in seg_res[2]] == [32.0, 32.0, 40.0]
    assert infos["em"][0]["data_type"] == "uint8" and infos["seg"][0]["data_type"] == "uint64"

    # 2. voxels
    data, notes = {}, {FINE: [], COARSE: []}
    for name in (FINE, COARSE):
        raw, seg, cle, nuc = fetch_crop(name, src, infos, notes[name])
        data[name] = {"raw": raw, "seg": seg, "cle": cle, "nuc": nuc, "cls": make_classes(seg, cle, nuc)}
    nest = nested_check((data[FINE]["raw"], data[FINE]["seg"]), (data[COARSE]["raw"], data[COARSE]["seg"]))
    log("nested check:", json.dumps(nest))

    # 3. small tables
    cells = {name: build_cells(name, data[name]["seg"], src, sources) for name in (FINE, COARSE)}

    def write(syn, syn_state):
        srcs = list(dict.fromkeys([LAYERS[k] for k in LAYERS] + sources))
        for name in (FINE, COARSE):
            d, c, o = data[name], CROPS[name], dirs[name]
            raw, seg, cls = d["raw"], d["seg"], d["cls"]
            save_npy(o / "raw.npy", raw)
            save_npy(o / "seg.npy", seg)
            save_npy(o / "classes.npy", cls)
            save_npy(o / "cleft_ids.npy", d["cle"])
            save_npy(o / "nucleus_ids.npy", d["nuc"])
            save_json(o / "cells.json", cells[name])
            if syn[name] is not None:
                save_json(o / "synapses.json", syn[name])
            elif (o / "synapses.json").exists():
                (o / "synapses.json").unlink()      # never leave a table that this run did not check
            ids, counts = np.unique(seg, return_counts=True)
            n_obj = int((ids != 0).sum())
            ccount = {k: int((cls == int(k)).sum()) for k in CLASS_NAMES}
            st = section_stats(raw)
            cal = caliber_stats(seg, c["res_xyz"])
            flagged = [e for e in cells[name]["cells"] if e["voxels_in_crop"] > 0 and (e["dendrite_cleaned"] or e["axon_cleaned"])]
            proof = []
            for e in cells[name]["cells"]:
                p = e["proofreading_status_and_strategy_v1412"]
                if e["voxels_in_crop"] > 0 and p and p["valid_id_equals_root_id"] and (p["status_dendrite"] or p["status_axon"]):
                    proof.append(e)
            pv = sum(e["voxels_in_crop"] for e in proof)
            idx, sc = find_scale(infos["em"][0], c["res_xyz"])
            _, ssc = find_scale(infos["seg"][0], c["res_xyz"])
            other = COARSE if name == FINE else FINE
            nn = [
                "Arrays are [z, y, x]. The source stores x, y, z; the arrays were transposed. voxel_nm, shape and "
                "offset_vox are z, y, x. offset_vox counts voxels of this level from the origin of the volume's physical "
                "frame, the way Neuroglancer and cloud-volume do (the first stored EM voxel of this level is at "
                "x, y, z = %s)." % [int(v) for v in sc.get("voxel_offset", [0, 0, 0])],
                "The image is the public EM layer as stored (level '%s'); the finest public level is 8 x 8 x 40 nm. "
                "The annotation tables give positions in voxels of 4 x 4 x 40 nm." % sc["key"],
                "seg.npy is level '%s' of the flat segmentation seg_m1300 as stored in the bucket (nothing was "
                "resampled here): %d objects, %.4f of the voxels are 0. It is the automatic segmentation with the "
                "proofreading edits made up to materialization v1300 (13 January 2025). Most objects are NOT "
                "proofread." % (ssc["key"], n_obj, float((seg == 0).mean())),
                "Proofread objects: %d root id(s) in this crop have a row with the same root id in the documented "
                "table proofreading_status_and_strategy (archive v1412) with status_dendrite or status_axon true; "
                "they cover %.4f of the voxels. They are listed in cells.json. For those objects 'proofread' "
                "means false merges were removed from the named compartment of that cell; it does not mean that "
                "every voxel of the object in this crop was checked. The cell table of v1300 itself (aibs_cell_info, "
                "columns dendrite_cleaned / axon_cleaned; a table that the documentation pages do not describe) flags "
                "%d root id(s) in this crop, covering %.4f of the voxels; the ones missing from the first count "
                "changed their root id between v1300 and v1412." % (len(proof), pv / seg.size, len(flagged),
                                                               sum(e["voxels_in_crop"] for e in flagged) / seg.size),
                "Caliber of the objects (per object: median over sections of the diameter of the circle with the "
                "object's area in the section): %s." % json.dumps(cal),
                "classes.npy: 1 = PSD of a detected synapse (%d voxels), 2 = nucleus (%d voxels), 3 = blood vessel "
                "lumen (%d voxels). All three come from automatic segmentations of the source, none is manual. Class 3 "
                "is segment %d of seg.npy, the object that the MICrONS Explorer page shows as 'blood vessels'; in "
                "this block it is the bright inside of the capillary (mean of raw there %s, of the whole crop %.1f), "
                "and the wall of the vessel belongs to other segments. Class 2 was read at 64 x 64 x 40 nm (the "
                "finest level of that layer) and repeated to this grid. Voxels of class 1 keep their id in seg.npy."
                % (ccount["1"], ccount["2"], ccount["3"], VESSEL_ROOT_ID,
                   ("%.1f" % float(raw[cls == 3].mean())) if ccount["3"] else "n/a", float(raw.mean())),
                "Extra files: cleft_ids.npy (uint32, PSD instance ids = synapse ids of the table), nucleus_ids.npy "
                "(uint32, nucleus ids = 'nucleus_id' in cells.json), cells.json, synapses.json. "
                "Coordinates in the JSON tables are nm relative to the crop origin (the corner of voxel [0, 0, 0]), "
                "axis order z, y, x; voxel index = floor(nm / voxel_nm).",
                "Root ids are larger than 2**53. Read the JSON tables with a parser that keeps integers "
                "(Python's json does); a float64 cannot hold them.",
                "Image statistics: %s. The image has low contrast as stored: the 1st and 99th percentile of raw are "
                "%d and %d." % ((json.dumps(st),) + tuple(int(v) for v in np.percentile(raw, [1, 99]))),
                "Nesting: the crop 'microns' is voxels z %s, y %s, x %s of the crop 'microns_coarse'; one coarse voxel "
                "is 1 x 4 x 4 fine voxels (z, y, x). Measured: %s. So the bucket's 32 nm image is the 4 x 4 mean of "
                "the 8 nm image, and its 32 nm segmentation is the 8 nm segmentation at every 4th voxel in y and x "
                "(first voxel of each 4 x 4 group), not the most frequent label."
                % (tuple(nest["fine_in_coarse_vox_zyx"]) + (json.dumps(
                    {k: nest[k] for k in ("raw_4x4_mean_vs_coarse", "seg_every_4th_voxel_equals_coarse")}),)),
                "In 'sources', the data URLs were read by the script; the documentation pages and the vessel mask "
                "were read by hand on 23 November 2026 for the facts stated here and for the choice of the place.",
            ]
            if syn[name] is not None:
                k = syn[name]["checks"]
                nn.append("synapses.json: %d rows of the public synapse table (v1300, automatic detection) with the centre "
                          "point in the crop. Checks against the voxels: centre on a PSD with the same id %d of %d "
                          "(%d if y and x are swapped, which tests the axis order); "
                          "pre point inside the crop %d, of which seg equals pre_root_id %d; post point inside %d, of "
                          "which seg equals post_root_id %d." % (k["n"], k["ctr_on_cleft_with_same_id"], k["n"],
                          k["ctr_on_cleft_with_same_id_if_y_and_x_are_swapped"],
                          k["pre_in_crop"], k["pre_seg_equals_pre_root_id"], k["post_in_crop"], k["post_seg_equals_post_root_id"]))
            else:
                nn.append("synapses.json is absent: " + syn_state)
            nn += notes[name]
            meta = {
                "name": name,
                "dataset": "MICrONS cortical mm^3, minnie65 (mouse visual cortex, serial-section TEM), public release; "
                           "flat segmentation of materialization v1300",
                "voxel_nm": [c["res_xyz"][2], c["res_xyz"][1], c["res_xyz"][0]],
                "shape": [int(v) for v in raw.shape],
                "offset_vox": [c["offset_xyz"][2], c["offset_xyz"][1], c["offset_xyz"][0]],
                "level": "scale '%s' of the EM layer (index %d of its info file) and scale '%s' of seg_m1300; voxel "
                         "%g x %g x %g nm (x, y, z)" % ((sc["key"], idx, ssc["key"]) + tuple(float(v) for v in c["res_xyz"])),
                "sources": srcs + DOC_URLS,
                "seg_kind": "automatic",
                "seg_meaning": "Each id is a root id of the flat MICrONS segmentation at v1300: one cell or fragment of a "
                               "cell (neurons, glia, and one object for the blood vessels), from automatic segmentation "
                               "plus the proofreading edits made so far; 0 = no label.",
                "class_names": CLASS_NAMES,
                "licence": LICENCE,
                "citation": CITATION,
                "fetched": datetime.date.today().isoformat(),
                "notes": nn,
                "axis_order": "z, y, x",
                "n_objects": n_obj,
                "purpose": c["purpose"],
                "related_crop": {"name": other, "fine_in_coarse_vox_zyx": nest["fine_in_coarse_vox_zyx"],
                                 "coarse_voxel_in_fine_voxels_zyx": nest["scale_zyx"],
                                 "note": "fine index = (coarse index - start) * scale, per axis"},
                "tables": {
                    "cells.json": "cells of the public cell table whose root id is in the crop; nm relative to the crop origin, z y x",
                    "synapses.json": ("rows of the public synapse table with the centre in the crop; nm relative to the "
                                      "crop origin, z y x") if syn[name] is not None else "absent: " + syn_state,
                },
                "extra_files": ["cleft_ids.npy", "nucleus_ids.npy", "cells.json"] + (["synapses.json"] if syn[name] is not None else []),
            }
            save_json(o / "meta.json", meta)
            preview(o / "preview.png", name, raw, seg, cls, syn[name])
            log("wrote", name, "->", str(o), "objects:", n_obj, "class voxels:", ccount,
                "proofread objects:", len(proof), "cells:", len(cells[name]["cells"]),
                "synapses:", None if syn[name] is None else syn[name]["checks"]["n"])

    # 4. write the crops, then read the synapse table (long) and write again with synapses.json
    syn = {FINE: None, COARSE: None}
    if args.skip_synapses:
        write(syn, "this run was started with --skip-synapses")
    else:
        write(syn, "the synapse table was still being read, or the run stopped before it was read to the end")
        df, stats = stream_synapses(src, sources)
        for name in (FINE, COARSE):
            syn[name] = build_synapses(name, df, stats, data[name]["seg"], data[name]["cle"])
            log(name, "synapse checks:", json.dumps(syn[name]["checks"]))
        write(syn, "")
    log("done")


if __name__ == "__main__":
    sys.exit(main())

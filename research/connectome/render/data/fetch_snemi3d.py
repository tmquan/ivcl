#!/usr/bin/env python3
"""fetch_snemi3d.py -- fetch the SNEMI3D training stack and store it in the common crop format.

SOURCE
  SNEMI3D: 3D Segmentation of neurites in EM images (ISBI 2013 challenge).
  Training stack of mouse cortex, serial-section SEM, from the Lichtman lab
  (Kasthuri et al. 2015), "manually annotated by Daniel R. Berger" (challenge page).
  Files: Zenodo record 7142003, https://doi.org/10.5281/zenodo.7142003, one file
  snemi.zip (194 608 530 bytes, md5 3d25a7025f66698f33c7850ace885939). The challenge
  page https://snemi3d.grand-challenge.org/ names this record as the host of the
  training and test data. The zip holds image/train-input.tif, seg/train-labels.tif
  and image/test-input.tif (the test stack has no labels and is not used here).

LICENCE
  Creative Commons Attribution 4.0 International (licence id "cc-by-4.0" in the
  Zenodo record).

CITATION
  Arganda-Carreras, I., Seung, H. S., Vishwanathan, A., & Berger, D. R. (2013).
  SNEMI3D: 3D Segmentation of neurites in EM images [Dataset]. Zenodo.
  https://doi.org/10.5281/zenodo.7142003
  The images and labels come from: Kasthuri, N., et al. (2015). Saturated
  Reconstruction of a Volume of Neocortex. Cell 162(3), 648-661.
  https://doi.org/10.1016/j.cell.2015.06.054

WHAT IT WRITES
  Two crops, side by side. --out names the folder of the first; the second gets
  the same path with "_full" appended.
    <out>/        "snemi3d":      64 sections x 512 x 512 voxels, the window of the
                                  training stack with the most thin neurites (rule below).
    <out>_full/   "snemi3d_full": the whole training stack, 100 x 1024 x 1024.
  In each folder:
    raw.npy         uint8  [z, y, x], C order: the EM image, bytes unchanged.
    seg.npy         uint64 [z, y, x]: the manual neurite labels, ids as in the source
                    (1..400), 0 = not labelled.
    meta.json       the description required by the crop format, plus the evidence
                    for the voxel size, the rule that chose the window, a check that
                    the labels lie on the image (border_check) and a list of
                    neighbouring raw sections that show the same tissue twice
                    (raw_repeated_sections; source sections 33 and 34 are such a pair).
    objects.json    one row per label id present: voxel count, bounding box, centroid,
                    in-plane diameter, number of connected components. Derived here,
                    not part of the source.
    preview.png     middle section: raw | raw with labels in colour and label boundaries.
    preview_zoom.png  the 128 x 128 pixels of the middle section with the most label
                    boundaries, enlarged 8 times: raw | raw with label boundaries only,
                    to check that boundaries lie on membranes.
    preview_zx.png  a section through z and x at the middle y, drawn at the physical
                    aspect ratio: raw | raw with labels.
  There are no class labels, so no classes.npy.

VOXEL SIZE (z, y, x) = 29 x 6 x 6 nm. See VOXEL_EVIDENCE below: the two primary
  sources disagree on z by one nanometre (paper: 29 nm sections; TIFF header: 30 nm).

USE
  python fetch_snemi3d.py --out ~/connectome/data/crops/snemi3d
  The download is cached under ~/connectome/data/src/snemi3d/ (--src to change) and
  checked against the md5 of the Zenodo record, so the script can be run again: it
  downloads nothing the second time and rewrites the same files.
  The script stops, and writes nothing for that crop, if the source files are not the
  ones described here (shape, dtype, header, 400 ids) or if raw and seg do not look
  registered (see border_check).

NEEDS  numpy, scipy, tifffile, pillow, edt, connected-components-3d.
"""

import argparse
import colorsys
import datetime
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np

# --------------------------------------------------------------------------- source

ZENODO_RECORD_API = "https://zenodo.org/api/records/7142003"
ZIP_URLS = [
    "https://zenodo.org/api/records/7142003/files/snemi.zip/content",
    "https://zenodo.org/records/7142003/files/snemi.zip?download=1",
]
ZIP_NAME = "snemi.zip"
ZIP_BYTES = 194608530
ZIP_MD5 = "3d25a7025f66698f33c7850ace885939"       # "checksum" of the Zenodo record
MEMBER_RAW = "image/train-input.tif"
MEMBER_SEG = "seg/train-labels.tif"
EXPECT_SHAPE = (100, 1024, 1024)
EXPECT_N_IDS = 400

DATASET = "SNEMI3D (ISBI 2013 challenge), training stack"
LICENCE = ("Creative Commons Attribution 4.0 International (CC BY 4.0); "
           "licence id 'cc-by-4.0' in Zenodo record 7142003")
CITATION = (
    "Arganda-Carreras, I., Seung, H. S., Vishwanathan, A., & Berger, D. R. (2013). "
    "SNEMI3D: 3D Segmentation of neurites in EM images [Dataset]. Zenodo. "
    "https://doi.org/10.5281/zenodo.7142003. "
    "Images and labels from: Kasthuri, N., et al. (2015). Saturated Reconstruction of a "
    "Volume of Neocortex. Cell 162(3), 648-661. https://doi.org/10.1016/j.cell.2015.06.054"
)

# --------------------------------------------------------------------------- voxel size

VOXEL_NM = (29.0, 6.0, 6.0)   # z, y, x

VOXEL_EVIDENCE = {
    "chosen_zyx_nm": list(VOXEL_NM),
    "why": (
        "x and y: the TIFF headers of the challenge files give 0.166666 pixels per nm, "
        "that is 6.00002 nm per pixel. z: one TIFF page is one section, and the paper "
        "that produced the sections gives their thickness as 29 nm. The TIFF headers "
        "give the z spacing as 30.0 nm, and the data page of the paper's project "
        "also says 30. The primary sources therefore differ by 1 nm (3 %) in z; 29 is "
        "used because it is the thickness the authors state they cut, and 30 is kept "
        "here as 'file_header_zyx_nm'. No source read explains the difference."
    ),
    "file_header": "filled at run time from the TIFF tags",
    "paper": {
        "ref": "Kasthuri et al. 2015, Cell 162(3):648-661, doi:10.1016/j.cell.2015.06.054",
        "quotes": [
            "we collected 2,250 29-nm coronal brain slices",
            "We chose 29 nm as section thickness in order to trace the finest neuronal wires",
            "We acquired high-resolution images with 3-nm pixels",
        ],
        "note": (
            "The paper gives the acquisition pixel as 3 nm. The challenge files have "
            "6 nm pixels, so they are at half that resolution in x and y; the paper "
            "does not mention the challenge stack, so the 2 x 2 reduction is an "
            "inference from the two numbers, not a quoted fact."
        ),
    },
    "project_data_page": {
        "ref": "https://neurodata.io/data/kasthuri15/ (read 2026-11-23)",
        "quote": ("acquired a data set from mouse cortex with a 3x3x30 cubic nanometer "
                  "spatial resolution"),
    },
    "challenge_description": (
        "The public challenge pages (snemi3d.grand-challenge.org, read 2026-11-23; "
        "archived copies of the 2013 site brainiac.mit.edu/SNEMI3D of 2013-02-25, "
        "2016-03-15 and 2021-07-23) and the Zenodo record give no voxel size. They "
        "say only that x and y have a high and z a low resolution."
    ),
    "secondary": {
        "ref": ("Lee, Zung, Li, Jain, Seung 2017, Superhuman Accuracy on the SNEMI3D "
                "Connectomics Challenge, arXiv:1706.00120, section 6.1"),
        "quote": ("The SNEMI3D challenge provides a single labeled image stack of size "
                  "1024 x 1024 x 100 for training and the same-sized image stack for "
                  "testing. The voxel resolution is 6 x 6 x 29 nm3"),
    },
}

# --------------------------------------------------------------------------- crop rule

CROP_SHAPE = (64, 512, 512)          # z, y, x
THIN_DIAMETER_NM = 200.0             # an object is "thin" below this in-plane diameter
THIN_MIN_VOXELS = 10000              # ... and counts only with this many voxels in the window
WINDOW_STRIDE_YX = 32                # windows are tried every 32 pixels in y and x
# Two neighbouring raw sections that correlate above this show the same tissue twice
# (measured on this stack: one pair at 0.92, all other pairs between 0.34 and 0.65).
REPEAT_CORR = 0.8
# The design (research/connectome/supervision.md, "Held out") holds out sections 80..99
# of SNEMI3D and leaves 75..79 unused. A 64-section window that starts at z <= 11 ends
# at section 74 at the latest, so it lies wholly in the training part.
WINDOW_MAX_Z0 = 11

USER_AGENT = "fetch_snemi3d.py (python-urllib; research data fetch)"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- download

def http_open(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_record(src_dir):
    """Read the Zenodo record and compare it with what this script was written for.

    Returns (url_read or None, facts). A network failure is not fatal: the zip is
    still checked against ZIP_MD5. A record that no longer matches is fatal.
    """
    try:
        with http_open(ZENODO_RECORD_API, timeout=60) as r:
            text = r.read().decode("utf-8")
    except Exception as e:                                   # noqa: BLE001
        log(f"WARNING: could not read {ZENODO_RECORD_API}: {e!r}; using the values in the script")
        return None, {"record_read": False}
    rec = json.loads(text)
    with open(os.path.join(src_dir, "zenodo_record_7142003.json"), "w") as f:
        f.write(text)
    files = {f["key"]: f for f in rec.get("files", [])}
    lic = (rec.get("metadata", {}).get("license") or {}).get("id")
    facts = {
        "record_read": True,
        "doi": rec.get("doi"),
        "title": rec.get("metadata", {}).get("title"),
        "licence_id": lic,
        "publication_date": rec.get("metadata", {}).get("publication_date"),
        "file_checksum": files.get(ZIP_NAME, {}).get("checksum"),
        "file_size": files.get(ZIP_NAME, {}).get("size"),
    }
    problems = []
    if facts["file_checksum"] != "md5:" + ZIP_MD5:
        problems.append(f"checksum is {facts['file_checksum']}, expected md5:{ZIP_MD5}")
    if facts["file_size"] != ZIP_BYTES:
        problems.append(f"size is {facts['file_size']}, expected {ZIP_BYTES}")
    if lic != "cc-by-4.0":
        problems.append(f"licence id is {lic!r}, expected 'cc-by-4.0'")
    if problems:
        raise SystemExit("The Zenodo record has changed since this script was written: "
                         + "; ".join(problems))
    log(f"Zenodo record read: doi {facts['doi']}, licence {lic}, {ZIP_NAME} {facts['file_checksum']}")
    return ZENODO_RECORD_API, facts


def ensure_zip(src_dir):
    """Return (path, info). Download snemi.zip unless a copy with the right md5 is cached."""
    path = os.path.join(src_dir, ZIP_NAME)
    side = path + ".download.json"
    if os.path.exists(path):
        got = md5_of_file(path)
        if got == ZIP_MD5:
            if os.path.exists(side):
                with open(side) as f:
                    info = json.load(f)
            else:   # a copy put there by hand: date it by its modification time
                day = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
                info = {"url": ZIP_URLS[0], "downloaded": day, "bytes": os.path.getsize(path),
                        "md5": got, "note": "sidecar missing; date taken from the file time"}
                with open(side, "w") as f:
                    json.dump(info, f, indent=1)
            log(f"cached {path} has the expected md5; no download")
            return path, info
        log(f"cached {path} has md5 {got}, expected {ZIP_MD5}; downloading again")
        os.remove(path)
    part = path + ".part"
    last_error = None
    for url in ZIP_URLS:
        try:
            log(f"downloading {url}")
            h = hashlib.md5()
            n = 0
            next_report = 0
            with http_open(url, timeout=120) as r, open(part, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    h.update(block)
                    n += len(block)
                    if n >= next_report:
                        log(f"  {n / 1e6:7.1f} MB of {ZIP_BYTES / 1e6:.1f}")
                        next_report += 25 << 20
            if n != ZIP_BYTES or h.hexdigest() != ZIP_MD5:
                raise IOError(f"got {n} bytes, md5 {h.hexdigest()}; expected {ZIP_BYTES}, {ZIP_MD5}")
            os.replace(part, path)
            info = {"url": url, "downloaded": datetime.date.today().isoformat(),
                    "bytes": n, "md5": h.hexdigest()}
            with open(side, "w") as f:
                json.dump(info, f, indent=1)
            log(f"downloaded {n} bytes, md5 matches the Zenodo record")
            return path, info
        except Exception as e:                               # noqa: BLE001
            last_error = e
            log(f"WARNING: {url} failed: {e!r}")
            if os.path.exists(part):
                os.remove(part)
    raise SystemExit(f"could not download {ZIP_NAME}: {last_error!r}")


# --------------------------------------------------------------------------- read

def read_tiff_member(zip_path, member):
    """Return (array, header facts) of one TIFF inside the zip, read in memory."""
    import tifffile
    with zipfile.ZipFile(zip_path) as z:
        data = z.read(member)
    facts = {"member": member, "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}
    with tifffile.TiffFile(io.BytesIO(data)) as tf:
        series = tf.series[0]
        page = tf.pages[0]
        ij = dict(tf.imagej_metadata or {})
        xres = page.tags["XResolution"].value
        yres = page.tags["YResolution"].value
        facts.update({
            "shape": [int(s) for s in series.shape],
            "axes": series.axes,
            "dtype": str(series.dtype),
            "pages": len(tf.pages),
            "imagej_version": ij.get("ImageJ"),
            "unit": ij.get("unit"),
            "spacing": ij.get("spacing"),
            "XResolution": [int(xres[0]), int(xres[1])],
            "YResolution": [int(yres[0]), int(yres[1])],
            "pixel_nm_yx": [yres[1] / yres[0], xres[1] / xres[0]],
        })
        info = ij.get("Info") or ""
        stale = {}
        for key in ("Slice thickness", "Voxel height", "Voxel width", "Location"):
            m = re.search(r"^\s*" + re.escape(key) + r" = (.*)$", info, flags=re.M)
            if m:
                stale[key] = m.group(1).strip()
        if stale:
            facts["embedded_nifti_info"] = stale
        arr = tf.asarray()
    return arr, facts


def check_source(raw, seg, hraw, hseg):
    """Stop if the files are not what this script and its notes describe."""
    def need(cond, msg):
        if not cond:
            raise SystemExit("unexpected source data: " + msg)
    need(raw.shape == EXPECT_SHAPE and raw.dtype == np.uint8, f"raw is {raw.shape} {raw.dtype}")
    need(seg.shape == EXPECT_SHAPE and seg.dtype == np.uint16, f"labels are {seg.shape} {seg.dtype}")
    need(hraw["axes"] == "ZYX" and hseg["axes"] == "ZYX", "TIFF axes are not ZYX")
    for h in (hraw, hseg):
        need(h["unit"] == "nm", f"{h['member']}: unit is {h['unit']!r}, not nm")
        need(abs(h["pixel_nm_yx"][0] - 6.0) < 0.01 and abs(h["pixel_nm_yx"][1] - 6.0) < 0.01,
             f"{h['member']}: pixel is {h['pixel_nm_yx']} nm, not 6")
        need(h["spacing"] == 30.0, f"{h['member']}: spacing is {h['spacing']!r}, not 30.0")
    ids = np.unique(seg)
    need(ids[0] == 0 and len(ids) == EXPECT_N_IDS + 1 and int(ids[-1]) == EXPECT_N_IDS,
         f"label ids are not 0..{EXPECT_N_IDS} without gaps ({len(ids)} values, max {ids[-1]})")


# --------------------------------------------------------------------------- measures

def object_stats(seg, voxel_nm, n_ids):
    """Per-id measures of a [z, y, x] label array (ids < n_ids, 0 = no label).

    inplane_diameter_nm: in every section in which the id occurs, take the largest
    in-plane Euclidean distance from a pixel of the id to the nearest pixel that has
    another id or 0 (pixel centre to pixel centre, the volume face not counted as a
    border); twice that minus one pixel is the width of the widest place of the
    profile in that section. The median over the sections is the diameter. It is a
    measure of calibre that does not depend on the length of the object.
    """
    import cc3d
    import edt
    from scipy import ndimage

    Z, Y, X = seg.shape
    vz, vy, vx = voxel_nm
    area = np.zeros((Z, n_ids), np.int64)
    rad = np.zeros((Z, n_ids), np.float32)
    ysum = np.zeros(n_ids, np.float64)
    xsum = np.zeros(n_ids, np.float64)
    yy, xx = np.meshgrid(np.arange(Y, dtype=np.float64), np.arange(X, dtype=np.float64),
                         indexing="ij")
    yyr, xxr = yy.ravel(), xx.ravel()
    for z in range(Z):
        lab = np.ascontiguousarray(seg[z])
        flat = lab.ravel()
        dist = edt.edt(lab, anisotropy=(vy, vx), black_border=False)
        r = np.zeros(n_ids, np.float32)
        np.maximum.at(r, flat, dist.ravel())
        rad[z] = r
        area[z] = np.bincount(flat, minlength=n_ids)
        ysum += np.bincount(flat, weights=yyr, minlength=n_ids)
        xsum += np.bincount(flat, weights=xxr, minlength=n_ids)
    vox = area.sum(0)
    present = area > 0
    n_sections = present.sum(0)
    zsum = (area * np.arange(Z, dtype=np.float64)[:, None]).sum(0)
    diameter = np.full(n_ids, np.nan)
    for i in range(1, n_ids):
        if n_sections[i]:
            diameter[i] = float(np.median(2.0 * rad[present[:, i], i] - min(vy, vx)))
    cc, n_cc = cc3d.connected_components(seg, connectivity=26, return_N=True)
    owner = np.zeros(n_cc + 1, np.int64)
    owner[cc.ravel()] = seg.ravel()
    n_comp = np.bincount(owner[1:], minlength=n_ids)
    del cc
    boxes = ndimage.find_objects(seg, max_label=n_ids - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid_vox = np.stack([zsum / vox, ysum / vox, xsum / vox], 1)
    return {"voxels": vox, "n_sections": n_sections, "diameter_nm": diameter,
            "n_components_26": n_comp, "boxes": boxes, "centroid_vox": centroid_vox}


def objects_table(stats, voxel_nm, thin_nm):
    v = np.asarray(voxel_nm, np.float64)
    rows = []
    for i in range(1, len(stats["voxels"])):
        n = int(stats["voxels"][i])
        if n == 0:
            continue
        sl = stats["boxes"][i - 1]
        lo = [int(s.start) for s in sl]
        hi = [int(s.stop) for s in sl]
        cen = (stats["centroid_vox"][i] + 0.5) * v
        d = float(stats["diameter_nm"][i])
        rows.append({
            "id": i,
            "n_voxels": n,
            "volume_um3": round(n * float(v.prod()) * 1e-9, 6),
            "n_sections": int(stats["n_sections"][i]),
            "bbox_vox": lo + hi,
            "bbox_nm": [round(float(a * b), 1) for a, b in zip(lo, v)]
                       + [round(float(a * b), 1) for a, b in zip(hi, v)],
            "centroid_nm": [round(float(c), 1) for c in cen],
            "inplane_diameter_nm": round(d, 1),
            "thin": bool(d < thin_nm),
            "n_components_26": int(stats["n_components_26"][i]),
        })
    return rows


def border_check(raw, seg):
    """Do the labels sit on the image?  Membranes are dark here and are left unlabelled.

    Mean grey value of raw under the voxels with label 0, for the labels as stored and
    for the labels moved by one or two pixels in y or x or sections in z (np.roll, so
    the move wraps at the faces). If raw and seg are registered, the stored position is
    the darkest. 'darkest_shift_*_px' is the minimum of a parabola through the values
    at -1, 0, +1.
    """
    r = raw.astype(np.float32)
    m = seg == 0
    out = {
        "what": ("mean grey value of raw under label 0, with seg as stored (shift 0) and "
                 "shifted by whole voxels; the lowest value marks where the unlabelled "
                 "borders lie on the dark membranes"),
        "mean_raw_unlabelled": round(float(r[m].mean()), 2),
        "mean_raw_labelled": round(float(r[~m].mean()), 2),
    }
    for axis, ax in ((0, "z"), (1, "y"), (2, "x")):
        vals = {s: float(r[np.roll(m, s, axis=axis)].mean()) for s in (-2, -1, 0, 1, 2)}
        a, b, c = vals[-1], vals[0], vals[1]
        out["shift_" + ax] = {f"{s:+d}": round(v, 2) for s, v in vals.items()}
        out["darkest_shift_" + ax + "_px"] = round(0.5 * (a - c) / (a - 2.0 * b + c), 2)
    return out


def _corr(a, b):
    """Pearson correlation of two arrays of the same shape."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _border_mask(lab):
    """Label 0, or a pixel whose label differs from one of its four in-plane neighbours."""
    m = lab == 0
    d = lab[:, 1:] != lab[:, :-1]
    m[:, 1:] |= d
    m[:, :-1] |= d
    d = lab[1:, :] != lab[:-1, :]
    m[1:, :] |= d
    m[:-1, :] |= d
    return m


def repeated_sections(raw, seg, z0):
    """Neighbouring raw sections that show the same tissue, and which labels fit them.

    corr_next[z] is the Pearson correlation of raw section z with raw section z + 1
    (no shift). Two different sections of this stack correlate at about 0.4 to 0.65;
    a pair above REPEAT_CORR is reported as a repeat. label_fit(l, r) is the
    correlation between the darkness of raw section r and the mask 'label 0 or
    in-plane label edge' of seg section l: the higher, the better the label borders
    of l lie on the dark membranes of r. z0 is the source section of the first
    section, for the wording only.
    """
    Z = raw.shape[0]
    corr_next = [_corr(raw[z], raw[z + 1]) for z in range(Z - 1)]

    def fit(zl, zr):
        return -_corr(raw[zr], _border_mask(seg[zl]))

    own = [fit(z, z) for z in range(Z)]
    pairs = []
    for a in range(Z - 1):
        if corr_next[a] <= REPEAT_CORR:
            continue
        b = a + 1
        others = [c for i, c in enumerate(corr_next) if i != a]
        f = {"labels_a_on_raw_a": own[a], "labels_b_on_raw_b": own[b],
             "labels_b_on_raw_a": fit(b, a), "labels_a_on_raw_b": fit(a, b)}
        if f["labels_b_on_raw_a"] > f["labels_a_on_raw_a"] and f["labels_a_on_raw_b"] <= f["labels_b_on_raw_b"]:
            odd, like = a, b
        elif f["labels_a_on_raw_b"] > f["labels_b_on_raw_b"] and f["labels_b_on_raw_a"] <= f["labels_a_on_raw_a"]:
            odd, like = b, a
        else:
            odd, like = None, None
        # the whole-pixel shift (dy, dx), within 3 pixels, at which raw[a][y, x] and
        # raw[b][y + dy, x + dx] correlate best
        R = 3
        Y, X = raw.shape[1:]
        best = (-2.0, 0, 0)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                c = _corr(raw[a, R:Y - R, R:X - R], raw[b, R + dy:Y - R + dy, R + dx:X - R + dx])
                if c > best[0]:
                    best = (c, dy, dx)
        pairs.append({
            "z": [a, b],
            "source_sections": [a + z0, b + z0],
            "corr": round(corr_next[a], 3),
            "corr_at_best_shift": round(best[0], 3),
            "best_shift_yx_px": [best[1], best[2]],
            "identical_pixels_fraction": round(float((raw[a] == raw[b]).mean()), 4),
            "mean_raw": [round(float(raw[a].mean()), 2), round(float(raw[b].mean()), 2)],
            "corr_other_pairs_median": round(float(np.median(others)), 3),
            "corr_other_pairs_max": round(float(max(others)), 3),
            "label_fit": {k: round(v, 3) for k, v in f.items()},
            "label_fit_own_median_all_sections": round(float(np.median(own)), 3),
            "label_fit_own_rank_of_a": int(sorted(own).index(own[a]) + 1),
            "label_fit_own_rank_of_b": int(sorted(own).index(own[b]) + 1),
            "raw_section_that_fits_the_other_labels_better": odd,
            "it_looks_like_an_image_of_section": like,
        })
    return {
        "what": ("corr_next[z]: Pearson correlation of raw section z with raw section z + 1. "
                 f"pairs: neighbours above {REPEAT_CORR}, which show the same tissue twice. "
                 "best_shift_yx_px = (dy, dx): raw[a][y, x] and raw[b][y + dy, x + dx] correlate "
                 "best (searched within 3 pixels). "
                 "label_fit: correlation between the darkness of a raw section and the mask "
                 "'label 0 or in-plane label edge' of a seg section (a, b = the two sections "
                 "of the pair); label_fit_own[z] is that of every section with its own labels; "
                 "rank 1 = the lowest of all sections."),
        "threshold": REPEAT_CORR,
        "corr_next": [round(c, 3) for c in corr_next],
        "label_fit_own": [round(v, 3) for v in own],
        "pairs": pairs,
    }


def repeated_notes(rep, n_sections):
    """One note per repeated pair, in words, from the numbers of repeated_sections()."""
    notes = []
    for p in rep["pairs"]:
        a, b = p["z"]
        sa, sb = p["source_sections"]
        f = p["label_fit"]
        text = (f"Raw sections {a} and {b} of this crop (source sections {sa} and {sb}) show the "
                f"same tissue twice: their correlation is {p['corr']}, against a median of "
                f"{p['corr_other_pairs_median']} and a maximum of {p['corr_other_pairs_max']} for "
                "all other pairs of neighbouring sections (raw_repeated_sections). The two images "
                f"are not the same bytes: {p['identical_pixels_fraction'] * 100:.1f} % of the pixels "
                f"are equal, the mean grey values are {p['mean_raw'][0]} and {p['mean_raw'][1]}, and "
                f"the correlation is highest ({p['corr_at_best_shift']}) with section {b} moved by "
                f"(y, x) = ({-p['best_shift_yx_px'][0]:+d}, {-p['best_shift_yx_px'][1]:+d}) pixels. ")
        odd, like = p["raw_section_that_fits_the_other_labels_better"], p["it_looks_like_an_image_of_section"]
        if odd is not None:
            own_odd = f["labels_a_on_raw_a"] if odd == a else f["labels_b_on_raw_b"]
            cross = f["labels_b_on_raw_a"] if odd == a else f["labels_a_on_raw_b"]
            own_like = f["labels_b_on_raw_b"] if odd == a else f["labels_a_on_raw_a"]
            odd_on_like = f["labels_a_on_raw_b"] if odd == a else f["labels_b_on_raw_a"]
            rank = p["label_fit_own_rank_of_a"] if odd == a else p["label_fit_own_rank_of_b"]
            text += (f"The labels of section {like} fit the image stored as section {odd} "
                     f"(label_fit {cross}) about as well as their own image ({own_like}). The "
                     f"labels of section {odd} fit their own image with {own_odd} (rank {rank} "
                     f"from the bottom among the {n_sections} sections, whose median is "
                     f"{p['label_fit_own_median_all_sections']}) and the image of section {like} "
                     f"with {odd_on_like}. So raw section {odd} looks like a second image of "
                     f"section {like}, and the labels of section {odd} seem to describe tissue "
                     "whose image is not in the stack; the source does not say what happened. "
                     f"Comparing a rendering with raw at section {odd}, or holding that section "
                     "out, is affected by this.")
        else:
            text += ("Which of the two is the repeat could not be told from the fit of the "
                     "labels (label_fit in raw_repeated_sections).")
        notes.append(text)
    return notes


def choose_window(seg, thin, n_ids):
    """Find the CROP_SHAPE window with the most thin objects.

    Score of a window: the number of ids that are thin (in-plane diameter, measured
    on the whole stack, below THIN_DIAMETER_NM) and have at least THIN_MIN_VOXELS
    voxels inside the window. Ties: more voxels of such objects, then the smallest
    offset. Windows: every z0 in 0..WINDOW_MAX_Z0, y0 and x0 every WINDOW_STRIDE_YX.
    """
    Z, Y, X = seg.shape
    dz, dy, dx = CROP_SHAPE
    B = WINDOW_STRIDE_YX
    nby, nbx = Y // B, X // B
    block = (np.arange(Y) // B)[:, None] * nbx + (np.arange(X) // B)[None, :]
    key0 = (block * n_ids).astype(np.int64)
    integral = np.zeros((Z + 1, nby + 1, nbx + 1, n_ids), np.int64)
    for z in range(Z):
        h = np.bincount((key0 + seg[z]).ravel(), minlength=nby * nbx * n_ids)
        integral[z + 1, 1:, 1:] = h.reshape(nby, nbx, n_ids)
    integral = integral.cumsum(0).cumsum(1).cumsum(2)

    def counts(z0, y0, x0):
        a, b, c = z0, y0 // B, x0 // B
        d, e, f = z0 + dz, (y0 + dy) // B, (x0 + dx) // B
        I = integral
        return (I[d, e, f] - I[a, e, f] - I[d, b, f] - I[d, e, c]
                + I[a, b, f] + I[a, e, c] + I[d, b, c] - I[a, b, c])

    results = []
    for z0 in range(0, min(WINDOW_MAX_Z0, Z - dz) + 1):
        for y0 in range(0, Y - dy + 1, B):
            for x0 in range(0, X - dx + 1, B):
                c = counts(z0, y0, x0)
                sel = thin & (c >= THIN_MIN_VOXELS)
                results.append((int(sel.sum()), int(c[sel].sum()), z0, y0, x0,
                                int((c[1:] > 0).sum()), float(c[0]) / float(c.sum())))
    results.sort(key=lambda r: (-r[0], -r[1], r[2], r[3], r[4]))
    best = results[0]
    # the integral image must agree with a direct count
    direct = np.bincount(seg[best[2]:best[2] + dz, best[3]:best[3] + dy,
                             best[4]:best[4] + dx].ravel(), minlength=n_ids)
    assert np.array_equal(direct, counts(best[2], best[3], best[4])), "window count check failed"
    scores = np.array([r[0] for r in results])
    report = {
        "rule": (
            f"Window of {dz} x {dy} x {dx} voxels with the most thin neurites. An object is "
            f"thin if its in-plane diameter, measured on the whole training stack (see "
            f"objects.json of snemi3d_full for the definition), is below {THIN_DIAMETER_NM:g} nm; "
            f"it counts for a window if at least {THIN_MIN_VOXELS} of its voxels lie inside. "
            f"Ties go to the window with more voxels of such objects, then to the smallest "
            f"offset. Windows tried: z0 = 0..{WINDOW_MAX_Z0} (so that the window ends at "
            f"section 74 at the latest and stays out of the sections the design holds out, "
            f"80..99, and leaves unused, 75..79), y0 and x0 every {B} pixels."
        ),
        "thin_diameter_nm": THIN_DIAMETER_NM,
        "thin_min_voxels": THIN_MIN_VOXELS,
        "windows_tried": len(results),
        "score_chosen": best[0],
        "score_min": int(scores.min()),
        "score_median": float(np.median(scores)),
        "score_mean": round(float(scores.mean()), 2),
        "score_max": int(scores.max()),
        "top5": [{"offset_vox": [r[2], r[3], r[4]], "thin_objects": r[0], "thin_voxels": r[1],
                  "ids_present": r[5], "fraction_unlabelled": round(r[6], 4)} for r in results[:5]],
        "thin_ids_in_whole_stack": int(thin.sum()),
    }
    return (best[2], best[3], best[4]), report


# --------------------------------------------------------------------------- previews

def colour_table(n_ids):
    lut = np.zeros((n_ids, 3), np.float32)
    for i in range(1, n_ids):
        hue = (i * 0.618033988749895) % 1.0
        sat = 0.6 + 0.4 * ((i * 0.381966011250105) % 1.0)
        lut[i] = colorsys.hsv_to_rgb(hue, sat, 1.0)
    return lut * 255.0


def edges(lab):
    """Pixels whose right or lower neighbour has another label."""
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return b


def edges_both(lab):
    """Both pixels at every change of label: lines two pixels wide."""
    b = np.zeros(lab.shape, bool)
    t = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= t
    b[:, 1:] |= t
    t = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= t
    b[1:, :] |= t
    return b


def paint(raw2d, lab2d, lut, fill, line, wide=False):
    rgb = np.repeat(raw2d.astype(np.float32)[..., None], 3, axis=2)
    if fill:
        m = lab2d > 0
        rgb[m] = 0.55 * rgb[m] + 0.45 * lut[lab2d[m]]
    if line is not None:
        rgb[edges_both(lab2d) if wide else edges(lab2d)] = line
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def grey(raw2d):
    return np.repeat(raw2d[..., None], 3, axis=2)


def resample(a, rows, cols):
    """Nearest-neighbour resampling of a 2D array to rows x cols."""
    r = np.minimum((np.arange(rows) * a.shape[0] / rows).astype(np.int64), a.shape[0] - 1)
    c = np.minimum((np.arange(cols) * a.shape[1] / cols).astype(np.int64), a.shape[1] - 1)
    return a[r][:, c]


def save_panels(path, panels, captions):
    from PIL import Image, ImageDraw, ImageFont
    gap, head = 12, 30
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h + head), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    x = 0
    for p, cap in zip(panels, captions):
        canvas.paste(Image.fromarray(p), (x, head))
        draw.text((x + 4, 5), cap, fill=(0, 0, 0), font=font)
        x += p.shape[1] + gap
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        canvas.save(f, format="PNG")
    os.replace(tmp, path)


def write_previews(out_dir, name, raw, seg, offset, voxel_nm, lut):
    Z, Y, X = raw.shape
    zm = Z // 2
    mag = max(1, 1024 // max(Y, X))
    yellow = np.array([255.0, 255.0, 0.0], np.float32)
    white = np.array([255.0, 255.0, 255.0], np.float32)

    # preview.png: the middle section
    r = resample(raw[zm], Y * mag, X * mag)
    l = resample(seg[zm], Y * mag, X * mag)
    save_panels(os.path.join(out_dir, "preview.png"),
                [grey(r), paint(r, l, lut, True, white)],
                [f"{name}  raw  z={zm} (source section {zm + offset[0]})  {Y}x{X} px, shown x{mag}",
                 "seg: one colour per id, white lines = label boundaries, grey = 0 (no label)"])

    # preview_zoom.png: the 128 x 128 pixels of the middle section with the most label
    # boundaries (tried every 32 pixels), enlarged 8 times
    side, zmag, step = 128, 8, 32
    e = edges(seg[zm]).astype(np.int64)
    best = (-1, 0, 0)
    for yy in range(0, Y - side + 1, step):
        for xx in range(0, X - side + 1, step):
            n = int(e[yy:yy + side, xx:xx + side].sum())
            if n > best[0]:
                best = (n, yy, xx)
    _, y0, x0 = best
    r = resample(raw[zm, y0:y0 + side, x0:x0 + side], side * zmag, side * zmag)
    l = resample(seg[zm, y0:y0 + side, x0:x0 + side], side * zmag, side * zmag)
    save_panels(os.path.join(out_dir, "preview_zoom.png"),
                [grey(r), paint(r, l, lut, False, yellow, wide=True)],
                [f"{name}  raw  z={zm}, y={y0}..{y0 + side}, x={x0}..{x0 + side}, shown x{zmag}",
                 "yellow lines = label boundaries (no fill); they should run along the dark membranes"])

    # preview_zx.png: z against x at the middle y, at the physical aspect ratio
    ym = Y // 2
    rows = int(round(Z * voxel_nm[0] / voxel_nm[2] * mag))
    r = resample(raw[:, ym, :], rows, X * mag)
    l = resample(seg[:, ym, :], rows, X * mag)
    save_panels(os.path.join(out_dir, "preview_zx.png"),
                [grey(r), paint(r, l, lut, True, None)],
                [f"{name}  raw  z (down, {voxel_nm[0]:g} nm per section) against x at y={ym}",
                 "seg in colour; the same object should keep its colour from section to section"])


# --------------------------------------------------------------------------- write

def save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def write_crop(out_dir, name, raw, seg, offset, common, extra_notes, selection, lut):
    os.makedirs(out_dir, exist_ok=True)
    Z, Y, X = raw.shape
    n_ids = EXPECT_N_IDS + 1
    log(f"{name}: measuring objects in {raw.shape}")
    stats = object_stats(seg, VOXEL_NM, n_ids)
    rows = objects_table(stats, VOXEL_NM, THIN_DIAMETER_NM)
    ids = [r["id"] for r in rows]
    n_multi = sum(1 for r in rows if r["n_components_26"] > 1)
    section_mean = raw.reshape(Z, -1).mean(1)
    frac0 = float((seg == 0).mean())
    bc = border_check(raw, seg)
    registered = (abs(bc["darkest_shift_y_px"]) < 1.0 and abs(bc["darkest_shift_x_px"]) < 1.0
                  and bc["shift_z"]["+0"] < min(bc["shift_z"]["-1"], bc["shift_z"]["+1"])
                  and bc["mean_raw_unlabelled"] < bc["mean_raw_labelled"])
    if not registered:
        raise SystemExit(f"{name}: raw and seg do not look registered, nothing written: {bc}")
    rep = repeated_sections(raw, seg, int(offset[0]))
    for p in rep["pairs"]:
        log(f"{name}: raw sections {p['z']} (source {p['source_sections']}) repeat the same tissue, "
            f"correlation {p['corr']}; label_fit {p['label_fit']}")

    save_npy(os.path.join(out_dir, "raw.npy"), raw.astype(np.uint8, copy=False))
    save_npy(os.path.join(out_dir, "seg.npy"), seg.astype(np.uint64))

    save_json(os.path.join(out_dir, "objects.json"), {
        "what": ("One row per label id present in seg.npy. Derived by fetch_snemi3d.py from "
                 "the labels; the source has no such table."),
        "axis_order": "z, y, x everywhere",
        "units": ("*_nm in nanometres from the crop origin, which is the outer corner of voxel "
                  "(0, 0, 0); the centre of voxel (k, j, i) is at ((k+0.5)*vz, (j+0.5)*vy, "
                  "(i+0.5)*vx). *_vox in voxels; bbox_vox is [z0, y0, x0, z1, y1, x1] with "
                  "z1, y1, x1 exclusive."),
        "voxel_nm": list(VOXEL_NM),
        "inplane_diameter_nm": (
            "In every section in which the id occurs: the largest in-plane Euclidean distance "
            "from a pixel of the id to the nearest pixel with another id or 0 (centre to "
            "centre; the face of the crop is not a border), times two, minus one pixel: the "
            "width of the profile at its widest place. The value is the median over those "
            "sections, measured inside this crop."),
        "thin": f"inplane_diameter_nm < {THIN_DIAMETER_NM:g}",
        "n_components_26": "number of 26-connected pieces of the id inside this crop",
        "objects": rows,
    })

    notes = [
        ("Voxel size: z = 29 nm is the section thickness stated by Kasthuri et al. 2015; the "
         "TIFF headers of the challenge files say spacing = 30.0 nm. x = y = 6 nm from the TIFF "
         "headers (0.166666 pixels per nm). See voxel_nm_evidence."),
        ("Axis order [z, y, x]: z is the TIFF page (section) index, y the image row, x the image "
         "column, as in the source. The source does not say how the stack lies in the tissue."),
        ("seg ids are the ids of the source file (1..400 in the whole stack), not renumbered, "
         "so the same neurite has the same id in snemi3d and snemi3d_full. "
         f"{len(ids)} ids are present here."),
        (f"0 in seg means no label: {frac0 * 100:.2f} % of the voxels here. These are the gaps "
         "the annotator left between objects (membranes, extracellular space) and unlabelled "
         "areas. A challenge organiser wrote on the challenge forum (2013-02-20) that, to train "
         "the optional membrane-probability maps, membrane labels were set up from 'all pixels "
         "with label 0' and 'all pixels at the edge of any nonzero object', and spoke of 'large "
         "areas with label 0'."),
        (f"{n_multi} of the ids present have more than one 26-connected piece inside this crop "
         "(see n_components_26 in objects.json). An id is one cell process, not always one "
         "connected component."),
        ("Registration of seg to raw, measured by this script (border_check): the mean grey value "
         f"of raw is {bc['mean_raw_unlabelled']} under label 0 and {bc['mean_raw_labelled']} under "
         "the labels; it rises when seg is shifted by one section in z; in the plane the darkest "
         f"position is {bc['darkest_shift_y_px']:+.2f} pixel in y and "
         f"{bc['darkest_shift_x_px']:+.2f} pixel in x from the stored labels. Label borders lie "
         "on the dark membranes to within one pixel."),
        ("In the previews, what is inside a process (mitochondria, vesicles) carries the id of "
         "the process, and in preview.png of snemi3d_full (section 50) the myelin sheath around "
         "an axon is unlabelled. Both were seen in the images, not counted."),
        ("raw is the source bytes, not normalised. The mean grey value changes from section to "
         f"section: between {section_mean.min():.1f} and {section_mean.max():.1f} here "
         "(raw_section_mean lists it per section). In preview_zx.png this shows as bright and "
         "dark rows."),
        ("The authors aligned the sections with 'a single affine transformation per image' "
         "(Kasthuri et al. 2015, Experimental Procedures). How well neighbouring sections "
         "register in this stack was not measured here."),
        ("The labels TIFF carries the header of an earlier NIfTI file in its ImageJ 'Info' text "
         "('Slice thickness = 25.0', 'Voxel width = 4.8...'). Its affine entries are denormal "
         "numbers, so it was not trusted and is not used; it is recorded in source_files."),
        ("The test stack of the challenge (image/test-input.tif) has no public labels and is "
         "not written."),
    ] + repeated_notes(rep, Z) + extra_notes

    meta = {
        "name": name,
        "dataset": DATASET,
        "voxel_nm": list(VOXEL_NM),
        "shape": [int(Z), int(Y), int(X)],
        "offset_vox": [int(o) for o in offset],
        "level": ("0, the only level published: the challenge TIFF stacks as distributed, "
                  "voxel 29 x 6 x 6 nm (z, y, x). The microscope images were taken at 3 nm per "
                  "pixel (Kasthuri et al. 2015), so x and y appear to be at half the acquisition "
                  "resolution (an inference from the two numbers; no source read says so)."),
        "sources": common["sources"],
        "seg_kind": "manual",
        "seg_meaning": ("One id per cell process (the challenge calls them neurites), traced by "
                        "hand through the sections ('manually annotated by Daniel R. Berger', "
                        "challenge page); the source does not say which ids are axons, dendrites "
                        "or glia; 0 = no label."),
        "class_names": {},
        "licence": LICENCE,
        "citation": CITATION,
        "fetched": common["fetched"],
        "notes": notes,
        "axis_order": "zyx",
        "dtypes": {"raw.npy": "uint8", "seg.npy": "uint64 (uint16 in the source)"},
        "n_objects": len(ids),
        "id_min": int(min(ids)),
        "id_max": int(max(ids)),
        "fraction_unlabelled": round(frac0, 6),
        "n_ids_with_several_components_26": n_multi,
        "n_thin_objects": sum(1 for r in rows if r["thin"]),
        "raw_section_mean": [round(float(m), 2) for m in section_mean],
        "border_check": bc,
        "raw_repeated_sections": rep,
        "voxel_nm_evidence": common["voxel_evidence"],
        "source_files": common["source_files"],
        "zenodo_record": common["record"],
        "documentation_read": common["documentation"],
        "tables": {"objects.json": "per-id measures; axis order z, y, x; nanometres from the crop origin"},
        "extra_files": ["objects.json", "preview_zoom.png", "preview_zx.png"],
        "written_by": "research/connectome/render/data/fetch_snemi3d.py",
    }
    if selection is not None:
        meta["selection"] = selection
    save_json(os.path.join(out_dir, "meta.json"), meta)
    write_previews(out_dir, name, raw, seg, offset, VOXEL_NM, lut)
    log(f"{name}: wrote {out_dir}  shape {list(raw.shape)}  offset {list(offset)}  "
        f"{len(ids)} ids  unlabelled {frac0 * 100:.2f} %")
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True,
                    help="folder of the crop 'snemi3d'; 'snemi3d_full' is written beside it as <out>_full")
    ap.add_argument("--src", default=os.path.join("~", "connectome", "data", "src", "snemi3d"),
                    help="cache for the download (default ~/connectome/data/src/snemi3d)")
    ap.add_argument("--offset", default=None,
                    help="z,y,x of the 64 x 512 x 512 crop, to override the rule (default: chosen by the rule)")
    args = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out)).rstrip("/")
    out_full = out + "_full"
    src = os.path.abspath(os.path.expanduser(args.src))
    os.makedirs(src, exist_ok=True)

    record_url, record = check_record(src)
    zip_path, dl = ensure_zip(src)
    log("reading the TIFF stacks from the zip")
    raw, hraw = read_tiff_member(zip_path, MEMBER_RAW)
    seg, hseg = read_tiff_member(zip_path, MEMBER_SEG)
    check_source(raw, seg, hraw, hseg)
    log(f"raw {raw.shape} {raw.dtype}; labels {seg.shape} {seg.dtype}, ids 1..{int(seg.max())}; "
        f"header: pixel {hraw['pixel_nm_yx'][0]:.5f} nm, spacing {hraw['spacing']} {hraw['unit']}")

    evidence = dict(VOXEL_EVIDENCE)
    evidence["file_header"] = {
        "file_header_zyx_nm": [hraw["spacing"], round(hraw["pixel_nm_yx"][0], 5),
                               round(hraw["pixel_nm_yx"][1], 5)],
        "tags": {"ImageDescription": f"unit={hraw['unit']}, spacing={hraw['spacing']}",
                 "XResolution": hraw["XResolution"], "YResolution": hraw["YResolution"]},
        "same_in": [MEMBER_RAW, MEMBER_SEG],
    }
    common = {
        "sources": [u for u in (record_url, dl["url"]) if u],
        "fetched": dl["downloaded"],
        "voxel_evidence": evidence,
        "source_files": {"zip": {"name": ZIP_NAME, "bytes": dl["bytes"], "md5": dl["md5"],
                                 "md5_in_zenodo_record": ZIP_MD5},
                         "raw": hraw, "seg": hseg},
        "record": record,
        "documentation": [
            "https://snemi3d.grand-challenge.org/ (challenge page: who made the images and labels; names the Zenodo record)",
            "https://doi.org/10.5281/zenodo.7142003 (record: licence, checksum, creators)",
            "https://doi.org/10.1016/j.cell.2015.06.054 (Kasthuri et al. 2015: section thickness, pixel size, alignment, manual segmentation)",
            "https://arxiv.org/abs/1706.00120 (Lee et al. 2017: 6 x 6 x 29 nm3, 1024 x 1024 x 100)",
            "https://neurodata.io/data/kasthuri15/ (data page of the paper's project: 3x3x30 nm)",
            "http://web.archive.org/web/20130222052540/http://brainiac.mit.edu:80/SNEMI3D/content/training-and-test-data-sets (2013 forum post by an organiser on label 0)",
        ],
    }

    n_ids = EXPECT_N_IDS + 1
    lut = colour_table(n_ids)

    # the whole stack first: its per-object diameters decide which objects are thin
    meta_full, full_rows = write_crop(out_full, "snemi3d_full", raw, seg, (0, 0, 0), common, [
        ("This is the whole training stack. The design (research/connectome/supervision.md) "
         "holds out sections 80..99 and leaves 75..79 unused; they are included here."),
    ], None, lut)

    thin = np.zeros(n_ids, bool)
    for r in full_rows:
        thin[r["id"]] = r["thin"]

    log("scoring windows")
    offset, selection = choose_window(seg, thin, n_ids)
    if args.offset:
        offset = tuple(int(v) for v in args.offset.split(","))
        selection = dict(selection)
        selection["overridden_by_--offset"] = list(offset)
    z0, y0, x0 = offset
    dz, dy, dx = CROP_SHAPE
    if not (0 <= z0 <= raw.shape[0] - dz and 0 <= y0 <= raw.shape[1] - dy and 0 <= x0 <= raw.shape[2] - dx):
        raise SystemExit(f"offset {offset} does not fit a {CROP_SHAPE} window in {raw.shape}")
    log(f"window chosen: offset (z, y, x) = {offset}, {selection['score_chosen']} thin objects "
        f"(all windows: min {selection['score_min']}, median {selection['score_median']}, "
        f"max {selection['score_max']})")
    craw = np.ascontiguousarray(raw[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])
    cseg = np.ascontiguousarray(seg[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])
    write_crop(out, "snemi3d", craw, cseg, offset, common, [
        (f"Crop of snemi3d_full at offset (z, y, x) = ({z0}, {y0}, {x0}): source sections "
         f"{z0}..{z0 + dz - 1}, rows {y0}..{y0 + dy - 1}, columns {x0}..{x0 + dx - 1}. Chosen as "
         "the window with the most thin neurites; see selection."),
        ("The window lies in sections 0..74, the part the design uses for training; it does not "
         "touch the held-out sections 80..99."),
        ("Objects are cut by the faces of the crop. objects.json measures them inside the crop; "
         "the diameters that chose the window were measured on the whole stack."),
    ], selection, lut)

    log("done: " + out + " and " + out_full + f" ({meta_full['n_objects']} ids in the whole stack)")


if __name__ == "__main__":
    sys.exit(main())

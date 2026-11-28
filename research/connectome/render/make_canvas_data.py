"""Make the data files that research/connectome/canvas.html shows.

For a sub-block of each crop (see data/fetch_*.py) it writes, into --out:
  <name>_raw.jpg, <name>_fit.jpg   atlases of sections: the image, and the slab render of the fitted Gaussians
  <name>_seg.png                   atlas of the labels, one colour per id
  <name>.js                        window.CANVAS_DATA[<name>] = facts, scores, Gaussians, Frame, Graph

Voxel: the image and the labels. Frame: TEASAR skeletons of the labelled objects
(kimimaro), nodes with radius. Graph: objects and their contacts, from touching
labels. Gaussians: a regular grid fitted to the image with the prototype slab
renderer (slab_proto.py), one seed, scored on the sections it was fitted to.

Run on the Spark:  python make_canvas_data.py --crops ~/connectome/data/crops --out canvas_data
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from slab_proto import render, psnr

SETS = [  # name, title, z0, y0, x0, nz, ny, nx, in-plane spacing of the Gaussians in pixels
    ("microns", "MICrONS minnie65", 20, 128, 128, 24, 256, 256, 6),
    ("snemi3d", "SNEMI3D", 20, 128, 128, 24, 256, 256, 6),
    ("cremi3d", "CREMI3D", 20, 128, 128, 24, 256, 256, 8),
    ("cellmap_313", "CellMap crop 313", 88, 0, 0, 24, 200, 200, 4),
]
COLS = 6


def atlas(vol):  # [Z, Y, X] or [Z, Y, X, 3] uint8 -> one picture, COLS sections per row
    z, y, x = vol.shape[:3]
    rows = -(-z // COLS)
    out = np.zeros((rows * y, COLS * x) + vol.shape[3:], np.uint8)
    for i in range(z):
        out[(i // COLS) * y:(i // COLS + 1) * y, (i % COLS) * x:(i % COLS + 1) * x] = vol[i]
    return out


def colours(seg):
    ids = seg.astype(np.uint64)
    h = (ids * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
    rgb = np.stack([(h >> np.uint64(s)) & np.uint64(255) for s in (16, 8, 0)], -1).astype(np.uint8)
    rgb = (rgb // 2 + 90).astype(np.uint8)
    rgb[seg == 0] = 0
    return rgb


def fit(tgt, voxel, sp_px, iso, steps, dev):
    Z, Y, X = tgt.shape
    vz, vy, vx = voxel
    sp = sp_px * vy
    zsp = max(vz, sp)
    gz = torch.arange(zsp / 2, Z * vz, zsp)
    gy = torch.arange(sp / 2, Y * vy, sp)
    gx = torch.arange(sp / 2, X * vx, sp)
    mu = torch.stack(torch.meshgrid(gz, gy, gx, indexing="ij"), -1).reshape(-1, 3).to(dev)
    N = len(mu)
    log_s = torch.log(torch.tensor([zsp * 0.5, sp * 0.6, sp * 0.6], device=dev)).repeat(N, 1)
    q = torch.zeros(N, 4, device=dev); q[:, 0] = 1
    bg = float(tgt.mean())
    idx = (mu / torch.tensor(voxel, device=dev)).long()
    amp = (tgt[idx[:, 0], idx[:, 1], idx[:, 2]] - bg) * 0.5
    for p in (mu, log_s, q, amp):
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [mu], "lr": 0.05 * sp}, {"params": [log_s], "lr": 0.01},
                            {"params": [q], "lr": 0.005}, {"params": [amp], "lr": 0.01}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scales = lambda: log_s.mean(1, keepdim=True).expand(-1, 3) if iso else log_s
    torch.manual_seed(0)
    for _ in range(steps):
        zsel = torch.randperm(Z, device=dev)[:6].sort().values
        img = render(mu, scales(), q, amp, (Z, Y, X), voxel, zsel, bg=bg)
        loss = (img - tgt[zsel]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        full = render(mu, scales(), q, amp, (Z, Y, X), voxel, None, bg=bg).clamp(0, 1)
        from slab_proto import covariance
        cov = covariance(scales(), q)
    return full, mu.detach(), cov, amp.detach(), torch.exp(scales()).detach()


def frame(seg, voxel):
    import kimimaro
    vz, vy, vx = voxel
    lab = np.ascontiguousarray(seg.transpose(2, 1, 0))  # x, y, z
    sk = kimimaro.skeletonize(lab, teasar_params={"scale": 1.5, "const": 4 * vx}, anisotropy=(vx, vy, vz),
                              dust_threshold=800, parallel=4, progress=False, fix_branching=True)
    nodes, edges, n_branch, n_end, length = [], [], 0, 0, 0.0
    for i, s in sk.items():
        base = len(nodes)
        v = s.vertices
        deg = np.bincount(s.edges.reshape(-1), minlength=len(v))
        n_branch += int((deg >= 3).sum()); n_end += int((deg == 1).sum())
        length += float(np.linalg.norm(v[s.edges[:, 0]] - v[s.edges[:, 1]], axis=1).sum()) if len(s.edges) else 0
        for p, r in zip(v, s.radius):
            nodes.append([round(float(p[2]), 1), round(float(p[1]), 1), round(float(p[0]), 1), round(float(r), 1), int(i) % 1000003])
        edges += (s.edges + base).tolist()
    return nodes, edges, {"objects": len(sk), "nodes": len(nodes), "branch_points": n_branch, "ends": n_end,
                          "path_um": round(length / 1000, 2)}


def graph(seg, voxel, min_vox=1500, min_contact=40):
    ids, inv, cnt = np.unique(seg, return_inverse=True, return_counts=True)
    inv = inv.reshape(seg.shape)
    keep = (cnt >= min_vox) & (ids != 0)
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in seg.shape], indexing="ij")
    cen = np.stack([np.bincount(inv.ravel(), w.ravel(), len(ids)) / cnt for w in (zz, yy, xx)], 1) * np.array(voxel)
    pairs = {}
    for ax in range(3):
        a = np.moveaxis(inv, ax, 0)[:-1].ravel(); b = np.moveaxis(inv, ax, 0)[1:].ravel()
        m = a != b
        k, c = np.unique(np.minimum(a[m], b[m]).astype(np.int64) * len(ids) + np.maximum(a[m], b[m]), return_counts=True)
        for kk, cc in zip(k, c):
            pairs[int(kk)] = pairs.get(int(kk), 0) + int(cc)
    sel = np.nonzero(keep)[0]; new = {int(o): n for n, o in enumerate(sel)}
    nodes = [[round(float(c), 1) for c in cen[o]] + [int(cnt[o]), int(ids[o]) % 1000003] for o in sel]
    edges = [[new[k // len(ids)], new[k % len(ids)], c] for k, c in pairs.items()
             if c >= min_contact and k // len(ids) in new and k % len(ids) in new]
    return nodes, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, title, z0, y0, x0, nz, ny, nx, sp in SETS:
        t0 = time.time()
        d = os.path.join(os.path.expanduser(a.crops), name)
        meta = json.load(open(os.path.join(d, "meta.json")))
        voxel = [float(v) for v in meta["voxel_nm"]]
        box = (slice(z0, z0 + nz), slice(y0, y0 + ny), slice(x0, x0 + nx))
        raw = np.load(os.path.join(d, "raw.npy"), mmap_mode="r")[box]
        seg = np.ascontiguousarray(np.load(os.path.join(d, "seg.npy"), mmap_mode="r")[box])
        tgt = torch.tensor(np.ascontiguousarray(raw), dtype=torch.float32, device=dev) / 255
        full, mu, cov, amp, sc = fit(tgt, voxel, sp, False, a.steps, dev)
        full_iso = fit(tgt, voxel, sp, True, a.steps, dev)[0]
        lo = F.avg_pool2d(tgt[:, None], 4, ceil_mode=True)
        up = F.interpolate(lo, size=(ny, nx), mode="bicubic", align_corners=False).clamp(0, 1)[:, 0]
        s = sc.sort(dim=1, descending=True).values
        shape = torch.stack([(s[:, 0] - s[:, 1]) / s[:, 0], (s[:, 1] - s[:, 2]) / s[:, 0], s[:, 2] / s[:, 0]], 1)
        kind = shape.argmax(1)
        N = len(mu)
        Image.fromarray(atlas(np.ascontiguousarray(raw))).save(os.path.join(a.out, f"{name}_raw.jpg"), quality=85)
        Image.fromarray(atlas((full * 255).round().byte().cpu().numpy())).save(os.path.join(a.out, f"{name}_fit.jpg"), quality=85)
        Image.fromarray(atlas(colours(seg))).save(os.path.join(a.out, f"{name}_seg.png"), optimize=True)
        fn, fe, fs = frame(seg, voxel)
        gn, ge = graph(seg, voxel)
        c = cov.cpu().numpy()
        g = np.concatenate([mu.cpu().numpy(), c[:, [0, 1, 2], [0, 1, 2]], c[:, [0, 0, 1], [1, 2, 2]],
                            amp.cpu().numpy()[:, None], kind.cpu().numpy()[:, None]], 1)
        out = {
            "name": name, "title": title, "voxel_nm": voxel, "shape": [nz, ny, nx], "cols": COLS,
            "box_in_crop": [z0, y0, x0], "seg_kind": meta.get("seg_kind"), "level": meta.get("level"),
            "citation": meta.get("citation"), "licence": meta.get("licence"),
            "scores": {
                "gaussians": N, "voxels_per_gaussian": round(nz * ny * nx / N, 1),
                "bytes_per_voxel_float32": round(N * 11 * 4 / (nz * ny * nx), 3),
                "psnr_aniso": round(psnr(full, tgt), 2), "psnr_iso": round(psnr(full_iso, tgt), 2),
                "psnr_down4_bicubic": round(psnr(up, tgt), 2), "bytes_per_voxel_down4": 0.0625,
                "psnr_next_section": round(psnr(tgt[:-1], tgt[1:]), 2),
                "median_scales_nm": [round(float(v), 1) for v in s.median(0).values],
                "share_line_sheet_blob": [round(float((kind == k).float().mean()), 3) for k in range(3)],
                "steps": a.steps, "seeds": 1,
            },
            "frame_stats": fs, "graph_stats": {"objects": len(gn), "contacts": len(ge)},
            "gaussians": [[round(float(v), 3 if j == 9 else 1) for j, v in enumerate(r)] for r in g],  # z y x | czz cyy cxx | czy czx cyx | amp*10 | kind
            "frame_nodes": fn, "frame_edges": fe, "graph_nodes": gn, "graph_edges": ge,
        }
        for row in out["gaussians"]:
            pass
        with open(os.path.join(a.out, f"{name}.js"), "w") as f:
            f.write("window.CANVAS_DATA=window.CANVAS_DATA||{};window.CANVAS_DATA[%s]=" % json.dumps(name))
            json.dump(out, f, separators=(",", ":"))
            f.write(";\n")
        print(name, out["scores"], fs, out["graph_stats"], f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

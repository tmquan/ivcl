"""Feasibility prototype of the slab renderer (not the final code).

Each Gaussian writes the closed-form integral of its density over one section
thickness into the voxels within 3 standard deviations; contributions are
summed with index_add. Axis order is (z, y, x); lengths are in nanometres.
"""
import argparse
import math
import time

import numpy as np
import torch


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def covariance(log_s, q):
    R = quat_to_rot(q)
    S = torch.exp(log_s)
    M = R * S[:, None, :]
    return M @ M.transpose(1, 2)


def render(mu, log_s, q, amp, shape, voxel, zsel=None, mode="slab", prefilter=True,
           chunk=40_000_000, bg=0.0):
    """Render sections ``zsel`` (tensor of section indices, or None for all).

    Returns [len(zsel), Y, X]. ``mode``: "slab" integrates over the section
    thickness, "point" samples the section centre.
    """
    dev = mu.device
    Z, Y, X = shape
    vz, vy, vx = voxel
    cov = covariance(log_s, q)
    if prefilter:  # a pixel averages over its own area: add the variance of a square
        cov = cov.clone()
        cov[:, 1, 1] = cov[:, 1, 1] + vy * vy / 12
        cov[:, 2, 2] = cov[:, 2, 2] + vx * vx / 12
    szz = cov[:, 0, 0]
    syy, sxx, syx = cov[:, 1, 1], cov[:, 2, 2], cov[:, 1, 2]
    szy, szx = cov[:, 0, 1], cov[:, 0, 2]
    det = syy * sxx - syx * syx
    iyy, ixx, iyx = sxx / det, syy / det, -syx / det          # inverse of the in-plane block
    gy = szy * iyy + szx * iyx                                  # Sigma_zp Sigma_pp^-1
    gx = szy * iyx + szx * ixx
    v = (szz - (gy * szy + gx * szx)).clamp_min(1e-6)            # conditional variance along z

    if zsel is None:
        zsel = torch.arange(Z, device=dev)
    nz = len(zsel)
    zmap = torch.full((Z,), -1, device=dev, dtype=torch.long)
    zmap[zsel] = torch.arange(nz, device=dev)
    out = torch.zeros(nz * Y * X, device=dev) + bg

    with torch.no_grad():
        hz = torch.ceil((3 * szz.sqrt() + vz / 2) / vz).long().clamp(1, 8)
        hy = torch.ceil(3 * syy.sqrt() / vy).long().clamp(1, 64)
        hx = torch.ceil(3 * sxx.sqrt() / vx).long().clamp(1, 64)
        # buckets: round the in-plane half width up to a power of two
        hp = torch.maximum(hy, hx)
        bp = (2 ** torch.ceil(torch.log2(hp.float()))).long()
        key = hz * 1000 + bp
        cz = torch.floor(mu[:, 0] / vz).long()
        cy = torch.floor(mu[:, 1] / vy).long()
        cx = torch.floor(mu[:, 2] / vx).long()

    for k in torch.unique(key).tolist():
        ids = torch.nonzero(key == k).squeeze(1)
        kz, kp = k // 1000, k % 1000
        oz = torch.arange(-kz, kz + 1, device=dev)
        op = torch.arange(-kp, kp + 1, device=dev)
        per = len(oz) * len(op) * len(op)
        step = max(1, chunk // per)
        for s in range(0, len(ids), step):
            i = ids[s:s + step]
            iz = cz[i, None] + oz[None, :]                       # [n, Kz]
            iy = cy[i, None] + op[None, :]                       # [n, Kp]
            ix = cx[i, None] + op[None, :]
            okz = (iz >= 0) & (iz < Z)
            zz = zmap[iz.clamp(0, Z - 1)]
            okz = okz & (zz >= 0)
            oky = (iy >= 0) & (iy < Y)
            okx = (ix >= 0) & (ix < X)
            dy = (iy.float() + 0.5) * vy - mu[i, 1, None]        # [n, Kp]
            dx = (ix.float() + 0.5) * vx - mu[i, 2, None]
            # in-plane exponent, [n, Ky, Kx]
            e = (iyy[i, None, None] * dy[:, :, None] ** 2 + ixx[i, None, None] * dx[:, None, :] ** 2
                 + 2 * iyx[i, None, None] * dy[:, :, None] * dx[:, None, :])
            plane = torch.exp(-0.5 * e)
            m = mu[i, 0, None, None] + gy[i, None, None] * dy[:, :, None] + gx[i, None, None] * dx[:, None, :]
            z0 = (iz.float() + 0.5) * vz                         # [n, Kz]
            sd = v[i].sqrt()[:, None, None, None]
            d = z0[:, :, None, None] - m[:, None, :, :]          # [n, Kz, Ky, Kx]
            if mode == "slab":
                r2 = math.sqrt(2.0)
                wz = (torch.erf((d + vz / 2) / (r2 * sd)) - torch.erf((d - vz / 2) / (r2 * sd))) \
                    * (math.sqrt(2 * math.pi) * sd / (2 * vz))
            else:
                wz = torch.exp(-0.5 * (d / sd) ** 2)
            w = amp[i, None, None, None] * plane[:, None] * wz
            ok = okz[:, :, None, None] & oky[:, None, :, None] & okx[:, None, None, :]
            flat = (zz.clamp_min(0)[:, :, None, None] * Y + iy.clamp(0, Y - 1)[:, None, :, None]) * X \
                + ix.clamp(0, X - 1)[:, None, None, :]
            out = out.index_add(0, flat[ok], w[ok])
    return out.reshape(nz, Y, X)


def psnr(a, b):
    return float(10 * torch.log10(1.0 / torch.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf", default=None)
    ap.add_argument("--shape", type=int, nargs=3, default=[16, 256, 256])
    ap.add_argument("--spacing", type=float, default=16.0, help="grid spacing of the initial Gaussians in the plane, nm")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--zbatch", type=int, default=4)
    ap.add_argument("--iso", action="store_true")
    ap.add_argument("--mode", default="slab")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    Z, Y, X = args.shape
    voxel = (40.0, 4.0, 4.0)
    if args.hdf:
        import h5py
        with h5py.File(args.hdf) as f:
            raw = f["volumes/raw"][40:40 + Z, 400:400 + Y, 400:400 + X]
        tgt = torch.tensor(raw, dtype=torch.float32, device=dev) / 255
    else:
        tgt = torch.rand(Z, Y, X, device=dev)
    bg = float(tgt.mean())

    # initial Gaussians: a regular grid, one layer per section
    sp = args.spacing
    gy = torch.arange(sp / 2, Y * voxel[1], sp)
    gx = torch.arange(sp / 2, X * voxel[2], sp)
    gz = (torch.arange(Z) + 0.5) * voxel[0]
    mu = torch.stack(torch.meshgrid(gz, gy, gx, indexing="ij"), -1).reshape(-1, 3).to(dev)
    N = len(mu)
    log_s = torch.log(torch.tensor([20.0, sp * 0.6, sp * 0.6], device=dev)).repeat(N, 1)
    q = torch.zeros(N, 4, device=dev); q[:, 0] = 1
    idx = (mu / torch.tensor(voxel, device=dev)).long()
    amp = (tgt[idx[:, 0], idx[:, 1], idx[:, 2]] - bg) * 0.5
    print(f"N = {N}  voxels = {Z * Y * X}  voxels per Gaussian = {Z * Y * X / N:.1f}")
    params = [mu, log_s, q, amp]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [mu], "lr": 0.5}, {"params": [log_s], "lr": 0.01},
        {"params": [q], "lr": 0.005}, {"params": [amp], "lr": 0.01}])
    t0 = time.time()
    for it in range(args.steps):
        zsel = torch.randperm(Z, device=dev)[:args.zbatch].sort().values
        ls = log_s if not args.iso else log_s.mean(1, keepdim=True).expand(-1, 3)
        img = render(mu, ls, q, amp, (Z, Y, X), voxel, zsel, mode=args.mode, bg=bg)
        loss = (img - tgt[zsel]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == args.steps - 1:
            torch.cuda.synchronize()
            print(f"step {it:4d}  loss {loss.item():.4f}  psnr {psnr(img.detach(), tgt[zsel]):.2f}  "
                  f"{(time.time() - t0) / (it + 1) * 1000:.0f} ms/step  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    with torch.no_grad():
        ls = log_s if not args.iso else log_s.mean(1, keepdim=True).expand(-1, 3)
        t1 = time.time()
        full = render(mu, ls, q, amp, (Z, Y, X), voxel, None, mode=args.mode, bg=bg)
        torch.cuda.synchronize()
        print(f"full render {time.time() - t1:.2f} s   PSNR {psnr(full, tgt):.2f} dB   "
              f"bytes/voxel {N * 11 * 4 / (Z * Y * X):.2f}")
        s = torch.exp(ls).sort(dim=1, descending=True).values
        print("median scales nm (s1, s2, s3):", [round(float(x), 1) for x in s.median(0).values])
        np.save("/tmp/proto_full.npy", full.cpu().numpy())
        np.save("/tmp/proto_tgt.npy", tgt.cpu().numpy())


if __name__ == "__main__":
    main()

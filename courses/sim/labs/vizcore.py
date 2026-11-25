"""vizcore: the visualization algorithms of Chapter 1, in plain NumPy.

Nothing here needs a simulator or a plotting library. Every lab of
this chapter imports from it, so you can read every line of every
algorithm you use.

Self-test:  python labs/vizcore.py
"""
import numpy as np

# ----------------------------------------------------------------
# Datasets: topology + geometry + attributes
# ----------------------------------------------------------------


def make_terrain(n=65, size=4.0):
    """A smooth height field on an n x n grid covering size x size m.

    Returns X, Y, Z arrays of shape (n, n). Z is in metres.
    """
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = (0.45 * np.exp(-((X - 0.6) ** 2 + (Y - 0.5) ** 2) / 0.5)
         + 0.30 * np.exp(-((X + 0.9) ** 2 + (Y + 0.7) ** 2) / 0.3)
         + 0.05 * np.sin(2.5 * X) * np.cos(2.0 * Y))
    return X, Y, Z - Z.min()


def grid_to_mesh(X, Y, Z):
    """Turn a height field into a quad mesh.

    Returns
      points  (N, 3) float   geometry
      counts  (M,)   int     topology: vertices per face (all 4)
      indices (4M,)  int     topology: which points form each face
    """
    n, m = Z.shape
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1),
                       indexing="ij")
    a = (i * m + j).ravel()
    quads = np.stack([a, a + m, a + m + 1, a + 1], axis=1)
    counts = np.full(len(quads), 4)
    return points, counts, quads.ravel()


def subsample_grid(X, Y, Z, stride):
    """Data reduction: keep every stride-th sample."""
    return X[::stride, ::stride], Y[::stride, ::stride], \
        Z[::stride, ::stride]


def reduction_table(n=129, strides=(1, 2, 4, 8, 16, 32)):
    """What subsampling a terrain costs. For each stride returns
    (stride, points, quads, max error, rms error, Xs, Zs): the
    error is measured after rebuilding the full grid from the
    reduced one by linear interpolation."""
    X, Y, Z = make_terrain(n)
    xs, ys = X[:, 0], Y[0, :]
    rows = []
    for stride in strides:
        Xs, Ys, Zs = subsample_grid(X, Y, Z, stride)
        lines = np.array([np.interp(ys, Ys[0, :], r) for r in Zs])
        back = np.array([np.interp(xs, Xs[:, 0], c)
                         for c in lines.T]).T
        err = np.abs(back - Z)
        rows.append((stride, Zs.size, (Zs.shape[0] - 1) ** 2,
                     float(err.max()),
                     float(np.sqrt((err ** 2).mean())), Xs, Zs))
    return rows


def grid_normals(X, Y, Z):
    """Unit surface normals of a height field, shape (n, n, 3)."""
    dzdx = np.gradient(Z, X[:, 0], axis=0)
    dzdy = np.gradient(Z, Y[0, :], axis=1)
    nrm = np.stack([-dzdx, -dzdy, np.ones_like(Z)], axis=2)
    return nrm / np.linalg.norm(nrm, axis=2, keepdims=True)


# ----------------------------------------------------------------
# Colour mapping: scalar -> lookup table -> colour
# ----------------------------------------------------------------

_VIRIDIS = np.array([
    [0x44, 0x01, 0x54], [0x48, 0x28, 0x78], [0x3e, 0x49, 0x89],
    [0x31, 0x68, 0x8e], [0x26, 0x82, 0x8e], [0x1f, 0x9e, 0x89],
    [0x35, 0xb7, 0x79], [0x6e, 0xce, 0x58], [0xb5, 0xde, 0x2b],
    [0xfd, 0xe7, 0x25]]) / 255.0


def lut_viridis(s):
    """Perceptually uniform table. s in [0, 1] -> (..., 3) RGB."""
    s = np.clip(np.asarray(s, dtype=float), 0, 1)
    x = np.linspace(0, 1, len(_VIRIDIS))
    return np.stack([np.interp(s, x, _VIRIDIS[:, c])
                     for c in range(3)], axis=-1)


def lut_rainbow(s):
    """Blue -> cyan -> green -> yellow -> red. s in [0, 1]."""
    s = np.clip(np.asarray(s, dtype=float), 0, 1)
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = np.interp(s, x, [0, 0, 0, 1, 1])
    g = np.interp(s, x, [0, 1, 1, 1, 0])
    b = np.interp(s, x, [1, 1, 0, 0, 0])
    return np.stack([r, g, b], axis=-1)


def map_scalars(values, vmin, vmax, lut=lut_viridis):
    """Clamp to [vmin, vmax], normalise, look up. Returns RGB."""
    s = (np.asarray(values, dtype=float) - vmin) / (vmax - vmin)
    return lut(s)


def srgb_to_linear(rgb):
    """Screen colours (sRGB, as in hex codes) to linear light
    (what a renderer computes with)."""
    c = np.asarray(rgb, dtype=float)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def depth_to_rgb(depth, lo, hi, lut=lut_viridis):
    """Colour-map a depth image (metres) to an 8-bit picture.
    Pixels with no hit (inf) are drawn white."""
    hit = np.isfinite(depth)
    rgb = map_scalars(np.where(hit, depth, lo), lo, hi, lut)
    rgb[~hit] = 1.0
    return (rgb * 255).round().astype(np.uint8)


def luminance(rgb):
    """Perceived brightness (Rec. 709) of RGB colours."""
    return np.asarray(rgb) @ np.array([0.2126, 0.7152, 0.0722])


# ----------------------------------------------------------------
# Contouring: marching squares
# ----------------------------------------------------------------

# For each of the 16 cases: which cell edges the contour crosses.
# Corners: 0=(i,j) 1=(i+1,j) 2=(i+1,j+1) 3=(i,j+1)
# Edges:   0=(0-1) 1=(1-2) 2=(2-3) 3=(3-0)
_MS_CASES = {
    0: [], 15: [],
    1: [(3, 0)], 14: [(3, 0)],
    2: [(0, 1)], 13: [(0, 1)],
    4: [(1, 2)], 11: [(1, 2)],
    8: [(2, 3)], 7: [(2, 3)],
    3: [(3, 1)], 12: [(3, 1)],
    6: [(0, 2)], 9: [(0, 2)],
    5: None, 10: None,          # the two ambiguous cases
}
_MS_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def marching_squares(F, level, xs, ys):
    """Contour lines of F(x, y) = level on a regular grid.

    F has shape (len(xs), len(ys)). Returns
      segments   (K, 2, 2): K line segments, each two (x, y) points
      ambiguous  number of cells that hit case 5 or 10
    """
    segs, ambiguous = [], 0
    above = F > level
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            corner = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            case = sum(1 << k for k, c in enumerate(corner)
                       if above[c])
            pairs = _MS_CASES[case]
            if pairs is None:
                ambiguous += 1
                centre = np.mean([F[c] for c in corner])
                if (centre > level) == (case == 5):
                    pairs = [(0, 1), (2, 3)]   # join the "above"s
                else:
                    pairs = [(3, 0), (1, 2)]   # separate them
            for e0, e1 in pairs:
                seg = []
                for e in (e0, e1):
                    p, q = corner[_MS_EDGES[e][0]], \
                        corner[_MS_EDGES[e][1]]
                    w = (level - F[p]) / (F[q] - F[p])
                    seg.append((xs[p[0]] + w * (xs[q[0]] - xs[p[0]]),
                                ys[p[1]] + w * (ys[q[1]] - ys[p[1]])))
                segs.append(seg)
    return np.array(segs).reshape(-1, 2, 2), ambiguous


def saddle_check(n=8):
    """Contour F = x*y - 0.01 at 0 on an n x n grid. Its positive
    regions are two opposite corners that do not touch, so the cell
    in the middle is an ambiguous case with a known right answer.
    Returns (ambiguous cells, resolved correctly?, segments, g, F)."""
    g = np.linspace(-1, 1, n)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    field = GX * GY - 0.01
    segs, ambiguous = marching_squares(field, 0.0, g, g)
    centre = segs[np.abs(segs).max(axis=(1, 2)) < 0.2]
    mid = centre.mean(axis=1)
    correct = bool(np.all(mid[:, 0] * mid[:, 1] > 0))
    return ambiguous, correct, segs, g, field


# ----------------------------------------------------------------
# Time integration
# ----------------------------------------------------------------


def integrate(velocity, x0, dt, steps, method="euler"):
    """Follow a particle through a velocity field.

    velocity(x) -> v. method is "euler" or "rk2".
    Returns positions of shape (steps + 1, dim).
    """
    x = np.asarray(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        v = velocity(x)
        if method == "euler":
            x = x + dt * v
        elif method == "rk2":
            x = x + dt / 2 * (v + velocity(x + dt * v))
        else:
            raise ValueError(method)
        path.append(x.copy())
    return np.array(path)


def rotation_field(x):
    """v = (-y, x): every particle should travel on a circle."""
    return np.array([-x[1], x[0]])


def circle_test(dts=(0.2, 0.1, 0.05)):
    """One turn around rotation_field with both methods.
    Returns rows (dt, method, radius at the end, path)."""
    rows = []
    for dt in dts:
        steps = int(round(2 * np.pi / dt))
        for method in ("euler", "rk2"):
            path = integrate(rotation_field, [1, 0], dt, steps, method)
            rows.append((dt, method, float(np.hypot(*path[-1])), path))
    return rows


def drop(z0, half, dt, duration, g=9.81):
    """A body falling from height z0 onto the ground, integrated
    with semi-implicit Euler. half is the height at which its
    centre rests. Returns rows (t, z), the first at t = dt."""
    z, vz, rows = float(z0), 0.0, []
    for k in range(int(round(duration / dt))):
        vz -= g * dt
        z += vz * dt
        if z <= half:
            z, vz = half, 0.0
        rows.append(((k + 1) * dt, z))
    return np.array(rows)


def simulate_throw(pos0, vel0, dt, steps, half, g=9.81):
    """Bodies thrown with velocities vel0 from pos0, integrated with
    semi-implicit Euler; a body stops where it first touches the
    ground. Returns t (steps,), pos and vel (steps, N, 3)."""
    p = np.array(pos0, dtype=float)
    v = np.array(vel0, dtype=float)
    t, pos, vel = [], [], []
    for k in range(steps):
        v[:, 2] -= g * dt
        p += v * dt
        down = p[:, 2] <= half
        p[down, 2] = half
        v[down] = 0.0
        t.append((k + 1) * dt)
        pos.append(p.copy())
        vel.append(v.copy())
    return np.array(t), np.array(pos), np.array(vel)


# ----------------------------------------------------------------
# Tensors: inertia and its principal axes
# ----------------------------------------------------------------


def box_inertia(mass, a, b, c):
    """Inertia tensor of a solid box with edges a, b, c (x, y, z)."""
    return mass / 12.0 * np.diag([b * b + c * c, a * a + c * c,
                                  a * a + b * b])


def rotation_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def principal_axes(tensor):
    """Eigen-decomposition of a symmetric 3x3 tensor.

    Returns eigenvalues (ascending) and eigenvectors as columns.
    """
    tensor = np.asarray(tensor, dtype=float).reshape(3, 3)
    return np.linalg.eigh((tensor + tensor.T) / 2)


def ellipsoid_surface(evals, evecs, n=24):
    """Points of the tensor ellipsoid: radii = eigenvalues,
    axes = eigenvectors. Returns X, Y, Z of shape (n, n)."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n),
                       np.linspace(0, np.pi, n))
    sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v),
                       np.cos(v)], axis=-1)
    pts = (sphere * evals) @ np.asarray(evecs).T
    return pts[..., 0], pts[..., 1], pts[..., 2]


# ----------------------------------------------------------------
# Implicit functions and distance fields
# ----------------------------------------------------------------


def sdf_circle(X, Y, centre, radius):
    """Signed distance to a circle: < 0 inside, 0 on, > 0 outside."""
    return np.hypot(X - centre[0], Y - centre[1]) - radius


def sdf_box(X, Y, centre, half):
    """Signed distance to an axis-aligned rectangle."""
    dx = np.abs(X - centre[0]) - half[0]
    dy = np.abs(Y - centre[1]) - half[1]
    outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
    return outside + np.minimum(np.maximum(dx, dy), 0)


def op_union(a, b):
    return np.minimum(a, b)


def op_intersection(a, b):
    return np.maximum(a, b)


def op_difference(a, b):
    return np.maximum(a, -b)


def demo_part(xs):
    """A part built from three distance functions on the grid
    xs x xs: a disc, minus a hole, minus a slot."""
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    disc = sdf_circle(X, Y, (0.0, 0.0), 1.0)
    hole = sdf_circle(X, Y, (0.0, 0.0), 0.5)
    slot = sdf_box(X, Y, (0.9, 0.0), (0.5, 0.15))
    ring = op_difference(disc, hole)             # disc minus hole
    return op_difference(ring, slot)             # ... minus a slot


def ring_mesh(r_out=0.5, r_in=0.25, height=0.1, n=48):
    """A washer (flat ring with a hole) as a closed triangle mesh.

    Returns points (4n, 3) and triangles (8n, 3), centred on the
    origin, hole along Z. A convex hull of this shape has no hole.
    """
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    h = height / 2
    ob = np.stack([r_out * c, r_out * s, np.full(n, -h)], 1)
    ot = np.stack([r_out * c, r_out * s, np.full(n, h)], 1)
    ib = np.stack([r_in * c, r_in * s, np.full(n, -h)], 1)
    it = np.stack([r_in * c, r_in * s, np.full(n, h)], 1)
    points = np.concatenate([ob, ot, ib, it])
    OB, OT, IB, IT = 0, n, 2 * n, 3 * n
    tris = []
    for k in range(n):
        m = (k + 1) % n
        tris += [(OB + k, OB + m, OT + m), (OB + k, OT + m, OT + k)]
        tris += [(IB + k, IT + m, IB + m), (IB + k, IT + k, IT + m)]
        tris += [(OT + k, OT + m, IT + m), (OT + k, IT + m, IT + k)]
        tris += [(OB + k, IB + m, OB + m), (OB + k, IB + k, IB + m)]
    return points, np.array(tris)


def mesh_volume(points, tris):
    """Signed volume of a closed triangle mesh (positive when the
    triangles are wound counter-clockwise seen from outside)."""
    a, b, c = (points[tris[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


# ----------------------------------------------------------------
# Glyphs: orient an object along a vector
# ----------------------------------------------------------------


def quat_from_z(directions):
    """Quaternions (w, x, y, z) that rotate +Z onto each direction."""
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.broadcast_to(z, d.shape), d)
    w = 1.0 + d @ z
    q = np.concatenate([w[..., None], axis], axis=-1)
    flipped = w < 1e-9                    # direction is exactly -Z
    q[flipped] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


# ----------------------------------------------------------------
# Probing: sample a scene along rays
# ----------------------------------------------------------------


def ray_fan(n, fov_deg=60.0):
    """n unit directions in the XY plane, spread over fov_deg
    around +X. Returns (n, 3)."""
    ang = np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n))
    return np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)


def raycast_boxes_2d(origin, directions, boxes, max_dist=20.0):
    """Reference ray caster: distance from origin along each
    direction to the nearest box. boxes = [(cx, cy, hx, hy), ...].
    Returns distances (n,), np.inf where nothing is hit."""
    o = np.asarray(origin, dtype=float)[:2]
    out = np.full(len(directions), np.inf)
    for k, d in enumerate(np.asarray(directions)[:, :2]):
        for cx, cy, hx, hy in boxes:
            lo = np.array([cx - hx, cy - hy])
            hi = np.array([cx + hx, cy + hy])
            with np.errstate(divide="ignore", invalid="ignore"):
                t0, t1 = (lo - o) / d, (hi - o) / d
            tn = np.nanmax(np.minimum(t0, t1))
            tf = np.nanmin(np.maximum(t0, t1))
            if tn <= tf and tf > 0 and 0 < tn < min(out[k], max_dist):
                out[k] = tn
    return out


# ----------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------

if __name__ == "__main__":
    X, Y, Z = make_terrain(33)
    pts, counts, idx = grid_to_mesh(X, Y, Z)
    assert pts.shape == (33 * 33, 3) and len(counts) == 32 * 32
    assert idx.max() == len(pts) - 1
    print(f"terrain      {len(pts)} points, {len(counts)} quads,"
          f" height 0..{Z.max():.2f} m")

    assert np.allclose(lut_viridis(0.0), _VIRIDIS[0])
    assert np.allclose(lut_rainbow(1.0), [1, 0, 0])
    s = np.linspace(0, 1, 101)
    for name, lut in (("viridis", lut_viridis),
                      ("rainbow", lut_rainbow)):
        lum = luminance(lut(s))
        mono = bool(np.all(np.diff(lum) > 0))
        print(f"lut {name:8s} brightness always increasing: {mono}")

    lin = srgb_to_linear([0.463, 0.725, 0.0])       # NVIDIA green
    assert np.allclose(lin, [0.1812, 0.4846, 0.0], atol=1e-3)
    print("colour       sRGB (0.463, 0.725, 0) = linear",
          np.round(lin, 3))

    xs = np.linspace(-2, 2, 81)
    GX, GY = np.meshgrid(xs, xs, indexing="ij")
    segs, amb = marching_squares(np.hypot(GX, GY), 1.0, xs, xs)
    r = np.hypot(segs[..., 0], segs[..., 1])
    length = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1).sum()
    assert abs(r - 1).max() < 2e-3
    print(f"contour      {len(segs)} segments, circle length"
          f" {length:.4f} (2*pi = {2*np.pi:.4f}), ambiguous {amb}")

    for method in ("euler", "rk2"):
        path = integrate(rotation_field, [1, 0], 0.1, 63, method)
        print(f"integrate    {method:5s} radius after one turn:"
              f" {np.hypot(*path[-1]):.4f} (exact 1.0)")

    I = box_inertia(2.0, 0.6, 0.5, 0.2)
    R = rotation_z(30)
    vals, vecs = principal_axes(R @ I @ R.T)
    assert np.allclose(vals, np.sort(np.diag(I)))
    print("inertia      principal moments", np.round(vals, 4))

    p, t = ring_mesh()
    vol = mesh_volume(p, t)
    exact = np.pi * (0.5**2 - 0.25**2) * 0.1
    assert vol > 0 and abs(vol - exact) / exact < 0.01
    print(f"ring mesh    volume {vol:.5f} m^3 (exact {exact:.5f})")

    n = grid_normals(X, Y, Z).reshape(-1, 3)
    q = quat_from_z(n)
    assert np.allclose(quat_rotate(q, np.array([0, 0, 1.0])), n)
    print(f"glyphs       {len(q)} quaternions rotate +Z onto normals")

    rows = drop(2.0, 0.15, 1 / 60, 1.2)
    t, pos, vel = simulate_throw([[0, 0, 0.3]], [[1, 0, 3.0]],
                                 1 / 120, 120, 0.06)
    assert abs(rows[-1, 1] - 0.15) < 1e-12 and pos[-1, 0, 2] == 0.06
    print(f"physics      drop lands at t = "
          f"{rows[rows[:, 1] <= 0.15][0, 0]:.3f} s; a throw at"
          f" 3 m/s rises to {pos[:, 0, 2].max():.2f} m")

    d = raycast_boxes_2d([-3, 0, 0.3], ray_fan(5),
                         [(0, 0, 0.5, 0.5)])
    assert abs(d[2] - 2.5) < 1e-9
    print("probe        centre ray hits the box at", d[2], "m")
    print("all vizcore self-tests passed")

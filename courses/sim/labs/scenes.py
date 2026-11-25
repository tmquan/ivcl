"""scenes: the scene each lab shows, written down once.

Every labNN() function returns (prims, view): a list of meshes and
a camera. A lab puts them on the stage with look.usd_scene().
"""
import numpy as np

import look
import vizcore as vc

# ---- Lab 1: a terrain mesh -------------------------------------


def lab01(n=65, stride=1, taller=False):
    X, Y, Z = vc.subsample_grid(*vc.make_terrain(n), stride)
    prim = look.height_field("terrain", X, Y, Z, look.GREY_RGB)
    if taller:                        # the geometric transformation
        prim["points"][:, 2] *= 2.0
    return [prim], look.View((3.9, -3.9, 3.3), (0.0, 0.0, 0.25))


# ---- Lab 2: one scalar, two lookup tables ----------------------


def lab02(n=65):
    X, Y, Z = vc.make_terrain(n)
    scalar = Z                                   # colour by height
    lo, hi = float(scalar.min()), float(scalar.max())
    prims = []
    for name, lut, y in (("rainbow", vc.lut_rainbow, -2.2),
                         ("viridis", vc.lut_viridis, 2.2)):
        rgb = vc.map_scalars(scalar, lo, hi, lut)
        prims.append(look.height_field(name, X, Y, Z, rgb,
                                       origin=(0.0, y, 0.0)))
    return prims, look.View((6.2, 0.0, 5.1), (0.0, 0.0, 0.0))


# ---- Lab 3: three cubes in front of a depth camera -------------

LAB03_CUBES = [((1.5, -0.9, 0.25), (0.9, 0.2, 0.2)),
               ((3.0, 0.2, 0.25), (0.2, 0.7, 0.2)),
               ((5.0, 1.0, 0.25), (0.2, 0.3, 0.9))]


def lab03():
    prims = [look.ground(half=8.0)]
    for k, (centre, rgb) in enumerate(LAB03_CUBES):
        prims.append(look.box(f"cube{k}", centre, 0.5, rgb))
    return prims, look.View((-1.0, 0.0, 1.0), (3.0, 0.0, 0.25))


def lab03_depth(width=look.WIDTH // 2, height=look.HEIGHT // 2):
    """What a perfect depth camera returns for lab03(): metres to
    the image plane for every pixel, inf where nothing is hit.
    One ray per pixel through the scene's own camera."""
    prims, view = lab03()
    f = view.target - view.eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    th = np.tan(np.radians(look.FOV) / 2)
    xs = (np.arange(width) + 0.5) / width * 2 - 1
    ys = 1 - (np.arange(height) + 0.5) / height * 2
    D = (f + (xs * th * width / height)[None, :, None] * r
         + (ys * th)[:, None, None] * u)         # forward part = 1
    depth = np.full((height, width), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in prims:
            pts = look.world_points(p)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            if p["name"] == "ground":
                t = -view.eye[2] / D[..., 2]
                at = view.eye + t[..., None] * D
                hit = ((t > 0) & (at[..., 0] >= lo[0])
                       & (at[..., 0] <= hi[0]) & (at[..., 1] >= lo[1])
                       & (at[..., 1] <= hi[1]))
            else:
                t0, t1 = (lo - view.eye) / D, (hi - view.eye) / D
                t = np.nanmax(np.minimum(t0, t1), axis=-1)
                far = np.nanmin(np.maximum(t0, t1), axis=-1)
                hit = (t <= far) & (t > 0)
            depth = np.where(hit & (t < depth), t, depth)
    return depth.astype(np.float32)


# ---- Lab 4: contour lines on the terrain -----------------------

LAB04_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]


def lab04_segments(n=65):
    """{level: segments (K, 2, 2)} from marching squares."""
    X, Y, Z = vc.make_terrain(n)
    return {lv: vc.marching_squares(Z, lv, X[:, 0], Y[0, :])
            for lv in LAB04_LEVELS}


def lab04(n=65):
    X, Y, Z = vc.make_terrain(n)
    rgb = vc.map_scalars(Z, 0.0, float(Z.max()))
    starts, ends = [], []
    for level, (segs, _ambiguous) in lab04_segments(n).items():
        lift = np.full((len(segs), 1), level + 0.012)
        starts.append(np.hstack([segs[:, 0], lift]))
        ends.append(np.hstack([segs[:, 1], lift]))
    prims = [look.height_field("terrain", X, Y, Z, rgb),
             look.tubes("contours", np.vstack(starts), np.vstack(ends),
                        look.WHITE_RGB, width=0.016)]
    return prims, look.View((3.3, -3.3, 3.9), (0.0, 0.0, 0.0))


# ---- Lab 5: thrown cubes, pathlines and a hedgehog -------------

LAB05_SIZE = 0.12
LAB05_DT, LAB05_STEPS, LAB05_SNAPSHOT = 1 / 120, 120, 30


def lab05_start():
    """Start positions and launch velocities of the 25 cubes."""
    g = np.linspace(-1.0, 1.0, 5)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    pos = np.stack([GX.ravel(), GY.ravel(),
                    np.full(GX.size, 0.3)], axis=1)
    vel = np.stack([-1.5 * pos[:, 1], 1.5 * pos[:, 0],
                    np.full(len(pos), 3.0)], axis=1)
    return pos, vel


def lab05_recording():
    """The throw computed with vizcore's own integrator instead of
    a physics engine. Returns t, pos, vel."""
    pos, vel = lab05_start()
    return vc.simulate_throw(pos, vel, LAB05_DT, LAB05_STEPS,
                             LAB05_SIZE / 2)


def lab05_base():
    pos, _vel = lab05_start()
    prims = [look.ground(half=3.2)]
    for k, p in enumerate(pos):
        prims.append(look.box(f"cube{k:02d}", p, LAB05_SIZE,
                              (0.8, 0.8, 0.8)))
    return prims, look.View((3.3, -3.3, 2.3), (0.0, 0.0, 0.4))


def lab05_overlay(pos, vel, snapshot, every=4, scale=0.08):
    """Pathlines coloured by speed, and a hedgehog at one step."""
    speed = np.linalg.norm(vel, axis=2)
    a, b = pos[:-every:every], pos[every::every]
    rgb = vc.map_scalars(speed[:-every:every], 0.0, speed.max())
    p, v = pos[snapshot], vel[snapshot]
    return [look.tubes("pathlines", a, b, rgb.reshape(-1, 3), 0.014),
            look.tubes("hedgehog", p, p + scale * v, look.WHITE_RGB,
                       0.026)]


def lab05(pos, vel, snapshot):
    prims, view = lab05_base()
    return prims + lab05_overlay(pos, vel, snapshot), view


# ---- Lab 6: the same cube dropped at three time steps ----------

LAB06_Z0, LAB06_SIZE, LAB06_DURATION = 2.0, 0.3, 1.2     # m, m, s
LAB06_RATES = (240, 60, 15)                              # steps / s


def lab06():
    prims = [look.ground(half=1.5),
             look.box("cube", (0.0, 0.0, LAB06_Z0), LAB06_SIZE)]
    return prims, look.View((3.4, -3.4, 2.6), (0.0, 0.0, 0.9))


def lab06_error(rows, t=0.4):
    """Height error against free fall at time t, in metres."""
    exact = LAB06_Z0 - 0.5 * 9.81 * t ** 2
    return float(np.interp(t, rows[:, 0], rows[:, 1]) - exact)


# ---- Lab 7: a box and the principal axes of its inertia --------

LAB07_EDGES = (0.6, 0.5, 0.2)
LAB07_MASS = 2.0
LAB07_CENTRE = (0.0, 0.0, 0.1)


def lab07_axes(tensor=None):
    """Three arrows from the centre of the box along the principal
    axes, length ~ moment."""
    if tensor is None:
        tensor = vc.box_inertia(LAB07_MASS, *LAB07_EDGES)
    vals, vecs = vc.principal_axes(tensor)
    c = np.array(LAB07_CENTRE)
    prims = []
    for k, rgb in enumerate(((0.85, 0.1, 0.1), (0.1, 0.65, 0.1),
                             (0.1, 0.2, 0.9))):
        tip = c + vecs[:, k] * vals[k] / vals.max() * 0.85
        prims.append(look.arrows(f"axis{k + 1}", [c], [tip], rgb, 0.02))
    return prims


def lab07_base():
    prims = [look.ground(half=1.2),
             look.box("box", LAB07_CENTRE, LAB07_EDGES, look.GREEN_RGB)]
    return prims, look.View((1.25, -1.6, 1.15), (0.0, 0.0, 0.3))


def lab07(tensor=None):
    prims, view = lab07_base()
    return prims + lab07_axes(tensor), view


# ---- Lab 8: a ball above a ring --------------------------------

LAB08_RING_Z, LAB08_BALL_R = 0.5, 0.1


def lab08(ball_z=1.0):
    points, tris = vc.ring_mesh(r_out=0.5, r_in=0.25, height=0.1)
    prims = [look.ground(half=1.3),
             look.mesh("ring", points, tris, look.GREY_RGB,
                       origin=(0.0, 0.0, LAB08_RING_Z)),
             look.sphere("ball", (0.0, 0.0, ball_z), LAB08_BALL_R,
                         (0.9, 0.2, 0.2))]
    return prims, look.View((1.5, -1.5, 1.45), (0.0, 0.0, 0.5))


# ---- Lab 9: surface normals as cone glyphs ---------------------

LAB09_HEIGHT, LAB09_RADIUS = 0.12, 0.02


def lab09_glyphs(n=65, every=4):
    """Where each cone stands, how it is turned, how long it is."""
    X, Y, Z = vc.make_terrain(n)
    k = every
    bases = np.stack([X[::k, ::k], Y[::k, ::k], Z[::k, ::k]],
                     -1).reshape(-1, 3)
    normals = vc.grid_normals(X, Y, Z)[::k, ::k].reshape(-1, 3)
    slope = 1.0 - normals[:, 2]                 # 0 on flat ground
    lengths = LAB09_HEIGHT * (1.0 + 6.0 * slope)
    return bases, vc.quat_from_z(normals), lengths, normals


def lab09_terrain(n=65):
    X, Y, Z = vc.make_terrain(n)
    rgb = vc.map_scalars(Z, 0.0, float(Z.max()))
    return ([look.height_field("terrain", X, Y, Z, rgb)],
            look.View((3.2, -3.2, 2.8), (0.0, 0.0, 0.0)))


def lab09(n=65, every=4):
    prims, view = lab09_terrain(n)
    bases, quats, lengths, _normals = lab09_glyphs(n, every)
    return prims + [look.cones("glyphs", bases, quats, lengths,
                               radius=LAB09_RADIUS)], view


# ---- Lab 10: obstacles and two fans of rays --------------------

LAB10_ORIGIN = (-3.0, 0.0, 0.3)
LAB10_FOV, LAB10_MAX = 60.0, 20.0
# (centre x, centre y, half size x, half size y); the last one is a
# thin table leg, 6 cm wide
LAB10_BOXES = [(1.0, -0.8, 0.3, 0.3), (2.0, 0.9, 0.4, 0.25),
               (1.5, 0.0, 0.03, 0.03)]
LAB10_FANS = ((10, (0.9, 0.2, 0.2), 0.03, 0.02),
              (90, (0.45, 0.8, 0.1), 0.012, 0.0))


def lab10_reference(n):
    """Distances a perfect ray caster returns for a fan of n rays."""
    return vc.raycast_boxes_2d(LAB10_ORIGIN, vc.ray_fan(n, LAB10_FOV),
                               LAB10_BOXES)


def lab10_base():
    prims = [look.ground(half=3.6)]
    for k, (cx, cy, hx, hy) in enumerate(LAB10_BOXES):
        prims.append(look.box(f"obstacle{k}", (cx, cy, 0.5),
                              (2 * hx, 2 * hy, 1.0), look.GREY_RGB))
    return prims, look.View((-0.9, -4.2, 5.6), (0.1, 0.0, 0.0))


def lab10_rays(distances=None):
    """Rods from the sensor to each hit. distances = {n: array};
    by default the reference distances."""
    prims = []
    for n, rgb, width, lift in LAB10_FANS:
        dist = (lab10_reference(n) if distances is None
                else np.asarray(distances[n], dtype=float))
        reach = np.where(np.isfinite(dist), dist, 6.0)
        start = np.array(LAB10_ORIGIN) + [0.0, 0.0, lift]
        ends = start + vc.ray_fan(n, LAB10_FOV) * reach[:, None]
        prims.append(look.tubes(f"rays{n}", np.tile(start, (n, 1)),
                                ends, rgb, width))
    return prims


def lab10(distances=None):
    prims, view = lab10_base()
    return prims + lab10_rays(distances), view

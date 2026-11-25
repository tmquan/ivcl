"""look: one scene, one camera, one set of colours.

A scene is a list of triangle meshes ("prims") whose colours already
contain the shading. One call writes that list onto a USD stage,
with a matte material, a camera and an even light:

    usd_scene(stage, prims, view, theme)

Because the points, the triangles, the colours and the camera are
all decided here, every picture of a scene looks the same wherever
it is drawn.

    --theme light | dark    (or IVCL_THEME) picks the background.
"""
import os
import sys

import numpy as np

# ---------------------------------------------------------------
# The look
# ---------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720      # picture size, pixels
FOV = 45.0                     # vertical field of view, degrees
LIGHT = np.array([0.35, -0.55, 0.76])   # direction towards the light
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.45, 0.55  # shade = AMBIENT + DIFFUSE * cos(angle)

GREEN_RGB = (0.463, 0.725, 0.0)         # NVIDIA green, #76B900
GROUND_RGB = (0.80, 0.80, 0.80)
GREY_RGB = (0.55, 0.55, 0.55)
WHITE_RGB = (1.0, 1.0, 1.0)

THEMES = {
    "light": dict(background=(1.0, 1.0, 1.0)),
    "dark": dict(background=(0.07, 0.07, 0.07)),
}
# Brightness of the dome light in Isaac Sim. The colours are already
# shaded, so the scene only needs even light. Tune this one number
# until the ground is as bright as in the chapter's preview:
# light grey, RGB (204, 204, 204).
DOME_INTENSITY = 1000.0


def pick():
    """Return "light" or "dark"; remove --theme from sys.argv."""
    name = os.environ.get("IVCL_THEME", "light")
    if "--theme" in sys.argv:
        i = sys.argv.index("--theme")
        name = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        del sys.argv[i:i + 2]
    if name not in THEMES:
        sys.exit("--theme must be light or dark")
    return name


# ---------------------------------------------------------------
# Colour and shading (done once, here)
# ---------------------------------------------------------------


def linear(rgb):
    """Screen colour (sRGB, as in a hex code) to linear light
    (what USD stores)."""
    c = np.asarray(rgb, dtype=float)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def shade(rgb, normals):
    """Lambert shading from the shared light, as screen colours."""
    cos = np.clip(np.asarray(normals, dtype=float) @ LIGHT, 0.0, 1.0)
    k = np.clip(AMBIENT + DIFFUSE * cos, 0.0, 1.0)
    return np.asarray(rgb, dtype=float) * k[..., None]


def face_normals(points, tris):
    a, b, c = (points[tris[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True),
                          1e-12)


class View:
    """Where the camera is and what it looks at (world, metres)."""

    def __init__(self, eye, target):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)


# ---------------------------------------------------------------
# Prims: triangle meshes with the shading baked into their colours
# (colors). Each prim also keeps its colour before shading (rgb)
# and whether it is shaded at all (lit), for a renderer that lights
# the scene itself.
# ---------------------------------------------------------------


def mesh(name, points, tris, rgb, lit=True, origin=(0.0, 0.0, 0.0)):
    """Flat-shaded mesh, one colour per triangle.

    points are relative to origin; origin becomes the prim's
    position (what a rigid body reports as its pose).
    rgb is one colour or one per triangle. lit=False keeps the
    colour as it is (used for lines and markers).
    """
    points = np.asarray(points, dtype=float)
    tris = np.asarray(tris, dtype=int)
    rgb = np.broadcast_to(np.asarray(rgb, dtype=float),
                          (len(tris), 3))
    colors = shade(rgb, face_normals(points, tris)) if lit else rgb
    return dict(name=name, points=points, tris=tris, per="face",
                colors=np.array(colors), origin=np.array(origin, float),
                rgb=np.array(rgb), lit=lit)


def smooth(name, points, tris, rgb, normals, origin=(0.0, 0.0, 0.0)):
    """Smooth-shaded mesh, one colour per point."""
    points = np.asarray(points, dtype=float)
    rgb = np.broadcast_to(np.asarray(rgb, dtype=float),
                          (len(points), 3))
    return dict(name=name, points=points,
                tris=np.asarray(tris, dtype=int), per="vertex",
                colors=shade(rgb, normals),
                origin=np.array(origin, float),
                rgb=np.array(rgb), lit=True)


_BOX_QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def box(name, centre, size, rgb=GREEN_RGB):
    """A box: 8 points, 12 triangles. size is one edge or (x, y, z)."""
    size = np.broadcast_to(np.asarray(size, dtype=float), (3,))
    corners = np.array([[x, y, z] for x in (-1, 1)
                        for y in (-1, 1) for z in (-1, 1)])
    tris = [t for a, b, c, d in _BOX_QUADS
            for t in ((a, b, c), (a, c, d))]
    return mesh(name, corners * size / 2, tris, rgb, origin=centre)


def ground(half=1.5, rgb=GROUND_RGB, z=0.0):
    """A square piece of ground, 2 * half metres on a side."""
    pts = [(-half, -half, z), (half, -half, z), (half, half, z),
           (-half, half, z)]
    return mesh("ground", pts, [(0, 1, 2), (0, 2, 3)], rgb)


def sphere(name, centre, radius, rgb, n=24):
    """A smooth-shaded sphere."""
    u = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    v = np.linspace(0, np.pi, n + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    nrm = np.stack([np.cos(U) * np.sin(V), np.sin(U) * np.sin(V),
                    np.cos(V)], axis=-1).reshape(-1, 3)
    m = n + 1
    tris = []
    for i in range(2 * n):
        j = (i + 1) % (2 * n)
        for k in range(n):
            a, b = i * m + k, j * m + k
            tris += [(a, b, b + 1), (a, b + 1, a + 1)]
    return smooth(name, nrm * radius, tris, rgb, nrm, origin=centre)


def height_field(name, X, Y, Z, rgb, origin=(0.0, 0.0, 0.0)):
    """A surface z = Z(x, y) on a grid; rgb is one colour per grid
    point, shape (n, m, 3), or a single colour."""
    n, m = Z.shape
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    dzdx = np.gradient(Z, X[:, 0], axis=0)
    dzdy = np.gradient(Z, Y[0, :], axis=1)
    nrm = np.stack([-dzdx, -dzdy, np.ones_like(Z)], 2).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1),
                       indexing="ij")
    a = (i * m + j).ravel()
    tris = np.concatenate([np.stack([a, a + m, a + m + 1], 1),
                           np.stack([a, a + m + 1, a + 1], 1)])
    rgb = np.asarray(rgb, dtype=float)
    rgb = rgb.reshape(-1, 3) if rgb.ndim == 3 else rgb
    return smooth(name, pts, tris, rgb, nrm, origin=origin)


def tubes(name, starts, ends, rgb, width=0.02):
    """Line segments as thin square rods, drawn in their own colour
    (not shaded), so they read as lines."""
    a = np.asarray(starts, dtype=float).reshape(-1, 3)
    b = np.asarray(ends, dtype=float).reshape(-1, 3)
    col = np.broadcast_to(np.asarray(rgb, dtype=float), (len(a), 3))
    d = b - a
    length = np.linalg.norm(d, axis=1)
    keep = length > 1e-9
    a, b, d, col = a[keep], b[keep], d[keep], col[keep]
    d = d / length[keep, None]
    ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]],
                   [[1.0, 0.0, 0.0]])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    w = width / 2
    ring = [u * w + v * w, -u * w + v * w, -u * w - v * w,
            u * w - v * w]
    pts = np.concatenate([np.stack([a + o for o in ring], 1),
                          np.stack([b + o for o in ring], 1)],
                         axis=1).reshape(-1, 3)
    tris = []
    for s in range(len(a)):
        o = 8 * s
        for q in range(4):
            r = (q + 1) % 4
            tris += [(o + q, o + r, o + 4 + r),
                     (o + q, o + 4 + r, o + 4 + q)]
        tris += [(o, o + 2, o + 1), (o, o + 3, o + 2),
                 (o + 4, o + 5, o + 6), (o + 4, o + 6, o + 7)]
    return mesh(name, pts, tris, np.repeat(col, 12, axis=0),
                lit=False)


def cone_shape(radius=0.02, n=12):
    """A cone of height 1 standing on the origin, pointing along +Z.
    Returns points and triangles."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.stack([radius * np.cos(ang), radius * np.sin(ang),
                    np.zeros(n)], axis=1)
    pts = np.concatenate([rim, [[0, 0, 1.0]], [[0, 0, 0.0]]])
    tris = []
    for k in range(n):
        m = (k + 1) % n
        tris += [(k, m, n), (m, k, n + 1)]
    return pts, np.array(tris)


def quat_rotate(q, v):
    """Rotate vectors v by quaternions q = (w, x, y, z)."""
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def cones(name, bases, quats, lengths, rgb=WHITE_RGB, radius=0.02,
          n=12):
    """Many copies of cone_shape(): scaled along Z by length, turned
    by quaternion (w, x, y, z), moved to base. Not shaded."""
    pts, tris = cone_shape(radius, n)
    all_pts, all_tris = [], []
    for k, (base, q, length) in enumerate(zip(bases, quats, lengths)):
        p = pts * np.array([1.0, 1.0, length])
        p = quat_rotate(np.asarray(q, float)[None], p) + base
        all_pts.append(p)
        all_tris.append(tris + k * len(pts))
    return mesh(name, np.concatenate(all_pts),
                np.concatenate(all_tris), rgb, lit=False)


def world_points(prim):
    return prim["points"] + prim["origin"]


# ---------------------------------------------------------------
# Author the prims as USD (plain pxr, so it works inside Isaac
# Sim and in any other USD program)
# ---------------------------------------------------------------


def usd_material(stage, path="/World/Look/Matte"):
    """A matte material: the surface colour is the prim's own
    displayColor, with no highlights added by the renderer."""
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    reader = UsdShade.Shader.Define(stage, path + "/Color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(
        "displayColor")
    color = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    surface = UsdShade.Shader.Define(stage, path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f
                        ).ConnectToSource(color)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("useSpecularWorkflow",
                        Sdf.ValueTypeNames.Int).Set(1)
    surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f
                        ).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface")
    return material


def usd_mesh(stage, path, prim, material=None):
    """Author one prim as a UsdGeom.Mesh at the given path."""
    from pxr import Gf, UsdGeom, UsdShade
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(prim["points"].tolist())           # geometry
    m.CreateFaceVertexCountsAttr([3] * len(prim["tris"]))  # topology
    m.CreateFaceVertexIndicesAttr(prim["tris"].ravel().tolist())
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    how = (UsdGeom.Tokens.uniform if prim["per"] == "face"
           else UsdGeom.Tokens.vertex)
    m.CreateDisplayColorPrimvar(how).Set(                 # attribute
        linear(prim["colors"]).tolist())
    double = UsdGeom.XformOp.PrecisionDouble
    x = UsdGeom.Xformable(m)
    x.AddTranslateOp(double).Set(Gf.Vec3d(*prim["origin"].tolist()))
    x.AddOrientOp(double).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    x.AddScaleOp(double).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def usd_author(stage, prims, root="/World/Look"):
    """One UsdGeom.Mesh per prim. Returns {name: prim path}."""
    from pxr import UsdShade
    matte = UsdShade.Material.Get(stage, root + "/Matte")
    if not matte:
        matte = usd_material(stage, root + "/Matte")
    paths = {}
    for p in prims:
        paths[p["name"]] = f"{root}/{p['name']}"
        usd_mesh(stage, paths[p["name"]], p, matte)
    return paths


def usd_camera(stage, view, path="/World/Camera"):
    """The scene's camera, as a USD camera prim."""
    from pxr import Gf, UsdGeom
    cam = UsdGeom.Camera.Define(stage, path)
    look_at = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*view.eye.tolist()), Gf.Vec3d(*view.target.tolist()),
        Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_at.GetInverse())
    aperture_v = 20.25                      # 36 x 20.25 mm = 16:9
    cam.CreateHorizontalApertureAttr(aperture_v * WIDTH / HEIGHT)
    cam.CreateVerticalApertureAttr(aperture_v)
    cam.CreateFocalLengthAttr(
        aperture_v / (2 * np.tan(np.radians(FOV) / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    return path


def usd_light(stage, theme="light", path="/World/Dome"):
    """Even, white light from all directions. The shading is already
    in the colours, so no other light is wanted."""
    from pxr import Sdf, UsdLux
    dome = UsdLux.DomeLight.Define(stage, path)
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateColorAttr((1.0, 1.0, 1.0))
    if theme == "dark":       # keep the light, hide it from the camera
        dome.GetPrim().CreateAttribute(
            "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    return path


def usd_scene(stage, prims, view, theme="light"):
    """Everything Isaac Sim needs: prims, camera and light."""
    paths = usd_author(stage, prims)
    usd_camera(stage, view)
    usd_light(stage, theme)
    return paths

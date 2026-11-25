"""simscene: the figures of the Isaac Sim pages.

A scene is a list of shapes and a camera. It is written down once, in labs/scene.py and
labs/scenes.py, the files the lab scripts use on the DGX Spark. This module draws the same
scenes, and the charts and diagrams of the pages, with NumPy and matplotlib only:

    show(HelloCube())            a 3D scene: a canvas you can turn, with Play if it has time
    show(FallChart())            a chart or a diagram: every item answers to the pointer
    colour("chapter0.ipynb")     the code blocks of a page in the colours Notion gives them

Every figure also has a still picture, shown where a page cannot run scripts.

Four class families, each with methods that its subclasses override:

    Picture    Scene, Chart                  bundle()                 what a notebook cell displays
    Scene      HelloCube, Terrain, ...       shapes(), motion()       the triangle meshes, what moves with time
    Chart      SetupDiagram, FallChart, ...  plot(fig), items(sheet)  the still picture, the interactive figure
    Renderer   Still, Canvas, Plot, Stage    draw(picture)            pixels, a 3D canvas, a Plotly figure, a USD stage

The interactive charts are drawn in the page by plotly.js, which the page fetches from the
Plotly CDN; no Python package is needed for them.
"""
import base64
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))             # notioncode.py, beside this file
sys.path.insert(0, str(Path(__file__).resolve().parent / "labs"))
import look                      # noqa: E402  shapes, shading and the camera, shared with the lab scripts
import scene as chapter0         # noqa: E402  the scene of Chapter 0
import scenes as chapter1        # noqa: E402  the scenes of Chapter 1
import vizcore as vc             # noqa: E402  the algorithms of Chapter 1


# ------------------------------------------------------------------------------------------
# Pictures: everything a cell can show
# ------------------------------------------------------------------------------------------
class Picture:
    """Base class of everything show() displays."""

    def bundle(self):
        """The picture in the formats a notebook understands, keyed by MIME type."""
        raise NotImplementedError

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.bundle()


def show(*pictures):
    """Display pictures, one under the other. Works for scenes, charts and diagrams alike."""
    from IPython.display import display
    for picture in pictures:
        display(picture.bundle(), raw=True)


# ------------------------------------------------------------------------------------------
# Scenes
# ------------------------------------------------------------------------------------------
class Scene(Picture):
    """Base class: a list of shapes seen through a camera.

    A subclass overrides shapes() and sets view. Its first docstring line is its caption."""

    view = None                                   # a look.View: where the eye is, what it looks at

    def shapes(self):
        """The shapes of the scene: triangle meshes whose colours already contain the shading."""
        raise NotImplementedError

    def motion(self):
        """What moves with time, or None: dict(labels, seconds, moves, shows). labels has one text for each
        frame; moves[name] is the offset (frames, 3) of a shape from where shapes() puts it; shows[name] is
        how many of its triangles are there at each frame."""
        return None

    @property
    def caption(self):
        return (self.__doc__ or type(self).__name__).strip().split("\n")[0]

    def bundle(self):
        still = Still().draw(self)
        return {"text/html": Canvas().draw(self, still=still),
                "image/png": base64.b64encode(png_bytes(still)).decode("ascii"),
                "text/plain": f"<{type(self).__name__}: {self.caption}>"}


class HelloCube(Scene):
    """A cube above the ground"""

    view = chapter0.VIEW

    def __init__(self, height=chapter0.START[2]):
        self.height = height                      # of the cube's centre, in metres

    def shapes(self):
        return chapter0.build(centre=(0.0, 0.0, self.height))


class FallingCube(HelloCube):
    """The cube falls and comes to rest on the ground"""

    def __init__(self):
        super().__init__(height=chapter0.SIZE / 2)                 # the still picture shows the end of the fall

    def motion(self):
        z0, rest = chapter0.START[2], chapter0.SIZE / 2
        t = np.arange(37) / 60.0
        z = np.maximum(z0 - 0.5 * 9.81 * t ** 2, rest)             # the formula of free fall, not a simulation
        return dict(labels=[f"t = {v:.3f} s" for v in t], seconds=4.0, moves={"cube": np.stack([0 * t, 0 * t, z - rest], axis=1)}, shows={})


class LabScene(Scene):
    """A scene of Chapter 1: shapes and camera come from one function of labs/scenes.py."""

    def build(self):
        """Returns (shapes, view). Overridden by every lab."""
        raise NotImplementedError

    def shapes(self):
        return self.build()[0]

    @property
    def view(self):
        return self.build()[1]


class Terrain(LabScene):
    """A height field as a triangle mesh"""

    def build(self):
        return chapter1.lab01(taller=True)


class ColourMaps(LabScene):
    """One terrain under the rainbow table and under viridis"""

    def build(self):
        return chapter1.lab02()


class DepthCubes(LabScene):
    """Three cubes in front of the depth camera"""

    def build(self):
        return chapter1.lab03()


class Contours(LabScene):
    """Contour lines on the terrain"""

    def build(self):
        return chapter1.lab04()


class Throw(LabScene):
    """Pathlines coloured by speed, and a hedgehog at one instant"""

    def build(self):
        _t, pos, vel = chapter1.lab05_recording()
        return chapter1.lab05(pos, vel, chapter1.LAB05_SNAPSHOT)

    def motion(self):
        _t, pos, _vel = chapter1.lab05_recording()
        start, _ = chapter1.lab05_start()
        frames = np.concatenate([start[None], pos])                # frame 0 is the start, frame f the recording after f steps
        f, every = np.arange(len(frames)), 4
        a, b = pos[:-every:every], pos[every::every]               # the pieces of the pathlines, as lab05_overlay() cuts them
        kept = np.linalg.norm(b - a, axis=2) > 1e-9                # look.tubes() leaves out pieces without length
        drawn = np.concatenate([[0], np.cumsum(kept.sum(axis=1))]) * 12
        size = {s["name"]: len(s["tris"]) for s in self.shapes()}
        assert drawn[-1] == size["pathlines"]
        return dict(labels=[f"t = {v * chapter1.LAB05_DT:.3f} s" for v in f], seconds=5.0,
                    moves={f"cube{k:02d}": frames[:, k] - start[k] for k in range(len(start))},
                    shows={"pathlines": drawn[np.clip((f - 1) // every, 0, len(a))],
                           "hedgehog": np.where(f - 1 >= chapter1.LAB05_SNAPSHOT, size["hedgehog"], 0)})


class InertiaBox(LabScene):
    """A box and the principal axes of its inertia"""

    def build(self):
        return chapter1.lab07()


class BallAndRing(LabScene):
    """A ball above a ring"""

    def build(self):
        return chapter1.lab08()


class Glyphs(LabScene):
    """Cones along the surface normals"""

    def build(self):
        return chapter1.lab09()


class RayFans(LabScene):
    """Two fans of rays in front of two boxes and a thin leg"""

    def build(self):
        return chapter1.lab10()


# ------------------------------------------------------------------------------------------
# Renderers
# ------------------------------------------------------------------------------------------
class Renderer:
    """Base class: something that can draw a scene."""

    def draw(self, scene):
        raise NotImplementedError


def _camera(view):
    """Right, up and forward of a camera with Z up."""
    f = view.target - view.eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r)
    return r, np.cross(r, f), f


# The studio in which the 3D pictures are taken, as a 3D program would set it up: one lamp with soft
# shadows, light from the sky above and from the floor below, a floor that fades into the background,
# and every colour made pastel.
STUDIO = dict(pastel=0.38,                         # how far every colour moves towards white
              key=(0.620, 0.608, 0.583),           # the lamp, a little warm
              sky=(0.505, 0.515, 0.520),           # light from above, a little cool
              ground=(0.300, 0.291, 0.282),        # light thrown back by the floor
              floor=(0.940, 0.935, 0.925),         # the floor, before it is made pastel
              edge=0.016, soft=2)                  # a shadow's edge is this wide (in viewing distances): 2 * soft + 1 cells of the shadow map


def _background():
    """The colour of the floor in full light, which is also the colour behind the scene."""
    base = 1 - (1 - np.array(STUDIO["floor"])) * (1 - STUDIO["pastel"])
    light = np.array(STUDIO["sky"]) + np.array(STUDIO["key"]) * look.LIGHT[2]
    return np.clip(base ** 2.2 * light, 0, 1) ** (1 / 2.2)


def _kind(shape):
    """0 a line or marker that is not lit, 1 a surface that throws shadows, 3 a flat floor piece, which only receives them."""
    if not shape.get("lit", True):
        return 0
    return 1 if np.ptp(shape["points"][:, 2]) > 1e-9 else 3


def _reach(scene, shapes):
    """The middle of some shapes of a scene and how far they reach from it, with everything that moves."""
    motion = scene.motion() or dict(moves={})
    lows, highs = [], []
    for s in shapes:
        p = look.world_points(s)
        shift = np.asarray(motion["moves"].get(s["name"], np.zeros((1, 3))), dtype=float)
        lows.append(p.min(axis=0) + np.minimum(shift.min(axis=0), 0))
        highs.append(p.max(axis=0) + np.maximum(shift.max(axis=0), 0))
    lo, hi = np.min(lows, axis=0), np.max(highs, axis=0)
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2, lo


def _studio(scene):
    """Where the studio stands around a scene: its floor, and its lamp, which is also the camera of the shadow map.
    The lamp looks at the shapes that throw shadows; the map has as many cells as a soft shadow edge needs."""
    shapes = scene.shapes()
    centre, reach, lo = _reach(scene, shapes)
    aim, wide, _ = _reach(scene, [s for s in shapes if _kind(s) == 1] or shapes)
    away = 6.0 * wide
    edge = STUDIO["edge"] * float(np.linalg.norm(scene.view.eye - scene.view.target))
    cells = int(np.clip(round(2.16 * wide * (2 * STUDIO["soft"] + 1) / edge), 128, 1024))
    cell = 2.16 * wide / cells                                      # what one cell of the map covers, in metres
    return dict(centre=centre, reach=reach, floor=float(lo[2] - 0.002 * reach), fade=(1.6 * reach, 3.2 * reach),
                aim=aim, lamp=aim + look.LIGHT * away, half=1.08 * wide / math.sqrt(away ** 2 - wide ** 2),
                near=away - 1.2 * wide, far=away + 1.2 * wide, cells=cells, lift=1.5 * cell, slack=1.0 * cell)


def _normals(shape):
    """The direction each surface faces: one for each triangle, or one for each point of a smooth shape."""
    points, tris = shape["points"], shape["tris"]
    a, b, c = (points[tris[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)                                   # as long as twice the area of its triangle
    if shape["per"] == "vertex":
        total = np.zeros_like(points)
        for k in range(3):
            np.add.at(total, tris[:, k], face)
        face = total
    return face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)


def _surfaces(scene, studio):
    """Every triangle of a scene and of the floor under it: its corners in the world (N, 3, 3) and, for each corner,
    seven numbers: the colour of the surface, the direction it faces, and its kind (0 a line or marker that is
    not lit, 1 a surface, 2 the floor of the studio, 3 a flat floor piece of the scene)."""
    corners, values = [], []
    for s in scene.shapes():
        tris, normal, rgb = s["tris"], _normals(s), np.asarray(s.get("rgb", s["colors"]), dtype=float)
        if s["per"] == "face":
            rgb, normal = np.repeat(rgb[:, None], 3, axis=1), np.repeat(normal[:, None], 3, axis=1)
        else:
            rgb, normal = rgb[tris], normal[tris]
        kind = np.full((len(tris), 3, 1), float(_kind(s)))
        corners.append(look.world_points(s)[tris])
        values.append(np.concatenate([rgb, normal, kind], axis=2))
    (cx, cy, _), z, r = studio["centre"], studio["floor"], studio["fade"][1]
    quad = np.array([[cx - r, cy - r, z], [cx + r, cy - r, z], [cx + r, cy + r, z], [cx - r, cy + r, z]])
    corners.append(quad[[[0, 1, 2], [0, 2, 3]]])
    values.append(np.tile(np.r_[STUDIO["floor"], 0.0, 0.0, 1.0, 2.0], (2, 3, 1)))
    return np.concatenate(corners), np.concatenate(values)


class Still(Renderer):
    """A still picture, drawn with NumPy in three passes, the way a 3D program renders a studio shot.

    1. What the camera sees: each triangle is projected and filled; where triangles overlap the nearest
       wins. Every pixel then knows the colour of its surface, the direction it faces and how far it is.
    2. What the lamp sees: the same, from the lamp. That gives the distance to the first surface in every
       direction, the shadow map.
    3. The light on every pixel: light from the sky and the floor, plus the lamp where the shadow map says
       that nothing stands in between. Shadow edges are soft because the map is asked in 25 places."""

    NEAR = 0.05                                   # metres; nearer parts are cut away
    LARGE = 64                                    # pixels; larger triangles are halved first

    def __init__(self, width=look.WIDTH, height=look.HEIGHT, oversample=2):
        self.width, self.height, self.oversample = width, height, oversample

    def draw(self, scene):
        """Returns the picture as an array (height, width, 3) of bytes."""
        W, H = self.width * self.oversample, self.height * self.oversample
        studio = _studio(scene)
        world, values = _surfaces(scene, studio)
        eye, (r, u, f) = scene.view.eye, _camera(scene.view)
        t = 1.0 / np.tan(np.radians(look.FOV) / 2)
        seen, depth = self._raster(self._from(world, eye, r, u, f), values, t * H / W, t, W, H)
        lamp, (lr, lu, lf) = studio["lamp"], _camera(look.View(studio["lamp"], studio["aim"]))
        solid, cells = values[:, 0, 6] == 1, studio["cells"]
        _, first = self._raster(self._from(world[solid], lamp, lr, lu, lf), np.zeros((int(solid.sum()), 3, 1)),
                                1 / studio["half"], 1 / studio["half"], cells, cells)
        hit = np.flatnonzero(depth > 0)                             # the pixels that show a surface
        z = 1.0 / depth[hit]
        xc = ((hit % W + 0.5) / W - 0.5) * 2 * z / (t * H / W)
        yc = (0.5 - (hit // W + 0.5) / H) * 2 * z / t
        point = eye + xc[:, None] * r + yc[:, None] * u + z[:, None] * f
        colour, normal, kind = seen[hit, :3], seen[hit, 3:6].astype(float), np.rint(seen[hit, 6])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        normal[((eye - point) * normal).sum(axis=1) < 0] *= -1      # a surface is seen from its lit side
        base = 1 - (1 - colour) * (1 - STUDIO["pastel"])            # pastel: every colour part of the way to white
        open_ = self._open(point + normal * studio["lift"], studio, first, lamp, lr, lu, lf)
        facing = np.clip(normal @ look.LIGHT, 0, 1)
        sky, ground, key = (np.array(STUDIO[k]) for k in ("sky", "ground", "key"))
        light = ground + (sky - ground) * (0.5 + 0.5 * normal[:, 2:3]) + key * (facing * open_)[:, None]
        out = np.clip(base ** 2.2 * light, 0, 1) ** (1 / 2.2)
        out[kind == 0] = base[kind == 0]                            # lines and markers keep their colour
        floor, back = kind == 2, _background()
        away = np.hypot(point[floor, 0] - studio["centre"][0], point[floor, 1] - studio["centre"][1])
        w = np.clip((away - studio["fade"][0]) / (studio["fade"][1] - studio["fade"][0]), 0, 1)
        out[floor] += (back - out[floor]) * (w * w * (3 - 2 * w))[:, None]   # the floor fades into the background
        image = np.empty((H * W, 3), np.float32)
        image[:] = back
        image[hit] = out
        s = self.oversample
        image = image.reshape(H // s, s, W // s, s, 3).mean(axis=(1, 3))
        return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def _from(world, eye, r, u, f):
        """Corners in the world as a camera at eye sees them: x to the right, y up, z the distance."""
        rel = world - eye
        return np.stack([rel @ r, rel @ u, rel @ f], axis=-1)

    @staticmethod
    def _open(point, studio, first, lamp, r, u, f):
        """How much of the lamp reaches each point: 1 in the open, 0 in full shadow, in between at a shadow's edge."""
        cells, n = studio["cells"], STUDIO["soft"]
        rel = point - lamp
        z = rel @ f
        col = np.floor((rel @ r / (z * studio["half"]) * 0.5 + 0.5) * cells).astype(np.int64) + n
        row = np.floor((0.5 - rel @ u / (z * studio["half"]) * 0.5) * cells).astype(np.int64) + n
        side = cells + 2 * n
        padded = np.zeros((side, side), np.float32)                 # 1 / distance of what the lamp sees; 0 where it sees nothing
        padded[n:-n, n:-n] = first.reshape(cells, cells)
        padded, limit = padded.ravel(), (z - studio["slack"]).astype(np.float32)
        total = np.zeros(len(point), np.float32)
        for dj in range(-n, n + 1):
            q = np.clip(row + dj, 0, side - 1) * side
            for di in range(-n, n + 1):
                total += padded[q + np.clip(col + di, 0, side - 1)] * limit <= 1.0     # nothing nearer to the lamp than this point
        return total / (2 * n + 1) ** 2

    def _raster(self, cam, values, tx, ty, W, H):
        """Fills triangles given in camera space (N, 3, 3), with values at their corners (N, 3, K), into W by H pixels.
        Returns the values of the nearest surface in each pixel (H * W, K) and 1 / its distance (0: nothing there)."""
        cam, values = self._clip(cam, values.astype(np.float32))
        image = np.zeros((H * W, values.shape[2]), np.float32)
        depth = np.zeros(H * W, np.float32)
        for _ in range(60):                                         # halve large triangles until all are small
            inv = 1.0 / cam[..., 2]
            sx = (cam[..., 0] * tx * inv * 0.5 + 0.5) * W
            sy = (0.5 - cam[..., 1] * ty * inv * 0.5) * H
            x0 = np.clip(np.floor(sx.min(1)), 0, W).astype(int)
            x1 = np.clip(np.ceil(sx.max(1)), -1, W - 1).astype(int)
            y0 = np.clip(np.floor(sy.min(1)), 0, H).astype(int)
            y1 = np.clip(np.ceil(sy.max(1)), -1, H - 1).astype(int)
            on_screen = (x1 >= x0) & (y1 >= y0)
            span = np.maximum(x1 - x0, y1 - y0) + 1
            large = on_screen & (span > self.LARGE)
            if not large.any():
                break
            cam, values = self._halve(cam, values, sx, sy, on_screen, large)
        bounds = (0, 4, 8, 16, 24, 32, 48, self.LARGE)
        for lo, size in zip(bounds[:-1], bounds[1:]):               # triangles of similar size together
            group = np.flatnonzero(on_screen & (span > lo) & (span <= size))
            step = max(1, int(1.0e6 // (size * size)))
            for k in range(0, len(group), step):
                self._fill(group[k:k + step], size, sx, sy, inv, values, x0, y0, x1, y1, W, image, depth)
        return image, depth

    def _clip(self, cam, values):
        """Cut triangles at the near plane (most are in front of it and pass unchanged)."""
        d = cam[..., 2]
        keep = (d >= self.NEAR).all(axis=1)
        cross = ~keep & (d >= self.NEAR).any(axis=1)
        out_cam, out_col = [cam[keep]], [values[keep]]
        for tri, col in zip(cam[cross], values[cross]):
            poly = []
            for i in range(3):
                a, b, ca, cb = tri[i], tri[(i + 1) % 3], col[i], col[(i + 1) % 3]
                if a[2] >= self.NEAR:
                    poly.append((a, ca))
                if (a[2] >= self.NEAR) != (b[2] >= self.NEAR):
                    w = (self.NEAR - a[2]) / (b[2] - a[2])
                    poly.append((a + w * (b - a), ca + w * (cb - ca)))
            for i in range(1, len(poly) - 1):
                out_cam.append(np.array([[poly[0][0], poly[i][0], poly[i + 1][0]]]))
                out_col.append(np.array([[poly[0][1], poly[i][1], poly[i + 1][1]]]))
        return np.concatenate(out_cam), np.concatenate(out_col)

    @staticmethod
    def _halve(cam, values, sx, sy, keep, large):
        """Cut each large triangle in two at the middle of its longest edge on the screen."""
        edge = np.stack([np.hypot(sx[large][:, i] - sx[large][:, (i + 1) % 3],
                                  sy[large][:, i] - sy[large][:, (i + 1) % 3]) for i in range(3)], axis=1)
        first = edge.argmax(axis=1)[:, None]
        order = (first + np.arange(3)) % 3                           # the longest edge runs from corner 0 to 1
        rows = np.arange(len(first))[:, None]
        p, c = cam[large][rows, order], values[large][rows, order]
        pm, cm = (p[:, 0] + p[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2
        halves_p = np.concatenate([np.stack([p[:, 0], pm, p[:, 2]], 1), np.stack([pm, p[:, 1], p[:, 2]], 1)])
        halves_c = np.concatenate([np.stack([c[:, 0], cm, c[:, 2]], 1), np.stack([cm, c[:, 1], c[:, 2]], 1)])
        small = keep & ~large
        return np.concatenate([cam[small], halves_p]), np.concatenate([values[small], halves_c])

    @staticmethod
    def _fill(tri, size, sx, sy, inv, values, x0, y0, x1, y1, W, image, depth):
        """Fill a batch of triangles whose bounding boxes fit in size x size pixels."""
        grid = np.arange(size, dtype=np.float32)
        gx = x0[tri, None, None] + grid[None, None, :]              # pixel columns, (n, 1, size)
        gy = y0[tri, None, None] + grid[None, :, None]              # pixel rows,    (n, size, 1)
        px, py = gx + 0.5, gy + 0.5
        ax, bx, cx = (sx[tri, i].astype(np.float32)[:, None, None] for i in range(3))
        ay, by, cy = (sy[tri, i].astype(np.float32)[:, None, None] for i in range(3))
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        area = np.where(area == 0, np.float32(np.inf), area)
        w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area  # weights of the three corners
        w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area
        w2 = 1.0 - w0 - w1
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                  & (gx <= x1[tri, None, None]) & (gy <= y1[tri, None, None]))
        n, row, col = np.nonzero(inside)
        if len(n) == 0:
            return
        a, b, c = w0[n, row, col], w1[n, row, col], w2[n, row, col]
        ia, ib, ic = (inv[tri[n], i].astype(np.float32) for i in range(3))
        z = a * ia + b * ib + c * ic                                # 1 / distance of this pixel
        rgb = ((a * ia)[:, None] * values[tri[n], 0] + (b * ib)[:, None] * values[tri[n], 1]
               + (c * ic)[:, None] * values[tri[n], 2]) / z[:, None]
        pixel = (y0[tri[n]] + row) * W + (x0[tri[n]] + col)
        order = np.lexsort((z, pixel))                              # by pixel, the nearest candidate last
        pixel, z, rgb = pixel[order], z[order], rgb[order]
        last = np.ones(len(pixel), bool)
        last[:-1] = pixel[1:] != pixel[:-1]
        pixel, z, rgb = pixel[last], z[last], rgb[last]
        nearer = z > depth[pixel]
        depth[pixel[nearer]] = z[nearer]
        image[pixel[nearer]] = rgb[nearer]


def png_bytes(image):
    """An array (height, width, 3) of bytes as a PNG file."""
    from PIL import Image                         # comes with matplotlib
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG", optimize=True)
    return out.getvalue()


class Canvas(Renderer):
    """A canvas for a web page that the reader can turn with the mouse.

    The scene travels inside the page as numbers; a small script draws it with WebGL. If the
    browser cannot do that, the still picture stays."""

    def draw(self, scene, still=None):
        """Returns HTML. still is the picture to show until, or instead of, the canvas."""
        if still is None:
            still = Still().draw(scene)
        h, w = still.shape[0] // 2, still.shape[1] // 2
        small = still[:h * 2, :w * 2].reshape(h, 2, w, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
        picture = base64.b64encode(png_bytes(small)).decode("ascii")
        label = scene.caption.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        data = self.data(scene)
        clock = ('<div class="simscene-time" hidden><button type="button">Play</button>'
                 '<input type="range" min="0" max="1" step="1" value="1" aria-label="Time"><output></output></div>') if "time" in data else ""
        return (f'<figure class="simscene"><style>{_CSS}</style><div class="simscene-stage">'
                f'<img alt="{label}" src="data:image/png;base64,{picture}">'
                f'<canvas hidden tabindex="0" role="img" aria-label="{label}. Drag to turn it, arrow keys also turn it, '
                'plus and minus zoom, 0 resets."></canvas>'
                '<div class="simscene-tools" hidden><button type="button" data-act="in" aria-label="Zoom in">+</button>'
                '<button type="button" data-act="out" aria-label="Zoom out">&minus;</button>'
                f'<button type="button" data-act="reset">Reset view</button></div></div>{clock}'
                f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script></figure>'
                f'<script>{_VIEWER}</script><script>window.simsceneBoot();</script>')

    @staticmethod
    def data(scene):
        """The scene as the viewer reads it: corner positions as 16-bit numbers, triangles, the colour of each surface
        and the direction it faces, and the studio around it (the lamp, the floor, the colours of the light)."""
        shapes, studio = scene.shapes(), _studio(scene)
        points = [look.world_points(s) for s in shapes]
        lo = np.min([p.min(axis=0) for p in points], axis=0)
        size = float(np.max([p.max(axis=0) for p in points], axis=0).__sub__(lo).max())
        code = lambda array: base64.b64encode(array.tobytes()).decode("ascii")   # noqa: E731
        meshes = []
        for s, p in zip(shapes, points):
            wide = len(p) > 65535
            meshes.append(dict(
                name=s["name"], per=s["per"], ntri=int(len(s["tris"])), wide=wide, kind=_kind(s),
                pos=code(np.rint((p - lo) / size * 65535).astype("<u2")),
                idx=code(s["tris"].astype("<u4" if wide else "<u2")),
                col=code(np.rint(np.clip(s.get("rgb", s["colors"]), 0, 1) * 255).astype(np.uint8)),
                nrm=code(np.rint(_normals(s) * 127).astype(np.int8))))
        numbers = lambda values: [float(v) for v in values]                       # noqa: E731
        out = dict(name=type(scene).__name__, eye=numbers(scene.view.eye), target=numbers(scene.view.target), fov=look.FOV,
                   lo=numbers(lo), size=size, meshes=meshes,
                   studio=dict(lamp=numbers(studio["lamp"]), aim=numbers(studio["aim"]), centre=numbers(studio["centre"]), light=numbers(look.LIGHT),
                               half=studio["half"], near=studio["near"], far=studio["far"], lift=studio["lift"], slack=studio["slack"],
                               floor=studio["floor"], fade=numbers(studio["fade"]), back=numbers(_background()),
                               colour=numbers(STUDIO["floor"]), key=numbers(STUDIO["key"]), sky=numbers(STUDIO["sky"]),
                               ground=numbers(STUDIO["ground"]), pastel=STUDIO["pastel"], map=studio["cells"], soft=STUDIO["soft"]))
        motion = scene.motion()
        if motion:
            index = {s["name"]: str(i) for i, s in enumerate(shapes)}
            out["time"] = dict(labels=list(motion["labels"]), seconds=float(motion["seconds"]),
                               move={index[n]: code(np.asarray(v, dtype="<f4")) for n, v in motion["moves"].items()},
                               show={index[n]: [int(c) for c in v] for n, v in motion["shows"].items()})
        return out


class Stage(Renderer):
    """The scene on a USD stage, for Isaac Sim: one mesh prim per shape, a matte material,
    the camera and an even light. The lab scripts make the same call."""

    def __init__(self, stage, theme="light"):
        self.stage, self.theme = stage, theme

    def draw(self, scene):
        """Returns {shape name: prim path}."""
        return look.usd_scene(self.stage, scene.shapes(), scene.view, self.theme)


_TIME_CSS = (".simscene-time{display:flex;align-items:center;gap:.7em;margin:.45em 0 0;font-size:.9em}"
             ".simscene-time[hidden]{display:none}"
             ".simscene-time input{flex:1;min-width:0;margin:0;accent-color:#76B900}"
             ".simscene-time output{min-width:7em;text-align:right;font-variant-numeric:tabular-nums}"
             ".simscene-time button,.simplot-read button{font:inherit;line-height:1.2;padding:.25em .8em;border:1px solid #c9c9c9;"
             "border-radius:6px;background:#fff;color:#1a1a1a;cursor:pointer}"
             '.simscene-time button::before{content:"";display:inline-block;width:0;height:0;margin-right:.5em;vertical-align:-.05em;'
             "border-style:solid;border-width:.38em 0 .38em .62em;border-color:transparent transparent transparent currentColor}"
             ".simscene-time button[data-on]::before{width:.2em;height:.76em;border-width:0 .2em;border-color:currentColor;vertical-align:-.08em}")
_CSS = (".simscene{margin:0 0 .5em}"
        ".simscene-stage{position:relative;border:1px solid #d9d9d9;border-radius:8px;overflow:hidden;background:@@BACK@@}"
        '.simscene-stage::before{content:"";display:block;padding-top:56.25%}'
        ".simscene-stage img,.simscene-stage canvas{position:absolute;left:0;top:0;display:block;width:100%;height:100%;"
        "margin:0;border:0;border-radius:0;max-width:none}"
        ".simscene-stage canvas{cursor:grab;touch-action:pan-y}"
        ".simscene-stage canvas:active{cursor:grabbing}"
        ".simscene [hidden]{display:none !important}"
        ".simscene.active .simscene-stage{border-color:#76B900;box-shadow:0 0 0 2px #76B900}"
        ".simscene-tools{position:absolute;top:10px;right:10px;display:flex;gap:6px}"
        ".simscene-tools button{font:inherit;font-size:13px;line-height:1;min-width:34px;padding:8px 10px;border:1px solid #d9d9d9;"
        "border-radius:6px;background:rgba(255,255,255,.92);color:#1a1a1a;cursor:pointer}"
        ).replace("@@BACK@@", "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in _background())) + _TIME_CSS

# The viewer: WebGL 1, no library. It starts every figure.simscene of the page when it scrolls into view,
# and lights the scene as Still does: a shadow map from the lamp, then the lit picture.
_VIEWER = r"""(function () {
  'use strict';
  if (window.simsceneBoot) return;                      // defined once per page, however many figures there are

  // ---------- data: base64 -> typed arrays -> one flat list of triangles, with the floor of the studio at the end
  function bytes(b64) {
    var s = atob(b64), n = s.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function build(scene) {
    var total = 6, m, i, a, st = scene.studio;
    for (i = 0; i < scene.meshes.length; i++) total += scene.meshes[i].ntri * 3;
    var pos = new Float32Array(total * 3), col = new Uint8Array(total * 4), nrm = new Int8Array(total * 3), o = 0, ranges = [];
    for (i = 0; i < scene.meshes.length; i++) {
      m = scene.meshes[i];
      ranges.push({ start: o, count: m.ntri * 3, kind: m.kind });
      var q = new Uint16Array(bytes(m.pos).buffer);
      var raw = bytes(m.idx).buffer;
      var idx = m.wide ? new Uint32Array(raw) : new Uint16Array(raw);
      var c = bytes(m.col), d = new Int8Array(bytes(m.nrm).buffer), face = m.per === 'face';
      for (var t = 0; t < m.ntri; t++) {
        for (var k = 0; k < 3; k++) {
          var v = idx[t * 3 + k], from = face ? t : v;
          for (a = 0; a < 3; a++) {
            pos[o * 3 + a] = scene.lo[a] + q[v * 3 + a] / 65535 * scene.size;
            col[o * 4 + a] = c[from * 3 + a];
            nrm[o * 3 + a] = d[from * 3 + a];
          }
          col[o * 4 + 3] = m.kind * 85;                    // 0 a line or marker, 1 a surface, 2 the studio floor, 3 a flat floor piece
          o++;
        }
      }
    }
    var r = st.fade[1], x = st.centre[0], y = st.centre[1], z = st.floor;
    var quad = [[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y - r], [x + r, y + r], [x - r, y + r]];
    ranges.push({ start: o, count: 6, kind: 2 });
    for (i = 0; i < 6; i++, o++) {
      pos[o * 3] = quad[i][0]; pos[o * 3 + 1] = quad[i][1]; pos[o * 3 + 2] = z;
      for (a = 0; a < 3; a++) col[o * 4 + a] = Math.round(st.colour[a] * 255);
      col[o * 4 + 3] = 170; nrm[o * 3 + 2] = 127;
    }
    return { pos: pos, col: col, nrm: nrm, count: total, ranges: ranges };
  }

  // ---------- camera: Z is up, the eye orbits the target
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function unit(a) { var l = Math.sqrt(dot(a, a)); return [a[0] / l, a[1] / l, a[2] / l]; }
  function startCamera(scene) {
    var d = sub(scene.eye, scene.target), dist = Math.sqrt(dot(d, d));
    return { target: scene.target.slice(), dist: dist, az: Math.atan2(d[1], d[0]), el: Math.asin(d[2] / dist), fov: scene.fov, size: scene.size };
  }
  function eyeOf(cam) {
    var ce = Math.cos(cam.el);
    return [cam.target[0] + cam.dist * ce * Math.cos(cam.az), cam.target[1] + cam.dist * ce * Math.sin(cam.az), cam.target[2] + cam.dist * Math.sin(cam.el)];
  }
  function axes(cam) {                       // right, up, forward of the camera
    var f = unit(sub(cam.target, eyeOf(cam))), r = unit(cross(f, [0, 0, 1])), u = cross(r, f);
    return { r: r, u: u, f: f };
  }
  function matrix(cam, aspect) {             // projection * view, column-major
    var e = eyeOf(cam), x = axes(cam), r = x.r, u = x.u, f = x.f;
    var near = Math.max(0.01, cam.dist * 0.02), far = cam.dist + 8 * cam.size;
    var t = 1 / Math.tan(cam.fov * Math.PI / 360);
    var A = (far + near) / (near - far), B = 2 * far * near / (near - far);
    var sx = t / aspect, sy = t;
    return new Float32Array([
      sx * r[0], sy * u[0], -A * f[0], f[0],
      sx * r[1], sy * u[1], -A * f[1], f[1],
      sx * r[2], sy * u[2], -A * f[2], f[2],
      -sx * dot(r, e), -sy * dot(u, e), A * dot(f, e) + B, -dot(f, e)]);
  }

  function lampMatrix(st) {                  // the same, for the camera that stands where the lamp is
    var e = st.lamp, f = unit(sub(st.aim, e)), r = unit(cross(f, [0, 0, 1])), u = cross(r, f), s = 1 / st.half;
    var A = (st.far + st.near) / (st.near - st.far), B = 2 * st.far * st.near / (st.near - st.far);
    return new Float32Array([
      s * r[0], s * u[0], -A * f[0], f[0],
      s * r[1], s * u[1], -A * f[1], f[1],
      s * r[2], s * u[2], -A * f[2], f[2],
      -s * dot(r, e), -s * dot(u, e), A * dot(f, e) + B, -dot(f, e)]);
  }

  // ---------- drawing: first what the lamp sees (the shadow map), then what the camera sees, lit
  var HIGH = '#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\n';
  var PACK = 'vec4 pack(float d) { vec4 e = fract(d * vec4(1.0, 255.0, 65025.0, 16581375.0)); return e - e.yzww * vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0); }' +
             'float unpack(vec4 e) { return dot(e, vec4(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0)); }';
  var LAMP_VS = 'attribute vec3 p; attribute vec4 c; uniform mat4 lm; uniform vec3 o; varying float d; varying float k;' +
                'void main() { vec4 q = lm * vec4(p + o, 1.0); d = q.w; k = c.a * 3.0; gl_Position = q; }';
  var LAMP_FS = HIGH + PACK + 'varying float d; varying float k; uniform float far;' +
                'void main() { if (k < 0.5 || k > 1.5) discard; gl_FragColor = pack(clamp(d / far, 0.0, 0.999999)); }';
  var VS = 'attribute vec3 p; attribute vec4 c; attribute vec3 n; uniform mat4 m; uniform vec3 o;' +
           'varying vec3 vc; varying float vk; varying vec3 vn; varying vec3 vp;' +
           'void main() { vec3 w = p + o; vc = c.rgb; vk = c.a * 3.0; vn = n; vp = w; gl_Position = m * vec4(w, 1.0); }';
  function lit(soft) {
    var n = (2 * soft + 1) * (2 * soft + 1);
    return HIGH + PACK + 'varying vec3 vc; varying float vk; varying vec3 vn; varying vec3 vp;' +
      'uniform sampler2D map; uniform mat4 lm; uniform vec3 eye; uniform vec3 lamp; uniform vec3 key; uniform vec3 sky; uniform vec3 ground;' +
      'uniform vec3 back; uniform vec2 centre; uniform vec2 fade; uniform float far; uniform float lift; uniform float slack; uniform float pastel; uniform float cells;' +
      'void main() {' +
      '  vec3 base = 1.0 - (1.0 - vc) * (1.0 - pastel);' +                                      // pastel: every colour part of the way to white
      '  if (vk < 0.5) { gl_FragColor = vec4(base, 1.0); return; }' +                          // lines and markers keep their colour
      '  vec3 N = normalize(vn); if (dot(eye - vp, N) < 0.0) N = -N;' +                        // a surface is seen from its lit side
      '  vec4 q = lm * vec4(vp + N * lift, 1.0); vec2 cell = floor((q.xy / q.w * 0.5 + 0.5) * cells); float open = 0.0;' +
      '  for (int j = -' + soft + '; j <= ' + soft + '; j++) for (int i = -' + soft + '; i <= ' + soft + '; i++) {' +   // the shadow map, asked in several places: soft edges
      '    vec2 t = (cell + vec2(float(i), float(j)) + 0.5) / cells;' +
      '    float first = (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0) ? 1.0 : unpack(texture2D(map, t));' +
      '    if (first > 0.9999 || first * far >= q.w - slack) open += 1.0;' +                   // the lamp sees nothing there, or nothing nearer than this point
      '  }' +
      '  vec3 light = ground + (sky - ground) * (0.5 + 0.5 * N.z) + key * (max(dot(N, lamp), 0.0) * open / ' + n + '.0);' +
      '  vec3 col = pow(clamp(pow(base, vec3(2.2)) * light, 0.0, 1.0), vec3(1.0 / 2.2));' +
      '  if (vk > 1.5 && vk < 2.5) col = mix(col, back, smoothstep(fade.x, fade.y, distance(vp.xy, centre)));' +   // the studio floor fades into the background
      '  gl_FragColor = vec4(col, 1.0);' +
      '}';
  }
  function shader(gl, type, source) {
    var s = gl.createShader(type);
    gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function program(gl, vs, fs) {
    var prog = gl.createProgram();
    gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(prog, 0, 'p'); gl.bindAttribLocation(prog, 1, 'c'); gl.bindAttribLocation(prog, 2, 'n');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    return prog;
  }
  function start(figure) {
    var canvas = figure.querySelector('canvas'), still = figure.querySelector('img');
    var tools = figure.querySelector('.simscene-tools');
    var scene = JSON.parse(figure.querySelector('script[type="application/json"]').textContent);
    var gl = canvas.getContext('webgl', { antialias: true }) || canvas.getContext('experimental-webgl', { antialias: true });
    if (!gl) throw new Error('no WebGL');
    var data = build(scene), cam = startCamera(scene), home = startCamera(scene), queued = false;
    var T = scene.time || null, frame = -1, paths = {}, key;       // frame -1: the scene as it was written down
    if (T) for (key in T.move) paths[key] = new Float32Array(bytes(T.move[key]).buffer);

    var st = scene.studio, cells = st.map, lm = lampMatrix(st), mapped = -2;
    var seen = program(gl, VS, lit(st.soft)), cast = program(gl, LAMP_VS, LAMP_FS);
    function attribute(index, array, size, type, normalised) {
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, size, type, normalised, 0, 0);
    }
    attribute(0, data.pos, 3, gl.FLOAT, false);
    attribute(1, data.col, 4, gl.UNSIGNED_BYTE, true);
    attribute(2, data.nrm, 3, gl.BYTE, true);
    var target = gl.createFramebuffer(), map = gl.createTexture(), deep = gl.createRenderbuffer();   // where the lamp's view is kept
    gl.bindTexture(gl.TEXTURE_2D, map);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, cells, cells, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindRenderbuffer(gl.RENDERBUFFER, deep);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, cells, cells);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, map, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, deep);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('no shadow map');
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    function set(prog, name, value) {
      var at = gl.getUniformLocation(prog, name);
      if (value.length === 16) gl.uniformMatrix4fv(at, false, value);
      else if (value.length === 3) gl.uniform3f(at, value[0], value[1], value[2]);
      else if (value.length === 2) gl.uniform2f(at, value[0], value[1]);
      else gl.uniform1f(at, value);
    }
    gl.useProgram(cast); set(cast, 'lm', lm); set(cast, 'far', st.far);
    gl.useProgram(seen);
    set(seen, 'lm', lm); set(seen, 'lamp', st.light); set(seen, 'key', st.key); set(seen, 'sky', st.sky); set(seen, 'ground', st.ground);
    set(seen, 'back', st.back); set(seen, 'centre', [st.centre[0], st.centre[1]]); set(seen, 'fade', st.fade); set(seen, 'far', st.far);
    set(seen, 'lift', st.lift); set(seen, 'slack', st.slack); set(seen, 'pastel', st.pastel); set(seen, 'cells', cells);
    gl.uniform1i(gl.getUniformLocation(seen, 'map'), 0);

    function all(prog) {                                // every mesh where the time puts it, as much of it as is there yet
      var at = gl.getUniformLocation(prog, 'o');
      if (!T || frame < 0) { gl.uniform3f(at, 0, 0, 0); gl.drawArrays(gl.TRIANGLES, 0, data.count); return; }
      for (var i = 0; i < data.ranges.length; i++) {
        var path = paths[i], n = T.show[i] ? T.show[i][frame] * 3 : data.ranges[i].count;
        if (path) gl.uniform3f(at, path[frame * 3], path[frame * 3 + 1], path[frame * 3 + 2]); else gl.uniform3f(at, 0, 0, 0);
        if (n) gl.drawArrays(gl.TRIANGLES, data.ranges[i].start, n);
      }
    }
    function draw() {
      queued = false;
      if (mapped !== frame) {                           // what the lamp sees changes only when something moves
        gl.bindFramebuffer(gl.FRAMEBUFFER, target);
        gl.viewport(0, 0, cells, cells);
        gl.clearColor(1, 1, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(cast); all(cast);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        mapped = frame;
      }
      var ratio = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(1, Math.round(canvas.clientWidth * ratio)), h = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      gl.clearColor(st.back[0], st.back[1], st.back[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(seen);
      set(seen, 'm', matrix(cam, w / h)); set(seen, 'eye', eyeOf(cam));
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, map);
      all(seen);
    }
    function redraw() { if (!queued) { queued = true; window.requestAnimationFrame(draw); } }
    function zoom(factor) {
      cam.dist = Math.min(home.dist * 6, Math.max(home.dist * 0.15, cam.dist * factor)); redraw();
    }
    function reset() { cam = startCamera(scene); redraw(); }

    // drag turns, Shift-drag (or the right button) moves; touch turns sideways only
    var last = null, active = false;
    canvas.addEventListener('pointerdown', function (e) {
      last = { x: e.clientX, y: e.clientY, move: e.shiftKey || e.button === 2 };
      active = true; figure.classList.add('active');
      if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!last) return;
      var dx = e.clientX - last.x, dy = e.clientY - last.y;
      last.x = e.clientX; last.y = e.clientY;
      if (last.move) {
        var x = axes(cam), k = 2 * cam.dist * Math.tan(cam.fov * Math.PI / 360) / canvas.clientHeight;
        for (var a = 0; a < 3; a++) cam.target[a] += (-dx * x.r[a] + dy * x.u[a]) * k;
      } else {
        cam.az -= dx * 0.008;
        if (e.pointerType !== 'touch') cam.el = Math.min(1.55, Math.max(-1.55, cam.el + dy * 0.008));
      }
      redraw();
    });
    function release() { last = null; }
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('pointerleave', function () { if (!last) { active = false; figure.classList.remove('active'); } });
    canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    canvas.addEventListener('dblclick', reset);
    canvas.addEventListener('wheel', function (e) {     // only after a click, so the page still scrolls
      if (!active && !e.ctrlKey && !e.metaKey) return;
      e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0015));
    }, { passive: false });
    canvas.addEventListener('keydown', function (e) {     // arrows turn, + and - zoom, 0 resets
      var k = e.key, used = true;
      if (k === 'ArrowLeft') cam.az += 0.1; else if (k === 'ArrowRight') cam.az -= 0.1;
      else if (k === 'ArrowUp') cam.el = Math.min(1.55, cam.el + 0.1); else if (k === 'ArrowDown') cam.el = Math.max(-1.55, cam.el - 0.1);
      else if (k === '+' || k === '=') zoom(0.8); else if (k === '-') zoom(1.25); else if (k === '0') reset(); else used = false;
      if (used) { e.preventDefault(); redraw(); }
    });
    tools.addEventListener('click', function (e) {
      var act = e.target.getAttribute('data-act');
      if (act === 'in') zoom(0.8); else if (act === 'out') zoom(1.25); else if (act === 'reset') reset();
    });
    if (window.ResizeObserver) new window.ResizeObserver(redraw).observe(canvas); else window.addEventListener('resize', redraw);

    if (T) (function () {                               // play, pause and a slider for the time
      var bar = figure.querySelector('.simscene-time'), n = T.labels.length, playing = false, t0 = 0, f0 = 0;
      var btn = bar.querySelector('button'), range = bar.querySelector('input'), out = bar.querySelector('output');
      function set(f) { frame = f; range.value = f; out.textContent = T.labels[f]; redraw(); }
      function stop() { playing = false; btn.textContent = 'Play'; btn.removeAttribute('data-on'); }
      function tick(now) {
        if (!playing) return;
        var f = Math.min(n - 1, f0 + Math.floor((now - t0) / (T.seconds * 1000) * (n - 1)));
        if (f !== frame) set(f);
        if (f >= n - 1) return stop();
        window.requestAnimationFrame(tick);
      }
      btn.addEventListener('click', function () {
        if (playing) return stop();
        f0 = frame < 0 || frame >= n - 1 ? 0 : frame;
        if (f0 === 0) set(0);
        playing = true; btn.textContent = 'Pause'; btn.setAttribute('data-on', '1');
        t0 = window.performance.now(); window.requestAnimationFrame(tick);
      });
      range.addEventListener('input', function () { stop(); set(Number(range.value)); });
      range.max = n - 1; range.value = n - 1; out.textContent = T.labels[n - 1];
      bar.hidden = false;
    })();

    still.hidden = true; canvas.hidden = false; tools.hidden = false;
    figure.classList.add('live');
    draw();
  }

  function boot() {
    var figures = document.querySelectorAll('figure.simscene');
    function open(figure) {
      if (figure.getAttribute('data-started')) return;
      figure.setAttribute('data-started', '1');
      try { start(figure); } catch (err) {                // keep the still picture
        var c = figure.querySelector('canvas'), s = figure.querySelector('img'), t = figure.querySelector('.simscene-tools');
        var b = figure.querySelector('.simscene-time');
        if (c) c.hidden = true; if (t) t.hidden = true; if (b) b.hidden = true; if (s) s.hidden = false;
        figure.classList.remove('live');
        if (window.console) window.console.warn('scene viewer: ' + err.message);
      }
    }
    if (window.IntersectionObserver) {                    // start a scene when it scrolls into view
      var watch = new window.IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) { watch.unobserve(en.target); open(en.target); } });
      }, { rootMargin: '300px' });
      Array.prototype.forEach.call(figures, function (f) { watch.observe(f); });
    } else Array.prototype.forEach.call(figures, open);
  }

  window.simsceneBoot = boot;
})();
"""


# ------------------------------------------------------------------------------------------
# Charts and diagrams
# ------------------------------------------------------------------------------------------
GREEN, DARK, GREY, RED, BLUE, PALE = "#76B900", "#1a1a1a", "#5E5E5E", "#d62728", "#1f77b4", "#f4f4f4"


def _fonts(*names):
    """Those of the named typefaces that this computer has, for matplotlib; the last name if it has none."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in names if n in have] or [names[-1]]


def _mono():
    """The typeface for names taken from code, in a still picture."""
    return _fonts("Consolas", "Menlo", "DejaVu Sans Mono")


def _style():
    """How matplotlib draws a still picture: regular text in NVIDIA Sans, mathematics in Computer Modern."""
    return {
        "font.family": "sans-serif", "font.sans-serif": _fonts("NVIDIA Sans", "Helvetica Neue", "Arial", "DejaVu Sans"), "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": True, "font.size": 13,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "legend.frameon": False, "lines.linewidth": 1.8, "svg.hashsalt": "simscene", "svg.fonttype": "path",
    }


class Chart(Picture):
    """Base class of the charts and diagrams. Each one is drawn twice.

    A subclass overrides plot(fig), the still picture drawn with matplotlib, and items(sheet), the
    interactive figure in which every item can be pointed at and selected. Its first docstring line
    is its caption."""

    size = (10.0, 4.6)                            # inches for the still picture; the interactive sheet is 900 pixels wide
    layout = "constrained"

    def plot(self, fig):
        """Draws the still picture on a matplotlib figure."""
        raise NotImplementedError

    def items(self, sheet):
        """Puts the items of the interactive figure on a Sheet."""
        raise NotImplementedError

    @property
    def caption(self):
        return (self.__doc__ or type(self).__name__).strip().split("\n")[0]

    def still(self, dpi=160):
        """The still picture as a PNG file."""
        import matplotlib
        from matplotlib.figure import Figure
        out = io.BytesIO()
        with matplotlib.rc_context(_style()):
            fig = Figure(figsize=self.size, layout=self.layout)
            self.plot(fig)
            fig.savefig(out, format="png", dpi=dpi)
        return out.getvalue()

    def bundle(self):
        return {"text/html": Plot().draw(self),
                "image/png": base64.b64encode(self.still()).decode("ascii"),
                "text/plain": f"<{type(self).__name__}: {self.caption}>"}


class FallChart(Chart):
    """The fall of the cube: the formula, and a loop that steps in time"""

    def plot(self, fig):
        ax = fig.add_subplot()
        g, z0, rest = 9.81, chapter0.START[2], chapter0.SIZE / 2
        t = np.linspace(0, 0.6, 400)
        ax.plot(t, np.maximum(z0 - 0.5 * g * t**2, rest), color=DARK, ls="--", label="formula")
        for rate, colour in ((15, RED), (60, BLUE)):
            rows = vc.drop(z0, rest, 1 / rate, 0.6)
            ax.plot(np.r_[0, rows[:, 0]], np.r_[z0, rows[:, 1]], color=colour, marker="o", ms=3.5,
                    label=f"loop, {rate} steps per second")
        ax.set(xlabel="time (s)", ylabel="height of the centre (m)", xlim=(0, 0.6), ylim=(0, 1.05))
        ax.legend()

    def items(self, sheet):
        k = sheet.panel((78, 62, 800, 330), x="time (s)", y="height of the centre (m)", xrange=(0, 0.6), yrange=(0, 1.05))
        _falls(sheet, k, chapter0.START[2], chapter0.SIZE / 2, 0.6, ((15, RED), (60, BLUE)), "formula", "loop, {} steps per second")
        sheet.legend(k, "top right")


class BrightnessChart(Chart):
    """How bright each lookup table is from its low end to its high end"""

    def plot(self, fig):
        grid = fig.add_gridspec(3, 1, height_ratios=[10, 1, 1])
        ax = fig.add_subplot(grid[0])
        s = np.linspace(0, 1, 256)
        for name, table, colour in (("rainbow", vc.lut_rainbow, RED), ("viridis", vc.lut_viridis, DARK)):
            ax.plot(s, vc.luminance(table(s)), color=colour, label=name)
        ax.set(ylabel="brightness", xlim=(0, 1), ylim=(0, 1))
        ax.tick_params(labelbottom=False)
        ax.legend()
        for row, (name, table) in enumerate((("rainbow", vc.lut_rainbow), ("viridis", vc.lut_viridis)), 1):
            strip = fig.add_subplot(grid[row])
            strip.imshow(table(s)[None], aspect="auto", extent=(0, 1, 0, 1))
            strip.set_yticks([])
            strip.set_ylabel(name, rotation=0, ha="right", va="center")
            strip.grid(False)
            if row == 1:
                strip.tick_params(labelbottom=False)
            else:
                strip.set_xlabel("scalar, from low to high")

    def items(self, sheet):
        s = np.linspace(0, 1, 65)
        tables = (("rainbow", vc.lut_rainbow, RED), ("viridis", vc.lut_viridis, DARK))
        k = sheet.panel((78, 150, 800, 244), y="brightness", xrange=(0, 1), yrange=(0, 1))
        sheet.axis(k, "x").update(showticklabels=False)
        for name, table, colour in tables:
            rgb = np.clip(np.asarray(table(s))[:, :3], 0, 1)
            sheet.curve(k, s, vc.luminance(rgb), name, [formula(f"s = {a:.3f}") + "<br>brightness " + formula(f"Y = {b:.3f}") + f"<br>colour ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
                                                        for a, b, c in zip(s, vc.luminance(rgb), rgb)], name, colour, legend=True)
        sheet.legend(k, "top left")
        cells = (np.arange(64) + 0.5) / 64
        for row, (name, table, _colour) in enumerate(tables):                      # each table as a strip of its colours
            strip = sheet.panel((78, 100 - 46 * row, 800, 26), x="scalar, from low to high", xrange=(0, 1), yrange=(-0.5, 0.5), frame=True if row else "box")
            sheet.axis(strip, "y").update(visible=False)
            sheet.axis(strip, "x").update(showgrid=False)
            sheet.heat(strip, cells, [0], [cells], f"the {name} table", [measure("s", "x", 2)], table, 0.0, 1.0)
            sheet.note(70, 113 - 46 * row, name, anchor="right")


class DepthImages(Chart):
    """One depth image under two lookup tables and two ranges"""

    size = (10.0, 6.0)

    def plot(self, fig):
        depth = chapter1.lab03_depth()
        hit = depth[np.isfinite(depth)]
        lo, hi = float(hit.min()), float(hit.max())
        settings = (("rainbow, fixed range", vc.lut_rainbow, 0.0, 8.0), ("viridis, fixed range", vc.lut_viridis, 0.0, 8.0),
                    ("rainbow, range of this frame", vc.lut_rainbow, lo, hi), ("viridis, range of this frame", vc.lut_viridis, lo, hi))
        for k, (title, table, near, far) in enumerate(settings, 1):
            ax = fig.add_subplot(2, 2, k)
            ax.imshow(vc.depth_to_rgb(depth, near, far, table))
            ax.set_title(title, fontsize=13)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            for side in ax.spines.values():
                side.set(visible=True, color="#bbbbbb")

    def items(self, sheet):
        depth = chapter1.lab03_depth(128, 72)
        hit = depth[np.isfinite(depth)]
        low, high = float(hit.min()), float(hit.max())
        settings = (("rainbow, fixed range", vc.lut_rainbow, 0.0, 8.0), ("viridis, fixed range", vc.lut_viridis, 0.0, 8.0),
                    ("rainbow, range of this frame", vc.lut_rainbow, low, high), ("viridis, range of this frame", vc.lut_viridis, low, high))
        rows, cols = depth.shape
        for n, (title, table, near, far) in enumerate(settings):
            k = sheet.panel((30 + 430 * (n % 2), 285 - 265 * (n // 2), 410, 230.6), xrange=(-0.5, cols - 0.5), yrange=(rows - 0.5, -0.5), frame="box", title=title)
            sheet.heat(k, np.arange(cols), np.arange(rows), np.where(np.isfinite(depth), depth, np.nan), title,
                       [("pixel column ", "x", 0, ""), ("row ", "y", 0, ""), measure("d", "z", 2, "m")], table, near, far)


class ContourChart(Chart):
    """Marching squares on the terrain, and on a saddle with one ambiguous cell"""

    size = (10.0, 4.9)

    def plot(self, fig):
        left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        X, Y, Z = vc.make_terrain(65)
        left.pcolormesh(X[:, 0], Y[0, :], Z.T, cmap="viridis", shading="nearest", rasterized=True)
        for segments, _ambiguous in chapter1.lab04_segments().values():
            for a, b in segments:
                left.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=1.4)
        left.set_title("the terrain from above, five levels", fontsize=13)
        left.set(xlabel="$x$ (m)", ylabel="$y$ (m)", aspect="equal")
        _count, _correct, segments, g, field = vc.saddle_check()
        limit = float(np.abs(field).max())
        right.pcolormesh(g, g, field.T, cmap="RdBu", vmin=-limit, vmax=limit, shading="nearest", edgecolors="white", linewidth=0.5)
        for a, b in segments:
            right.plot([a[0], b[0]], [a[1], b[1]], color=DARK, lw=2.5)
        mid = len(g) // 2
        right.plot([g[mid - 1], g[mid], g[mid], g[mid - 1], g[mid - 1]], [g[mid - 1], g[mid - 1], g[mid], g[mid], g[mid - 1]],
                   color=GREEN, lw=2.5)
        right.set_title("a saddle, the ambiguous cell in green", fontsize=13)
        right.set(xlabel="$x$", ylabel="$y$", aspect="equal")
        for ax in (left, right):
            ax.grid(False)

    def items(self, sheet):
        import matplotlib
        X, Y, Z = vc.make_terrain(65)
        left = sheet.panel((70, 62, 340, 340), x=formula("x") + " (m)", y=formula("y") + " (m)", xrange=(-2.03, 2.03), yrange=(-2.03, 2.03), title="the terrain from above, five levels")
        sheet.heat(left, X[:, 0], Y[0, :], Z.T, "terrain", [measure("x", "x", 2, "m"), measure("y", "y", 2, "m"), ("height ", "z", 3, " m")],
                   vc.lut_viridis, 0.0, float(Z.max()))
        for level, (segments, _ambiguous) in chapter1.lab04_segments().items():
            xs, ys = _broken(segments)
            sheet.curve(left, xs, ys, f"contour at {level} m", f"height {level} m<br>{len(segments)} segments", f"level{level}", "white", width=1.8)
        _count, _correct, segments, g, field = vc.saddle_check()
        half, limit = (g[1] - g[0]) / 2, float(np.abs(field).max())
        right = sheet.panel((520, 62, 340, 340), x=formula("x"), y=formula("y"), xrange=(g[0] - half, g[-1] + half), yrange=(g[0] - half, g[-1] + half),
                            title="a saddle, the ambiguous cell in green")
        sheet.heat(right, g, g, field.T, "saddle", [measure("x", "x", 2), measure("y", "y", 2), measure("F", "z", 3)],
                   matplotlib.colormaps["RdBu"], -limit, limit, gap=1)
        xs, ys = _broken(segments)
        sheet.curve(right, xs, ys, "contour of the saddle", formula("F = 0") + f"<br>{len(segments)} segments", "zero", DARK, width=3.0)
        m = len(g) // 2
        corners = [(m - 1, m - 1), (m, m - 1), (m, m), (m - 1, m)]
        sheet.area(right, [g[i] for i, _j in corners], [g[j] for _i, j in corners], "the ambiguous cell",
                   "corners " + ", ".join(formula(f"F = {field[i, j]:+.3f}") for i, j in corners) + "<br>two corners above 0 and two below, on opposite sides",
                   "cell", "rgba(118,185,0,0)", GREEN, width=3.0)


class IntegrationChart(Chart):
    """One turn around the circle with two methods, and the error against the step"""

    size = (10.0, 4.6)

    def plot(self, fig):
        left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        names, colours = {"euler": "Euler", "rk2": "Runge-Kutta 2"}, {"euler": RED, "rk2": GREEN}
        a = np.linspace(0, 2 * np.pi, 200)
        left.plot(np.cos(a), np.sin(a), color=DARK, ls="--", label="exact")
        for _dt, method, _radius, path in vc.circle_test((0.1,)):
            left.plot(path[:, 0], path[:, 1], color=colours[method], marker="o", ms=2.5, label=names[method])
        left.set_title("the paths, step 0.1", fontsize=13)
        left.set(xlabel="$x$", ylabel="$y$", aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
        rows = vc.circle_test((0.2, 0.1, 0.05, 0.025))
        for method in ("euler", "rk2"):
            steps = [r[0] for r in rows if r[1] == method]
            errors = [abs(r[2] - 1) for r in rows if r[1] == method]
            right.plot(steps, errors, color=colours[method], marker="o", ms=4, label=names[method])
        right.set_title("distance from the circle after one turn", fontsize=13)
        right.set(xscale="log", yscale="log", xlabel="step", ylabel="error")
        right.set_xticks([0.025, 0.05, 0.1, 0.2], labels=["0.025", "0.05", "0.1", "0.2"])
        right.minorticks_off()
        right.legend()

    def items(self, sheet):
        names, colours = {"euler": "Euler", "rk2": "Runge-Kutta 2"}, {"euler": RED, "rk2": GREEN}
        left = sheet.panel((70, 62, 318, 318), x=formula("x"), y=formula("y"), xrange=(-1.6, 1.6), yrange=(-1.6, 1.6), title="the paths, step 0.1")
        a = np.linspace(0, 2 * np.pi, 121)
        sheet.curve(left, np.cos(a), np.sin(a), "exact", "a circle of radius 1", "exact", DARK, dash="dash")
        paths = {method: path for _dt, method, _radius, path in vc.circle_test((0.1,))}
        steps = np.arange(len(paths["euler"]))
        for method, path in paths.items():
            sheet.curve(left, path[:, 0], path[:, 1], names[method], [f"step {i}<br>" + formula(f"x = {p[0]:.3f}") + ", " + formula(f"y = {p[1]:.3f}") + f"<br>radius {np.hypot(*p):.4f}"
                                                                      for i, p in enumerate(path)], method, colours[method], markers=5)
        moving = [("exact", "exact", DARK, np.stack([np.cos(0.1 * steps), np.sin(0.1 * steps)], 1))] + [(names[m], m, colours[m], paths[m]) for m in paths]
        for name, group, colour, path in moving:                                    # the particle of each method, step by step
            sheet.cursor(left, path[-1, 0], path[-1, 1], f"the particle, {name}", group, colour, path[:, 0] - path[-1, 0], path[:, 1] - path[-1, 1],
                         [f"step {i}<br>" + formula(f"x = {p[0]:.3f}") + ", " + formula(f"y = {p[1]:.3f}") + f"<br>radius {np.hypot(*p):.4f}" for i, p in enumerate(path)])
        right = sheet.panel((530, 62, 350, 318), x="step", y="error", xrange=(0.02, 0.25), yrange=(5e-6, 2.0), log="xy",
                            title="distance from the circle after one turn")
        sheet.axis(right, "x").update(tickmode="array", tickvals=[0.025, 0.05, 0.1, 0.2], ticktext=["0.025", "0.05", "0.1", "0.2"], dtick=None)
        rows = vc.circle_test((0.2, 0.1, 0.05, 0.025))
        for method in ("euler", "rk2"):
            mine = [r for r in rows if r[1] == method]
            sheet.curve(right, [r[0] for r in mine], [abs(r[2] - 1) for r in mine], names[method],
                        [f"step {r[0]}<br>radius after one turn {r[2]:.5f}<br>error {abs(r[2] - 1):.5f}" for r in mine], method, colours[method], markers=8, legend=True)
        sheet.legend(right, "bottom right")
        sheet.clock([f"step {i} of {steps[-1]}" for i in steps], seconds=6.0)


class TimestepChart(Chart):
    """The same drop at three time steps, against the exact curve"""

    def plot(self, fig):
        ax = fig.add_subplot()
        z0, rest, duration = chapter1.LAB06_Z0, chapter1.LAB06_SIZE / 2, chapter1.LAB06_DURATION
        t = np.linspace(0, duration, 400)
        ax.plot(t, np.maximum(z0 - 0.5 * 9.81 * t**2, rest), color=DARK, ls="--", label="exact")
        for rate, colour in zip(chapter1.LAB06_RATES, (GREEN, BLUE, RED)):
            rows = vc.drop(z0, rest, 1.0 / rate, duration)
            ax.plot(np.r_[0, rows[:, 0]], np.r_[z0, rows[:, 1]], color=colour, marker="o", ms=3,
                    label=f"{rate} steps per second")
        ax.set(xlabel="time (s)", ylabel="height of the centre (m)", xlim=(0, duration), ylim=(0, 2.1))
        ax.legend()

    def items(self, sheet):
        z0, rest, duration = chapter1.LAB06_Z0, chapter1.LAB06_SIZE / 2, chapter1.LAB06_DURATION
        k = sheet.panel((78, 62, 800, 330), x="time (s)", y="height of the centre (m)", xrange=(0, duration), yrange=(0, 2.1))
        _falls(sheet, k, z0, rest, duration, tuple(zip(chapter1.LAB06_RATES, (GREEN, BLUE, RED))), "exact", "{} steps per second", seconds=5.0)
        sheet.legend(k, "top right")


class FieldChart(Chart):
    """A part described by one function: the field, and a cut through it"""

    size = (10.0, 4.9)

    def plot(self, fig):
        xs = np.linspace(-1.5, 1.5, 121)
        part = vc.demo_part(xs)
        left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        limit = float(np.abs(part).max())
        left.imshow(part.T, origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap="RdBu", vmin=-limit, vmax=limit)
        for level, colour, label in ((0.0, DARK, "surface"), (0.1, GREEN, "offset by 10 cm")):
            segments, _ = vc.marching_squares(part, level, xs, xs)
            for k, (a, b) in enumerate(segments):
                left.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=2.0, label=label if k == 0 else None)
        row = 45                                                    # the cut, at y = -0.375 m
        left.axhline(xs[row], color=GREY, lw=1.0, ls=":")
        left.set_title("the field: red inside, blue outside", fontsize=13)
        left.set(xlabel="$x$ (m)", ylabel="$y$ (m)", aspect="equal")
        left.grid(False)
        right.plot(xs, part[:, row], color=DARK)
        right.axhline(0.0, color=DARK, lw=1.0)
        right.axhline(0.1, color=GREEN, lw=1.0)
        right.fill_between(xs, part[:, row], 0, where=part[:, row] < 0, color=RED, alpha=0.25)
        right.set_title("the field along the dotted line", fontsize=13)
        right.set(xlabel="$x$ (m)", ylabel="distance to the surface (m)", xlim=(xs[0], xs[-1]))

    def items(self, sheet):
        import matplotlib
        xs = np.linspace(-1.5, 1.5, 121)
        part, row = vc.demo_part(xs), 45
        limit = float(np.abs(part).max())
        left = sheet.panel((70, 62, 340, 340), x=formula("x") + " (m)", y=formula("y") + " (m)", xrange=(xs[0], xs[-1]), yrange=(xs[0], xs[-1]), title="the field: red inside, blue outside")
        sheet.heat(left, xs, xs, part.T, "field", [measure("x", "x", 2, "m"), measure("y", "y", 2, "m"), measure("F", "z", 3, "m")],
                   matplotlib.colormaps["RdBu"], -limit, limit)
        for level, colour, name, group in ((0.0, DARK, "surface", "surface"), (0.1, GREEN, "offset by 10 cm", "offset")):
            segments, _ = vc.marching_squares(part, level, xs, xs)
            length = float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum())
            bx, by = _broken(segments)
            sheet.curve(left, bx, by, name, formula(f"F = {level}") + f" m<br>{len(segments)} segments, {length:.2f} m long", group, colour, width=2.2)
        cut = np.linspace(xs[0], xs[-1], 31)
        sheet.curve(left, cut, np.full(31, xs[row]), "the cut", "along " + formula(f"y = {xs[row]:.3f}") + " m", "cut", GREY, dash="dot", width=1.6)
        profile = part[:, row]
        right = sheet.panel((530, 62, 350, 340), x=formula("x") + " (m)", y="distance to the surface (m)", xrange=(xs[0], xs[-1]),
                            yrange=(float(profile.min()) - 0.04, float(profile.max()) + 0.04), title="the field along the dotted line")
        inside = profile < 0
        edges = np.flatnonzero(np.diff(np.r_[False, inside, False].astype(int)))    # where runs of "inside" begin and end
        for n, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            sheet.area(right, np.r_[xs[a], xs[a:b], xs[b - 1]], np.r_[0.0, profile[a:b], 0.0], "inside the part",
                       "from " + formula(f"x = {xs[a]:.2f}") + " m to " + formula(f"x = {xs[b - 1]:.2f}") + " m", f"inside{n}", "rgba(214,39,40,0.25)", "rgba(214,39,40,0)", width=0)
        sheet.curve(right, xs[[0, -1]], [0.0, 0.0], "surface", formula("F = 0"), "surface", DARK, width=1.4)
        sheet.curve(right, xs[[0, -1]], [0.1, 0.1], "offset by 10 cm", formula("F = 0.1") + " m", "offset", GREEN, width=1.4)
        sheet.curve(right, xs, profile, "the field along the cut", [formula(f"x = {a:.3f}") + " m<br>" + formula(f"F = {b:+.3f}") + " m<br>" + ("inside" if b < 0 else "outside") + " the part"
                                                                    for a, b in zip(xs, profile)], "cut", DARK)


class ProbeChart(Chart):
    """Two fans of rays seen from above: what each one hits"""

    size = (10.0, 4.8)

    def plot(self, fig):
        origin, boxes = np.array(chapter1.LAB10_ORIGIN[:2]), chapter1.LAB10_BOXES
        for k, (n, *_rest) in enumerate(chapter1.LAB10_FANS, 1):
            ax = fig.add_subplot(1, 2, k)
            ax.set_axisbelow(True)
            directions = vc.ray_fan(n, chapter1.LAB10_FOV)[:, :2]
            distance = chapter1.lab10_reference(n)
            reach = np.where(np.isfinite(distance), distance, 6.5)
            ends = origin + directions * reach[:, None]
            for end in ends:
                ax.plot([origin[0], end[0]], [origin[1], end[1]], color=GREEN, lw=0.7)
            for cx, cy, hx, hy in boxes:
                ax.fill([cx - hx, cx + hx, cx + hx, cx - hx], [cy - hy, cy - hy, cy + hy, cy + hy], color="#9a9a9a")
            hit = np.isfinite(distance)
            ax.plot(ends[hit, 0], ends[hit, 1], ls="", marker="o", ms=3.5, color=DARK)
            ax.set_title(f"{n} rays, {int(hit.sum())} hits", fontsize=13)
            ax.set(xlabel="$x$ (m)", aspect="equal", xlim=(-3.2, 3.8), ylim=(-3.6, 3.6))
            if k == 1:
                ax.set_ylabel("$y$ (m)")

    def items(self, sheet):
        origin, boxes = np.array(chapter1.LAB10_ORIGIN[:2]), chapter1.LAB10_BOXES
        for p, (n, *_rest) in enumerate(chapter1.LAB10_FANS):
            directions, distance = vc.ray_fan(n, chapter1.LAB10_FOV)[:, :2], chapter1.lab10_reference(n)
            hit = np.isfinite(distance)
            ends = origin + directions * np.where(hit, distance, 6.5)[:, None]
            k = sheet.panel((62 + 440 * p, 58, 330.5, 340), x=formula("x") + " (m)", y=formula("y") + " (m)" if p == 0 else "", xrange=(-3.2, 3.8), yrange=(-3.6, 3.6),
                            title=f"{n} rays, {int(hit.sum())} hits")
            for b, (cx, cy, hx, hy) in enumerate(boxes):
                name = "thin leg" if b == len(boxes) - 1 else f"box {b + 1}"
                sheet.box(k, cx - hx, cy - hy, 2 * hx, 2 * hy, name, f"centre at ({cx}, {cy}) m<br>{2 * hx:.2f} m by {2 * hy:.2f} m<br>{cx - origin[0]:.1f} m in front of the sensor",
                          f"fan{n}box{b}", "#9a9a9a", "#9a9a9a", width=1.0, small=b == len(boxes) - 1)
            for i, (end, d, way) in enumerate(zip(ends, distance, directions)):
                angle = math.degrees(math.atan2(way[1], way[0]))
                ray = sheet.curve(k, [origin[0], end[0]], [origin[1], end[1]], f"ray {i + 1} of {n}",
                                  f"angle {angle:+.1f} degrees<br>" + (f"hits at {d:.2f} m" if hit[i] else "hits nothing"), f"fan{n}/ray{i}", GREEN, width=1.0,
                                  markers=1)
                ray["marker"].update(color=DARK, size=[0] * (len(ray["x"]) - 1) + [7 if hit[i] else 0])     # a dot where the ray hits
            sheet.dots(k, [origin[0]], [origin[1]], "sensor", [f"at ({origin[0]:.0f}, {origin[1]:.0f}) m<br>a fan of {n} rays, {chapter1.LAB10_FOV:.0f} degrees wide"], f"fan{n}", DARK, size=9)


class ReductionChart(Chart):
    """Subsampling a terrain: a profile at each stride, and the error it costs"""

    def plot(self, fig):
        rows = vc.reduction_table()
        ys = np.linspace(-2.0, 2.0, 129)
        j = int(np.argmin(np.abs(ys - 0.5)))
        left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        for (stride, _points, _quads, _worst, _rms, X, Z), colour in zip(rows, (DARK, BLUE, GREEN, "#9467bd", "#ff7f0e", RED)):
            left.plot(X[:, 0], Z[:, j // stride], color=colour, marker="o" if stride > 2 else None, ms=3, label=f"stride {stride}")
        left.set(xlabel="$x$ (m)", ylabel="height (m)")
        left.legend(fontsize=11)
        right.plot([r[1] for r in rows], [r[3] for r in rows], color=DARK, marker="o", ms=4)
        right.set(xscale="log", xlabel="points kept", ylabel="largest height error (m)")

    def items(self, sheet):
        rows = vc.reduction_table()
        j = int(np.argmin(np.abs(np.linspace(-2.0, 2.0, 129) - 0.5)))
        colours = (DARK, BLUE, GREEN, "#9467bd", "#ff7f0e", RED)
        left = sheet.panel((70, 62, 360, 318), x=formula("x") + " (m)", y="height (m)", xrange=(-2.1, 2.1), yrange=(-0.02, 0.56), title="one profile at each stride")
        for (stride, _points, _quads, _worst, _rms, X, Z), colour in zip(rows, colours):
            xs, zs = X[:, 0], Z[:, j // stride]
            sheet.curve(left, xs, zs, f"stride {stride}", [formula(f"x = {a:.2f}") + " m<br>height " + formula(f"z = {b:.3f}") + " m" for a, b in zip(xs, zs)],
                        f"stride{stride}", colour, markers=6 if stride > 2 else 0, legend=True)
        sheet.legend(left, "top left")
        right = sheet.panel((540, 62, 340, 318), x="points kept", y="largest height error (m)", xrange=(15, 30000), yrange=(-0.015, 0.32), log="x",
                            title="what each stride costs")
        line = sheet.curve(right, [r[1] for r in rows], [r[3] for r in rows], "error", "", None, GREY, width=1.4)
        line.update(hoverinfo="skip", hovertemplate=None)
        for (stride, points, _quads, worst, rms, _X, _Z), colour in zip(rows, colours):
            sheet.dots(right, [points], [worst], f"stride {stride}", [f"{points} points<br>largest error {100 * worst:.2f} cm<br>rms error {100 * rms:.2f} cm"],
                       f"stride{stride}", colour, size=11)


class Diagram(Chart):
    """Base class of the diagrams: boxes, arrows and text on a 16 by 9 sheet."""

    size = (10.0, 5.625)
    layout = None
    H1, H2, BODY, SMALL, MONO = 15, 13, 11.5, 10, 9.5        # the only text sizes

    def sheet(self, fig):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 16), ylim=(0, 9))
        ax.axis("off")
        return ax

    @staticmethod
    def box(ax, x, y, w, h, fill="white", edge=DARK):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec=edge, lw=1.6, zorder=1))

    @staticmethod
    def arrow(ax, start, end, colour=DARK, style="-|>"):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=18, lw=2.2, color=colour, zorder=2,
                                     shrinkA=0, shrinkB=0))


class SetupDiagram(Diagram):
    """Where the picture is made: the Spark simulates and renders, the Mac shows it"""

    def plot(self, fig):
        ax = self.sheet(fig)
        xl, xr, w, y0, h = 0.6, 10.2, 5.2, 2.7, 4.7
        boxes = ((xl, "DGX Spark", ["aarch64, DGX OS 7", "USD stage: the scene", "PhysX: moves it", "RTX: draws every frame",
                                    "encodes the frames as video"], True),
                 (xr, "MacBook", ["no simulation, no rendering", "terminal and editor over SSH", "WebRTC Streaming Client",
                                  "shows the video", "sends mouse and keyboard"], False))
        for x, title, lines, filled in boxes:
            self.box(ax, x, y0, w, h, fill=GREEN if filled else "white", edge=GREEN if filled else DARK)
            ax.text(x + w / 2, y0 + h - 0.55, title, ha="center", va="center", fontsize=self.H1,
                    color="white" if filled else DARK)
            for i, line in enumerate(lines):
                ax.text(x + w / 2, y0 + h - 1.45 - i * 0.62, line, ha="center", va="center", fontsize=self.BODY,
                        color="white" if filled else GREY)
        for y, label, colour, style in ((6.15, "SSH, TCP 22", GREY, "<|-"), (4.91, "signalling, TCP 49100", GREEN, "<|-|>"),
                                        (3.67, "video and input, UDP 47998", GREEN, "<|-|>")):
            self.arrow(ax, (xl + w + 0.2, y), (xr - 0.2, y), colour, style)
            ax.text(8, y + 0.36, label, ha="center", va="center", fontsize=self.SMALL, color=DARK)
        ax.text(8, 1.6, "Same LAN or VPN.  One client per instance.  No password on the stream: trusted networks only.",
                ha="center", va="center", fontsize=self.BODY, color=GREY)

    def items(self, sheet):
        k = sheet.panel((0, 0, sheet.W, sheet.h), xrange=(0, 16), yrange=(0, 9), frame=False)
        xl, xr, w, y0, h = 0.6, 10.2, 5.2, 2.7, 4.7
        machines = (
            ("spark", xl, "DGX Spark", "simulates the scene and draws every frame", GREEN, GREEN, "white", "white", (
                ("aarch64, DGX OS 7", "the processor and the system of the Spark"),
                ("USD stage: the scene", "a tree of prims with attributes<br>saved as out/hello_cube.usda in Step 3"),
                ("PhysX: moves it", "the physics engine<br>steps of 1/60 s in Step 5"),
                ("RTX: draws every frame", "the renderer, on the GPU of the Spark<br>1280 by 720 pixels in Step 4"),
                ("encodes the frames as video", "the picture leaves the Spark as video, not as geometry"))),
            ("mac", xr, "MacBook", "shows the picture and sends your input", "white", DARK, DARK, GREY, (
                ("no simulation, no rendering", "the Mac computes nothing of the scene"),
                ("terminal and editor over SSH", "where the commands marked # Spark are typed"),
                ("WebRTC Streaming Client", "the viewer application of Isaac Sim<br>one client for each running instance"),
                ("shows the video", "decodes the stream that arrives on UDP port 47998"),
                ("sends mouse and keyboard", "your input travels back on the same stream"))))
        for group, x, title, role, fill, edge, ink, soft, rows in machines:
            sheet.box(k, x, y0, w, h, title, role, group, fill, edge, radius=0.12)
            sheet.label(k, x + w / 2, y0 + h - 0.55, title, group, size=19, colour=ink)
            for i, (line, meaning) in enumerate(rows):
                y = y0 + h - 1.45 - i * 0.62
                sheet.box(k, x + 0.2, y - 0.29, w - 0.4, 0.58, line, meaning, f"{group}/{i}", "rgba(255,255,255,0)", "rgba(255,255,255,0)", width=0)
                sheet.label(k, x + w / 2, y, line, f"{group}/{i}", size=14.5, colour=soft)
        links = (("ssh", 6.15, "SSH, TCP 22", "TCP port 22<br>commands and files, started from the Mac", GREY, False, (xr - 0.2, xl + w + 0.2)),
                 ("signal", 4.91, "signalling, TCP 49100", "TCP port 49100<br>sets the stream up, in both directions", GREEN, True, (xl + w + 0.2, xr - 0.2)),
                 ("video", 3.67, "video and input, UDP 47998", "UDP port 47998<br>video frames to the Mac, mouse and keyboard back", GREEN, True, (xl + w + 0.2, xr - 0.2)))
        for group, y, text, meaning, colour, both, (a, b) in links:
            sheet.link(k, (a, y), (b, y), text, meaning, group, colour, both=both)
            sheet.label(k, 8, y + 0.36, text, group, size=12.5)
        sheet.label(k, 8, 1.6, "Same LAN or VPN.  One client per instance.  No password on the stream: trusted networks only.", size=14, colour=GREY)


class DatasetDiagram(Diagram):
    """The three parts of a dataset, on a small mesh"""

    def plot(self, fig):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon
        ax = self.sheet(fig)
        n, ox, oy, s = 4, 1.1, 2.1, 1.45
        at = lambda i, j: (ox + i * s, oy + j * s)                   # noqa: E731
        ax.add_patch(Polygon([at(0, 0), at(1, 0), at(1, 1)], fc=GREEN, alpha=0.30, ec="none", zorder=0))
        for i in range(n):
            ax.add_line(Line2D([at(i, 0)[0]] * 2, [at(0, 0)[1], at(0, n - 1)[1]], color=GREY, lw=1.3, zorder=1))
            ax.add_line(Line2D([at(0, 0)[0], at(n - 1, 0)[0]], [at(0, i)[1]] * 2, color=GREY, lw=1.3, zorder=1))
        for i in range(n - 1):
            for j in range(n - 1):
                ax.add_line(Line2D([at(i, j)[0], at(i + 1, j + 1)[0]], [at(i, j)[1], at(i + 1, j + 1)[1]], color=GREY, lw=0.8, zorder=1))
        for i in range(n):
            for j in range(n):
                ax.add_patch(Circle(at(i, j), 0.27, fc=GREEN, ec=DARK, lw=1, zorder=3))
                ax.text(*at(i, j), str(i * n + j), ha="center", va="center", fontsize=self.MONO, color="white", zorder=4)
        cx = ox + 1.5 * s
        ax.text(cx, 7.55, "a 4 by 4 height field as a mesh", ha="center", va="center", fontsize=self.BODY, color=GREY)
        ax.text(cx, 1.15, "shaded face: points 0, 4, 5", ha="center", va="center", fontsize=self.BODY, color=DARK)
        X, Y, Z = vc.make_terrain(n)
        c = np.clip(np.asarray(vc.lut_viridis(Z / Z.max()))[..., :3], 0, 1)[0, 0]
        rows = (("GEOMETRY", "where each point is: 16 points", "points",
                 f"[({X[0, 0]:.1f}, {Y[0, 0]:.1f}, {Z[0, 0]:.2f}), ({X[0, 1]:.1f}, {Y[0, 1]:.1f}, {Z[0, 1]:.2f}), ...]"),
                ("TOPOLOGY", "who is connected: 18 triangles", "faceVertexCounts, faceVertexIndices", "[3, 3, 3, ...]   [0, 4, 5,  0, 5, 1, ...]"),
                ("ATTRIBUTES", "data carried by the points", "primvars:displayColor, primvars:elevation",
                 f"[({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), ...]   [{Z[0, 0]:.2f}, {Z[0, 1]:.2f}, ...]"))
        bx, bw, bh, gap = 7.3, 8.1, 2.0, 0.3
        for k, (head, what, names, example) in enumerate(rows):
            y = 7.9 - (k + 1) * bh - k * gap
            self.box(ax, bx, y, bw, bh)
            ax.text(bx + 0.3, y + bh - 0.42, head, va="center", fontsize=self.H2, color="#5f9400")
            ax.text(bx + 3.0, y + bh - 0.42, what, va="center", fontsize=self.SMALL, color=GREY)
            ax.text(bx + 0.3, y + bh - 1.02, names, va="center", fontsize=self.BODY, family=_mono(), color=DARK)
            ax.text(bx + 0.3, y + bh - 1.60, example, va="center", fontsize=self.MONO, family=_mono(), color=GREY)

    def items(self, sheet):
        k = sheet.panel((0, 0, sheet.W, sheet.h), xrange=(0, 16), yrange=(0, 9), frame=False)
        n, ox, oy, s = 4, 1.1, 2.1, 1.45
        at = lambda i, j: (ox + i * s, oy + j * s)                                  # noqa: E731
        X, Y, Z = vc.make_terrain(n)
        colours = np.clip(np.asarray(vc.lut_viridis(Z / Z.max()))[..., :3], 0, 1)
        faces = [f for i in range(n - 1) for j in range(n - 1)
                 for f in (((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1)))]
        for f, corners in enumerate(faces):                                          # topology: which points make a face
            ids = [i * n + j for i, j in corners]
            sheet.area(k, [at(*c)[0] for c in corners], [at(*c)[1] for c in corners], f"face {f}",
                       f"points {ids[0]}, {ids[1]}, {ids[2]}<br>entries {3 * f} to {3 * f + 2} of " + code("faceVertexIndices"), f"face{f}",
                       "rgba(118,185,0,0.30)" if f == 0 else "rgba(118,185,0,0)", GREY, width=1.1, also=[f"point{p}" for p in ids] + ["topology"])
        for i in range(n):                                                           # geometry and attributes: what each point carries
            for j in range(n):
                p, c = i * n + j, colours[i, j]
                sheet.dots(k, [at(i, j)[0]], [at(i, j)[1]], f"point {p}",
                           ["position " + formula(f"({X[i, j]:.2f}, {Y[i, j]:.2f}, {Z[i, j]:.2f})") + f" m<br>elevation {Z[i, j]:.3f} m<br>colour ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"],
                           f"point{p}", GREEN, size=30, edge=DARK, label=[str(p)], also=("geometry", "attributes"))
        cx = ox + 1.5 * s
        sheet.label(k, cx, 7.55, "a 4 by 4 height field as a mesh", size=14.5, colour=GREY)
        sheet.label(k, cx, 1.15, "shaded face: points 0, 4, 5", size=14.5)
        first, second = (X[0, 0], Y[0, 0], Z[0, 0]), (X[0, 1], Y[0, 1], Z[0, 1])
        parts = (("geometry", "GEOMETRY", "where each point is: 16 points", "points", f"[({first[0]:.1f}, {first[1]:.1f}, {first[2]:.2f}), ({second[0]:.1f}, {second[1]:.1f}, {second[2]:.2f}), ...]",
                  code("points") + ": 16 positions of three numbers each"),
                 ("topology", "TOPOLOGY", "who is connected: 18 triangles", "faceVertexCounts, faceVertexIndices", "[3, 3, 3, ...]   [0, 4, 5,  0, 5, 1, ...]",
                  code("faceVertexCounts") + ": 18 numbers, all of them 3<br>" + code("faceVertexIndices") + ": 54 numbers, three for each face"),
                 ("attributes", "ATTRIBUTES", "data carried by the points", "primvars:displayColor, primvars:elevation",
                  f"[({colours[0, 0][0]:.2f}, {colours[0, 0][1]:.2f}, {colours[0, 0][2]:.2f}), ...]   [{Z[0, 0]:.2f}, {Z[0, 1]:.2f}, ...]",
                  code("primvars:displayColor") + ": 16 colours<br>" + code("primvars:elevation") + ": 16 numbers"))
        bx, bw, bh, gap = 7.3, 8.1, 2.0, 0.3
        for m, (group, head, what, names, example, meaning) in enumerate(parts):
            y = 7.9 - (m + 1) * bh - m * gap
            sheet.box(k, bx, y, bw, bh, head.capitalize(), meaning, group, "white", DARK, radius=0.12)
            sheet.label(k, bx + 0.3, y + bh - 0.42, head, group, size=16, colour="#5f9400", anchor="left")
            sheet.label(k, bx + 3.0, y + bh - 0.42, what, group, size=12.5, colour=GREY, anchor="left")
            sheet.label(k, bx + 0.3, y + bh - 1.02, names, group, size=14, anchor="left", mono=True)
            sheet.label(k, bx + 0.3, y + bh - 1.60, example, group, size=11.5, colour=GREY, anchor="left", mono=True)


class PipelineDiagram(Diagram):
    """Source, filter, sink"""

    size = (10.0, 4.4)

    def plot(self, fig):
        ax = self.sheet(fig)
        ax.set(ylim=(1.0, 8.04))
        w, h, y = 4.2, 3.4, 2.9
        columns = (("SOURCE", ["vc.make_terrain()", "a height field,", "65 by 65 samples"], "white", DARK, DARK),
                   ("FILTER", ["vc.map_scalars()", "height to colour,", "through a lookup table"], DARK, "white", DARK),
                   ("SINK", ["look.usd_scene()", "meshes on the USD stage,", "drawn by RTX"], GREEN, "white", GREEN))
        ax.text(8, 7.3, "scenes.lab02() in labs/scenes.py", ha="center", va="center", fontsize=self.H1, family=_mono(), color=DARK)
        for x, (title, lines, fill, ink, edge) in zip((0.9, 5.9, 10.9), columns):
            self.box(ax, x, y, w, h, fill=fill, edge=edge)
            ax.text(x + w / 2, y + h - 0.6, title, ha="center", va="center", fontsize=self.H1, color=ink)
            for i, line in enumerate(lines):
                mono = line.endswith("()")
                ax.text(x + w / 2, y + h - 1.5 - i * 0.6, line, ha="center", va="center", fontsize=self.SMALL if mono else self.BODY,
                        color=ink if fill != "white" else GREY, family=_mono() if mono else None)
        for x in (0.9, 5.9):
            self.arrow(ax, (x + w + 0.12, y + h / 2), (x + w + 0.68, y + h / 2))
        ax.text(8, 1.9, "Every lab on this page has these three parts.", ha="center", va="center", fontsize=self.BODY, color=GREY)

    def items(self, sheet):
        k = sheet.panel((0, 0, sheet.W, sheet.h), xrange=(0, 16), yrange=(1.0, 8.04), frame=False)
        _X, _Y, Z = vc.make_terrain(65)
        w, h, y = 4.2, 3.4, 2.9
        stages = (("source", 0.9, "SOURCE", "white", DARK, DARK, GREY, ("vc.make_terrain()", "a height field,", "65 by 65 samples"),
                   f"makes the data<br>X, Y, Z: three arrays of 65 by 65 numbers<br>heights from 0 to {Z.max():.3f} m"),
                  ("filter", 5.9, "FILTER", DARK, DARK, "white", "white", ("vc.map_scalars()", "height to colour,", "through a lookup table"),
                   f"turns data into other data<br>clamps each height to 0 .. {Z.max():.3f} m, scales it to 0 .. 1, looks the colour up<br>{Z.size} colours"),
                  ("sink", 10.9, "SINK", GREEN, GREEN, "white", "white", ("look.usd_scene()", "meshes on the USD stage,", "drawn by RTX"),
                   f"hands the result to the renderer<br>two meshes of {Z.size} points and {2 * 64 * 64} triangles each"))
        sheet.label(k, 8, 7.3, "scenes.lab02() in labs/scenes.py", size=18, mono=True)
        for group, x, title, fill, edge, ink, soft, lines, meaning in stages:
            sheet.box(k, x, y, w, h, title.capitalize(), meaning, group, fill, edge, radius=0.12)
            sheet.label(k, x + w / 2, y + h - 0.6, title, group, size=19, colour=ink)
            for i, line in enumerate(lines):
                sheet.label(k, x + w / 2, y + h - 1.5 - i * 0.6, line, group, size=12.5 if i == 0 else 14.5, colour=ink if i == 0 else soft, mono=i == 0)
        flows = (("heights", 0.9, "the height of each sample<br>65 by 65 numbers"), ("colours", 5.9, "the colour of each sample<br>65 by 65 RGB triples"))
        for name, x, meaning in flows:
            sheet.link(k, (x + w + 0.1, y + h / 2), (x + w + 0.7, y + h / 2), name, meaning, name, DARK, head=0.26)
        sheet.label(k, 8, 1.9, "Every lab on this page has these three parts.", size=14.5, colour=GREY)


class ColliderDiagram(Diagram):
    """What a convex hull does to a ring"""

    def plot(self, fig):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Rectangle
        ax = self.sheet(fig)
        ax.add_line(Line2D([8, 8], [0.6, 8.4], color="#cccccc", lw=1.3, zorder=0))
        for x0, title, solid in ((0.5, '"convexHull"', True), (8.5, '"sdf"   "convexDecomposition"   "none"', False)):
            cx, ring_y, ring_h, ground = x0 + 3.5, 4.0, 0.5, 1.7
            ax.text(cx, 8.0, title, ha="center", va="center", fontsize=self.H2, family=_mono(), color=DARK)
            if solid:
                ax.add_patch(Rectangle((cx - 1.25, ring_y), 2.5, ring_h, fc=GREEN, alpha=0.35, ec=GREEN, lw=1.6, ls="--", zorder=1))
                ax.text(cx, ring_y - 0.45, "the hull fills the hole", ha="center", va="center", fontsize=self.BODY, color="#5f9400")
            for sx in (cx - 2.5, cx + 1.25):
                ax.add_patch(Rectangle((sx, ring_y), 1.25, ring_h, fc=GREY, ec=DARK, lw=1.3, zorder=2))
            ax.text(cx - 3.05, ring_y + ring_h / 2, "ring", ha="center", va="center", fontsize=self.BODY, color=GREY)
            ax.add_line(Line2D([x0 + 0.3, x0 + 6.7], [ground, ground], color=DARK, lw=2, zorder=1))
            ax.text(cx - 3.05, ground - 0.38, "ground", ha="center", va="center", fontsize=self.BODY, color=GREY)
            rest = ring_y + ring_h + 0.5 if solid else ground + 0.5
            ax.add_patch(Circle((cx, 6.6), 0.5, fc="none", ec=RED, lw=1.3, ls="--", zorder=2))
            self.arrow(ax, (cx, 6.05), (cx, rest + 0.58), RED)
            ax.add_patch(Circle((cx, rest), 0.5, fc=RED, ec=DARK, lw=1.3, zorder=3))
            ax.text(cx, 0.95, "the ball rests ON the ring" if solid else "the ball falls THROUGH the hole", ha="center",
                    va="center", fontsize=self.H2, color=DARK)

    def items(self, sheet):
        frames = np.arange(55) / 120.0                                              # seconds
        fall = 1.0 - 0.5 * 9.81 * frames ** 2                                       # free fall from 1 m, no bounce
        circle = np.linspace(0, 2 * np.pi, 49)[:-1]
        cases = (("hull", code('"convexHull"'), 0.65, "the ball rests on the ring"),
                 ("open", code('"sdf"') + "   " + code('"convexDecomposition"') + "   " + code('"none"'), 0.10, "the ball falls through the hole"))
        for p, (group, title, rest, verdict) in enumerate(cases):
            k = sheet.panel((62 + 438 * p, 92, 380, 304), x=formula("x") + " (m)", y=formula("z") + " (m)" if p == 0 else "", xrange=(-0.8, 0.8), yrange=(-0.1, 1.18))
            sheet.note(62 + 438 * p + 190, 420, title, size=14.5)
            sheet.note(62 + 438 * p + 190, 22, verdict, size=16)
            sheet.curve(k, [-0.8, 0.8], [0.0, 0.0], "ground", formula("z = 0"), f"{group}/ground", DARK, width=2.6)
            if group == "hull":
                sheet.box(k, -0.5, 0.45, 1.0, 0.1, "convex hull of the ring", "the smallest shape without dents around the ring<br>it has no hole",
                          f"{group}/shape", "rgba(118,185,0,0.35)", GREEN, dash="dash")
            for side in (-0.5, 0.25):
                sheet.box(k, side, 0.45, 0.25, 0.1, "ring", "outer radius 0.5 m, hole radius 0.25 m<br>from " + formula("z = 0.45") + " m to " + formula("z = 0.55") + " m",
                          f"{group}/ring", "#5E5E5E", DARK, width=1.3)
            sheet.area(k, 0.1 * np.cos(circle), 1.0 + 0.1 * np.sin(circle), "the ball at the start", "radius 0.1 m<br>centre at " + formula("z = 1") + " m",
                       f"{group}/start", "rgba(214,39,40,0)", RED, width=1.4, dash="dash")
            z = np.maximum(fall, rest)
            ball = sheet.area(k, 0.1 * np.cos(circle), rest + 0.1 * np.sin(circle), "the ball", "", f"{group}/ball", RED, DARK, width=1.3)
            sheet.moves(ball, np.zeros(len(frames)), z - rest, [formula(f"t = {a:.3f}") + " s<br>centre at " + formula(f"z = {b:.3f}") + " m" for a, b in zip(frames, z)])
        sheet.clock([f"t = {v:.3f} s" for v in frames], seconds=4.0)


# ------------------------------------------------------------------------------------------
# Interactive charts: every item is one Plotly trace with a name and values
# ------------------------------------------------------------------------------------------
SANS = "NVIDIA Sans, NVIDIASans, Helvetica Neue, Arial, sans-serif"                       # regular text
MATH = "Computer Modern Serif, Latin Modern Roman, CMU Serif, Times New Roman, serif"    # mathematics: the typeface of LaTeX (the site has it in _static/fonts)
CODE = "Consolas, Cascadia Mono, Menlo, DejaVu Sans Mono, Liberation Mono, monospace"    # names taken from code
PLOTLY_JS = ("https://cdn.plot.ly/plotly-4.1.1.min.js", "sha256-O24V1F27f8pb0glCkelh3cVHLNiHAJ5gCaVtq2aNch8=")


def _plain(values, digits=5):
    """Arrays as lists a JSON file can hold: numbers to a few significant digits, None where there is no number."""
    a = np.asarray(values, dtype=float)
    out = np.array([float(f"{v:.{digits}g}") if np.isfinite(v) else None for v in a.ravel()], dtype=object)
    return out.reshape(a.shape).tolist()


def _scale(table, steps=32):
    """A lookup table (a function from 0..1 to RGB) as a Plotly colour scale."""
    s = np.linspace(0.0, 1.0, steps + 1)
    rgb = np.rint(np.clip(np.asarray(table(s))[:, :3], 0, 1) * 255).astype(int)
    return [[round(float(v), 4), f"rgb({r},{g},{b})"] for v, (r, g, b) in zip(s, rgb)]


def formula(text):
    """A formula for a label, set as LaTeX sets it: single letters are variables and stand in italics,
    x_1 and x^2 (or x_{12}) become indices. Units and words stay outside: formula("t = 0.25") + " s"."""
    def index(m):
        tag = "sub" if (m.group(1) or m.group(3)) == "_" else "sup"
        return f"<{tag}>{m.group(2) if m.group(2) is not None else m.group(4)}</{tag}>"
    s = re.sub(r"([_^])\{([^}]*)\}|([_^])(\w)", index, text)
    s = re.sub(r"(?<![A-Za-z<&/])([A-Za-z])(?![A-Za-z>])", r"<i>\1</i>", s)
    return f'<span style="font-family:{MATH}">{s}</span>'


def measure(name, key, digits, unit=""):
    """For Sheet.heat: shows the value of one axis as name = number unit, the name set as a variable."""
    return (f'<span style="font-family:{MATH}"><i>{name}</i> = ', key, digits, "</span>" + (" " + unit if unit else ""))


def code(text):
    """A name taken from code, for a label: in the typeface of code."""
    return f'<span style="font-family:{CODE}">{text}</span>'


def _broken(segments):
    """Separate line segments (K, 2, 2) as one line with breaks, for a single trace."""
    xs, ys = [], []
    for a, b in np.asarray(segments, dtype=float):
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
    return xs[:-1], ys[:-1]


def _falls(sheet, k, z0, rest, duration, rates, exact_name, loop_name, seconds=4.0):
    """A drop from z0 to rest: the formula, a loop that steps in time at each rate, and the cube of each at the time of the clock."""
    exact = lambda t: np.maximum(z0 - 0.5 * 9.81 * np.asarray(t) ** 2, rest)        # noqa: E731
    frames = np.arange(int(round(duration * 60)) + 1) / 60.0
    t = np.linspace(0, duration, 61)
    sheet.curve(k, t, exact(t), exact_name, [formula(f"t = {a:.2f}") + " s<br>" + formula(f"z = {b:.3f}") + " m" for a, b in zip(t, exact(t))],
                "exact", DARK, dash="dash", legend=True)
    series = [(f"the cube of the {exact_name}" if exact_name == "formula" else "the cube, exact", "exact", DARK, frames, exact(frames))]
    for rate, colour in rates:
        rows = vc.drop(z0, rest, 1.0 / rate, duration)
        ts, zs = np.r_[0, rows[:, 0]], np.r_[z0, rows[:, 1]]
        name = loop_name.format(rate)
        sheet.curve(k, ts, zs, name, [f"step {i}<br>" + formula(f"t = {a:.3f}") + " s<br>" + formula(f"z = {b:.3f}") + f" m<br>below the formula by {100 * (float(exact(a)) - b):.1f} cm"
                                      for i, (a, b) in enumerate(zip(ts, zs))], f"rate{rate}", colour, markers=6 if rate <= 60 else 3, legend=True)
        last = np.minimum(np.floor(frames * rate + 1e-9).astype(int), len(ts) - 1)   # the last step the loop has made
        series.append((f"the cube, {name}", f"rate{rate}", colour, frames, zs[last]))
    for name, group, colour, xs, zs in series:
        sheet.cursor(k, xs[-1], zs[-1], name, group, colour, xs - xs[-1], zs - zs[-1],
                     [formula(f"t = {a:.3f}") + " s<br>" + formula(f"z = {b:.3f}") + " m" for a, b in zip(frames, zs)])
    sheet.clock([f"t = {v:.3f} s" for v in frames], seconds=seconds)


class Sheet:
    """The interactive form of a chart: a Plotly figure in which every item is one trace.

    An item has a name and values. Pointing at it shows them and makes it stand out; a click keeps it
    selected. Items of one group stand out together; "a/b" is a part of "a". Positions are in pixels
    of a sheet 900 wide, and the whole sheet scales with the page."""

    W = 900

    def __init__(self, size):
        self.h = int(round(self.W * size[1] / size[0]))
        self.data, self.panels, self.time, self.named = [], [], None, {}
        self.layout = dict(
            font=dict(family=SANS, size=15, color=DARK), paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode="closest", dragmode=False,
            hoverlabel=dict(font=dict(family=SANS, size=14), bgcolor="white", bordercolor=GREY, align="left"))

    # ----- panels: a pair of axes in a box given in pixels (left, bottom, width, height)
    def panel(self, box, x=None, y=None, xrange=None, yrange=None, log="", frame=True, title=None):
        k = len(self.panels) + 1
        left, bottom, width, height = box
        for name, label, span, lo, extent, total in (("x", x, xrange, left, width, self.W), ("y", y, yrange, bottom, height, self.h)):
            axis = dict(domain=[round(lo / total, 5), round((lo + extent) / total, 5)], anchor=("y" if name == "x" else "x") + ("" if k == 1 else str(k)),
                        fixedrange=True, automargin=False, zeroline=False, visible=bool(frame))
            if span is not None:
                axis["range"] = [math.log10(v) for v in span] if name in log else list(span)
            if name in log:
                axis.update(type="log", exponentformat="power", dtick=1)
            if frame == "box":
                axis.update(showline=True, mirror=True, linecolor="#bbbbbb", linewidth=1, ticks="", showticklabels=False, showgrid=False)
            elif frame:
                axis.update(title=dict(text=label or "", font=dict(size=15), standoff=8), showline=True, linecolor=DARK, linewidth=1.2,
                            ticks="outside", ticklen=5, tickwidth=1, tickcolor=DARK, tickfont=dict(size=14),
                            showgrid=True, gridcolor="#e6e6e6", gridwidth=1)
            self.layout[f"{name}axis" + ("" if k == 1 else str(k))] = axis
        self.panels.append(box)
        if title:
            self.note(left + width / 2, bottom + height + 14, title, size=16)
        return k

    def axis(self, k, name):
        return self.layout[f"{name}axis" + ("" if k == 1 else str(k))]

    def _add(self, k, trace):
        trace.update(xaxis="x" + ("" if k == 1 else str(k)), yaxis="y" + ("" if k == 1 else str(k)))
        trace.setdefault("showlegend", False)
        self.data.append(trace)
        return trace

    # ----- items
    def curve(self, k, x, y, name, values, group, colour, dash="solid", width=2.2, markers=0, legend=False):
        """A line through points. values: one text for each point, or one text for the whole line.

        A line with one text gets extra points along its straight pieces, so that the pointer finds it anywhere along it."""
        whole = isinstance(values, str)
        if whole:
            x, y = self._along(k, x, y)
        trace = dict(type="scatter", mode="lines+markers" if markers else "lines", x=_plain(x), y=_plain(y), name=name,
                     hovertemplate=f"<b>{name}</b><br>" + (values if whole else "%{text}") + "<extra></extra>",
                     line=dict(color=colour, width=width, dash=dash), meta=dict(g=group, v=values.replace("<br>", "; ")) if whole else dict(g=group))
        if not whole:
            trace["text"] = values
        if markers:
            trace["marker"] = dict(size=markers, color=colour)
        if legend:
            self.named.setdefault(k, []).append(trace)
        return self._add(k, trace)

    def _along(self, k, x, y, gap=10.0):
        """The same line with a point every few pixels (gap) along each straight piece."""
        ax, ay = self.axis(k, "x"), self.axis(k, "y")
        if ax.get("type") == "log" or ay.get("type") == "log" or "range" not in ax or "range" not in ay:
            return x, y
        _left, _bottom, width, height = self.panels[k - 1]
        sx, sy = abs(ax["range"][1] - ax["range"][0]) / width * gap, abs(ay["range"][1] - ay["range"][0]) / height * gap
        xs, ys = [], []
        for i in range(len(x)):
            a = None if i == 0 else (x[i - 1], y[i - 1])
            b = (x[i], y[i])
            if a is not None and a[0] is not None and b[0] is not None:
                n = int(math.ceil(max(abs(b[0] - a[0]) / sx, abs(b[1] - a[1]) / sy, 1.0)))
                for j in range(1, n):
                    xs.append(a[0] + (b[0] - a[0]) * j / n)
                    ys.append(a[1] + (b[1] - a[1]) * j / n)
            xs.append(b[0])
            ys.append(b[1])
        return xs, ys

    def dots(self, k, x, y, name, values, group, colour, size=9, edge=None, label=None, legend=False, also=()):
        """Separate points; label writes a text inside each of them."""
        trace = dict(type="scatter", mode="markers+text" if label else "markers", x=_plain(x), y=_plain(y), name=name,
                     customdata=values, hovertemplate=f"<b>{name}</b><br>%{{customdata}}<extra></extra>",
                     marker=dict(size=size, color=colour, line=dict(color=edge or colour, width=1.2)), meta=dict(g=group, c=1, w=list(also)))
        if label:
            trace.update(text=label, textposition="middle center", textfont=dict(size=12, color="white"))
        return self._add(k, trace)

    def area(self, k, x, y, name, values, group, fill, edge=None, width=1.6, dash="solid", also=(), small=False):
        """A closed outline, filled. It answers to the pointer anywhere inside; a small one also close to its corners."""
        x, y = list(x) + [x[0]], list(y) + [y[0]]
        return self._add(k, dict(type="scatter", mode="lines", x=_plain(x), y=_plain(y), fill="toself", fillcolor=fill, hoveron="points+fills" if small else "fills",
                                 name=name, text=f"<b>{name}</b><br>{values}", hoverinfo="text",
                                 line=dict(color=edge or fill, width=width, dash=dash, simplify=False), meta=dict(g=group, v=values.replace("<br>", "; "), w=list(also))))

    def box(self, k, x, y, w, h, name, values, group, fill, edge=None, radius=0.0, **more):
        """A rectangle, with rounded corners if radius is given (in the units of the panel)."""
        if radius:
            a = np.linspace(0, np.pi / 2, 7)
            corners = ((x + w - radius, y + h - radius, 0), (x + radius, y + h - radius, 1), (x + radius, y + radius, 2), (x + w - radius, y + radius, 3))
            xs = np.concatenate([cx + radius * np.cos(a + q * np.pi / 2) for cx, cy, q in corners])
            ys = np.concatenate([cy + radius * np.sin(a + q * np.pi / 2) for cx, cy, q in corners])
        else:
            xs, ys = [x, x + w, x + w, x], [y, y, y + h, y + h]
        return self.area(k, xs, ys, name, values, group, fill, edge, **more)

    def link(self, k, start, end, name, values, group, colour, both=False, width=2.4, head=0.3):
        """An arrow from start to end (in a panel with the same scale on both axes)."""
        a, b = np.asarray(start, float), np.asarray(end, float)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        tips = [(b, d)] + ([(a, -d)] if both else [])
        steps = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.25)))
        shaft_a, shaft_b = (a + d * head * 0.8 if both else a), b - d * head * 0.8
        xs, ys = list(shaft_a[0] + (shaft_b[0] - shaft_a[0]) * steps), list(shaft_a[1] + (shaft_b[1] - shaft_a[1]) * steps)
        for tip, way in tips:
            base = tip - way * head
            xs += [None, tip[0], base[0] + n[0] * head * 0.42, base[0] - n[0] * head * 0.42, tip[0]]
            ys += [None, tip[1], base[1] + n[1] * head * 0.42, base[1] - n[1] * head * 0.42, tip[1]]
        return self._add(k, dict(type="scatter", mode="lines", x=[None if v is None else round(float(v), 4) for v in xs],
                                 y=[None if v is None else round(float(v), 4) for v in ys], fill="toself", fillcolor=colour, hoveron="points+fills",
                                 name=name, text=f"<b>{name}</b><br>{values}", hoverinfo="text", line=dict(color=colour, width=width),
                                 meta=dict(g=group, v=values.replace("<br>", "; "))))

    def label(self, k, x, y, text, group=None, size=15, colour=DARK, anchor="center", mono=False):
        """Text that belongs to an item (it fades with its group) and does not answer to the pointer itself."""
        position = {"center": "middle center", "left": "middle right", "right": "middle left"}[anchor]
        return self._add(k, dict(type="scatter", mode="text", x=[x], y=[y], text=[text], textposition=position, hoverinfo="skip", cliponaxis=False,
                                 textfont=dict(size=size, color=colour, family=CODE if mono else SANS), meta=dict(g=group) if group else {}))

    def note(self, x, y, text, size=15, colour=DARK, anchor="center"):
        """Text on the sheet itself, placed in pixels: a title or a remark."""
        self.layout.setdefault("annotations", []).append(dict(
            x=round(x / self.W, 5), y=round(y / self.h, 5), xref="paper", yref="paper", text=text, showarrow=False,
            xanchor=anchor, yanchor="middle", font=dict(size=size, color=colour)))

    def heat(self, k, x, y, z, name, fmt, table, low, high, gap=0):
        """A grid of values in colour. fmt: (text before, "x" | "y" | "z", decimals, text after) for each value to show."""
        template = "<br>".join(f"{before}%{{{key}:.{digits}f}}{after}" for before, key, digits, after in fmt)
        return self._add(k, dict(type="heatmap", x=_plain(x), y=_plain(y), z=_plain(z), name=name, colorscale=_scale(table), zmin=low, zmax=high,
                                 showscale=False, xgap=gap, ygap=gap, hovertemplate=f"<b>{name}</b><br>{template}<extra></extra>",
                                 meta=dict(fmt=[list(f) for f in fmt])))

    def cursor(self, k, x, y, name, group, colour, dx, dy, texts, size=13):
        """A marker that moves with time: at frame f it is at (x + dx[f], y + dy[f]) and says texts[f]."""
        return self._add(k, dict(type="scatter", mode="markers", x=[round(float(x), 4)], y=[round(float(y), 4)], name=name, text=[texts[-1]],
                                 hovertemplate=f"<b>{name}</b><br>%{{text}}<extra></extra>", cliponaxis=False,
                                 marker=dict(size=size, color="white", line=dict(color=colour, width=2.6)),
                                 meta=dict(g=group, texts=list(texts), move=dict(x=[round(float(x), 4)], y=[round(float(y), 4)], dx=_plain(dx), dy=_plain(dy)))))

    def moves(self, trace, dx, dy, texts):
        """Lets an area move with time."""
        trace["meta"].update(texts=list(texts), move=dict(x=list(trace["x"]), y=list(trace["y"]), dx=_plain(dx), dy=_plain(dy)))
        trace["text"] = f"<b>{trace['name']}</b><br>{texts[-1]}"
        return trace

    def spot(self, k, x, y):
        """A place given in pixels of the sheet, in the units of panel k."""
        left, bottom, width, height = self.panels[k - 1]
        out = []
        for name, frac in (("x", (x - left) / width), ("y", (y - bottom) / height)):
            axis = self.axis(k, name)
            value = axis["range"][0] + frac * (axis["range"][1] - axis["range"][0])
            out.append(10 ** value if axis.get("type") == "log" else value)
        return out

    def legend(self, k, corner="top right", inset=12):
        """The names of the lines of panel k that were made with legend=True, in a corner of the panel.
        Each name is an item too: pointing at it makes its line stand out."""
        entries = self.named.get(k, [])
        left, bottom, width, height = self.panels[k - 1]
        pad, row, sample, size = 8, 22, 30, 14
        w = 2 * pad + sample + 10 + max(0.56 * size * len(t["name"]) for t in entries)
        h = 2 * pad + row * len(entries) - 4
        x0 = left + width - inset - w if "right" in corner else left + inset
        y1 = bottom + height - inset if "top" in corner else bottom + inset + h
        (ax, ay), (bx, by) = self.spot(k, x0, y1 - h), self.spot(k, x0 + w, y1)
        back = self.area(k, [ax, bx, bx, ax], [ay, ay, by, by], "", "", None, "rgba(255,255,255,0.85)", "rgba(255,255,255,0)", width=0)
        back.update(hoverinfo="skip", meta={})
        for n, t in enumerate(entries):
            y = y1 - pad - row * n - (row - 4) / 2
            points = [self.spot(k, x0 + pad + sample * f, y) for f in (0.0, 0.5, 1.0)]
            entry = dict(type="scatter", mode=t["mode"], x=_plain([q[0] for q in points]), y=_plain([q[1] for q in points]), name=t["name"],
                         hovertemplate=f"<b>{t['name']}</b><extra></extra>", line=dict(t["line"]), meta=dict(g=t["meta"]["g"], key=1))
            if "marker" in t:
                big = t["marker"]["size"] if isinstance(t["marker"]["size"], (int, float)) else 6
                entry["marker"] = dict(color=t["marker"]["color"], size=[0, big, 0])
            self._add(k, entry)
            tx, ty = self.spot(k, x0 + pad + sample + 10, y)
            self.label(k, tx, ty, t["name"], t["meta"]["g"], size=size, anchor="left")

    def clock(self, labels, seconds=4.0):
        """Gives the sheet a time: one label for each frame, and how long playing all of them takes."""
        self.time = dict(labels=list(labels), seconds=seconds)

    def figure(self):
        """The sheet as plotly.js reads it."""
        data = list(self.data)
        for k in range(1, len(self.panels) + 1):               # a circle that marks the selected point of each panel
            n = "" if k == 1 else str(k)
            data.append(dict(type="scatter", mode="markers", x=[], y=[], xaxis="x" + n, yaxis="y" + n, hoverinfo="skip", showlegend=False, cliponaxis=False,
                             marker=dict(symbol="circle-open", size=20, color=DARK, line=dict(color=DARK, width=2.6)), meta=dict(ring=f"x{n}y{n}")))
        hint = "Point at an item to see its values. Click it to keep it selected."
        if self.time:
            hint += " Play runs the time."
        first = lambda family: family.split(",")[0]                    # noqa: E731
        fonts = [f"16px {first(SANS)}", f"16px {first(MATH)}", f"italic 16px {first(MATH)}"]   # what the page loads before it draws
        return dict(w=self.W, h=self.h, data=data, layout=self.layout, time=self.time, hint=hint, bar=False, fonts=fonts)


class Plot(Renderer):
    """A chart for a web page, drawn by plotly.js. Every item answers to the pointer: it stands out and
    shows its values, and a click keeps it selected. Until plotly.js has arrived, and if it cannot,
    the still picture is shown."""

    def draw(self, chart, still=None):
        """Returns HTML. still is a PNG file to show until, or instead of, the interactive figure."""
        sheet = Sheet(chart.size)
        chart.items(sheet)
        spec = sheet.figure()
        if still is None:
            still = chart.still(dpi=100)
        label = chart.caption.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        clock = ('<div class="simscene-time" hidden><button type="button">Play</button>'
                 '<input type="range" min="0" max="1" step="1" value="1" aria-label="Time"><output></output></div>') if spec["time"] else ""
        text = json.dumps(spec, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
        return (f'<figure class="simplot"><style>{_PLOT_CSS}</style>'
                f'<div class="simplot-stage" style="padding-top:{100 * spec["h"] / spec["w"]:.3f}%">'
                f'<img alt="{label}" src="data:image/png;base64,{base64.b64encode(still).decode("ascii")}">'
                f'<div class="simplot-plot" hidden></div></div>{clock}<p class="simplot-read" hidden></p>'
                f'<script type="application/json">{text}</script></figure>'
                f'<script>{_PLOTTER}</script><script>window.simplotBoot();</script>')


_PLOT_CSS = (".simplot{margin:0 0 .5em}"
             ".simplot-stage{position:relative;height:0;overflow:hidden;background:#fff}"
             ".simplot-stage img,.simplot-plot{position:absolute;left:0;top:0;display:block;width:100%;height:100%;margin:0;border:0;max-width:none}"
             ".simplot-stage [hidden],.simplot-read[hidden]{display:none !important}"
             ".simplot-read{margin:.35em 0 0;min-height:1.7em;font-size:.9em;text-align:left;opacity:.85}"
             ".simplot-read button{margin-left:.6em;font-size:.9em}") + _TIME_CSS
_PLOTTER = r"""(function () {
  'use strict';
  if (window.simplotBoot) return;                       // defined once per page, however many figures there are
  var SRC = 'https://cdn.plot.ly/plotly-4.1.1.min.js', SRI = 'sha256-O24V1F27f8pb0glCkelh3cVHLNiHAJ5gCaVtq2aNch8=', DIM = 0.22;
  var state = 0, waiting = [];                          // plotly.js: 0 not asked for, 1 on its way, 2 here, 3 not available

  // ---------- plotly.js, loaded once, and the typeface of the page
  function library(done) {
    if (window.Plotly) { state = 2; return done(true); }
    if (state === 3) return done(false);
    waiting.push(done);
    if (state === 1) return;
    state = 1;
    var s = document.createElement('script'), amd = window.define;
    function finish() {
      if (amd) window.define = amd;
      state = window.Plotly ? 2 : 3;
      waiting.splice(0).forEach(function (f) { f(state === 2); });
    }
    if (amd && amd.amd) window.define = undefined;      // a module loader on the page would keep Plotly to itself
    s.src = SRC; s.integrity = SRI; s.crossOrigin = 'anonymous';
    s.onload = finish; s.onerror = finish;
    document.head.appendChild(s);
  }
  function typeface(fonts, done) {                      // text is measured when it is drawn, so the fonts come first
    var called = false, left = fonts.length;
    function go() { if (!called) { called = true; done(); } }
    function one() { if (--left <= 0) go(); }
    try { fonts.forEach(function (f) { document.fonts.load(f).then(one, one); }); window.setTimeout(go, 1500); } catch (err) { go(); }
  }

  // ---------- a copy of a figure in which every length in pixels is k times as long
  var PX = { size: 1, ticklen: 1, tickwidth: 1, linewidth: 1, gridwidth: 1, zerolinewidth: 1, standoff: 1, borderwidth: 1,
             thickness: 1, borderpad: 1, xgap: 1, ygap: 1, tracegroupgap: 1 };
  var KEEP = { x: 1, y: 1, z: 1, text: 1, meta: 1, hoverlabel: 1, colorscale: 1, customdata: 1, ticktext: 1, tickvals: 1 };
  function scaled(v, k, key, parent) {
    if (typeof v === 'number') return PX[key] || (key === 'width' && parent === 'line') || parent === 'margin' ? v * k : v;
    if (!v || typeof v !== 'object' || KEEP[key]) return v;
    if (Array.isArray(v)) return v.map(function (e) { return typeof e === 'number' ? (key === 'size' ? e * k : e) : scaled(e, k, key, parent); });
    var o = {};
    for (var n in v) o[n] = scaled(v[n], k, n, key);
    return o;
  }

  function start(figure, spec) {
    var stage = figure.querySelector('.simplot-stage'), still = figure.querySelector('img'), plot = figure.querySelector('.simplot-plot');
    var read = figure.querySelector('.simplot-read'), bar = figure.querySelector('.simscene-time');
    var P = window.Plotly, hot = null, sel = null, frame = -1, shown = '', timer = 0, width = 0;
    var config = { displaylogo: false, responsive: false, scrollZoom: false, displayModeBar: spec.bar ? 'hover' : false,
                   modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'] };

    // ----- highlight: the item under the pointer, or else the selected one, stands out; the others fade
    function group(i) { var m = spec.data[i].meta; return m && m.g ? m.g : null; }
    function near(g, on) { return g === on || g.indexOf(on + '/') === 0 || on.indexOf(g + '/') === 0; }
    function paint() {                                  // an item stands out with its parts, the whole it is part of, and its partners
      var i0 = hot !== null ? hot : (sel ? sel.i : -1), on = i0 < 0 ? null : group(i0), m0 = i0 < 0 ? {} : spec.data[i0].meta || {};
      var lit = on ? [on].concat(m0.w || []) : [], op = spec.data.map(function (t, i) {
        var g = group(i), base = t.opacity === undefined ? 1 : t.opacity;
        return on && g && !lit.some(function (o) { return near(g, o); }) ? base * DIM : base;
      });
      var key = op.join(',');
      if (key !== shown) { shown = key; P.restyle(plot, { opacity: op }); }
    }

    // ----- values: what the label of an item says, as one line
    function values(i, p) {
      var t = spec.data[i], m = t.meta || {}, s;
      if (m.fmt) s = m.fmt.map(function (f) { return f[0] + Number(p[f[1]]).toFixed(f[2]) + f[3]; }).join('; ');
      else if (m.move && m.texts) s = m.texts[frame < 0 ? m.texts.length - 1 : frame];
      else if (m.v !== undefined) s = m.v;
      else {
        var src = m.c ? t.customdata : t.text;
        s = Array.isArray(src) ? (Array.isArray(src[0]) ? src[p.pointNumber[0]][p.pointNumber[1]] : src[p.pointNumber]) : src;
      }
      return String(s === undefined ? '' : s).replace(/<br>/g, '; ');
    }
    function say() {
      read.innerHTML = sel ? '<b>' + sel.name + '</b>: ' + sel.values + ' <button type="button">Release</button>' : spec.hint;
    }
    function ring() {                                   // a circle around the selected point
      spec.data.forEach(function (t, i) {
        if (!t.meta || !t.meta.ring) return;
        var here = sel && sel.at && sel.axes === t.meta.ring;
        t.x = here ? [sel.at[0]] : []; t.y = here ? [sel.at[1]] : [];
        P.restyle(plot, { x: [t.x], y: [t.y] }, [i]);
      });
    }
    function release() { sel = null; say(); ring(); paint(); }
    function wire() {
      plot.on('plotly_hover', function (e) {
        var p = e.points && e.points[0];
        if (!p) return;
        window.clearTimeout(timer); hot = group(p.curveNumber) ? p.curveNumber : null; paint();
      });
      plot.on('plotly_unhover', function () {
        window.clearTimeout(timer);
        timer = window.setTimeout(function () { hot = null; paint(); }, 80);
      });
      plot.on('plotly_click', function (e) {
        var p = e.points && e.points[0];
        if (!p) return;
        var i = p.curveNumber, t = spec.data[i], g = group(i), m = t.meta || {};
        if ((!g && !m.fmt) || m.key) return;                // a name in a legend only points at its line
        var id = i + ':' + (p.pointNumber === undefined ? '' : String(p.pointNumber));
        if (sel && sel.id === id) return release();
        var point = p.pointNumber !== undefined && !m.move && typeof p.x === 'number' && typeof p.y === 'number';
        sel = { id: id, i: i, g: g, name: t.name, values: values(i, p), axes: p.xaxis && p.yaxis ? p.xaxis._id + p.yaxis._id : '', at: point ? [p.x, p.y] : null };
        say(); ring(); paint();
      });
      read.addEventListener('click', function (e) { if (e.target.tagName === 'BUTTON') release(); });
    }

    // ----- time: items that have a path move along it
    function at(f) {
      var idx = [], xs = [], ys = [], ts = [];
      frame = f;
      spec.data.forEach(function (t, i) {
        var m = t.meta && t.meta.move;
        if (!m) return;
        t.x = m.x.map(function (v) { return v === null ? null : v + m.dx[f]; });
        t.y = m.y.map(function (v) { return v === null ? null : v + m.dy[f]; });
        if (t.meta.texts) t.text = t.fill ? '<b>' + t.name + '</b><br>' + t.meta.texts[f] : t.x.map(function () { return t.meta.texts[f]; });
        idx.push(i); xs.push(t.x); ys.push(t.y); ts.push(t.text);
      });
      if (idx.length) P.restyle(plot, { x: xs, y: ys, text: ts }, idx);
      if (sel && spec.data[sel.i].meta.move) { sel.values = values(sel.i, {}); say(); }
    }
    function clock() {
      var T = spec.time, n = T.labels.length, playing = false, t0 = 0, f0 = 0;
      var btn = bar.querySelector('button'), range = bar.querySelector('input'), out = bar.querySelector('output');
      function set(f) { range.value = f; out.textContent = T.labels[f]; at(f); }
      function stop() { playing = false; btn.textContent = 'Play'; btn.removeAttribute('data-on'); }
      function tick(now) {
        if (!playing) return;
        var f = Math.min(n - 1, f0 + Math.floor((now - t0) / (T.seconds * 1000) * (n - 1)));
        if (f !== frame) set(f);
        if (f >= n - 1) return stop();
        window.requestAnimationFrame(tick);
      }
      btn.addEventListener('click', function () {
        if (playing) return stop();
        f0 = frame < 0 || frame >= n - 1 ? 0 : frame;
        if (f0 === 0) set(0);
        playing = true; btn.textContent = 'Pause'; btn.setAttribute('data-on', '1');
        t0 = window.performance.now(); window.requestAnimationFrame(tick);
      });
      range.addEventListener('input', function () { stop(); set(Number(range.value)); });
      range.max = n - 1; range.value = n - 1; out.textContent = T.labels[n - 1];
      bar.hidden = false;
    }

    // ----- drawing: the figure fills the width it is given, and everything in it scales with that width
    function draw() {
      width = stage.clientWidth || spec.w;
      var k = Math.max(0.4, Math.min(1.4, width / spec.w)), layout = scaled(spec.layout, k, 'layout', '');
      layout.width = Math.round(spec.w * k); layout.height = Math.round(spec.h * k); layout.autosize = false;
      shown = '';
      return P.react(plot, scaled(spec.data, k, 'data', ''), layout, config).then(paint);
    }

    plot.hidden = false;
    return draw().then(function () {
      still.hidden = true; read.hidden = false;
      say(); wire();
      if (spec.time) clock();
      figure.classList.add('live');
      if (window.ResizeObserver) {
        var again = 0;
        new window.ResizeObserver(function () {
          window.clearTimeout(again);
          again = window.setTimeout(function () { if (stage.clientWidth && stage.clientWidth !== width) draw(); }, 150);
        }).observe(stage);
      }
    });
  }

  function boot() {
    var figures = document.querySelectorAll('figure.simplot');
    function fall(figure, err) {                        // keep the still picture
      var q = function (s) { return figure.querySelector(s); };
      if (err && window.console) window.console.warn('figure: ' + (err.message || err));
      figure.setAttribute('data-error', String(err && (err.stack || err.message) || err));
      if (q('.simplot-plot')) q('.simplot-plot').hidden = true;
      if (q('.simplot-read')) q('.simplot-read').hidden = true;
      if (q('.simscene-time')) q('.simscene-time').hidden = true;
      if (q('img')) q('img').hidden = false;
      figure.classList.remove('live');
    }
    function open(figure) {
      if (figure.getAttribute('data-started')) return;
      figure.setAttribute('data-started', '1');
      library(function (ok) {
        if (!ok) return fall(figure, 'plotly.js did not arrive');
        var spec;
        try { spec = JSON.parse(figure.querySelector('script[type="application/json"]').textContent); } catch (err) { return fall(figure, err); }
        typeface(spec.fonts || [], function () {
          try { start(figure, spec).then(null, function (err) { fall(figure, err); }); } catch (err) { fall(figure, err); }
        });
      });
    }
    if (window.IntersectionObserver) {                  // start a figure when it scrolls into view
      var watch = new window.IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) { watch.unobserve(en.target); open(en.target); } });
      }, { rootMargin: '400px' });
      Array.prototype.forEach.call(figures, function (f) { watch.observe(f); });
    } else Array.prototype.forEach.call(figures, open);
  }

  window.simplotBoot = boot;
})();
"""


# ------------------------------------------------------------------------------------------
# Code in colour: the code blocks of a page, coloured the way Notion colours them
# ------------------------------------------------------------------------------------------
def _key(text):
    """A short name for a piece of code (the same sum the page script computes)."""
    x, raw = 0x811C9DC5, text.encode("utf-16-le")
    for i in range(0, len(raw), 2):
        x = ((x ^ int.from_bytes(raw[i:i + 2], "little")) * 0x01000193) & 0xFFFFFFFF
    return format(x, "x")


def code_blocks(notebook):
    """Every piece of code in a notebook as (language, text): the fenced blocks of its text and its code cells."""
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            yield "python", source
            continue
        lines, i = source.split("\n"), 0
        while i < len(lines):
            m = re.match(r"^( *)```([a-z][a-z ]*)$", lines[i])
            if not m:
                i += 1
                continue
            indent, end = len(m.group(1)), i + 1
            while lines[end] != m.group(1) + "```":
                end += 1
            yield m.group(2), "\n".join(line[indent:] for line in lines[i + 1:end])
            i = end + 1


def colour(notebook):
    """Colours the code blocks of the page made from this notebook the way Notion colours them.

    The site colours code with Pygments, which leaves shell commands almost bare. This reads the
    notebook, cuts every piece of code into tokens with the grammar Notion uses, and gives the page
    a small script that puts those colours in. A page without the script keeps the colours it has."""
    from IPython.display import display
    import notioncode
    names = {"text": "plain text"}
    blocks = {_key(text.rstrip()): notioncode.highlight(text.rstrip(), names.get(language, language)) for language, text in code_blocks(notebook)}
    data = json.dumps(blocks, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    display({"text/html": f'<style>{_CODE_CSS}</style><script type="application/json" class="simcode-data">{data}</script><script>{_CODE_JS}</script>',
             "text/plain": f"<{len(blocks)} pieces of code in colour>"}, raw=True)


_CODE_CSS = ("div.highlight.simcode,div.highlight.simcode pre{background:#F7F6F3;color:#37352F}"
             ".simcode .token.comment,.simcode .token.prolog,.simcode .token.doctype,.simcode .token.cdata{color:slategray}"
             ".simcode .token.punctuation{color:#999}"
             ".simcode .token.namespace{opacity:.7}"
             ".simcode .token.property,.simcode .token.tag,.simcode .token.boolean,.simcode .token.number,.simcode .token.constant,"
             ".simcode .token.symbol,.simcode .token.deleted{color:#905}"
             ".simcode .token.selector,.simcode .token.attr-name,.simcode .token.string,.simcode .token.char,.simcode .token.builtin,"
             ".simcode .token.inserted{color:#690}"
             ".simcode .token.operator,.simcode .token.entity,.simcode .token.url{color:#9a6e3a}"
             ".simcode .token.atrule,.simcode .token.attr-value,.simcode .token.keyword{color:#07a}"
             ".simcode .token.function,.simcode .token.class-name{color:#DD4A68}"
             ".simcode .token.regex,.simcode .token.important,.simcode .token.variable{color:#e90}"
             ".simcode .token.important,.simcode .token.bold{font-weight:bold}"
             ".simcode .token.italic{font-style:italic}"
             'html[data-theme="dark"] div.highlight.simcode,html[data-theme="dark"] div.highlight.simcode pre{background:#2d2d2d;color:#ccc}'
             'html[data-theme="dark"] .simcode .token.comment,html[data-theme="dark"] .simcode .token.prolog{color:#999}'
             'html[data-theme="dark"] .simcode .token.punctuation{color:#ccc}'
             'html[data-theme="dark"] .simcode .token.property,html[data-theme="dark"] .simcode .token.tag,html[data-theme="dark"] .simcode .token.constant,'
             'html[data-theme="dark"] .simcode .token.symbol,html[data-theme="dark"] .simcode .token.deleted{color:#e2777a}'
             'html[data-theme="dark"] .simcode .token.boolean,html[data-theme="dark"] .simcode .token.number,html[data-theme="dark"] .simcode .token.function{color:#f08d49}'
             'html[data-theme="dark"] .simcode .token.class-name{color:#f8c555}'
             'html[data-theme="dark"] .simcode .token.selector,html[data-theme="dark"] .simcode .token.important,html[data-theme="dark"] .simcode .token.atrule,'
             'html[data-theme="dark"] .simcode .token.keyword,html[data-theme="dark"] .simcode .token.builtin{color:#cc99cd}'
             'html[data-theme="dark"] .simcode .token.string,html[data-theme="dark"] .simcode .token.char,html[data-theme="dark"] .simcode .token.attr-value,'
             'html[data-theme="dark"] .simcode .token.regex,html[data-theme="dark"] .simcode .token.variable{color:#7ec699}'
             'html[data-theme="dark"] .simcode .token.operator,html[data-theme="dark"] .simcode .token.entity,html[data-theme="dark"] .simcode .token.url{color:#67cdcc}')
_CODE_JS = r"""(function () {
  'use strict';
  function key(s) {
    var x = 0x811c9dc5;
    for (var i = 0; i < s.length; i++) x = Math.imul(x ^ s.charCodeAt(i), 0x01000193) >>> 0;
    return x.toString(16);
  }
  function run() {
    var map = {}, n, i;
    var data = document.querySelectorAll('script.simcode-data');
    for (i = 0; i < data.length; i++) { var part = JSON.parse(data[i].textContent); for (n in part) map[n] = part[n]; }
    var pre = document.querySelectorAll('div.highlight > pre');
    for (i = 0; i < pre.length; i++) {
      var html = map[key(pre[i].textContent.replace(/\s+$/, ''))];
      if (html === undefined || pre[i].parentNode.classList.contains('simcode')) continue;
      pre[i].innerHTML = html;
      pre[i].parentNode.classList.add('simcode');
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run); else run();
})();"""

"""simscene: the figures of the Isaac Sim pages.

A scene is a list of shapes and a camera. It is written down once, in labs/scene.py and
labs/scenes.py, the files the lab scripts use on the DGX Spark. This module draws the same
scenes for the notebook, with NumPy and matplotlib only:

    show(HelloCube())            a 3D scene: a canvas you can turn, and a still picture
    show(FallChart())            a chart or a diagram, as SVG

Three class families, each with one method that its subclasses override:

    Picture    Scene, Chart                    bundle()   what a notebook cell displays
    Scene      HelloCube, Terrain, ...         shapes()   the triangle meshes of the scene
    Renderer   Still, Canvas, Stage            draw(scene) a picture, a web canvas, a USD stage
    Chart      SetupDiagram, FallChart, ...    plot(fig)  the matplotlib drawing
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

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


class CubeAtRest(HelloCube):
    """The cube at rest on the ground"""

    def __init__(self):
        super().__init__(height=chapter0.SIZE / 2)


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


def _triangles(scene):
    """Every triangle of a scene: world corners (N, 3, 3) and corner colours (N, 3, 3)."""
    corners, colours = [], []
    for s in scene.shapes():
        points = look.world_points(s)
        corners.append(points[s["tris"]])
        c = s["colors"]
        colours.append(np.repeat(c[:, None, :], 3, axis=1) if s["per"] == "face" else c[s["tris"]])
    return np.concatenate(corners), np.concatenate(colours)


class Still(Renderer):
    """A still picture, drawn with NumPy: each triangle is projected through the scene's camera
    and filled; where triangles overlap the nearest one wins (a depth buffer)."""

    NEAR = 0.05                                   # metres; nearer parts are cut away
    LARGE = 64                                    # pixels; larger triangles are halved first

    def __init__(self, width=look.WIDTH, height=look.HEIGHT, oversample=2):
        self.width, self.height, self.oversample = width, height, oversample

    def draw(self, scene, background=(1.0, 1.0, 1.0)):
        """Returns the picture as an array (height, width, 3) of bytes."""
        W, H = self.width * self.oversample, self.height * self.oversample
        r, u, f = _camera(scene.view)
        world, colour = _triangles(scene)
        rel = world - scene.view.eye
        cam = np.stack([rel @ r, rel @ u, rel @ f], axis=-1)        # x right, y up, z depth
        cam, colour = self._clip(cam, colour)
        t = 1.0 / np.tan(np.radians(look.FOV) / 2)
        image = np.empty((H * W, 3), np.float32)
        image[:] = background
        depth = np.zeros(H * W, np.float32)                         # 1 / distance; 0 is infinitely far
        for _ in range(40):                                         # halve large triangles until all are small
            inv = 1.0 / cam[..., 2]
            sx = (cam[..., 0] * t * H / W * inv * 0.5 + 0.5) * W
            sy = (0.5 - cam[..., 1] * t * inv * 0.5) * H
            x0 = np.clip(np.floor(sx.min(1)), 0, W).astype(int)
            x1 = np.clip(np.ceil(sx.max(1)), -1, W - 1).astype(int)
            y0 = np.clip(np.floor(sy.min(1)), 0, H).astype(int)
            y1 = np.clip(np.ceil(sy.max(1)), -1, H - 1).astype(int)
            on_screen = (x1 >= x0) & (y1 >= y0)
            span = np.maximum(x1 - x0, y1 - y0) + 1
            large = on_screen & (span > self.LARGE)
            if not large.any():
                break
            cam, colour = self._halve(cam, colour, sx, sy, on_screen, large)
        bounds = (0, 4, 8, 16, 24, 32, 48, self.LARGE)
        for lo, size in zip(bounds[:-1], bounds[1:]):               # triangles of similar size together
            group = np.flatnonzero(on_screen & (span > lo) & (span <= size))
            step = max(1, int(1.5e6 // (size * size)))
            for k in range(0, len(group), step):
                self._fill(group[k:k + step], size, sx, sy, inv, colour, x0, y0, x1, y1, W, image, depth)
        s = self.oversample
        image = image.reshape(H // s, s, W // s, s, 3).mean(axis=(1, 3))
        return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    def _clip(self, cam, colour):
        """Cut triangles at the near plane (most are in front of it and pass unchanged)."""
        d = cam[..., 2]
        keep = (d >= self.NEAR).all(axis=1)
        cross = ~keep & (d >= self.NEAR).any(axis=1)
        out_cam, out_col = [cam[keep]], [colour[keep]]
        for tri, col in zip(cam[cross], colour[cross]):
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
    def _halve(cam, colour, sx, sy, keep, large):
        """Cut each large triangle in two at the middle of its longest edge on the screen."""
        edge = np.stack([np.hypot(sx[large][:, i] - sx[large][:, (i + 1) % 3],
                                  sy[large][:, i] - sy[large][:, (i + 1) % 3]) for i in range(3)], axis=1)
        first = edge.argmax(axis=1)[:, None]
        order = (first + np.arange(3)) % 3                           # the longest edge runs from corner 0 to 1
        rows = np.arange(len(first))[:, None]
        p, c = cam[large][rows, order], colour[large][rows, order]
        pm, cm = (p[:, 0] + p[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2
        halves_p = np.concatenate([np.stack([p[:, 0], pm, p[:, 2]], 1), np.stack([pm, p[:, 1], p[:, 2]], 1)])
        halves_c = np.concatenate([np.stack([c[:, 0], cm, c[:, 2]], 1), np.stack([cm, c[:, 1], c[:, 2]], 1)])
        small = keep & ~large
        return np.concatenate([cam[small], halves_p]), np.concatenate([colour[small], halves_c])

    @staticmethod
    def _fill(tri, size, sx, sy, inv, colour, x0, y0, x1, y1, W, image, depth):
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
        rgb = ((a * ia)[:, None] * colour[tri[n], 0] + (b * ib)[:, None] * colour[tri[n], 1]
               + (c * ic)[:, None] * colour[tri[n], 2]) / z[:, None]
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
        return (f'<figure class="simscene"><style>{_CSS}</style><div class="simscene-stage">'
                f'<img alt="{label}" src="data:image/png;base64,{picture}">'
                f'<canvas hidden tabindex="0" role="img" aria-label="{label}. Drag to turn it, arrow keys also turn it, '
                'plus and minus zoom, 0 resets."></canvas>'
                '<div class="simscene-tools" hidden><button type="button" data-act="in" aria-label="Zoom in">+</button>'
                '<button type="button" data-act="out" aria-label="Zoom out">&minus;</button>'
                '<button type="button" data-act="reset">Reset view</button></div></div>'
                f'<script type="application/json">{json.dumps(self.data(scene), separators=(",", ":"))}</script></figure>'
                f'<script>{_VIEWER}</script><script>window.simsceneBoot();</script>')

    @staticmethod
    def data(scene):
        """The scene as the viewer reads it: corner positions as 16-bit numbers, triangles, colours."""
        shapes = scene.shapes()
        points = [look.world_points(s) for s in shapes]
        lo = np.min([p.min(axis=0) for p in points], axis=0)
        size = float(np.max([p.max(axis=0) for p in points], axis=0).__sub__(lo).max())
        code = lambda array: base64.b64encode(array.tobytes()).decode("ascii")   # noqa: E731
        meshes = []
        for s, p in zip(shapes, points):
            wide = len(p) > 65535
            meshes.append(dict(
                name=s["name"], per=s["per"], ntri=int(len(s["tris"])), wide=wide,
                pos=code(np.rint((p - lo) / size * 65535).astype("<u2")),
                idx=code(s["tris"].astype("<u4" if wide else "<u2")),
                col=code(np.rint(np.clip(s["colors"], 0, 1) * 255).astype(np.uint8))))
        return dict(name=type(scene).__name__, eye=[float(v) for v in scene.view.eye],
                    target=[float(v) for v in scene.view.target], fov=look.FOV,
                    lo=[float(v) for v in lo], size=size, bg=[255, 255, 255], meshes=meshes)


class Stage(Renderer):
    """The scene on a USD stage, for Isaac Sim: one mesh prim per shape, a matte material,
    the camera and an even light. The lab scripts make the same call."""

    def __init__(self, stage, theme="light"):
        self.stage, self.theme = stage, theme

    def draw(self, scene):
        """Returns {shape name: prim path}."""
        return look.usd_scene(self.stage, scene.shapes(), scene.view, self.theme)


_CSS = (".simscene{margin:0 0 .5em}"
        ".simscene-stage{position:relative;border:1px solid #d9d9d9;border-radius:8px;overflow:hidden;background:#fff}"
        '.simscene-stage::before{content:"";display:block;padding-top:56.25%}'
        ".simscene-stage img,.simscene-stage canvas{position:absolute;left:0;top:0;display:block;width:100%;height:100%;"
        "margin:0;border:0;border-radius:0;max-width:none}"
        ".simscene-stage canvas{cursor:grab;touch-action:pan-y}"
        ".simscene-stage canvas:active{cursor:grabbing}"
        ".simscene [hidden]{display:none !important}"
        ".simscene.active .simscene-stage{border-color:#76B900;box-shadow:0 0 0 2px #76B900}"
        ".simscene-tools{position:absolute;top:10px;right:10px;display:flex;gap:6px}"
        ".simscene-tools button{font:13px/1 sans-serif;min-width:34px;padding:8px 10px;border:1px solid #d9d9d9;"
        "border-radius:6px;background:rgba(255,255,255,.92);color:#1a1a1a;cursor:pointer}")

# The viewer: WebGL 1, no library. It starts every figure.simscene of the page when it scrolls into view.
_VIEWER = r"""(function () {
  'use strict';
  if (window.simsceneBoot) return;                      // defined once per page, however many figures there are

  // ---------- data: base64 -> typed arrays -> one flat list of coloured triangles
  function bytes(b64) {
    var s = atob(b64), n = s.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function build(scene) {
    var total = 0, m, i;
    for (i = 0; i < scene.meshes.length; i++) total += scene.meshes[i].ntri * 3;
    var pos = new Float32Array(total * 3), col = new Uint8Array(total * 3), o = 0;
    for (i = 0; i < scene.meshes.length; i++) {
      m = scene.meshes[i];
      var q = new Uint16Array(bytes(m.pos).buffer);
      var raw = bytes(m.idx).buffer;
      var idx = m.wide ? new Uint32Array(raw) : new Uint16Array(raw);
      var c = bytes(m.col), face = m.per === 'face';
      for (var t = 0; t < m.ntri; t++) {
        for (var k = 0; k < 3; k++) {
          var v = idx[t * 3 + k], from = face ? t : v;
          for (var a = 0; a < 3; a++) {
            pos[o * 3 + a] = scene.lo[a] + q[v * 3 + a] / 65535 * scene.size;
            col[o * 3 + a] = c[from * 3 + a];
          }
          o++;
        }
      }
    }
    return { pos: pos, col: col, count: total };
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
    var near = Math.max(0.01, cam.dist * 0.02), far = cam.dist + 3 * cam.size;
    var t = 1 / Math.tan(cam.fov * Math.PI / 360);
    var A = (far + near) / (near - far), B = 2 * far * near / (near - far);
    var sx = t / aspect, sy = t;
    return new Float32Array([
      sx * r[0], sy * u[0], -A * f[0], f[0],
      sx * r[1], sy * u[1], -A * f[1], f[1],
      sx * r[2], sy * u[2], -A * f[2], f[2],
      -sx * dot(r, e), -sy * dot(u, e), A * dot(f, e) + B, -dot(f, e)]);
  }

  // ---------- drawing
  var VS = 'attribute vec3 p; attribute vec3 c; uniform mat4 m; varying vec3 v;' +
           'void main() { v = c; gl_Position = m * vec4(p, 1.0); }';
  var FS = 'precision mediump float; varying vec3 v;' +
           'void main() { gl_FragColor = vec4(v, 1.0); }';
  function shader(gl, type, source) {
    var s = gl.createShader(type);
    gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function start(figure) {
    var canvas = figure.querySelector('canvas'), still = figure.querySelector('img');
    var tools = figure.querySelector('.simscene-tools');
    var scene = JSON.parse(figure.querySelector('script[type="application/json"]').textContent);
    var gl = canvas.getContext('webgl', { antialias: true }) || canvas.getContext('experimental-webgl', { antialias: true });
    if (!gl) throw new Error('no WebGL');
    var data = build(scene), cam = startCamera(scene), home = startCamera(scene), queued = false;

    var prog = gl.createProgram();
    gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    function attribute(name, array, size, type, normalised) {
      var loc = gl.getAttribLocation(prog, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, normalised, 0, 0);
    }
    attribute('p', data.pos, 3, gl.FLOAT, false);
    attribute('c', data.col, 3, gl.UNSIGNED_BYTE, true);
    var uM = gl.getUniformLocation(prog, 'm');
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(scene.bg[0] / 255, scene.bg[1] / 255, scene.bg[2] / 255, 1);

    function draw() {
      queued = false;
      var ratio = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(1, Math.round(canvas.clientWidth * ratio)), h = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniformMatrix4fv(uM, false, matrix(cam, w / h));
      gl.drawArrays(gl.TRIANGLES, 0, data.count);
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
        if (c) c.hidden = true; if (t) t.hidden = true; if (s) s.hidden = false;
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
STYLE = {
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "legend.frameon": False, "lines.linewidth": 1.8, "svg.hashsalt": "simscene", "svg.fonttype": "path",
}


class Chart(Picture):
    """Base class of the charts, drawn with matplotlib.

    A subclass overrides plot(fig). Its first docstring line is its caption."""

    size = (10.0, 4.6)                            # inches
    layout = "constrained"

    def plot(self, fig):
        raise NotImplementedError

    @property
    def caption(self):
        return (self.__doc__ or type(self).__name__).strip().split("\n")[0]

    def figure(self):
        import matplotlib
        from matplotlib.figure import Figure
        with matplotlib.rc_context(STYLE):
            fig = Figure(figsize=self.size, layout=self.layout)
            self.plot(fig)
        return fig

    def bundle(self):
        import matplotlib
        fig, svg, png = self.figure(), io.StringIO(), io.BytesIO()
        with matplotlib.rc_context(STYLE):
            fig.savefig(svg, format="svg", metadata={"Date": None})
            fig.savefig(png, format="png", dpi=160)
        return {"image/svg+xml": svg.getvalue(),
                "image/png": base64.b64encode(png.getvalue()).decode("ascii"),
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
        left.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
        _count, _correct, segments, g, field = vc.saddle_check()
        limit = float(np.abs(field).max())
        right.pcolormesh(g, g, field.T, cmap="RdBu", vmin=-limit, vmax=limit, shading="nearest", edgecolors="white", linewidth=0.5)
        for a, b in segments:
            right.plot([a[0], b[0]], [a[1], b[1]], color=DARK, lw=2.5)
        mid = len(g) // 2
        right.plot([g[mid - 1], g[mid], g[mid], g[mid - 1], g[mid - 1]], [g[mid - 1], g[mid - 1], g[mid], g[mid], g[mid - 1]],
                   color=GREEN, lw=2.5)
        right.set_title("a saddle, the ambiguous cell in green", fontsize=13)
        right.set(xlabel="x", ylabel="y", aspect="equal")
        for ax in (left, right):
            ax.grid(False)


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
        left.set(xlabel="x", ylabel="y", aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
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
        left.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
        left.grid(False)
        right.plot(xs, part[:, row], color=DARK)
        right.axhline(0.0, color=DARK, lw=1.0)
        right.axhline(0.1, color=GREEN, lw=1.0)
        right.fill_between(xs, part[:, row], 0, where=part[:, row] < 0, color=RED, alpha=0.25)
        right.set_title("the field along the dotted line", fontsize=13)
        right.set(xlabel="x (m)", ylabel="distance to the surface (m)", xlim=(xs[0], xs[-1]))


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
            ax.set(xlabel="x (m)", aspect="equal", xlim=(-3.2, 3.8), ylim=(-3.6, 3.6))
            if k == 1:
                ax.set_ylabel("y (m)")


class ReductionChart(Chart):
    """Subsampling a terrain: a profile at each stride, and the error it costs"""

    def plot(self, fig):
        rows = vc.reduction_table()
        ys = np.linspace(-2.0, 2.0, 129)
        j = int(np.argmin(np.abs(ys - 0.5)))
        left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        for (stride, _points, _quads, _worst, _rms, X, Z), colour in zip(rows, (DARK, BLUE, GREEN, "#9467bd", "#ff7f0e", RED)):
            left.plot(X[:, 0], Z[:, j // stride], color=colour, marker="o" if stride > 2 else None, ms=3, label=f"stride {stride}")
        left.set(xlabel="x (m)", ylabel="height (m)")
        left.legend(fontsize=11)
        right.plot([r[1] for r in rows], [r[3] for r in rows], color=DARK, marker="o", ms=4)
        right.set(xscale="log", xlabel="points kept", ylabel="largest height error (m)")


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
        rows = (("GEOMETRY", "where each point is: 16 points", "points", "[(-2.0, -2.0, 0.03), (-2.0, -0.7, 0.08), ...]"),
                ("TOPOLOGY", "who is connected: 18 triangles", "faceVertexCounts, faceVertexIndices", "[3, 3, 3, ...]   [0, 4, 5,  0, 5, 1, ...]"),
                ("ATTRIBUTES", "data carried by the points", "primvars:displayColor, primvars:elevation", "[(0.27, 0.0, 0.33), ...]   [0.03, 0.08, ...]"))
        bx, bw, bh, gap = 7.3, 8.1, 2.0, 0.3
        for k, (head, what, names, example) in enumerate(rows):
            y = 7.9 - (k + 1) * bh - k * gap
            self.box(ax, bx, y, bw, bh)
            ax.text(bx + 0.3, y + bh - 0.42, head, va="center", fontsize=self.H2, color="#5f9400")
            ax.text(bx + 3.0, y + bh - 0.42, what, va="center", fontsize=self.SMALL, color=GREY)
            ax.text(bx + 0.3, y + bh - 1.02, names, va="center", fontsize=self.BODY, family="DejaVu Sans Mono", color=DARK)
            ax.text(bx + 0.3, y + bh - 1.60, example, va="center", fontsize=self.MONO, family="DejaVu Sans Mono", color=GREY)


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
        ax.text(8, 7.3, "scenes.lab02() in labs/scenes.py", ha="center", va="center", fontsize=self.H1, family="DejaVu Sans Mono", color=DARK)
        for x, (title, lines, fill, ink, edge) in zip((0.9, 5.9, 10.9), columns):
            self.box(ax, x, y, w, h, fill=fill, edge=edge)
            ax.text(x + w / 2, y + h - 0.6, title, ha="center", va="center", fontsize=self.H1, color=ink)
            for i, line in enumerate(lines):
                mono = line.endswith("()")
                ax.text(x + w / 2, y + h - 1.5 - i * 0.6, line, ha="center", va="center", fontsize=self.SMALL if mono else self.BODY,
                        color=ink if fill != "white" else GREY, family="DejaVu Sans Mono" if mono else None)
        for x in (0.9, 5.9):
            self.arrow(ax, (x + w + 0.12, y + h / 2), (x + w + 0.68, y + h / 2))
        ax.text(8, 1.9, "Every lab on this page has these three parts.", ha="center", va="center", fontsize=self.BODY, color=GREY)


class ColliderDiagram(Diagram):
    """What a convex hull does to a ring"""

    def plot(self, fig):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Rectangle
        ax = self.sheet(fig)
        ax.add_line(Line2D([8, 8], [0.6, 8.4], color="#cccccc", lw=1.3, zorder=0))
        for x0, title, solid in ((0.5, '"convexHull"', True), (8.5, '"sdf"   "convexDecomposition"   "none"', False)):
            cx, ring_y, ring_h, ground = x0 + 3.5, 4.0, 0.5, 1.7
            ax.text(cx, 8.0, title, ha="center", va="center", fontsize=self.H2, family="DejaVu Sans Mono", color=DARK)
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

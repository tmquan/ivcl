"""Write an interactive 3D view of a scene package, as one self-contained HTML file.

The view shows what scene.py puts on the USD stage (one mesh per object, the
cleft mesh, the three image sections and the wire frame of the block) with
WebGL 2 in the browser, so a notebook can turn the scene without Isaac Sim.
The meshes are simplified by vertex clustering until the whole scene has about
--triangles triangles, so that the file stays a few megabytes. No library is
loaded from the network.

Needs numpy and pillow. Usage:

    python web_view.py out/isaac/snemi3d --title SNEMI3D --triangles 250000

Writes <package>/viewer.html; the notebook shows it with IPython.display.HTML,
and a browser opens it as it is.
"""

import argparse
import base64
import gzip
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def cluster(points, faces, cell):
    """Vertex clustering on a grid of the given cell. Returns points and faces."""
    keys = np.floor(points / cell).astype(np.int64)
    keys -= keys.min(0)
    span = keys.max(0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    merged = np.stack([np.bincount(inverse, points[:, a], len(counts)) for a in range(3)], 1) / counts[:, None]
    f = inverse[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, 1), axis=0, return_index=True)
    f = f[np.sort(first)]
    used, f = np.unique(f, return_inverse=True)
    return merged[used].astype(np.float32), f.reshape(-1, 3).astype(np.int64)


def simplify(meshes, budget, cell0):
    """Pick one cell size for all meshes so that the total is near budget triangles."""
    total = sum(len(f) for _, f in meshes)
    if total <= budget:
        return meshes, 0.0
    cell = cell0 * np.sqrt(total / budget) * 0.5
    for _ in range(4):
        out = [cluster(p, f, cell) for p, f in meshes]
        total = sum(len(f) for _, f in out)
        if abs(total - budget) < 0.15 * budget:
            break
        cell *= np.sqrt(total / budget)
    return out, float(cell)


def build(package, title, budget):
    package = Path(package).expanduser().resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    entries = [o for o in manifest["objects"] if (package / o["file"]).exists()]
    meshes = []
    for o in entries:
        data = np.load(package / o["file"])
        meshes.append((data["points"].astype(np.float32), data["faces"].astype(np.int64)))
    before = sum(len(f) for _, f in meshes)
    meshes, cell = simplify(meshes, budget, min(manifest["mesh_voxel_um"]))

    size = np.array(manifest["size"], np.float64)
    lo = np.minimum(np.zeros(3), np.min([p.min(0) for p, _ in meshes], 0))
    hi = np.maximum(size, np.max([p.max(0) for p, _ in meshes], 0))
    scale = (hi - lo) / 65535.0

    blobs, objects, vbase, ibase = [], [], 0, 0
    positions, indices = [], []
    for o, (p, f) in zip(entries, meshes):
        positions.append(np.round((p - lo) / scale).astype(np.uint16))
        indices.append((f + vbase).astype(np.uint32))
        objects.append({"name": o["name"], "class": o["class"], "color": o["color"],
                        "voxels": o.get("voxels"), "first": ibase, "count": int(f.size)})
        vbase += len(p)
        ibase += int(f.size)
    blobs.append(np.concatenate(positions).tobytes())
    blobs.append(np.concatenate(indices).tobytes())

    slices = []
    for sl in manifest.get("slices", []):
        # The PNGs are stored for USD (first row at the top). WebGL takes the first row
        # of a byte upload as t = 0, at the bottom, so the rows are reversed here.
        image = np.asarray(Image.open(package / sl["file"]).convert("L"))[::-1]
        slices.append({"name": sl["name"], "corners": sl["corners"],
                       "width": int(image.shape[1]), "height": int(image.shape[0])})
        blobs.append(np.ascontiguousarray(image).tobytes())

    offsets, at = [], 0
    for b in blobs:
        offsets.append(at)
        at += len(b) + (-len(b)) % 4
    payload = b"".join(b + b"\0" * ((-len(b)) % 4) for b in blobs)
    scene = {"title": title, "lo": lo.tolist(), "scale": scale.tolist(), "size": size.tolist(),
             "vertices": vbase, "indices": ibase, "objects": objects, "slices": slices,
             "offsets": offsets, "unit": "\u00b5m"}
    data = base64.b64encode(gzip.compress(payload, 9)).decode()
    html = (TEMPLATE.replace("__ID__", "cv" + uuid.uuid4().hex[:10])
            .replace("__TITLE__", title)
            .replace("__SCENE__", json.dumps(scene, separators=(",", ":")))
            .replace("__DATA__", data))
    target = package / "viewer.html"
    target.write_text(html)
    return target, before, ibase // 3, cell


TEMPLATE = r"""<div id="__ID__" style="position:relative;width:100%;max-width:980px;height:600px;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden;font:12px/1.4 system-ui,sans-serif;color:#222">
<canvas style="width:100%;height:100%;display:block;cursor:grab;touch-action:none"></canvas>
<div style="position:absolute;top:8px;left:8px;background:rgba(255,255,255,.88);padding:5px 9px;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,.15);user-select:none">
<b>__TITLE__</b>&nbsp;&nbsp;
<label><input type="checkbox" data-k="slices" checked> sections</label>
<label><input type="checkbox" data-k="box" checked> box</label>
<label><input type="checkbox" data-k="spin"> spin</label>
<label>&nbsp;objects <input type="range" data-k="count" min="1" value="1000" style="width:90px;vertical-align:middle"></label>
<button data-k="reset" style="margin-left:6px">reset view</button>
</div>
<div data-k="legend" style="position:absolute;top:8px;right:8px;max-width:300px;background:rgba(255,255,255,.88);padding:5px 9px;border-radius:5px;user-select:none;display:none"></div>
<div data-k="info" style="position:absolute;bottom:8px;left:8px;background:rgba(255,255,255,.88);padding:4px 8px;border-radius:5px;user-select:none">drag: turn &middot; shift-drag or right-drag: move &middot; wheel: zoom &middot; click an object: show it alone &middot; double-click: reset</div>
</div>
<script>
(async function () {
const root = document.getElementById("__ID__");
const S = __SCENE__;
const B64 = "__DATA__";
const canvas = root.querySelector("canvas"), info = root.querySelector('[data-k="info"]');
const ui = k => root.querySelector('[data-k="' + k + '"]');
const gl = canvas.getContext("webgl2", {antialias: true});
if (!gl) { info.textContent = "This browser has no WebGL 2."; return; }
let bytes = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
const buf = bytes.buffer;

// Geometry: positions in the unit of the package, normals from the faces.
const q = new Uint16Array(buf, S.offsets[0], S.vertices * 3);
const idx = new Uint32Array(buf, S.offsets[1], S.indices);
const pos = new Float32Array(q.length), nrm = new Float32Array(q.length);
for (let i = 0; i < q.length; i++) pos[i] = S.lo[i % 3] + q[i] * S.scale[i % 3];
for (let t = 0; t < idx.length; t += 3) {
  const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
  const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
  const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  for (const k of [a, b, c]) { nrm[k] += nx; nrm[k + 1] += ny; nrm[k + 2] += nz; }
}

function program(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, "#version 300 es\nprecision highp float;\n" + src);
    gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
    gl.attachShader(p, s);
  }
  gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
  const u = {}; for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const n = gl.getActiveUniform(p, i).name; u[n] = gl.getUniformLocation(p, n); }
  return {p, u};
}
const mesh = program(
  `layout(location=0) in vec3 P; layout(location=1) in vec3 N; uniform mat4 V, M;
   out vec3 n; out vec3 e; void main(){ vec4 w = V * vec4(P, 1.0); e = w.xyz; n = mat3(V) * N; gl_Position = M * w; }`,
  `in vec3 n; in vec3 e; uniform vec3 C; out vec4 o;
   void main(){ vec3 m = normalize(n); vec3 v = normalize(-e); if (dot(m, v) < 0.0) m = -m;
     vec3 k = normalize(vec3(-0.4, 0.6, 0.7)), f = normalize(vec3(0.5, -0.3, 0.8));
     float d = 0.42 + 0.48 * max(dot(m, k), 0.0) + 0.22 * max(dot(m, f), 0.0) + 0.12 * max(dot(m, v), 0.0);
     float s = pow(max(dot(reflect(-k, m), v), 0.0), 24.0) * 0.12;
     o = vec4(min(C * d + s, vec3(1.0)), 1.0); }`);
const pick = program(
  `layout(location=0) in vec3 P; uniform mat4 V, M; void main(){ gl_Position = M * V * vec4(P, 1.0); }`,
  `uniform vec3 C; out vec4 o; void main(){ o = vec4(C, 1.0); }`);
const flat = program(
  `layout(location=0) in vec3 P; layout(location=2) in vec2 T; uniform mat4 V, M; out vec2 t;
   void main(){ t = T; gl_Position = M * V * vec4(P, 1.0); }`,
  `in vec2 t; uniform sampler2D I; uniform vec3 C; uniform float X; out vec4 o;
   void main(){ o = X > 0.5 ? vec4(vec3(texture(I, t).r), 1.0) : vec4(C, 1.0); }`);

const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
for (const [loc, data] of [[0, pos], [1, nrm]]) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
}
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

// Image sections: textured quads; then the 12 edges of the block.
gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
const quads = S.slices.map((sl, i) => {
  const v = gl.createVertexArray(); gl.bindVertexArray(v);
  const c = sl.corners, uv = [[0, 0], [1, 0], [1, 1], [0, 1]], data = [];
  for (const j of [0, 1, 2, 0, 2, 3]) data.push(...c[j], ...uv[j]);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 20, 12);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, sl.width, sl.height, 0, gl.RED, gl.UNSIGNED_BYTE,
                new Uint8Array(buf, S.offsets[2 + i], sl.width * sl.height));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return {vao: v, tex};
});
const Z = S.size, box = [];
const corner = i => [(i & 1) ? Z[0] : 0, (i & 2) ? Z[1] : 0, (i & 4) ? Z[2] : 0];
for (const [a, b] of [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]) box.push(...corner(a), 0, 0, ...corner(b), 0, 0);
const boxVao = gl.createVertexArray(); gl.bindVertexArray(boxVao);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(box), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
gl.bindVertexArray(null);

// Camera: the overview camera of scene.py (50 mm lens on a 36 mm aperture, Z up).
const centre0 = Z.map(v => v / 2), radius = Math.hypot(...Z) / 2, fov = 2 * Math.atan(18 / 50);
let cam;
const reset = () => { cam = {az: -60, el: 28, dist: 3 * radius, at: centre0.slice()}; };
reset();
const sub = (a, b) => a.map((v, i) => v - b[i]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => { const l = Math.hypot(...a); return a.map(v => v / l); };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function basis() {
  const a = cam.az * Math.PI / 180, e = cam.el * Math.PI / 180;
  const eye = cam.at.map((v, i) => v + cam.dist * [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)][i]);
  const f = unit(sub(cam.at, eye)), r = unit(cross(f, [0, 0, 1])), u = cross(r, f);
  return {eye, f, r, u};
}
function matrices() {
  const {eye, f, r, u} = basis();
  const V = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, eye), -dot(u, eye), dot(f, eye), 1];
  const near = cam.dist * 0.02, far = cam.dist + radius * 4, t = 1 / Math.tan(fov / 2), asp = canvas.width / canvas.height;
  const s = Math.min(1, asp);
  const M = [t * s / asp, 0, 0, 0, 0, t * s, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
  return {V: new Float32Array(V), M: new Float32Array(M)};
}

// Drawing. The objects come largest first; "count" keeps the first ones, "alone" is the one clicked on.
const nObj = S.objects.filter(o => o.class !== "cleft").length;
ui("count").max = nObj; ui("count").value = nObj;
let alone = -1, dirty = true, fb = null;
const shown = i => S.objects[i].class === "cleft" || i < +ui("count").value;
function scene(target) {
  const {V, M} = matrices();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const P = target === "pick" ? pick : mesh;
  gl.useProgram(P.p); gl.uniformMatrix4fv(P.u.V, false, V); gl.uniformMatrix4fv(P.u.M, false, M);
  gl.bindVertexArray(vao);
  S.objects.forEach((o, i) => {
    if (alone >= 0 ? alone !== i : !shown(i)) return;
    if (target === "pick") gl.uniform3f(P.u.C, ((i + 1) & 255) / 255, ((i + 1) >> 8) / 255, 0);
    else gl.uniform3fv(P.u.C, o.color);
    gl.drawElements(gl.TRIANGLES, o.count, gl.UNSIGNED_INT, o.first * 4);
  });
  if (target === "pick") return;
  gl.useProgram(flat.p); gl.uniformMatrix4fv(flat.u.V, false, V); gl.uniformMatrix4fv(flat.u.M, false, M);
  if (ui("slices").checked) {
    gl.uniform1f(flat.u.X, 1); gl.activeTexture(gl.TEXTURE0); gl.uniform1i(flat.u.I, 0);
    for (const s of quads) { gl.bindVertexArray(s.vao); gl.bindTexture(gl.TEXTURE_2D, s.tex); gl.drawArrays(gl.TRIANGLES, 0, 6); }
  }
  if (ui("box").checked) {
    gl.uniform1f(flat.u.X, 0); gl.uniform3f(flat.u.C, 0.3, 0.3, 0.3);
    gl.bindVertexArray(boxVao); gl.drawArrays(gl.LINES, 0, 24);
  }
  gl.bindVertexArray(null);
}
function frame() {
  const w = Math.round(canvas.clientWidth * devicePixelRatio), h = Math.round(canvas.clientHeight * devicePixelRatio);
  if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; fb = null; dirty = true; }
  if (ui("spin").checked) { cam.az += 0.3; dirty = true; }
  if (dirty && canvas.width) { scene("screen"); dirty = false; }
  if (root.isConnected) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function picked(x, y) {
  if (!fb) {
    fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    const c = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, c);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, canvas.width, canvas.height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, c);
    const d = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, d);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, canvas.width, canvas.height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, d);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb); scene("pick");
  const px = new Uint8Array(4), r = canvas.getBoundingClientRect();
  gl.readPixels(Math.round((x - r.left) * canvas.width / r.width), Math.round((r.bottom - y) * canvas.height / r.height), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return px[0] + (px[1] << 8) - 1;
}

// Legend of the classes that have their own colour.
const LONG = {cleft: "synaptic clefts", mito: "mitochondria", er: "endoplasmic reticulum", golgi: "Golgi",
              ves: "vesicles", endo: "endosomes", lyso: "lysosomes", perox: "peroxisomes", mt: "microtubules", ld: "lipid droplets"};
const kinds = new Map(S.objects.filter(o => o.class !== "object").map(o => [o.class, o.color]));
if (kinds.size) {
  ui("legend").style.display = "block";
  ui("legend").innerHTML = [...kinds].map(([k, c]) => '<span style="white-space:nowrap;margin-right:8px"><span style="display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px;background:rgb(' + c.map(v => Math.round(v * 255)) + ')"></span>' + (LONG[k] || k) + "</span>").join(" ");
}
function describe() {
  if (alone < 0) { info.textContent = S.objects.filter((o, i) => shown(i)).length + " meshes, " + S.size.map(v => v.toFixed(2)).join(" \u00d7 ") + " " + S.unit + " \u00b7 click an object to show it alone"; return; }
  const o = S.objects[alone];
  info.textContent = o.name + (o.class !== "object" ? " (" + (LONG[o.class] || o.class) + ")" : "") + (o.voxels ? ", " + o.voxels.toLocaleString() + " voxels" : "") + " \u00b7 click elsewhere to show all";
}

// Mouse.
let drag = null;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => { drag = {x: e.clientX, y: e.clientY, moved: 0, pan: e.shiftKey || e.button === 2}; try { canvas.setPointerCapture(e.pointerId); } catch (_) {} canvas.style.cursor = "grabbing"; });
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
  if (drag.pan) {
    const {r, u} = basis(), k = cam.dist * Math.tan(fov / 2) * 2 / canvas.clientHeight;
    cam.at = cam.at.map((v, i) => v - (r[i] * dx - u[i] * dy) * k);
  } else { cam.az -= dx * 0.4; cam.el = Math.max(-89, Math.min(89, cam.el + dy * 0.4)); }
  dirty = true;
});
canvas.addEventListener("pointerup", e => {
  canvas.style.cursor = "grab";
  if (drag && drag.moved < 4 && e.button === 0) { const i = picked(e.clientX, e.clientY); alone = i >= 0 && i !== alone ? i : -1; describe(); dirty = true; }
  drag = null;
});
canvas.addEventListener("wheel", e => { e.preventDefault(); cam.dist = Math.max(radius * 0.2, Math.min(radius * 20, cam.dist * Math.exp(e.deltaY * 0.0015))); dirty = true; }, {passive: false});
canvas.addEventListener("dblclick", () => { reset(); alone = -1; describe(); dirty = true; });
ui("reset").addEventListener("click", () => { reset(); dirty = true; });
for (const k of ["slices", "box", "spin", "count"]) ui(k).addEventListener("input", () => { describe(); dirty = true; });
describe();
})().catch(e => { const r = document.getElementById("__ID__"); if (r) r.lastElementChild.textContent = "viewer error: " + e; });
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("--title", default=None)
    ap.add_argument("--triangles", type=int, default=250_000, help="triangle budget of the whole scene")
    a = ap.parse_args()
    target, before, after, cell = build(a.package, a.title or Path(a.package).name, a.triangles)
    print(f"{target}: {after:,} of {before:,} triangles"
          + (f" (clustering cell {cell * 1000:.0f} nm)" if cell else "")
          + f", {target.stat().st_size / 1e6:.1f} MB")

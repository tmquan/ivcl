"""Lab 9 (handbook 1.5.4): glyphs. One cone per sample, turned
along the surface normal and stretched by the slope.

Run (env isaac):  python labs/lab09_glyphs.py --stream
Output:           out/lab09.usda
"""
from simlab import finish, new_stage, parse_args, save, show, start

args = parse_args(n=65, every=4)
app = start(args.stream)

from pxr import Gf, UsdGeom, UsdShade

import look
import scenes

stage = new_stage()
prims, view = scenes.lab09_terrain(args.n)
show(stage, prims, view)

# Where each cone stands, how it is turned, how long it is.
bases, quats, lengths, normals = scenes.lab09_glyphs(args.n, args.every)

# ONE cone is stored; the instancer draws it once per sample.
glyphs = UsdGeom.PointInstancer.Define(stage, "/World/Glyphs")
cone = look.mesh("cone", *look.cone_shape(scenes.LAB09_RADIUS),
                 look.WHITE_RGB, lit=False)
matte = UsdShade.Material.Get(stage, "/World/Look/Matte")
proto = look.usd_mesh(stage, "/World/Glyphs/Proto", cone, matte)
glyphs.CreatePrototypesRel().AddTarget(proto.GetPath())
glyphs.CreateProtoIndicesAttr([0] * len(bases))
glyphs.CreatePositionsAttr(bases.tolist())
glyphs.CreateOrientationsAttr(
    [Gf.Quath(*map(float, q)) for q in quats])
glyphs.CreateScalesAttr([(1.0, 1.0, float(s)) for s in lengths])

steep = lengths.argmax()
print(f"{len(bases)} glyphs from one prototype prim")
print(f"steepest sample: normal {normals[steep].round(2)},"
      f" cone length {lengths[steep]:.3f} m")
save("lab09.usda")
finish(app, args.stream)

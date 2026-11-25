"""Lab 1 (handbook 1.1): a dataset is topology + geometry +
attributes, and algorithms are transformations of them.

Run (env isaac):  python labs/lab01_dataset.py
                  python labs/lab01_dataset.py --stride 4
Output:           out/lab01.usda
"""
from simlab import finish, new_stage, parse_args, save, show, start

args = parse_args(n=65, stride=1)
app = start(args.stream)

import os

import numpy as np
from pxr import Sdf, UsdGeom

import scenes

stage = new_stage()

# SOURCE: a height field becomes a mesh prim on the stage.
prims, view = scenes.lab01(args.n, args.stride)
paths = show(stage, prims, view)
mesh = UsdGeom.Mesh(stage.GetPrimAtPath(paths["terrain"]))


def read():
    return (np.array(mesh.GetPointsAttr().Get()),
            np.array(mesh.GetFaceVertexCountsAttr().Get()),
            np.array(mesh.GetFaceVertexIndicesAttr().Get()))


pts0, cnt0, idx0 = read()
print("--- the dataset ---")
print(f"geometry : points              {pts0.shape}")
print(f"topology : faceVertexCounts    {cnt0.shape}"
      f"  (all {cnt0[0]}-sided)")
print(f"topology : faceVertexIndices   {idx0.shape}")
print(f"first face uses points {idx0[:3]}")
print("attribute: primvars:displayColor  (one colour per point)")

# GEOMETRIC transformation: double every height.
taller, _ = scenes.lab01(args.n, args.stride, taller=True)
mesh.GetPointsAttr().Set(taller[0]["points"].tolist())
pts1, cnt1, idx1 = read()
print("--- after a geometric transformation (z * 2) ---")
print("max height   :", round(float(pts0[:, 2].max()), 3), "->",
      round(float(pts1[:, 2].max()), 3))
print("topology same:", bool(np.array_equal(idx0, idx1)
                              and np.array_equal(cnt0, cnt1)))

# ATTRIBUTE transformation: create a scalar from the geometry.
elevation = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
    "elevation", Sdf.ValueTypeNames.FloatArray,
    UsdGeom.Tokens.vertex)
elevation.Set(pts1[:, 2].tolist())
pts2, cnt2, idx2 = read()
print("--- after an attribute transformation (elevation) ---")
print("new attribute:", elevation.GetName(),
      f"({len(elevation.Get())} values, one per point)")
print("geometry same:", bool(np.array_equal(pts1, pts2)),
      " topology same:", bool(np.array_equal(idx1, idx2)))

path = save("lab01.usda")
print(f"file size: {os.path.getsize(path) / 1024:.0f} kB"
      f" for {len(pts2)} points")
finish(app, args.stream)

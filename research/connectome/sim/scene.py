"""Build the USD stage of a scene package. Uses only pxr and numpy.

Used by ``render_kit.py`` (the Kit runtime of Isaac Sim, without the Isaac Sim
extensions) and by the notebook, when Isaac Sim starts in its kernel.

Stage layout (one unit is one unit of the package, micrometres):

    /World/Volume/Objects/<name>     one mesh per object
    /World/Volume/Slices/<name>      image slices on the far faces of the block
    /World/Volume/Box                wire frame of the block
    /World/Looks/<name>              one material per object
    /World/Lights, /World/Ground, /World/Cameras/Overview

The look follows the model viewer of https://cellmapchallenge.janelia.org/:
one flat colour per class, a white background, even light from all sides, and
a soft shadow under the block.
"""

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

CAMERA = "/World/Cameras/Overview"


def build(stage, package, aspect=1.0):
    """Fill ``stage`` from the package folder. Returns (centre, radius)."""
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    volume = UsdGeom.Xform.Define(stage, "/World/Volume")
    if manifest.get("up_axis", "Z") == "Y":
        # The package is Y-up (glTF); the stage is Z-up.
        volume.AddRotateXOp().Set(90.0)
    UsdGeom.Scope.Define(stage, "/World/Volume/Objects")
    UsdGeom.Scope.Define(stage, "/World/Looks")

    def preview_material(name, color, metallic, roughness, opacity=1.0, texture=None):
        path = f"/World/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if texture is None:
            diffuse.Set(Gf.Vec3f(*[float(c) for c in color]))
        else:
            reader = UsdShade.Shader.Define(stage, f"{path}/St")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(stage, f"{path}/Image")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result")
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(image.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def add_mesh(path, points, faces, material):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(np.ascontiguousarray(faces, np.int32).reshape(-1)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        lo, hi = points.min(0), points.max(0)
        mesh.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return mesh

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for obj in manifest["objects"]:
        data = np.load(package / obj["file"])
        points, faces = data["points"], data["faces"]
        name = obj["name"]
        look = preview_material(name, obj["color"], obj["metallic"], obj["roughness"],
                                obj.get("opacity", 1.0))
        mesh = add_mesh(f"/World/Volume/Objects/{name}", points, faces, look)
        mesh.GetPrim().SetCustomDataByKey("connectome:class", obj["class"])
        lo, hi = np.minimum(lo, points.min(0)), np.maximum(hi, points.max(0))
    if "size" in manifest:
        lo, hi = np.zeros(3), np.array(manifest["size"], float)

    if manifest.get("slices"):
        UsdGeom.Scope.Define(stage, "/World/Volume/Slices")
    for sl in manifest.get("slices", []):
        look = preview_material(sl["name"], (1, 1, 1), 0.0, 1.0, texture=f"./{sl['file']}")
        quad = add_mesh(f"/World/Volume/Slices/{sl['name']}",
                        np.array(sl["corners"], np.float32), np.array([[0, 1, 2], [0, 2, 3]]), look)
        st = UsdGeom.PrimvarsAPI(quad.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Wire frame of the block.
    corners = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    edges = [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    box = UsdGeom.BasisCurves.Define(stage, "/World/Volume/Box")
    box.CreateTypeAttr(UsdGeom.Tokens.linear)
    box.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in corners[i]]) for i in edges])
    box.CreateCurveVertexCountsAttr([2] * 12)
    size = float(np.linalg.norm(hi - lo))
    box.CreateWidthsAttr([size * 0.002])
    box.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    box.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.3)])

    # World-space bounds of the block, after the change of up axis.
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Volume/Objects")).ComputeAlignedRange()
    wlo, whi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    if "size" in manifest:
        wlo, whi = lo, hi
    center = (wlo + whi) / 2
    radius = float(np.linalg.norm(whi - wlo)) / 2

    # Light: white from all sides, and one soft light from above for the shadow.
    UsdGeom.Scope.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(12.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25.0, 20.0, 0.0))

    # Ground: a white plane under the block, to catch the shadow.
    ground_look = preview_material("ground", (1, 1, 1), 0.0, 1.0)
    g = radius * 20
    z0 = float(wlo[2]) - radius * 0.02
    add_mesh("/World/Ground",
             np.array([[center[0] - g, center[1] - g, z0], [center[0] + g, center[1] - g, z0],
                       [center[0] + g, center[1] + g, z0], [center[0] - g, center[1] + g, z0]],
                      np.float32),
             np.array([[0, 1, 2], [0, 2, 3]]), ground_look)

    # Camera.
    UsdGeom.Scope.Define(stage, "/World/Cameras")
    camera = UsdGeom.Camera.Define(stage, "/World/Cameras/Overview")
    camera.CreateFocalLengthAttr(50.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * aspect)
    camera.CreateClippingRangeAttr(Gf.Vec2f(radius * 0.01, radius * 100))
    UsdGeom.Xformable(camera).AddTransformOp()

    set_view(stage, center, radius, -60.0)
    stage.SetMetadata("comment", f"{manifest['name']}: {manifest['source']}. "
                                 f"Unit: {manifest['unit']}.")
    return center, radius


def set_view(stage, center, radius, azimuth_deg, elevation_deg=28.0, distance=3.0):
    """Point the overview camera at the block from the given direction."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    eye = np.asarray(center) + radius * distance * np.array(
        [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*np.asarray(center).tolist()),
                                   Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA)).GetOrderedXformOps()[0].Set(view.GetInverse())

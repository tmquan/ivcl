# Representation

The same tissue is held in three forms: Voxel, Frame and Graph. Each is best at one thing, and the agent moves information between them.

| | Voxel | Frame | Graph |
|---|---|---|---|
| **What it is** | A value at every point of a regular grid | The medial axes of one connected component, with a radius at every point | The components and the links between them |
| **Answers** | Where exactly is the boundary? | What shape is this object, and how does it run? | What is connected to what? |
| **Scope** | The whole block | One connected component | All components |
| **Size** | Largest | Small | Smallest |
| **In Isaac Sim** | NanoVDB volume in Warp | Curves and splats under the prim of the component | Prims and relationships of the stage |

A **connected component** is a maximal set of touching voxels with one label: one neurite, one mitochondrion, one cleft, one piece of endoplasmic reticulum. Where a dataset gives only a class and no instances, the components are those of the class mask.

## Voxel

The voxel form holds what was measured and what must be delivered.

| Volume | Type | Levels |
|---|---|---|
| Raw image | 8-bit | All levels of the [mipmap](multiscale.md) |
| Segmentation, where a dataset has it | 64-bit identifier | All levels, by most frequent label |
| Class type, where a dataset has it | 8-bit class | All levels |
| Membrane probability, predicted | 8-bit | Working level |
| Output labels | 64-bit identifier | Working level |

Storage is OME-Zarr on disk, chunked 128 × 128 × 32. Tradeoff: Zarr is chosen because the CellMap data is already in it; the tools that need Neuroglancer precomputed format get an export.

In memory, a block is a set of NanoVDB grids on the GPU. NanoVDB is sparse, so label and output volumes cost memory only where there is something. Warp kernels sample these grids at any point given in nanometres: trilinear for the image, nearest neighbour for labels.

A **block** is 1024 × 1024 × 256 voxels at the working level, with a halo of 128 × 128 × 32. It is the unit of work and of restart.

## Frame

The medial axis of a shape is the set of centres of the largest balls that fit inside it. With the radius of the ball at each centre, it is a complete description: the union of the balls is the shape again.

What the medial axis looks like depends on the component.

| Component | Its medial axis | Examples |
|---|---|---|
| Tube | A curve, with branch points | Axon, dendrite, spine neck, microtubule |
| Sheet | A surface | Synaptic cleft, cisterna of the endoplasmic reticulum or Golgi |
| Ball | A point | Vesicle, lysosome, lipid droplet |
| Mixed | Curves and surfaces joined | A dendrite with a soma; a mitochondrion with a flat part |

So a Frame is more than a curve skeleton: it has curve parts and surface parts. The Frame of this design is a set of **nodes** and **edges** per component.

| | Holds |
|---|---|
| **Node** | A centre; three radii with their directions; a shape type (line, sheet, blob); a level |
| **Edge** `adjacent` | Two nodes of the component whose ellipsoids overlap |

A node is a ball stretched to an ellipsoid: three radii in place of one. A tube needs far fewer ellipsoids along its axis than balls, and a sheet far fewer across its surface. A node of a Frame is exactly one Gaussian of [Features](features.md). The Frame is not stored beside the Gaussians: it is the Gaussians of one component and their `adjacent` edges.

From the Frame of a component:

| Wanted | Taken from the Frame |
|---|---|
| The skeleton of a neurite, as a tree | The line nodes, joined by the minimum spanning tree of the `adjacent` edges weighted by distance |
| Branch points and ends | Nodes of the tree with three or more neighbours, or with one |
| Path length, caliber along the path | Sum of distances between neighbours; the middle radius of each node |
| The shape again | The union of the ellipsoids |

The membrane between cells is a component too, of class membrane. Its Frame is a surface: the set of cell boundaries. It is the counterpart of the Frames of the cells, and the final voxel step uses it (see [Features](features.md)).

In USD, per component:

```text
/World/Block_{i}_{j}_{k}/Objects/o_{id}      Xform; class type as a semantic label
    /Frame      BasisCurves through the line nodes, with widths = caliber
    /Splats     all nodes: positions, orientations, radii
```

## Graph

The Graph has one node per component and three kinds of edge.

| Edge | Between | Meaning | Found by |
|---|---|---|---|
| `same` | Two components | They are one object, cut apart by an error or by a block border | The agent, or stitching |
| `contact` | Two components | They touch | Overlap between a node of one Frame and a node of the other |
| `synapse` | A presynaptic and a postsynaptic component | A synapse, with a direction | A cleft component in contact with both |

An **object** is a set of components joined by `same` edges. A **neuron** is an object with one cell body. Joining two components adds a `same` edge and an `adjacent` edge between their Frames; no voxel is changed.

Overlap between two nodes has a closed form (see [Features](features.md)), so the Graph is built from the Frames without touching voxels.

In USD, the Graph is the stage itself: each component prim has relationships `connectome:same`, `connectome:contact` and `connectome:synapseTo` to other prims, and attributes for volume, path length, class scores and confidence. For analysis outside Isaac Sim it is exported as three Parquet tables: Frame nodes, branches, synapses.

## Moving between the three

| From | To | Operation |
|---|---|---|
| Voxel | Frame | Distance transform of the component; its ridge is the medial axis; nodes are placed along it and fitted |
| Frame | Voxel | Draw the union of the ellipsoids; then a watershed on the membrane map puts the boundary exactly |
| Frame | Graph | Each Frame becomes one node; overlaps between Frames become edges |
| Graph | Frame | A `same` edge joins two Frames; cutting it, or cutting an `adjacent` edge, separates them |
| Voxel | Graph | Through the Frame. Directly, from touching labels, as a check |
| Graph | Voxel | Relabel: each voxel takes the object of its component |

Each operation is written once, as a Warp kernel or a USD operation, and tested alone. The test of the whole is a round trip: true labels to Frames to labels, with the loss measured (milestone M3).

## Views

The agent does not read the three forms directly. It looks at them through a camera, and what the camera gives is a **view**. A view is an observation, not a fourth representation.

The camera is a model of the microtome and the microscope together. It is orthographic, and its near and far clipping planes are one section thickness apart, so it sees a **slab**. Everything inside the slab is summed. Nothing hides anything else, because an EM image is a thin slice that electrons pass through or scan across.

| Parameter | Meaning | CREMI | CellMap |
|---|---|---|---|
| Slab thickness | Section thickness | 40 nm | 8 nm |
| Pixel size | Level of the mipmap | 4 nm at level 0 | 8 nm at level 0 |
| Size | Field of view | 256 × 256 pixels by default | The same |
| Orientation | Normal of the slab | Along z, or along a Frame | Any |

One camera gives aligned channels of the same slab.

| Channel | Shows | Made from |
|---|---|---|
| Raw | The image | Voxel: sampled over the slab |
| Model | The image that the Frames predict | Frame: the slab renderer |
| Residual | What the Frames do not explain | Raw minus model |
| Identity, class | Which component, and of what kind | Frame: the same renderer |
| Labels, in training | The truth | Voxel |

Two ways of moving the camera are used:

- **Along the sections.** The camera steps along z, one slab per view. This is the stack as acquired.
- **Along a Frame.** The camera follows the curve of a neurite with its normal on the tangent. The neurite stays in the centre of every view as a round profile. A merge shows as a profile that doubles; a split, as a profile that ends while the image goes on.

```{admonition} Oblique views and anisotropy
:class: tip
A view along a Frame is oblique to the sections. Its model channel is exact at any angle, because Frame nodes are continuous. Its raw channel is interpolated between sections, and with sections 10 times thicker than a pixel, interpolation invents membranes. The rule: an oblique raw view is taken only at a level where the voxel is within a factor of 2.5 of isotropic. For CREMI that is level 2 (16 × 16 × 40 nm) and coarser. For CellMap it is every level.
```

Two renderers make views, both inside Isaac Sim, both reading the pose of the same camera prim.

| Renderer | Used for | Gradients |
|---|---|---|
| Slab renderer, in Warp | Fitting, training, the agent's observation | Yes |
| RTX, with a render product and Replicator annotators | Display and review: meshes, Frames and ellipsoids, with instance and class annotations | No |

## Tests

- View: the raw channel at level 0, normal along z, equals the numpy slice; at level ℓ, the slice of the level-ℓ volume.
- Frame: for a synthetic tube, Y, sheet and ball, the nodes lie on the true medial axis within one voxel, with the right radii, shape types, branch points and ends.
- Frame: the union of the ellipsoids of a Frame covers the component with an intersection over union above 0.9 before the watershed.
- Graph: on synthetic tubes that touch, cross or branch, the `contact` edges are the expected ones.
- Round trip: labels to Frames to labels on CREMI A.

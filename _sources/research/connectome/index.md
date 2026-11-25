# Connectome Agent

An agent that reconstructs an electron-microscopy (EM) volume inside Isaac Sim. It holds the tissue as a set of 3D Gaussians, looks at it through a camera that cuts sections, and edits the Gaussians until they explain the image, the segmentation and the class of every structure.

Three words are used in a fixed sense throughout. **Voxel** is the image grid. **Frame** is the medial axes of one connected component, with a radius at every point. **Graph** is the set of components and their links. A 2D image from the camera is called a **view**, never a frame.

```{note}
This is a design, written on 23 November 2026. It replaces an earlier design built around a flood-filling network. One part has been run: a scene of the CellMap crop 313 was built and rendered on the lab's DGX Spark with the Kit runtime of Isaac Sim. Everything else is to be built. Numbers marked *estimate* are arithmetic, not measurements.
```

## The idea

The design has four axes. Each has three members, and each member has a job.

| Axis | Members | Page |
|---|---|---|
| **Data representation** | Voxel, Frame, Graph | [Representation](representation.md) |
| **Data supervision** | Raw, segmentation, class type | [Supervision](supervision.md) |
| **Feature** | Geometry, anisotropy, Gaussian splats | [Features](features.md) |
| **Multi-scale** | Coarse-to-fine, fine-to-coarse, mipmap | [Multi-scale](multiscale.md) |

One object ties the four together: the **Gaussian**. A Gaussian is an ellipsoid of stained material with a centre, a shape, an intensity, an identity and a class.

- It can be drawn into voxels. The Gaussians of one connected component, with their neighbour links, are its Frame. The components and their contacts are the Graph. So it joins the three representations.
- Its intensity is trained by the raw image, its identity by the segmentation, its class by the class labels. So it carries the three kinds of supervision.
- Its centre and shape are the geometry; the ratio of its axes is the anisotropy. So it is the feature.
- Small Gaussians merge into large ones and large ones split into small ones. So it has a scale, and a pyramid of them is a mipmap.

The [agent](agent.md) is the loop that edits this set.

## Outputs

| Output | How it is obtained from the Gaussians |
|---|---|
| Dense segmentation of neurites, mitochondria, clefts (and the organelle classes of CellMap) | Gaussians are the seeds; a watershed on the membrane map at the finest level gives each voxel to one seed |
| A skeleton of every neurite | The Frame of the neurite: its elongated Gaussians, joined by a minimum spanning tree |
| A graph of neurons, branches, synapses | The Graph: one node per component, with the contact and synapse links between Frames |

## Isaac Sim

Isaac Sim is the runtime, not a viewer beside it. Each part of the design maps to a part of Isaac Sim.

| Part of the design | In Isaac Sim | Seen in the lab's build |
|---|---|---|
| Voxel | NanoVDB volumes on the GPU through Warp; a USD `Volume` prim for display | `omni.volume`, `omni.warp.core` 1.17 |
| Frame | Curves with widths under the prim of each component; its nodes as splats | USD 25.11 |
| Graph | The USD stage: one prim per component, relationships for links, layers for versions | USD 25.11 |
| Views | A camera prim with a render product; Replicator annotators for colour, instance and class | `omni.replicator.core` |
| Gaussians | Warp arrays for fitting; the USD particle-field schema and NuRec for display | `omni.usd.schema.usd_particle_field`, `isaacsim.replicator.nurec_utils` |
| Differentiable rendering | Warp kernels with automatic differentiation | `omni.warp.core` |
| The agent's body | A camera rig that the policy moves | Camera prim |
| Many blocks at once | A vectorised environment of the Isaac Lab kind | Needs Isaac Lab |
| Human review | The Isaac Sim window, streamed to the Mac | `isaac-sim.streaming.sh` |

```{admonition} State of the build on the Spark
:class: warning
On `spark-tiger`, the Isaac Sim 6.2.0 build starts, but 28 of its own extensions fail to load: their compiled parts were built and never copied into the application (`isaacsim.core.simulation_manager.bindings` is the first one missing). With `SimulationApp`, the renderer then returns empty images. The Kit runtime of the same build, started with the renderer and Replicator only, does render. Milestone M0 is to finish that build. Until then the prototype uses the Kit route.
```

## Datasets

The redesign adds the CellMap challenge data to the three volumes named before, because it is the only one with dense class labels for many organelles, and the only isotropic one.

| | CREMI | SNEMI3D | MICrONS | CellMap |
|---|---|---|---|---|
| **Tissue** | Fly brain | Mouse cortex | Mouse visual cortex | Cells and tissues, several species |
| **Imaging** | Serial-section TEM | Serial-section SEM | Serial-section TEM | FIB-SEM |
| **Voxel, nm** | 4 × 4 × 40 | 6 × 6 × 29 | 4 × 4 × 40 | Isotropic, 8 or finer |
| **Raw** | Yes | Yes | Yes | Yes |
| **Segmentation** | Neurites, dense | Neurites, dense | Automatic; some neurons proofread | Instances for some classes |
| **Class type** | Clefts, with partners | None | Synapse table; cell types for some neurons | Dozens of organelle classes, dense in labelled crops |
| **Role** | First slice | Second slice | The full volume | Class types and the isotropic case |

No dataset has all three kinds of supervision for all structures. That is why supervision is an axis of its own: each kind trains its own part of the Gaussian, and a volume with only raw data still trains the first part.

## Decisions

| # | Decision | Turned down | Why, and what it costs |
|---|---|---|---|
| D1 | The state of the agent is a set of anisotropic 3D Gaussians. | A voxel canvas per object (flood filling). | One primitive serves Voxel, Frame and Graph, and is trained by all three kinds of supervision. The cost: a Gaussian is smooth, so the exact membrane position comes from a final voxel step. |
| D2 | The agent's view is a **slab**: the sum of everything inside one section thickness. No depth order, no occlusion. | Perspective splatting with alpha blending, as in novel-view synthesis. | An EM image is a thin slice of tissue, not a view of a surface. The slab integral of a Gaussian has a closed form. The cost: the standard splat renderers cannot be used for training; the project writes its own in Warp. |
| D3 | Isaac Sim is the runtime. One Isaac Sim process per GPU holds many blocks. | A separate core with Isaac Sim as a viewer. | Required. It also gives cameras, annotators, USD layers and Warp in one process. The cost: the build must work, and each process takes tens of seconds to start. |
| D4 | Continuous values (centre, shape, intensity) are fitted by gradient descent. Discrete edits (split, merge, link, label) are chosen by a policy. | A policy that outputs everything; or optimisation only. | Gradients are exact and cheap for continuous values and do not exist for structure. The cost: two mechanisms to keep in step. |
| D5 | Every length is in nanometres. The section thickness is a parameter of the camera. | Working in voxels. | One model then serves 4 × 4 × 40 nm sections and 8 nm isotropic blocks. |
| D6 | The stage is the record. The data, the agent's edits and the reviewer's edits are separate USD layers. | A separate event log. | A reviewer's layer overrides the agent's without changing it, and removing a layer undoes it. The cost: layers record the result of edits, not their order; the order is kept in a small side table. |
| D7 | The policy's view encoder can be swapped. The first one is a small convolutional network; Cosmos 3 Edge is an option for later. | Cosmos in the main loop from the start. | A sequence of views along a Frame is a video, which suits Cosmos, but nothing in the design depends on it. |

## Build order

Each milestone has one test.

| | Milestone | Data | Done when |
|---|---|---|---|
| M0 | Isaac Sim build finished; scene viewer | CellMap crop 313 | `SimulationApp` renders a non-empty image on the Spark. (The viewer already renders through the Kit route.) |
| M1 | Voxel store, mipmap, slab camera | CREMI A | A view from the camera at any level equals the same slice computed with numpy. |
| M2 | Slab renderer in Warp, with gradients; fit to raw | A 2 µm crop of CREMI A | Gradients match finite differences. The fitted Gaussians reproduce the image above a set quality, with the count reported. |
| M3 | Frames from the true labels; Graph; back to voxels | CREMI A | The segmentation rebuilt from those Frames has adapted Rand error below 0.02 against the labels. This shows the representation is sufficient. |
| M4 | The multi-scale cycle | CREMI A, SNEMI3D | Same quality as M2 and M3 with fewer Gaussians and fewer steps. |
| M5 | The policy for discrete edits | CREMI, SNEMI3D | Scores on the held-out sections no worse than a baseline affinity network with watershed. |
| M6 | Class types: clefts and synapses; organelles | CREMI; CellMap crops | Cleft and synapse scores on CREMI; per-class scores on held-out CellMap crops. |
| M7 | Vectorised environment, layers, review in the window | CREMI | A reviewer's edit in the Isaac Sim window overrides the agent, and removing the reviewer's layer restores the agent's result. |
| M8 | Blocks and stitching | A 100 µm cube of MICrONS | Killing workers at random does not change the result. |

M3 comes before any learning on purpose. If Frames made from perfect labels cannot be turned back into a good segmentation, the design is wrong, and it is cheap to find that out first.

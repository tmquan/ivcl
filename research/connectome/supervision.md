# Supervision

Three kinds of data teach the agent: the raw image, the segmentation, and the class type. They differ in how much of each exists.

| Kind | What it says | How much exists | Trains |
|---|---|---|---|
| **Raw** | What the tissue looks like | All of every volume | Centre, radii, amplitude and intensity of the Frame nodes |
| **Segmentation** | Which voxels belong to the same component | Small labelled volumes | The identity of each node; the edges of Frame and Graph |
| **Class type** | What kind of thing a component is | Small labelled crops; tables for some neurons | The class scores |

Raw data is unlimited and needs no person. The part of the model that it trains is therefore trained on everything, including the whole of MICrONS. The labelled parts only have to teach identity and class, on top of a geometry that already fits.

## A loss for every pair

Each kind of supervision can be applied in each [representation](representation.md). That gives nine places for a loss. Eight are used.

| | Voxel | Frame | Graph |
|---|---|---|---|
| **Raw** | **Main.** The slab rendered from the Frames against the raw slab | The cross-section along a Frame changes slowly; predict the next one | Not used |
| **Segmentation** | Rendered identity against the label slab; a node inside a labelled component takes its identity | Nodes against the true medial axis and its radius; `adjacent` edges against the true Frame | `same` and `contact` edges against the labels |
| **Class type** | Rendered class against the class slab | The class of a component from its Frame: caliber, branching, shape | Cell type from connections; `synapse` edges and their direction against tables |

## Raw

### Raw on Voxel

The main loss of the whole system. For a camera pose drawn at random in a block, at a level drawn at random, the model channel of the [view](representation.md) is compared with the raw channel:

$$
L_{\text{raw}} = (1 - \lambda) \, \lVert I_{\text{model}} - I_{\text{raw}} \rVert_1 + \lambda \, \bigl(1 - \mathrm{SSIM}(I_{\text{model}}, I_{\text{raw}})\bigr),
\qquad \lambda = 0.2
$$

Two variants make it harder and more useful.

- **Held-out section.** One section in ten is never shown during fitting. The Frames fitted to its neighbours must render it. This measures whether the model has learned structure and not copied pixels. It is also how a missing or damaged section is filled.
- **Across levels.** The slab is rendered at one level of the mipmap and compared with the raw data of that level (see [Multi-scale](multiscale.md)).

The small network that predicts the membrane map is pretrained the same way, without labels: blocks of raw voxels are masked and predicted.

### Raw on Frame

Along the Frame of a neurite, the cross-section changes slowly: the same profile, the same organelles, sliding past. A small predictor takes the last 8 views along a Frame and predicts the next one. It needs no labels: any Frame gives training sequences, right or wrong, because most stretches of most Frames are right.

Its error is used at run time. A sudden rise marks the place where the Frame leaves its neurite: a merge, a break, a damaged section. This is the only supervision that looks along an object and not across it.

By the rule on oblique views, it is trained at near-isotropic levels only.

## Segmentation

### Segmentation on Voxel

Where labels exist, a node whose centre falls in component $o$ is told so directly. This is the strongest signal and is used first.

For everything else, the identity vectors of the nodes are rendered into a slab with 16 channels, and the label slab says which pixels are the same component. The loss pulls the vectors of one component together and pushes the means of different components apart by a margin:

$$
L_{\text{id}} = \frac{1}{N} \sum_{o} \frac{1}{|o|} \sum_{p \in o} \bigl[\lVert e(p) - \bar e_o \rVert - \delta_v\bigr]_+^2
\; + \; \frac{1}{N(N-1)} \sum_{o \ne o'} \bigl[2\delta_d - \lVert \bar e_o - \bar e_{o'} \rVert\bigr]_+^2
$$

with $N$ the number of components in the view, $\bar e_o$ the mean vector of component $o$, $\delta_v = 0.5$ and $\delta_d = 1.5$. Tradeoff: an identity vector is trained, and not an identity number, because a number has no gradient and the count of components is not known in advance. The number is given afterwards, by clustering the vectors along the `adjacent` edges.

### Segmentation on Frame

The labels give the true Frame of every labelled component: the ridge of its distance transform, with the distance as radius. Three losses use it.

| Loss | Says |
|---|---|
| Position | Each line node lies on the true medial curve; each sheet node on the true medial surface |
| Radius | The middle radius of a node equals the distance transform at its centre |
| Topology | The `adjacent` edges give the same branch points and ends as the true Frame, matched within 500 nm |

This is where shape and continuity are learned. It also makes labels go further: a component labelled only by its skeleton, as in many proofreading tools, still gives the position and topology losses.

### Segmentation on Graph

A classifier scores each pair of components that touch: one object or two? Its inputs come from the two Frames near the contact (overlap, angle between tangents, change of caliber, distance between identity vectors, and the error of the raw-on-Frame predictor carried across the contact). Its target comes from the labels. This is agglomeration, learned.

Sources, in order of trust: manual labels (CREMI, SNEMI3D); proofread neurons of MICrONS; the automatic segmentation of MICrONS, with a low weight and never for evaluation.

## Class type

"Class type" covers three things at three sizes, and each has its natural representation.

| Size | Classes | Source | Applied on |
|---|---|---|---|
| Structure | Membrane, cytoplasm, mitochondrion, cleft, vesicle, and the organelle classes of CellMap | CREMI clefts; CellMap crops | Voxel |
| Compartment | Axon, dendrite, soma, glia | MICrONS tables for some neurons | Frame |
| Cell | Cell type of a neuron; direction of a synapse | MICrONS tables; CREMI partners | Graph |

**On Voxel.** Class scores are rendered and compared with the class slab by cross-entropy, each class weighted by the inverse of its frequency. A cleft is a few hundred voxels among millions; without weights it is never learned.

**On Frame.** An axon and a dendrite differ in their Frames: caliber, how often they branch and at what angle, spines, the shape numbers of [Features](features.md). A small graph network on the nodes of one Frame predicts the compartment. The loss applies only where a label exists.

**On Graph.** A cell type shows in what a neuron connects to. A graph network on the Graph predicts it for neurons with a cell body. For a synapse, the target on CREMI is the annotated pair: which side of the cleft is presynaptic.

## One model, partial labels

A training step takes a block from any dataset and applies the losses for which that block has data.

| Dataset | Raw | Segmentation | Class type |
|---|---|---|---|
| CREMI | Yes | Yes | Clefts, synapses |
| SNEMI3D | Yes | Yes | None |
| MICrONS, anywhere | Yes | Weak | Weak |
| MICrONS, proofread neurons | Yes | Yes, for those neurons | Compartment, cell type |
| CellMap crops | Yes | For some classes | Many structure classes |

A class that a crop does not label is left out of the loss for that crop. It is not counted as background. CellMap crops label different sets of classes, so the rule is needed even inside one dataset.

## Labels from Isaac Sim

Views of what the agent has built need labels too: for display, for review, and for training the policy on its own state. Replicator annotators on the camera give them, aligned with the colour image.

| Annotator | Gives |
|---|---|
| Colour | The scene as drawn |
| Instance segmentation | Which component prim each pixel shows |
| Semantic segmentation | The class label of that prim |
| Normals, depth | The surface of drawn meshes and ellipsoids |

The class of a component is stored on its prim as a USD semantic label. A change of class by the agent or by a reviewer therefore shows in the next view with no other step.

## Held out

| Dataset | Held out | Unused between training and test |
|---|---|---|
| CREMI A, B, C | Sections 100 to 124 of each | Sections 95 to 99 |
| SNEMI3D | Sections 80 to 99 | Sections 75 to 79 |
| MICrONS | One cube of 100 µm: the cube with the greatest path length of fully proofread neurons | A shell of 10 µm |
| CellMap | Every fifth labelled crop of each dataset, by crop number | None: crops do not touch |

The raw loss is not applied in the held-out parts either. Otherwise the geometry would already be fitted there, and the test of identity and class would be easier than on new data.

## Measures

| On | Measure |
|---|---|
| Voxel | Raw: peak signal-to-noise ratio and SSIM on held-out sections, per level. Segmentation: variation of information, split and merge apart; adapted Rand error. Class: intersection over union per class; for clefts also the mean distance to the nearest true cleft |
| Frame | Expected run length; precision and recall of nodes within 200 nm of the true Frame; F1 of branch points and of ends; radius error; nodes per cubic micrometre |
| Graph | Synapse F1 with partners; precision and recall of connected pairs; correlation of synapse counts; accuracy of compartment and cell type |

Every score is given beside the same score of a baseline trained on the same split: a 3D affinity network with watershed for segmentation, a 3D U-Net for classes, TEASAR on the baseline segmentation for Frames.

## Tests

- Each loss, on a synthetic scene where the answer is known, reaches zero.
- A crop with one class left unlabelled gives no gradient to that class.
- The held-out section of a synthetic stack is rendered within a set error from its neighbours.
- The raw-on-Frame predictor, on a synthetic tube joined to another, has its largest error at the join.

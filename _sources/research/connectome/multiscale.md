# Multi-scale

Tissue has structure at every size, from a 40 nm vesicle to a 10 µm cell body. The agent works at several scales and passes information both ways between them.

| Member | What it is | Job |
|---|---|---|
| **Mipmap** | The same data stored at several levels of detail | Gives every scale something to look at and to write to |
| **Coarse-to-fine** | A pass from the coarsest level down | Says where to look; gives each fine part its context |
| **Fine-to-coarse** | A pass from the finest level up | Gathers evidence; decides what belongs together over long distances |

## Mipmap

There is a mipmap of each [representation](representation.md).

### Of Voxel

Each level halves the one below. The rule for which axes are halved follows the [anisotropy](features.md) of the acquisition: x and y only, until the voxel is within a factor of 2.5 of isotropic; then all three.

| Level | CREMI and MICrONS, nm | SNEMI3D, nm | CellMap, nm | Resolves |
|---|---|---|---|---|
| 0 | 4 × 4 × 40 | 6 × 6 × 29 | 8 × 8 × 8 | Membranes, clefts, vesicles |
| 1 | 8 × 8 × 40 | 12 × 12 × 29 | 16 × 16 × 16 | Thin axons, spine necks |
| 2 | 16 × 16 × 40 | 24 × 24 × 58 | 32 × 32 × 32 | Dendrites, mitochondria |
| 3 | 32 × 32 × 80 | 48 × 48 × 116 | 64 × 64 × 64 | Thick dendrites, cell surface |
| 4 | 64 × 64 × 160 | 96 × 96 × 232 | 128 × 128 × 128 | Cell bodies, nuclei, vessels |

Image levels are made by averaging; label levels by the most frequent label. The **working level**, at which output labels are written, is level 0 for CREMI, SNEMI3D and CellMap, and level 1 for MICrONS, the level of its public segmentation.

### Of Frame

Every node of a Frame has a level. The rule: the smallest radius of a node of level ℓ is at least half a voxel of level ℓ. A Frame therefore exists at each level as a coarser or finer set of nodes:

- at a coarse level, a dendrite is a few long nodes and its spines are absent;
- at a fine level, the same dendrite is many short nodes, with each spine a side branch.

The levels are joined by `parent` edges: each node of level ℓ has one parent at level ℓ + 1 that contains it. The Frames of all levels with their `parent` edges form a tree per component. This tree is the mipmap of the Frame.

Rendering a node at a level other than its own uses a prefilter. A pixel of width $p$ averages over its own area. Averaging a Gaussian over a square of width $p$ is close to adding the variance of that square, $p^2/12$, to its covariance in the plane:

$$
\Sigma_{pp} \;\to\; \Sigma_{pp} + \frac{p_\ell^2}{12} I
$$

and scaling the amplitude so that the total stays the same. Without it, a fine node seen at a coarse level falls between pixels and flickers; with it, the render at level ℓ of the fine nodes agrees with the level-ℓ image.

### Of Graph

Contracting the Frame tree gives the Graph at several sizes: all components; components joined into objects by `same` edges; objects grouped into cells. A query about connectivity is answered at the coarsest level that holds the answer.

## Coarse-to-fine

The pass down starts at level 4 and ends at the working level.

1. **At the top.** Fit a small number of large nodes to the level-4 image. They capture cell bodies, nuclei, vessels and the thickest dendrites. A block of 8 × 8 × 10 µm has a few hundred.
2. **Going down one level.** For each node of level ℓ + 1, render its region at level ℓ and compare with the raw data of level ℓ. Where the residual is above a threshold, the node is **split**: it gets children at level ℓ, placed along its longest axis, with half its radii. Where the residual is low, the node gets one child equal to itself.
3. **Inherit.** Children start with the identity vector and class scores of their parent.
4. **Fit.** The children are fitted to the level-ℓ raw data by gradient descent, with the parents fixed.

What the pass gives:

- **Attention.** Fine work is done only where the coarse level cannot explain the image. The inside of a cell body is settled at level 4 and never visited at level 0.
- **Context.** A fine node knows which coarse structure it is in before it has looked at anything.
- **A good start.** Gradient descent on fine nodes started from their parents converges in a few steps; started from nothing it does not.

The risk of this direction: a coarse node can cover two objects that the coarse image cannot tell apart, and its children inherit one identity. The pass up finds this.

## Fine-to-coarse

The pass up starts at the working level and ends at level 4.

1. **Merge.** The children of a node are merged into it by matching moments. With child amplitudes $a_i$, centres $\mu_i$ and covariances $\Sigma_i$, and weights $m_i = a_i \sqrt{|\Sigma_i|}$:

   $$
   \mu = \frac{\sum_i m_i \mu_i}{\sum_i m_i},
   \qquad
   \Sigma = \frac{\sum_i m_i \left[\Sigma_i + (\mu_i - \mu)(\mu_i - \mu)^\top\right]}{\sum_i m_i}
   $$

   The parent is now the best single Gaussian for what its children found.

2. **Vote.** The parent takes the mean identity vector and the summed class scores of its children.
3. **Check.** If the identity vectors of the children fall into two clusters further apart than the margin of the [identity loss](supervision.md), the parent covers two objects. It is replaced by two parents, one per cluster, and their `adjacent` edge is removed.
4. **Link.** `adjacent` and `contact` edges are computed at the level just written. At a coarse level a neurite is a short chain, so a gap of one damaged section, which breaks the chain at level 0, is a small step between two overlapping nodes at level 2.

What the pass gives:

- **Evidence.** A coarse decision rests on everything measured below it.
- **Continuity.** Joining across gaps and across block borders is done where it is easy: at the level where the two ends overlap.
- **Error detection.** Step 3 is a test for merges that costs nothing extra.

## The cycle

Down, then up, then down again: the pattern of a multigrid solver.

```text
level 4   fit ──────────────┐                    ┌── check, link ──┐
level 3      split, fit ────┤                    ├── merge, vote   │
level 2         split, fit ─┤                    ├── merge, vote   │   second
level 1            split, fit                    merge, vote       │   cycle,
level 0               split, fit ── policy edits ─┘                ▼   only where
                                                                       changed
```

- The first pass down builds the Frames from the raw data alone.
- At the bottom, the [agent](agent.md) makes its discrete edits: grow, join, cut, label.
- The pass up carries the result to every level and reports conflicts.
- A second cycle runs only in the regions that the pass up changed. The cycle stops when a pass changes fewer than 0.1% of the nodes, or after 3 cycles.

One loss keeps the levels in agreement during fitting: the render of a parent at its own level should equal the render of its children at that level.

$$
L_{\text{scale}} = \bigl\lVert I_\ell(\text{parent}) - I_\ell(\text{children}) \bigr\rVert_1
$$

## Choosing a level for a view

When the agent looks at a place, it chooses the level at which the structure there is 4 to 12 pixels across. The caliber is known from the node: twice its middle radius. Below 4 pixels, membranes merge. Above 12, the view shows only interior.

| Caliber | Level for CREMI |
|---|---|
| Below 50 nm | 0 |
| 50 to 100 nm | 1 |
| 100 to 200 nm | 2 |
| 200 to 400 nm | 3 |
| Above 400 nm | 4 |

At a cleft, a branch point, or a place of low confidence, the agent also takes a second view one level finer.

## Cost

*Estimates*, to be measured in milestone M4. For one CREMI sample, with 2 to 5 million nodes at level 0:

- each level up has about 4 to 8 times fewer nodes, so all coarser levels together add less than a third to the count at level 0;
- with the pass down, the fine fit visits only where the coarse residual is high. In neuropil that is most of the volume; in a block with a cell body it is much less.

The saving from scale is therefore large in MICrONS, which has cell bodies, vessels and white matter, and small in CREMI, which is all fine neuropil. Milestone M4 reports both.

## Tests

- Mipmap of Voxel: each level equals the average of the one below, on the axes that the rule halves.
- Prefilter: fine nodes rendered at level ℓ equal the level-0 render averaged down to level ℓ, within a set error.
- Merge: the parent of two equal Gaussians a distance d apart has their centre in the middle and a variance larger by d²/4 along the line that joins them.
- Check: a parent whose children come from two synthetic tubes is split in two.
- Cycle: on a synthetic tube with one missing section, the pass up joins the two halves at the level where they overlap.

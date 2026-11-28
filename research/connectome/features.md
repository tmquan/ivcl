# Features

Three features describe the tissue: its geometry, its anisotropy, and the Gaussian splats that carry both.

A Gaussian splat is one node of a [Frame](representation.md): a ball of the medial axis, stretched to an ellipsoid.

## Gaussian splats

A Gaussian is a soft ellipsoid of material. Its density at a point $x$ (in nanometres) is

$$
g(x) = a \, \exp\!\left(-\tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)\right),
\qquad
\Sigma = R \, \mathrm{diag}(s_1^2, s_2^2, s_3^2) \, R^\top
$$

| Field | Size | Meaning | Trained by |
|---|---|---|---|
| Centre $\mu$ | 3 | Position, nm | Raw |
| Rotation $R$, as a quaternion | 4 | Orientation of the axes | Raw |
| Scales $s_1 \ge s_2 \ge s_3$ | 3 | Half-lengths of the axes, nm | Raw |
| Amplitude $a$ | 1 | How much material | Raw |
| Intensity $c$ | 1 | Grey value it contributes | Raw |
| Identity vector $e$ | 16 | Which object it belongs to | Segmentation |
| Class scores $k$ | One per class | What kind of structure it is | Class type |
| Level, parent | 2 | Place in the scale hierarchy | The [multi-scale cycle](multiscale.md) |
| Component | 1 | The Frame it is a node of | The agent |
| Confidence | 1 | How well it is supported | All |

About 46 numbers, 184 bytes, per Gaussian.

### The slab renderer

A pixel of a [view](representation.md) is the mean, over the thickness $t$ of the slab, of what the Gaussians contribute. In the coordinates of the camera, write a point as $(p, z)$ with $p$ in the plane and $z$ along the normal, and split the centre and covariance in the same way:

$$
\mu = \begin{pmatrix} \mu_p \\ \mu_z \end{pmatrix},
\qquad
\Sigma = \begin{pmatrix} \Sigma_{pp} & \Sigma_{pz} \\ \Sigma_{zp} & \sigma_{zz} \end{pmatrix}
$$

Along the normal through a pixel at $p$, the Gaussian is a 1D Gaussian with centre and variance

$$
m(p) = \mu_z + \Sigma_{zp} \Sigma_{pp}^{-1} (p - \mu_p),
\qquad
v = \sigma_{zz} - \Sigma_{zp} \Sigma_{pp}^{-1} \Sigma_{pz}
$$

Integrating it over the slab from $z_0 - t/2$ to $z_0 + t/2$ gives the weight of the Gaussian at that pixel:

$$
w(p) = a \, \exp\!\left(-\tfrac{1}{2} (p - \mu_p)^\top \Sigma_{pp}^{-1} (p - \mu_p)\right)
\cdot \frac{\sqrt{2 \pi v}}{2t}
\left[ \operatorname{erf}\!\frac{z_0 + \tfrac{t}{2} - m(p)}{\sqrt{2v}} - \operatorname{erf}\!\frac{z_0 - \tfrac{t}{2} - m(p)}{\sqrt{2v}} \right]
$$

The view is a sum over the Gaussians near the slab, with no ordering:

$$
I(p) = b + \sum_k c_k \, w_k(p)
$$

where $b$ is the grey value of empty resin. The identity and class channels use the same weights with $e_k$ or $k_k$ in place of $c_k$, divided by $\sum_k w_k(p)$.

Three properties follow.

- **No sorting.** Standard splatting sorts by depth and blends front to back. A slab has no front. The sum is simpler, faster, and has exact gradients.
- **Section thickness is a parameter.** A Gaussian thinner than the slab is averaged by it, as the microscope averages it. The same Gaussians render a 40 nm section and an 8 nm one.
- **Any orientation.** Rotating $\mu$ and $\Sigma$ into the coordinates of the camera is all that an oblique view needs.

The renderer is a Warp kernel. Gaussians are put in a grid of tiles by their extent of 3 standard deviations; each pixel sums the Gaussians of its tile. Warp records the operations and gives gradients with respect to every field.

Tradeoff: the additive model treats grey value as linear in the amount of stain. Real contrast saturates. A monotone curve with 3 parameters per section, fitted with the Gaussians, corrects for it and also absorbs brightness steps between sections.

### How many

A Gaussian should stand for one piece of structure, not for one voxel. The budget is set by a rule: the smallest scale of a Gaussian at level ℓ is half a voxel of that level, and a Gaussian is split only while it lowers the error by more than a threshold.

*Estimate:* a CREMI sample is 5 × 5 × 5 µm, 195 million voxels. With membranes as thin sheets about 100 nm across, and interiors as tubes and blobs, about 2 to 5 million Gaussians, 0.4 to 0.9 GB. The count per cubic micrometre is reported with every result, because it is the cost of the representation.

### In Isaac Sim

| Use | Form |
|---|---|
| Fitting and the agent's views | Warp arrays, one structure of arrays per block |
| The stage | Per component, arrays of positions, orientations and radii under `/Objects/o_{id}/Splats`; the line nodes also as curves under `/Objects/o_{id}/Frame` |
| Display as solid ellipsoids, with instance and class annotations | A point instancer of unit spheres with those arrays |
| Display as splats | The USD particle-field schema, rendered by RTX |

The particle-field display is the 3D look of the model. It is not the slab renderer and is not used for training.

## Geometry

Geometry is read from the nodes of the Frames, with no extra network. The scales $s_1, s_2, s_3$ of a node are its three radii.

| Quantity | From |
|---|---|
| Position | $\mu$ |
| Tangent of a neurite; normal of a sheet | The axis of $R$ with the largest scale; the axis with the smallest |
| Caliber | $2 s_2$ for a tube: twice the middle scale |
| Thickness of a sheet | $2 s_3$ |
| Length along a neurite | Sum of distances between adjacent centres |
| Curvature | Change of tangent between adjacent Gaussians, per unit length |
| Medial axis of a component | The centres of its nodes: a curve through the line nodes, a surface through the sheet nodes |
| Volume of a component | $\sum_k a_k (2\pi)^{3/2} s_1 s_2 s_3$ |
| Surface | The level set of the summed density at half its interior value |
| Contact area between two components | Sum of overlaps between the nodes of their Frames |

**Overlap** of two Gaussians $i$ and $j$ is the integral of their product:

$$
O_{ij} = a_i a_j \, (2\pi)^{3/2} \sqrt{\frac{|\Sigma_i|\,|\Sigma_j|}{|\Sigma_i + \Sigma_j|}}
\exp\!\left(-\tfrac{1}{2} (\mu_i - \mu_j)^\top (\Sigma_i + \Sigma_j)^{-1} (\mu_i - \mu_j)\right)
$$

It is the test behind the `adjacent` edges of a Frame and the `contact` edges of the [Graph](representation.md), and behind the matching of nodes across the halo of two blocks.

Two voxel fields complete the geometry, at the working level only:

- the **membrane map**, predicted by a small 3D network from the raw image;
- its **distance transform**, the distance from each voxel to the nearest membrane.

They are needed once, at the end: the nodes of the Frames are seeds, and a watershed on the distance transform gives every voxel to one seed. This is the step that puts the boundary on the membrane exactly, which a smooth Gaussian cannot do.

## Anisotropy

The word has two meanings here, and the design keeps them apart.

### Anisotropy of the acquisition

Sections are thicker than pixels are wide: 10 times in CREMI and MICrONS, 5 times in SNEMI3D, not at all in CellMap.

| Where it matters | What is done |
|---|---|
| Units | Every length is in nanometres; the voxel size is data, not an assumption of the code |
| Views | The slab thickness is the section thickness |
| The pyramid | x and y are halved until the voxel is within a factor of 2.5 of isotropic; then all three axes (see [Multi-scale](multiscale.md)) |
| Gaussians | No scale along the section normal can be fitted below half the section thickness. Below that, the image holds no information, and the scale is set by a prior: the middle scale $s_2$ |
| Oblique views of raw data | Only at near-isotropic levels |
| Networks on voxels | Kernels of 3 × 3 × 1 at anisotropic levels, 3 × 3 × 3 at near-isotropic ones |

### Anisotropy of the tissue

Structures have a direction. A neurite is long in one direction; a membrane is thin in one. The three scales of a Gaussian say which, through three numbers that sum to 1:

$$
c_{\text{line}} = \frac{s_1 - s_2}{s_1},
\qquad
c_{\text{sheet}} = \frac{s_2 - s_3}{s_1},
\qquad
c_{\text{blob}} = \frac{s_3}{s_1}
$$

| Shape | Largest of the three | Structures |
|---|---|---|
| Line | $c_{\text{line}}$ | Axons, thin dendrites, spine necks, microtubules |
| Sheet | $c_{\text{sheet}}$ | Membranes, synaptic clefts, endoplasmic reticulum, Golgi cisternae |
| Blob | $c_{\text{blob}}$ | Vesicles, lysosomes, lipid droplets, cell bodies, short mitochondria |

These numbers are used in four places:

- as an input to the class scores: a cleft is a sheet between two objects, a vesicle is a blob of 40 nm;
- to give each node of a Frame its shape type, and so to say which nodes form the medial curve (line) and which the medial surface (sheet);
- to start a Gaussian: before fitting, its axes are those of the **structure tensor** of the raw image at that place and scale, the smoothed outer product of the image gradient, whose eigenvectors give the local direction;
- to detect errors: a line-like Gaussian whose tangent turns sharply against its neighbours, or a caliber that jumps, marks a likely merge.

## Tests

- Renderer: a single Gaussian rendered as a slab equals the numerical integral of its density over the slab, to 4 decimal places, at 5 orientations.
- Renderer: gradients from Warp equal finite differences for every field.
- Thickness: one set of Gaussians rendered at 8 nm and summed over 5 slabs equals the render at 40 nm.
- Overlap: the closed form equals a numerical integral on a grid.
- Shape: Gaussians fitted to a synthetic tube, sheet and ball get the right largest shape number.

# Agent

The agent is the loop that builds and corrects the Frames. It runs inside Isaac Sim: its body is a camera in the stage, what it sees are views from that camera, and what it changes are prims of the stage.

## Body and state

| | What | In Isaac Sim |
|---|---|---|
| **Body** | A slab camera: position, orientation, level, slab thickness | A camera prim, `/World/Block_*/Cameras/c_{n}` |
| **Attention** | The Frame node it is working at, and a queue of nodes to visit | Attributes of the camera prim |
| **State of the world** | The Frames and the Graph of the block | Warp arrays, mirrored to the prims of the components |
| **Memory** | Which nodes are settled; confidence of each | Attributes of the nodes |

One Isaac Sim process holds many blocks side by side in one stage, each with several agents. A block is an environment in the sense of Isaac Lab: the same sub-tree of prims, cloned on a grid, stepped together.

## The loop

One step of the agent has four parts.

```text
        ┌──────────────────────────────────────────────────────────┐
        ▼                                                          │
   1. look        views from the camera: raw, model, residual,     │
                  identity, class; the nodes near the camera       │
   2. fit         a few gradient steps on the nodes in view        │
                  (centre, radii, intensity, identity, class)      │
   3. edit        the policy chooses one discrete action           │
   4. move        the camera goes to the next place ───────────────┘
```

Fitting and editing are separate on purpose (decision D4). Where a node sits and how large it is are continuous, and the slab renderer gives exact gradients for them. Whether a node should exist, and what it is joined to, are discrete, and a policy chooses them.

## Actions

| On | Action | Effect |
|---|---|---|
| **Frame** | `grow` | Add a node at the tip of the Frame, one caliber ahead along the tangent. This is how a neurite is traced. |
| | `branch` | Add a node to the side of the current node, and put it on the queue |
| | `retract` | Remove the last k nodes of the tip and mark the place as tried |
| | `split` | Replace a node by two children along its longest axis |
| | `merge` | Replace two adjacent nodes by their moment-matched parent |
| | `prune` | Remove a node that explains nothing |
| | `cut` | Remove an `adjacent` edge: the component becomes two |
| **Graph** | `join` | Add a `same` edge between two components |
| | `separate` | Remove a `same` edge |
| | `link` | Add a `synapse` edge from one component to another through a cleft |
| | `label` | Set the class of a component |
| **Body** | `step` | Move the camera to the next node along the Frame |
| | `rescale` | Go one level up or down |
| | `jump` | Move the camera to the next node on the queue |
| **Control** | `flag` | Mark the place for review, with a reason |
| | `close` | Finish the component: settled, or abandoned |

Tracing a neurite is `grow`, `step`, `grow`, `step`, with `branch` where the view shows a second profile leaving. Compared with flood filling, the agent writes a few nodes per micrometre and not thousands of voxels, and each node already carries the caliber and the direction.

Three rules are enforced by the environment, so that no policy can break them:

- a node cannot be placed where the summed density of settled components of other identities is above one half;
- a component that grows beyond 3 times the 99th percentile of path length in the training set is stopped and flagged;
- every action is written with its actor: the agent with the hash of its weights, or a person by name.

## Policy

```text
 views (5 channels, 256 × 256, up to 8 along the Frame) ─► view encoder ──┐
 nodes within 2 µm: centre, radii, shape, identity, class ─► graph network ┼─► action type
 scalars: level, caliber, queue length, confidence ────────► MLP ──────────┘    action arguments
                                                                                confidence
```

| Part | Choice |
|---|---|
| View encoder | A 2D convolutional network with a small transformer over the sequence of views. It can be swapped for Cosmos 3 Edge, which takes a sequence of views along a Frame as a video (decision D7). |
| Graph network | Message passing over the nodes near the camera and their `adjacent` and `contact` edges, 4 rounds |
| Outputs | The action type; its arguments (which node, which neighbour, which class); a confidence |

Each action type has a rule that it starts from, and that takes over when the confidence of the policy is low. `grow` continues along the tangent while the residual ahead is high; `rescale` follows the table of [Multi-scale](multiscale.md); `flag` fires on the error signals below. A rule is easy to debug and cannot improve; a network can improve and can fail quietly. Starting from rules gives a working agent at every milestone.

### Training

1. **The teacher.** Milestone M3 builds the true Frames from the labels. A scripted agent that can see them takes the right action at every state.
2. **Imitation.** The policy learns to copy the teacher on CREMI and SNEMI3D.
3. **Correction.** The learned policy is run; the states it reaches are labelled by the teacher; training repeats. So the policy learns to recover from its own mistakes, which the teacher never makes.
4. **Review as data.** Every edit by a human reviewer is one more labelled state.

Reinforcement learning, with expected run length as reward, is left as an option. Imitation has a clear target and is stable.

## Finding and fixing errors

A merge joins two neurons and corrupts the connections of both. A split loses connections and invents none. The agent is therefore set to prefer splits: components are grown carefully, and joining them is a separate, checked action.

| Signal | From | Catches |
|---|---|---|
| Children disagree on identity | The pass up of the [multi-scale cycle](multiscale.md) | A coarse node over two objects |
| The cross-section predictor is surprised | [Raw on Frame](supervision.md) | The Frame leaving its neurite |
| Tangent turns by more than 60° between neighbours, or caliber changes by more than 3 times | Geometry of the Frame | Two neurites that touch and cross |
| Four branches at one node with two straight paths through | Topology of the Frame | A crossing taken for a branch point |
| An end inside the block that is not a natural end | Topology of the Frame | A split |
| Two cell bodies in one object | The Graph | A large merge |
| A cleft whose two sides are one object | The Graph | A merge at a synapse |

Each signal has a score, calibrated against held-out labels so that 0.9 means right 9 times in 10. Above 0.9 the agent acts: `cut` for a merge, `join` for a split. Between 0.5 and 0.9 it raises a flag. The priority of a flag is the path length that depends on the answer, times how uncertain the score is.

## Review

A reviewer works in the Isaac Sim window, streamed from the Spark.

1. The reviewer takes the open flag with the highest priority.
2. The stage shows the components involved in different colours, their Frames as curves, and three views at the place: across the sections, and along each of the two Frames.
3. The reviewer uses the same actions as the agent: `join`, `cut`, `label`, `grow` from a clicked point, or accept.

### Layers

The stage is composed of USD layers, strongest first.

| Layer | Written by | Holds |
|---|---|---|
| `review.usd` | People | Their edits |
| `agent.usd` | The agent | Frames, Graph, flags |
| `data.usd` | Ingest | Volumes, block layout, cameras |

A reviewer's edit is an opinion in a stronger layer. It overrides the agent's result and leaves it in place. Muting `review.usd` shows what the agent alone produced; that comparison is how the value of review is measured. When the agent is retrained and run again, `agent.usd` is replaced and the reviewers' edits still apply where their prims still exist; edits whose prims are gone are listed for a second look.

Layers hold the result of edits, not their order. The order, with actor and time, is kept in a table beside the stage (decision D6).

## Many blocks, many GPUs

| Unit | Rule |
|---|---|
| Process | One Isaac Sim process per GPU, started once, kept running. Starting takes tens of seconds, so it is not done per block. |
| Block | 16 blocks per process at a time, as cloned environments (*estimate*, to be set by memory in M7) |
| Task | A row in a task table: block, stage, state, lease. A worker takes a task for 30 minutes and renews the lease while it works. A dead worker stops renewing, and the task returns to the queue. |
| Checkpoint | The Warp arrays of a block are written every 5 minutes, and at the end of each pass of the cycle, to a new file that is then renamed. A reader sees a whole checkpoint or none. |
| Restart | A worker loads the last checkpoint of its block and continues. Components that were open are reopened from their last settled node. |
| Border | Nodes in the halo are computed by both neighbours. Two nodes from two blocks are matched when their overlap is above a threshold; a match becomes an `adjacent` edge. Disagreements become flags. |
| Result | The stage of a block is a payload of the stage of the volume, loaded only when looked at |

```{admonition} Warp arrays and USD prims
:class: tip
During fitting, nodes change thousands of times a second. They live in Warp arrays. Writing each change to USD would be far slower than the fit itself. The prims of a component are therefore updated when the component is closed, when a flag is raised, and at display rate while a person is watching. The Warp arrays are the working copy; the stage is the record.
```

*Estimate* of size for MICrONS, about 1.3 million blocks at 8 × 8 × 40 nm: at 1 to 3 million nodes per block and 184 bytes per node, 240 to 720 TB of nodes at the finest level. That is the same order as the image itself, and too much to keep. So the finest nodes of a block are kept only until its labels and its Frames at level 2 and above are written; the fine nodes can be fitted again from the image where needed. What is kept for the whole volume is the output labels, the Frames from level 2 up, and the Graph.

## Tests

- Rules: a policy that outputs random actions never places a node inside a settled component.
- Teacher: the scripted agent on CREMI A rebuilds the true Frames with expected run length equal to the maximum.
- Layers: an edit in `review.usd` changes the composed stage; muting the layer restores the agent's result exactly.
- Restart: a block killed at a random moment and restarted gives the same Frames and Graph as an uninterrupted run.
- Border: a synthetic neurite crossing 8 blocks comes out as one object with one Frame.

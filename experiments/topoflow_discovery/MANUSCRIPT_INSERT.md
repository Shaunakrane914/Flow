# TopoFlow: solver-free throat discovery — verified manuscript text

## Suggested title

Geometry-Driven Discovery and Response Prediction of Critical Pore Throats: A Controlled Graph-Learning Study

## Abstract

Localized throat restrictions can alter pore-network permeability while leaving a node-only graph representation unchanged. We evaluate an edge-aware graph attention model that predicts global hydraulic importance and single-throat restriction responses from geometry, connectivity and boundary membership. Training uses eight crops from four development scans. A prospectively frozen evaluation uses 10 nonoverlapping 128³-voxel crops from five previously unused Imperial College source scans, S2, S4, S6, S7 and S8. Five fixed training seeds are ensembled; matched guided mean-aggregation and edge-MLP models, an unguided GAT, conductance ranking and an affine-pressure ranking provide controls. All geometry-driven throat selections are sealed before flow-derived outcomes are calculated. Individual 50% restrictions of the guided GAT's three highest-ranked throats produce a scan-macro mean permeability loss of 3.57%, compared with 0.37% for random selection and 4.08% for affine-pressure ranking. On a common intervention pool spanning 10%, 25% and 50% restrictions, guided GAT response MAE is 2.04 percentage points, compared with 1.92 for the guided edge MLP and 0.46 for no response. Thus, the GAT ranks useful intervention locations but does not beat the strongest analytical ranking or the no-response accuracy control. These results quantify the capabilities and limits of learned restriction surrogates under a specified cylindrical-throat pore-network model; they do not establish physical permeability accuracy or a unique advantage from attention.

## Evaluation procedure

The five scan IDs are source groups, not independently certified geological specimens. Two opposite-corner crops per scan are fixed before extraction; failed geometry attempts are retained in the audit. Eligible throats connect physical pores within a topology-spanning inlet–outlet component. Eligibility does not use pressure, flux or dissipation. Each learned model ranks every eligible throat with its global importance head and selects three highest and three lowest scores. A shared uniform sample supplies three random controls. The importance head differs from native GAT attention coefficients. The conductance control uses d⁴/L; the affine-pressure control ranks g(Δx)², approximating dissipation under an affine pressure field without solving flow. Neither analytical ranking uses a fitted coefficient or flow-derived test inputs.

The source is the official [Imperial College ImagesICPSC2007 archive](https://imperialcollegelondon.box.com/v/ImagesICPSC2007). All five parent arrays contain 300³ voxels. Isotropic voxel sizes for S2, S4, S6, S7 and S8 are 4.956, 8.960, 5.100, 4.803 and 4.892 μm, respectively. Array value zero denotes pore space; whole-volume pore fractions agree with the source-header porosities to their reported precision. Published full-volume permeabilities are not used as labels for the crops. Exact voxel-byte comparisons rule out duplicate crops against the locally available earlier development and external data, without establishing geological independence.

All four model architectures, twenty checkpoints, seed ensemble and evaluation settings are fixed before extracting these validation scans. The guided mean/MLP controls were added after the earlier four-scan pilot, but are predefined for this new validation group. Their training objective, optimizer, guidance weight and development-only epoch-selection rule match the guided GAT; parameter counts and selected epoch counts differ and are disclosed. The mean-aggregation network is edge-conditioned and GraphSAGE-style rather than a literal SAGEConv implementation. The edge MLP has no message passing but shares the global edge-score normalization and bounded response readout.

The GAT uses three single-head, 32-channel edge-aware layers with ELU activations. Seven node inputs encode scale-normalized log diameter and volume, normalized coordinates and inlet/outlet flags. Six throat inputs encode normalized log diameter, length, a d⁴/L proxy and three absolute direction components. A symmetric edge head maps the summed and absolute-difference endpoint embeddings plus throat features through 70→32→2 channels. Its first output is globally softmax-normalized over throats to importance s; the second gives leverage l=(1−s)·sigmoid(z). For a restriction q, r=(1−q)⁴ and the predicted fractional loss is (1−r)s/[1−(1−r)l]. This constrained readout follows the known rank-one identity of a linear resistor network and is not claimed as a new physical law. Training uses AdamW with learning rate 0.003, weight decay 0.0001, batch size two, gradient clipping at five and importance-guidance weight 0.05. Development-only epoch selection yields 15 epochs for guided GAT, 12 for guided mean GNN, 22 for guided edge MLP and seven for unguided GAT. No epoch or seed is chosen using the new scans.

Reference outcomes use fixed-pressure, incompressible linear flow with cylindrical inscribed-diameter throats and direct center-to-center lengths, including SNOW2 boundary conduits. A diameter restriction q changes its conductance by (1−q)⁴. Single-throat responses are independently recomputed by fresh sparse solves; an exact rank-one update provides a numerical cross-check and an efficient solver-based control. Individual restrictions are separate counterfactuals, not simultaneous three-throat narrowing. The primary endpoint averages the three selected throat responses, then crops within a scan, then the five scans equally. Shared-pool accuracy uses the union of all geometry-selected top, low and random throats; solver-oracle selections are excluded. Interventions, crops and training seeds are not treated as independent rock specimens.

## Discovery results

| Selection method | Actual top-throat loss (%) | Cluster interval low | Cluster interval high | Sensitivity Spearman |
| --- | --- | --- | --- | --- |
| Guided GAT | 3.570 | 1.158 | 7.258 | 0.514 |
| Guided mean GNN | 1.855 | 0.588 | 3.113 | 0.371 |
| Guided edge MLP | 2.609 | 0.953 | 4.266 | 0.308 |
| Unguided GAT | 1.588 | 0.348 | 3.464 | 0.265 |
| Affine pressure | 4.076 | 0.897 | 8.481 | 0.390 |
| Conductance | 1.693 | 0.163 | 3.766 | 0.221 |
| Random | 0.368 | 0.115 | 0.621 | n/a |
| PNM sensitivity (solver) | 11.146 | 7.343 | 17.411 | n/a |

The GAT identifies higher-impact throats than the random control on this aggregate endpoint. The affine-pressure ranking identifies larger actual losses than GAT without a flow solve. The paired scan-level sign-flip comparison with affine-pressure ranking gives p=0.6250. Only five scan groups are available, so the smallest possible two-sided exact p-value is 0.0625. Bootstrap intervals describe the observed groups and do not establish broad geological generalization.

## Response results

| Response predictor | Common-pool MAE (pp) | Cluster interval low | Cluster interval high |
| --- | --- | --- | --- |
| Guided GAT | 2.039 | 1.329 | 2.776 |
| Guided mean GNN | 1.858 | 1.232 | 2.595 |
| Guided edge MLP | 1.923 | 1.117 | 2.917 |
| Unguided GAT | 1.309 | 0.731 | 2.038 |
| Affine pressure | 0.748 | 0.538 | 0.920 |
| Conductance | 0.595 | 0.377 | 0.844 |
| No response | 0.461 | 0.247 | 0.717 |

These errors use a different, common selection pool from the previous oracle-selected pilot. The previous 24.8% reduction and the new errors are separate experiments and must not be combined. Reported seed variability is secondary to the fixed ensemble endpoint.

## Computational cost and physical scope

Warm CPU timings include the complete neural ensemble response output and separate exact-PNM setup and reuse costs. The guided GAT ensemble takes 25.69 ms, compared with 2.60 ms for a PNM base solve, 13.67 ms for exact all-throat response precomputation and 0.058 ms for the cached all-throat update. These operations have different scopes and setup requirements, but the measurements establish no runtime advantage for the GAT on these graphs. Extraction, feature conversion, batching and disk/model loading are outside forward-only times. The cached exact update uses flow-derived sensitivity and leverage from a base solve and factorization; it is not solver-free. The complete measured timings accompany the results rather than extrapolating to larger volumes or other hardware.

The previously observed S1/S5 full-volume PNM values remain approximately one-quarter of published numerical reference values. This study neither calibrates away that discrepancy nor establishes matched direct numerical simulation. Every intervention conclusion is conditional on the stated linear cylindrical-throat model. No LBM ground truth, verified physical clogging dynamics, attention-specific superiority or universal novelty is claimed.

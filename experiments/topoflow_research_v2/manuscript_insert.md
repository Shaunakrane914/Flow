# Manuscript replacement text

## Evaluation protocol

We evaluate TopoFlow on 24 accepted, non-overlapping 128-cubed micro-CT crops from four natural parent scans: Berea sandstone and the C1, C2 and Estaillades carbonates. The accepted binary crops and voxel resolutions are verified against their source archives. We use nested leave-one-parent-out evaluation. Each outer fold holds out an entire scan; inner folds rotate the remaining scans to select model settings and training duration. We report five training seeds per outer fold. Synthetic media are excluded from this primary experiment. Permeability targets are calculated with a cylindrical-throat pore-network model and are not lattice-Boltzmann or experimental reference measurements.

We compare the standardized original GAT with a scale-normalized formulation that predicts log10(k/d_g^2), where d_g is the geometric mean equivalent pore diameter. The latter uses dimensionless pore features and graph descriptors. GraphSAGE, uniform-attention GAT and an edge-free pore-set network receive the same descriptors. Normalization statistics are fitted exclusively to the training parents. Additional comparators are a training-mean predictor, calibrated porosity and diameter-scaled Kozeny-Carman surrogates, ExtraTrees descriptor regression and a dimensionally normalized ridge model.

## Results

The GAT formulation selected by inner validation achieves a parent-balanced log10 RMSE of 1.4565 +/- 0.0825 across five seeds, with a crop-level log10 MAE of 1.2633, pooled log-space R2 of -2.6589, and factor-of-two accuracy of 13.3%. The fixed physics-and-geometry GAT has parent-balanced RMSE 0.8311. The Porosity KC comparator has parent-balanced RMSE 0.7525. Per-scan results and the uniform-attention and edge-free controls are reported alongside aggregate metrics; aggregation does not establish a distinct advantage for learned attention.

## Scope of conclusions

Measured pore-radius CV spans 0.459 to 0.806. The data do not test a threshold at CV=1.5. Numerical mass conservation assesses the PNM solve rather than its physical accuracy against a voxel-resolved reference. These four previously examined source scans support an exploratory comparison; they do not establish population-level geological generalization or an untouched external validation result. Seed variation measures optimization sensitivity and cannot replace independent scan replication.

The scale-normalized formulation changes both the feature representation and the graph-level inputs. Its comparison with the standardized GAT is therefore a combined design comparison, not a causal isolation of dimensional scaling. Global descriptors are supplied equally to the physics-model controls to assess whether graph message passing adds information.

## Computational measurements

On CPU, the trained scale-normalized GAT has median forward latency 1.719 ms and 95th-percentile latency 2.673 ms across 24 held-out crops and 720 timed calls. Fresh SNOW2 extraction on one predeclared crop per parent has median latency 4.057 s, while the cylindrical-throat PNM reference solve has median latency 1.854 ms. The extraction statistics use four crops; repeated forward calls are timing observations rather than independent rock samples. These measurements support millisecond evaluation of a precomputed graph, but do not demonstrate a large solver acceleration or a measured memory advantage over a 3D CNN.

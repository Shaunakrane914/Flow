# TopoFlow natural rock evaluation

The primary endpoint is permeability error on a held-out natural scan. Synthetic media are excluded from all primary training, validation and testing. All four available natural scans rotate through the outer test fold; inner parent validation selects settings and epoch counts without the outer test labels. Five seeds measure optimization variation, not five independent datasets.

These existing scans have already been examined during development. This is an exploratory nested cross-validation study, not a newly acquired untouched external validation set.

| Model | Parent macro RMSE log10 K | Crop RMSE log10 K | MAE log10 K | R2 log | R2 linear | Within factor 2 |
|---|---:|---:|---:|---:|---:|---:|
| porosity_kc | 0.7525 +/- 0.0000 | 0.6990 | 0.5254 | 0.2537 | -3.3437 | 37.5% |
| physics_ridge | 0.7881 +/- 0.0000 | 0.8071 | 0.6883 | 0.0051 | -11.6206 | 16.7% |
| scaled_kc | 0.8133 +/- 0.0000 | 0.7862 | 0.5652 | 0.0560 | -15.5409 | 54.2% |
| gat_physics | 0.8311 +/- 0.0200 | 0.8488 | 0.7207 | -0.1007 | -11.3928 | 22.5% |
| descriptors | 0.8327 +/- 0.0124 | 0.8525 | 0.7521 | -0.1104 | -0.1541 | 12.5% |
| pore_set_physics | 0.8462 +/- 0.0856 | 0.8474 | 0.6654 | -0.1044 | -17.0516 | 32.5% |
| uniform_physics | 0.8659 +/- 0.1314 | 0.8137 | 0.6377 | -0.0202 | -1.0120 | 30.8% |
| sage_physics | 0.8713 +/- 0.0781 | 0.8782 | 0.7361 | -0.1851 | -4.9386 | 20.0% |
| constant | 0.9224 +/- 0.0000 | 0.8787 | 0.6762 | -0.1793 | -0.1775 | 20.8% |
| selected_all | 0.9943 +/- 0.1069 | 0.9767 | 0.8387 | -0.4663 | -17.3654 | 17.5% |
| selected_gat | 1.4565 +/- 0.0825 | 1.5462 | 1.2633 | -2.6589 | -11.5479 | 13.3% |
| gat_standard | 1.5528 +/- 0.0393 | 1.6167 | 1.3305 | -2.9942 | -0.2249 | 16.7% |

The +/- values are sample standard deviations over training seeds. Parent macro RMSE is the square root of the mean of four parent MSE values; crop RMSE weights individual crops equally. R2 is calculated in both explicitly named spaces. Targets are cylindrical-throat PNM approximations in square metres, not LBM or measured permeability.

## Per scan results

| Model | Held-out scan | Crops | RMSE log10 K | Within factor 2 |
|---|---|---:|---:|---:|
| constant | parent_berea | 8 | 0.4505 | 25.0% |
| constant | parent_c1 | 7 | 0.7255 | 28.6% |
| constant | parent_c2 | 6 | 1.3169 | 0.0% |
| constant | parent_estaillades | 3 | 0.9693 | 33.3% |
| descriptors | parent_berea | 8 | 0.8685 | 0.0% |
| descriptors | parent_c1 | 7 | 0.7837 | 14.3% |
| descriptors | parent_c2 | 6 | 0.9819 | 33.3% |
| descriptors | parent_estaillades | 3 | 0.6634 | 0.0% |
| gat_physics | parent_berea | 8 | 0.8024 | 5.0% |
| gat_physics | parent_c1 | 7 | 0.6798 | 31.4% |
| gat_physics | parent_c2 | 6 | 1.1264 | 26.7% |
| gat_physics | parent_estaillades | 3 | 0.5699 | 40.0% |
| gat_standard | parent_berea | 8 | 0.8064 | 12.5% |
| gat_standard | parent_c1 | 7 | 2.4925 | 0.0% |
| gat_standard | parent_c2 | 6 | 1.3460 | 33.3% |
| gat_standard | parent_estaillades | 3 | 0.9568 | 33.3% |
| physics_ridge | parent_berea | 8 | 0.6388 | 12.5% |
| physics_ridge | parent_c1 | 7 | 0.7860 | 0.0% |
| physics_ridge | parent_c2 | 6 | 1.1059 | 33.3% |
| physics_ridge | parent_estaillades | 3 | 0.4851 | 33.3% |
| pore_set_physics | parent_berea | 8 | 0.3725 | 47.5% |
| pore_set_physics | parent_c1 | 7 | 0.9717 | 17.1% |
| pore_set_physics | parent_c2 | 6 | 1.1246 | 33.3% |
| pore_set_physics | parent_estaillades | 3 | 0.6450 | 26.7% |
| porosity_kc | parent_berea | 8 | 0.2343 | 62.5% |
| porosity_kc | parent_c1 | 7 | 0.4213 | 28.6% |
| porosity_kc | parent_c2 | 6 | 1.1473 | 33.3% |
| porosity_kc | parent_estaillades | 3 | 0.8462 | 0.0% |
| sage_physics | parent_berea | 8 | 0.6048 | 12.5% |
| sage_physics | parent_c1 | 7 | 0.9960 | 17.1% |
| sage_physics | parent_c2 | 6 | 1.0895 | 26.7% |
| sage_physics | parent_estaillades | 3 | 0.6708 | 33.3% |
| scaled_kc | parent_berea | 8 | 0.1336 | 100.0% |
| scaled_kc | parent_c1 | 7 | 0.6580 | 42.9% |
| scaled_kc | parent_c2 | 6 | 1.3007 | 16.7% |
| scaled_kc | parent_estaillades | 3 | 0.7094 | 33.3% |
| selected_all | parent_berea | 8 | 0.8024 | 5.0% |
| selected_all | parent_c1 | 7 | 0.9717 | 17.1% |
| selected_all | parent_c2 | 6 | 1.1246 | 33.3% |
| selected_all | parent_estaillades | 3 | 0.9708 | 20.0% |
| selected_gat | parent_berea | 8 | 0.8024 | 5.0% |
| selected_gat | parent_c1 | 7 | 2.4925 | 0.0% |
| selected_gat | parent_c2 | 6 | 1.1264 | 26.7% |
| selected_gat | parent_estaillades | 3 | 0.5699 | 40.0% |
| uniform_physics | parent_berea | 8 | 0.3764 | 45.0% |
| uniform_physics | parent_c1 | 7 | 0.7442 | 25.7% |
| uniform_physics | parent_c2 | 6 | 1.1042 | 23.3% |
| uniform_physics | parent_estaillades | 3 | 0.9708 | 20.0% |

## Interpretation and paper use

gat_standard retains the original GAT and mean readout with train-only feature and target standardization. The physics variants predict a dimensionless residual plus twice the log geometric mean pore diameter. They also receive the same dimensionless graph descriptors; GAT, uniform attention, GraphSAGE and the edge-free pore-set control have matched descriptor heads. The physics variant changes both dimensional scaling and graph inputs, so it is a combined ablation rather than evidence isolating either change.

selected_gat chooses between the two GAT formulations using inner validation only. selected_all chooses among all model families by the same rule. Their outer metrics evaluate selection strategies rather than retrospectively choosing the best test score.

The four-scan sample cannot establish a universal topology threshold or statistically significant architectural superiority. A threshold outside the observed CV range is untested, not refuted. A PNM conservation residual validates the linear solve numerically; it does not validate the pore geometry or cylindrical closure against voxel-resolved flow.

See protocol.json, inner_selection.csv, predictions.csv, metrics.csv, per_parent_metrics.csv, paired_parent_comparisons.json and checkpoints/ for the complete audit trail. Runtime measurements are reported separately; no CNN resource advantage is inferred from parameter count.

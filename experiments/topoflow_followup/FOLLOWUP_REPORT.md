# TopoFlow follow-up experiments

This study completes the controlled GAT ablation, a locked test on four additional natural scans, and a critical-throat intervention pilot. The measured gap is an input-identifiability limitation: a GAT that receives only node size/volume and adjacency cannot distinguish networks that differ only in throat diameter. The pilot extension supplies throat geometry and predicts counterfactual permeability loss with a separate learned hydraulic-importance head.

## Controlled design comparison

| Physical normalization | Graph descriptors | Development parent-balanced log RMSE |
|---|---|---:|
| No | No | 1.5528 |
| No | Yes | 0.8933 |
| Yes | No | 0.9504 |
| Yes | Yes | 0.8311 |

All cells use the same original GAT encoder, inner parent validation, candidate grid and five seeds. The normalization factor changes both dimensionless input and dimensionless target representation; it is not an isolated comparison of input z-scoring alone. Two existing cells are reused from their exact compatible V2 runs; two missing cells are newly trained.

## Locked external permeability evaluation

The official archive supplies S1, S3, S5 and S9. All 32 predeclared corner crops are accepted. The scans are locked before crop preparation, and all model settings are selected on the four development scans only. These are four named sandstone datasets from the same collection; independent geological provenance beyond the specimen identifiers is not certified.

| Model | Log RMSE | Log R2 | Factor-two accuracy |
|---|---:|---:|---:|
| physics_ridge | 0.2534 | 0.6547 | 75.0% |
| pore_set_physics | 0.2980 | 0.5034 | 66.9% |
| scaled_kc | 0.3428 | 0.3682 | 53.1% |
| gat_descriptors_only | 0.4111 | 0.0233 | 51.2% |
| gat_normalized_only | 0.4131 | 0.0339 | 51.2% |
| selected_all | 0.4369 | -0.2506 | 53.8% |
| uniform_physics | 0.4369 | -0.2506 | 53.8% |
| porosity_kc | 0.5060 | -0.3767 | 40.6% |
| descriptors | 0.5435 | -0.5889 | 51.9% |
| gat_physics | 0.5687 | -0.7671 | 36.9% |
| selected_gat | 0.5687 | -0.7671 | 36.9% |
| constant | 0.6307 | -1.1389 | 34.4% |
| gat_standard | 1.9294 | -19.6411 | 7.5% |

Physics ridge remains the strongest permeability comparator in this external experiment. The primary fixed GAT and the GAT selected on development data do not beat it. These external permeability results were not used to adjust the original permeability models afterward.

## Existing GAT counterfactual failure

On one predeclared accepted crop from each development scan, three highest-dissipation and three other random internal throats are narrowed by 10%, 25%, 50% or fully removed. Partial narrowing does not change node features, adjacency or the V2 graph descriptors. Both existing GAT formulations therefore emit exactly zero predicted change. Full removal alters adjacency and is reported separately. A disconnected full-blockage case has k=0 and no finite log label.

At 50% diameter reduction, the mean loss across the three high-dissipation throats is 13.7% (berea), 44.5% (c1), 42.9% (c2), 29.9% (estaillades). This is behavior under the declared PNM approximation, not a real clogging measurement.

## Throat-aware prototype and external pilot

The prototype is trained on eight development graphs, two per development scan. One graph from each of the four additional scans is reserved for external response evaluation. The original primary guided-GAT and three unguided architectures are specified before evaluating those pilot responses. Equally guided mean-aggregation and edge-MLP controls are added after inspecting the first pilot results to test a supervision confound; those two comparisons are explicitly post hoc.

Features comprise dimensionless pore geometry, coordinates, inlet/outlet flags, throat diameter, length, directional geometry and a conductance proxy. No pressure, flow, dissipation label or reference permeability enters the model forward pass. The guided model receives PNM hydraulic importance only as an auxiliary training target. Its global importance head is not a native GAT attention coefficient, so it does not establish that the attention coefficients are explanations.

| Model | Permeability-loss MAE in percentage points | Seed SD in points | Importance Spearman |
|---|---:|---:|---:|
| edge_gat | 4.888 | 0.208 | 0.209 |
| guided_edge_gat | 4.573 | 0.377 | 0.422 |
| edge_sage | 5.003 | 0.138 | 0.301 |
| guided_edge_sage | 5.033 | 0.389 | 0.363 |
| edge_mlp | 5.287 | 0.217 | 0.323 |
| guided_edge_mlp | 4.840 | 0.125 | 0.291 |
| no_response | 6.078 | 0.000 | undefined |
| conductance_proxy | 6.105 | 0.000 | 0.089 |

The primary guided GAT reduces mean absolute response error by 24.8% relative to predicting no response. This is an exploratory four-graph result. Seed variation is not specimen uncertainty, and thousands of dependent edge/dose/seed rows do not create additional independent rocks. The exact parent sign-flip comparisons are exported; with four parents their minimum possible two-sided p-value is 0.125.

The no-response control has lower error for random throats than the guided GAT. The gain is concentrated in the high-dissipation subset. Test throats are selected using an oracle PNM dissipation field to construct the experiment; this selector is not claimed as a solver-free deployment mechanism. Separately reported ranking metrics assess the learned geometry-only head.

## Independent published full-volume consistency check

Complete 300-cubed S1 and S5 images are processed in isolated workers, rather than assigning published full-scan values to crops. Our PNM gives 342.49 and 1187.48 mD, compared with published Kz values of 1312 and 4440 mD. Boundary settings are incompletely matched and the GAT inputs are larger than their 128-cube training domains. These are cross-method consistency checks, not certified matched DNS validation or newly run LBM simulations. The approximately fourfold discrepancies prevent treating numerical mass conservation as physical accuracy.

## Research positioning

The defensible question is whether geometry-only learned hydraulic importance predicts controlled single-throat restrictions better than equally informed controls on unseen scans. The current evidence supports a promising pilot, not a universal critical-throat detector or an established attention-superiority claim. The sensitivity and finite-update identities are known consequences of linear network physics; they are not presented as new mathematical results.

[RTG-GNN (2024)](https://doi.org/10.1016/j.geoen.2024.213358) already uses pore/throat topology for permeability prediction. [Zhao and Xiao (2026)](https://doi.org/10.1016/j.jhydrol.2026.135602) already learns conductances inside a PNM and studies physically consistent geometric sensitivities. Therefore edge features and sensitivity analysis alone are not novelty. Controlled intervention prediction and comparison of learned importance with hydraulic sensitivity are the narrower contribution being investigated; a first-of-its-kind claim is not established.

Sources: [official Imperial collection](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/) and [Dong and Blunt (2009)](https://doi.org/10.1103/PhysRevE.80.036307).

## Artifacts

- `ablation/`: controlled four-cell results, configurations and 40 new checkpoints.
- `external_protocol.json` and `external_data/`: scan lock, 32 crop attempts, source headers and hashes.
- `external_test/`: 30 trained models, prediction records, tuning records and per-scan results.
- `critical_throats_verified/`: 96 physical interventions, two existing GAT evaluations each, attention rankings and physics checks.
- `throat_aware_pilot/`: 12-graph inventory, 30 trained pilot models, response records and parent-level comparisons.
- `full_reference/`: complete-domain reference calculations and published-value scope.
- `figures/`: four figures in PNG, SVG and PDF.
- Raw source archives, images and graph tensors remain in the ignored `data/` directory.

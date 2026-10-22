# Public data pilot results

Run completed on 22 October 2026. This is an implementation check for Experiments 1 and 2, not evidence that the manuscript is publication ready.

## Data recovered and prepared

The configured Supabase downloads failed with connection errors. Public institutional sources were used instead:

| Specimen | Source | Role | Accepted crops | Rejected crops |
|---|---|---|---:|---:|
| C1 carbonate | [Figshare C1](https://doi.org/10.6084/m9.figshare.1189257) | Training | 3 | 1 |
| C2 carbonate | [Figshare C2](https://doi.org/10.6084/m9.figshare.1189258) | Training | 3 | 1 |
| Berea sandstone | [Figshare Berea](https://doi.org/10.6084/m9.figshare.1153794) | Validation | 4 | 0 |
| Estaillades carbonate | [Imperial College images](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/) | Test | 1 | 3 |

Estaillades is the same named 1000³ archive referenced by the existing processor. Its header specifies 3.31136 micrometre voxels. C1, C2 and Berea are additional public specimens, not replacements for MEC, ILS and Savonnières. The deleted original 1,231-sample collection has not been reconstructed.

Sixteen fixed crops of 128³ voxels were attempted. Five lacked a percolating pore-network path under the declared pilot extraction and flow settings. Their exclusions and reasons are recorded. Accepted targets are newly calculated cylindrical-throat PNM permeabilities. Published full-scan direct-simulation permeability values are retained in the source header but are not used as crop labels.

All crops from a source scan remain in one split. There are two training parents, one validation parent, and one test parent. Only one positive-permeability test crop survived. This is far too little independent evidence for a geological generalization or heterogeneity finding.

## What was run

The original GAT, a specified GraphSAGE comparator, a uniform-attention GAT control and a pore-feature set model were trained using seeds 11, 22, 33, 44 and 55. Each used a maximum of 100 epochs, patience of 20, and checkpoint selection by validation MSE. Training-fitted constant, KC and descriptor-regression baselines were also evaluated. This is one fixed pilot configuration, not a completed fair hyperparameter search.

The run produced 20 neural checkpoints and 385 sample/model/seed prediction rows, with metrics, loss histories, source and graph hashes, physical pore-size CV, paired comparisons, and a pilot-labelled figure.

## Numerical diagnostic

These values describe **one held-out crop**, averaged over five model seeds. They must not be used as paper-level performance estimates or compared with the manuscript's historical numbers.

| Model | Mean held-out absolute log10 error | Standard deviation across seeds |
|---|---:|---:|
| Training constant | 0.7457 | 0.0000 |
| Calibrated KC | 0.5389 | 0.0000 |
| Descriptor regression | 0.5463 | 0.0813 |
| Original GAT | 1.0149 | 0.1142 |
| GraphSAGE | 1.2538 | 0.1389 |
| Uniform GAT | 1.0338 | 0.0605 |
| Pore set | 1.0062 | 0.0257 |

For one sample, RMSE in log space equals absolute log error per seed. The table is derived from `results/experiments/public_pilot/summary.json`. Seed variability is not uncertainty over independent specimens.

The pilot does **not** establish a GAT advantage over simpler models: KC and descriptor regression had smaller error on this single crop. It also cannot establish whether learned attention helps across heterogeneous rocks.

The exporter correctly leaves test R² undefined and declines to estimate a parent-bootstrap interval because only one independent test parent exists. CV bin boundaries were calculated solely from training samples, at approximately 0.4771 and 0.6089. No threshold was fitted or validated.

## Verification and limits

All 13 automated tests passed. They include perfect-prediction paired loss, target-transform inversion, parent leakage and duplicate detection, physical CV, model controls, analytical Poiseuille flow, viscosity/pressure invariance, length-squared permeability scaling, and full pipeline execution. Changing only test labels in the integration test leaves selected model weights identical. Source files also passed Python syntax checks. The largest relative mass imbalance in accepted pilot labels was approximately 4.72e-13.

Numerical conservation does not validate the pore geometry or conductance closure against real rocks. Pore-label interpretation, boundary conduits, crop representativeness and the PNM approximation still need independent assessment. The failed-crop filtering also limits the population represented. No LBM run, CNN benchmark, runtime comparison, memory benchmark or reconstruction of the original five-category dataset was performed.

The current experiment environment used PyTorch 2.10.0 CPU, PyG 2.3.1, PoreSpy 3.0.3 and OpenPNM 3.6.0 installed locally. Actual training environment versions are recorded in the run metadata; they differ from the repository's pinned PyTorch version. Pilot reference flow is solved with the explicit SciPy mass-balance routine rather than the legacy OpenPNM label function.

## Files and next scientific step

- [Workflow and commands](README.md)
- [Run metadata](../results/experiments/public_pilot/run.json)
- [Predictions](../results/experiments/public_pilot/predictions.csv)
- [Per-category metrics](../results/experiments/public_pilot/metrics.csv)
- [Heterogeneity comparisons](../results/experiments/public_pilot/heterogeneity.json)
- [Pilot heterogeneity figure](../results/experiments/public_pilot/heterogeneity.png)
- `data/pilot_public/manifest.json`, `sources.json` and `rejections.json` contain the local data inventory and preparation evidence. These files and the recovered source archives remain under the gitignored data directory.

The next scientific step is to acquire additional independently identified parent scans, validate the label pipeline on matching independent references, and freeze a research split/tuning protocol. Merely extracting many more crops from the same Estaillades scan would not fix the lack of independent test specimens. The implemented workflow can run that study once its manifest and data are available.

# TopoFlow experiments 1 and 2

This workflow retains the original three-layer GAT and tests it against GraphSAGE, uniform incoming attention, a model of the unordered pore features, and inexpensive regressors. It corrects paired loss, selects checkpoints exclusively on validation data, and exports paired test errors against physical pore-size heterogeneity. No historical dashboard score is used as an experimental result.

## Run a small public-data pilot

From the repository root, using the scientific environment described in `requirements.txt`:

```powershell
python scripts/prepare_pilot_data.py
python scripts/run_experiments.py --manifest data/pilot_public/manifest.json --output results/experiments/public_pilot --purpose pilot --epochs 100
```

The preparation script downloads the original Imperial College Estaillades archive and three small legacy Imperial/Figshare specimens: C1 carbonate, C2 carbonate, and Berea sandstone. C1 and C2 are training specimens, Berea is validation, and Estaillades is test. The four crops of any specimen remain together. These are **pilot splits**, not a representative geological-generalization benchmark. C1/C2 are additional public carbonates; they do not replace the deleted MEC, ILS, or Savonnières records.

The archived Estaillades header gives 3.31136 micrometre spacing. Source descriptions, archive hashes, original headers, resolutions, crop coordinates, assumptions and rejections are written to `data/pilot_public`. Published full-volume permeability is not assigned to crops. Labels are newly computed **cylindrical-throat pore-network approximations**, not LBM labels. The preparation routine uses SNOW2 equivalent pore diameters, inscribed throat diameters, direct centre distances, explicit boundary labels, positive conductances and mass-balance checks. Boundary nodes are excluded from GNN pore features and CV. This defines a documented pilot model; it does not certify the closure against direct simulation. In particular, boundary conduits and pore-label interpretation still need validation.

Downloads and generated graphs stay in the gitignored `data/` directory. Existing `.pth` files and historical results are not overwritten. Preparation and experimental outputs must use fresh directories; an existing source archive can be reused after integrity checks. `tar` with 7z support (available with Windows) is used to read the small Figshare archives. Raw files are read without extracting arbitrary archive paths. Estaillades crops are read through the compressed stream without writing its 1 GB volume to disk.

The earlier cloud library may also contain recoverable volumes:

```powershell
python scripts/recover_cloud_samples.py --per-category 2
```

This reads the existing `.env` configuration without printing credentials. Recovery does not establish original parent scans, crop origins, physical resolution, or research suitability. Its inventory is not a training manifest. The cloud endpoint was unavailable during initial recovery.

## Declare an actual research dataset

Copy `experiments/manifest.example.json` to a directory near your trusted `.pt` graph files and fill `samples` with records having:

| Field | Meaning |
|---|---|
| `id` | Unique sample identifier |
| `graph` | Trusted local PyG graph path, relative to the manifest; absolute paths also work |
| `parent_id` | Actual parent scan/specimen identity shared by all its crops and augmentations; never invent one parent per crop |
| `formation` | Explicit rock category used for summaries; name synthetic media as synthetic |
| `split` | `train`, `validation`, or `test` |
| `porosity` | Physical pore fraction measured from the binary image, strictly between zero and one |
| `label_source` | Exact reference method, such as a validated OpenPNM configuration or independently produced LBM data |

Also retain source DOI/URL, licence, physical resolution, crop coordinates, phase labels, flow axis, exclusions and reference-solver settings. Extra fields are preserved in the resolved manifest.

Graphs must have two node features in this order: `log10(diameter_m + diameter_epsilon_m)` and `log10(volume_m3 + volume_epsilon_m3)`. The scalar `y` must be `log10(k_m2 + target_epsilon_m2)`. All epsilons must be explicitly declared in `encoding`; a sample can override them using its own `encoding` object. The loader removes the declared epsilons, checks positivity, and converts to canonical **log10 of SI values without an additive epsilon** for common model inputs and targets. It does not silently assume that legacy scripts used identical transforms. If the original transform destroyed information or the geometry was incorrectly derived, regenerate the graph from valid physical data.

All three splits must be nonempty. The loader rejects parents appearing in different splits, duplicate IDs, duplicate file content or graph tensors, nonfinite features, invalid or one-way edges, and missing graphs. These checks cannot discover neighbouring crops given falsely distinct parent identities; parent metadata must be verified against source records. Graphs are loaded through PyTorch pickle; only use graphs you prepared or trust.

```powershell
python scripts/run_experiments.py --manifest data/research/manifest.json --output results/experiments/research_run_01 --epochs 100 --seeds 11 22 33 44 55
```

For a category held out from both training and validation:

```powershell
python scripts/run_experiments.py --manifest data/research/heldout_manifest.json --output results/experiments/heldout_run_01 --protocol formation-holdout
```

There is no automatic random crop split. If only one parent exists in a category, independent within-category specimen generalization cannot be established. Prepare one manifest and fresh run directory for each outer fold. Keep hyperparameter searches inside the training/validation portion. Test results must not be used to revise configurations and then reported as an untouched final test.

The historical `src/train_ils.py`, `train_synthetic.py`, `train_estaillades.py` and `train_savonnieres.py` now route to this CLI. They require the same explicit manifest/output arguments. Their names do not filter samples. The older hybrid and standalone baseline scripts are historical, are not used in this protocol, and retain known scientific limitations; do not use them to produce the new paper's tables.

## Experiment 1

- **GAT:** unchanged `TopoFlowGNN`, including two heads in the first layer, ELU, dropout, global mean pooling and the existing MLP.
- **GraphSAGE:** mean aggregation with the same 128/64/64 output widths, activations, dropout, pooling and prediction head. Parameter counts differ and are reported. This is a specified comparator, not a claim of equal parameter capacity.
- **Uniform GAT:** the same GAT feature projections/heads/self loops, with equal incoming neighbour weights and frozen unused attention parameters. It is a closer control for learned attention than comparing two distinct architectures alone.
- **Pore set:** linear layers on pore features followed by mean pooling; ignores edges, with the same output widths and prediction head. It tests whether message passing adds value beyond distributions of pore features.
- **Constant:** training-mean log permeability.
- **KC:** `log10(k) = log10(C) + log10(phi^3/(1-phi)^2)`. Fit `log10(C)` on training samples only, minimizing the common log-error objective. No test calibration or hidden phi clipping.
- **Descriptors:** ExtraTrees regression on measured porosity, physical pore size/volume summaries, node and edge counts, degree summaries and pore-radius CV. This control includes coarse graph descriptors; it is not a purely geometry-only model. Uses 200 trees, minimum leaf size two, fixed seed, and training data only.

Every learned neural model uses the same optimizer settings and matched shuffle seeds. Checkpoints are selected by validation MSE with early stopping; the test set is predicted after selection. Different architectures consume different random numbers and have different capacities. Fix the tuning budget in advance or use an inner validation search; equal defaults alone do not establish that every architecture was optimally tuned. The default seeds are 11, 22, 33, 44 and 55. Hyperparameters are configurable with `--help`.

Metrics explicitly distinguish log10(m²) errors, linear m² errors, linear and log R², mean absolute relative error, and factor-of-two accuracy. Undefined constant-target R² is null, not forced to one or zero. `summary.json` reports equal-category macro RMSE and descriptive variation across seeds; `metrics.csv` retains per-category and pooled results. Seed variation does not measure the number of independent specimens.

## Experiment 2

CV is calculated **per sample from physical pore radii**, as population standard deviation divided by mean. Radius and diameter give the same CV under a constant factor of two. Permeability labels are never used to construct CV.

Low/medium/high boundaries are fixed at the **training** CV tertiles. Test plots and pair tables retain continuous CV and category identities. For each test sample, squared log-error differences are paired by seed, then averaged across seeds. Positive difference means GAT has lower error than its comparator. Parent-balanced mean effects are reported overall, by training-defined CV bin and by category. Percentile bootstrap intervals resample parent-level effects (2,000 repeats by default); intervals are not estimated with fewer than two parents.

These are exploratory associations. A CV bin plot does not control for lithology or prove a causal attention mechanism, and three bins cannot establish a universal threshold. No changepoint or `CV=1.5` claim is fitted. A confirmed heterogeneity-dependent effect needs additional independent parents, coverage across the descriptor range, and a prospectively specified or independently validated analysis. Attention weights are not automatically physical explanations.

## Evidence exported by each run

| File | Contents |
|---|---|
| `run.json` | Configuration, completion status, software versions, source hashes, manifest hash, split counts, fitted baseline parameters and limitations |
| `manifest_resolved.json` | Exact graph paths, hashes, parent IDs, physical labels, feature encodings, CV and descriptors |
| `*_seed*.pth` | Newly trained weights, validation-selected epoch and configuration |
| `*_seed*_history.csv` | Training and validation losses only, with no per-epoch test selection |
| `models.json` | Model sizes, selection scores and checkpoint hashes |
| `predictions.csv` | Every sample, split, model and seed, with true/predicted physical and log permeability |
| `metrics.csv` | Per-category and pooled metrics, including train/validation/test |
| `summary.json` | Macro test error across categories and descriptive seed variability |
| `heterogeneity_pairs.csv` | Sample-level paired GAT/comparator errors and physical CV |
| `heterogeneity.json` | Training bin boundaries, paired effects, parent counts, intervals and interpretation limits |
| `heterogeneity.png` | Continuous CV versus paired GAT/GraphSAGE error, identified as PILOT or RESEARCH |
| `descriptors_seed*.pkl` | Fitted training-only descriptor baseline |

Archive or commit the scripts and compact evidence for a paper. Large binaries may live in a licensed data repository with stable hashes. A checkpoint is linked to the configuration and graph hashes through `run.json`, `models.json` and the resolved manifest. These runs do not measure end-to-end latency or memory (Experiment 3), so do not reuse the old speed and memory claims.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover paired-loss broadcasting, metric spaces, inverse transforms, leakage and duplicate detection, CV construction, attention/pore-set controls, analytical flow/scaling, and full-run reproducibility. In the integration test, changing only held-out labels must leave validation-selected model parameters identical. Generated test fixtures are temporary and are not research data.

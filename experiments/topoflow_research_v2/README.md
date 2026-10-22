# TopoFlow research evaluation

Completed on 22 October 2026. Start with [the research report](research_report.md),
[the paper comparison table](tables/paper_comparison.md), and
[replacement manuscript passages](manuscript_insert.md).

The primary dataset contains **24 accepted crops from four natural scans**, not
36 independent rocks. The other 12 saved graphs are synthetic and are excluded
from primary training, validation, and testing. The four source scans have been
examined before this experiment, so this is exploratory nested cross-validation,
not a newly acquired external validation set.

## What was executed

- Four outer leave-one-natural-parent-out folds.
- Three inner validation folds per outer fold; all outer test labels excluded
  from feature fitting, parameter selection, checkpoint selection and epoch selection.
- Five neural formulations, two candidate optimizer/regularization settings each.
- Five analytical or descriptor baselines with declared candidate parameters.
- 240 recorded inner candidate/fold evaluations and 100 final neural checkpoints.
- Five final training seeds and 1,440 prediction rows across ten model families
  plus two strategies selected inside the training parents.
- Exact source-byte and resolution checks for every accepted natural crop.
- Fresh SNOW2/PNM extraction on one fixed crop from each natural scan; all four
  fresh permeability labels agree with their saved labels within 0.0001 log units.
- Measured forward latency for 24 held-out crops, 10 warmups and 30 repeats each,
  for each of the five neural formulations: 3,600 timed calls in total.
- Sixteen automated scientific/integration tests passed.

## Findings

The scale-normalized GAT with graph descriptors reduces parent-balanced log RMSE
from **1.5528** for the standardized original GAT to **0.8311**. The change combines
dimensional normalization with additional input descriptors and does not isolate
the effect of either change. The calibrated porosity baseline is still better
on the primary endpoint, at **0.7525**. The GAT variant's pooled log-space R2 is
**-0.1007** and factor-of-two accuracy is **22.5%**. All model rows, including
weaker outcomes, are retained.

Choosing a model family from inner validation is also evaluated explicitly.
The inner-selected GAT has parent-balanced RMSE **1.4565**; its poor result shows
that model selection itself is unstable with only three training scans. The
best-looking fixed test row must not be presented as a model selected without
examining test results.

The final resource pass measures median GAT forward latency **1.7185 ms**,
PNM solve latency **1.8537 ms**, and SNOW2 latency **4.0572 s**. The solve and forward
distributions cover different crop counts; paired crop comparisons are exported
separately. There is no basis for a large solver speedup claim. These data also
do not establish end-to-end millisecond inference or a CNN memory advantage.

The current result strengthens the evaluation but **does not substantiate the
paper's existing high-accuracy or attention-superiority claims**. It supports
reporting a limited model comparison and a substantial cross-scan failure mode.

## Artifact map

| File or directory | Contents |
|---|---|
| `protocol.json` | Frozen candidate grid, seed list, evaluation rule, run duration and source hashes |
| `inner_selection.csv` | Every inner evaluation, its training parents and validation parent |
| `selection_parent_*.json` | Selected settings and training epochs for each outer fold |
| `predictions.csv` | Every outer prediction, true label, seed, model and selection strategy |
| `metrics.csv` | Mean single-model metrics and seed standard deviations |
| `per_parent_metrics.csv` | Separate error for each held-out natural scan |
| `ensemble_predictions.csv` | Mean log predictions and descriptive seed spread |
| `tables/ensemble_metrics.csv` | Explicitly separated five-seed ensemble metrics |
| `paired_parent_comparisons.json` | Paired scan effects and exploratory exact sign-flip probabilities |
| `checkpoints/` | Trained models, fitted transforms and training-parent IDs |
| `data_audit/` | Source headers, hashes, exact crop matching and limitations |
| `resources_verified/` | Final measured timing data, 3,600 raw repeats, memory scopes and fresh label checks |
| `resources/` | Preserved first resource pass; superseded by the verified pass |
| `figures/` | Four visually checked figures in PNG, PDF and editable SVG |
| `tables/` | Markdown and LaTeX paper tables |
| `run_sources/` | Exact source revisions used during the training run |

Parent macro RMSE is the square root of the average parent MSE. Seed SD is not
a confidence interval over independent rocks. PNM labels are approximations and
numerical conservation is not physical validation against direct simulation.
The four-scan design cannot test population-wide geological generalization.
The observed physical pore-radius CV is 0.459–0.806, so CV=1.5 is untested.

## Reproduction

Use the recorded package versions and local trusted data. Choose a fresh output
directory; the scripts refuse to overwrite completed runs.

```powershell
python scripts/run_research_v2.py --output experiments/reproduction_v2
python scripts/audit_research_data.py --output experiments/reproduction_v2
python scripts/measure_research_resources.py --output experiments/reproduction_v2 --resource-directory resources_verified
python scripts/build_research_artifacts.py --output experiments/reproduction_v2
python -m unittest discover -s tests -v
```

`run_sources/` preserves the files whose hashes were recorded during training.
The workspace runner has since received a comment correction; no training logic,
settings, predictions or selection outcomes were changed after evaluation.
The resource-measurement and artifact scripts retain their own provenance in
the supplementary source hashes. Original archives remain under the ignored
`data/public/` directory, and no credentials are included in this package.

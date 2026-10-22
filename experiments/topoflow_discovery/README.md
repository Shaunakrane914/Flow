# Sealed critical-throat discovery evaluation

Read [REPORT.md](REPORT.md) for findings and [MANUSCRIPT_INSERT.md](MANUSCRIPT_INSERT.md)
for paper-ready passages. The experiment addresses the review of commit `0ad1731`.

The new five-scan test supports useful geometry-only ranking against random
selection, but establishes no advantage over the strongest analytical ranking,
no common-pool response-accuracy advantage and no CPU runtime advantage.
The study remains conditional on the specified cylindrical-throat PNM.

## Evidence

- [Protocol](protocol.json): five new scan IDs, fixed crops, models, twenty checkpoint hashes and source hashes.
- [Selection seal](selection_seal.json): saved rankings and model-selected top/low/random throats, sealed before outcomes.
- [Crop attempts](crop_attempts.csv): all ten fixed attempts; no fallback crops or flow-based selection.
- [Source-byte audit](source_byte_audit.json): phase fractions, full-image hashes and no exact duplicates against 56 earlier crops.
- [Fresh outcomes](outcomes.csv), [common-pool predictions](common_pool_predictions.csv) and [physics checks](physics_checks.json).
- [Discovery metrics](discovery_metrics.csv), [response errors](accuracy_metrics.csv), [paired scan comparisons](paired_scan_comparisons.json).
- [Measured timing calls](timing_observations.csv), [timing summary](timing_summary.csv), [observed memory](resources.json).
- [Verification](verification.json), [test output](test_output.txt), [source ledger](source_ledger.json).
- [Figures](figures/model_selected_throat_loss.pdf), also available as PNG and SVG.

## Reproduce

Run from the repository root. Python package versions are in `verification.json`.
The twenty checkpoints are already committed under the preceding follow-up.
Raw images and derived geometry remain in ignored `data/`; no new redistribution
licence is asserted. Retrieve the official source archive if it is absent:

```powershell
python scripts/fetch_official_archives.py
```

Frozen hashes require the exact source bytes, including line endings. If a checkout
changes line endings, restore the five frozen source files from the committed
`source_snapshot/` using the paths in `protocol.json` before reproduction. This
overwrites only those five research files and should be done in a fresh checkout.

To preserve the committed evidence and the original local data, use two fresh
directories:

```powershell
python scripts/reproduce_throat_discovery.py --output experiments/discovery_replication --data data/discovery_replication
```

The wrapper runs the unchanged freeze → select → evaluate stages and generates
the report. It refuses existing directories. Reproduction evaluates fixed
checkpoints and does not require recovering or retraining the old development
images. Model outputs should agree within numerical tolerance; timings depend
on hardware and load. A local timestamp/hash seal records execution order; it is
not an externally registered preregistration.

To regenerate the released report or verify its evidence in the original tree:

```powershell
python scripts/build_discovery_report.py
python scripts/verify_throat_discovery.py
python -m unittest discover -s tests -v
```

`freeze`, `select` and `evaluate` are also available separately in
`scripts/run_throat_discovery.py`; their default directories are the released
paths, so existing artifacts intentionally prevent rerunning those stages there.

## Statistical and physical scope

The independent grouping unit is the named source scan. Two crops, many throats,
three doses and five seeds do not create more independent geological specimens.
Five source groups permit no two-sided exact paired p-value below 0.0625.
Bootstrap intervals are descriptive. The PNM sensitivity control requires a base
flow solve and the exact response control also requires factorization/leverage
precomputation; neither is solver-free. No voxel-flow, real-clogging or matched
physical validation has been added. The earlier S1/S5 reference discrepancy
persists and is disclosed in the report and manuscript text.

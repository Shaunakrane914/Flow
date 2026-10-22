# Comparative consolidation after review of 4c48050

- [Diagnostic report](REPORT.md): what the new analyses establish and leave unresolved.
- [Consolidated manuscript](MANUSCRIPT.md): complete Markdown research draft with all three authors.
- [Analysis plan](analysis_plan.json): explicitly post-hoc, including the initial debug case and OpenPNM API correction.
- [Prospective protocol](VALIDATION_PROTOCOL.md): unexecuted work kept separate from paper findings.

No new architecture, trained checkpoint, fitted switching rule or test-set
calibration is introduced. The preceding frozen study remains unchanged.

## Main evidence

- [Error strata](response_strata.csv), [per-graph counts/errors](response_per_graph_strata.csv), [outcome distribution](outcome_distribution.csv).
- [Geometry metrics by graph](network_diagnostics_per_graph.csv) and [by scan](network_diagnostics_per_scan.csv), [all four exploratory associations](exploratory_associations.csv).
- [Identical-conductance assembly checks](independent_solver_checks.csv), [fixed crop physics variants](crop_physics_variants.csv).
- [Original S1/S5 network closure results](original_network_reference.csv), [source hashes and boundary conventions](original_network_provenance.json).
- Three figures in [PNG](figures/topology_and_ranking_advantage.png), PDF and SVG; [verification](verification.json), [test log](test_output.txt) and exact [source ledger](source_ledger.json).

## Reproduction

Run from the repository root with the versions recorded in `verification.json`.
The official `data/public/ImagesICPSC2007.zip` and the ten saved geometry NPZ
files in `data/throat_discovery/` are required. Restore the preceding discovery
inputs using its documented reproduction workflow if necessary. No model
retraining or full-volume SNOW extraction is needed for this diagnostic analysis.
Follow that workflow's source-byte restoration instructions if a checkout changes
line endings in the older frozen files; the new diagnostic sources preserve their
bytes through repository attributes and `source_snapshot/`.

The released output directory is intentionally protected from overwrite. For a
fresh analysis using the original local inputs:

```powershell
python -c "from pathlib import Path; from scripts import analyze_discovery_review as a; a.OUT=Path('experiments/diagnostic_replication'); a.run(); from scripts import build_diagnostic_paper as b; b.OUT=a.OUT; b.main()"
```

If the discovery reproduction used alternate input/output directories, set
`a.DATA` and `a.OLD` to those directories before `a.run()`. Use a fresh output
directory. The native-network phase uses the official original maximal-ball
files in the archive, preserves pore-endpoint length correspondence, and records
its reservoir-removal convention. These choices do not reproduce every aspect
of the historical network model or its boundary handling.

To rebuild only the released documents/figures or run checks:

```powershell
python scripts/build_diagnostic_paper.py
python scripts/verify_research_diagnostics.py
python -m unittest discover -s tests -v
```

Numerical agreement between implementations is not physical validation. This
post-hoc analysis does not add independent test specimens or fix the S1/S5
SNOW/image discrepancy. Generated paper statements remain conditional on the
declared PNM approximation.

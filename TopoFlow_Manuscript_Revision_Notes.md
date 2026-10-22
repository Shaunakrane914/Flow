# TopoFlow manuscript revision notes

Revised 23 October 2026.

The editorial follow-up applies the five accepted changes: a lattice-BGK source,
rank-one-update provenance with a [derivation](docs/critical_throat_response_derivation.md),
a precise Pore-GNN description, a concrete abstract ending and shorter closing
sections. The Results text and quantitative tables are unchanged except for
reference renumbering. The author table, eleven equations, four figures and page
sections match the preceding Word version. All six final pages were inspected.

The current paper is [TopoFlow_IEEE_V4_Cited_Revised.docx](TopoFlow_IEEE_V4_Cited_Revised.docx).
It edits the retained `TopoFlow_IEEE_V4_Cited_Equations.docx` manuscript; the source
file remains unchanged. The permeability objective, SNOW2 extraction, GAT encoder,
section sequence, aligned three-author table and two-column page geometry remain.

The title now reads **Permeability Prediction in Micro CT Scans using Graph
Attention Networks**. Removing “Millisecond” avoids presenting forward-only
latency as the time for image extraction and prediction together.

The abstract and methods identify ImagesICPSC2007 and the separate Estaillades
archive, fixed 128³ crops, physical voxel sizes, source grouping and PNM targets.
The GAT architecture and attention equations replace the GraphSAGE description.
The paper distinguishes bulk regression from the later edge-aware restriction
extension. It describes source-separated model selection and the locked primary
model rather than selecting a winner from external outcomes.

Results now use the completed experiments:

| Evidence | Findings included | Source |
| --- | --- | --- |
| Nested natural-rock validation | 24 crops, four development scans, five seeds; parent-balanced log-RMSE 0.8311 for GAT, 0.8713 for GraphSAGE and 0.7525 for calibrated porosity KC | [Comparison](experiments/topoflow_research_v2/tables/paper_comparison.md) |
| Controlled GAT ablation | Standard 1.5528; normalization 0.9504; descriptors 0.8933; both 0.8311 | [Metrics](experiments/topoflow_followup/ablation/metrics.csv) |
| Locked external permeability | 32 crops from S1/S3/S5/S9; primary GAT log-RMSE 0.5687 versus physics ridge 0.2534; negative GAT log-R² retained | [Predictions](experiments/topoflow_followup/external_test/predictions.csv), [metrics](experiments/topoflow_followup/external_test/metrics.csv) |
| Frozen throat selection | Ten crops from five new scan IDs; 3.57% top-selected mean loss for GAT, 0.37% for random and 4.08% for affine pressure | [Discovery study](experiments/topoflow_discovery/REPORT.md) |
| Response calibration | Shared pool of 897 interventions; GAT MAE 2.04 pp versus no-response 0.46 pp; consequential-intervention errors also reported | [Diagnostics](experiments/topoflow_diagnostics/REPORT.md) |
| Cost and reference checks | Measured CPU scopes, numerical consistency, closure sensitivity and unresolved physical-reference differences | [Component timing](experiments/topoflow_research_v2/resources_verified/resource_report.md), [frozen timing](experiments/topoflow_discovery/timing_summary.csv), [reference checks](experiments/topoflow_diagnostics/verification.json) |

The old R² 0.963, MSE 0.0024, 512 GB comparison, 9,000-fold speedup and asserted
topological threshold are removed from this revision. Four figures now show the
correct workflow and measured comparisons. The original duplicate lists of figures
and tables are omitted. The bibliography contains seventeen checked references,
numbered by first appearance; prior GNN permeability work is acknowledged.

The discussion separates ranking from response calibration, and numerical
agreement from physical validation. Completed outcomes are stated directly.
Unperformed validation remains confined to future scope. The evidence supports a
comparative PNM surrogate study, without an established general GAT advantage.

Verification used the managed document runtime. Microsoft Word rendered six pages;
all six page images were inspected. Eleven display equations remain native Word
math. The author table and original section properties were checked against the
source; a closing continuous section balances the last page. Eight external-model
metric sets and seven ranking and response summaries were independently recomputed
from saved predictions. No model was retrained or test-set calibration fitted for
this revision.

The packaged LibreOffice renderer was attempted but its executable is absent from
this Windows runtime; installed Word COM and bundled Poppler supplied the visual
check. QA PDFs and page images remain local. The generation script is
[scripts/revise_existing_paper.py](scripts/revise_existing_paper.py).

Source DOCX SHA-256:
`0297f6ad073d2c5003d713b63f36a27d2bbb36e912f5fa2ba950f08485b4da9c`.

Revised DOCX SHA-256:
`27541b2bef816a024b0383e511167bf939eeb3598ab135a9700aad7d31b0c940`.

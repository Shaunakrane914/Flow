# TopoFlow manuscript revision notes

Revised 23 October 2026.

**Current deliverable:** [restored LaTeX proceedings PDF](output/pdf/TopoFlow_ICISEA2026_Restored.pdf),
13 pages, with [editable source](output/pdf/TopoFlow_ICISEA2026_LaTeX.zip).
It uses the official publisher-linked SVPROC package with a balanced native
title/author block, serif contact information, a corresponding-author footnote
and corrected spacing and float placement. All pages were visually inspected;
content preservation was checked for the complete abstract and research prose,
11 equations, four images, three tables, all captions and 18 references.
Both Word manuscripts remain unchanged. See the
[restoration record](docs/latex_format_restoration.md).

The previous Word conversion is retained below as revision history.

The earlier deliverable is the separate
[Atlantis proceedings copy](TopoFlow_Atlantis_ICISEA2026.docx), built from the
Word template linked on the official Atlantis Press author-resources page.
All 14 rendered pages were inspected. The six-page IEEE input remains unchanged.
The body uses one column and numbered sections, with template title/author styles,
captions and reference formatting. Shaunak is the user-confirmed corresponding
author; the confirmed no-competing-interests statement precedes References.
All eleven native equations, four figure images, three data tables, eighteen
sources and substantive prose were verified against the IEEE input. See the
[conversion notes](docs/atlantis_format_conversion.md) for the current output hash,
template provenance and remaining venue-specific requirements. No conference
page limit has been verified.

The revision history of the retained IEEE manuscript follows.

The latest update adds Shivaji D. Pawar as fourth author with the user-supplied
email `shivaji.pawar@universalai.in`. Four equal-width author columns preserve the
original order and centered alignment. Affiliations remain omitted as requested.
The user's renamed file is retained as
[TopoFlow_Physics_Guided_Graph_Attention_ICISEA2026.docx](TopoFlow_Physics_Guided_Graph_Attention_ICISEA2026.docx).

One verified paper is added to the image-learning paragraph, comparing attention
over image features with attention over graph neighbors. Its pages are corrected
to 1003–1027 from the publisher PDF and Crossref record. It becomes reference [7]
in first-citation order; the bibliography now has eighteen references.
The [citation selection review](docs/pawar_citation_review.md) records why the
other supplied publications were not added. No author-contribution statement or
new experimental claim was introduced.

All six rendered pages were inspected. Result tables, eleven native equations,
four figure images and section properties match the preceding draft. Results
prose differs only in citation numbering. The same six cached source papers were
compared with 69 current text blocks (3,494 normalized tokens), with no exact
eight-token match. This is not a Turnitin or AI-detector result.

The preceding reviewer corrections were applied to the existing manuscript. The title
reads **Evaluating Physics Guided Graph Attention for Permeability Prediction
and Critical Pore Throat Discovery**. Shaunak's confirmed email is
`shaunak.rane@universalai.in`; the preceding three-author layout retained its alignment.
Affiliations are omitted at the user's explicit request.

The feature paragraph now distinguishes the raw ten-element vector from the
transformed vector used by the GAT descriptor branch and physics ridge. The last
feature is pore-radius CV, and the edge count includes both directions of each
represented throat. ExtraTrees uses the raw descriptors. The two porosity
transforms in the neural branch are deliberate, so replacing the entire paragraph
with the raw vector alone would also misdescribe the experiment. See the
[ordered descriptor definitions](docs/graph_descriptor_definitions.md).
The relevant source files match their frozen follow-up hashes. The checkpoint
sentence now says "20 checkpoints (four models × five seeds)."

Microsoft Word rendered the revised file to six pages and every page image was
inspected. All eleven native equations, four figure images, feature and result
tables, and page-section properties match the preceding revision. Results prose
is identical except for the checkpoint clarification. The latest comparison of
69 text blocks (3,469 normalized tokens) against the six cached source papers
found no exact eight-token match; this remains a bounded source comparison, not
an AI detector or plagiarism clearance.

The [official ICISEA 2026 homepage](https://conference.aiktc.ac.in/) and the
[public Meteor About page](https://meteor.springer.com/project/dashboard.jsf?id=3188&tab=About)
confirm MULTINOVA 2.0 and its AI-focused scope. Neither inspected page establishes
the required manuscript template. The retained IEEE layout is therefore not
claimed to satisfy the publisher's submission instructions; template compliance
remains unresolved. No unrelated ICISEA event's template was substituted.

The earlier editorial work is recorded below.

The editorial follow-up applies the five accepted changes: a lattice-BGK source,
rank-one-update provenance with a [derivation](docs/critical_throat_response_derivation.md),
a precise Pore-GNN description, a concrete abstract ending and shorter closing
sections. The Results text and quantitative tables are unchanged except for
reference renumbering. The author table, eleven equations, four figures and page
sections match the preceding Word version. All six final pages were inspected.

The current paper is [TopoFlow_Physics_Guided_Graph_Attention_ICISEA2026.docx](TopoFlow_Physics_Guided_Graph_Attention_ICISEA2026.docx).
It edits the retained `TopoFlow_IEEE_V4_Cited_Equations.docx` manuscript; the source
file remains unchanged. The permeability objective, SNOW2 extraction, GAT encoder,
section sequence, aligned three-author table and two-column page geometry remain.

The preceding title was **Permeability Prediction in Micro CT Scans using Graph
Attention Networks**. Removing “Millisecond” avoided presenting forward-only
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
`d9b7f24f5f74370cf1491e4172ec5a9e750cba27643fb9a6c4b1002c4ce281cf`.

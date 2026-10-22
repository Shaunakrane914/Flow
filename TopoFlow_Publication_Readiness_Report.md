# TopoFlow publication readiness and recovery report

Prepared for the TopoFlow authors on 22 October 2026

**Reviewer corrections completed, 23 October 2026:** The existing Word paper now
uses the comparative title recommended in the review, distinguishes raw and
transformed graph descriptors, correctly identifies pore-radius CV, clarifies the
20 frozen checkpoints and uses Shaunak's confirmed `universalai.in` email.
Affiliations are omitted as requested. Six pages were rendered and inspected;
measured results, tables, figures and eleven native equations are preserved.
The [descriptor definitions](docs/graph_descriptor_definitions.md) record the
exact feature order. Required publisher-template compliance remains unresolved:
the inspected [official conference site](https://conference.aiktc.ac.in/) and
public Meteor About page identify the event but do not supply a template.

**Editorial follow-up completed, 23 October 2026:** The
[existing Word paper](TopoFlow_IEEE_V4_Cited_Revised.docx) now includes the two
missing equation sources, an explanation of the rank-one response readout,
a corrected Pore-GNN description and a more concrete abstract conclusion.
Future Scope and Conclusion are shorter. The results, figures, native equations,
authors and section geometry are preserved. All six pages were inspected; the
bibliography has seventeen references. A
[technical derivation](docs/critical_throat_response_derivation.md) documents
the existing response formula. The [originality review](TopoFlow_Originality_Review.md)
records the new file hash and a repeat comparison against the six cached papers.
No AI detector score is claimed.

**Originality review, 23 October 2026:** The
[source-overlap and attribution review](TopoFlow_Originality_Review.md) and
[paragraph audit](TopoFlow_Originality_Paragraph_Audit.md) inspect the pre-edit
Word version. No exact eight-token prose match was found against six
downloaded source papers. This is not a Turnitin result or an AI-authorship
score. The review identifies missing equation provenance for the LBM background
and restriction readout, plus a distinction needed between Pore-GNN's CNN feature
graphs and physical pore networks. The manuscript was not edited in this audit.
Those three findings were addressed in the later editorial follow-up above.

**Current Word paper:** [TopoFlow IEEE revised manuscript](TopoFlow_IEEE_V4_Cited_Revised.docx)
updates the existing paper while preserving its SNOW2–GAT permeability objective,
section sequence, three-author block and two-column layout. The six-page revision
incorporates the completed source-separated benchmark, controlled ablations,
locked external test and frozen critical-throat experiment. Native Word equations,
measured figures and corrected references replace the unsupported claims.
[Revision notes and evidence](TopoFlow_Manuscript_Revision_Notes.md) identify the
changes and verification. The original Word files are retained.

**Comparative consolidation after review of 4c48050:** The new
[diagnostic report](experiments/topoflow_diagnostics/REPORT.md) adds outcome-stratified
response errors, four exploratory topology associations and sixteen matched-conductance
assembly checks. On interventions causing at least one percentage point of loss,
guided GAT MAE is 6.66 points versus 4.51 for no response. Overprediction therefore
persists beyond the low-impact majority. Shortest-path tortuosity correlates with
ranking advantage across these five already inspected scans (Spearman 0.60),
but no validated regime or selector is established. Independent numerical assemblies
agree to approximately 2e-14; original-network closure results still differ from
published network values by 12.5% and 33.5%, so physical validation remains unresolved.
The [consolidated manuscript](experiments/topoflow_diagnostics/MANUSCRIPT.md) includes
all three authors and presents completed comparative findings. Unperformed validation
is kept in a separate protocol, with no new architecture, calibration or switching rule fitted.

**Latest completed review response:** [The sealed discovery study](experiments/topoflow_discovery/REPORT.md)
tests five previously unused Imperial scan IDs (S2/S4/S6/S7/S8), ten fixed crops
and twenty frozen checkpoints. All geometry-driven selections precede outcome
calculation; 966 fresh intervention solves check numerical consistency. GAT top
throats yield 3.57% mean loss at 50% diameter restriction versus 0.37% for random,
while the affine-pressure control reaches 4.08%. On the shared pool, guided GAT
MAE is 2.04 percentage points versus 0.46 for no response. CPU timing establishes
no GAT advantage over exact PNM alternatives. This strengthens the evaluation
but weakens a claim of superior response prediction. The physical-reference
discrepancy remains unresolved. Use the new [manuscript insert](experiments/topoflow_discovery/MANUSCRIPT_INSERT.md)
for the completed findings; the earlier pilot below is historical context.

**Further completed experiments:** [The follow-up report](experiments/topoflow_followup/FOLLOWUP_REPORT.md)
now adds four named external scans and 32 crops, the controlled four-cell GAT
ablation, and a trained critical-throat response pilot. The guided GAT achieves
4.57 percentage-point response MAE versus 6.08 for predicting no response, on one
graph per external scan. Its learned hydraulic-importance head is distinct from
native GAT attention. This is preliminary PNM-based evidence with close controls,
not a demonstrated population-level advantage or real-clogging validation.

**Experimental update, 22 October 2026:** A new [completed research evaluation](experiments/topoflow_research_v2/README.md)
now provides four-scan nested validation, five seeds, 100 trained checkpoints,
source-byte checks for 24 natural-rock crops, measured component timings, and
[replacement manuscript text](experiments/topoflow_research_v2/manuscript_insert.md).
The physics-and-geometry GAT reaches parent-balanced log RMSE 0.8311; calibrated
porosity KC reaches 0.7525. The experiment improves reproducibility but does not
support the original high-accuracy, architectural-superiority or large-speedup
claims. The assessment below is retained as the earlier review, rather than a
description of the subsequently completed work.

## 1 Executive assessment

**The project can support a renewed research study, but the current manuscript should not be submitted. Its main numerical claims need to be rebuilt from traceable experiments. Editing the language will not resolve the problems in the labels, training, comparisons, or claimed contribution.**

The recommended direction is a focused empirical paper asking: **Under what conditions does graph connectivity improve permeability prediction beyond geometric descriptors and calibrated analytical models, and what is the complete computational cost?** Treat heterogeneity as an explanatory variable to investigate. Do not begin with a commitment to discovering a threshold at 1.5 or reproducing the current headline accuracy.

This direction reuses the existing extraction pipeline, models, and application. It still requires correction of the scientific pipeline, new evaluation, and a clear contribution relative to earlier work. Publication remains conditional on what those experiments show and the scope of the eventual venue.

This is an assessment and proposed research plan. No application code, training scripts, model weights, or manuscript content were changed while preparing it.

## 2 Evidence and limits of this assessment

The assessment combines the two supplied reviews with the text and tables in [the manuscript](TopoFlow_IEEE_V4_Cited.docx), the current source files, saved result summaries, and selected primary literature checked online. The repository has existing uncommitted work; these findings describe the inspected working tree, rather than a clean historical release.

Evidence is described as follows:

- **Confirmed:** visible in the manuscript, implementation, or stored artifacts.
- **Derived:** an inference or calculation whose assumptions are stated.
- **Unresolved:** requires original data, prediction files, historical logs, or further experiments.

The local `data/` directory is absent. Checkpoints and several result summaries exist, but their existence does not establish which data, settings, and code revision produced them. This report does not reproduce the complete benchmark, validate the original scans, or establish the provenance of the manuscript figures. The literature check covers important overlapping work; it is not an exhaustive systematic review.

## 3 How the existing reviews should be interpreted

The reviews correctly identify serious evidence gaps, but several conclusions need updating now that code is available.

| Review concern | What the workspace establishes | Consequence |
|---|---|---|
| The title says attention but the method describes GraphSAGE | The standard model actually contains three `GATConv` layers. The hybrid model uses `GATv2Conv`. | Describe the implemented GAT correctly; changing the title to GraphSAGE would introduce another error. |
| Edge geometry may be unused | The standard graph contains diameter and volume as node inputs and connectivity as edges. Its forward pass does not accept throat attributes. | Remove the claim that throat geometry is directly supplied to this model, or test an explicit extension later. |
| Code and checkpoints were unavailable to reviewers | They are present locally, but a reproducible release and run provenance are missing. | Package and document the actual experiment; do not describe the project as having no implementation. |
| Calibrated Kozeny–Carman is missing | Several baseline scripts fit a coefficient. Other paths use invalid porosity proxies or calibrate before splitting. | Audit each comparison separately. Fitting a coefficient exists, but does not establish a sound baseline. |
| A random split probably leaks overlapping crops | Inspected training scripts split sorted files. Estaillades and Savonnières processing uses non-overlapping 128-voxel crops. | Overlap leakage is not demonstrated. Shared parent scans, spatial correlation, and test-based selection still need attention. |
| 0.8 seconds cannot be called milliseconds | 0.8 seconds equals 800 milliseconds. | The important issue is which processing stages were timed. Use a precise title without promotional timing language. |
| Non-monotonic benefits disprove a threshold | Different benefits above a threshold do not mathematically disprove a crossover. | The stronger objections are inconsistent CV definitions, sparse coverage, and lack of independent threshold validation. |

Sources: [standard model](src/model.py), [hybrid model](src/model_hybrid.py), [Estaillades processing](src/process_estaillades.py), [Savonnières processing](src/process_savonnieres.py), and [baseline comparison](src/baseline_savonnieres.py).

Do not treat the review scores as publisher decisions, or their concerns as proof of fabrication. Resolve each concern against evidence. The present records are insufficient to substantiate the headline results, but that is different from establishing misconduct.

## 4 Scientific problems that must be resolved first

### 4.1 Establish what the prediction target actually is

**Confirmed:** The manuscript repeatedly describes LBM-derived supervision and a 32-core LBM experiment. The inspected label-generation path calls `OpenPNM` network flow, assigns hydraulic conductances, solves pore pressures, and obtains permeability using Darcy's law. This is a pore-network approximation; it is not a voxel-resolved LBM simulation. See [physics.py](src/physics.py), especially the label-generation function beginning at line 23.

Recover the original label records before deciding what the paper can claim. Every sample needs a label source, physical units, solver version, geometry settings, and flow direction. If a separate LBM dataset exists, establish its sample-by-sample correspondence to the graphs. If it does not, rewrite the study as a surrogate for the specified pore-network solver.

This distinction changes the baseline design. **If OpenPNM generated the targets using the same networks and settings, rerunning that solver reproduces the reference by construction. A GNN cannot demonstrate greater physical accuracy than that reference merely by matching its outputs.** It can demonstrate an approximation error versus runtime trade-off. Claims about actual rock permeability require an independent reference, such as suitable published simulations, independently run voxel-resolved flow, or experimental measurements with compatible conditions.

Independent reference data need not cover every sample initially. A deliberately selected subset spanning formations, porosity, connectivity, and permeability can reveal whether the network labels are credible. Keep that validation separate from GNN training and acknowledge its limits.

### 4.2 Validate geometry and physical scaling before regenerating labels

The current pipeline has several confirmed inconsistencies:

- [The physics solver](src/physics.py) fixes voxel size at 2.68 micrometres. ILS processing uses 1 micrometre and Estaillades processing uses 3.31 micrometres. Domain dimensions and boundary margins therefore do not consistently follow the extraction resolution.
- The processing functions in [Estaillades](src/process_estaillades.py), [Savonnières](src/process_savonnieres.py), and [ILS](src/process_ils.py) can take `pore.equivalent_diameter` and apply a square-root formula intended for area. A diameter cannot be substituted for an area in that formula. The numerical impact depends on which properties existed in the original extraction outputs.
- Some extraction paths estimate throat dimensions from connected pores instead of using a documented extracted throat measure. These approximations must be identified and justified.
- Boundary selection uses coordinate extrema, while domain length is derived from array dimensions. The correspondence between array axes, network coordinates, padded boundary nodes, and flow direction needs verification.
- Sparse, disconnected, or failed networks are excluded. These rules change the population studied and must be reported rather than silently interpreted as representative of all crops.

PoreSpy explicitly distinguishes equivalent and inscribed diameters and requires the user to choose the diameter definition. It also cautions that added boundary nodes have fictitious geometry. Use those distinctions when auditing the original network properties. [PoreSpy SNOW2 documentation](https://porespy.org/examples/networks/reference/snow2.html).

Before training again, establish analytical sanity cases, pressure-flow linearity, inlet–outlet mass balance, sensitivity to the conductance model, and consistency of physical dimensions. Check crop-size sensitivity on selected heterogeneous samples before calling them representative elementary volumes. State viscosity, pressure difference, convergence checks, and treatment of non-percolating samples. A successful solver return alone is not physical validation.

**Required decision:** If geometry or unit errors affected the original graphs, regenerate the affected graphs and labels. Retraining on invalid labels will not repair the study.

### 4.3 Correct the training objective and evaluation procedure

**Confirmed:** Four standard training scripts compare model output shaped `[B, 1]` with targets shaped `[B]`. Broadcasting turns paired regression into comparisons across the batch. The earlier workspace check reproduced a loss of 0.5 for perfectly paired predictions that should have zero loss. The same shape problem affects the manually calculated R² in these scripts.

Affected scripts: [Savonnières](src/train_savonnieres.py), [Estaillades](src/train_estaillades.py), [ILS](src/train_ils.py), and [synthetic](src/train_synthetic.py). The hybrid loss explicitly aligns target dimensions and should not be described as having this same bug.

The standard scripts also select the best checkpoint using test loss. [Hybrid training](src/train_hybrid.py) fits the Kozeny–Carman coefficient on all loaded graphs before splitting, exposing test labels to calibration. These are separate validity problems.

Future experiments need aligned prediction and target shapes, an independent validation process for all selection, and a final test set that is evaluated after choices are fixed. All fitted preprocessing and baseline parameters must use training data only. Record seeds, configuration, checkpoint selection, and sample assignments. Correcting the loss requires retraining affected models; recomputing metrics alone is insufficient.

### 4.4 Rebuild all numerical comparisons from predictions

**Confirmed:** [The Estaillades comparison](src/baseline_estaillades.py), line 79, assigns `gnn_mse = 0.0802` rather than evaluating a checkpoint in that comparison. Because the training script has the broadcasting problem, this copied training value is especially unsuitable for a matched comparison. The Savonnières comparison does evaluate predictions and flattens them, but also contains a placeholder branch when weights are absent. [Dashboard values](web/routers/dashboard.py) are manually embedded numbers.

These files are useful records of project development. They are not sufficient evidence for the paper's results. Missing models or failed runs should be reported as unavailable, never substituted with a plausible score.

Export a prediction record for every evaluated sample, including sample ID, parent scan, formation or synthetic category, split, label source, true permeability, predicted permeability, model, seed, and checkpoint identifier. Derive every table and plot from those records.

Define the target transformation exactly. The code uses log transforms with additive constants, and constants differ across extraction paths. Inverting `log10(V + epsilon)` as `10**x` does not recover the original volume; it recovers `V + epsilon`. Reconstructing porosity from transformed features can therefore introduce bias. Prefer stored physical metadata and porosity measured directly from the binary image.

### 4.5 Reassess whether the existing wins are useful

The result summaries allow an important conditional calculation. For the same evaluation targets and scale:

`R²_GNN = 1 − (MSE_GNN / MSE_KC) × (1 − R²_KC)`

| Dataset | Stored KC MSE | Stored KC R² | Stored GNN MSE | Implied GNN R² |
|---|---:|---:|---:|---:|
| Savonnières | 5.5457 | −1.0784 | 2.9849 | approximately −0.119 |
| Estaillades | 0.1120 | −0.5466 | 0.0802 | approximately −0.107 |

**Derived, not a fresh evaluation:** These values use rounded numbers from [Savonnières results](results/savonnieres_results.txt) and [Estaillades results](results/estaillades_results.txt). They apply only if each row uses the same targets, transformation, and ordinary MSE/R² definitions. The copied Estaillades score makes that assumption particularly uncertain.

If the assumptions hold, both apparent wins remain worse than predicting the test-target mean in the R² definition. That mean is an explanatory reference, not a deployable model: the actual baseline must predict a constant estimated from training data. Add that baseline explicitly, along with a training-fitted porosity model and descriptor regression. An improvement percentage over a poor baseline does not establish useful predictive performance.

The manuscript's R² of 0.963 and MSE of 0.0024 remain unresolved. No inspected prediction record establishes their origin. Do not retain either number until both can be reproduced on identified observations with stated transformations. Scores from different transformations or test populations cannot be compared directly.

## 5 Select one defensible contribution

Earlier research already covers graph-based permeability prediction. I'Anson and colleagues use extracted pore networks and LBM supervision, compare with Carman–Kozeny, and include extraction in their timing discussion. This is direct overlap with the existing narrative. [I'Anson et al. 2025](https://ipjournal.interpore.org/index.php/interpore/article/view/47).

RTG-GNN uses pore and throat attributes with gating and attention for permeability prediction, so introducing attention or throat features alone would not establish novelty. [Zhao et al. 2024](https://www.sciencedirect.com/science/article/pii/S2949891024007280).

Pore-GNN combines CNN-derived features with graph models, providing another relevant representation comparator. [Alzahrani et al. 2023](https://www.sciopen.com/article/10.46690/ager.2023.10.05).

Zhao and Xiao embed learned conductances in a differentiable pore-network model. Its publisher record now identifies it as a 2026 Journal of Hydrology article, so the manuscript's 2025 Water Resources Research entry needs updating. [Zhao and Xiao 2026](https://www.sciencedirect.com/science/article/pii/S0022169426006992).

These primary records establish substantial overlap. They do not establish that every implementation detail is identical. Read the full closest papers before finalizing an architectural comparison or claiming a new gap.

| Possible direction | Evidence required | Recommendation |
|---|---|---|
| A reproducible study of generalization and when connectivity helps | Independent evaluation, strong descriptors, connectivity ablations, full runtime accounting | Preferred starting direction; a focused empirical contribution if results reveal a useful insight |
| Heterogeneity-dependent model selection | A measurable input-only descriptor, broad coverage, training-only rule selection, independent validation | Potential extension; do not promise a universal threshold |
| A new physical architecture | A specific unmet limitation, related-method comparisons, mechanistic ablations, independent physical validation | Higher effort; standard attention or residual learning is insufficient by itself |
| A software demonstration | Documented workflow, usability, reproducibility, and a venue accepting software or demonstration contributions | Suitable alternative if scientific gains are limited; the web interface is supporting software rather than proof of a new permeability method |

A suitable working title for the recommended study is **Evaluating Graph Connectivity and Generalization in Pore Network Permeability Prediction**. Use a title referring to a pore-network solver surrogate if that is the only validated target. Select the final title after the results establish the contribution.

## 6 Redesign the experiments around the claims

### Dataset and independence

Recover the raw data and produce a manifest containing source citation or DOI, licence, parent scan ID, lithology, physical resolution, phase labels, crop coordinates, crop size, orientation, acceptance status, graph statistics, and label provenance. Preserve rejection counts and reasons.

The manuscript says 256³ inputs, while inspected processors use 128³ crops. It also refers to Berea sandstone and then lists other rock categories. Resolve this from records. Describe synthetic media as synthetic, and identify MEC and ILS precisely; synthetic samples are not a fifth geological formation. Verify the 1,231 accepted samples and the number of independent parent scans rather than treating a dashboard count as a dataset inventory.

Use two explicitly different evaluation questions where data allow:

1. **New specimens within known categories:** group all crops from a parent specimen together. Use independent parent scans for validation and testing.
2. **Transfer to an unseen rock category:** hold out the entire category and select hyperparameters using only the remaining data, for example through inner grouped validation.

If only one parent scan exists for a category, an independent within-category specimen test is impossible with the current collection. Acquire additional parents or limit the claim. Spatially separated blocks with a buffer can evaluate within-scan transfer, but do not establish generalization to independent specimens. Keep duplicates, overlapping crops, augmentations, and resampled versions in the same split.

Five nominal categories do not create five equally credible geological transfer tests. Explain the role of synthetic data and report each holdout separately. Do not use formation-specific checkpoints selected for a supposedly unseen formation.

### Minimum experimental package

| Experiment | Status for recommended paper | Evidence to retain |
|---|---|---|
| Validate geometry and reference labels | Required before training | Physical settings, sanity checks, solver residuals, reference-subset comparisons, exclusions |
| Train-only constant, calibrated KC, descriptor regressor, and corrected current GAT | Required | Common sample partitions, tuning protocol, predictions, per-category scores |
| Independent parent or category holdout evaluation | Required to support generalization | Parent counts, split manifest, held-out predictions, explicit limits where data are insufficient |
| Connectivity ablation | Required to claim connectivity helps | Matched feature-only/set model and controlled edge-removal or rewiring comparison |
| Repeated training and uncertainty | Required to quantify stability | A practical starting point is five seeds; paired errors and intervals respecting parent-scan grouping |
| Extraction and inference benchmark | Required if retaining efficiency claims | Per-stage timings, hardware, memory profiles, sample dimensions, cold and warm conditions |
| Independent voxel-resolved or experimental reference subset | Required for claims of physical accuracy beyond reproducing PNM | Correspondence to samples, reference uncertainty, agreement and failure cases |
| 3D CNN comparison | Required if claiming superiority to voxel CNNs | Specified architecture, matched data, reasonable tuning, accuracy and resource measurements |
| Direct related-GNN comparison | Required for a claimed architectural advance | Public implementation or clearly labelled faithful reproduction, matched protocol |
| Threshold and segmentation sensitivity | Required only for those respective claims | Defined descriptor, independent rule validation, perturbation settings and uncertainty |

Start with a small validated pilot and inexpensive baselines. Do not launch an extensive CNN or GNN sweep before the labels and paired loss are correct. Add an edge-aware architecture only if the research question warrants it and the current experiments reveal a specific limitation.

For connectivity ablations, retain the same node geometry and model capacity where possible. A model can use edges without establishing that topology adds value beyond size or distribution statistics. Rewiring is a diagnostic intervention, not necessarily a physically realizable rock; explain that limitation. Boundary and flow-axis information should be investigated if directional behavior is claimed. An undirected network is not intrinsically invalid for pressure-driven flow.

Report RMSE and MAE in a clearly defined logarithmic space, R² in the stated space, and factor-of-two accuracy on positive permeability. Include physical-unit errors where meaningful. Specify a reference unit when defining logarithms, and keep any epsilon or non-percolation convention explicit. Present per-category results and both pooled and category-balanced summaries. Pooled R² can be inflated by differences between categories while hiding poor predictions within each category.

Estimate uncertainty at the level of independent parent scans where possible. Training seeds measure optimization variability; many correlated crops do not substitute for additional independent specimens. With few parents, present the limited evidence honestly rather than producing artificially narrow intervals.

## 7 Replace the threshold claim with a testable hypothesis

**Confirmed:** README and the manuscript describe pore-size CV near 1.5. [The heterogeneity analysis](src/analysis_heterogeneity.py) instead computes CV from permeability labels, uses a log-CV threshold of 0.15, and estimates porosity from pore count divided by voxel count. These are different quantities, and the porosity proxy is not a pore-volume fraction.

The displayed category CV values also leave a large gap between approximately 0.85 and 2.5. Even if those values were valid pore-size CVs, any boundary inside that gap could separate the displayed winners. These observations do not identify 1.5 as a unique physical crossover.

For the first corrected paper, describe heterogeneity dependence as exploratory unless independent evidence becomes available. If pursued:

1. Define CV from a specified physical pore-size measure for each sample. An input-only measure can support prospective model selection; a descriptor requiring the unknown target cannot do so.
2. Measure porosity directly and examine confounding by formation, resolution, pore count, and permeability range.
3. Compare paired held-out errors of GAT, calibrated KC, and descriptor regression across the descriptor range.
4. Estimate any switching rule using training and validation data only. Evaluate the fixed rule on independent parents or categories, including against always selecting one model.
5. Use grouped resampling to assess uncertainty and check whether one category drives the pattern. If coverage near the proposed crossover is sparse, collect data there.

If no stable threshold appears, remove it. A measured account of where graph models fail or succeed can still be informative. Do not search many descriptors on the test set and present the best apparent breakpoint as a discovery.

## 8 Make computational claims comparable

Remove the 512 GB, 2.4 GB, 99.5%, 0.8-second, and 9,000-fold headline comparisons until their measurement records are available or they have been rerun. Treat this as an evidence requirement, not a claim that such measurements are impossible.

Measure image loading, any segmentation, SNOW2 extraction, feature construction, device transfer, and forward prediction separately. Report both complete image-to-prediction latency and prediction from an already extracted graph. Keep chart rendering and web communication separate from scientific inference.

For PNM supervision, the relevant acceleration comparison is GNN approximation versus solving the network equations, with the shared extraction cost accounted for. For independent LBM comparisons, include the actual voxel solver workflow and its accuracy conditions. Training cost and the number of predictions needed to amortize preparation may matter for a deployment claim.

Report CPU, GPU, memory capacity, threads, precision, batch size, software versions, warm-up, timing synchronization, repetitions, median and tail latency. Distinguish process CPU RAM, GPU allocated memory, and GPU reserved memory. Do not compare CPU RAM for one method with GPU VRAM for another as a single memory-saving percentage.

Node count reduction is not a measured memory reduction: features, bidirectional edge storage, intermediate tensors, solver arrays, and framework overhead all contribute. Report actual graph-size distributions and identify whether an edge count means physical throats or stored directed connections.

## 9 Reconstruct the manuscript from the evidence

Keep the application and existing figures as development assets. Rebuild the scientific narrative only after the corrected pilot determines what can be claimed.

| Manuscript section | Required revision |
|---|---|
| Title and abstract | State the actual model and validated target. Remove unverified numbers and speed language. Mention independent test design and the main limitation. |
| Introduction | State one unresolved question. Credit existing pore-network extraction and graph learning. Remove the claim that TopoFlow bypasses SNOW2 preprocessing. |
| Related work | Compare graph construction, attributes, supervision, physical constraints, holdout design, and end-to-end costs. Avoid comparing published MSE values across incompatible scales. |
| Data and labels | Give parent scans, accepted/rejected crops, sources, licences, resolution, solver and units. Reconcile 128³ versus 256³ and Berea versus the actual collection. |
| Method | Describe the three GAT layers, attention heads, two node inputs, activations, dropout, mean pooling, MLP, and exact target transform. Describe the hybrid separately if evaluated. |
| Physics | Correct Darcy's-law wording and symbols. Explain the network approximation and conductance assumptions. Limit the demonstrated task to single-phase scalar permeability. |
| Evaluation | Explain all splits, fitted transforms, checkpoint selection, tuning budget, baselines, seeds, and uncertainty. |
| Results | Supply the missing comparison Table 3 from actual matched evaluations. Include all test samples or label any plotting subset with its selection rule. |
| Discussion | State where the model helps, where it fails, label uncertainty, limited parent counts, and cost of extraction. Do not claim pressure or velocity prediction from a scalar output. |
| Conclusion | Answer the research question using validated results. Keep R² distinct from Pearson correlation r. Remove unsupported multiphase, universal-threshold, and real-time simulation claims. |
| References and disclosures | Verify authors, titles, venues, DOIs, and the claim each citation supports. Correct SNOW attribution. Add availability, contributions, funding, conflicts, and disclosures required by the selected venue. |

The original SNOW reference is Gostick's watershed extraction paper; version-specific PoreSpy documentation should accompany the software description. [Gostick 2017](https://doi.org/10.1103/PhysRevE.96.023307). Describe SNOW2 as watershed-based pore-network extraction rather than implying it is a newly invented skeletonization algorithm.

Rewrite unclear passages for their scientific meaning rather than replacing words with synonyms. Check author details, equations, units, section numbering, captions, and the final compiled layout. Add a technical workflow diagram whose features and outputs match the implementation. The prose alone does not establish whether a paraphrasing tool was used; do not treat the reviews' suspicions as proof.

## 10 Ordered recovery plan and decision points

### Phase 1 Recover and reconcile the evidence

Preserve the current manuscript, weights, summaries, and logs. Recover raw scans, graphs, source records, and any LBM outputs. Produce a claim-to-artifact register linking every retained number to its samples, predictions, configuration, and run. Identify which historical results cannot be traced.

**Proceed when:** the actual target, units, sample inventory, parent identities, crop dimensions, and split history are known. If records cannot be recovered, treat this as a new experiment rather than an attempted recreation of the old scores.

### Phase 2 Validate a small scientific pilot

Resolve geometry, scaling, porosity, transformations, reference-label validity, and the training objective. Evaluate corrected GAT against a training-fitted constant, calibrated KC, and one strong descriptor regressor on a defensible held-out group.

**Proceed when:** labels pass the agreed physical checks, predictions are traceable, and the pilot shows what question is worth pursuing. If labels are unreliable, repair or replace them first. If GAT offers no useful advantage, investigate that result before expanding the architecture.

### Phase 3 Complete the chosen study

Run the frozen grouped evaluation, repeated training, relevant connectivity ablations, and computational benchmarks. Add independent references or additional parent scans as required by the chosen claim. Record failures and negative outcomes.

**Proceed when:** the contribution survives matched baselines and uncertainty analysis. If simpler descriptors match GAT, reconsider the contribution as a carefully supported limitation or benchmark finding. If evidence is too narrow, pursue a software or student-research presentation with appropriately limited claims.

### Phase 4 Write and release the supported paper

Rewrite the manuscript from the final tables and figures. Prepare a reproducibility package with source revision, dependency versions, configurations, split manifests, model checkpoints, prediction records, dataset retrieval instructions, and licensing information. Release only material the authors are entitled to distribute; keep service credentials and unrelated generated files out of the package.

Select a specific venue by subject and article type, then check its current author instructions, review model, length, availability requirements, and disclosure policies. An IEEE-style file is not evidence of suitability for a particular conference or journal. Venue selection should follow the validated contribution rather than a desired acceptance probability.

## 11 Submission readiness criteria

Submit only when the authors can answer yes to all applicable items:

- Every retained numerical claim has a traceable run and prediction or profiling record.
- Architecture, graph attributes, target transformations, and solver descriptions match the implementation.
- Geometry, units, conductance assumptions, boundaries, and reference labels have been validated to the level claimed.
- Dataset sources, licences, parent scans, crop positions, and exclusions are documented.
- Validation and test roles are separate, with no test labels used in calibration or selection.
- Baselines use matched data and include a training-fitted constant and a strong geometry-only predictor.
- Any claim that connectivity helps has a controlled supporting comparison.
- Generalization claims match the actual level of independent holdout evidence.
- Uncertainty reflects both training variability and the limited number of independent samples.
- Efficiency comparisons include preprocessing and distinguish CPU RAM from GPU memory.
- The threshold is independently supported or removed, and negative results are reported.
- The missing results table is supplied, references are checked, and venue-specific presentation requirements are satisfied.

**Recommended immediate priority:** recover the dataset and establish label provenance, then validate the geometry and paired regression objective. Those steps determine whether a credible empirical paper can be built. Preserve the option to publish a narrower finding; do not preserve a numerical claim merely because it already appears in the abstract.

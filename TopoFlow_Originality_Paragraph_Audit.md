# TopoFlow paragraph audit

Reviewed 23 October 2026. This is a close-reading and exact-source-overlap ledger, not an AI detector report. See the [main review](TopoFlow_Originality_Review.md) for sources, findings and limitations. The reviewed Word manuscript remains unchanged.

IDs are body-element positions in the reviewed DOCX, including tables. “No 8-token match” means only that no consecutive sequence of eight normalized tokens occurred in the six downloaded PDFs. It does not certify originality against other sources or identify authorship.

| ID | Opening words | Direct source comparison | Close-reading note |
| --- | --- | --- | --- |
| P004 | Abstract—Pore-network graphs offer a compact representation for predicting permeability from micro-CT… | No 8-token match | Project-specific abstract or caption; compare quantitative claims with saved experiment results. |
| P005 | The workflow retains SNOW2 extraction, graph message passing and permeability regression.… | No 8-token match | Project-specific abstract or caption; compare quantitative claims with saved experiment results. |
| P006 | A separate frozen evaluation on ten crops from five previously unused… | No 8-token match | Clarity edit suggested for the abstract ending; no authorship conclusion. |
| P007 | Keywords—Graph attention networks, pore-network modeling, permeability, digital rock physics, SNOW2, critical… | No 8-token match | Project-specific abstract or caption; compare quantitative claims with saved experiment results. |
| P009 | Accurate prediction of absolute permeability in heterogeneous porous structures is important… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P010 | Direct numerical methods such as the lattice Boltzmann method (LBM) resolve… | No 8-token match | Seven-token overlap is conventional LBM terminology; not copied explanatory prose. |
| P011 | Data-driven models, including three-dimensional convolutional neural networks (CNNs), have been used… | No 8-token match | Short overlap is the conventional CNN name and abbreviation; image-permeability work is cited. |
| P012 | Graph representations provide another route by processing extracted pores and their… | No 8-token match | Optional direct statement of the comparison would improve clarity. |
| P013 | TopoFlow retains this image-to-network approach and uses graph attention for permeability… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P015 | 1) A source-separated comparison of GAT, GraphSAGE [4] and simpler permeability… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P016 | 2) A controlled ablation separating scale normalization from graph descriptors, followed… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P017 | 3) Component-level CPU timing and hydraulic-reference checks that distinguish learning error,… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P018 | Section II reviews related methods; Section III defines the comparison. Section… | No 8-token match | Standard paper roadmap; optional space saving, not an AI finding. |
| P020 | A. Physics Based Analytical Models: Traditional permeability estimates include the Kozeny–Carman… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P021 | Chapuis and Aubertin discuss the applicability of KC to soil hydraulic… | No 8-token match | Optional removal of defensive framing; KC source is cited. |
| P024 | where Φ is porosity, τ is tortuosity, c is a dimensionless… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P025 | B. Lattice Boltzmann Simulation: LBM advances discrete particle distributions on a… | No 8-token match | Add a source for the lattice-BGK background formulation. |
| P028 | Here fᵢ is the particle distribution for lattice direction i, cᵢ… | No 8-token match | Symbol definitions accompany the uncited lattice-BGK equation; add its source. |
| P029 | C. Image Based Machine Learning: Wu, Yin and Xiao predict permeability… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P030 | D. Graph Neural Networks: Pore-GNN predicts flow properties from micro-CT-derived graphs… | No 8-token match | Distinguish Pore-GNN CNN feature graphs from physical pore–throat networks. |
| P031 | Zhao and Xiao embed learned conductances in a differentiable PNM [9].… | No 8-token match | Background or literature summary; standard terminology and nearby attribution inspected. |
| P035 | The permeability benchmark compares a common log-permeability target under identical source… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P036 | The porosity KC baseline fits an intercept to log₁₀[Φ³/(1−Φ)²]. The scaled… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P037 | For critical-throat selection, the guided GAT is compared with a guided… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P038 | The analytical throat rankings use conductance proportional to d⁴/L and an… | No 8-token match | Keep analytical-control status explicit; provide traceable derivation for exact updates. |
| P041 | TopoFlow converts segmented micro-CT images to pore-network graphs and predicts absolute… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P043 | 1) Image input: Natural-rock binary volumes come from the Imperial College… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P044 | 2) Network extraction: SNOW2 partitions the void space using watershed regions… | No 8-token match | Original watershed extraction paper is cited; a software/version reference would further identify SNOW2. |
| P045 | 3) Graph prediction: The bulk encoder receives log pore diameter and… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P047 | Fig. 1. TopoFlow workflow. The PNM supplies training and evaluation targets… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P048 | C. Model Description: The bulk model uses graph attention [13]. Its… | No 8-token match | GAT source is cited; implementation-specific dimensions are reported. |
| P049 | The pooled GAT output is a scalar z used for standardized-target… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P051 | Each attention layer updates a pore embedding by a weighted neighborhood… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P054 | The coefficients α are softmax-normalized over the neighbors and self-loop of… | No 8-token match | Attention mechanism is attributed to GAT; no distinctive copied passage found. |
| P055 | For the scale-normalized variant, d₀ is the geometric-mean diameter of physical… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P059 | D. Mathematical Details: The hydraulic reference assumes steady, incompressible creeping flow.… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P062 | where ρ is density, v is characteristic velocity and Dₕ is… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P064 | Here Q is total volumetric flow rate, L is the physical… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P065 | Training uses mean squared error in y = log₁₀(k/m²), or its… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P067 | Predictive R² is calculated in log space using the mean reference… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P069 | Here N is the number of evaluated crops, yᵢ is reference… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P071 | 1) Input and separation: The development set contains 24 natural crops… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P072 | 2) Locked external test: After development selection, models are trained on… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P073 | 3) Feature assignment: The ten bulk graph descriptors comprise two porosity… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P074 | 4) Training: The neural benchmark uses seeds 11, 22, 33, 44… | No 8-token match | Project methods or mathematical definitions; conventional terminology and implementation details inspected. |
| P075 | 5) Restriction extension: A separate edge-aware model trains on eight development… | No 8-token match | Add rank-one-update provenance and a derivation reference for the restriction readout. |
| P079 | The evaluation uses natural-rock crops with parent-scan separation. Synthetic development examples… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P080 | Nested validation gives parent-balanced log-RMSE 0.8311 ± 0.0200 for the normalized… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P081 | The controlled GAT ablation yields parent-balanced log-RMSE 1.5528 for standard inputs,… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P082 | On the locked external test, the primary normalized GAT with descriptors… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P083 | Figure 2 summarizes external errors. Values for neural models are means… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P085 | Fig. 2. Locked external permeability comparison. Error bars show five-seed standard… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P087 | Serial CPU measurements use one Torch/BLAS thread. Bulk-model forward timing uses… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P089 | Fig. 3. Measured component costs in the development benchmark. Extraction is… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P093 | Partial throat narrowing leaves the original node-only GAT inputs unchanged, so… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P094 | The test uses S2, S4, S6, S7 and S8, with two… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P096 | Fig. 4. Actual loss from individual 50% restrictions of top-selected throats.… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P097 | Averaging first within crop and then equally across scans, guided-GAT top… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P098 | On the shared union of geometry-selected throats and three doses, 897… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P099 | Fresh sparse solves check 966 selected throat-dose interventions. Subsequent diagnostics reuse… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P101 | Sixteen independently assembled, identical-conductance OpenPNM checks agree with the sparse reference… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P102 | Full-volume SNOW calculations give 342.49 and 1187.48 mD for S1 and… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P103 | In the frozen restriction benchmark, one-thread warm CPU times average scan-wise… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P104 | Named scans are the grouping units; they are not independently certified… | No 8-token match | Experiment-specific result, caption or limitation; no copied passage established in the checked corpus. |
| P106 | Independent voxel-resolved flow or measured permeability on source-separated specimens is the… | No 8-token match | Future validation is correctly separated from completed results. |
| P107 | Broader source coverage would permit testing the exploratory geometry associations and… | No 8-token match | Potentially consolidate with adjacent future-scope paragraphs; preserve validation requirements. |
| P108 | Model development should address the measured calibration error and compare its… | No 8-token match | Potentially consolidate future scope; fresh validation remains a necessary condition. |
| P109 | These extensions retain the existing SNOW2-to-graph workflow while targeting its principal… | No 8-token match | Optional redundant closing sentence in future scope. |
| P111 | TopoFlow predicts PNM permeability from SNOW2-extracted pore networks using graph attention.… | No 8-token match | Concluding interpretation or future scope; preserve evidence limits. |
| P112 | The frozen restriction study finds useful geometry-driven throat ranking relative to… | No 8-token match | Concluding interpretation or future scope; preserve evidence limits. |

## Separately inspected material

| Elements | Treatment |
| --- | --- |
| P000, T001 | Title and author information read; not treated as AI or plagiarism indicators. |
| T033 | Method taxonomy read. Clarify CNN feature graphs versus physical pore networks in the GNN row. |
| T058 | Model feature assignments read; parameter names are conventional technical descriptions. |
| T091 | Quantitative result table read. Numerals and model labels are not prose-authorship evidence. |
| P023 | KC equation inspected; method citation appears in the surrounding text. |
| P027 | Lattice-BGK equation inspected; explicit source needed. |
| P050, P052, P053 | Pooling and attention equations inspected; GAT mechanism is attributed nearby. |
| P056 | Scale-reconstruction equation inspected; project normalization is described. |
| P061, P063 | Reynolds and Darcy equations inspected as standard background relations. |
| P066, P068 | MSE and R² definitions inspected; standard metric formulas are not distinctive prose. |
| P076 | Restriction-response formula inspected; rank-one-update provenance and derivation should be added. |
| P046, P084, P088, P095 | Figure provenance checked against the preceding revision script and experiment sources; image similarity was not tested. |
| P114–P128 | Fifteen references read. Exact bibliographic titles and author names are expected similarities and were excluded from automated prose comparisons. |
| Other short labels and empty section markers | Read for structure; not classified for authorship. |

Reviewed manuscript SHA-256: `0bf4d965714d28d44f8ec9fe7415812d17184467c01dabe49d51f6a521fd6b8f`.

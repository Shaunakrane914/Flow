# TopoFlow originality and attribution review

Reviewed 23 October 2026 against
[TopoFlow_IEEE_V4_Cited_Revised.docx](TopoFlow_IEEE_V4_Cited_Revised.docx).

No exact eight-word match was found between the manuscript's checked text and
the six downloaded source papers. This is a limited textual-overlap finding,
not a plagiarism clearance or a Turnitin similarity percentage. Two attribution
gaps and one related-work description need attention before submission.

The revised prose is AI-assisted: the assistant drafted substantial wording in
the preceding revision. No AI detector was run. An AI percentage or a claim of
human-only authorship cannot be inferred from this review.

## What was checked

| Check | Scope and result |
| --- | --- |
| Close reading | All manuscript text blocks, four tables including the author table, eleven native equations and the bibliography were inspected. |
| Direct wording comparison | Seventy non-bibliographic text blocks, including captions, contain 3,486 normalized word/number tokens. Each consecutive eight-token window was compared with extracted text from six source PDFs. No match was found. |
| Shorter wording overlaps | The longest observed overlap in the supplementary four-to-eight-token scan was seven tokens: “such as the lattice Boltzmann method LBM.” This is conventional terminology, not evidence of copied explanatory prose. |
| Public web queries | Twenty-four quoted-phrase queries sampled the abstract, introduction, background, methods, results and conclusion. Some queries returned unrelated or approximate results despite quotation marks; those results were not treated as matches. No copied passage was confirmed from the inspected results. |
| References and formulas | Bibliographic titles, author names, standard formulas and technical terms were inspected separately. They were excluded from the automated prose-overlap comparison. |
| Figures and tables | Captions and table wording were read. Figure provenance follows the preceding revision's generation script and experiment CSVs; no reverse-image or image-plagiarism service was run. |
| AI authorship | No detector result. Known drafting history establishes AI assistance, independently of writing style. |

Normalization used Unicode NFKC, lowercasing, punctuation tokenization and
dehyphenation at extracted PDF line breaks. This procedure checks exact sequences;
it does not detect semantic copying, translated copying, ideas without attribution,
private student submissions or every form of PDF extraction error. The token
count is not Microsoft Word's word count. Tables, formula symbols and reference
titles were not included in that count.

## Sources compared directly

| Source | Version obtained | PDF pages | Exact eight-token matches in checked manuscript text |
| --- | --- | ---: | ---: |
| [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | Author PDF from arXiv | 12 | 0 |
| [GraphSAGE](https://arxiv.org/abs/1706.02216) | Author PDF from arXiv | 19 | 0 |
| [Wu, Yin and Xiao on image-based permeability prediction](https://arxiv.org/abs/1809.02996) | Author PDF from arXiv | 17 | 0 |
| [I’Anson et al. on GNN permeability prediction](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/276560759/I_AnsonJ2025Using.pdf) | University repository PDF | 22 | 0 |
| [Zhao and Xiao on a GNN-embedded PNM](https://arxiv.org/abs/2509.13841) | Author PDF from arXiv; not asserted to be the final journal text | 37 | 0 |
| [OpenPNM](https://eprints.whiterose.ac.uk/id/eprint/101460/) | Accepted manuscript from White Rose | 36 | 0 |

These PDFs contain approximately 64,135 normalized tokens in total. Their
references, cover sheets and other extracted text were not removed, so this
is a corpus size rather than a count of research prose alone.

The SNOW paper's full PDF returned HTTP 403 and was not compared in full. Its
[publisher abstract](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.023307)
was inspected. The [Pore-GNN publisher abstract](https://www.sciopen.com/article/10.46690/ager.2023.10.05)
was also inspected. RTG-GNN full-text access was unavailable during this audit.
No source in these three categories is included in the six-PDF comparison count.

## Attribution findings

The paragraph identifiers below refer to body-element positions in the reviewed
DOCX. They provide stable locations for this file version, not Word page numbers.
The [paragraph audit](TopoFlow_Originality_Paragraph_Audit.md) records every
compared text block and the separately inspected tables, equations and references.

| Location | Finding | Specific action |
| --- | --- | --- |
| P025–P028, LBM background | The standard single-relaxation-time lattice-BGK equation is presented without a direct source. No copied wording was established, but equation provenance is incomplete. | Add a source for the lattice-BGK formulation, for example Qian, d’Humières and Lallemand, EPL 17, 479–484 (1992), [DOI](https://doi.org/10.1209/0295-5075/17/6/001). Keep the statement that these experiments did not generate LBM labels. |
| P075–P076, restriction response | The bounded readout uses a known rank-one network-update relationship. The present paper describes the readout without explaining this origin. The repository insert already acknowledges it. | Explain that exact hydraulic sensitivity and leverage give the finite-change response through a rank-one update of the linear network equations; learned estimates are substituted in the surrogate. Cite the underlying matrix-update method and provide a short derivation or a specific technical supplement. A Sherman–Morrison citation supports the algebra, not the entire permeability application by itself. |
| P030 and Table I | Pore-GNN is described broadly as predicting from micro-CT-derived graphs; the table groups GNN studies under “Pore network.” Pore-GNN's abstract actually describes CNN feature vectors embedded into graphs. | Distinguish CNN feature graphs from extracted physical pore–throat graphs. Suggested sentence: “Pore-GNN combines CNN-derived image features with graph models for permeability prediction [6].” Broaden Table I's representation entry to “Graph representations” or distinguish the approaches. This is a related-work precision issue, not a plagiarism finding. |
| P038, P103 | The affine-pressure proxy and exact-update control should remain clearly identified as analytical/reference methods. | Retain the existing distinction between geometry-only scores, flow-derived quantities and cached exact responses. Link any derivation to these definitions. Do not present the matrix identity as a new physical law. |

The [existing restriction-study insert](experiments/topoflow_discovery/MANUSCRIPT_INSERT.md)
already calls the finite-change expression a known rank-one identity. Carrying
that explanation into the Word manuscript would improve attribution without
changing the experiment or the paper's central idea.

Standard expressions for Darcy permeability, Reynolds number, MSE and R² are
not evidence of textual plagiarism. Their definitions, units and conditions
still need to be correct. Existing KC and GAT citations provide attribution for
those method families; renaming standard concepts to reduce text matching would
reduce clarity.

## Editorial findings rather than AI predictions

Several sentences can be more concrete, but none provides a valid word-level
AI diagnosis. Technical vocabulary, formal grammar and regular sentence length
cannot establish authorship.

| Location | Current wording | Editorial recommendation |
| --- | --- | --- |
| P006 | “delimit the benefit of attention under the stated PNM reference” | State the measured distinction: “The ranking advantage over random selection did not translate into better response accuracy.” |
| P012 | “Whether this representation improves accuracy or runtime must be established…” | State the experiment directly: “We compare prediction accuracy and runtime with analytical, descriptor-based and network-solver baselines.” |
| P021 | “rather than assuming that graph learning must outperform them” | End after identifying the two calibrated KC variants; the comparison already makes the rationale clear. |
| P018 | The standard section-by-section roadmap | Optional space saving. It is ordinary academic boilerplate, not an AI or plagiarism finding. |
| P106–P109 | Several future-scope paragraphs repeat the need for fresh validation | Consolidate the repeated points if the conference page limit requires it. Keep the substantive limitation about independent physical validation. |

These recommendations concern clarity. They are not a technique for predicting,
lowering or bypassing an AI detector score. Necessary scientific limitations and
standard terminology should remain.

## AI assistance and similarity are different questions

Turnitin explains that its AI indicator and similarity score are independent,
and that its AI model can misidentify both human and AI writing. The indicator
is not a conclusive judgment about misconduct.
[Turnitin's official AI Writing Report guide](https://guides.turnitin.com/hc/en-us/articles/22774058814093-Using-the-AI-Writing-Report).

For this manuscript, drafting history is known: AI assisted the revision. Follow
the actual conference's disclosure requirements and describe that assistance
accurately. A possible factual starting point is: “Generative AI assisted with
manuscript drafting and language revision.” Add statements that the authors
verified every claim, source and result only after the authors have completed
that review. This audit does not establish the conference's exact required
disclosure format or policy.

The project's own experiment reports and manuscript drafts are already present
in the repository. If a similarity service indexes them, it may report overlaps
with the authors' public material. Such matches need source-by-source review;
they are neither automatically third-party plagiarism nor automatically
irrelevant to a venue's prior-publication rules.

## Limits and recommended next steps

Address the two equation-attribution gaps and refine the Pore-GNN description.
Have all three authors read the final manuscript and verify its interpretation
of the experiments. A genuine institutional similarity check remains necessary
if the conference or college requires one. Its matched sources and exclusions
must be inspected; no acceptable percentage can be promised from this audit.

The Word manuscript was not changed during this review. No manuscript was
uploaded to a third-party AI/plagiarism checker. Selected quoted phrases were
sent to web search, and public source PDFs were downloaded for local comparison.

Reviewed manuscript SHA-256:
`0bf4d965714d28d44f8ec9fe7415812d17184467c01dabe49d51f6a521fd6b8f`.

The local audit folder `.paper_revision/originality/` contains the paragraph
inventory, query log, source hashes and exact-overlap results. Source PDFs and
raw search results are kept local rather than added to the repository.

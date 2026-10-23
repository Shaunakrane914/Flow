# TopoFlow formatting restoration - 23 October 2026

The previous proceedings conversion remained a Word document. This restoration
creates actual LaTeX source and a compiled PDF without rewriting the research.

## Deliverables

- [Restored PDF](../output/pdf/TopoFlow_ICISEA2026_Restored.pdf): 13 pages.
- [Editable source bundle](../output/pdf/TopoFlow_ICISEA2026_LaTeX.zip).
- [LaTeX source and build instructions](../paper/latex/README.md).
- [Verification record](../paper/latex/verification.json).

## Official template and layout

The [Atlantis Press policies page](https://www.atlantis-press.com/policies) links
the [official LaTeX package](https://assets.ctfassets.net/b4d0t67iwrjs/6HjuuxMB8tryYfLtWhcT3a/715489d26bdf68def035a1e24f67e2d0/LaTex_Package.zip).
The downloaded archive has SHA-256
`51a9a27a80d02ac45218abc83f465557ae96cc315adaebcaffe60de3fcdc7e9b`.
It supplies the SVPROC class, version 1.3. Its three required local class/style
files are included unchanged; their hashes appear in the verification record.

Native title and author commands replace the Word title-page construction.
The complete title wraps naturally into three balanced lines, with all four
authors on one line. Email links use serif type. Shaunak's corresponding-author
marker leads to the native first-page footnote; the other three addresses share
the compact contact block. Affiliations remain omitted as previously requested.

The preamble tightens title/author spacing, retains the class text dimensions,
uses a ragged-right abstract without forced word hyphenation, and controls
figure/table placement. Equations are native LaTeX mathematics. Headings and
references use SVPROC's hierarchy and bibliography environment.

## Content preservation

The Word source `TopoFlow_Atlantis_ICISEA2026.docx` is unchanged, SHA-256
`f9e3c1c178d6845e1e8dc44a3e5b7a97fb872c6fc1ca8d9f5aa29d072d40bf8a`.
The retained six-page IEEE Word copy is also unchanged, SHA-256
`d9b7f24f5f74370cf1491e4172ec5a9e750cba27643fb9a6c4b1002c4ce281cf`.

Verification compared 85 prose/heading blocks, the complete 182-word abstract,
all table cells, seven captions, 18 references, all four author names and email
addresses, and the corresponding-author designation. Text conversion was
round-tripped to compare words, numbers, citations and symbols. All 11 equations
were checked against independently transcribed formulas and their source OMML
hashes. The four image files are byte-identical to the Word images; embedded PDF
figure pixels also match the originals. Scientific claims, reported values and
the confirmed disclosure are preserved.

## Compilation and visual checks

Built with portable Tectonic 0.17.0; no system TeX installation was replaced.
Every page of the final 13-page PDF was rendered with Poppler and visually
inspected. The final log has no overfull boxes, missing characters, unresolved
references/citations or compilation errors. Five nonfatal underfull-box warnings
were visually checked. Text bounds remain inside the page; figures, equations,
tables and references are legible without clipping or overlaps.

The converter and preservation checker are in `scripts/restore_paper_latex.py`
and `scripts/verify_paper_latex.py`. The verification record includes hashes for
the Word source, LaTeX source, PDF and original template files. No experiments,
results or scientific prose were changed. No AI-detection percentage or plagiarism-screening result is
claimed. Conference-specific page limits and submission-stage requirements
remain unverified.

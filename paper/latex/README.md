# TopoFlow proceedings LaTeX source

This is a formatting-only conversion of `TopoFlow_Atlantis_ICISEA2026.docx`.
The original Word files remain unchanged. The compiled, visually inspected copy
is `../../output/pdf/TopoFlow_ICISEA2026_Restored.pdf` (13 pages).

## Build

Run from this directory with a current TeX distribution and the standard packages
listed in `topoflow.tex`:

```sh
xelatex -interaction=nonstopmode -halt-on-error topoflow.tex
xelatex -interaction=nonstopmode -halt-on-error topoflow.tex
```

Alternatively, use Tectonic, which resolves its TeX dependencies automatically:

```sh
tectonic --keep-logs topoflow.tex
```

The delivered PDF was built with Tectonic 0.17.0. Bibliography entries are embedded
in the source; no separate BibTeX step or bibliography database is required.
Different TeX/font versions can change line and page breaks; inspect a new build
before submission.

## Template provenance

The [official Atlantis Press policies page](https://www.atlantis-press.com/policies)
links the [LaTeX package](https://assets.ctfassets.net/b4d0t67iwrjs/6HjuuxMB8tryYfLtWhcT3a/715489d26bdf68def035a1e24f67e2d0/LaTex_Package.zip).
It supplies Springer SVPROC class version 1.3 (20 July 2016). The accompanying
`svproc.cls`, `aliascnt.sty` and `remreset.sty` are unmodified publisher-package
files. `TEMPLATE_README.txt` is the original package readme.

The preamble uses native title/author/section commands, serif email links,
natural title wrapping, compact title spacing, a ragged-right abstract and float
placement controls. Class margins and text dimensions are retained. Affiliations
remain omitted at the authors' request. Shaunak Rane is the corresponding author.
The contact block contains the other three addresses, and the corresponding
address appears in the first-page footnote.

## Preservation and verification

`verification.json` records the checked source and output hashes. All 85 prose
and heading blocks, the 182-word abstract, 11 equations, four figure images,
three data tables, all captions, 18 references and all author details were
compared with the Word source. Embedded PDF figure pixels match the originals.
All 13 rendered pages were inspected; no overfull boxes, missing characters or
unresolved citations remain. Five nonfatal underfull-box warnings were visually
checked.

For regeneration within the repository, run `python scripts/restore_paper_latex.py`
and `python scripts/verify_paper_latex.py` from the repository root. They require
`python-docx` and `lxml`. Regeneration asserts the retained Word source hash and
does not alter that source.

This follows the publisher-linked package; a conference-specific page limit and
the required submission-stage format have not been independently confirmed.

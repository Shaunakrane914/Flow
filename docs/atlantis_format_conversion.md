# TopoFlow Atlantis proceedings conversion

Completed 23 October 2026. The separate
[proceedings manuscript](../TopoFlow_Atlantis_ICISEA2026.docx) follows the Word
template linked by the [official Atlantis Press author resources](https://www.atlantis-press.com/policies).
It renders to **14 pages** in Microsoft Word. Every final page was inspected.
The [six-page IEEE copy](../TopoFlow_Physics_Guided_Graph_Attention_ICISEA2026.docx)
is retained unchanged.

The official [Word template download](https://assets.ctfassets.net/b4d0t67iwrjs/61D4gQKQaJuqyRIB8i5h8h/0e1b497360ae6c6d0e04fdf5d776a9e2/Word_Template.zip)
contains SPLNPROC, a Springer proceedings template. This specific publisher-linked
package controls the conversion. No unrelated conference template was substituted.
The package includes a macro-enabled sample; its macros were not executed.
The final DOCX contains no macros or template sample text.

| Element | Proceedings copy |
| --- | --- |
| Layout | One A4 portrait section and one column, with the template's exact margins and header/footer positions |
| Title | Centered bold Times New Roman, 14 points, using a Word Title style based on the template's papertitle style |
| Authors | All four names in the existing order; centered author and email components |
| Correspondence | Shaunak Rane, explicitly confirmed by the user; `shaunak.rane@universalai.in` |
| Affiliations | Omitted as previously requested; no affiliation or ORCID invented |
| Abstract | Existing 182-word abstract merged into the template's single abstract paragraph with bold run-in label |
| Body | Times New Roman, 10 points; template Normal and first-paragraph styles |
| Sections | Arabic-numbered sections 1–7 and numbered subsections; prose section/table cross-references updated |
| Equations | Eleven native Word math blocks, unchanged, centered with template equation spacing |
| Figures | Four unchanged image blobs, resized proportionately and kept with their captions |
| Tables | Three data tables with identical cells, proportional widths, automatic row heights and template-style horizontal rules |
| References | Eighteen sources in the same first-citation order; numbered entries, surname/initials, year in parentheses and existing DOI/URL identifiers |
| Interests | User-confirmed no-competing-interests statement before References |
| Running heads | Short title and “S. Rane et al.” with native PAGE fields; blank first-page header |

The downloaded template uses “Abstract.” as a bold run-in label. The conversion
follows that actual component rather than introducing a separate standalone
heading based on a general description of Springer layouts. Bibliographic
punctuation and author-name order change to match the template; titles, DOI/arXiv
identifiers, years, journal metadata and access dates are retained.

The conversion does not change experiments, measured outcomes or conclusions.
The added corresponding-author designation and interests statement were confirmed
by the user. No author-contribution statement, acceptance claim or AI-detection
score was introduced. The preceding originality comparison remains a limited
check of the retained IEEE manuscript, not a new similarity score for this file.

Verification compared substantive prose after the documented heading and
cross-reference changes, all table cells, exclusive canonical XML for all eleven
equations, image hashes, reference titles and DOI/arXiv identifiers. The section
properties match the official reference exactly. Numbering definitions, theme,
font table, blank footers, endnotes, web settings and custom XML parts were
preserved byte-for-byte. Existing template styles were retained; Normal's implicit
10-point size was made explicit and a Title alias added. The reference's macros,
sample chart, sample footnote, editing protection and publisher placeholders were
removed from the working copy.

The packaged render and render-diff tools were attempted and reported that
LibreOffice `soffice.exe` is absent. Installed Word COM and bundled Poppler
provided the reference and final page renders. All three reference pages and all
14 final pages were inspected. QA PDFs, PNGs and template downloads remain local.

Reproduce with [scripts/format_atlantis_paper.py](../scripts/format_atlantis_paper.py)
using the managed document Python runtime. If the reference cache is absent, the
script downloads the exact official package, verifies its hash, and removes the
macro parts without executing them. It verifies the derived reference by a
content fingerprint independent of ZIP timestamps. A changed input manuscript
or publisher package fails the checks rather than being silently reformatted.

Template ZIP SHA-256:
`db03865ce8a5c654c4deb8bdaacb8b6ff76f37fe3391e9c32e4815d7edcf8c24`.

Retained IEEE input SHA-256:
`d9b7f24f5f74370cf1491e4172ec5a9e750cba27643fb9a6c4b1002c4ce281cf`.

Proceedings DOCX SHA-256:
`f9e3c1c178d6845e1e8dc44a3e5b7a97fb872c6fc1ca8d9f5aa29d072d40bf8a`.

The conference-specific page limit, initial-submission requirements and any
additional required metadata are not verified. Following the publisher's linked
template does not establish compliance with those venue-specific conditions.

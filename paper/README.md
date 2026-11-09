# Research Manuscript

This folder contains the revised IEEEtran manuscript and the material required to compile it.

## Files

- `main.tex` — 12-page IEEEtran two-column manuscript source.
- `references.bib` — BibTeX references cited by `main.tex`.
- `images/` — five figures referenced by the manuscript.
- `make_figures.py` — script used to generate the manuscript figures from the reported summary values.
- `research_paper_12pages.pdf` — compiled PDF included for convenient review.

## Build

Compile from this folder using pdfLaTeX and BibTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

On Overleaf, upload the files in this folder as a project, keeping `main.tex`, `references.bib`, and `images/` at the project root. Select pdfLaTeX as the compiler.

## Reporting notes

The 99th-percentile benign-calibration threshold is the primary result. The 97.5th-percentile threshold is reported as exploratory because the unknown-test file was inspected during earlier threshold exploration. The manuscript explicitly states that the code does not reproduce the full FedGAN-ID system: GAN training, federated aggregation, differential privacy, CAN message graphs, MTF, and READ are not implemented.

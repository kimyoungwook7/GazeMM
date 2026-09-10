# GazeMM — project page

Project page for **GazeMM: Robust Off-Angle Iris Recognition via Gaze Manifold Matching**
(Min, Kim, Ryu, Lee — ETRI).

Static HTML, no build step. Deployed with GitHub Pages.

```
.
├── index.html            # the project page
├── static/
│   ├── css/style.css
│   ├── images/           # figures rendered from the Overleaf vector sources
│   └── pdfs/             # supplementary PDF (the paper points readers here)
└── .nojekyll             # tell GitHub Pages to serve files as-is
```

## Deploy

```bash
git remote add origin https://github.com/<account>/GazeMM.git
git push -u origin main
```

Then **Settings → Pages → Source: `main` / `/ (root)`**.
The site appears at `https://<account>.github.io/GazeMM/` after ~1 minute.

To preview locally:

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Before pushing

- [ ] **Decide whether to publish `static/pdfs/GazeMM_paper.pdf`.** It is the submitted
      manuscript. ICASSP does not review double-blind and permits preprints, so this is
      normally fine — but delete the file and its button in `index.html` if you would
      rather wait for the decision.
- [ ] Confirm the venue line in `index.html` (`<div class="venue">`) — currently
      "Submitted to ICASSP 2027".
- [ ] Check the code repo link. It currently points at
      `github.com/kimyoungwook7/GAZEMM`, matching the paper.
- [ ] Update the BibTeX entry once the paper is accepted.

## Supplementary

`static/pdfs/GazeMM_supplementary.pdf` — 6 pages, built from
`Template_supplemental_v0.2.tex` in the Overleaf project. The PDF is the only form;
there is deliberately no HTML version to keep in sync.

Coverage of the five items the paper promises the project page will host:

| Paper says | Where it lives |
| --- | --- |
| Dataset statistics | § A, Table S1 + Figs. S1–S2 |
| Baseline configurations | § B, per-method |
| Training settings | § B, proposed method |
| Hyperparameter analyses | § C, Tables S9 ($k$) and S10 ($c$) |
| Extended results | § C, Tables S2–S8 |

### Rebuilding the PDF

The published PDF is built from a patched copy that neutralises the co-authors' red
revision marks — the body text is untouched, only the colour definitions are overridden:

```latex
\usepackage[table]{xcolor}
\definecolor{red}{rgb}{0,0,0}        % <- added
\definecolor{orange}{rgb}{0,0,0}     % <- added
\definecolor{darkgreen}{rgb}{0,0,0}  % <- added
```

Then `pdflatex` → `bibtex` → `pdflatex` ×2, and copy the result over
`static/pdfs/GazeMM_supplementary.pdf`.

When the `.tex` changes, rebuild the PDF as above — that is the only step.

## Hosting large files

GitHub Pages caps a repo at ~1 GB and individual files at 100 MB, with 100 GB/month of
bandwidth. Keep checkpoints, extracted features and video off this repo — put them on
Hugging Face Hub or Zenodo (which also mints a DOI) and link out.

## Credits

Layout follows the [Nerfies](https://github.com/nerfies/nerfies.github.io) project page
convention, released under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

# GazeMM: Robust Off-Angle Iris Recognition via Gaze Manifold Matching

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://kimyoungwook7.github.io/GazeMM/)
[![Supplementary](https://img.shields.io/badge/Supplementary-PDF-orange)](https://kimyoungwook7.github.io/GazeMM/static/pdfs/GazeMM_supplementary.pdf)
[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#citation)
<!-- TODO: add a license badge once the LICENSE file is chosen, e.g.
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) -->

[Yewon Min](#)\*, [Youngwook Kim](#)\*, [Moonwook Ryu](#), [Jun Seong Lee](#)†<br>
Electronics and Telecommunications Research Institute (ETRI)<br>
Daejeon, Republic of Korea<br>
\* Equal contribution &nbsp;&nbsp; † Corresponding author

Submitted to ICASSP 2027.

## Abstract

Recent deep learning-based iris recognition methods encode iris images as high-dimensional
embeddings and typically authenticate a query by directly comparing it with enrollment embeddings
using point-to-point matching. Under off-angle acquisition, however, gaze-induced appearance
changes introduce substantial intra-class variation, separating embeddings of the same identity and
degrading recognition performance.

To address this challenge, we propose **GazeMM**, an identity-specific gaze manifold-based iris
matching framework that explicitly models such gaze-induced variation. Given multi-view enrollment
embeddings, GazeMM constructs a low-dimensional affine subspace for each identity and matches a
query by orthogonal residual distance, followed by cohort-based score normalization. Experimental
results show that GazeMM achieves an average equal error rate of **0.30%** under off-angle
conditions, while maintaining competitive performance under frontal acquisition. Moreover, GazeMM
remains effective across different iris feature extractors, demonstrating the generality of the
proposed matching strategy.

## Installation

Tested on Windows 11 with Python 3.10.11, PyTorch 2.11.0 (CUDA 13.0) and an NVIDIA RTX 5080.
A CUDA-capable GPU is required.

```bash
git clone https://github.com/kimyoungwook7/GAZEMM.git
cd GAZEMM
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The datasets are not redistributed; please obtain them from their owners. Place each dataset under
`data/<DATASET>/` with one folder per subject and one sub-folder per eye:

```
data/<DATASET>/<subject>/<L|R>/<image>
```

Identities are defined per eye, so the left and right eyes of a subject are different classes.
The twelve datasets used in the paper and their folder names are listed at the top of `main.py`
(`DATA_1` … `DATA_12`): eight frontal datasets (CASIA v1.0, CASIA-Iris-Interval,
CASIA-Iris-Thousand, CASIA-Iris-Lamp, PolyU Iris DB, MMU-v1, and the Cogent and Vista subsets of
the IIITD Contact Lens Iris DB) and four off-angle datasets (EyeMovement from AI-Hub, OpenEDS,
CASIA-Iris-Degradation, and CASIA-Iris-Africa). Datasets whose folder is missing are skipped.

## Usage

All settings are constants at the top of `main.py`. Edit them if needed, then run

```bash
python main.py
```

| Constant | Meaning | Paper setting |
|---|---|---|
| `LOAD_DATASETS` | datasets to evaluate | all twelve |
| `CV_FOLDS`, `CV_REPEATS`, `CV_SEED` | subject-disjoint k-fold cross-validation, repeated with different shuffles | 2-fold, 3 repeats, seed 42 |
| `MANIFOLD_COMPONENTS` | dimension of each identity's affine subspace | 3 |
| `MANIFOLD_MIN_IMGS` | minimum number of enrollment images per identity for subspace modeling | 3 |
| `FOCUS_GATE_C` | quality filtering coefficient `c` (single value for all datasets) | 2.0 |

For every dataset the script prints the per-fold and averaged FRR@FAR=10⁻³, EER and Rank-1
(mean ± std over folds) together with the FTE, and ends with a summary table over all datasets.
Training hyperparameters (150 epochs with early stopping
on validation EER, batch-hard sampling with 16 identities × 4 images, Adagrad with learning rate
0.075) are defined in `argument_settings()` in `modules/GazeMM/irisRecognitionTrain.py`.

## Repository structure

```
main.py                              experiment driver: datasets, cross-validation, summary table
modules/
  utils.py                           data listing, subject-disjoint splits, EER / FRR@FAR / Rank-1
  GazeMM/
    irisRecognitionTrain.py          triplet training, gaze manifold matching, score normalization
    models/resnet.py                 ResNet34 embedding network
    models/utils_resnet.py           ResNet34 definition (from torchvision)
requirements.txt
```

## Reproducibility

Random seeds are fixed per fold and deterministic CUDA kernels are enabled. Results may still vary
slightly across hardware and software versions.

## Citation

The paper is currently under review. A citation entry will be provided here upon acceptance.

<!-- TODO: replace with the final BibTeX after acceptance
```bibtex
@inproceedings{min2027gazemm,
  title     = {GazeMM: Robust Off-Angle Iris Recognition via Gaze Manifold Matching},
  author    = {Min, Yewon and Kim, Youngwook and Ryu, Moonwook and Lee, Jun Seong},
  booktitle = {ICASSP},
  year      = {2027}
}
```
-->

## Acknowledgements

The ResNet34 definition in `modules/GazeMM/models/` is reduced from
[tamerthamoqa/facenet-pytorch-glint360k](https://github.com/tamerthamoqa/facenet-pytorch-glint360k)
(commit `38fc241a`), which in turn derives from [torchvision](https://github.com/pytorch/vision);
the ImageNet weights are the torchvision release.

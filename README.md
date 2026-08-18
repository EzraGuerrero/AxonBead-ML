# AxonBead-ML

Deep-learning extension of [AxonBead](https://github.com/EzraGuerrero/AxonBead), a published image
analysis tool that quantifies axonal beads (a biomarker of axonal damage) in confocal microscopy
images of neurons.

**Status:** 🚧 Step 7 — Streamlit demo, public launch.

## Try me

**Live App:** https://axonbead-ml.streamlit.app/
**Live API:** https://axonbead-ml.onrender.com/docs

## Why this project exists

AxonBead's bead-detection step relies on a manually tuned intensity threshold, which has to be
re-checked by eye for every new imaging batch. This project replaces that step with a model trained
to detect beads directly from the raw image — removing the manual tuning step while keeping the
same validated output metric (beads per 1000 µm² of neurite area) that AxonBead already reports.

The classical AxonBead pipeline is used here as both a dependency (for shared I/O utilities) and as
the baseline that any ML model has to match or beat.

## Project structure

```
axonbead-ml/
├── data/
│   ├── raw/          # original, untouched images (DVC-tracked, not committed to git)
│   ├── interim/       # intermediate outputs (candidate crops, cached arrays)
│   └── processed/     # final training-ready datasets
├── notebooks/          # exploratory analysis, documented with markdown cells
├── src/
│   └── axonbead_ml/      # the installable package (`pip install -e .`)
│       ├── data/          # dataset loading and preprocessing code
│       ├── annotation/    # the labeling tool
│       ├── models/        # model architectures
│       ├── training/      # training loops, MLflow logging
│       └── api/           # FastAPI serving code (added in week 7)
├── streamlit_app/        # public demo app (added in week 8)
├── docker/               # containerization files (added in week 7)
├── docs/
│   ├── data_card.md      # dataset documentation
│   ├── model_card.md     # model documentation (added once a model exists)
│   └── experiments/      # notes on individual training runs
├── tests/
├── pyproject.toml
└── CHANGELOG.md
```

## Relationship to AxonBead

This repo does not fork or modify [AxonBead](https://github.com/EzraGuerrero/AxonBead) — that repo
stays stable and citable as-is, since it's tied to a published method (Guerrero Gonzalez et al.,
2025, *European Journal of Neuroscience*). Instead, AxonBead is installed here as a dependency and
used as the classical-method baseline for validation.

## Tech stack

Python 3.10+, PyTorch, scikit-image, DVC (data versioning), MLflow (experiment tracking), FastAPI
(model serving), Docker (containerization), Streamlit (public demo).

## Roadmap

| Steps | Milestone |
|---|---|
| 1 | Repo scaffolding, data collection |
| 2 | Annotation tool, annotation begins |
| 3 | Annotation complete, baseline model + MLflow tracking |
| 4–5 | U-Net bead detector trained |
| 6 | FastAPI + Docker |
| 7 | Streamlit demo, public launch |
| 8–12 (stretch) | Neurite segmentation model |

## License

MIT — see [LICENSE](LICENSE)

## Author

Ezra Guerrero González, PhD

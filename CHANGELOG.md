# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Step 7 - Streamlit demo app + public deployment

### Added
- `streamlit_app/app.py` — public-facing demo: example image gallery (preview all three
  conditions before choosing), file upload (.czi/.tif/.png/.jpg), detection overlay
  (predicted bead locations drawn on the image), bead-count interpretation bands, model
  statistics sidebar, and a documented "next step" note on neurofilament-area normalization.
- `streamlit_app/requirements.txt` — deliberately lightweight (streamlit, requests, pillow,
  matplotlib, numpy only) — the app is a thin client and needs neither torch nor bioio,
  since all model inference happens via calls to the deployed API.
- `GET /examples/{name}/image` endpoint added to the FastAPI app, so Streamlit can display
  and overlay bundled example images without duplicating them locally.

### Fix
- Resolved an out-of-memory (OOM) crash on Render's free tier (512MB limit), which caused
  `/detect` to fail for `control` and `high_beads` (worked intermittently for `low_beads`
  only). Root cause: `pip install torch` pulls the default PyPI wheel, which bundles CUDA
  libraries loaded into memory at import time even on a CPU-only container. Fixed by
  installing the CPU-only torch build explicitly (`--index-url
  https://download.pytorch.org/whl/cpu`) and setting `torch.set_num_threads(1)`.

### Deployed
- API live on Render: [https://axonbead-ml.onrender.com/docs](https://axonbead-ml.onrender.com/docs)
- Streamlit app live on Streamlit Community Cloud: [https://axonbead-ml.streamlit.app/](https://axonbead-ml.streamlit.app/)

### Documentation
- `README.md` updated with both live URLs.
---

## Step 6 - FastAPI serving endpoint + Docker

### Added
- `src/axonbead_ml/inference.py` — single shared preprocess → predict → peak-find pipeline,
  used by both the API and (going forward) any future evaluation code, so served predictions
  can't silently diverge from validated notebook results.
- `src/axonbead_ml/api/image_loading.py` — loads `.czi`, `.tif`/`.tiff`, `.png`, `.jpg` for
  inference; non-`.czi` formats are assumed single-channel grayscale (auto-converted from RGB
  if needed, since they carry no channel metadata).
- `src/axonbead_ml/api/main.py` — FastAPI app with `GET /health`, `GET /examples`, and
  `POST /detect` (accepts an uploaded image or a bundled example by name).
- `scripts/export_sample_images.py` — one-time script exporting one representative image per
  condition (control/low_beads/high_beads) as PNGs bundled into the API for `/examples`.
- `docker/Dockerfile`, `.dockerignore` — containerizes the API (Python 3.11-slim base, model
  checkpoint baked into the image, `uvicorn` as the entrypoint).
- Render deployment — created Render web service linked to Github repo. Render builds image
  from GitHub repo, so `checkpoints/unet_best.pt` needed to be added by modifying .gitignore
  `!checkpoints/unet_best.pt`

### Infrastructure notes
- Docker Desktop on Windows repeatedly failed (WSL2 backend errors, a reinstall attempt that
  crashed the machine and required a backup restore). Resolved by installing Docker Engine
  directly inside a Debian VM instead of using Docker Desktop.
- First VM build ran out of disk space mid-build; resolved by increasing the VM's allocated
  disk — Docker's layer caching meant the rebuild reused already-completed layers rather than
  starting over.

### Verified
- `/detect` tested against all three bundled examples; bead counts (control: 3, low_beads: 10,
  high_beads: 18) are consistent with the EDA's per-condition means, both run locally via
  `uvicorn` and inside the built Docker container.
---

## Step 4–5 — U-Net bead detector

### Added
- `src/axonbead_ml/data/loading.py` — shared annotation/condition-loading logic, extracted
  from the EDA notebook once a second notebook needed the same code.
- `src/axonbead_ml/data/heatmap.py` — converts point annotations to Gaussian heatmap targets.
- `src/axonbead_ml/data/splits.py` — fixed, seeded, condition-stratified train/val/test split
  (70/15/15), saved to `data/processed/splits.csv`.
- `src/axonbead_ml/data/dataset.py` — PyTorch `Dataset` loading images + heatmaps on demand.
- `src/axonbead_ml/models/unet.py` — small U-Net (16 base channels, 3 downsampling levels),
  sized deliberately small for a 14-image training set.
- `src/axonbead_ml/training/predict.py` — converts predicted heatmaps to point coordinates via
  local peak-finding.
- `notebooks/04_unet_data_prep.ipynb` — creates the split, visually sanity-checks heatmap
  generation (overlay on raw image).
- `notebooks/05_unet_training.ipynb` — full training loop, loss curves, MLflow logging.

### Results
- Trained 60 epochs, tuned `peak_threshold` (final: 0.25) against the validation set.
- Final held-out **test set** evaluation: F1 0.628 (vs. classical baseline F1 0.341 on the
  full dataset) — logged to MLflow as `unet_v1_test_final`.
- Known weakness: recall on `low_beads` condition is lower than other conditions, likely due
  to MSE loss under-penalizing missed faint/dim beads. Weighted loss noted as future work,
  not implemented in this version.

### Known issues
- Checkpoint files are overwritten by a fixed filename (`unet_best.pt`) rather than versioned
  per run — to be fixed before any future retraining, to avoid ambiguity about which weights
  are loaded.

### Documentation
- `docs/model_card.md` updated to v0.2 (U-Net), with baseline vs. U-Net comparison table and
  limitations.
---

## Step 3 — EDA, classical baseline, MLflow

### Added
- `notebooks/02_eda.ipynb` — bead count distributions by condition, outlier check, visual
  spot-check of annotations overlaid on a raw image.
- `src/axonbead_ml/models/baseline.py` — classical detector (manual intensity threshold = 220,
  8-bit; connected-component + circularity filtering), matching AxonBead's own approach.
- `src/axonbead_ml/training/evaluate.py` — precision/recall/F1 via optimal (Hungarian) point
  matching; reused for every model evaluated afterward.
- `notebooks/03_baseline_model.ipynb` — runs the baseline across all 60 images, logs to MLflow.
- `docs/model_card.md` created (v0.1, classical baseline).

### Fixed
- MLflow tracking URI pinned explicitly to the project root (was defaulting to the notebook's
  working directory, `notebooks/mlruns/`, causing runs to appear "missing" from the UI).
- Migrated MLflow backend from the deprecated filesystem store to SQLite
  (`sqlite:///mlflow.db`), per MLflow's own deprecation notice.

### Results
- Classical baseline (all 60 images): precision 0.237, recall 0.607, F1 0.341.
- Data card updated with real annotation counts (68 control / 250 low_beads / 448 high_beads
  bead clicks; see `docs/experiments/eda_summary_stats.csv`).
---

## Step 2 — Annotation tooling

### Added
- `notebooks/01_bead_annotation.ipynb` — napari-based interactive annotation walkthrough.
- `src/axonbead_ml/annotation/annotate.py` — batch annotation script: loops a folder of
  `.czi` images, skips already-annotated ones, merges results into one combined CSV.
- Confirmed SMI-31 channel index (C=1) by inspecting real file metadata.

### Changed
- Switched image loading from `aicsimageio` (discontinued) to `bioio` (its actively
  maintained successor), after `aicsimageio` broke against a newer `aicspylibczi` release.

### Results
- All 60 images annotated (20 control / 20 low_beads / 20 high_beads); combined into
  `data/interim/all_annotations.csv`.
---

## Step 1 — Repo scaffolding, environment, data collection

### Added
- Initial repo scaffold: `data/`, `notebooks/`, `src/axonbead_ml/` installable package layout,
  `docs/`, `tests/`, `streamlit_app/`, `docker/`.
- `README.md`, `docs/data_card.md` (Datasheets for Datasets template), `pyproject.toml`,
  `.gitignore`, `LICENSE` (MIT).
- Conda environment (Python 3.11, matching AxonBead's requirement).
- Git repo initialized and pushed to GitHub; DVC initialized for data versioning.

### Fixed
- Resolved a git/DVC tracking conflict (`data/raw` needed to be untracked from git via
  `git rm -r --cached` before DVC could take it over).
- Recovered empty folder structure lost during a clone mixup, via `.gitkeep` placeholders.

### Data
- Gathered and organized 60 raw `.czi` images (20 control / 20 low_beads / 20 high_beads)
  across multiple experimental rounds and imaging sessions; added to DVC tracking.
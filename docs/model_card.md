# Model Card — AxonBead-ML Bead Detector

## 1. Model details
- **Name:** U-Net bead detector (v0.2), served via a public API + Streamlit demo
- **Version:** v0.2 (model weights unchanged since training; this update reflects deployment)
- **Date:** 2026-08-16
- **Type:** Deep learning (U-Net, PyTorch), served behind a FastAPI inference API
- **Owner:** Ezra Guerrero González

## 2. Intended use
- **Purpose:** Detect axonal bead locations in SMI-31 confocal images, replacing manual
  intensity thresholding. Publicly demoed via a Streamlit app for portfolio/showcase purposes.
- **Out of scope:** Not validated for other markers, magnifications, or imaging modalities.
  Not intended for clinical or diagnostic use — bead-count interpretation shown in the demo
  app uses heuristic thresholds derived from this project's own dataset, not an established
  clinical standard.

## 3. Architecture / how it works
**Current model (v0.2):** Small U-Net (16 base channels, 3 downsampling levels) trained to
predict a Gaussian heatmap of bead locations from the raw SMI-31 image. Predicted heatmap
peaks (threshold 0.25, min distance 5px) are converted to point coordinates and matched
against ground truth via Hungarian assignment. See `src/axonbead_ml/models/unet.py`. See
`src/axonbead_ml/models/unet.py` and `src/axonbead_ml/inference.py` (the same inference function
used both for evaluation and by the live API, so served predictions can't silently diverge from
validated results).

**Baseline (v0.1, still in repo for comparison):** Manual intensity threshold (220, 8-bit) +
connected-component shape filtering. See `src/axonbead_ml/models/baseline.py`.

## 4. Training data
See `docs/data_card.md` — 60 annotated confocal images (20 control / 20 low_beads /
20 high_beads), split 14 train / 9 val / 9 test, stratified by condition, fixed seed.

## 5. Evaluation data & metrics
Evaluated on a held-out test set (9 images, never used for training or threshold tuning),
using the same precision/recall/F1 methodology (Hungarian point-matching, max distance 15px)
for both models — see `src/axonbead_ml/training/evaluate.py`.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| v0.1 Classical baseline | 0.237 | 0.607 | 0.341 |
| v0.2 U-Net | 0.639 | 0.617 | 0.628 |

U-Net trained for 60 epochs; `peak_threshold=0.25` selected via validation-set tuning before
this final, single test-set evaluation.

## 6. Limitations
- Recall on `low_beads` condition was the model's weakest point during validation-set tuning
  (~0.38 at initial evaluation) — likely due to MSE loss under-penalizing missed faint/dim
  beads. A foreground-weighted loss is planned future work, not yet implemented.
- Trained on only 14 images — small dataset increases variance in reported metrics; the
  val/test sets (~9 images, ~3 per condition) are thin enough that individual-image results
  should be treated with caution alongside the aggregate numbers.
- Checkpoint files are currently overwritten by a fixed filename (`unet_best.pt`) rather than
  versioned by run — to be fixed before any future retraining, to avoid ambiguity about which
  weights are loaded/deployed.
- Bead-count interpretation bands (used in the demo app) are heuristics derived from this
  project's own EDA, not clinically validated cutoffs.

## 7. Ethical considerations
Not intended for clinical or diagnostic use; research and portfolio demonstration tool only.

## 8. Deployment
- **Inference API:** FastAPI + Docker, deployed on Render. Endpoints: `GET /health`,
  `GET /examples`, `GET /examples/{name}/image`, `POST /detect`. Live at: _[https://axonbead-ml.onrender.com/docs]_
- **Demo UI:** Streamlit app, deployed on Streamlit Community Cloud — thin client calling
  the API above (no model code duplicated in the UI layer). Live at: _[https://axonbead-ml.streamlit.app/]_
- **Known operational constraints:**
  - Render's free tier spins down after inactivity; first request after idle time can take
    up to ~60 seconds (cold start).
  - Free tier has a 512MB memory limit. The default `pip install torch` wheel bundles CUDA
    libraries that get loaded into memory even on this CPU-only deployment, causing an OOM
    crash under real use. Fixed by installing the CPU-only torch build explicitly.

### Deployment log
| Date | Change |
|---|---|
| 2026-08-17 | API deployed to Render; `/detect` initially failed for `control`/`high_beads` due to OOM |
| 2026-08-18 | Fixed via CPU-only torch wheel + `torch.set_num_threads(1)`; all conditions verified working |
| 2026-08-18 | Streamlit app deployed, connected to live API |

## 8. Version history
| Version | Date | Model | F1 (overall) | Notes |
|---|---|---|---|---|
| v0.1 | 2026-08-14 | Classical baseline | 0.341 | First recorded baseline via MLflow |
| v0.2 | 2026-08-16 | U-Net (60 epochs) | 0.628 | Beats baseline; low_beads recall gap noted as future work |
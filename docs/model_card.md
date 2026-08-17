# Model Card — AxonBead-ML Bead Detector

## 1. Model details
- **Name:** U-Net bead detector (v0.2)
- **Version:** v0.2
- **Date:** 2026-08-16
- **Type:** Deep learning (U-Net, PyTorch) — supersedes the v0.1 classical baseline
- **Owner:** Ezra Guerrero González

## 2. Intended use
- **Purpose:** Detect axonal bead locations in SMI-31 confocal images, as a baseline reference
  for the deep-learning detector under development.
- **Out of scope:** Not validated for other markers, magnifications, or imaging modalities.

## 3. Architecture / how it works
**Current model (v0.2):** Small U-Net (16 base channels, 3 downsampling levels) trained to
predict a Gaussian heatmap of bead locations from the raw SMI-31 image. Predicted heatmap
peaks (threshold 0.25, min distance 5px) are converted to point coordinates and matched
against ground truth via Hungarian assignment. See `src/axonbead_ml/models/unet.py`.

- Note: model served via a FastAPI + Docker API (/detect, /examples, /health), with the manual
sigma/peak_threshold/image_size values used at inference matching the validated notebook config
exactly (via the shared inference.py).

**Baseline (v0.1, still in repo for comparison):** Manual intensity threshold (220, 8-bit) +
connected-component shape filtering. See `src/axonbead_ml/models/baseline.py`.

## 4. Training data
Not applicable (rule-based, not trained). Evaluated against the annotated dataset described
in `docs/data_card.md`.

## 5. Evaluation data & metrics
Evaluated on the same held-out test set (9 images, stratified by condition), never used for
training or threshold tuning.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| v0.1 Classical baseline | 0.237 | 0.607 | 0.341 |
| v0.2 U-Net | 0.639 | 0.617 | 0.628 |

## 6. Limitations
- Recall on low_beads condition is weaker than other conditions (~0.38 at initial evaluation) —
  likely due to MSE loss under-penalizing missed faint/dim beads; a foreground-weighted loss is
  a planned future improvement, not yet implemented.
- Trained on only 14 images — small dataset increases variance in reported metrics.
- Checkpoint files are overwritten by filename on each training run rather than versioned by
  epoch count; if retraining, checkpoint naming should be fixed first to avoid ambiguity about
  which weights are loaded.

## 7. Ethical considerations
Not intended for clinical or diagnostic use; research tool only.

## 8. Version history
| Version | Date | Model | F1 (overall) | Notes |
|---|---|---|---|---|
| v0.1 | 2026-08-14 | Classical baseline | 0.341 | First recorded baseline via MLflow |
| v0.2 | 2026-08-16 | U-Net (60 epochs) | 0.628 | Beats baseline; low_beads recall gap noted as future work |
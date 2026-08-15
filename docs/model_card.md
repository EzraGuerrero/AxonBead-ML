# Model Card — AxonBead-ML Bead Detector

## 1. Model details
- **Name:** Classical baseline (manual threshold + shape filtering)
- **Version:** v0.1
- **Date:** 2026-08-XX
- **Type:** Rule-based / classical image processing (not a trained model)
- **Owner:** Ezra Guerrero González

## 2. Intended use
- **Purpose:** Detect axonal bead locations in SMI-31 confocal images, as a baseline reference
  for the deep-learning detector under development.
- **Out of scope:** Not validated for other markers, magnifications, or imaging modalities.

## 3. Architecture / how it works
Manual intensity threshold (220, 8-bit) isolates high-intensity bead structures against
neurite background, followed by connected-component labeling and filtering by area
(5–200 px) and circularity (≥0.6). See `src/axonbead_ml/models/baseline.py`.

## 4. Training data
Not applicable (rule-based, not trained). Evaluated against the annotated dataset described
in `docs/data_card.md`.

## 5. Evaluation data & metrics
- **Evaluated on:** All 60 annotated images (20 control / 20 low_beads / 20 high_beads).
- **Metric:** Precision/recall/F1 via optimal point-matching (Hungarian assignment,
  max distance 15 px) — see `src/axonbead_ml/training/evaluate.py`.
- **Results:** Overall precision 0.237, recall 0.607, F1 0.341.
  By condition: control F1 0.165, low_beads F1 0.316, high_beads F1 0.455.

## 6. Limitations
- Low precision — many false positives, likely background/neurite structures crossing the
  intensity threshold without being filtered out by shape alone.
- Not tested on images from other experiments, magnifications, or staining protocols.

## 7. Ethical considerations
Not intended for clinical or diagnostic use; research tool only.

## 8. Version history
| Version | Date | Model | F1 (overall) | Notes |
|---|---|---|---|---|
| v0.1 | 2026-08-XX | Classical baseline | 0.341 | First recorded baseline via MLflow |
# Data Card — AxonBead-ML Training Set

_This follows the "Datasheets for Datasets" convention (Gebru et al., 2018) — a standard way of
documenting ML datasets so anyone (including future-you) can understand what's in them, how they
were collected, and what their limitations are. Fill this in as data collection progresses._

## 1. Overview

- **Purpose:** Training/validation data for a bead-detection model, intended to replace manual
  intensity thresholding in the AxonBead pipeline.
- **Modality:** Confocal microscopy images, SMI-31 (neurofilament) channel, `.czi` format.
- **Status:** _TBD — fill in once collection is complete._

## 2. Source

- **Origin:** iPSC-derived neurons exposed to varying glutamate concentrations (see AxonBead
  publication for full experimental protocol).
- **Imaging equipment / settings:** SMI-31 (488 nm) neurofilament channel, 20x magnification.
- **Collection date(s):** Multiple experimental rounds/imaging sessions — see composition table.

## 3. Composition

| Condition | # images | Notes |
|---|---|---|
| Control (non-treated / control) | 20 | Negative examples — teaches the model what "not a bead" looks like |
| Low_beads (low-dose glutamate) | 20 | |
| High_beads (high-dose glutamate) | 20 | |

- **Batches/imaging sessions represented:** Multiple experimental rounds and imaging sessions
  (encoded in filename via experiment ID, e.g. NI240119).
- **Class balance (once annotated):** _TBD — filled in after Week 2 annotation._

## 4. Collection process

- **Selection criteria:** _How were these specific images chosen from the larger experiment?_
- **Who collected/imaged them:** _TBD._

## 5. Known limitations / biases

- _e.g., all images from a single cell line / donor? Single imaging instrument? Any known
  artifacts (uneven illumination, bleed-through) to flag?_

## 6. Ethical considerations

- iPSC line source and any relevant consent/ethics approvals (reference the original publication's
  methods section).

## 7. Annotation protocol

_To be completed in Week 2 — will describe how bead locations were labeled, by whom, and any
quality-control steps (e.g., second-pass review, inter-rater agreement)._

## 8. Version history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-08-11 | Initial raw data collected: 60 images total (20 control / 20 low_beads / 20 high_beads),
 spanning experiments NI231117 and NI240119 (first 5 images of NT, HCl as control, Pio, LiCl, CHIR as low_beads, 
Glut and DMSO as high_beads) |

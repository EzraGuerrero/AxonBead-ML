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
- **Imaging equipment / settings:** _TBD._
- **Collection date(s):** _TBD._

## 3. Composition

| Condition | # images | Notes |
|---|---|---|
| Control (no glutamate) | _TBD_ | Needed as negative examples — teaches the model what "not a bead" looks like |
| Low-dose glutamate | _TBD_ | |
| High-dose glutamate | _TBD_ | |

- **Batches/imaging sessions represented:** _TBD — note if data spans more than one session, which
  matters for how well the model generalizes._
- **Class balance (once annotated):** _TBD — record the ratio of bead : non-bead candidate regions._

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
| 0.1 | _TBD_ | Initial raw data collected |

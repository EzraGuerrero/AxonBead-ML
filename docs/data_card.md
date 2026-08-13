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

| Condition | n_images | total_beads | mean_beads_per_image | std_beads_per_image |
|---|---|---|---|---|
| Control | 20 | 68 | 3.4 | 2.5 |
| High_beads | 20 | 448 | 22.4 | 6.5 |
| Low_beads | 20 | 250 | 12.5 | 3.9 |

- **Batches/imaging sessions represented:** Multiple experimental rounds and imaging sessions
  (encoded in filename via experiment ID, e.g. NI240119).
- **Class balance (once annotated):** control < low_beads < high_beads
- **Outliers:** NI240119_SMI31-488_20x_HCl_03.czi is a known outlier in the "control" condition. Annotation is correct

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

True beads were labelled by Dr. Ezra Guerrero González, considering an user-determined size threshold, intensity value,
and whether they appear connected to neurites on both sides.

## 8. Version history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-08-11 | Initial raw data collected: 60 images total (20 control / 20 low_beads / 20 high_beads),
 spanning experiments NI231117 and NI240119 (first 5 images of NT, HCl as control, Pio, LiCl, CHIR as low_beads, 
Glut and DMSO as high_beads) |

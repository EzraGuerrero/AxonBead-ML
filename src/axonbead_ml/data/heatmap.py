"""
Convert (y, x) bead-center annotations into Gaussian heatmap targets.

A U-Net predicts an image-shaped output, so point annotations need to become
an image-shaped target: a small Gaussian "blob" at each bead's location.
During evaluation (not here — that happens in training/predict.py, added
next), we'll find local peaks in the *predicted* heatmap to recover
coordinates again, and score them with the same match_points function
used for the classical baseline.
"""

import numpy as np


def points_to_heatmap(
    points: np.ndarray, image_shape: tuple, sigma: float = 4.0
) -> np.ndarray:
    """Build a Gaussian heatmap from (y, x) points.

    sigma: controls how large/spread-out each blob is, in pixels. Roughly,
    a bead's true radius is a reasonable starting point — too small and the
    network gets almost no gradient signal near misses during training;
    too large and nearby beads blur into one blob.
    """
    heatmap = np.zeros(image_shape, dtype=np.float32)
    if len(points) == 0:
        return heatmap

    yy, xx = np.meshgrid(
        np.arange(image_shape[0]), np.arange(image_shape[1]), indexing="ij"
    )

    for y, x in points:
        gaussian = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
        # take the max, not the sum, so overlapping blobs from nearby beads
        # don't produce an artificially bright combined peak
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap
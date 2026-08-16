"""
Convert a predicted heatmap into discrete bead coordinates.

The U-Net outputs a heatmap, not a list of points — this finds local
maxima in that heatmap and treats each one as a predicted bead location,
so the result can be scored with the exact same match_points function
used for the classical baseline.
"""

import numpy as np
from skimage.feature import peak_local_max


def heatmap_to_points(
    heatmap: np.ndarray, threshold: float = 0.3, min_distance: int = 5
) -> np.ndarray:
    """Find local peaks in a predicted heatmap. Returns (N, 2) array of (y, x).

    threshold: minimum heatmap value (0-1) to count as a real detection —
        below this is "no bead here." Lower it if the model is missing
        real beads (recall too low); raise it if it's hallucinating too
        many (precision too low).
    min_distance: minimum pixel distance required between two separate
        peaks, so one blurry blob doesn't get counted as several beads.
    """
    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return coords  # already (row, col) == (y, x), same convention as annotations
"""
Classical baseline bead detector.

This exists so the eventual deep-learning model has something concrete to
beat, measured on your own annotated ground truth — not just "the model
seemed to work." It's a self-contained threshold + connected-components
detector, independent of AxonBead's internal code, for the same reason the
annotation tool avoided depending on unverified internal functions: it's
transparent, easy to reason about, and won't break if AxonBead's internals
change.

This mirrors the *idea* behind AxonBead's own classical bead detection
(intensity threshold + size/shape filtering) without depending on its
exact implementation.
"""

import numpy as np
from skimage import filters, measure


def detect_beads(
    image: np.ndarray,
    threshold_method: str = "manual",
    manual_threshold: float = 220,
    min_area: int = 25,
    max_area: int = 200,
    min_circularity: float = 0.6,
) -> np.ndarray:
    """Detect bead-like blobs in a 2D image, return an (N, 2) array of (y, x) centroids.

    Parameters
    ----------
    threshold_method: "manual", "otsu" or "triangle" — manual thesholding and 
        two standard automatic thresholding methods from scikit-image. Otsu
        assumes a roughly bimodal intensity histogram (background vs. foreground);
        triangle tends to work better when foreground pixels are a small minority,
        which is often the case for sparse bead-like structures.
    min_area, max_area: pixel-area bounds, filters out noise (too small)
        and large clumps/artifacts (too big).
    min_circularity: 4*pi*area / perimeter^2, ranges 0-1 where 1 is a
        perfect circle. Filters out elongated/irregular blobs that don't
        look bead-like.
    """
    if threshold_method == "manual":
        threshold = int(manual_threshold)
    elif threshold_method == "otsu":
        threshold = filters.threshold_otsu(image)
    elif threshold_method == "triangle":
        threshold = filters.threshold_triangle(image)
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    binary = image > threshold
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    centroids = []
    for region in regions:
        if not (min_area <= region.area <= max_area):
            continue
        if region.perimeter == 0:
            continue
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        if circularity < min_circularity:
            continue
        centroids.append(region.centroid)  # (row, col) == (y, x)

    return np.array(centroids) if centroids else np.empty((0, 2))
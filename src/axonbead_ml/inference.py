"""
End-to-end inference: raw 2D image in, predicted bead (y, x) coordinates out.

This is deliberately a single shared function rather than duplicated logic
in the training notebook and the API — the exact resize/normalize/predict/
peak-find sequence has to match precisely between validated evaluation
results and what the live API actually does, or the API's behavior would
silently diverge from the reported F1 score.
"""

import numpy as np
import torch
from skimage.transform import resize

from axonbead_ml.training.predict import heatmap_to_points


def predict_bead_locations(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512,
    peak_threshold: float = 0.25,
    min_peak_distance: int = 5,
) -> np.ndarray:
    """Run the full pipeline on one 2D image, return (N, 2) array of (y, x)
    bead coordinates in the ORIGINAL image's resolution.

    image: 2D array, any resolution, values in [0, 255] (8-bit).
    """
    original_shape = image.shape
    scale = image_size / original_shape[0]  # assumes square images, matching training

    image_normalized = image.astype(np.float32) / 255.0
    image_resized = resize(image_normalized, (image_size, image_size), anti_aliasing=True)

    image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_heatmap = model(image_tensor).cpu().numpy()[0, 0]

    points_resized_scale = heatmap_to_points(
        predicted_heatmap, threshold=peak_threshold, min_distance=min_peak_distance
    )

    if len(points_resized_scale) == 0:
        return np.empty((0, 2))

    return points_resized_scale / scale

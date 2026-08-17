"""
Load an image for bead detection, regardless of source format.

.czi files carry channel metadata, so we know exactly which channel is
SMI-31. Plain image formats (PNG, TIFF, JPG) don't carry that information,
so we assume single-channel grayscale input for those, converting from RGB
automatically if needed. Document this assumption clearly wherever the API
is described — it's a silent behavior difference between formats otherwise.
"""

from pathlib import Path

import numpy as np
from skimage import color, io

SMI31_CHANNEL_INDEX = 1


def load_image_for_inference(path: Path) -> np.ndarray:
    """Return a 2D uint8/float array ready for predict_bead_locations."""
    suffix = path.suffix.lower()

    if suffix == ".czi":
        from bioio import BioImage

        img = BioImage(str(path))
        return img.get_image_data("YX", C=SMI31_CHANNEL_INDEX, T=0, Z=0)

    if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        image = io.imread(str(path))
        if image.ndim == 3:
            # RGB (or RGBA) — collapse to grayscale rather than silently
            # picking one channel, since we don't know which one might
            # correspond to a meaningful signal for a non-.czi upload
            image = color.rgb2gray(image[..., :3]) * 255.0
        return image.astype(np.float32)

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported: .czi, .tif, .tiff, .png, .jpg, .jpeg"
    )

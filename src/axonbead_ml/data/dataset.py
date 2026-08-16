"""
PyTorch Dataset for bead detection: loads a raw image and builds its
Gaussian heatmap target on the fly.

Images are resized down from their native 1024x1024 to `image_size`
(default 512x512). This is purely a practical/tractability choice for a
14-image training set on a CPU or modest GPU — full-resolution training is
worth revisiting later once the pipeline works, not before.
"""

from pathlib import Path

import numpy as np
import torch
from bioio import BioImage
from skimage.transform import resize
from torch.utils.data import Dataset

from axonbead_ml.data.heatmap import points_to_heatmap

SMI31_CHANNEL_INDEX = 1


class BeadDataset(Dataset):
    def __init__(
        self,
        split_df,
        annotations_df,
        split: str,
        image_size: int = 512,
        sigma: float = 4.0,
    ):
        """
        split_df: DataFrame from splits.py, with columns image/condition/path/split.
        annotations_df: DataFrame from loading.py, with columns image/y/x/condition.
        split: "train", "val", or "test" — filters split_df to this subset.
        image_size: images and heatmaps are resized to (image_size, image_size).
        sigma: passed through to points_to_heatmap; scaled automatically to
            account for the resize (so blob size looks consistent regardless
            of image_size).
        """
        self.rows = split_df[split_df["split"] == split].reset_index(drop=True)
        self.annotations = annotations_df
        self.image_size = image_size
        self.sigma = sigma

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        image_path = Path(row["path"])

        img = BioImage(str(image_path))
        image = img.get_image_data("YX", C=SMI31_CHANNEL_INDEX, T=0, Z=0).astype(np.float32)
        original_shape = image.shape

        # normalize 8-bit intensities to [0, 1] — standard for feeding into a
        # network; keeps gradients well-scaled regardless of input bit depth
        image = image / 255.0

        points = self.annotations.loc[
            self.annotations["image"] == row["image"], ["y", "x"]
        ].values

        scale = self.image_size / original_shape[0]  # assumes square images
        image_resized = resize(image, (self.image_size, self.image_size), anti_aliasing=True)
        points_resized = points * scale
        sigma_resized = self.sigma * scale

        heatmap = points_to_heatmap(
            points_resized, (self.image_size, self.image_size), sigma=sigma_resized
        )

        # add a channel dimension: (H, W) -> (1, H, W), which is what
        # PyTorch's 2D conv layers expect (channels-first)
        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)

        return image_tensor, heatmap_tensor, row["image"]
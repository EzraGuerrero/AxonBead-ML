"""
Train/validation/test split, stratified by condition, at the image level.

Splitting must happen per-image, not per-pixel-patch: if patches from the
same image ended up in both train and validation, validation performance
would be inflated by the model having effectively seen that image's
specific noise/background pattern already. Stratifying by condition
ensures each split has a representative mix of control/low_beads/high_beads,
rather than e.g. validation accidentally landing mostly on one condition.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make_split(
    images: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each image to train/val/test, stratified by condition.

    images: DataFrame with at least an "image" and "condition" column
        (as returned by list_all_images_with_condition).
    seed: fixed for reproducibility — re-running this produces the exact
        same split every time, which matters for fair comparison between
        the baseline, this U-Net, and any future model.
    """
    rng = np.random.default_rng(seed)
    assigned = []

    for condition, group in images.groupby("condition"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        split_labels = (
            ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        )
        for idx, label in zip(indices, split_labels):
            assigned.append({"index": idx, "split": label})

    split_df = pd.DataFrame(assigned).set_index("index")
    result = images.copy()
    result["split"] = split_df["split"]
    return result


def save_split(split_df: pd.DataFrame, path: Path) -> None:
    split_df.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
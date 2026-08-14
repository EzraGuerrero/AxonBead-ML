"""
Shared utilities for loading annotations and mapping images to conditions.

Extracted from the EDA notebook (02_eda.ipynb) once the baseline model
notebook needed the exact same logic — see the project README for the rule
of thumb: code stays in a notebook until a second notebook needs it too.
"""

from pathlib import Path

import pandas as pd


def build_filename_to_condition(raw_dir: Path) -> dict:
    """Map each image filename to its condition, based on its parent folder.

    Uses folder location rather than parsing the filename text, since
    filenames use inconsistent condition tokens across experiments
    (e.g. "NT" vs "HCl" for what is organized as the same "control" folder).
    """
    return {p.name: p.parent.name for p in raw_dir.rglob("*.czi")}


def load_annotations_with_condition(annotations_path: Path, raw_dir: Path) -> pd.DataFrame:
    """Load the combined bead-click annotations and attach each row's condition."""
    annotations = pd.read_csv(annotations_path)
    filename_to_condition = build_filename_to_condition(raw_dir)
    annotations["condition"] = annotations["image"].map(filename_to_condition)

    missing = annotations[annotations["condition"].isna()]
    if len(missing):
        raise ValueError(
            f"{missing['image'].nunique()} annotated images had no matching file "
            f"in {raw_dir} — check for renamed/moved files."
        )
    return annotations


def list_all_images_with_condition(raw_dir: Path) -> pd.DataFrame:
    """One row per .czi file found under raw_dir, with its condition and full path."""
    rows = [
        {"image": p.name, "condition": p.parent.name, "path": p}
        for p in raw_dir.rglob("*.czi")
    ]
    return pd.DataFrame(rows)


def counts_per_image(annotations: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Bead count per image, including images with zero annotated beads.

    A plain groupby on the annotations table silently drops zero-bead images
    (there's no row to group), so this explicitly merges against the full
    image list to add them back in.
    """
    counts = (
        annotations.groupby(["image", "condition"]).size().reset_index(name="bead_count")
    )
    all_images = list_all_images_with_condition(raw_dir)[["image", "condition"]]
    counts = all_images.merge(counts, on=["image", "condition"], how="left")
    counts["bead_count"] = counts["bead_count"].fillna(0).astype(int)
    return counts
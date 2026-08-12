 
"""
Batch bead-center annotation tool.

Loops through every .czi image in a folder, opens each one in napari for
you to click bead centers, and saves one CSV per image to
data/interim/annotations/. Skips images that already have an annotation
file, so you can stop mid-way and resume later without redoing work.

Usage (from the project root, with the `annotation` extra installed):

    python src/axonbead_ml/annotation/annotate.py --input-dir data/raw

Optionally merge everything into one master file when you're done:

    python src/axonbead_ml/annotation/annotate.py --input-dir data/raw --combine

Note: run this directly with `python path/to/annotate.py`, not
`python -m axonbead_ml.annotation.annotate` — running by path sidesteps an
import-discovery issue some editable-install setups hit, and works exactly
the same for a self-contained script like this one.

Controls inside napari, for each image:
    - Make sure the "annotations" Points layer is selected (left panel).
    - Click directly on a bead to add a point at that location.
    - Select the point-selection tool (arrow icon), click a point, then
      press Backspace/Delete to remove a mistaken click.
    - Close the napari window when you've marked every bead in that image —
      this saves your annotations AND automatically opens the next image.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Confirmed by inspecting img.channel_names in the walkthrough notebook:
# SMI-31 is channel index 1 for these files (Ch2-T2).
SMI31_CHANNEL_INDEX = 1


def load_smi31_channel(image_path: Path) -> np.ndarray:
    """Load the SMI-31 (neurofilament) channel from a .czi file as a 2D array."""
    from bioio import BioImage

    img = BioImage(str(image_path))
    return img.get_image_data("YX", C=SMI31_CHANNEL_INDEX, T=0, Z=0)


def annotate_image(image_path: Path, output_dir: Path) -> Path:
    """Open napari for one image, collect clicked points, save to CSV."""
    import napari

    print(f"\nLoading {image_path.name} ...")
    image = load_smi31_channel(image_path)

    viewer = napari.Viewer(title=f"Annotating: {image_path.name}")
    viewer.add_image(image, name="SMI-31", colormap="green")
    points_layer = viewer.add_points(
        name="annotations",
        ndim=2,
        size=10,
        face_color="red",
        border_color="white",
    )
    points_layer.mode = 'add'

    print("Click on each bead to mark its center. Close this window when done.")
    napari.run()  # blocks until this image's viewer window is closed

    coords = points_layer.data
    df = pd.DataFrame(coords, columns=["y", "x"])
    df.insert(0, "image", image_path.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_annotations.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} bead annotations to {out_path}")
    return out_path


def annotate_folder(input_dir: Path, output_dir: Path) -> None:
    """Annotate every .czi under input_dir, skipping already-annotated ones."""
    image_paths = sorted(input_dir.rglob("*.czi"))
    if not image_paths:
        print(f"No .czi files found under {input_dir}")
        return

    remaining = [
        p for p in image_paths
        if not (output_dir / f"{p.stem}_annotations.csv").exists()
    ]
    already_done = len(image_paths) - len(remaining)
    print(
        f"Found {len(image_paths)} images total — "
        f"{already_done} already annotated, {len(remaining)} remaining."
    )

    for i, image_path in enumerate(remaining, start=1):
        print(f"\n=== Image {i} of {len(remaining)} ===")
        annotate_image(image_path, output_dir)

    print("\nAll images in this folder are annotated.")


def combine_annotations(output_dir: Path, combined_path: Path) -> None:
    """Merge every per-image annotation CSV into one master file."""
    csv_files = sorted(output_dir.glob("*_annotations.csv"))
    if not csv_files:
        print(f"No annotation CSVs found in {output_dir}")
        return
    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"Combined {len(csv_files)} files ({len(combined)} total beads) into {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate bead centers across a folder of .czi images."
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Folder containing .czi images (searched recursively, e.g. data/raw)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/annotations"),
        help="Folder to save per-image annotation CSVs into",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="After annotating, merge all per-image CSVs into one master file",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input folder not found at {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    annotate_folder(args.input_dir, args.output_dir)

    if args.combine:
        combine_annotations(args.output_dir, args.output_dir.parent / "all_annotations.csv")


if __name__ == "__main__":
    main()
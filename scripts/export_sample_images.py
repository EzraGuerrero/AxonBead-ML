"""
Export one representative image per condition as a PNG, for the API's
bundled /examples feature.

Run once, from the project root:
    python scripts/export_sample_images.py

Picks the first image found in each condition folder — edit
SPECIFIC_IMAGES below if you'd rather hand-pick particular files (e.g. ones
you know look clean/representative rather than whatever sorts first).
"""

from pathlib import Path

from bioio import BioImage
from PIL import Image

SMI31_CHANNEL_INDEX = 1
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("src/axonbead_ml/api/sample_images")

# Set specific filenames here if you want to hand-pick which image
# represents each condition, e.g.:
# SPECIFIC_IMAGES = {"control": "NI240119_SMI31-488_20x_NT_03.czi", ...}

SPECIFIC_IMAGES = {
    "control": "NI231117_SMI31-488_20x_HCl_01.czi",
    "low_beads": "NI240119_SMI31-488_20x_Pio_01.czi",
    "high_beads": "NI231117_SMI31-488_20x_DMSOx1_01.czi"
}

def export_condition(condition: str) -> None:
    condition_dir = RAW_DIR / condition
    if condition in SPECIFIC_IMAGES:
        image_path = condition_dir / SPECIFIC_IMAGES[condition]
    else:
        candidates = sorted(condition_dir.glob("*.czi"))
        if not candidates:
            print(f"WARNING: no .czi files found in {condition_dir}, skipping.")
            return
        image_path = candidates[0]

    img = BioImage(str(image_path))
    array = img.get_image_data("YX", C=SMI31_CHANNEL_INDEX, T=0, Z=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{condition}.png"
    Image.fromarray(array.astype("uint8")).save(out_path)
    print(f"{condition}: exported {image_path.name} -> {out_path}")


if __name__ == "__main__":
    for condition in ["control", "low_beads", "high_beads"]:
        export_condition(condition)

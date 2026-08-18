"""
FastAPI service for the AxonBead-ML bead detector.

Run locally with:
    uvicorn axonbead_ml.api.main:app --reload --app-dir src

Then open http://localhost:8000/docs for interactive API documentation —
FastAPI generates this automatically from the type hints and Pydantic
models below, including a "Try it out" button that lets you upload a real
test image through the browser without writing any client code.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from axonbead_ml.api.image_loading import load_image_for_inference
from axonbead_ml.inference import predict_bead_locations
from axonbead_ml.models.unet import SmallUNet

app = FastAPI(
    title="AxonBead-ML API",
    description="Detects axonal beads in confocal microscopy images using a trained U-Net.",
    version="0.2.0",
)

CHECKPOINT_PATH = Path("checkpoints/unet_best.pt")
EXAMPLES_DIR = Path(__file__).parent / "sample_images"
IMAGE_SIZE = 512
PEAK_THRESHOLD = 0.25
MIN_PEAK_DISTANCE = 5

EXAMPLES = {
    "control": "Control condition — minimal axonal damage, few beads expected.",
    "low_beads": "Low-dose glutamate — moderate axonal damage.",
    "high_beads": "High-dose glutamate — extensive axonal damage, many beads expected.",
}

# Loaded once at startup, not per-request — reloading a model from disk on
# every API call would make each request far slower than necessary.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=1, out_channels=1, base_channels=16).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Limit PyTorch's thread count to reduce peak memory on small CPU instances
torch.set_num_threads(1)

class DetectionResponse(BaseModel):
    source: str
    n_beads: int
    points: List[List[float]]  # each inner list is [y, x] in original image pixel coordinates


class ExampleInfo(BaseModel):
    name: str
    description: str


@app.get("/health")
def health() -> dict:
    """Cheap liveness check — deployment platforms poll this to confirm the service is up."""
    return {"status": "ok"}


@app.get("/examples", response_model=List[ExampleInfo])
def list_examples() -> List[ExampleInfo]:
    """List the bundled sample images available for /detect, for users without their own file."""
    return [ExampleInfo(name=name, description=desc) for name, desc in EXAMPLES.items()]


@app.get("/examples/{name}/image")
def get_example_image(name: str) -> FileResponse:
    """Return the raw PNG for one bundled example, so a client (e.g. Streamlit) can
    display it and draw a detection overlay, without needing its own copy of the file."""
    if name not in EXAMPLES:
        raise HTTPException(404, f"Unknown example '{name}'. Options: {list(EXAMPLES)}")
    image_path = EXAMPLES_DIR / f"{name}.png"
    if not image_path.exists():
        raise HTTPException(500, f"Example image file missing on server: {image_path}")
    return FileResponse(image_path, media_type="image/png")


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: Optional[UploadFile] = File(default=None),
    example: Optional[str] = None,
) -> DetectionResponse:
    """Detect bead locations in an uploaded image, or a bundled example.

    Provide exactly one of:
    - `file`: an uploaded .czi, .tif/.tiff, .png, or .jpg image
    - `example`: one of "control", "low_beads", "high_beads" (see /examples)
    """
    if file is not None and example is not None:
        raise HTTPException(400, "Provide either `file` or `example`, not both.")
    if file is None and example is None:
        raise HTTPException(400, "Provide either `file` or `example`.")

    if example is not None:
        if example not in EXAMPLES:
            raise HTTPException(404, f"Unknown example '{example}'. Options: {list(EXAMPLES)}")
        image_path = EXAMPLES_DIR / f"{example}.png"
        source_name = f"example:{example}"
        image = load_image_for_inference(image_path)
    else:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            image = load_image_for_inference(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        source_name = file.filename

    points = predict_bead_locations(
        image,
        model,
        device,
        image_size=IMAGE_SIZE,
        peak_threshold=PEAK_THRESHOLD,
        min_peak_distance=MIN_PEAK_DISTANCE,
    )

    return DetectionResponse(
        source=source_name,
        n_beads=len(points),
        points=points.tolist(),
    )
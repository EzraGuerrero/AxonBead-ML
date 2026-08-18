# Base image: a minimal Python 3.11 install (matches AxonBead's requirement).
# "slim" means most non-essential OS packages are left out, keeping the image
# smaller than the full python:3.11 image.
FROM python:3.11-slim

WORKDIR /app

# System-level libraries some image libraries (scikit-image, Pillow) need
# under the hood, plus git — needed because pyproject.toml installs AxonBead
# directly from GitHub, which requires git to clone during the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what the API actually needs — not the whole repo. Data,
# notebooks, and docs would just make the image larger for no benefit
# (enforced by .dockerignore alongside this).
COPY pyproject.toml .
COPY src/ src/
COPY checkpoints/unet_best.pt checkpoints/unet_best.pt

# Install the project. Plain "pip install ." (not "-e .[dev]") — this is a
# production image, so no editable install and no dev-only tools
# (jupyter, pytest, napari) that the API doesn't need.

# Install CPU-only torch explicitly first — the default PyPI wheel bundles
# CUDA libraries that get loaded into memory even though this container has
# no GPU, which is a common cause of OOM on small deployments.

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir .

EXPOSE 8000

# The command that runs when the container starts. --host 0.0.0.0 is
# required (not "localhost") so the server accepts connections from outside
# the container, not just from within it.
CMD ["uvicorn", "axonbead_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]

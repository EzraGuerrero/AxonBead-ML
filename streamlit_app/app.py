"""
AxonBead-ML Streamlit demo.

A thin client — all detection happens via calls to the deployed FastAPI service.
No model, no torch, no bioio here: this app only needs to send images and
display results, which is why its requirements.txt is so much lighter than
the API's.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

# TODO: replace with your actual Render URL once confirmed
API_BASE_URL = "https://axonbead-ml.onrender.com"

st.set_page_config(page_title="AxonBead-ML", layout="wide")

st.title("AxonBead-ML: Axonal Bead Detector")
st.markdown(
    "Detects axonal beads in confocal microscopy images (SMI-31 neurofilament channel) "
    "using a U-Net trained to replace manual intensity thresholding. Upload your own "
    "image or try a bundled example below."
)

# --- Sidebar: model info ---
with st.sidebar:
    st.header("About this model")
    st.markdown(
        """
**Architecture:** Small U-Net (16 base channels), trained on 60 annotated confocal images
(14 train / 9 val / 9 test, stratified by condition).

**Held-out test set performance:**

| Metric | Classical baseline | U-Net |
|---|---|---|
| Precision | 0.237 | — |
| Recall | 0.607 | — |
| F1 | 0.341 | **0.628** |

**Known limitation:** recall on low-bead-density images is the model's current weak point
(~0.38 at initial evaluation) — likely because the training loss under-penalizes missed
faint beads. A foreground-weighted loss is planned future work.

**Planned next step:** normalize bead counts to neurofilament-positive area
(beads / 1000 µm²), matching the original AxonBead publication's metric, so results are
comparable across images with differing amounts of visible neurite — currently this app
shows raw counts only.

[Source code & full documentation](https://github.com/EzraGuerrero/AxonBead-ML)
        """
    )
    st.caption("Research tool — not for clinical or diagnostic use.")

st.divider()


# --- Helpers ---
@st.cache_data(show_spinner=False)
def fetch_examples():
    response = requests.get(f"{API_BASE_URL}/examples")
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def fetch_example_image(name: str) -> bytes:
    response = requests.get(f"{API_BASE_URL}/examples/{name}/image")
    response.raise_for_status()
    return response.content


def interpret_bead_count(n_beads: int):
    """Heuristic bands derived from this project's own EDA — the midpoints between
    condition means (control ~3.4, low_beads ~12.5, high_beads ~22.4 beads/image).
    NOT a clinically validated cutoff; specific to this dataset and imaging setup."""
    if n_beads < 8:
        return (
            "Low bead count",
            "Neurites appear largely healthy — no strong signs of early axonal damage.",
            "green",
        )
    elif n_beads < 17:
        return (
            "Moderate bead count",
            "Possible early signs of axonal damage. Bead formation can indicate disrupted "
            "axonal transport — an early step that may progress toward axonal transection "
            "and eventual retrograde neuronal degeneration if damage continues.",
            "orange",
        )
    else:
        return (
            "High bead count",
            "Strong signs of axonal damage. This level of beading is consistent with "
            "significant disruption of axonal transport, which can lead to axonal "
            "transection and retrograde neuronal degeneration and death.",
            "red",
        )


def draw_overlay(image: np.ndarray, points: list):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    if points:
        points_arr = np.array(points)
        ax.scatter(
            points_arr[:, 1], points_arr[:, 0],
            s=60, facecolors="none", edgecolors="red", linewidths=1.5,
        )
    ax.axis("off")
    return fig


# --- Step 1: choose an image ---
st.subheader("1. Choose an image")

examples = fetch_examples()
mode = st.radio("Image source", ["Use a bundled example", "Upload my own image"], horizontal=True)

selected_example = None
uploaded_file = None
image_bytes_for_display = None

if mode == "Use a bundled example":
    st.caption("Preview all three conditions below, then pick one to analyze.")
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            img_bytes = fetch_example_image(example["name"])
            st.image(img_bytes, caption=example["name"], use_container_width=True)
            st.caption(example["description"])

    selected_example = st.selectbox("Select example to analyze", [e["name"] for e in examples])
    image_bytes_for_display = fetch_example_image(selected_example)
else:
    uploaded_file = st.file_uploader(
        "Upload a .czi, .tif/.tiff, .png, or .jpg image",
        type=["czi", "tif", "tiff", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        image_bytes_for_display = uploaded_file.getvalue()

st.divider()

# --- Step 2: run detection ---
st.subheader("2. Run detection")

can_run = uploaded_file is not None or selected_example is not None
if st.button("Detect beads", type="primary", disabled=not can_run):
    with st.spinner(
        "Running detection... (if the API has been idle, this can take up to a "
        "minute while it wakes back up)"
    ):
        if mode == "Use a bundled example":
            response = requests.post(f"{API_BASE_URL}/detect", params={"example": selected_example})
        else:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE_URL}/detect", files=files)

    if response.status_code != 200:
        st.error(f"Detection failed: {response.text}")
    else:
        result = response.json()
        n_beads = result["n_beads"]
        points = result["points"]

        st.divider()
        st.subheader("3. Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            # PIL can open PNG/JPG/TIFF for display, but not .czi (a proprietary
            # microscopy format) — for .czi uploads specifically, show a plain
            # results summary instead of attempting a preview that would fail.
            try:
                image = np.array(Image.open(io.BytesIO(image_bytes_for_display)).convert("L"))
                fig = draw_overlay(image, points)
                st.pyplot(fig)
            except Exception:
                st.info(
                    f"Detection succeeded ({n_beads} bead(s) found), but an in-browser preview "
                    "isn't available for .czi files directly — only PNG/JPG/TIFF can be "
                    "previewed here. The detection itself is unaffected."
                )

        with col2:
            st.metric("Beads detected", n_beads)

            label, explanation, color = interpret_bead_count(n_beads)
            st.markdown(f":{color}[**{label}**]")
            st.write(explanation)

            st.caption(
                "Thresholds are heuristic, derived from this project's own annotated "
                "dataset distribution — not an established clinical or diagnostic standard."
            )
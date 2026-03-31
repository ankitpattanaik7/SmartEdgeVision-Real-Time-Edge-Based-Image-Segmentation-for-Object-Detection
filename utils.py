"""Utility helpers for image I/O, visualization, and analytics."""

from __future__ import annotations

from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR OpenCV image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image.")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for Streamlit/Matplotlib display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize arbitrary numeric image arrays into [0, 255] uint8 format."""
    norm = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def encode_png(image: np.ndarray, is_bgr: bool = False) -> bytes:
    """Encode an image to PNG bytes for Streamlit download buttons."""
    if image.ndim == 2:
        img = image
    elif is_bgr:
        img = image
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image for download.")
    return buffer.tobytes()


def stage_figure(images: list[np.ndarray], titles: list[str], cmap: str = "gray"):
    """Create a compact matplotlib figure for intermediate stage inspection."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig


def edge_density(edge_map: np.ndarray) -> float:
    """Compute proportion of edge pixels for sensitivity analysis."""
    return float(np.count_nonzero(edge_map) / edge_map.size)


def performance_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Format benchmark rows into a tidy DataFrame."""
    return pd.DataFrame(rows)

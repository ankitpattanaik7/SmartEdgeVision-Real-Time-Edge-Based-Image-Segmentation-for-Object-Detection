"""Custom Canny edge detection implementation for academic inspection.

This module provides a step-wise implementation of the Canny pipeline:
1) Gaussian smoothing
2) Gradient computation (Sobel)
3) Non-maximum suppression
4) Double thresholding
5) Edge tracking by hysteresis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CannyResult:
    """Container for full Canny outputs, including intermediate stages."""

    edges: np.ndarray
    blurred: np.ndarray
    gradient_magnitude: np.ndarray
    gradient_direction: np.ndarray
    non_max_suppressed: np.ndarray
    thresholded: np.ndarray


def _validate_thresholds(low_threshold: int, high_threshold: int) -> tuple[int, int]:
    """Ensure thresholds are bounded and ordered."""
    low = int(np.clip(low_threshold, 0, 255))
    high = int(np.clip(high_threshold, 0, 255))
    if low > high:
        low, high = high, low
    return low, high


def gaussian_smoothing(gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Apply Gaussian smoothing to suppress sensor and texture noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def compute_gradients(blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel derivatives, gradient magnitude, and direction."""
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(grad_x, grad_y)
    magnitude = magnitude / (magnitude.max() + 1e-8) * 255.0

    direction = np.arctan2(grad_y, grad_x)
    return grad_x, grad_y, magnitude.astype(np.float32), direction


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin edges by preserving local maxima along gradient directions."""
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.float32)

    angle = np.rad2deg(direction)
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 0.0
            r = 0.0

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def double_threshold(
    nms: np.ndarray,
    low_threshold: int,
    high_threshold: int,
    weak_value: int = 75,
    strong_value: int = 255,
) -> np.ndarray:
    """Classify pixels into strong, weak, and non-edge categories."""
    low, high = _validate_thresholds(low_threshold, high_threshold)
    thresholded = np.zeros_like(nms, dtype=np.uint8)

    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms >= low) & (nms < high))

    thresholded[strong_i, strong_j] = strong_value
    thresholded[weak_i, weak_j] = weak_value

    return thresholded


def edge_tracking_by_hysteresis(
    thresholded: np.ndarray, weak_value: int = 75, strong_value: int = 255
) -> np.ndarray:
    """Promote weak edges connected to strong edges via 8-neighborhood traversal."""
    rows, cols = thresholded.shape
    edges = thresholded.copy()

    strong_points = np.argwhere(edges == strong_value)
    stack = [tuple(p) for p in strong_points]

    while stack:
        x, y = stack.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and edges[nx, ny] == weak_value:
                    edges[nx, ny] = strong_value
                    stack.append((nx, ny))

    edges[edges != strong_value] = 0
    return edges


def custom_canny(
    gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    kernel_size: int = 5,
    sigma: float = 1.4,
    return_intermediates: bool = True,
) -> CannyResult:
    """Run the full custom Canny pipeline on a single-channel image."""
    if gray.ndim != 2:
        raise ValueError("custom_canny expects a grayscale image.")

    blurred = gaussian_smoothing(gray, kernel_size=kernel_size, sigma=sigma)
    _, _, magnitude, direction = compute_gradients(blurred)
    nms = non_maximum_suppression(magnitude, direction)
    thresholded = double_threshold(nms, low_threshold=low_threshold, high_threshold=high_threshold)
    edges = edge_tracking_by_hysteresis(thresholded)

    if not return_intermediates:
        zero = np.zeros_like(edges, dtype=np.float32)
        return CannyResult(edges=edges, blurred=gray, gradient_magnitude=zero, gradient_direction=zero, non_max_suppressed=zero, thresholded=np.zeros_like(edges))

    return CannyResult(
        edges=edges,
        blurred=blurred,
        gradient_magnitude=magnitude,
        gradient_direction=direction,
        non_max_suppressed=nms,
        thresholded=thresholded,
    )


def opencv_canny(gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Reference OpenCV Canny implementation used for benchmarking."""
    low, high = _validate_thresholds(low_threshold, high_threshold)
    return cv2.Canny(gray, threshold1=low, threshold2=high)

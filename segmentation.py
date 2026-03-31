"""Segmentation methods built on top of edge maps and regional similarity."""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np


def contour_segmentation(image_bgr: np.ndarray, edge_map: np.ndarray, min_area: int = 120) -> dict[str, np.ndarray]:
    """Segment objects using contour closure inferred from edge maps."""
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = image_bgr.copy()
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 128, 255), 1)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    isolated = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return {"annotated": annotated, "mask": mask, "isolated": isolated}


def region_growing_segmentation(
    image_bgr: np.ndarray,
    seed_point: Optional[tuple[int, int]] = None,
    tolerance: int = 15,
) -> dict[str, np.ndarray]:
    """Simple region-growing segmentation around a seed based on intensity similarity."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if seed_point is None:
        seed_point = (h // 2, w // 2)

    sx, sy = seed_point
    sx = int(np.clip(sx, 0, h - 1))
    sy = int(np.clip(sy, 0, w - 1))

    seed_value = int(gray[sx, sy])
    visited = np.zeros_like(gray, dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=np.uint8)

    q = deque([(sx, sy)])
    visited[sx, sy] = 1

    while q:
        x, y = q.popleft()

        if abs(int(gray[x, y]) - seed_value) <= tolerance:
            mask[x, y] = 255
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        visited[nx, ny] = 1
                        q.append((nx, ny))

    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return {"mask": mask, "segmented": segmented}


def watershed_segmentation(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Marker-controlled watershed segmentation for region partitioning."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_ws = cv2.watershed(image_bgr.copy(), markers)

    boundaries = image_bgr.copy()
    boundaries[markers_ws == -1] = [0, 0, 255]

    marker_vis = cv2.normalize(markers_ws.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    marker_vis = marker_vis.astype(np.uint8)

    return {
        "binary": binary,
        "markers": marker_vis,
        "boundaries": boundaries,
    }

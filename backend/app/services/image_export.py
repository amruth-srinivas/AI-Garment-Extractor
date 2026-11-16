"""Write full-resolution exports and lighter previews."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.utils.image_utils import preview_rgb, save_jpeg, save_png, white_background


def render_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    selected = mask.astype(bool)
    if selected.any():
        colour = np.array([22, 110, 78], dtype=np.float32)
        output[selected] = (
            output[selected].astype(np.float32) * 0.58 + colour * 0.42
        ).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    thickness = max(1, int(round(min(rgb.shape[:2]) / 420)))
    cv2.drawContours(output, contours, -1, (14, 64, 48), thickness)
    return output


def mask_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(bool).astype(np.uint8)) * 255


class ImageExporter:
    def write_previews(self, folder: Path, rgba: np.ndarray, original: np.ndarray) -> None:
        settings = get_settings()
        white = white_background(rgba)
        preview_rgba = _preview_rgba(rgba, settings.preview_max_side)
        preview_white = preview_rgb(white, settings.preview_max_side)
        thumb = preview_rgb(white, 480)
        save_png(folder / "garment.png", rgba)
        save_jpeg(folder / "garment_white.jpg", white, quality=95)
        save_png(folder / "preview_transparent.png", preview_rgba)
        save_jpeg(folder / "preview_white.jpg", preview_white, quality=90)
        save_jpeg(folder / "thumbnail.jpg", thumb, quality=86)
        # The original preview already lives with the upload. A copy keeps
        # the result folder usable if the upload is later cleaned up.
        save_jpeg(folder / "original_preview.jpg", preview_rgb(original, settings.preview_max_side), quality=90)

    def write_masks(
        self,
        folder: Path,
        *,
        garment: np.ndarray,
        core: np.ndarray,
        holes: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        save_png(folder / "mask.png", mask_u8(garment))
        save_png(folder / "core_mask.png", mask_u8(core))
        save_png(folder / "inpaint_mask.png", mask_u8(holes))
        save_png(folder / "labels.png", labels.astype(np.uint8))
        save_png(folder / "confidence.png", np.clip(confidence * 255.0, 0, 255).astype(np.uint8))


def _preview_rgba(rgba: np.ndarray, max_side: int) -> np.ndarray:
    height, width = rgba.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return rgba
    scale = max_side / float(long_side)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    rgb = cv2.resize(rgba[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(rgba[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.dstack([rgb, alpha])


image_exporter = ImageExporter()

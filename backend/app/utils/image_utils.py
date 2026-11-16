"""Image IO helpers. Colour arrays are RGB uint8 unless noted."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def working_size(width: int, height: int, max_side: int) -> tuple[int, int, float]:
    """Return (new_height, new_width, scale) with the long side capped."""

    long_side = max(width, height)
    if long_side <= max_side or max_side <= 0:
        return height, width, 1.0
    scale = max_side / float(long_side)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale))), scale


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    interpolation = cv2.INTER_AREA if width < rgb.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(rgb, (width, height), interpolation=interpolation)


def resize_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def resize_linear(values: np.ndarray, width: int, height: int) -> np.ndarray:
    if values.shape[1] == width and values.shape[0] == height:
        return values.astype(np.float32)
    return cv2.resize(values.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def preview_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width, _ = working_size(rgb.shape[1], rgb.shape[0], max_side)
    return resize_rgb(rgb, width, height)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", compress_level=3)


def save_jpeg(path: Path, rgb: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
    )


def encode_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def white_background(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    composite = rgb * alpha + 255.0 * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

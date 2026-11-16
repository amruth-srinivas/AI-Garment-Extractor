"""Upload checks for fashion photographs."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_settings
from app.errors import AppError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _extension(filename: str) -> str:
    dot = filename.rfind(".")
    if dot < 0:
        return ""
    return filename[dot:].lower()


def _magic_matches(data: bytes, extension: str) -> bool:
    if extension in {".jpg", ".jpeg"}:
        return data[:3] == b"\xff\xd8\xff"
    if extension == ".png":
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    if extension == ".webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def decode_upload(data: bytes, filename: str) -> tuple[np.ndarray, dict]:
    settings = get_settings()
    extension = _extension(filename or "")
    if extension not in ALLOWED_EXTENSIONS:
        raise AppError("Upload a JPG, JPEG, PNG, or WEBP image.")
    if not data:
        raise AppError("The uploaded file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise AppError(f"File exceeds the {settings.max_upload_mb} MB limit.", 413)
    if not _magic_matches(data, extension):
        raise AppError("The file contents do not match a JPG, PNG, or WEBP image.")

    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        if getattr(image, "is_animated", False):
            image.seek(0)
        rgb_image = image.convert("RGB")
    except AppError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AppError("The image could not be decoded.") from exc

    width, height = rgb_image.size
    if width < 64 or height < 64:
        raise AppError("Image is too small. Use a photo at least 64 pixels on the shorter side.")
    if width * height > settings.max_pixels:
        raise AppError(
            "Image resolution is too high for this workstation. "
            f"The limit is {settings.max_pixels:,} pixels."
        )

    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise AppError("The image could not be read as RGB.")
    return rgb, {
        "filename": filename,
        "width": int(width),
        "height": int(height),
        "bytes": len(data),
    }

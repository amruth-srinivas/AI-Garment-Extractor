"""Load stored uploads and write the lossless working copy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.errors import AppError
from app.utils.image_utils import load_rgb, preview_rgb, save_jpeg, save_png


def upload_dir(image_id: str) -> Path:
    return get_settings().uploads_dir / image_id


def original_path(image_id: str) -> Path:
    path = upload_dir(image_id) / "original.png"
    if not path.exists():
        raise AppError("That upload could not be found.", 404)
    return path


def save_upload(image_id: str, rgb: np.ndarray, meta: dict) -> dict:
    settings = get_settings()
    folder = upload_dir(image_id)
    folder.mkdir(parents=True, exist_ok=True)
    save_png(folder / "original.png", rgb)
    preview = preview_rgb(rgb, settings.preview_max_side)
    save_jpeg(folder / "preview.jpg", preview, quality=90)
    payload = {
        **meta,
        "image_id": image_id,
        "preview_url": f"/api/uploads/{image_id}/file?kind=preview",
        "original_url": f"/api/uploads/{image_id}/file?kind=original",
    }
    (folder / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_upload_meta(image_id: str) -> dict:
    path = upload_dir(image_id) / "meta.json"
    if not path.exists():
        raise AppError("That upload could not be found.", 404)
    return json.loads(path.read_text(encoding="utf-8"))


def load_original(image_id: str) -> np.ndarray:
    return load_rgb(original_path(image_id))

"""Photograph upload."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.api.ids import require_id
from app.config import get_settings
from app.errors import AppError
from app.services.image_loader import load_upload_meta, save_upload, upload_dir
from app.utils.validation import decode_upload

router = APIRouter(prefix="/uploads", tags=["uploads"])


async def read_limited(upload: UploadFile, max_mb: int | None = None) -> bytes:
    settings = get_settings()
    limit = (max_mb or settings.max_upload_mb) * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise AppError(f"File exceeds the {max_mb or settings.max_upload_mb} MB limit.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("", status_code=201, summary="Upload a fashion photograph")
async def create_upload(file: UploadFile = File(...)) -> dict:
    """Accept a JPG, JPEG, PNG, or WEBP fashion photograph.

    The stored original is a lossless PNG after EXIF orientation is applied,
    so later exports are not recompressed before the garment is cut out.
    """

    payload = await read_limited(file)
    filename = file.filename or "upload.png"
    rgb, info = await asyncio.to_thread(decode_upload, payload, filename)
    image_id = uuid.uuid4().hex
    return await asyncio.to_thread(save_upload, image_id, rgb, info)


@router.get("/{image_id}", summary="Read upload details")
def get_upload(image_id: str) -> dict:
    require_id(image_id)
    return load_upload_meta(image_id)


@router.get("/{image_id}/file", summary="Fetch the original or the preview")
def get_upload_file(image_id: str, kind: str = "preview") -> FileResponse:
    require_id(image_id)
    load_upload_meta(image_id)
    folder = upload_dir(image_id)
    if kind == "preview":
        path, media = folder / "preview.jpg", "image/jpeg"
    elif kind == "original":
        path, media = folder / "original.png", "image/png"
    else:
        raise AppError("kind must be preview or original.", 404)
    if not path.exists():
        raise AppError("That file could not be found.", 404)
    return FileResponse(path, media_type=media)

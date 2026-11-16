"""Result preview, download, mask edits, and history."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.api.ids import require_id
from app.api.routes.upload import read_limited
from app.errors import AppError
from app.services.job_manager import jobs
from app.services.pipeline import delete_result, list_results, load_result, result_file, update_product_tag

router = APIRouter(prefix="/results", tags=["results"])


class TagUpdate(BaseModel):
    product_tag: str


@router.get("", summary="List finished extractions")
def history(limit: int = 12) -> dict:
    items = list_results()
    bounded = max(1, min(limit, 50))
    recent = []
    for item in items[:bounded]:
        recent.append(
            {
                "id": item.get("id"),
                "product_tag": item.get("product_tag"),
                "garment_label": item.get("garment_label"),
                "confidence": item.get("confidence"),
                "low_confidence": item.get("low_confidence"),
                "parser": item.get("parser"),
                "created_at": item.get("created_at"),
                "thumbnail_url": item.get("thumbnail_url"),
            }
        )
    return {"count": len(items), "items": recent}


@router.get("/{result_id}", summary="Read one extraction result")
def get_result(result_id: str) -> dict:
    require_id(result_id)
    return load_result(result_id)


@router.delete("/{result_id}", status_code=204, summary="Delete one extraction")
def remove_result(result_id: str) -> Response:
    """Delete a finished extraction from the studio history.

    The photograph upload is kept. A result that is still being edited cannot
    be deleted until that job finishes.
    """

    require_id(result_id)
    delete_result(result_id)
    return Response(status_code=204)


@router.patch("/{result_id}", summary="Set the product tag")
def patch_result(result_id: str, body: TagUpdate) -> dict:
    require_id(result_id)
    return update_product_tag(result_id, body.product_tag)


@router.get("/{result_id}/file", summary="Preview a result image")
def get_result_file(result_id: str, kind: str = "transparent") -> FileResponse:
    require_id(result_id)
    path, media = result_file(result_id, kind)
    return FileResponse(path, media_type=media)


@router.get("/{result_id}/download", summary="Download the garment PNG or JPEG")
def download_result(result_id: str, variant: str = "transparent", format: str = "png") -> FileResponse:
    """Transparent PNG keeps the alpha. JPEG is flattened on white.

    `variant` is accepted so clients can ask for the white JPEG explicitly.
    A PNG request always returns the transparent garment. A JPEG request always
    returns the white-background product image.
    """

    require_id(result_id)
    meta = load_result(result_id)
    tag = meta.get("product_tag") or "garment"
    if format == "png":
        if variant not in {"transparent", "white"}:
            raise AppError("variant must be transparent or white.", 400)
        path, media = result_file(result_id, "garment")
        filename = f"garmentai-{tag}.png"
    elif format in {"jpeg", "jpg"}:
        path, media = result_file(result_id, "garment_white")
        filename = f"garmentai-{tag}.jpg"
    else:
        raise AppError("format must be png or jpeg.", 400)
    return FileResponse(path, media_type=media, filename=filename)


@router.post("/{result_id}/mask", status_code=202, summary="Apply a hand-edited mask")
async def edit_mask(
    result_id: str,
    mask: UploadFile = File(...),
    regenerate: bool = False,
) -> dict:
    """Replace the garment alpha with the uploaded mask.

    White pixels are kept. Restored pixels come from the original photograph.
    Set regenerate=true to estimate hidden cloth again inside the new mask.
    """

    require_id(result_id)
    meta = load_result(result_id)
    payload = await read_limited(mask, max_mb=20)
    if len(payload) < 32:
        raise AppError("The mask file is empty.")
    job = jobs.create(
        meta["image_id"],
        {
            "task": "edit",
            "result_id": result_id,
            "regenerate": regenerate,
        },
    )
    path = result_file_folder(result_id) / f"edit_{job['id']}.png"
    path.write_bytes(payload)
    jobs.patch_options(job["id"], mask_name=path.name)
    jobs.submit(job["id"])
    return jobs.snapshot(job["id"])


@router.post("/{result_id}/reconstruct", status_code=202, summary="Regenerate hidden cloth")
def regenerate(result_id: str) -> dict:
    """Run inpainting again on body-occluded regions of the current mask.

    The fill is an estimate. Original garment pixels are not replaced.
    """

    require_id(result_id)
    meta = load_result(result_id)
    job = jobs.create(
        meta["image_id"],
        {
            "task": "edit",
            "result_id": result_id,
            "regenerate": True,
        },
    )
    jobs.submit(job["id"])
    return jobs.snapshot(job["id"])


def result_file_folder(result_id: str):
    from app.config import get_settings

    return get_settings().results_dir / result_id

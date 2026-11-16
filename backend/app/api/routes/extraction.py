"""Garment detection and extraction jobs.

Progress reported by these routes is the stage the worker is actually in.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.ids import require_id
from app.errors import AppError
from app.services.image_loader import load_upload_meta
from app.services.job_manager import jobs
from app.services.pipeline import garment_overlay_path

router = APIRouter(tags=["extraction"])


class ExtractionRequest(BaseModel):
    image_id: str
    parser: Literal["auto", "segformer", "mediapipe"] = "auto"
    reconstruct: bool = True
    force_reconstruct: bool = False
    refine_sam2: bool = False
    clean_background: bool = True


class ContinueRequest(BaseModel):
    garment_ids: list[str] = Field(min_length=1)
    product_tag: str | None = None
    reconstruct: bool | None = None
    refine_sam2: bool | None = None
    force_reconstruct: bool = False


@router.post("/extractions", status_code=202, summary="Start garment detection")
def start_extraction(body: ExtractionRequest) -> dict:
    """Detect clothing on an uploaded photograph.

    When more than one region is found, the job pauses in `awaiting_selection`
    so the caller can choose. A single region also pauses, with that region
    preselected by the client, so the mask can be checked before export.
    Pass `selection` only through the continue call.
    """

    require_id(body.image_id)
    load_upload_meta(body.image_id)
    job = jobs.create(
        body.image_id,
        {
            "task": "extract",
            "parser": body.parser,
            "reconstruct": body.reconstruct,
            "force_reconstruct": body.force_reconstruct,
            "refine_sam2": body.refine_sam2,
            "clean_background": body.clean_background,
            "selection": "ask",
            "reuse_parse": False,
            "garment_ids": None,
            "product_tag": None,
        },
    )
    jobs.submit(job["id"])
    return job


@router.get("/extractions/{job_id}", summary="Poll extraction progress")
def get_extraction(job_id: str) -> dict:
    require_id(job_id)
    return jobs.snapshot(job_id)


@router.post("/extractions/{job_id}/continue", status_code=202, summary="Extract the chosen garments")
def continue_extraction(job_id: str, body: ContinueRequest) -> dict:
    """Run human removal and optional reconstruction for the selected regions."""

    require_id(job_id)
    job = jobs.get(job_id)
    if job["status"] != "awaiting_selection":
        raise AppError("This extraction is not waiting for a garment choice.", 409)
    changes: dict = {
        "garment_ids": body.garment_ids,
        "reuse_parse": True,
        "selection": "ask",
        "force_reconstruct": body.force_reconstruct,
    }
    if body.product_tag is not None:
        changes["product_tag"] = body.product_tag
    if body.reconstruct is not None:
        changes["reconstruct"] = body.reconstruct
    if body.refine_sam2 is not None:
        changes["refine_sam2"] = body.refine_sam2
    jobs.patch_options(job_id, **changes)
    jobs.submit(job_id)
    return jobs.snapshot(job_id)


@router.get(
    "/extractions/{job_id}/garments/{garment_id}/overlay",
    summary="Preview a detected garment mask",
)
def garment_overlay(job_id: str, garment_id: str) -> FileResponse:
    require_id(job_id)
    jobs.get(job_id)
    path = garment_overlay_path(job_id, garment_id)
    return FileResponse(path, media_type="image/jpeg")

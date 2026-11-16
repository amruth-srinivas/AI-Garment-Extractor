"""Health, hardware, and model availability."""

from __future__ import annotations

import importlib.util
import sys

from fastapi import APIRouter

from app.config import get_settings
from app.constants import (
    LAMA_LICENSE,
    MEDIAPIPE_LICENSE,
    PRODUCT_TAGS,
    REMBG_LICENSE,
    SAM2_LICENSE,
    SEGFORMER_LICENSE,
)
from app.models.downloads import hf_ready
from app.models.model_manager import gpu_status

router = APIRouter(tags=["system"])


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@router.get("/health", summary="Runtime, GPU, and model status")
def health() -> dict:
    settings = get_settings()
    gpu = gpu_status()
    vram = gpu.get("vram_mb") or 0
    return {
        "app": "GarmentAI",
        "python": sys.version.split()[0],
        "torch": gpu.get("torch"),
        "gpu": gpu,
        "defaults": {
            "parser": "auto",
            "reconstruct": True,
            "refine_sam2": bool(vram >= 7000),
            "clean_background": True,
        },
        "product_tags": PRODUCT_TAGS,
        "models": {
            "segformer": {
                "id": settings.segformer_model_id,
                "ready": hf_ready(settings.segformer_model_id, settings),
                "installed": _installed("transformers"),
                "license": SEGFORMER_LICENSE,
                "purpose": "Separate upper clothes, dress, skirt, pants, hair, face, arms, and legs.",
            },
            "mediapipe": {
                "id": "selfie_multiclass_256x256",
                "ready": settings.mediapipe_path.exists(),
                "installed": _installed("mediapipe"),
                "license": MEDIAPIPE_LICENSE,
                "purpose": "Separate background, hair, body skin, face skin, clothes, and accessories.",
            },
            "sam2": {
                "id": settings.sam2_model_id,
                "ready": hf_ready(settings.sam2_model_id, settings),
                "installed": _installed("transformers"),
                "license": SAM2_LICENSE,
                "purpose": "Optional edge refinement. It cannot replace the clothing parser.",
            },
            "lama": {
                "id": "big-lama",
                "ready": settings.lama_path.exists() and settings.lama_path.stat().st_size > 50_000_000,
                "installed": _installed("torch"),
                "license": LAMA_LICENSE,
                "purpose": "Estimate cloth hidden by the body. It does not recover the original hidden textile.",
            },
            "rembg": {
                "id": "u2net",
                "ready": _installed("rembg"),
                "installed": _installed("rembg"),
                "license": REMBG_LICENSE,
                "purpose": "Remove leftover background on the garment edge after clothing has been separated from the person.",
            },
        },
    }

"""Lazy model loading, GPU detection, and CPU fallback.

Models are loaded one at a time. On a 4GB GPU the parser is released before
SAM 2 or LaMa so the activations of two networks are not resident together.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

from app.config import get_settings
from app.errors import AppError

logger = logging.getLogger("garmentai.models")


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise AppError(
            "PyTorch is not installed in this environment. Install the CUDA build "
            "described in README.md before running extraction."
        ) from exc
    return torch


def gpu_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "name": None, "vram_mb": None, "torch": None}
    info = {
        "available": bool(torch.cuda.is_available()),
        "name": None,
        "vram_mb": None,
        "torch": torch.__version__,
    }
    if info["available"]:
        properties = torch.cuda.get_device_properties(0)
        info["name"] = properties.name
        info["vram_mb"] = int(properties.total_memory / (1024 * 1024))
    return info


def pick_device(prefer_gpu: bool = True):
    torch = import_torch()
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segformer: tuple[Any, Any, str] | None = None
        self._mediapipe: Any = None
        self._sam2: tuple[Any, Any, str] | None = None
        self._lama: tuple[Any, str] | None = None
        self._rembg: Any = None

    def unload(self, name: str | None = None) -> None:
        with self._lock:
            if name in (None, "segformer"):
                self._segformer = None
            if name in (None, "mediapipe"):
                segmenter = self._mediapipe
                self._mediapipe = None
                if segmenter is not None:
                    close = getattr(segmenter, "close", None)
                    if callable(close):
                        close()
            if name in (None, "sam2"):
                self._sam2 = None
            if name in (None, "lama"):
                self._lama = None
            if name in (None, "rembg"):
                self._rembg = None
        release_cuda()

    def segformer(self, device: str | None = None):
        from app.models.downloads import ensure_hf_repo

        settings = get_settings()
        torch = import_torch()
        target = device or str(pick_device())
        with self._lock:
            if self._segformer is not None and self._segformer[2] == target:
                return self._segformer[0], self._segformer[1]
        ensure_hf_repo(settings.segformer_model_id, settings)
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise AppError("transformers is not installed.") from exc
        logger.info("Loading SegFormer clothes parser on %s", target)
        processor = AutoImageProcessor.from_pretrained(
            settings.segformer_model_id,
            cache_dir=str(settings.hf_home / "hub"),
            local_files_only=not settings.auto_download,
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            settings.segformer_model_id,
            cache_dir=str(settings.hf_home / "hub"),
            local_files_only=not settings.auto_download,
        )
        model.eval()
        model.to(torch.device(target))
        with self._lock:
            self._segformer = (processor, model, target)
        return processor, model

    def mediapipe(self):
        from app.models.downloads import ensure_mediapipe

        with self._lock:
            if self._mediapipe is not None:
                return self._mediapipe
        path = ensure_mediapipe()
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise AppError(
                "mediapipe is not installed. Install backend\\requirements.txt."
            ) from exc
        base = mp_python.BaseOptions(model_asset_path=str(path))
        try:
            options = vision.ImageSegmenterOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=True,
            )
            segmenter = vision.ImageSegmenter.create_from_options(options)
        except (TypeError, ValueError):
            options = vision.ImageSegmenterOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
            segmenter = vision.ImageSegmenter.create_from_options(options)
        with self._lock:
            self._mediapipe = segmenter
        return segmenter

    def sam2(self, device: str | None = None):
        from app.models.downloads import ensure_hf_repo

        settings = get_settings()
        torch = import_torch()
        target = device or str(pick_device())
        with self._lock:
            if self._sam2 is not None and self._sam2[2] == target:
                return self._sam2[0], self._sam2[1]
        ensure_hf_repo(settings.sam2_model_id, settings)
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise AppError(
                "This transformers build does not export Sam2Model. "
                "Upgrade transformers to 4.46 or newer."
            ) from exc
        logger.info("Loading SAM 2.1 hiera-tiny on %s", target)
        processor = Sam2Processor.from_pretrained(
            settings.sam2_model_id,
            cache_dir=str(settings.hf_home / "hub"),
            local_files_only=not settings.auto_download,
        )
        model = Sam2Model.from_pretrained(
            settings.sam2_model_id,
            cache_dir=str(settings.hf_home / "hub"),
            local_files_only=not settings.auto_download,
        )
        model.eval()
        model.to(torch.device(target))
        with self._lock:
            self._sam2 = (processor, model, target)
        return processor, model

    def lama(self, device: str | None = None):
        from app.models.downloads import ensure_lama

        torch = import_torch()
        target = device or str(pick_device())
        with self._lock:
            if self._lama is not None and self._lama[1] == target:
                return self._lama[0]
        path = ensure_lama()
        logger.info("Loading LaMa inpainting on %s", target)
        model = torch.jit.load(str(path), map_location=target)
        model.eval()
        with self._lock:
            self._lama = (model, target)
        return model

    def rembg_session(self):
        with self._lock:
            if self._rembg is not None:
                return self._rembg
        try:
            from rembg import new_session
        except ImportError as exc:
            raise AppError("rembg is not installed.") from exc
        session = new_session("u2net")
        with self._lock:
            self._rembg = session
        return session


models = ModelManager()

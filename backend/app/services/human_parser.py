"""Human parsing.

Two real models are supported:

* SegFormer B2 clothes, which separates upper clothes, skirt, pants, dress,
  hair, face, arms, and legs.
* MediaPipe selfie multiclass, which separates background, hair, body skin,
  face skin, clothes, and accessories.

Neither model is a background remover. Clothing and the person are different
classes. If the requested weights are missing, parsing stops with a setup
error instead of inventing a mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings
from app.constants import MEDIAPIPE_LABELS, SEGFORMER_LABELS
from app.errors import AppError
from app.models.model_manager import models, pick_device, release_cuda
from app.utils.image_utils import resize_linear, resize_nearest, working_size

logger = logging.getLogger("garmentai.parser")


@dataclass
class ParseResult:
    labels: np.ndarray
    confidence: np.ndarray
    source: str
    label_names: dict[int, str]
    notes: list[str] = field(default_factory=list)
    license_note: str = ""


def _label_map(config_map: dict) -> dict[int, str]:
    names: dict[int, str] = {}
    for key, value in config_map.items():
        names[int(key)] = str(value)
    return names or dict(SEGFORMER_LABELS)


def _upsample(labels: np.ndarray, confidence: np.ndarray, width: int, height: int):
    full_labels = resize_nearest(labels.astype(np.uint8), width, height)
    full_conf = resize_linear(confidence.astype(np.float32), width, height)
    return full_labels.astype(np.uint8), np.clip(full_conf, 0.0, 1.0).astype(np.float32)


def parse_segformer(rgb: np.ndarray) -> ParseResult:
    from app.constants import SEGFORMER_LICENSE

    settings = get_settings()
    height, width = rgb.shape[:2]
    work_h, work_w, scale = working_size(width, height, settings.mask_max_side)
    device = pick_device()
    try:
        processor, model = models.segformer(str(device))
        return _run_segformer(rgb, processor, model, device, work_h, work_w, scale, SEGFORMER_LICENSE)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or str(device) == "cpu":
            raise
        logger.warning("SegFormer ran out of GPU memory. Retrying on CPU.")
        release_cuda()
        models.unload("segformer")
        processor, model = models.segformer("cpu")
        note_prefix = "SegFormer used the CPU because the GPU ran out of memory. "
        result = _run_segformer(
            rgb, processor, model, "cpu", work_h, work_w, scale, SEGFORMER_LICENSE
        )
        result.notes.insert(0, note_prefix.strip())
        return result


def _run_segformer(rgb, processor, model, device, work_h, work_w, scale, license_note) -> ParseResult:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    source = rgb
    if scale < 1.0:
        source = np.asarray(Image.fromarray(rgb).resize((work_w, work_h), Image.Resampling.LANCZOS))
    image = Image.fromarray(source)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    logits = logits.float().cpu()
    del inputs
    release_cuda()
    upsampled = F.interpolate(logits, size=(work_h, work_w), mode="bilinear", align_corners=False)
    probabilities = torch.softmax(upsampled, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    labels = prediction[0].numpy().astype(np.uint8)
    conf = confidence[0].numpy().astype(np.float32)
    full_labels, full_conf = _upsample(labels, conf, rgb.shape[1], rgb.shape[0])
    names = _label_map(getattr(model.config, "id2label", SEGFORMER_LABELS))
    notes = []
    if scale < 1.0:
        notes.append(
            "The garment outline was estimated at "
            f"{work_w}x{work_h} and applied to the original "
            f"{rgb.shape[1]}x{rgb.shape[0]} photograph, so the cloth texture "
            "is still the original pixels."
        )
    return ParseResult(
        labels=full_labels,
        confidence=full_conf,
        source="segformer",
        label_names=names,
        notes=notes,
        license_note=license_note,
    )


def parse_mediapipe(rgb: np.ndarray) -> ParseResult:
    from app.constants import MEDIAPIPE_LICENSE

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise AppError("mediapipe is not installed.") from exc

    segmenter = models.mediapipe()
    contiguous = np.ascontiguousarray(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous)
    result = segmenter.segment(mp_image)
    if result.category_mask is None:
        raise AppError("MediaPipe did not return a category mask.")
    labels = np.squeeze(np.array(result.category_mask.numpy_view()))
    if labels.ndim != 2:
        raise AppError("MediaPipe returned a category mask with an unexpected shape.")
    labels = labels.astype(np.uint8)

    confidence = np.full(labels.shape[:2], 0.35, dtype=np.float32)
    if result.confidence_masks:
        # Channel 4 is clothes. Use the winning class confidence everywhere.
        stacked = []
        for mask in result.confidence_masks:
            channel = np.squeeze(np.array(mask.numpy_view())).astype(np.float32)
            if channel.ndim != 2:
                continue
            stacked.append(channel)
        if stacked:
            cube = np.stack(stacked, axis=0)
            if cube.shape[1:] != labels.shape[:2]:
                resized = [
                    resize_linear(channel, labels.shape[1], labels.shape[0]) for channel in stacked
                ]
                cube = np.stack(resized, axis=0)
            confidence = cube.max(axis=0)

    full_labels, full_conf = _upsample(labels, confidence, rgb.shape[1], rgb.shape[0])
    notes = [
        "MediaPipe selfie multiclass has a single clothes class. "
        "Shirt, dress, saree, kurti, lehenga, skirt, and trousers are not "
        "separate trained categories. A top and bottom are split only when "
        "skin creates a reliable gap."
    ]
    return ParseResult(
        labels=full_labels,
        confidence=full_conf,
        source="mediapipe",
        label_names=dict(MEDIAPIPE_LABELS),
        notes=notes,
        license_note=MEDIAPIPE_LICENSE,
    )


def parse_person(rgb: np.ndarray, parser: str) -> ParseResult:
    choice = (parser or "auto").lower()
    if choice not in {"auto", "segformer", "mediapipe"}:
        raise AppError("Parser must be auto, segformer, or mediapipe.")

    if choice == "mediapipe":
        return parse_mediapipe(rgb)
    if choice == "segformer":
        return parse_segformer(rgb)

    try:
        return parse_segformer(rgb)
    except AppError as exc:
        logger.warning("SegFormer unavailable (%s). Falling back to MediaPipe.", exc.message)
        try:
            result = parse_mediapipe(rgb)
        except AppError as media_exc:
            raise AppError(
                "No human parser is available. "
                f"SegFormer: {exc.message} MediaPipe: {media_exc.message}"
            ) from media_exc
        result.notes.insert(0, f"SegFormer was not used. {exc.message}")
        return result


class HumanParser:
    """Clothing-versus-body parser. The backend can be SegFormer or MediaPipe."""

    def parse(self, rgb: np.ndarray, parser: str) -> ParseResult:
        return parse_person(rgb, parser)


human_parser = HumanParser()

"""End-to-end extraction.

The parse tells cloth from the person. Refinement only moves the boundary.
Reconstruction writes into body-shaped holes and nowhere else. Export uses
the original photograph's pixels for every unoccluded part of the garment.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.constants import BODY_CLASS_NAMES, PRODUCT_TAGS
from app.errors import AppError
from app.models.model_manager import models, release_cuda
from app.services.background_removal import background_removal
from app.services.garment_detector import GarmentCandidate, garment_detector
from app.services.garment_reconstruction import (
    LOW_CONFIDENCE_MESSAGE as RECON_LOW,
    garment_reconstruction,
)
from app.services.human_parser import ParseResult, human_parser
from app.services.human_removal import human_removal
from app.services.image_export import image_exporter, render_overlay
from app.services.image_loader import load_original
from app.services.job_manager import jobs, utc_now
from app.services.mask_refinement import mask_refiner
from app.services.segmentation import confidence_is_low, mask_confidence, segmentation
from app.utils.image_utils import load_gray, preview_rgb, save_jpeg, save_png
from app.utils.mask_utils import class_mask, find_occlusion_mask, ids_for

logger = logging.getLogger("garmentai.pipeline")


def execute_job(job_id: str) -> None:
    try:
        job = jobs.get(job_id)
        if job["options"].get("task") == "edit":
            _edit(job)
        else:
            _extract(job)
    finally:
        release_cuda()


def _extract(job: dict) -> None:
    job_id = job["id"]
    options = job["options"]
    rgb = load_original(job["image_id"])
    jobs.stage(job_id, "uploading_image", "Image stored")
    parsed = _load_or_parse(job_id, rgb, options.get("parser", "auto"), bool(options.get("reuse_parse")))
    jobs.stage(job_id, "identifying_garment", "Finding garment regions in the parse")
    garments = garment_detector.detect(parsed.labels, parsed.confidence, parsed.label_names, parsed.source)
    if not garments:
        raise AppError(
            "No clothing was detected. Use a photo where the garment is visible, "
            "or switch the parser and try again."
        )
    _write_garment_files(job_id, rgb, garments)
    public = [_public_garment(job_id, item) for item in garments]
    garment_ids = options.get("garment_ids") or None
    selection = options.get("selection", "ask")
    if not garment_ids and selection != "largest":
        jobs.await_selection(job_id, public)
        return
    chosen = garment_detector.choose(garments, garment_ids, "largest" if not garment_ids else "ask")
    if not chosen:
        raise AppError("Select at least one detected garment.")
    _finish(job, rgb, parsed, chosen, public)


def _finish(job, rgb, parsed: ParseResult, chosen: list[GarmentCandidate], public: list[dict]) -> None:
    settings = get_settings()
    job_id = job["id"]
    options = job["options"]
    notes = list(parsed.notes)
    notes.append(parsed.license_note)
    jobs.stage(job_id, "generating_mask", "Building the garment mask and tightening its edge")
    raw = np.zeros(parsed.labels.shape[:2], dtype=bool)
    for item in chosen:
        raw |= item.mask
    refined = mask_refiner.grabcut(rgb, raw)
    notes.append("GrabCut adjusted the garment edge from the photograph's colour.")
    if options.get("refine_sam2"):
        models.unload("segformer")
        body = class_mask(parsed.labels, ids_for(parsed.label_names, BODY_CLASS_NAMES))
        jobs.stage(job_id, "generating_mask", "Refining the garment edge with SAM 2")
        refined, sam_note = mask_refiner.sam2(rgb, refined, body)
        notes.append(sam_note)
    else:
        notes.append(
            "SAM 2 edge refinement is off. Enable it on the upload step if you want "
            "a second model to adjust the outline. A 4GB GPU runs it only after the parser is unloaded."
        )
    mask = segmentation.build(rgb, parsed.labels, refined, parsed.confidence, parsed.label_names)
    if options.get("clean_background", True):
        jobs.stage(job_id, "generating_mask", "Checking the garment edge for leftover background")
        mask, background_note = background_removal.trim(rgb, mask, parsed.confidence)
        notes.append(background_note)
    if int(mask.sum()) < int(settings.min_garment_fraction * mask.size):
        raise AppError(
            "The garment mask is too small to export. The parse may have missed the clothing. "
            "Try the other parser or a clearer photograph."
        )

    score = mask_confidence(parsed.confidence, mask)
    low = confidence_is_low(score)
    jobs.stage(job_id, "removing_human", "Removing face, hair, skin, and limbs from the garment")
    holes = find_occlusion_mask(parsed.labels, mask, parsed.label_names)
    method, message, fill, used_holes = _maybe_reconstruct(job_id, rgb, mask, holes, score, low, options)
    notes.append(message)
    if low:
        notes.append(
            f"Segmentation confidence is {score:.0%}. Review the mask before relying on the export."
        )

    jobs.stage(job_id, "refining_output", "Writing the full-resolution garment")
    if method in {"lama", "opencv_telea", "pattern_continue"}:
        rgba = human_removal.compose(rgb, mask, fill, used_holes)
        core = mask & ~used_holes
        saved_holes = used_holes
    else:
        rgba = human_removal.compose(rgb, mask)
        core = mask
        saved_holes = np.zeros(mask.shape, dtype=bool)

    result_id = uuid.uuid4().hex
    folder = _result_dir(result_id)
    folder.mkdir(parents=True, exist_ok=True)
    tag = _product_tag(options, chosen)
    labels = ", ".join(dict.fromkeys(item.label for item in chosen))
    meta = _base_meta(
        result_id=result_id,
        image_id=job["image_id"],
        job_id=job_id,
        rgb=rgb,
        confidence=score,
        low_confidence=low,
        parser=parsed.source,
        parser_license=parsed.license_note,
        method=method,
        message=message,
        product_tag=tag,
        garment_label=labels,
        garment_ids=[item.id for item in chosen],
        notes=notes,
    )
    _persist(folder, rgba, rgb, mask, core, saved_holes, parsed, meta)
    jobs.succeed(job_id, result_id, "Garment export is ready")


def _maybe_reconstruct(job_id, rgb, mask, holes, score, low, options):
    if not options.get("reconstruct", True):
        jobs.skip(job_id, "reconstructing", "Reconstruction is turned off")
        return "disabled", "Reconstruction is off. Areas hidden by the body stay transparent.", None, holes
    if low and not options.get("force_reconstruct"):
        jobs.skip(job_id, "reconstructing", RECON_LOW)
        return "skipped_low_confidence", RECON_LOW, None, holes
    if not holes.any():
        jobs.stage(job_id, "reconstructing", "Looking for cloth hidden by the body")
        return "none", "No occluded garment region was found, so nothing was generated.", None, holes
    jobs.stage(
        job_id,
        "reconstructing",
        "Estimating cloth hidden by the body. This fill is not the original fabric.",
    )
    models.unload("segformer")
    models.unload("sam2")
    fill, method, message = garment_reconstruction.fill(rgb, mask, holes, "lama")
    if method not in {"lama", "opencv_telea", "pattern_continue"}:
        return method, message, None, holes
    return method, message, fill, holes


def _edit(job: dict) -> None:
    job_id = job["id"]
    options = job["options"]
    result_id = options.get("result_id")
    if not result_id:
        raise AppError("The edit is missing its result.")
    meta = load_result(result_id)
    folder = _result_dir(result_id)
    rgb = load_original(meta["image_id"])
    jobs.stage(job_id, "uploading_image", "Image stored")
    jobs.stage(job_id, "detecting_person", "Using the saved human parse")
    parsed = _read_parse(folder, rgb)
    jobs.stage(job_id, "identifying_garment", "Keeping the selected garment")
    jobs.stage(job_id, "generating_mask", "Reading the corrected garment mask")
    previous = np.asarray(Image.open(folder / "garment.png").convert("RGBA"))
    if options.get("mask_name"):
        new_mask = _decode_mask(folder / options["mask_name"], rgb.shape[1], rgb.shape[0])
    else:
        new_mask = previous[:, :, 3] > 10
    core = _load_bool(folder / "core_mask.png", new_mask.shape, fallback=previous[:, :, 3] > 10)
    inpaint = _load_bool(folder / "inpaint_mask.png", new_mask.shape, fallback=np.zeros(new_mask.shape, dtype=bool))
    added = new_mask & ~(previous[:, :, 3] > 10)
    known = ((core & new_mask) | added) & new_mask
    jobs.stage(job_id, "removing_human", "Applying the corrected mask to the original photograph")

    if options.get("regenerate"):
        jobs.stage(job_id, "reconstructing", "Regenerating the hidden cloth estimate")
        models.unload("segformer")
        models.unload("sam2")
        if parsed is not None:
            holes = find_occlusion_mask(parsed.labels, known, parsed.label_names)
        else:
            holes = np.zeros(known.shape, dtype=bool)
        holes = (holes | (inpaint & new_mask)) & ~known
        fill, method, message = garment_reconstruction.fill(rgb, known, holes, "lama")
        if method in {"lama", "opencv_telea", "pattern_continue"}:
            rgba = human_removal.compose(rgb, known, fill, holes)
            saved_holes = holes
        else:
            rgba = human_removal.compose(rgb, known)
            saved_holes = np.zeros(known.shape, dtype=bool)
            message = message or "No hidden region was regenerated."
    else:
        jobs.skip(job_id, "reconstructing", "The previous reconstruction was left as it is")
        remain = inpaint & new_mask & ~known
        rgba = human_removal.compose(rgb, known, previous[:, :, :3], remain)
        saved_holes = remain
        method = meta.get("reconstruction_method", "disabled")
        message = "Mask edits use original photograph pixels where you restored cloth. " + str(
            meta.get("reconstruction_message") or ""
        )

    jobs.stage(job_id, "refining_output", "Updating the export")
    notes = list(meta.get("notes") or [])
    notes.append("The garment mask was edited by hand.")
    if parsed is not None:
        labels = parsed.labels
        confidence_map = parsed.confidence
        license_note = parsed.license_note
        source = parsed.source
    else:
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        confidence_map = np.zeros(rgb.shape[:2], dtype=np.float32)
        license_note = meta.get("parser_license", "")
        source = meta.get("parser", "unknown")
    updated = _base_meta(
        result_id=result_id,
        image_id=meta["image_id"],
        job_id=meta.get("job_id", job_id),
        rgb=rgb,
        confidence=float(meta.get("confidence") or 0),
        low_confidence=bool(meta.get("low_confidence")),
        parser=source,
        parser_license=license_note,
        method=method,
        message=message.strip(),
        product_tag=meta.get("product_tag", "other"),
        garment_label=meta.get("garment_label", "Garment"),
        garment_ids=meta.get("garment_ids") or [],
        notes=notes,
    )
    updated["created_at"] = meta.get("created_at", updated["created_at"])
    updated["manually_edited"] = True
    parsed_for_masks = parsed or ParseResult(labels, confidence_map, source, {}, [], license_note)
    _persist(folder, rgba, rgb, known | saved_holes, known, saved_holes, parsed_for_masks, updated)
    jobs.succeed(job_id, result_id, "Updated garment export is ready")


def load_result(result_id: str) -> dict:
    path = _result_dir(result_id) / "meta.json"
    if not path.exists():
        raise AppError("That result could not be found.", 404)
    return json.loads(path.read_text(encoding="utf-8"))


def list_results() -> list[dict]:
    root = get_settings().results_dir
    items = []
    if not root.exists():
        return items
    for child in root.iterdir():
        path = child / "meta.json"
        if not path.exists():
            continue
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return items


def delete_result(result_id: str) -> None:
    """Remove one finished extraction and its job folder.

    The original upload is left in place, because another garment from the
    same photograph may still refer to it.
    """

    if jobs.result_in_progress(result_id):
        raise AppError("That extraction is still updating. Wait until it finishes, then delete it.", 409)
    meta = load_result(result_id)
    folder = _result_dir(result_id)
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise AppError("The extraction could not be deleted.", 500) from exc
    job_id = meta.get("job_id")
    if isinstance(job_id, str) and job_id:
        jobs.discard(job_id)


def update_product_tag(result_id: str, product_tag: str) -> dict:
    if product_tag not in PRODUCT_TAGS:
        raise AppError("Unknown product tag.")
    meta = load_result(result_id)
    meta["product_tag"] = product_tag
    meta["updated_at"] = utc_now()
    _write_json(_result_dir(result_id) / "meta.json", meta)
    return meta


def result_file(result_id: str, kind: str) -> tuple[Path, str]:
    mapping = {
        "transparent": ("preview_transparent.png", "image/png"),
        "white": ("preview_white.jpg", "image/jpeg"),
        "mask": ("mask.png", "image/png"),
        "thumb": ("thumbnail.jpg", "image/jpeg"),
        "original": ("original_preview.jpg", "image/jpeg"),
        "garment": ("garment.png", "image/png"),
        "garment_white": ("garment_white.jpg", "image/jpeg"),
    }
    if kind not in mapping:
        raise AppError("Unknown result file.", 404)
    name, media = mapping[kind]
    path = _result_dir(result_id) / name
    if not path.exists():
        raise AppError("That file is not ready yet.", 404)
    return path, media


def garment_overlay_path(job_id: str, garment_id: str) -> Path:
    safe = _safe_garment_id(garment_id)
    path = _job_dir(job_id) / "garments" / f"{safe}_overlay.jpg"
    if not path.exists():
        raise AppError("That garment overlay could not be found.", 404)
    return path


def _load_or_parse(job_id: str, rgb: np.ndarray, parser: str, reuse: bool) -> ParseResult:
    folder = _job_dir(job_id)
    if reuse:
        parsed = _read_parse(folder, rgb)
        if parsed is not None:
            jobs.stage(job_id, "detecting_person", "Reusing the saved human parse")
            return parsed
    label = {"auto": "auto", "segformer": "SegFormer clothes", "mediapipe": "MediaPipe multiclass"}.get(
        parser, parser
    )
    jobs.stage(job_id, "detecting_person", f"Separating clothing from the person with {label}")
    parsed = human_parser.parse(rgb, parser)
    _write_parse(folder, parsed)
    return parsed


def _write_parse(folder: Path, parsed: ParseResult) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_png(folder / "labels.png", parsed.labels.astype(np.uint8))
    save_png(folder / "confidence.png", np.clip(parsed.confidence * 255.0, 0, 255).astype(np.uint8))
    _write_json(
        folder / "parse_meta.json",
        {
            "source": parsed.source,
            "label_names": {str(key): value for key, value in parsed.label_names.items()},
            "notes": parsed.notes,
            "license_note": parsed.license_note,
            "height": int(parsed.labels.shape[0]),
            "width": int(parsed.labels.shape[1]),
        },
    )


def _read_parse(folder: Path, rgb: np.ndarray) -> ParseResult | None:
    meta_path = folder / "parse_meta.json"
    labels_path = folder / "labels.png"
    if not meta_path.exists() or not labels_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels = np.asarray(Image.open(labels_path).convert("L"), dtype=np.uint8)
    confidence_path = folder / "confidence.png"
    if confidence_path.exists():
        confidence = load_gray(confidence_path).astype(np.float32) / 255.0
    else:
        confidence = np.full(labels.shape[:2], 0.5, dtype=np.float32)
    height, width = rgb.shape[:2]
    if labels.shape != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    names = {int(key): value for key, value in meta.get("label_names", {}).items()}
    return ParseResult(
        labels=labels,
        confidence=confidence.astype(np.float32),
        source=meta.get("source", "unknown"),
        label_names=names,
        notes=list(meta.get("notes") or []),
        license_note=meta.get("license_note") or "",
    )


def _write_garment_files(job_id: str, rgb: np.ndarray, garments: list[GarmentCandidate]) -> None:
    folder = _job_dir(job_id) / "garments"
    folder.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    for item in garments:
        save_png(folder / f"{item.id}.png", (item.mask.astype(np.uint8)) * 255)
        overlay = render_overlay(rgb, item.mask)
        save_jpeg(folder / f"{item.id}_overlay.jpg", preview_rgb(overlay, settings.preview_max_side), quality=88)


def _public_garment(job_id: str, item: GarmentCandidate) -> dict:
    data = item.summary()
    data["overlay_url"] = f"/api/extractions/{job_id}/garments/{item.id}/overlay"
    return data


def _product_tag(options: dict, chosen: list[GarmentCandidate]) -> str:
    requested = options.get("product_tag")
    if requested:
        if requested not in PRODUCT_TAGS:
            raise AppError("Unknown product tag.")
        return requested
    tags = list(dict.fromkeys(item.product_tag for item in chosen))
    if len(tags) == 1:
        return tags[0]
    return "other"


def _base_meta(**fields) -> dict:
    result_id = fields["result_id"]
    image_id = fields["image_id"]
    now = utc_now()
    return {
        "id": result_id,
        "image_id": image_id,
        "job_id": fields["job_id"],
        "width": int(fields["rgb"].shape[1]),
        "height": int(fields["rgb"].shape[0]),
        "confidence": round(float(fields["confidence"]), 4),
        "low_confidence": bool(fields["low_confidence"]),
        "reliable": not bool(fields["low_confidence"]),
        "parser": fields["parser"],
        "parser_license": fields["parser_license"],
        "reconstruction_method": fields["method"],
        "reconstruction_message": fields["message"],
        "product_tag": fields["product_tag"],
        "garment_label": fields["garment_label"],
        "garment_ids": fields["garment_ids"],
        "notes": [note for note in fields["notes"] if note],
        "manually_edited": False,
        "created_at": now,
        "updated_at": now,
        "original_preview_url": f"/api/results/{result_id}/file?kind=original",
        "transparent_preview_url": f"/api/results/{result_id}/file?kind=transparent",
        "white_preview_url": f"/api/results/{result_id}/file?kind=white",
        "mask_url": f"/api/results/{result_id}/file?kind=mask",
        "thumbnail_url": f"/api/results/{result_id}/file?kind=thumb",
        "download_png_url": f"/api/results/{result_id}/download?variant=transparent&format=png",
        "download_jpeg_url": f"/api/results/{result_id}/download?variant=white&format=jpeg",
        "upload_preview_url": f"/api/uploads/{image_id}/file?kind=preview",
    }


def _persist(folder, rgba, rgb, garment, core, holes, parsed: ParseResult, meta: dict) -> None:
    image_exporter.write_previews(folder, rgba, rgb)
    image_exporter.write_masks(
        folder,
        garment=garment,
        core=core,
        holes=holes,
        labels=parsed.labels,
        confidence=parsed.confidence,
    )
    _write_json(
        folder / "parse_meta.json",
        {
            "source": parsed.source,
            "label_names": {str(key): value for key, value in parsed.label_names.items()},
            "notes": parsed.notes,
            "license_note": parsed.license_note,
            "height": int(parsed.labels.shape[0]),
            "width": int(parsed.labels.shape[1]),
        },
    )
    _write_json(folder / "meta.json", meta)


def _decode_mask(path: Path, width: int, height: int) -> np.ndarray:
    if not path.exists():
        raise AppError("The edited mask was not uploaded.")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.shape != (height, width):
        ratio = array.shape[1] / max(1, array.shape[0])
        expected = width / max(1, height)
        if abs(ratio - expected) > 0.03:
            raise AppError("The mask aspect ratio does not match the photograph.")
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
    return array > 127


def _load_bool(path: Path, shape: tuple[int, int], fallback: np.ndarray) -> np.ndarray:
    if not path.exists():
        return fallback.astype(bool)
    array = load_gray(path)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return array > 127


def _safe_garment_id(garment_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in garment_id)
    if not cleaned or cleaned != garment_id:
        raise AppError("Unknown garment id.", 404)
    return cleaned


def _job_dir(job_id: str) -> Path:
    return get_settings().jobs_dir / job_id


def _result_dir(result_id: str) -> Path:
    return get_settings().results_dir / result_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


# Imported for callers that run an image without the job queue.
def extract_to_folder(image_path: Path, output_dir: Path, parser: str = "auto", reconstruct: bool = True) -> dict:
    """CLI helper. Uses the largest garment and does not pause for a selection."""

    from app.services.image_loader import save_upload
    from app.utils.validation import decode_upload

    rgb, info = decode_upload(image_path.read_bytes(), image_path.name)
    image_id = uuid.uuid4().hex
    save_upload(image_id, rgb, info)
    job = jobs.create(
        image_id,
        {
            "task": "extract",
            "parser": parser,
            "reconstruct": reconstruct,
            "refine_sam2": False,
            "clean_background": True,
            "selection": "largest",
            "reuse_parse": False,
        },
    )
    execute_job(job["id"])
    finished = jobs.snapshot(job["id"])
    if finished["status"] != "completed" or not finished["result_id"]:
        raise AppError(finished["error"] or "Extraction did not finish.")
    meta = load_result(finished["result_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # The canonical files stay in the result directory. Copy the two deliverables.
    target_png = output_dir / "garment.png"
    target_jpg = output_dir / "garment_white.jpg"
    target_png.write_bytes((_result_dir(meta["id"]) / "garment.png").read_bytes())
    target_jpg.write_bytes((_result_dir(meta["id"]) / "garment_white.jpg").read_bytes())
    return meta

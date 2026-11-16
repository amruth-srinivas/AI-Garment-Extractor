"""Edge refinement with GrabCut, and optional SAM 2 when its weights are present.

GrabCut is constrained to a narrow band around the parser mask so it can follow
fabric edges without repainting the garment as a new design. SAM 2 is asked
only to correct that same band. Its mask cannot delete the confident core or
add background far from the clothes.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import get_settings
from app.errors import AppError
from app.models.model_manager import models, pick_device, release_cuda
from app.utils.image_utils import resize_nearest, resize_rgb, working_size
from app.utils.mask_utils import dilate, erode, remove_small_components

logger = logging.getLogger("garmentai.refine")


def _keep_related(refined: np.ndarray, garment: np.ndarray) -> np.ndarray:
    """Drop speckles, but keep every piece that still touches the parsed garment."""

    min_island = max(80, int(0.002 * refined.size))
    refined = remove_small_components(refined.astype(bool), min_island)
    count, labels, _, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), 8)
    if count <= 1:
        return refined
    touching = dilate(garment.astype(bool), 2)
    kept = np.zeros(refined.shape[:2], dtype=bool)
    for index in range(1, count):
        component = labels == index
        if (component & touching).any():
            kept |= component
    return kept if kept.any() else garment.astype(bool)


def refine_grabcut(rgb: np.ndarray, garment: np.ndarray) -> np.ndarray:
    garment = garment.astype(bool)
    if int(garment.sum()) < 200 or int((~garment).sum()) < 200:
        return garment

    settings = get_settings()
    work_h, work_w, scale = working_size(rgb.shape[1], rgb.shape[0], settings.grabcut_max_side)
    small_rgb = resize_rgb(rgb, work_w, work_h) if scale < 1.0 else rgb
    small = resize_nearest(garment.astype(np.uint8), work_w, work_h).astype(bool) if scale < 1.0 else garment
    core = erode(small, 4)
    band = dilate(small, 6)
    if int(core.sum()) < 30:
        return garment

    gc_mask = np.full(small.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[band] = cv2.GC_PR_BGD
    gc_mask[small] = cv2.GC_PR_FGD
    gc_mask[core] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(small_rgb), gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        logger.warning("GrabCut skipped: %s", exc)
        return garment

    refined_small = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    refined_small = (refined_small & dilate(small, 8)) | erode(small, 2)
    if scale < 1.0:
        refined = resize_nearest(refined_small.astype(np.uint8), rgb.shape[1], rgb.shape[0]).astype(bool)
        # Keep the full-resolution core, and only let GrabCut edit the edge band.
        refined = (refined & dilate(garment, 8)) | erode(garment, 2)
    else:
        refined = refined_small
    refined = _keep_related(refined, garment)
    if int(refined.sum()) < int(0.4 * garment.sum()):
        logger.info("GrabCut shrank the garment too far. Keeping the parser mask.")
        return garment
    return refined


def refine_sam2(rgb: np.ndarray, garment: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the refined mask and a note describing what SAM 2 did."""

    garment = garment.astype(bool)
    if int(garment.sum()) < 200:
        return garment, "SAM 2 was skipped because the garment mask is too small."

    work_h, work_w, scale = working_size(rgb.shape[1], rgb.shape[0], 1024)
    small_rgb = resize_rgb(rgb, work_w, work_h) if scale < 1.0 else rgb
    small_garment = (
        resize_nearest(garment.astype(np.uint8), work_w, work_h).astype(bool) if scale < 1.0 else garment
    )
    small_body = resize_nearest(body.astype(np.uint8), work_w, work_h).astype(bool) if scale < 1.0 else body
    points, point_labels = _prompt_points(small_garment, small_body)
    if len(points) < 2:
        return garment, "SAM 2 was skipped because the mask did not yield stable prompts."

    device = pick_device()
    try:
        mask_small, score = _run_sam(small_rgb, points, point_labels, str(device))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or str(device) == "cpu":
            raise
        logger.warning("SAM 2 ran out of GPU memory. Retrying on CPU.")
        release_cuda()
        models.unload("sam2")
        mask_small, score = _run_sam(small_rgb, points, point_labels, "cpu")
    except AppError:
        raise

    if scale < 1.0:
        sam_mask = resize_nearest(mask_small.astype(np.uint8), rgb.shape[1], rgb.shape[0]).astype(bool)
    else:
        sam_mask = mask_small

    band = dilate(garment, 18)
    core = erode(garment, 3)
    refined = (sam_mask & band) | core
    recall = float((refined & garment).sum()) / max(1, int(garment.sum()))
    if recall < 0.72 or int(refined.sum()) > int(2.2 * garment.sum()):
        return garment, (
            f"SAM 2 (score {score:.2f}) disagreed with the clothing parser, "
            "so the parser mask was kept."
        )
    refined = _keep_related(refined, garment)
    return refined, f"SAM 2.1 hiera-tiny adjusted the garment edge (quality score {score:.2f})."


def _prompt_points(garment: np.ndarray, body: np.ndarray) -> tuple[list[list[int]], list[int]]:
    positive = _peaks(garment, limit=5, radius=28)
    negative_body = _peaks(body & ~dilate(garment, 2), limit=4, radius=24)
    background = ~dilate(garment, 10)
    negative_bg = _peaks(background, limit=4, radius=36)
    points = positive + negative_body + negative_bg
    labels = [1] * len(positive) + [0] * (len(negative_body) + len(negative_bg))
    return points, labels


def _peaks(mask: np.ndarray, limit: int, radius: int) -> list[list[int]]:
    if int(mask.sum()) < 5:
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[int]] = []
    for _ in range(limit):
        index = int(np.argmax(distance))
        y, x = np.unravel_index(index, distance.shape)
        if distance[y, x] < 2:
            break
        points.append([int(x), int(y)])
        cv2.circle(distance, (int(x), int(y)), radius, 0, -1)
    return points


def _run_sam(rgb: np.ndarray, points: list[list[int]], point_labels: list[int], device: str):
    import torch
    from PIL import Image

    processor, model = models.sam2(device)
    image = Image.fromarray(rgb)
    inputs = processor(
        images=image,
        input_points=[[points]],
        input_labels=[[point_labels]],
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    pred = outputs.pred_masks
    pred = pred.detach().cpu()
    try:
        processed = processor.post_process_masks(pred, inputs["original_sizes"])
    except TypeError:
        processed = processor.post_process_masks(
            pred,
            inputs["original_sizes"],
            inputs.get("reshaped_input_sizes"),
        )
    masks = processed[0]
    array = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
    # Accept (objects, predictions, H, W) or (predictions, H, W) or (H, W).
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    scores = getattr(outputs, "iou_scores", None)
    score_list = [0.0]
    if scores is not None:
        score_list = np.asarray(scores.detach().cpu()).reshape(-1).tolist() or [0.0]
    if array.ndim == 2:
        chosen = array > 0
        score = float(score_list[0])
    else:
        count = array.shape[0]
        best_index = 0
        best_score = -1e9
        garment_hint = None
        for index in range(count):
            candidate = array[index] > 0
            score = float(score_list[index]) if index < len(score_list) else 0.0
            # Prefer the prompt score. Area is only a tie break against empty masks.
            rank = score + min(0.05, float(candidate.mean()))
            if rank > best_score and int(candidate.sum()) > 20:
                best_score = rank
                best_index = index
                garment_hint = candidate
        chosen = garment_hint if garment_hint is not None else array[0] > 0
        score = float(score_list[best_index]) if best_index < len(score_list) else 0.0
    del outputs, inputs
    release_cuda()
    return chosen.astype(bool), score


class MaskRefiner:
    """Tightens the clothing boundary without inventing a new garment."""

    def grabcut(self, rgb: np.ndarray, garment: np.ndarray) -> np.ndarray:
        return refine_grabcut(rgb, garment)

    def sam2(
        self,
        rgb: np.ndarray,
        garment: np.ndarray,
        body: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        return refine_sam2(rgb, garment, body)


mask_refiner = MaskRefiner()

"""Build a garment mask that excludes the person wearing it."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import get_settings
from app.constants import BODY_CLASS_NAMES
from app.utils.mask_utils import (
    class_mask,
    dilate,
    erode,
    fill_small_nonbody_holes,
    ids_for,
    skin_pixels,
    smooth_mask,
)


def combine_garments(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    return combined


def separate_garment_from_body(
    rgb: np.ndarray,
    labels: np.ndarray,
    garment: np.ndarray,
    confidence: np.ndarray,
    label_names: dict[int, str],
) -> np.ndarray:
    """Keep parser cloth pixels and drop face, hair, skin, limbs, and edge residue."""

    kept = garment.astype(bool).copy()
    body = class_mask(labels, ids_for(label_names, BODY_CLASS_NAMES))
    kept &= ~body

    hair = class_mask(labels, ids_for(label_names, {"Hair"}))
    if hair.any():
        # A couple of pixels of hair halo on the shoulder, not a hole through the print.
        kept &= ~dilate(hair, 2)

    face = class_mask(labels, ids_for(label_names, {"Face", "Face-skin"}))
    if face.any():
        kept &= ~dilate(face, 2)
        kept = _clear_neck(rgb, kept, face, confidence)

    kept = _clear_skin_touching_body(rgb, kept, body, confidence)
    kept = _clear_small_boundary_skin(rgb, kept)
    kept = fill_small_nonbody_holes(kept, labels, label_names)
    min_island = max(40, int(0.0004 * kept.shape[0] * kept.shape[1]))
    kept = smooth_mask(kept, min_island)
    return kept


def _clear_neck(
    rgb: np.ndarray,
    garment: np.ndarray,
    face: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Drop a low-confidence skin-coloured neck under the face.

    A high-confidence turtleneck stays, because those pixels are cloth.
    """

    ys, xs = np.where(face)
    if len(xs) == 0:
        return garment
    face_bottom = int(ys.max())
    face_left, face_right = int(xs.min()), int(xs.max())
    face_height = max(1, int(ys.max() - ys.min() + 1))
    band_bottom = min(garment.shape[0], face_bottom + max(4, int(face_height * 0.22)))
    rows = np.arange(garment.shape[0])[:, None]
    cols = np.arange(garment.shape[1])[None, :]
    region = (
        (rows >= face_bottom)
        & (rows <= band_bottom)
        & (cols >= face_left - 4)
        & (cols <= face_right + 4)
    )
    skin = skin_pixels(rgb)
    drop = region & skin & garment & (confidence < 0.72)
    garment = garment.copy()
    garment[drop] = False
    return garment


def _clear_skin_touching_body(
    rgb: np.ndarray,
    garment: np.ndarray,
    body: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Remove skin-coloured residue on the boundary next to a body label.

    Beige and brown cloth away from the body is left alone, including the
    interior of the garment.
    """

    if not body.any() or not garment.any():
        return garment
    touching = dilate(body, 8) & garment
    drop = touching & skin_pixels(rgb) & (confidence < 0.62)
    garment = garment.copy()
    garment[drop] = False
    return garment


def _clear_small_boundary_skin(rgb: np.ndarray, garment: np.ndarray) -> np.ndarray:
    """Drop fingertips and similar skin that the parser folded into the cloth.

    Only a small skin-coloured component that touches the silhouette is removed.
    A skin-coloured print, or a large skin-toned panel, stays because it is either
    inside the garment or too big to be a hand.
    """

    garment = garment.astype(bool)
    if int(garment.sum()) < 50:
        return garment
    skin = skin_pixels(rgb) & garment
    if not skin.any():
        return garment
    boundary = garment & ~erode(garment, 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
    drop = np.zeros(garment.shape, dtype=bool)
    limit = max(60, int(0.035 * int(garment.sum())))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area == 0 or area > limit:
            continue
        component = labels == index
        if (component & boundary).any():
            drop |= component
    if not drop.any():
        return garment
    return garment & ~dilate(drop, 1)


def mask_confidence(confidence: np.ndarray, mask: np.ndarray) -> float:
    pixels = confidence[mask]
    if pixels.size == 0:
        return 0.0
    return float(pixels.mean())


def confidence_is_low(score: float) -> bool:
    return score < get_settings().low_confidence


class SegmentationService:
    """Unions the selected garments and removes body pixels from that mask."""

    def build(
        self,
        rgb: np.ndarray,
        labels: np.ndarray,
        garment: np.ndarray,
        confidence: np.ndarray,
        label_names: dict[int, str],
    ) -> np.ndarray:
        return separate_garment_from_body(rgb, labels, garment, confidence, label_names)


segmentation = SegmentationService()

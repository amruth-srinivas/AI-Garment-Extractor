"""Mask geometry used by detection, cleanup, and occlusion finding."""

from __future__ import annotations

import cv2
import numpy as np

from app.constants import BODY_CLASS_NAMES, OCCLUDER_CLASS_NAMES


def ids_for(label_names: dict[int, str], wanted: set[str]) -> list[int]:
    return [index for index, name in label_names.items() if name in wanted]


def class_mask(labels: np.ndarray, label_ids: list[int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(labels.shape[:2], dtype=bool)
    return np.isin(labels, np.asarray(label_ids, dtype=labels.dtype))


def odd_size(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def ellipse(width: int, height: int) -> np.ndarray:
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (odd_size(width, 1), odd_size(height, 1)),
    )


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    kernel = ellipse(pixels * 2 + 1, pixels * 2 + 1)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    kernel = ellipse(pixels * 2 + 1, pixels * 2 + 1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return eroded.astype(bool)


def close(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    kernel = ellipse(width, height)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    kept = np.zeros(mask.shape[:2], dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == index] = True
    return kept


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    winner = 1 + int(np.argmax(areas))
    return labels == winner


def grouped_components(mask: np.ndarray, dilate_px: int, min_area: int) -> list[np.ndarray]:
    """Split a class mask into pieces, joining fragments a few pixels apart."""

    mask = mask.astype(bool)
    if int(mask.sum()) < min_area:
        return []
    grouping = dilate(mask, dilate_px) if dilate_px > 0 else mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouping.astype(np.uint8), 8)
    pieces: list[np.ndarray] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        piece = (labels == index) & mask
        if int(piece.sum()) >= min_area:
            pieces.append(piece)
    pieces.sort(key=lambda item: int(item.sum()), reverse=True)
    return pieces


def component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fill_small_nonbody_holes(
    mask: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str],
    max_hole_area: int = 120,
) -> np.ndarray:
    """Fill tiny segmentation gaps that are not body pixels.

    Body-sized holes are left open so reconstruction can target them instead
    of copying skin into the garment.
    """

    body_ids = ids_for(label_names, BODY_CLASS_NAMES)
    inverted = (~mask.astype(bool)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(inverted, 8)
    filled = mask.astype(bool).copy()
    height, width = mask.shape[:2]
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area == 0 or area > max_hole_area:
            continue
        x, y, w, h, _ = stats[index]
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if touches_border:
            continue
        hole = components == index
        if body_ids and float(np.isin(labels[hole], body_ids).mean()) > 0.4:
            continue
        filled[hole] = True
    return filled


def smooth_mask(mask: np.ndarray, min_island: int) -> np.ndarray:
    """Close single-pixel gaps. Thin straps are not opened away."""

    kernel = ellipse(3, 3)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return remove_small_components(closed.astype(bool), min_island)


def skin_pixels(rgb: np.ndarray) -> np.ndarray:
    """Skin test that does not treat yellow or green cloth as skin.

    Red must sit clearly above green, which is true of skin and not of the
    yellow garments this tool is often asked to keep. Callers still limit the
    test to a boundary or a body label so a skin-toned fabric panel survives.
    """

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 170, 130], dtype=np.uint8)
    in_range = cv2.inRange(ycrcb, lower, upper).astype(bool)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    ordered = (red > green + 12) & (green > blue + 5) & (red > 70)
    return in_range & ordered


def _chest_center(garment: np.ndarray, top: int, span: int) -> float:
    """Center of the neck opening, or of the cloth when there is no opening."""

    height = garment.shape[0]
    gap_centers = []
    for y in range(top, min(height, top + max(8, int(span * 0.22)))):
        cols = np.flatnonzero(garment[y])
        if cols.size < 8:
            continue
        breaks = np.where(np.diff(cols) > 40)[0]
        if breaks.size:
            left = int(cols[int(breaks[0])])
            right = int(cols[int(breaks[0]) + 1])
            gap_centers.append(0.5 * (left + right))
    if gap_centers:
        return float(np.median(gap_centers))
    centers = []
    for y in range(top + int(span * 0.3), top + int(span * 0.6)):
        cols = np.flatnonzero(garment[y])
        if cols.size > 40:
            centers.append(0.5 * (int(cols[0]) + int(cols[-1])))
    if centers:
        return float(np.median(centers))
    xs = np.where(garment)[1]
    return float(np.median(xs)) if xs.size else 0.0


def mirrored_sleeve(garment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a missing flutter sleeve from the sleeve that is still visible.

    Returns the new pixels and, for each of them, the source column on the
    same row. A bare arm beside the skirt is not a sleeve and is not copied.
    """

    height, width = garment.shape
    holes = np.zeros((height, width), dtype=bool)
    source_x = np.full((height, width), -1, dtype=np.int32)
    ys = np.where(garment)[0]
    if ys.size < 80:
        return holes, source_x
    top = int(ys.min())
    span = int(ys.max() - top + 1)
    center = _chest_center(garment, top, span)
    sleeve_bottom = top + max(8, int(span * 0.17))
    right_ext: list[float] = []
    left_ext: list[float] = []
    for y in range(top, min(height, sleeve_bottom)):
        cols = np.flatnonzero(garment[y])
        if cols.size == 0:
            continue
        right_ext.append(float(int(cols[-1]) - center))
        left_ext.append(float(center - int(cols[0])))
    if not right_ext:
        return holes, source_x
    right_med = float(np.median(right_ext))
    left_med = float(np.median(left_ext))
    if abs(right_med - left_med) < 48:
        return holes, source_x
    mirror_right = right_med > left_med
    band: list[int] = []
    band_end = min(height, sleeve_bottom + max(4, int(span * 0.06)))
    for y in range(sleeve_bottom, band_end):
        cols = np.flatnonzero(garment[y])
        if cols.size:
            band.append(int(cols[-1] if mirror_right else cols[0]))
    if not band:
        return holes, source_x
    bodice_edge = int(np.median(band))
    anchors: list[int] = []
    for y in range(top, min(height, top + int(span * 0.22))):
        cols = np.flatnonzero(garment[y])
        if cols.size == 0:
            continue
        edge = int(cols[0] if mirror_right else cols[-1])
        if mirror_right and edge < center - 30:
            anchors.append(edge)
        elif not mirror_right and edge > center + 30:
            anchors.append(edge)
    if not anchors:
        return holes, source_x
    # The shoulder corner is the edge closest to the neck.
    anchor = int(max(anchors) if mirror_right else min(anchors))
    overlap = 8
    for y in range(top, min(height, sleeve_bottom + 1)):
        cols = np.flatnonzero(garment[y])
        if cols.size == 0:
            continue
        if mirror_right:
            sleeve_cols = cols[cols >= bodice_edge - 2]
        else:
            sleeve_cols = cols[cols <= bodice_edge + 2]
        for x in sleeve_cols:
            if mirror_right:
                dest = int(anchor + overlap - (int(x) - bodice_edge))
            else:
                dest = int(anchor - overlap + (bodice_edge - int(x)))
            if dest < 0 or dest >= width or garment[y, dest]:
                continue
            holes[y, dest] = True
            source_x[y, dest] = int(x)
        placed = np.flatnonzero(holes[y])
        if placed.size == 0:
            continue
        holes[y, int(placed[0]) : int(placed[-1]) + 1] = True
        known = np.flatnonzero(source_x[y] >= 0)
        if known.size == 0:
            continue
        for x in range(int(placed[0]), int(placed[-1]) + 1):
            if source_x[y, x] < 0:
                nearest = known[int(np.argmin(np.abs(known - x)))]
                source_x[y, x] = source_x[y, nearest]
    holes &= ~garment
    source_x[~holes] = -1
    return holes, source_x


def _bracketed(garment: np.ndarray, holes: np.ndarray, reach: int = 36) -> np.ndarray:
    """Keep a hole only when cloth sits on both sides of it, close by."""

    kept = np.zeros(holes.shape, dtype=bool)
    for y in np.where(holes.any(axis=1))[0]:
        cloth = np.flatnonzero(garment[y])
        if cloth.size < 2:
            continue
        for x in np.flatnonzero(holes[y]):
            left = cloth[cloth < x]
            right = cloth[cloth > x]
            if left.size == 0 or right.size == 0:
                continue
            if int(x - left[-1]) <= reach and int(right[0] - x) <= reach:
                kept[y, x] = True
    return kept


def find_occlusion_mask(
    labels: np.ndarray,
    garment: np.ndarray,
    label_names: dict[int, str],
) -> np.ndarray:
    """Cloth hidden by hair on a missing sleeve, or by a limb lying on the cloth.

    A sleeve that was covered by hair is copied from the sleeve that is still
    visible. An arm or a hand that only stands beside the cloth is left out,
    because painting it adds a ribbon next to the garment. The face and the
    neckline are left open.
    """

    garment = garment.astype(bool)
    if int(garment.sum()) < 50:
        return np.zeros(garment.shape, dtype=bool)

    ys, xs = np.where(garment)
    top = int(ys.min())
    bottom = int(ys.max())
    garment_width = int(xs.max() - xs.min() + 1)
    garment_height = int(bottom - top + 1)
    # Small enough to fill a limb lying on the cloth, and too small to close
    # a square neckline.
    kernel_w = odd_size(min(21, int(garment_width * 0.08)), 9)
    kernel_h = odd_size(min(15, int(garment_height * 0.025)), 7)

    occluder_ids = ids_for(label_names, OCCLUDER_CLASS_NAMES)
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])[None, :]
    in_span = (rows >= top) & (rows <= bottom)
    hair_ids = ids_for(label_names, {"Hair"})
    hair = class_mask(labels, hair_ids) if hair_ids else np.zeros(garment.shape, dtype=bool)
    limb_ids = ids_for(label_names, {"Left-arm", "Right-arm", "Body-skin"})
    limbs = class_mask(labels, limb_ids) if limb_ids else np.zeros(garment.shape, dtype=bool)
    sleeve, _source = mirrored_sleeve(garment)
    interior = np.zeros(garment.shape, dtype=bool)
    if occluder_ids:
        closed = close(garment, kernel_w, kernel_h)
        interior = _bracketed(garment, closed & ~garment & in_span & dilate(limbs | hair, 1))
    outside = (~garment).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    holes = (sleeve | interior) & ~garment
    face = class_mask(labels, ids_for(label_names, {"Face", "Face-skin"}))
    if face.any():
        holes &= ~dilate(face, 3)
        fys, fxs = np.where(face)
        face_height = max(1, int(fys.max() - fys.min() + 1))
        neck_bottom = min(labels.shape[0] - 1, int(fys.max()) + max(4, int(face_height * 0.28)))
        neck = (
            (rows >= int(fys.max()))
            & (rows <= neck_bottom)
            & (cols >= int(fxs.min()))
            & (cols <= int(fxs.max()))
        )
        holes &= ~neck
    # The parser's hair and skin halo is a couple of pixels of cloth that was
    # cleared off the edge. Pull that back into the fill, not into the backdrop.
    halo = dilate(holes, 2) & ~garment & (distance <= 3.0)
    holes = holes | halo
    holes = remove_small_components(holes, max(24, int(0.00025 * garment.size)))
    garment_area = int(garment.sum())
    if int(holes.sum()) > int(0.45 * garment_area):
        # A hole this large means the parser failed, not that half the
        # garment was hidden. Refuse to invent that much cloth.
        return np.zeros(garment.shape, dtype=bool)
    return holes


def soft_alpha(mask: np.ndarray) -> np.ndarray:
    """Binary interior with a one-pixel soft edge. The print itself is not blurred."""

    binary = mask.astype(bool)
    if not binary.any():
        return np.zeros(mask.shape[:2], dtype=np.float32)
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 0.7)
    alpha = np.clip(blurred, 0.0, 1.0)
    core = erode(binary, 1)
    outer = dilate(binary, 2)
    alpha[core] = 1.0
    alpha[~outer] = 0.0
    return alpha


def scale_bbox(
    bbox: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    if scale == 1.0:
        return bbox
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 / scale)),
        int(round(y0 / scale)),
        int(round(x1 / scale)),
        int(round(y1 / scale)),
    )

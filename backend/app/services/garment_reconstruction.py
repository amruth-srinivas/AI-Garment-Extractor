"""Reconstruct only the cloth hidden by the body.

Original garment pixels are never replaced. LaMa (or Telea, if the LaMa
weights cannot be loaded) writes into the occlusion mask, and a light colour
match pulls the fill toward the surrounding cloth. The fill is an estimate.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.errors import AppError
from app.models.model_manager import models, pick_device, release_cuda
from app.utils.mask_utils import dilate, mirrored_sleeve

logger = logging.getLogger("garmentai.reconstruct")

LAMA_MESSAGE = (
    "Hidden cloth was estimated with LaMa from the surrounding visible fabric. "
    "It is not a recovery of the original hidden textile, and it can miss a "
    "plaid, stripe, or print that was fully covered."
)
PATTERN_MESSAGE = (
    "Cloth hidden by hair was continued from the sleeve that is still visible. "
    "It is an estimate of the covered area, not the textile that was behind the "
    "hair, and a plaid or print can shift. A hand or arm that only rests beside "
    "the cloth is left out, so the garment does not grow a ribbon."
)
TELEA_MESSAGE = (
    "Hidden cloth was estimated with OpenCV Telea inpainting because the LaMa "
    "weights were not available. This is a classical fill, not the original fabric."
)
DISABLED_MESSAGE = "Reconstruction is off. Areas hidden by the body stay transparent."
LOW_CONFIDENCE_MESSAGE = (
    "Reconstruction was skipped because the garment mask confidence is low. "
    "Correct the mask, then regenerate the hidden areas if you want a fill."
)


def reconstruct(
    original: np.ndarray,
    garment: np.ndarray,
    holes: np.ndarray,
    method: str = "lama",
) -> tuple[np.ndarray, str, str]:
    holes = holes.astype(bool) & ~garment.astype(bool)
    if not holes.any():
        return (
            original.copy(),
            "none",
            "No occluded garment region was found, so nothing was generated.",
        )
    if method == "disabled":
        return original.copy(), "disabled", DISABLED_MESSAGE

    if method == "opencv_telea":
        filled = _inpaint_holes(original, holes, garment, engine="opencv_telea")
        return filled, "opencv_telea", TELEA_MESSAGE

    try:
        filled = _paint_covered_cloth(original, garment.astype(bool), holes)
        return filled, "pattern_continue", PATTERN_MESSAGE
    except (ValueError, cv2.error) as exc:
        logger.warning("Pattern continuation failed (%s). Using LaMa.", exc)
    try:
        filled = _inpaint_holes(original, holes, garment, engine="lama")
        return filled, "lama", LAMA_MESSAGE
    except AppError as exc:
        logger.warning("LaMa unavailable (%s). Using Telea.", exc.message)
        filled = _inpaint_holes(original, holes, garment, engine="opencv_telea")
        return filled, "opencv_telea", TELEA_MESSAGE + " " + exc.message


def _inpaint_holes(
    original: np.ndarray,
    holes: np.ndarray,
    garment: np.ndarray,
    engine: str,
) -> np.ndarray:
    output = original.copy()
    count, components, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    order = list(range(1, count))
    order.sort(key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
    for index in order[:12]:
        component = components == index
        if int(component.sum()) < 30:
            continue
        output = _inpaint_component(output, component, engine, garment)
    _match_ring_colour(output, original, holes)
    # Put the original cloth back even if a crop overlapped it.
    output[garment_guard(original, holes)] = original[garment_guard(original, holes)]
    return output


def _paint_covered_cloth(original: np.ndarray, garment: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Copy the visible sleeve into a missing one, then continue other holes."""

    filled = original.copy()
    _sleeve_holes, source = mirrored_sleeve(garment)
    copied = holes & (source >= 0)
    if copied.any():
        ys, xs = np.where(copied)
        filled[ys, xs] = original[ys, source[ys, xs]]
    remain = holes & ~copied
    if remain.any():
        continued = _continue_from_cloth(original, garment, remain)
        filled[remain] = continued[remain]
    if not copied.any() and not remain.any():
        raise ValueError("Nothing to continue from the visible cloth.")
    return filled


def _continue_from_cloth(original: np.ndarray, garment: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Copy real cloth from the same row into areas a body part covered.

    The sample is a run of garment pixels in from the edge, so the pale
    fringe and a neck opening are not repeated into the fill.
    """

    if int(garment.sum()) < 80:
        raise ValueError("The garment has no pixels to continue from.")
    filled = original.copy()
    rows = np.where(holes.any(axis=1))[0]
    for y in rows:
        cloth_x = np.flatnonzero(garment[y])
        if cloth_x.size < 20:
            continue
        hole_x = np.flatnonzero(holes[y])
        left_edge = int(cloth_x[0])
        right_edge = int(cloth_x[-1])
        period = 36
        fringe = 18
        left_src = cloth_x[fringe : fringe + period]
        right_src = cloth_x[::-1][fringe : fringe + period]
        if left_src.size < 8:
            left_src = cloth_x[:period]
        if right_src.size < 8:
            right_src = cloth_x[::-1][:period]
        on_left = hole_x <= (left_edge + right_edge) // 2
        # Tile a run of real cloth pixels. Indexing the garment columns
        # skips the neck opening, so the sample cannot land on skin.
        left_i = np.abs(hole_x - left_edge) % left_src.size
        right_i = np.abs(hole_x - right_edge) % right_src.size
        sample = np.where(on_left, left_src[left_i], right_src[right_i])
        filled[y, hole_x] = original[y, sample]
    return filled


def garment_guard(original: np.ndarray, holes: np.ndarray) -> np.ndarray:
    # Caller restores non-hole pixels explicitly.
    return ~holes.astype(bool)


def _inpaint_component(
    image: np.ndarray,
    hole: np.ndarray,
    engine: str,
    garment: np.ndarray | None = None,
) -> np.ndarray:
    ys, xs = np.where(hole)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(0.55 * (y1 - y0)) + 24
    pad_x = int(0.55 * (x1 - x0)) + 24
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(image.shape[0], y1 + pad_y)
    x1 = min(image.shape[1], x1 + pad_x)
    crop = image[y0:y1, x0:x1]
    crop_hole = hole[y0:y1, x0:x1]
    garment_crop = None if garment is None else garment[y0:y1, x0:x1]
    # The photograph still shows the person. Inpainting that image redraws
    # skin and hair. The context keeps original cloth only, so the fill
    # continues the fabric instead of the body.
    context = np.full_like(crop, 255)
    if garment_crop is not None:
        cloth = garment_crop.astype(bool) & ~crop_hole
        context[cloth] = crop[cloth]
    else:
        context[~crop_hole] = crop[~crop_hole]
    filled_crop = _run_engine(context, crop_hole, engine)
    filled_crop = _match_local_colour(crop, filled_crop, crop_hole, garment_crop)
    image = image.copy()
    region = image[y0:y1, x0:x1]
    region[crop_hole] = filled_crop[crop_hole]
    image[y0:y1, x0:x1] = region
    return image


def _run_engine(crop: np.ndarray, hole: np.ndarray, engine: str) -> np.ndarray:
    settings = get_settings()
    long_side = max(crop.shape[0], crop.shape[1])
    scale = 1.0
    view = crop
    view_hole = hole
    if engine == "lama" and long_side > settings.lama_max_side:
        scale = settings.lama_max_side / float(long_side)
        new_w = max(8, int(round(crop.shape[1] * scale)))
        new_h = max(8, int(round(crop.shape[0] * scale)))
        view = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        view_hole = cv2.resize(hole.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    if engine == "lama":
        filled_view = _lama(view, view_hole)
    else:
        mask = view_hole.astype(np.uint8) * 255
        filled_view = cv2.inpaint(view, mask, 3, cv2.INPAINT_TELEA)

    if scale != 1.0:
        filled = cv2.resize(
            filled_view,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        filled = filled_view
    return filled


def _lama(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    import torch

    device = pick_device()
    try:
        return _lama_on(rgb, hole, str(device))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or str(device) == "cpu":
            raise
        logger.warning("LaMa ran out of GPU memory. Retrying on CPU.")
        release_cuda()
        models.unload("lama")
        return _lama_on(rgb, hole, "cpu")


def _lama_on(rgb: np.ndarray, hole: np.ndarray, device: str) -> np.ndarray:
    import torch

    model = models.lama(device)
    image_t, mask_t = _prepare_lama(rgb, hole, device)
    with torch.inference_mode():
        output = model(image_t, mask_t)
    array = output[0].permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    array = array[: rgb.shape[0], : rgb.shape[1]]
    del output, image_t, mask_t
    release_cuda()
    return array


def _prepare_lama(rgb: np.ndarray, hole: np.ndarray, device: str):
    import torch

    image = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    mask = (hole.astype(np.float32) > 0).astype(np.float32)[None, ...]
    image = _pad_modulo(image, 8)
    mask = _pad_modulo(mask, 8)
    image_t = torch.from_numpy(image).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    mask_t = (mask_t > 0).float()
    return image_t, mask_t


def _pad_modulo(array: np.ndarray, modulo: int) -> np.ndarray:
    channels, height, width = array.shape
    out_h = height if height % modulo == 0 else (height // modulo + 1) * modulo
    out_w = width if width % modulo == 0 else (width // modulo + 1) * modulo
    return np.pad(
        array,
        ((0, 0), (0, out_h - height), (0, out_w - width)),
        mode="symmetric",
    )


def _match_local_colour(
    original: np.ndarray,
    filled: np.ndarray,
    hole: np.ndarray,
    garment: np.ndarray | None = None,
) -> np.ndarray:
    ring = dilate(hole, 6) & ~hole
    if garment is not None:
        cloth = ring & garment.astype(bool)
        if int(cloth.sum()) >= 40:
            ring = cloth
    if int(ring.sum()) < 40 or int(hole.sum()) < 20:
        return filled
    source = original[ring].astype(np.float32).mean(axis=0)
    current = filled[hole].astype(np.float32).mean(axis=0)
    delta = (source - current) * 0.45
    adjusted = filled.copy()
    adjusted[hole] = np.clip(filled[hole].astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return adjusted


def _match_ring_colour(image: np.ndarray, original: np.ndarray, holes: np.ndarray) -> None:
    # Colour is matched around each hole. A second global recolour would shift
    # the original print, so this hook does not modify the garment.
    return None


class GarmentReconstruction:
    """Fills body-occluded holes only. Original garment pixels stay untouched."""

    def fill(
        self,
        original: np.ndarray,
        garment: np.ndarray,
        holes: np.ndarray,
        method: str = "lama",
    ) -> tuple[np.ndarray, str, str]:
        return reconstruct(original, garment, holes, method)


garment_reconstruction = GarmentReconstruction()

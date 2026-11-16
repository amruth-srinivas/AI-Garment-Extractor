"""Residue cleanup after the garment has already been separated from the body.

rembg is not the extractor. It only drops low-confidence edge pixels that both
sit on the outer rim of the clothing mask and are labelled background by u2net.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.config import get_settings
from app.errors import AppError
from app.models.model_manager import models
from app.utils.image_utils import resize_nearest, resize_rgb, working_size
from app.utils.mask_utils import erode

logger = logging.getLogger("garmentai.background")


def trim_background(
    rgb: np.ndarray,
    garment: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, str]:
    garment = garment.astype(bool)
    if int(garment.sum()) < 50:
        return garment, "Background cleanup was skipped because the garment mask is empty."
    try:
        from rembg import remove

        session = models.rembg_session()
    except AppError as exc:
        return garment, f"Background residue cleanup was skipped. {exc.message}"
    except Exception as exc:
        logger.warning("rembg unavailable: %s", exc)
        return garment, "Background residue cleanup was skipped because rembg could not start."

    settings = get_settings()
    work_h, work_w, scale = working_size(rgb.shape[1], rgb.shape[0], 1024)
    small = resize_rgb(rgb, work_w, work_h) if scale < 1.0 else rgb
    try:
        cutout = remove(Image.fromarray(small), session=session)
    except Exception as exc:
        logger.warning("rembg failed: %s", exc)
        return garment, "Background residue cleanup failed, so the clothing mask was left unchanged."

    alpha = np.asarray(cutout)
    if alpha.ndim != 3 or alpha.shape[2] < 4:
        return garment, "Background cleanup did not return an alpha channel."
    foreground = alpha[:, :, 3] > 30
    if scale < 1.0:
        foreground = resize_nearest(foreground.astype(np.uint8), rgb.shape[1], rgb.shape[0]).astype(bool)

    outer = garment & ~erode(garment, 5)
    drop = outer & ~foreground & (confidence < 0.70)
    if int(drop.sum()) > int(0.12 * max(1, garment.sum())):
        return garment, (
            "Background cleanup would have removed too much of the garment, "
            "so it was ignored."
        )
    cleaned = garment & ~drop
    removed = int((garment & ~cleaned).sum())
    if removed == 0:
        return cleaned, "Background cleanup found no residue on the garment edge."
    return cleaned, (
        f"Background cleanup removed {removed} low-confidence edge pixels "
        "that were outside the clothing."
    )


class BackgroundRemoval:
    def trim(
        self,
        rgb: np.ndarray,
        garment: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        return trim_background(rgb, garment, confidence)


background_removal = BackgroundRemoval()

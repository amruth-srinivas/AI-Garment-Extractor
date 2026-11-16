"""Composite the original garment pixels and drop the person.

Reconstruction may supply colours only for the occlusion mask. Every other
kept pixel is copied from the photograph, so the print, stitching, and colour
are the original image.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.utils.mask_utils import erode, soft_alpha


def compose_garment(
    original: np.ndarray,
    keep: np.ndarray,
    fill_rgb: np.ndarray | None = None,
    holes: np.ndarray | None = None,
) -> np.ndarray:
    height, width = original.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    keep = keep.astype(bool)
    hole_mask = np.zeros((height, width), dtype=bool)
    if holes is not None and fill_rgb is not None:
        hole_mask = holes.astype(bool) & ~keep
    rgba[keep, :3] = original[keep]
    rgba[keep, 3] = 255
    if fill_rgb is not None and hole_mask.any():
        rgba[hole_mask, :3] = fill_rgb[hole_mask]
        rgba[hole_mask, 3] = 255
    covered = keep | hole_mask
    alpha = soft_alpha(covered)
    rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    # Interior of the kept cloth stays fully opaque. Only the outer edge is soft.
    interior = erode(covered, 1)
    rgba[interior, 3] = 255
    return defringe(rgba)


def defringe(rgba: np.ndarray) -> np.ndarray:
    """Replace the thin edge halo with nearby garment colour.

    The fill is limited to pixels that are not fully opaque, so the fabric
    interior is not filtered or smoothed.
    """

    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3]
    interior = alpha > 245
    edge = (alpha > 0) & ~erode(interior, 1)
    if int(edge.sum()) < 5 or int(interior.sum()) < 20:
        return rgba
    # Telea copies colour from the opaque cloth into the 1px fringe only.
    fringe = edge.astype(np.uint8) * 255
    filled = cv2.inpaint(rgb, fringe, 2, cv2.INPAINT_TELEA)
    rgb[edge] = filled[edge]
    out = rgba.copy()
    out[:, :, :3] = rgb
    return out


class HumanRemoval:
    """Copies original cloth pixels and drops every other pixel to transparency."""

    def compose(
        self,
        original: np.ndarray,
        keep: np.ndarray,
        fill_rgb: np.ndarray | None = None,
        holes: np.ndarray | None = None,
    ) -> np.ndarray:
        return compose_garment(original, keep, fill_rgb, holes)


human_removal = HumanRemoval()

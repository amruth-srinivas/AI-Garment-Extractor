"""Turn a human-parsing map into selectable garment regions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.constants import (
    ACCESSORY_CLASS_NAMES,
    GARMENT_CLASS_NAMES,
    TAG_BY_LABEL,
)
from app.utils.mask_utils import (
    class_mask,
    component_bbox,
    grouped_components,
    ids_for,
    largest_component,
)


@dataclass
class GarmentCandidate:
    id: str
    label: str
    product_tag: str
    confidence: float
    area_fraction: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "product_tag": self.product_tag,
            "confidence": round(float(self.confidence), 4),
            "area_fraction": round(float(self.area_fraction), 4),
            "bbox": list(self.bbox),
            "notes": list(self.notes),
        }


def _slug(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _mean_confidence(confidence: np.ndarray, mask: np.ndarray) -> float:
    pixels = confidence[mask]
    if pixels.size == 0:
        return 0.0
    return float(pixels.mean())


def _split_clothes_on_skin(
    clothes: np.ndarray,
    skin: np.ndarray,
) -> list[tuple[np.ndarray, str, str]]:
    """Split one clothes class into a top and a bottom when a skin gap exists."""

    clothes = largest_component(clothes)
    area = int(clothes.sum())
    if area < 500 or int(skin.sum()) < 50:
        return [(clothes, "Clothes", "dress" if _looks_like_dress(clothes) else "top")]

    ys, xs = np.where(clothes)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    height = max(1, y1 - y0 + 1)
    best: tuple[float, int] | None = None
    search_start = y0 + int(height * 0.28)
    search_end = y0 + int(height * 0.72)
    for y in range(search_start, search_end):
        cloth_row = float(clothes[y, x0:x1].mean()) if x1 > x0 else 1.0
        skin_row = float(skin[y, x0:x1].mean()) if x1 > x0 else 0.0
        if skin_row > 0.22 and cloth_row < 0.28:
            top_area = int(clothes[y0:y].sum())
            bottom_area = int(clothes[y + 1 : y1 + 1].sum())
            if top_area > 0.14 * area and bottom_area > 0.14 * area:
                score = skin_row - cloth_row
                if best is None or score > best[0]:
                    best = (score, y)
    if best is None:
        tag = "dress" if _looks_like_dress(clothes) else "top"
        return [(clothes, "Clothes", tag)]

    split_y = best[1]
    top = clothes.copy()
    top[split_y:] = False
    bottom = clothes.copy()
    bottom[:split_y] = False
    top = largest_component(top)
    bottom = largest_component(bottom)
    bottom_tag = _bottom_tag(bottom)
    return [
        (top, "Clothes", "top"),
        (bottom, "Clothes", bottom_tag),
    ]


def _looks_like_dress(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    return height > width * 1.15 and height > mask.shape[0] * 0.45


def _bottom_tag(mask: np.ndarray) -> str:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return "skirt"
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    # A narrow, long lower garment is more like trousers than a skirt.
    # This is a shape guess, not a trained jeans/trousers classifier.
    if height > width * 0.85:
        return "trousers"
    return "skirt"


def detect_garments(
    labels: np.ndarray,
    confidence: np.ndarray,
    label_names: dict[int, str],
    source: str,
) -> list[GarmentCandidate]:
    height, width = labels.shape[:2]
    image_area = float(height * width)
    min_area = max(400, int(0.004 * image_area))
    found: list[GarmentCandidate] = []

    garment_ids = ids_for(label_names, GARMENT_CLASS_NAMES)
    if source == "mediapipe":
        clothes = class_mask(labels, ids_for(label_names, {"Clothes"}))
        skin = class_mask(labels, ids_for(label_names, {"Body-skin", "Face-skin"}))
        pieces = _split_clothes_on_skin(clothes, skin) if clothes.any() else []
        for index, (mask, label, tag) in enumerate(pieces):
            if int(mask.sum()) < min_area:
                continue
            notes = []
            if len(pieces) > 1:
                notes.append(
                    "Split from the single MediaPipe clothes class at a skin gap. "
                    "Confirm the piece before exporting."
                )
            else:
                notes.append(
                    "MediaPipe does not distinguish a dress from a saree, kurti, "
                    "lehenga, or long top. The product tag is a shape suggestion."
                )
            conf = _mean_confidence(confidence, mask)
            if len(pieces) > 1:
                conf *= 0.8
            found.append(
                _candidate(f"clothes_{index}", label, tag, conf, mask, image_area, notes)
            )
    else:
        for label_id in garment_ids:
            name = label_names[label_id]
            pieces = grouped_components(labels == label_id, dilate_px=8, min_area=min_area)
            for index, mask in enumerate(pieces):
                tag = TAG_BY_LABEL.get(name, "other")
                notes = []
                if name == "Dress":
                    notes.append(
                        "The parser class is Dress. Saree, gown, kurti, and lehenga "
                        "are not separate classes; change the product tag if needed."
                    )
                if name == "Upper-clothes":
                    notes.append(
                        "The parser class is Upper-clothes. Shirt, t-shirt, jacket, "
                        "and coat share this class."
                    )
                if name == "Pants":
                    notes.append("The parser class is Pants. Jeans are not a separate class.")
                found.append(
                    _candidate(
                        f"{_slug(name)}_{index}",
                        name,
                        tag,
                        _mean_confidence(confidence, mask),
                        mask,
                        image_area,
                        notes,
                    )
                )

    accessory_ids = ids_for(label_names, ACCESSORY_CLASS_NAMES)
    for label_id in accessory_ids:
        name = label_names[label_id]
        pieces = grouped_components(labels == label_id, dilate_px=4, min_area=max(min_area, int(0.01 * image_area)))
        for index, mask in enumerate(pieces[:2]):
            found.append(
                _candidate(
                    f"{_slug(name)}_{index}",
                    name,
                    TAG_BY_LABEL.get(name, "other"),
                    _mean_confidence(confidence, mask),
                    mask,
                    image_area,
                    ["Accessory region, not selected by default."],
                )
            )

    # Shoes are a pair. If both feet were found, keep them as separate
    # optional regions; the UI does not preselect accessories.
    found.sort(key=lambda item: (item.label in ACCESSORY_CLASS_NAMES, -item.area_fraction))
    return found


def _candidate(identifier, label, tag, confidence, mask, image_area, notes) -> GarmentCandidate:
    return GarmentCandidate(
        id=identifier,
        label=label,
        product_tag=tag,
        confidence=float(confidence),
        area_fraction=float(mask.sum()) / image_area,
        bbox=component_bbox(mask),
        mask=mask.astype(bool),
        notes=notes,
    )


def person_fraction(labels: np.ndarray, label_names: dict[int, str]) -> float:
    background_ids = ids_for(label_names, {"Background"})
    if not background_ids:
        return 1.0
    background = class_mask(labels, background_ids)
    return float((~background).mean())


class GarmentDetector:
    """Finds selectable garment regions and reports a confidence for each."""

    def detect(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        label_names: dict[int, str],
        source: str,
    ) -> list[GarmentCandidate]:
        return detect_garments(labels, confidence, label_names, source)

    def choose(
        self,
        garments: list[GarmentCandidate],
        garment_ids: list[str] | None,
        selection: str,
    ) -> list[GarmentCandidate]:
        return choose_garments(garments, garment_ids, selection)


garment_detector = GarmentDetector()


def choose_garments(
    garments: list[GarmentCandidate],
    garment_ids: list[str] | None,
    selection: str,
) -> list[GarmentCandidate]:
    from app.errors import AppError

    primary = [item for item in garments if item.label not in ACCESSORY_CLASS_NAMES]
    pool = primary or garments
    if garment_ids:
        wanted = set(garment_ids)
        chosen = [item for item in garments if item.id in wanted]
        missing = wanted.difference(item.id for item in chosen)
        if missing:
            raise AppError("Unknown garment id: " + ", ".join(sorted(missing)))
        if not chosen:
            raise AppError("Select at least one detected garment.")
        return chosen
    if not pool:
        return []
    if selection == "largest" or len(pool) == 1:
        return [max(pool, key=lambda item: item.area_fraction)]
    return []

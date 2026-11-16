import numpy as np

from app.services.garment_detector import detect_garments
from app.services.segmentation import separate_garment_from_body
from app.utils.mask_utils import find_occlusion_mask

NAMES = {
    0: "Background",
    2: "Hair",
    7: "Dress",
    11: "Face",
    14: "Left-arm",
}


def _scene():
    labels = np.zeros((100, 80), dtype=np.uint8)
    labels[30:90, 20:60] = 7
    labels[40:70, 38:43] = 14
    labels[8:24, 30:50] = 11
    labels[50:58, 28:34] = 2
    garment = labels == 7
    confidence = np.full(labels.shape, 0.25, dtype=np.float32)
    confidence[garment] = 0.92
    return labels, garment, confidence


def test_occlusion_keeps_the_face_out_and_the_arm_in():
    labels, garment, _confidence = _scene()
    holes = find_occlusion_mask(labels, garment, NAMES)
    assert holes[55, 40]
    assert holes[54, 30]
    assert not holes[15, 40]
    assert not (holes & garment).any()


def test_occlusion_does_not_paint_an_arm_beside_the_cloth():
    labels = np.zeros((140, 100), dtype=np.uint8)
    labels[20:120, 25:75] = 7
    labels[50:90, 60:88] = 14
    labels[2:16, 40:60] = 11
    garment = labels == 7
    holes = find_occlusion_mask(labels, garment, NAMES)
    assert not holes[70, 80]
    assert not holes[10, 50]
    assert not (holes & garment).any()


def test_body_pixels_are_removed_from_the_garment():
    labels, garment, confidence = _scene()
    rgb = np.full((*labels.shape, 3), 235, dtype=np.uint8)
    rgb[garment] = (20, 120, 70)
    rgb[labels == 11] = (214, 164, 140)
    claimed = garment | (labels == 11)
    cleaned = separate_garment_from_body(rgb, labels, claimed, confidence, NAMES)
    assert not cleaned[15, 40]
    assert cleaned[80, 40]


def test_detector_reports_a_dress_with_confidence():
    labels, garment, confidence = _scene()
    found = detect_garments(labels, confidence, NAMES, "segformer")
    dresses = [item for item in found if item.label == "Dress"]
    assert len(dresses) == 1
    assert dresses[0].product_tag == "dress"
    assert dresses[0].confidence > 0.8
    assert dresses[0].mask[80, 40]
    assert not dresses[0].mask[15, 40]

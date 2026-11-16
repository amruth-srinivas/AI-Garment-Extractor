"""Labels, product tags, and processing stages shared by the API."""

from __future__ import annotations

# SegFormer B2 clothes (ATR) label ids. MediaPipe uses a different map,
# carried on each parse result rather than assumed here.
SEGFORMER_LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

MEDIAPIPE_LABELS = {
    0: "Background",
    1: "Hair",
    2: "Body-skin",
    3: "Face-skin",
    4: "Clothes",
    5: "Others",
}

# Classes that are garments the user can extract.
GARMENT_CLASS_NAMES = {
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Scarf",
    "Clothes",
}

# Optional accessories. Shown when they cover enough of the photo, never
# auto-selected ahead of a real garment.
ACCESSORY_CLASS_NAMES = {
    "Hat",
    "Belt",
    "Bag",
    "Left-shoe",
    "Right-shoe",
    "Sunglasses",
    "Others",
}

# Body regions subtracted from the garment. Face is removed but not treated
# as cloth to reconstruct. Arms, legs, skin, and hair that sit inside the
# garment silhouette are occlusion candidates.
BODY_CLASS_NAMES = {
    "Hair",
    "Face",
    "Face-skin",
    "Body-skin",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
}

OCCLUDER_CLASS_NAMES = {
    "Hair",
    "Body-skin",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
}

PRODUCT_TAGS = [
    "dress",
    "shirt",
    "t-shirt",
    "top",
    "trousers",
    "jeans",
    "skirt",
    "jacket",
    "coat",
    "saree",
    "kurti",
    "lehenga",
    "scarf",
    "other",
]

TAG_BY_LABEL = {
    "Upper-clothes": "top",
    "Skirt": "skirt",
    "Pants": "trousers",
    "Dress": "dress",
    "Scarf": "scarf",
    "Hat": "other",
    "Belt": "other",
    "Bag": "other",
    "Left-shoe": "other",
    "Right-shoe": "other",
    "Sunglasses": "other",
    "Others": "other",
    "Clothes": "other",
}

STAGE_DEFS = [
    ("uploading_image", "Uploading image"),
    ("detecting_person", "Detecting person"),
    ("identifying_garment", "Identifying garment"),
    ("generating_mask", "Generating garment mask"),
    ("removing_human", "Removing human body"),
    ("reconstructing", "Reconstructing hidden regions"),
    ("refining_output", "Refining final output"),
]

SEGFORMER_LICENSE = (
    "Non-commercial research and evaluation only. This checkpoint is a "
    "fine-tune of NVIDIA SegFormer (mit-b2), whose licence restricts the "
    "weights and derivative works to research or evaluation. Do not ship "
    "it in a commercial product. The training images come from the ATR "
    "human-parsing set, which is a research dataset."
)

MEDIAPIPE_LICENSE = (
    "Apache-2.0. MediaPipe and the selfie multiclass model are published "
    "by Google under Apache-2.0."
)

SAM2_LICENSE = (
    "Apache-2.0. SAM 2.1 hiera-tiny weights are published by Meta under Apache-2.0."
)

LAMA_LICENSE = (
    "Apache-2.0 for the LaMa code and the published torchscript weights. "
    "The original training used Places imagery; confirm that dataset's terms "
    "before commercial deployment. LaMa fills texture from surrounding pixels. "
    "It does not recover the hidden original cloth."
)

REMBG_LICENSE = (
    "rembg is MIT. The default u2net weights are MIT. Background cleanup is "
    "only a residue step after clothing has already been separated from the body."
)


def fresh_stages() -> list[dict]:
    return [
        {"id": stage_id, "label": label, "status": "pending"}
        for stage_id, label in STAGE_DEFS
    ]

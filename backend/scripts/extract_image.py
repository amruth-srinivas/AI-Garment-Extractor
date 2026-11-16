"""Run one photograph through the extraction pipeline.

Example:

    .venv\\Scripts\\python.exe scripts\\extract_image.py samples\\female-model-posing.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import ensure_dirs  # noqa: E402
from app.services.pipeline import extract_to_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a garment from one photograph")
    parser.add_argument("image", type=Path)
    parser.add_argument("--parser", default="auto", choices=["auto", "segformer", "mediapipe"])
    parser.add_argument("--no-reconstruct", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    ensure_dirs()
    output = args.out or (BACKEND_ROOT / "samples" / "output" / args.image.stem)
    meta = extract_to_folder(
        args.image,
        output,
        parser=args.parser,
        reconstruct=not args.no_reconstruct,
    )
    print(json.dumps({key: meta[key] for key in (
        "id",
        "parser",
        "product_tag",
        "garment_label",
        "confidence",
        "low_confidence",
        "reconstruction_method",
        "reconstruction_message",
        "notes",
    ) if key in meta}, indent=2))
    print("Transparent PNG:", output / "garment.png")
    print("White JPEG:", output / "garment_white.jpg")


if __name__ == "__main__":
    main()

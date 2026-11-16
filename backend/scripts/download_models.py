"""Download the parser, SAM 2, MediaPipe, and LaMa weights used by GarmentAI.

Run this from the backend directory with the project virtualenv:

    .venv\\Scripts\\python.exe scripts\\download_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import ensure_dirs, get_settings  # noqa: E402
from app.models.downloads import ensure_hf_repo, ensure_lama, ensure_mediapipe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GarmentAI model weights")
    parser.add_argument("--skip-sam2", action="store_true", help="Do not download SAM 2.1 hiera-tiny")
    parser.add_argument("--skip-lama", action="store_true", help="Do not download LaMa")
    parser.add_argument("--skip-segformer", action="store_true", help="Do not download SegFormer clothes")
    args = parser.parse_args()

    settings = get_settings()
    # The download script must be allowed to fetch weights even if the API
    # process was started with auto-download disabled.
    settings.auto_download = True
    ensure_dirs(settings)
    print("Weights directory:", settings.weights_dir)
    print("Hugging Face cache:", settings.hf_home)

    print("MediaPipe selfie multiclass...")
    path = ensure_mediapipe(settings)
    print("  saved", path)

    if not args.skip_segformer:
        print("SegFormer B2 clothes (non-commercial research / evaluation licence)...")
        ensure_hf_repo(settings.segformer_model_id, settings)
        print("  cached", settings.segformer_model_id)

    if not args.skip_sam2:
        print("SAM 2.1 hiera-tiny...")
        ensure_hf_repo(settings.sam2_model_id, settings)
        print("  cached", settings.sam2_model_id)

    if not args.skip_lama:
        print("LaMa inpainting...")
        print("  saved", ensure_lama(settings))

    print("Done.")


if __name__ == "__main__":
    main()

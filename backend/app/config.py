"""Runtime configuration.

Environment variables use the GARMENTAI_ prefix, for example
GARMENTAI_DATA_DIR and GARMENTAI_AUTO_DOWNLOAD.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GARMENTAI_",
        env_file=str(BACKEND_ROOT / ".env"),
        extra="ignore",
    )

    data_dir: Path = BACKEND_ROOT / "data"
    weights_dir: Path = BACKEND_ROOT / "weights"
    hf_home: Path = BACKEND_ROOT / ".cache" / "huggingface"
    log_dir: Path = BACKEND_ROOT / "logs"
    max_upload_mb: int = 30
    max_pixels: int = 40_000_000
    mask_max_side: int = 1280
    grabcut_max_side: int = 960
    lama_max_side: int = 640
    preview_max_side: int = 1400
    low_confidence: float = 0.45
    very_low_confidence: float = 0.22
    min_garment_fraction: float = 0.008
    job_ttl_hours: int = 48
    auto_download: bool = True
    segformer_model_id: str = "mattmdjaga/segformer_b2_clothes"
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    mediapipe_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    )
    mediapipe_sha256: str = (
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
    )
    lama_model_url: str = (
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def mediapipe_path(self) -> Path:
        return self.weights_dir / "selfie_multiclass_256x256.tflite"

    @property
    def lama_path(self) -> Path:
        return self.weights_dir / "big-lama.pt"

    @property
    def log_path(self) -> Path:
        return self.log_dir / "garmentai.log"


def apply_runtime_env(settings: Settings) -> None:
    os.environ["HF_HOME"] = str(settings.hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(settings.hf_home / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    apply_runtime_env(settings)
    return settings


def ensure_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for path in (
        settings.uploads_dir,
        settings.jobs_dir,
        settings.results_dir,
        settings.weights_dir,
        settings.log_dir,
        settings.hf_home,
    ):
        path.mkdir(parents=True, exist_ok=True)

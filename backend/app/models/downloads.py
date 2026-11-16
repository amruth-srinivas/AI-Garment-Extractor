"""Checkpoint downloads. Missing weights raise a setup error instead of a fake mask."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from app.config import Settings, get_settings
from app.errors import AppError

logger = logging.getLogger("garmentai.models")

USER_AGENT = "GarmentAI/1.0"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    logger.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise AppError(
            f"Could not download {dest.name}. Check the network and run "
            "scripts\\download_models.py. "
            f"Source: {url}. Detail: {exc}"
        ) from exc
    temporary.replace(dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_mediapipe(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = settings.mediapipe_path
    if path.exists():
        return path
    if not settings.auto_download:
        raise AppError(
            "MediaPipe selfie multiclass weights are missing at "
            f"{path}. Run scripts\\download_models.py."
        )
    _download(settings.mediapipe_model_url, path)
    digest = _sha256(path)
    if digest != settings.mediapipe_sha256:
        path.unlink(missing_ok=True)
        raise AppError(
            "The MediaPipe checkpoint hash did not match the published file. "
            "The download was deleted. Run scripts\\download_models.py again."
        )
    return path


def ensure_lama(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = settings.lama_path
    if path.exists() and path.stat().st_size > 50_000_000:
        return path
    if path.exists():
        path.unlink()
    if not settings.auto_download:
        raise AppError(
            "LaMa weights are missing at "
            f"{path}. Run scripts\\download_models.py, or turn reconstruction off."
        )
    _download(settings.lama_model_url, path)
    if path.stat().st_size < 50_000_000:
        path.unlink(missing_ok=True)
        raise AppError("The LaMa download looks incomplete. Run scripts\\download_models.py again.")
    return path


def hf_ready(repo_id: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    resolved = try_to_load_from_cache(
        repo_id,
        "config.json",
        cache_dir=str(settings.hf_home / "hub"),
    )
    return isinstance(resolved, str)


def ensure_hf_repo(repo_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if hf_ready(repo_id, settings):
        return
    if not settings.auto_download:
        raise AppError(
            f"Model {repo_id} is not in the local Hugging Face cache. "
            "Run scripts\\download_models.py."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AppError("huggingface_hub is not installed.") from exc
    logger.info("Downloading Hugging Face repo %s", repo_id)
    try:
        snapshot_download(repo_id, cache_dir=str(settings.hf_home / "hub"))
    except Exception as exc:
        raise AppError(
            f"Could not download {repo_id}. Detail: {exc}"
        ) from exc

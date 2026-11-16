"""GarmentAI API."""

from __future__ import annotations

import logging
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.extraction import router as extraction_router
from app.api.routes.results import router as results_router
from app.api.routes.system import router as system_router
from app.api.routes.upload import router as upload_router
from app.config import ensure_dirs, get_settings
from app.errors import AppError
from app.services.job_manager import jobs

logger = logging.getLogger("garmentai")


def configure_logging() -> None:
    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if getattr(root, "_garmentai_configured", False):
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root._garmentai_configured = True  # type: ignore[attr-defined]


def cleanup_expired() -> None:
    settings = get_settings()
    ttl = settings.job_ttl_hours * 3600
    now = time.time()
    for folder in (settings.uploads_dir, settings.jobs_dir, settings.results_dir):
        if not folder.exists():
            continue
        for child in folder.iterdir():
            try:
                if child.is_dir() and now - child.stat().st_mtime > ttl:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                logger.warning("Could not clean %s", child)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    ensure_dirs()
    cleanup_expired()
    jobs.recover()
    jobs.start()
    logger.info("GarmentAI API ready")
    yield
    jobs.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="GarmentAI",
        summary="Extract a garment from a fashion photograph.",
        description=(
            "Detect clothing on a person, remove the body, and optionally estimate "
            "cloth that the body was hiding. Hidden fills are estimates, not the "
            "original covered fabric. Interactive docs are at /docs."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_prefix = "/api"
    app.include_router(system_router, prefix=api_prefix)
    app.include_router(upload_router, prefix=api_prefix)
    app.include_router(extraction_router, prefix=api_prefix)
    app.include_router(results_router, prefix=api_prefix)

    @app.exception_handler(AppError)
    async def handle_app_error(_request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def handle_unexpected(_request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "The server hit an unexpected error. Check the log for details."},
        )

    return app


app = create_app()

"""One GPU worker, with job state stored on disk so the UI can poll it."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.constants import fresh_stages
from app.errors import AppError

logger = logging.getLogger("garmentai.jobs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="garment")

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def recover(self) -> None:
        folder = get_settings().jobs_dir
        if not folder.exists():
            return
        with self._lock:
            for child in folder.iterdir():
                path = child / "job.json"
                if not path.exists():
                    continue
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if job.get("status") in {"queued", "running"}:
                    job["status"] = "failed"
                    job["error"] = "The server restarted before this extraction finished. Run it again."
                    job["message"] = job["error"]
                    self._write(job)
                self._jobs[job["id"]] = job

    def create(self, image_id: str, options: dict) -> dict:
        job = {
            "id": uuid.uuid4().hex,
            "status": "queued",
            "stage": "uploading_image",
            "stages": fresh_stages(),
            "message": "Waiting for the extraction worker",
            "progress": 0,
            "image_id": image_id,
            "result_id": None,
            "error": None,
            "garments": [],
            "options": options,
            "preview_url": f"/api/uploads/{image_id}/file?kind=preview",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self._lock:
            self._jobs[job["id"]] = job
            self._write(job)
        return self.public(job)

    def submit(self, job_id: str) -> None:
        self.start()
        assert self._executor is not None
        from app.services.pipeline import execute_job

        self._executor.submit(self._run, job_id, execute_job)

    def _run(self, job_id: str, fn) -> None:
        try:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["status"] in {"completed", "failed"}:
                    return
                job["status"] = "running"
                job["updated_at"] = utc_now()
                self._write(job)
            fn(job_id)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            message = exc.message if isinstance(exc, AppError) else str(exc)[:700]
            if not message:
                message = "Extraction failed before a garment could be exported."
            self.fail(job_id, message)

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                path = self._path(job_id)
                if not path.exists():
                    raise AppError("That extraction could not be found.", 404)
                job = json.loads(path.read_text(encoding="utf-8"))
                self._jobs[job_id] = job
            return json.loads(json.dumps(job))

    def public(self, job: dict) -> dict:
        hidden = {"mask_name"}
        options = {key: value for key, value in job.get("options", {}).items() if key not in hidden}
        return {
            "id": job["id"],
            "status": job["status"],
            "stage": job["stage"],
            "stages": job["stages"],
            "message": job["message"],
            "progress": job["progress"],
            "image_id": job["image_id"],
            "result_id": job["result_id"],
            "error": job["error"],
            "garments": job["garments"],
            "options": options,
            "preview_url": job["preview_url"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    def snapshot(self, job_id: str) -> dict:
        return self.public(self.get(job_id))

    def result_in_progress(self, result_id: str) -> bool:
        with self._lock:
            stored = list(self._jobs.values())
        for job in stored:
            if job.get("status") not in {"queued", "running"}:
                continue
            if job.get("result_id") == result_id or job.get("options", {}).get("result_id") == result_id:
                return True
        return False

    def discard(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.get("status") in {"queued", "running"}:
                raise AppError("That extraction is still running.", 409)
            self._jobs.pop(job_id, None)
        folder = get_settings().jobs_dir / job_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def patch_options(self, job_id: str, **changes) -> dict:
        with self._lock:
            job = self._require(job_id)
            job["options"].update(changes)
            job["updated_at"] = utc_now()
            self._write(job)
            return self.public(job)

    def stage(self, job_id: str, stage_id: str, message: str) -> None:
        self._set_stage(job_id, stage_id, "active", message)

    def skip(self, job_id: str, stage_id: str, message: str) -> None:
        with self._lock:
            job = self._require(job_id)
            for stage in job["stages"]:
                if stage["id"] == stage_id:
                    stage["status"] = "skipped"
            job["message"] = message
            job["updated_at"] = utc_now()
            job["progress"] = self._progress(job)
            self._write(job)
        logger.info("Job %s skipped %s: %s", job_id, stage_id, message)

    def await_selection(self, job_id: str, garments: list[dict]) -> None:
        with self._lock:
            job = self._require(job_id)
            job["status"] = "awaiting_selection"
            job["garments"] = garments
            job["message"] = "Several garments were detected. Choose which one to extract."
            job["updated_at"] = utc_now()
            job["progress"] = self._progress(job)
            self._write(job)

    def succeed(self, job_id: str, result_id: str, message: str) -> None:
        with self._lock:
            job = self._require(job_id)
            for stage in job["stages"]:
                if stage["status"] != "skipped":
                    stage["status"] = "done"
            job["status"] = "completed"
            job["result_id"] = result_id
            job["message"] = message
            job["progress"] = 100
            job["error"] = None
            job["updated_at"] = utc_now()
            self._write(job)

    def fail(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job["status"] == "completed":
                return
            job["status"] = "failed"
            job["error"] = message
            job["message"] = message
            job["updated_at"] = utc_now()
            self._write(job)

    def _set_stage(self, job_id: str, stage_id: str, status: str, message: str) -> None:
        with self._lock:
            job = self._require(job_id)
            seen = False
            known = False
            for stage in job["stages"]:
                if stage["id"] == stage_id:
                    seen = True
                    known = True
                    if stage["status"] != "skipped":
                        stage["status"] = status
                elif not seen and stage["status"] != "skipped":
                    stage["status"] = "done"
            if not known:
                raise AppError(f"Unknown processing stage {stage_id}.")
            job["stage"] = stage_id
            job["message"] = message
            job["progress"] = self._progress(job)
            job["updated_at"] = utc_now()
            if job["status"] == "queued":
                job["status"] = "running"
            self._write(job)
        logger.info("Job %s %s: %s", job_id, stage_id, message)

    def _progress(self, job: dict) -> int:
        stages = job["stages"]
        if not stages:
            return 0
        finished = sum(1 for stage in stages if stage["status"] in {"done", "skipped"})
        return int(round(finished / len(stages) * 100))

    def _require(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError("That extraction could not be found.", 404)
        return job

    def _path(self, job_id: str) -> Path:
        return get_settings().jobs_dir / job_id / "job.json"

    def _write(self, job: dict) -> None:
        path = self._path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)


jobs = JobManager()

import io
import time

from PIL import Image


def _png() -> bytes:
    image = Image.new("RGB", (128, 160), (30, 90, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "GarmentAI"
    assert "segformer" in body["models"]
    assert "product_tags" in body
    assert "dress" in body["product_tags"]
    assert "saree" in body["product_tags"]


def test_upload_and_preview(client):
    response = client.post(
        "/api/uploads",
        files={"file": ("studio.png", _png(), "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["width"] == 128
    assert body["height"] == 160
    preview = client.get(body["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/jpeg")


def test_rejects_invalid_upload(client):
    response = client.post(
        "/api/uploads",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_rejects_mismatched_jpeg_bytes(client):
    response = client.post(
        "/api/uploads",
        files={"file": ("look.jpg", b"not really a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400


def test_delete_result(client):
    from app.config import get_settings

    result_id = "c" * 32
    job_id = "d" * 32
    folder = get_settings().results_dir / result_id
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(
        '{"id":"%s","job_id":"%s","product_tag":"dress","garment_label":"Dress","confidence":0.9}'
        % (result_id, job_id),
        encoding="utf-8",
    )
    (folder / "garment.png").write_bytes(b"png")
    job_folder = get_settings().jobs_dir / job_id
    job_folder.mkdir(parents=True)
    (job_folder / "job.json").write_text("{}", encoding="utf-8")

    listed = client.get("/api/results")
    assert any(item["id"] == result_id for item in listed.json()["items"])
    deleted = client.delete(f"/api/results/{result_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/results/{result_id}").status_code == 404
    assert not folder.exists()
    assert not job_folder.exists()


def test_missing_result_is_404(client):
    response = client.get("/api/results/" + ("a" * 32))
    assert response.status_code == 404


def test_missing_job_is_404(client):
    response = client.get("/api/extractions/" + ("b" * 32))
    assert response.status_code == 404


def test_extraction_without_models_fails_cleanly(client):
    uploaded = client.post(
        "/api/uploads",
        files={"file": ("studio.png", _png(), "image/png")},
    )
    assert uploaded.status_code == 201
    started = client.post(
        "/api/extractions",
        json={"image_id": uploaded.json()["image_id"], "parser": "auto", "reconstruct": False},
    )
    assert started.status_code == 202
    job_id = started.json()["id"]
    deadline = time.time() + 90
    status = started.json()
    while status["status"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.4)
        status = client.get(f"/api/extractions/{job_id}").json()
    assert status["status"] == "failed"
    assert status["error"]
    continued = client.post(
        f"/api/extractions/{job_id}/continue",
        json={"garment_ids": ["dress_0"]},
    )
    assert continued.status_code == 409

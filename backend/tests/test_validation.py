import io

import pytest
from PIL import Image

from app.errors import AppError
from app.utils.validation import decode_upload


def _png_bytes() -> bytes:
    image = Image.new("RGB", (96, 120), (180, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_png_upload_decodes():
    rgb, info = decode_upload(_png_bytes(), "look.png")
    assert info["width"] == 96
    assert info["height"] == 120
    assert rgb.shape == (120, 96, 3)


def test_rejects_text_disguised_as_jpeg():
    with pytest.raises(AppError):
        decode_upload(b"this is not a photo", "look.jpg")


def test_rejects_unknown_extension():
    with pytest.raises(AppError):
        decode_upload(_png_bytes(), "look.bmp")


def test_rejects_tiny_image():
    image = Image.new("RGB", (16, 16), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(AppError):
        decode_upload(buffer.getvalue(), "tiny.png")

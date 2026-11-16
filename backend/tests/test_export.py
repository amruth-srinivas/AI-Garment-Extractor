import io

import numpy as np
from PIL import Image

from app.services.garment_reconstruction import reconstruct
from app.services.human_removal import compose_garment
from app.utils.image_utils import encode_png, white_background


def _garment_scene():
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:] = (240, 240, 240)
    rgb[20:60, 20:60] = (12, 150, 48)
    garment = np.zeros((80, 80), dtype=bool)
    garment[20:60, 20:60] = True
    holes = np.zeros((80, 80), dtype=bool)
    holes[36:46, 36:46] = True
    rgb[holes] = (220, 40, 40)
    garment[holes] = False
    return rgb, garment, holes


def test_original_garment_pixels_survive_reconstruction():
    rgb, garment, holes = _garment_scene()
    filled, method, message = reconstruct(rgb, garment, holes, method="opencv_telea")
    assert method == "opencv_telea"
    assert "estimate" in message.lower() or "classical" in message.lower()
    assert np.array_equal(filled[garment], rgb[garment])
    assert not np.array_equal(filled[holes], rgb[holes])


def test_disabled_reconstruction_does_not_invent_pixels():
    rgb, garment, holes = _garment_scene()
    filled, method, _message = reconstruct(rgb, garment, holes, method="disabled")
    assert method == "disabled"
    assert np.array_equal(filled, rgb)


def test_transparent_png_keeps_the_background_clear():
    rgb, garment, holes = _garment_scene()
    rgba = compose_garment(rgb, garment)
    assert rgba[10, 10, 3] == 0
    assert rgba[40, 28, 3] == 255
    assert rgba[40, 28, 1] == 150
    assert rgba[40, 40, 3] == 0
    encoded = encode_png(rgba)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((10, 10))[3] == 0
        assert image.getpixel((28, 40))[3] == 255


def test_white_background_flattens_transparency():
    rgb, garment, _holes = _garment_scene()
    rgba = compose_garment(rgb, garment)
    white = white_background(rgba)
    assert white.shape[2] == 3
    assert white[10, 10].tolist() == [255, 255, 255]
    assert abs(int(white[40, 28, 1]) - 150) <= 2

import numpy as np

from tktomo.colormaps import available_colormaps, get_lut


def test_grayscale_always_available():
    assert "grayscale" in available_colormaps()


def test_grayscale_lut_shape():
    lut = get_lut("grayscale")
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    # monotonic ramp
    assert lut[0, 0] == 0 and lut[-1, 0] == 255

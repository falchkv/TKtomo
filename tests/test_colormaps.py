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


def test_albula_is_always_available_and_shaped_like_albula():
    names = available_colormaps()
    assert "albula" in names and "albula-hot" in names
    lut = get_lut("albula")
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert (lut[0] == 255).all()            # white at the bottom
    assert (lut[127] < 10).all()            # black at the split
    assert (lut[-1] == 255).all()           # white at the top
    assert lut[170, 0] == 255 and lut[170, 2] == 0   # red before yellow

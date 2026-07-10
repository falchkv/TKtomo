import numpy as np
import pytest

from tktomo.io import ProjectionData, load_projections, save_projections
from tktomo.io.phantom import generate_phantom, generate_volume


def test_projection_data_validation():
    with pytest.raises(ValueError):
        ProjectionData(data=np.zeros((3, 4)), angles=np.zeros(3))  # 2D not allowed
    with pytest.raises(ValueError):
        ProjectionData(data=np.zeros((3, 4, 5)), angles=np.zeros(2))  # angle mismatch


def test_sinogram_and_projection_slicing():
    proj = ProjectionData(data=np.random.rand(5, 6, 7), angles=np.linspace(0, np.pi, 5))
    assert proj.sinogram(2).shape == (5, 7)
    assert proj.projection(1).shape == (6, 7)


def test_phantom_shapes():
    proj = generate_phantom(n_angles=30, size=64, n_slices=4)
    assert proj.shape == (30, 4, 64)
    assert proj.angles.shape == (30,)
    vol = generate_volume(size=64, n_slices=4)
    assert vol.shape == (4, 64, 64)


def test_phantom_slices_differ_structurally():
    proj = generate_phantom(n_angles=30, size=64, n_slices=8)
    first, last = proj.sinogram(0), proj.sinogram(7)
    assert not np.allclose(first, last)
    # not merely a scalar multiple of each other:
    ratio = last / np.where(first == 0, np.nan, first)
    assert np.nanstd(ratio) > 1e-6
    vol = generate_volume(size=64, n_slices=8)
    assert not np.allclose(vol[0], vol[-1])


def test_hdf5_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    proj = generate_phantom(n_angles=20, size=48, n_slices=3)
    path = tmp_path / "proj.h5"
    save_projections(path, proj)
    loaded = load_projections(path)
    np.testing.assert_allclose(loaded.data, proj.data)
    np.testing.assert_allclose(loaded.angles, proj.angles)

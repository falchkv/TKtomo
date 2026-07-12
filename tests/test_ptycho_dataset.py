"""Loading HDF5 files that do not follow a conventional layout.

The convention-driven loader probes NXtomo / DXchange / blender-sim locations. A real
ptycho pipeline writes wherever it likes, so the alignment app has to be able to read
an explicitly-named dataset out of an arbitrary tree -- and cope with the angle axis
not being first.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from tktomo.ptycho_align.core import (  # noqa: E402
    list_hdf5_datasets,
    load_dataset,
    load_hdf5,
    suggest_hdf5_paths,
)


@pytest.fixture
def odd_file(tmp_path):
    """An HDF5 file no layout probe would recognise: nested, oddly named, decoy arrays."""
    path = tmp_path / "beamline.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        group = f.create_group("recon/2024_08/ptycho")
        group.create_dataset("object_phase", data=rng.random((12, 8, 6)).astype(np.float32))
        group.create_dataset("rotation", data=np.linspace(0.0, 180.0, 12, endpoint=False))
        # Decoys: a 2-D probe, a scalar, and a 1-D array of the wrong length.
        group.create_dataset("probe", data=rng.random((8, 6)))
        f.create_dataset("meta/energy_keV", data=8.9)
        f.create_dataset("meta/ring_current", data=np.arange(37.0))
    return path


def test_list_hdf5_datasets_finds_arrays_at_any_depth(odd_file):
    entries = {e.path: e for e in list_hdf5_datasets(odd_file)}

    assert "/recon/2024_08/ptycho/object_phase" in entries
    assert "/meta/energy_keV" in entries  # scalars are listed too; the GUI greys them out

    stack = entries["/recon/2024_08/ptycho/object_phase"]
    assert stack.shape == (12, 8, 6)
    assert stack.is_stack and not stack.is_angles
    assert entries["/recon/2024_08/ptycho/rotation"].is_angles
    assert not entries["/recon/2024_08/ptycho/probe"].is_stack  # 2-D
    assert not entries["/meta/energy_keV"].is_angles  # scalar


def test_suggest_picks_the_stack_and_its_angle_array(odd_file):
    data_path, angle_path = suggest_hdf5_paths(list_hdf5_datasets(odd_file))

    assert data_path == "/recon/2024_08/ptycho/object_phase"
    # /meta/ring_current is 1-D too, but the wrong length and the wrong name.
    assert angle_path == "/recon/2024_08/ptycho/rotation"


def test_suggest_returns_nothing_when_there_is_no_stack(tmp_path):
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=np.zeros((4, 4)))
    assert suggest_hdf5_paths(list_hdf5_datasets(path)) == (None, None)


def test_load_hdf5_reads_the_named_datasets(odd_file):
    data = load_hdf5(
        odd_file,
        data_path="/recon/2024_08/ptycho/object_phase",
        angle_path="/recon/2024_08/ptycho/rotation",
    )

    assert data.data.shape == (12, 8, 6)
    assert data.data.dtype == np.float32
    # The angles were stored in degrees; the whole toolkit speaks radians.
    assert data.angles.max() == pytest.approx(np.deg2rad(165.0))
    assert data.metadata["data_path"] == "/recon/2024_08/ptycho/object_phase"


def test_load_dataset_routes_explicit_paths_to_load_hdf5(odd_file):
    data = load_dataset(
        odd_file,
        data_path="/recon/2024_08/ptycho/object_phase",
        angle_path="/recon/2024_08/ptycho/rotation",
    )
    assert data.data.shape == (12, 8, 6)


def test_axis_order_rotates_an_angle_last_stack(tmp_path):
    """A stack saved as (v, u, angle) must be readable without rewriting the file."""
    path = tmp_path / "angle_last.h5"
    stored = np.arange(8 * 6 * 12, dtype=np.float32).reshape(8, 6, 12)  # (v, u, angle)
    with h5py.File(path, "w") as f:
        f.create_dataset("stack", data=stored)
        f.create_dataset("theta", data=np.linspace(0.0, np.pi, 12, endpoint=False))

    data = load_hdf5(path, data_path="/stack", angle_path="/theta", axis_order=(2, 0, 1))

    assert data.data.shape == (12, 8, 6)
    np.testing.assert_allclose(data.data, np.transpose(stored, (2, 0, 1)))


def test_missing_angles_fall_back_to_a_uniform_scan(odd_file):
    data = load_hdf5(odd_file, data_path="/recon/2024_08/ptycho/object_phase")

    assert data.angles.shape == (12,)
    np.testing.assert_allclose(data.angles, np.linspace(0.0, np.pi, 12, endpoint=False))


def test_a_mismatched_angle_array_says_which_two_things_disagree(odd_file):
    """The commonest way to get this wrong: right stack, wrong angle dataset."""
    with pytest.raises(ValueError, match="37 angles but .* 12 projections"):
        load_hdf5(
            odd_file,
            data_path="/recon/2024_08/ptycho/object_phase",
            angle_path="/meta/ring_current",
        )


def test_a_non_3d_dataset_is_rejected(odd_file):
    with pytest.raises(ValueError, match="must be 3-D"):
        load_hdf5(odd_file, data_path="/recon/2024_08/ptycho/probe")


def test_a_missing_dataset_names_the_file(odd_file):
    with pytest.raises(KeyError, match="beamline.h5"):
        load_hdf5(odd_file, data_path="/nope")

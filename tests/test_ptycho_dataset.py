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


# -- complex projections and cropping ----------------------------------------------------


@pytest.fixture
def complex_file(tmp_path):
    """A ptycho reconstruction as it really arrives: complex, angle-in-degrees, huge."""
    path = tmp_path / "recon.h5"
    v, u = np.mgrid[0:20, 0:30]
    # A stack whose phase and amplitude are different, so a mix-up is visible.
    phase = (v / 20.0 - 0.5).astype(np.float32)
    amplitude = (1.0 + u / 30.0).astype(np.float32)
    stack = (amplitude * np.exp(1j * phase))[None].repeat(9, axis=0).astype(np.complex64)
    with h5py.File(path, "w") as f:
        f.create_dataset("obj", data=stack)
        f.create_dataset("angle", data=np.linspace(0.0, 180.0, 9, endpoint=False))
        f.create_dataset("pr", data=np.zeros((9, 1, 4, 4), dtype=np.complex64))  # to be ignored
    return path, phase, amplitude


def test_a_4d_dataset_is_not_offered_as_a_stack(complex_file):
    """The 'pr' probe array is 4-D; it must never be mistaken for the projections."""
    path, _phase, _amplitude = complex_file
    entries = {e.path: e for e in list_hdf5_datasets(path)}

    assert not entries["/pr"].is_stack
    assert suggest_hdf5_paths(list_hdf5_datasets(path)) == ("/obj", "/angle")


def test_complex_projections_load_as_phase_or_amplitude(complex_file):
    path, phase, amplitude = complex_file

    as_phase = load_hdf5(path, data_path="/obj", angle_path="/angle", component="phase")
    as_amplitude = load_hdf5(path, data_path="/obj", angle_path="/angle", component="amplitude")

    assert as_phase.data.dtype == np.float32
    np.testing.assert_allclose(as_phase.data[0], phase, atol=1e-6)
    np.testing.assert_allclose(as_amplitude.data[0], amplitude, atol=1e-6)
    assert as_phase.metadata["component"] == "phase"


def test_complex_data_is_never_silently_cast(complex_file):
    """numpy only *warns* on complex -> float and keeps the real part, which for a
    ptycho object is meaningless. Every path must choose a component explicitly."""
    from tktomo.ptycho_align.core.dataset import _finalise

    path, _phase, _amplitude = complex_file
    with pytest.raises(ValueError, match="complex"):
        _finalise(np.zeros((2, 2, 2), dtype=np.complex64), np.zeros(2), name="x")

    with pytest.raises(ValueError, match="Unknown component"):
        load_hdf5(path, data_path="/obj", component="magnitude")


def test_crop_reads_only_the_requested_region(complex_file):
    path, phase, _amplitude = complex_file

    cropped = load_hdf5(
        path, data_path="/obj", angle_path="/angle", component="phase", crop=(4, 12, 6, 20)
    )

    assert cropped.data.shape == (9, 8, 14)
    np.testing.assert_allclose(cropped.data[0], phase[4:12, 6:20], atol=1e-6)
    assert cropped.metadata["crop"] == (4, 12, 6, 20)
    assert cropped.metadata["full_shape"] == (9, 20, 30)


def test_crop_is_clipped_to_the_stack_rather_than_reading_out_of_bounds(complex_file):
    path, _phase, _amplitude = complex_file
    data = load_hdf5(path, data_path="/obj", crop=(0, 999, 0, 999))
    assert data.data.shape == (9, 20, 30)


def test_crop_and_axis_order_compose(tmp_path):
    """The crop is in the post-axis-order frame, and is still a hyperslab in the file."""
    path = tmp_path / "angle_last.h5"
    stored = np.arange(20 * 30 * 9, dtype=np.float32).reshape(20, 30, 9)  # (v, u, angle)
    with h5py.File(path, "w") as f:
        f.create_dataset("stack", data=stored)

    data = load_hdf5(path, data_path="/stack", axis_order=(2, 0, 1), crop=(4, 12, 6, 20))

    assert data.data.shape == (9, 8, 14)
    np.testing.assert_allclose(data.data, np.transpose(stored, (2, 0, 1))[:, 4:12, 6:20])


def test_progress_is_reported_and_cancelling_raises(complex_file):
    from tktomo.ptycho_align.core import DatasetProblem

    path, _phase, _amplitude = complex_file

    seen = []
    load_hdf5(path, data_path="/obj", progress=lambda done, total: seen.append((done, total)) is None)
    assert seen[0] == (1, 9) and seen[-1] == (9, 9)

    # A cancelled read must not return a half-filled stack.
    with pytest.raises(DatasetProblem, match="cancelled"):
        load_hdf5(path, data_path="/obj", progress=lambda done, total: done < 3)


def test_crop_rejects_an_empty_region():
    from tktomo.ptycho_align.core import Crop

    with pytest.raises(ValueError, match="Empty crop"):
        Crop(10, 10, 0, 5)

    assert Crop(2, 6, 3, 9).shifted_by(10, 100).as_tuple() == (12, 16, 103, 109)
    assert Crop.full((5, 20, 30)).as_tuple() == (0, 20, 0, 30)


def test_load_kwargs_survive_a_json_round_trip():
    """The browser builds these kwargs; the file they describe may be on another machine.

    `Crop` is not serialisable at all, and tuples come back as lists. `load_hdf5` happens
    to tolerate a list for both today, but the dict is also stored in `metadata` and
    compared against later, where a list-vs-tuple mismatch is a silent inequality.
    """
    import json

    from tktomo.ptycho_align.core import Crop, coerce_load_kwargs, jsonable_load_kwargs

    kwargs = {
        "data_path": "/obj",
        "angle_path": "/theta",
        "axis_order": (1, 0, 2),
        "angles_in_degrees": True,
        "component": "phase",
        "crop": Crop(2, 30, 5, 100),
    }

    restored = coerce_load_kwargs(json.loads(json.dumps(jsonable_load_kwargs(kwargs))))

    assert restored == kwargs
    assert isinstance(restored["crop"], Crop)
    assert isinstance(restored["axis_order"], tuple)


def test_load_kwargs_coercion_tolerates_absent_optional_keys():
    from tktomo.ptycho_align.core import coerce_load_kwargs, jsonable_load_kwargs

    kwargs = {"data_path": "/obj", "crop": None, "axis_order": None}
    assert coerce_load_kwargs(jsonable_load_kwargs(kwargs)) == kwargs


def test_load_kwargs_coercion_rejects_a_bad_axis_order():
    from tktomo.ptycho_align.core import coerce_load_kwargs

    with pytest.raises(ValueError, match="permutation"):
        coerce_load_kwargs({"axis_order": [0, 1, 1]})

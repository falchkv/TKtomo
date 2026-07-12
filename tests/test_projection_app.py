"""GUI tests for the projection app's load / reconstruct controls (pytest-qt)."""

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QInputDialog  # noqa: E402

from tktomo.recon import register_backend  # noqa: E402
from tktomo.ui.projection_app import ProjectionWindow  # noqa: E402


class FakeBackend:
    """Records the reconstruct call and returns a fixed volume."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def reconstruct(self, projections, angles, **kwargs):
        self.calls.append(kwargs)
        return np.ones((projections.shape[1], 8, 8))

    def reproject(self, volume, angles, **kwargs):
        return np.zeros((len(angles), volume.shape[0], volume.shape[2]))


@pytest.fixture
def fake_backend():
    backend = FakeBackend()
    register_backend(backend)
    return backend


def test_reconstruct_saves_and_opens_tomogram(qtbot, tmp_path, monkeypatch, fake_backend):
    window = ProjectionWindow()
    qtbot.addWidget(window)
    window.backend_combo.setCurrentText("fake")
    window.algorithm_combo.setCurrentText("gridrec")
    window.center_edit.setText("24.5")

    out = tmp_path / "tomogram.h5"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    window._reconstruct()

    assert fake_backend.calls == [
        {"algorithm": "gridrec", "center": 24.5, "filter_name": "shepp"}
    ]
    assert out.exists()
    h5py = pytest.importorskip("h5py")
    with h5py.File(out, "r") as f:
        assert f["tomogram/data"].shape == (window.data.shape[1], 8, 8)
        np.testing.assert_allclose(f["tomogram/angles"][()], window.data.angles)
        assert f["tomogram"].attrs["backend"] == "fake"

    assert len(window._tomogram_windows) == 1
    tomo = window._tomogram_windows[0]
    qtbot.addWidget(tomo)
    np.testing.assert_allclose(tomo.angles, window.data.angles)
    assert tomo.volume.shape == (window.data.shape[1], 8, 8)


def test_iterative_algorithm_kwargs(qtbot, tmp_path, monkeypatch, fake_backend):
    window = ProjectionWindow()
    qtbot.addWidget(window)
    window.backend_combo.setCurrentText("fake")
    window.algorithm_combo.setCurrentText("sirt")
    assert not window.filter_combo.isEnabled()
    assert window.iterations_spin.isEnabled()
    window.iterations_spin.setValue(7)

    monkeypatch.setattr(  # cancel the save dialog: reconstruct still proceeds
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )
    window._reconstruct()

    assert fake_backend.calls == [{"algorithm": "sirt", "num_iter": 7}]
    assert len(window._tomogram_windows) == 1
    qtbot.addWidget(window._tomogram_windows[0])


def test_load_blender_sim_file_with_group_choice(qtbot, tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "proj.h5"
    angles = np.linspace(0.0, np.pi, 6, endpoint=False)
    with h5py.File(path, "w") as f:
        for i, name in enumerate(("attenuation", "phase")):
            group = f.create_group(name)
            group.create_dataset("data", data=np.full((6, 8, 8), float(i)))
            group.create_dataset("angles", data=angles)

    window = ProjectionWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("phase", True))
    )
    window._load()

    assert window.data.shape == (6, 8, 8)
    np.testing.assert_allclose(window.data.data, 1.0)  # the "phase" group
    np.testing.assert_allclose(window.data.angles, angles)

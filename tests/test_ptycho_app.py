"""GUI tests for ptycho-align.

Exercises the step-wise workflow the whole tool exists for: load, preprocess, COM,
step, run, stop, revert, save/reload, and the binned-preview shift rescaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")
pytest.importorskip("tomopy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from make_phantom import make_misaligned_dataset  # noqa: E402

from tktomo.io import save_projections  # noqa: E402
from tktomo.ptycho_align.core import io as session_io  # noqa: E402
from tktomo.ptycho_align.core import load_dataset  # noqa: E402
from tktomo.ptycho_align.ui.main_window import PtychoAlignWindow, bin_stack  # noqa: E402
from tktomo.ptycho_align.ui.panels import Hdf5BrowserDialog  # noqa: E402
from tktomo.ptycho_align.ui.panels.base import (  # noqa: E402
    MODE_ALIGNED,
    MODE_DIFFERENCE,
    MODE_REPROJECTION,
)


@pytest.fixture(scope="module")
def phantom_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "phantom.h5"
    data, _sx, _sy = make_misaligned_dataset(
        size=32, n_angles=40, max_shift=1.5, margin=6, seed=3
    )[:3]
    save_projections(path, data)
    return path


@pytest.fixture
def window(qtbot, phantom_file):
    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    win.load_path(str(phantom_file))
    return win


def _run(window, n: int) -> None:
    """Run n iterations on the worker thread and wait for it to finish."""
    window._start_run(n)
    assert window._run.wait(120_000), "the alignment run did not finish"
    window._run_finished()


def test_bin_stack_halves_the_detector_axes():
    stack = np.ones((3, 8, 10), dtype=np.float32)
    assert bin_stack(stack, 2).shape == (3, 4, 5)
    assert bin_stack(stack, 1).shape == stack.shape
    # An odd size is trimmed, not padded.
    assert bin_stack(np.ones((3, 9, 9), dtype=np.float32), 2).shape == (3, 4, 4)


def test_loading_populates_the_engine(window):
    assert window.raw is not None
    assert window.engine is not None
    assert window.engine.iteration == 0


def test_preprocessing_pads_and_rebuilds_the_engine(window):
    before = window.engine.state.original.shape
    options = window.preprocess_panel.options()
    options.pad_percent = 20.0
    window._apply_preprocessing(options)

    after = window.engine.state.original.shape
    assert after[1] > before[1] and after[2] > before[2], "padding was not applied"
    assert window.engine.iteration == 0


def test_com_prealignment_populates_shifts_and_centre(window):
    window._run_com("mean")

    assert window.com is not None
    assert np.any(window.engine.state.sx != 0.0)
    assert window.engine.state.center > 0
    # The sinogram overlay and the fit plot both need the fitted curve.
    assert window.com.fitted_u.shape == window.engine.state.angles.shape


def test_step_produces_a_volume_and_fills_every_view(window):
    window._run_com("mean")
    _run(window, 1)

    assert window.engine.iteration == 1
    assert window.engine.state.volume is not None

    # The viewers are fed from cached stacks; all four modes must be populated.
    assert window._stacks[MODE_ALIGNED] is not None
    assert window._stacks[MODE_REPROJECTION] is not None
    assert window._stacks[MODE_DIFFERENCE] is not None
    assert window._stacks[MODE_DIFFERENCE].shape == window._stacks[MODE_ALIGNED].shape


def test_running_iterations_reduces_the_residual(window):
    window._run_com("mean")
    _run(window, 4)

    history = window.engine.history
    assert len(history) == 4
    assert history[-1].residual < history[0].residual


def test_stop_cancels_and_leaves_a_valid_state(window):
    window._run_com("mean")
    window._start_run(50)
    window._stop_run()  # cancel immediately
    assert window._run.wait(120_000)
    window._run_finished()

    assert window.engine.iteration < 50
    # Whatever it managed, the state must be complete and steppable.
    _run(window, 1)
    assert window.engine.state.volume is not None


def test_revert_rolls_back_the_iteration(window):
    window._run_com("mean")
    _run(window, 3)
    sx_at_1 = window.engine.history[0].sx.copy()

    window._revert(1)

    assert window.engine.iteration == 1
    np.testing.assert_allclose(window.engine.state.sx, sx_at_1)


def test_session_roundtrip_restores_the_state(window, tmp_path):
    window._run_com("mean")
    _run(window, 2)

    path = tmp_path / "session.h5"
    session_io.save_session(path, window.engine)
    restored = session_io.load_session(path)

    assert restored.iteration == 2
    np.testing.assert_allclose(restored.state.sx, window.engine.state.sx)
    np.testing.assert_allclose(restored.state.sy, window.engine.state.sy)
    assert restored.state.center == pytest.approx(window.engine.state.center)
    assert restored.config.recon_algorithm == window.engine.config.recon_algorithm


def test_bin_factor_rescales_the_existing_shifts(window):
    window._run_com("mean")
    sx_before = window.engine.state.sx.copy()
    center_before = window.engine.state.center

    window.align_panel.bin_spin.setValue(2)

    # Shifts are in pixels of the (now half-size) grid, so they halve with it.
    np.testing.assert_allclose(window.engine.state.sx, sx_before / 2, rtol=1e-6)
    assert window.engine.state.center == pytest.approx(center_before / 2)
    assert window.engine.state.original.shape[2] == pytest.approx(
        window.preprocessed.data.shape[2] // 2, abs=1
    )


def test_align_toggles_reach_the_engine(window):
    window.align_panel.horizontal_check.setChecked(False)
    window._run_com("mean")
    _run(window, 1)

    result = window.engine.history[-1]
    np.testing.assert_array_equal(result.dsx, np.zeros_like(result.dsx))


def test_engine_log_reaches_the_dock_from_the_worker_thread(window, qtbot):
    """The engine logs from inside step(), i.e. on the worker thread.

    Touching a QWidget off the GUI thread is undefined behaviour: it corrupted Qt's
    state and segfaulted the process inside a later, unrelated paint event. The handler
    must hop threads via a queued signal, so this asserts the line actually arrives.
    """
    window._run_com("mean")
    _run(window, 1)
    qtbot.wait(100)  # let the queued log signal be delivered

    text = window.log_widget.toPlainText()
    assert "iter 1:" in text, f"the worker thread's log never reached the dock: {text!r}"
    assert "COM pre-alignment" in text


def test_log_handler_is_detached_when_the_window_closes(qtbot, phantom_file):
    """The logger is module-global; a handler left pointing at a destroyed widget
    would fire again the next time a window opens."""
    import logging

    logger = logging.getLogger("tktomo.ptycho_align")
    before = len(logger.handlers)

    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    assert len(logger.handlers) == before + 1

    win.close()
    assert len(logger.handlers) == before


def test_reprojection_owns_its_memory(window):
    """tomopy.project returns a view onto its shared-memory buffer, which it recycles.

    The GUI caches these arrays and pyqtgraph paints them lazily, so a borrowed buffer
    is read after tomopy has reused it.
    """
    window._run_com("mean")
    _run(window, 1)

    assert window.engine.last_simulated.flags.owndata
    assert window.engine.state.volume.flags.owndata


def test_window_fits_on_a_small_screen(qtbot):
    """The action bar must stay reachable on a 1080p display.

    The four left docks stacked demanded ~1000 px of height, which pushed the window's
    minimum past the screen and put Step/Run/Stop off the bottom edge, where they could
    not be clicked or dragged back into view.
    """
    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    win.show()

    minimum = win.minimumSizeHint()
    assert minimum.height() <= 700, f"minimum height {minimum.height()} px is too tall"
    assert minimum.width() <= 1280, f"minimum width {minimum.width()} px is too wide"

    # And it must actually shrink that far, with the action bar still inside the window.
    win.resize(1000, 620)
    qtbot.wait(50)
    bottom = win.action_bar.mapTo(win, win.action_bar.rect().bottomLeft()).y()
    assert bottom <= win.height(), "the action bar is below the bottom of the window"


def test_export_without_a_volume_reports_clearly(window, tmp_path):
    # Nothing has been reconstructed yet, so there is no volume to write.
    with pytest.raises(ValueError, match="no volume to export"):
        session_io.export_volume(tmp_path / "v.h5", window.engine.state.volume)


# -- HDF5 dataset browser ---------------------------------------------------------------


def _ok(dialog):
    from PySide6.QtWidgets import QDialogButtonBox

    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)


@pytest.fixture
def odd_file(tmp_path):
    """A file no layout probe recognises: the stack is nested and oddly named."""
    import h5py

    path = tmp_path / "beamline.h5"
    data, _sx, _sy = make_misaligned_dataset(size=24, n_angles=20, max_shift=1.0, seed=5)[:3]
    with h5py.File(path, "w") as f:
        group = f.create_group("recon/ptycho")
        group.create_dataset("object_phase", data=data.data)
        group.create_dataset("rotation", data=np.rad2deg(data.angles))
    return path


def test_browser_preselects_the_stack_and_its_angles(qtbot, odd_file):
    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    dialog = Hdf5BrowserDialog(str(odd_file), win)
    qtbot.addWidget(dialog)

    selection = dialog.selection()
    assert selection["data_path"] == "/recon/ptycho/object_phase"
    assert selection["angle_path"] == "/recon/ptycho/rotation"
    assert selection["axis_order"] == (0, 1, 2)
    assert _ok(dialog).isEnabled()


def test_browser_selection_loads_into_the_window(qtbot, odd_file):
    """The whole point: a file the automatic probe cannot read still loads."""
    win = PtychoAlignWindow()
    qtbot.addWidget(win)

    # The conventional loader does not find it...
    with pytest.raises(KeyError):
        load_dataset(str(odd_file))

    # ...but an explicit dataset path does.
    dialog = Hdf5BrowserDialog(str(odd_file), win)
    qtbot.addWidget(dialog)
    assert win.load_path(str(odd_file), **dialog.selection())

    assert win.raw is not None
    assert win.raw.data.shape == (20, 24 + 16, 24 + 16)  # margin-padded phantom
    assert win.engine is not None


def test_browser_blocks_ok_on_an_angle_length_mismatch(qtbot, odd_file):
    """Right stack, wrong angle array: say so instead of loading garbage."""
    import h5py

    with h5py.File(odd_file, "a") as f:
        f.create_dataset("meta/ring_current", data=np.arange(99.0))

    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    dialog = Hdf5BrowserDialog(str(odd_file), win)
    qtbot.addWidget(dialog)

    wrong = next(e for e in dialog.entries if e.path == "/meta/ring_current")
    dialog._set_angles(wrong)
    dialog._validate()

    assert not _ok(dialog).isEnabled()
    assert "99 angles but 20" in dialog.status.text()


def test_browser_axis_order_offers_every_permutation(qtbot, odd_file):
    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    dialog = Hdf5BrowserDialog(str(odd_file), win)
    qtbot.addWidget(dialog)

    orders = [dialog.axis_combo.itemData(i) for i in range(dialog.axis_combo.count())]
    assert sorted(orders) == sorted(
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    )
    # Each is labelled by the shape it produces, so the right one is obvious.
    assert "angles=20" in dialog.axis_combo.itemText(orders.index((0, 1, 2)))


# -- crop and complex component ---------------------------------------------------------


@pytest.fixture
def auto_accept(monkeypatch):
    """Answer the modal warnings load_path raises (phase data has a negative integral)."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))


@pytest.fixture
def complex_file(tmp_path):
    """A complex stack under non-standard names, with a 4-D probe array to ignore."""
    import h5py

    path = tmp_path / "recon.h5"
    data, _sx, _sy = make_misaligned_dataset(size=24, n_angles=16, max_shift=1.0, seed=7)[:3]
    # Scale into (-pi, pi] so exp(1j*phase) does not wrap and np.angle round-trips.
    phase = data.data.astype(np.float32)
    phase = phase / (np.abs(phase).max() * 1.05) * np.pi
    with h5py.File(path, "w") as f:
        f.create_dataset("obj", data=(np.exp(1j * phase)).astype(np.complex64))
        f.create_dataset("angle", data=np.rad2deg(data.angles))
        f.create_dataset("pr", data=np.zeros((16, 1, 8, 8), dtype=np.complex64))
    return path, phase


def test_browser_enables_the_component_choice_only_for_complex_data(qtbot, complex_file, odd_file):
    win = PtychoAlignWindow()
    qtbot.addWidget(win)

    complex_dialog = Hdf5BrowserDialog(str(complex_file[0]), win)
    qtbot.addWidget(complex_dialog)
    assert complex_dialog.component_combo.isEnabled()
    assert complex_dialog.selection()["component"] == "phase"
    assert complex_dialog.selection()["data_path"] == "/obj"  # not the 4-D /pr

    real_dialog = Hdf5BrowserDialog(str(odd_file), win)
    qtbot.addWidget(real_dialog)
    assert not real_dialog.component_combo.isEnabled()


def test_loading_a_crop_of_a_complex_stack(qtbot, complex_file, auto_accept):
    from tktomo.ptycho_align.core import Crop

    path, phase = complex_file
    win = PtychoAlignWindow()
    qtbot.addWidget(win)

    assert win.load_path(
        str(path),
        data_path="/obj",
        angle_path="/angle",
        component="phase",
        crop=Crop(8, 24, 10, 30),
    )

    assert win.raw.data.shape == (16, 16, 20)
    # exp(1j*phase) round-trips through np.angle for |phase| < pi.
    np.testing.assert_allclose(win.raw.data[0], phase[0][8:24, 10:30], atol=1e-5)
    assert win.raw.metadata["crop"] == (8, 24, 10, 30)
    assert win.data_panel.crop_button.isEnabled()
    assert win._source["kwargs"]["data_path"] == "/obj"
    assert win._source_is_complex()


def test_crop_box_reports_the_size_and_clamps_to_the_stack(qtbot):
    from tktomo.ptycho_align.core import Crop
    from tktomo.ptycho_align.ui.panels import CropBox

    box = CropBox()
    qtbot.addWidget(box)
    box.set_full_shape((100, 200, 300))

    assert box.is_full_frame()
    assert box.crop() == Crop(0, 200, 0, 300)
    assert "100 x 200 x 300" in box.size_label.text()

    box.set_crop(Crop(10, 50, 20, 90))
    assert box.crop() == Crop(10, 50, 20, 90)
    assert not box.is_full_frame()

    # A crop from a bigger stack must be clamped, never allowed to read out of range.
    box.set_crop(Crop(0, 5000, 0, 5000))
    assert box.crop() == Crop(0, 200, 0, 300)

    box.reset_to_full()
    assert box.is_full_frame()


def test_crop_dialog_round_trips_the_selection(qtbot):
    from tktomo.ptycho_align.core import Crop
    from tktomo.ptycho_align.ui.panels import CropDialog

    dialog = CropDialog(
        (16, 64, 64), Crop(4, 40, 8, 56), "phase", complex_source=True, roi=Crop(1, 9, 2, 10)
    )
    qtbot.addWidget(dialog)

    assert dialog.selection()["crop"] == Crop(4, 40, 8, 56)
    assert dialog.component_combo.isEnabled()

    # The ROI drawn in the projection view becomes the crop.
    dialog.roi_button.click()
    assert dialog.selection()["crop"] == Crop(1, 9, 2, 10)

    dialog.component_combo.setCurrentText("amplitude")
    assert dialog.selection()["component"] == "amplitude"


def test_the_roi_maps_back_into_file_coordinates(qtbot, complex_file, auto_accept):
    """An ROI drawn on a cropped view indexes the crop, not the file."""
    from tktomo.ptycho_align.core import Crop

    path, _phase = complex_file
    win = PtychoAlignWindow()
    qtbot.addWidget(win)
    win.load_path(
        str(path), data_path="/obj", angle_path="/angle", crop=Crop(8, 24, 10, 30)
    )

    assert win._roi_as_file_crop() is None  # no ROI drawn yet

    win.projection_view.roi_check.setChecked(True)
    win.projection_view.roi.setPos((4, 2))  # (x=column, y=row) in pyqtgraph
    win.projection_view.roi.setSize((6, 5))
    win.projection_view.roi.show()

    mapped = win._roi_as_file_crop()
    assert mapped is not None
    # The ROI sits at row 2, column 4 of the crop, which starts at row 8, column 10.
    assert mapped.v0 == 2 + 8
    assert mapped.u0 == 4 + 10

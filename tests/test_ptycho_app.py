"""GUI tests for ptycho-align.

Exercises the step-wise workflow the whole tool exists for: load, preprocess, COM,
step, run, stop, revert, save/reload, and the binned-preview shift rescaling.
"""

from __future__ import annotations

import sys
from dataclasses import replace
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
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from tktomo.ptycho_align.core import center_is_plausible, load_dataset  # noqa: E402
from tktomo.ptycho_align.core.preprocess import bin_stack  # noqa: E402
from tktomo.ptycho_align.ui.main_window import PtychoAlignWindow  # noqa: E402
from tktomo.ptycho_align.ui.panels import Hdf5BrowserDialog  # noqa: E402
from tktomo.ptycho_align.ui.panels.base import (  # noqa: E402
    MODE_ALIGNED,
    MODE_DIFFERENCE,
    MODE_RAW,
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


def _engine(window):
    """The engine behind an in-process session.

    White-box, and valid only for a ``LocalSession`` -- these are tests of what the
    window does to local state. The behaviour that has to hold for a remote session too
    is pinned in ``test_session_conformance.py`` instead.
    """
    return window.session.host.engine


def _state(window):
    return _engine(window).state


def _run(window, n: int) -> None:
    """Run n iterations on the session's compute thread and wait for it to finish."""
    window._start_run(n)
    assert window._run_handle is not None, "the run did not start"
    window.session.wait(window._run_handle, timeout=120.0)
    window._run_finished()


def test_bin_stack_halves_the_detector_axes():
    stack = np.ones((3, 8, 10), dtype=np.float32)
    assert bin_stack(stack, 2).shape == (3, 4, 5)
    assert bin_stack(stack, 1).shape == stack.shape
    # An odd size is trimmed, not padded.
    assert bin_stack(np.ones((3, 9, 9), dtype=np.float32), 2).shape == (3, 4, 4)


def test_loading_populates_the_engine(window):
    assert window.session.summary().has_engine
    assert _engine(window) is not None
    assert _engine(window).iteration == 0


def test_preprocessing_pads_and_rebuilds_the_engine(window):
    before = _state(window).original.shape
    options = window.preprocess_panel.options()
    options.pad_percent = 20.0
    window._apply_preprocessing(options)

    after = _state(window).original.shape
    assert after[1] > before[1] and after[2] > before[2], "padding was not applied"
    assert _engine(window).iteration == 0


def test_com_prealignment_populates_shifts_and_centre(window):
    window._run_com("mean")

    assert window.session.summary().com is not None
    assert np.any(_state(window).sx != 0.0)
    assert _state(window).center > 0
    # The sinogram overlay and the fit plot both need the fitted curve.
    assert window.session.summary().com.fitted_u.shape == _state(window).angles.shape


def test_step_produces_a_volume_and_fills_every_view(window):
    window._run_com("mean")
    _run(window, 1)

    assert _engine(window).iteration == 1
    assert _state(window).volume is not None

    # The viewers pull one plane per mode; all four must have something to give.
    planes = window._planes
    aligned = planes.plane(MODE_ALIGNED, 0, 0)
    assert planes.plane(MODE_RAW, 0, 0) is not None
    assert aligned is not None
    assert planes.plane(MODE_REPROJECTION, 0, 0) is not None
    assert planes.plane(MODE_DIFFERENCE, 0, 0).shape == aligned.shape
    # A plane, not the stack behind it.
    assert aligned.shape == _state(window).original.shape[1:]


def test_running_iterations_reduces_the_residual(window):
    window._run_com("mean")
    _run(window, 4)

    history = _engine(window).history
    assert len(history) == 4
    assert history[-1].residual < history[0].residual


def test_stop_cancels_and_leaves_a_valid_state(window):
    window._run_com("mean")
    window._start_run(50)
    window._stop_run()  # cancel immediately
    window.session.wait(window._run_handle, timeout=120.0)
    window._run_finished()

    assert _engine(window).iteration < 50
    # Whatever it managed, the state must be complete and steppable.
    _run(window, 1)
    assert _state(window).volume is not None


def test_revert_rolls_back_the_iteration(window):
    window._run_com("mean")
    _run(window, 3)
    sx_at_1 = _engine(window).history[0].sx.copy()

    window._revert(1)

    assert _engine(window).iteration == 1
    np.testing.assert_allclose(_state(window).sx, sx_at_1)


def test_session_roundtrip_restores_the_state(window, tmp_path):
    window._run_com("mean")
    _run(window, 2)

    path = tmp_path / "session.h5"
    session_io.save_session(path, _engine(window))
    restored = session_io.load_session(path)

    assert restored.iteration == 2
    np.testing.assert_allclose(restored.state.sx, _state(window).sx)
    np.testing.assert_allclose(restored.state.sy, _state(window).sy)
    assert restored.state.center == pytest.approx(_state(window).center)
    assert restored.config.recon_algorithm == _engine(window).config.recon_algorithm


def test_bin_factor_rescales_the_existing_shifts(window):
    window._run_com("mean")
    sx_before = _state(window).sx.copy()
    center_before = _state(window).center

    window.align_panel.bin_spin.setValue(2)

    # Shifts are in pixels of the (now half-size) grid, so they halve with it.
    np.testing.assert_allclose(_state(window).sx, sx_before / 2, rtol=1e-6)
    assert _state(window).center == pytest.approx(center_before / 2)
    assert _state(window).original.shape[2] == pytest.approx(
        window.session.summary().raw_shape[2] // 2, abs=1
    )


def test_bin_factor_rescales_the_stored_com_result(window):
    window._run_com("mean")
    com_before = window.session.summary().com

    window.align_panel.bin_spin.setValue(2)

    # The COM result outlives the rebuild and is still in pixels, so it must follow the
    # grid. Left on the old scale it becomes the reference `_estimate_center` judges
    # against, and a correct estimate on the binned grid gets rejected as implausible.
    assert window.session.summary().com.center == pytest.approx(com_before.center / 2)
    np.testing.assert_allclose(window.session.summary().com.com_u, com_before.com_u / 2, rtol=1e-6)
    np.testing.assert_allclose(window.session.summary().com.fitted_u, com_before.fitted_u / 2, rtol=1e-6)
    assert window.session.summary().com.center == pytest.approx(_state(window).center)

    width = _state(window).original.shape[2]
    ok, reason = center_is_plausible(_state(window).center, width, window.session.summary().com.center)
    assert ok, reason


class _StuckRun:
    """A session that reports a run in flight for the first few polls.

    Models the case that matters: a reconstruction that outlives any *fixed* timeout,
    because a tomopy call already running cannot be interrupted. Cancelling records the
    request but does not stop it.
    """

    def __init__(self, session, polls_needed: int = 3) -> None:
        self._session = session
        self._polls = 0
        self._polls_needed = polls_needed
        self.cancelled = False
        self.running = True

    def summary(self, since_iteration: int = 0):
        self._polls += 1
        if self._polls >= self._polls_needed:
            self.running = False
        return replace(self._session.summary(since_iteration), running=self.running)

    def cancel_run(self) -> None:
        self.cancelled = True

    def __getattr__(self, name):
        return getattr(self._session, name)


def test_closing_mid_run_waits_however_long_the_compute_takes(window):
    window.session = stuck = _StuckRun(window.session)
    window._summary = replace(window._summary, running=True)

    window.close()

    # The old code called wait() once with a 30 s cap and tore the window down anyway
    # when it expired -- destroying a live thread mid-reconstruction, which can segfault
    # inside tomopy's shared-memory cleanup. A cancelled run must be waited out.
    assert stuck.cancelled, "the run was not cancelled on close"
    assert not stuck.running, "the window was destroyed while compute was still running"


def test_bin_change_is_refused_while_a_run_is_in_flight(window, monkeypatch):
    nagged = []
    monkeypatch.setattr(
        "tktomo.ptycho_align.ui.main_window.QMessageBox.information",
        lambda *args, **kwargs: nagged.append(args),
    )
    window._run_com("mean")
    engine_before = _engine(window)
    # Exactly what a live run does to the host: exclusive verbs are refused outright.
    window.session.host._run_active = True
    try:
        window.align_panel.bin_spin.setValue(2)

        # Rebuilding here would swap the engine out from under the compute thread, which
        # keeps reconstructing into an engine nothing reads: the iteration is silently
        # discarded -- minutes of work, no result, no error.
        assert _engine(window) is engine_before
        assert window.session.summary().bin_factor == 1
        assert window.align_panel.bin_factor() == 1, "the spin box was not put back"
        assert nagged, "the user was not told why the bin change was refused"
    finally:
        # Unconditionally: leaving it set makes closeEvent wait for a run that will never
        # finish, and a failed assertion would hang the suite rather than report itself.
        window.session.host._run_active = False


def test_run_footprint_is_cubic_in_width_so_binning_dominates_it(window):
    full, _text = window._run_footprint(5)

    window.align_panel.bin_spin.setValue(2)
    binned, _text = window._run_footprint(5)

    # The volumes are (rows, width, width), so halving the detector axes cuts them 8x.
    # The cached stacks only shrink 4x, so the total lands between the two -- the point
    # being that it is dominated by the cubic term, which is what makes binning the
    # effective lever.
    assert binned < full / 4


def test_a_run_that_would_exhaust_memory_is_refused(window, monkeypatch):
    footprint, _text = window._run_footprint(5)
    monkeypatch.setattr(
        "tktomo.ptycho_align.session.engine_host.available_ram_bytes", lambda: footprint // 2
    )
    warnings = []

    def refuse(*args, **kwargs):
        warnings.append(args)
        return QMessageBox.No

    monkeypatch.setattr(
        "tktomo.ptycho_align.ui.main_window.QMessageBox.warning", staticmethod(refuse)
    )

    window._start_run(5)

    # An OOM kill leaves no traceback and no log line -- the window just disappears -- so
    # the run must not start without the user having been told.
    assert warnings, "the user was not warned before a run that cannot fit in RAM"
    assert window._run_handle is None, "the run started despite the warning being declined"


def test_a_runaway_iteration_stops_the_run_immediately(window, qtbot, monkeypatch):
    monkeypatch.setattr(
        "tktomo.ptycho_align.ui.main_window.QMessageBox.warning",
        staticmethod(lambda *args, **kwargs: QMessageBox.Ok),
    )
    window._run_com("mean")
    # The engine should have taken the COM amplitude as its yardstick for a real shift.
    assert _engine(window).com_amplitude == pytest.approx(window.session.summary().com.amplitude)

    # Force every iteration to look like the noise-matching failure: a shift update far
    # larger than the object's actual motion.
    monkeypatch.setattr(
        "tktomo.ptycho_align.core.engine.shift_update_is_runaway",
        lambda error, width, amplitude=None, **kwargs: "forced runaway",
    )

    window._start_run(5)
    # waitUntil pumps the event loop. Blocking on run.wait() instead would deadlock the
    # very thing under test: the cancel rides a queued signal to the GUI thread, which a
    # blocked main thread never delivers.
    qtbot.waitUntil(lambda: not window.session.summary().running, timeout=120_000)
    window._run_finished()

    # It must stop at the first bad iteration, not run all five: the shifts are
    # cumulative, so each further iteration warps the data by a larger bogus offset.
    assert _engine(window).iteration == 1, "the run did not stop at the runaway iteration"
    assert _engine(window).history[-1].runaway


def test_iteration_cost_is_square_in_width_and_linear_in_inner_iterations():
    from tktomo.ptycho_align.core.estimates import iteration_cost_units

    base = iteration_cost_units(410, 33, 1137, "sirt", 2)
    # Doubling the width is 4x the work -- this is why the pad hurts and binning helps.
    assert iteration_cost_units(410, 33, 2274, "sirt", 2) == pytest.approx(4 * base)
    assert iteration_cost_units(410, 33, 1137, "sirt", 4) == pytest.approx(2 * base)
    # gridrec/fbp are direct: num_iter does not apply, so it must not inflate the cost.
    direct = iteration_cost_units(410, 33, 1137, "gridrec", 2)
    assert iteration_cost_units(410, 33, 1137, "gridrec", 8) == pytest.approx(direct)


def test_a_timed_iteration_predicts_what_a_finer_grid_would_cost(window):
    window._run_com("mean")
    _run(window, 1)

    measured = _engine(window).history[-1].wallclock_s
    assert window._run_duration(1) == pytest.approx(measured, rel=0.5)

    # The calibration is per-machine, not per-grid, so it survives a rebuild and can warn
    # about a grid that has never been run -- which is the only moment the warning helps.
    coarse = window._run_duration(1)
    window.align_panel.bin_spin.setValue(2)
    assert window._run_duration(1) < coarse


def test_the_resource_view_samples_cpu_and_memory(qtbot):
    from tktomo.ptycho_align.ui.panels import ResourceView

    view = ResourceView()
    qtbot.addWidget(view)
    view.sample()
    view.sample()  # CPU is a rate: it needs two readings to mean anything

    assert view.peak_rss_gb() > 0
    assert "GB" in view.info_label.text()


def test_align_toggles_reach_the_engine(window):
    window.align_panel.horizontal_check.setChecked(False)
    window._run_com("mean")
    _run(window, 1)

    result = _engine(window).history[-1]
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

    assert _engine(window).last_simulated.flags.owndata
    assert _state(window).volume.flags.owndata


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
        session_io.export_volume(tmp_path / "v.h5", _state(window).volume)


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

    summary = win.session.summary()
    assert summary.has_engine
    assert summary.raw_shape == (20, 24 + 16, 24 + 16)  # margin-padded phantom


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

    summary = win.session.summary()
    assert summary.raw_shape == (16, 16, 20)
    # exp(1j*phase) round-trips through np.angle for |phase| < pi.
    np.testing.assert_allclose(
        win.session.read_plane(MODE_RAW, 0, 0), phase[0][8:24, 10:30], atol=1e-5
    )
    assert summary.dataset.crop == (8, 24, 10, 30)
    assert win.data_panel.crop_button.isEnabled()
    source = win._source()
    assert source["kwargs"]["data_path"] == "/obj"
    assert win._source_is_complex(source)


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

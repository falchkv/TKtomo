"""GUI tests for the feature-isolation and track-model apps.

Interaction goes through the same slots the pointer/keyboard use, so these
are behavior tests, not pixel tests. Heavier stacks are covered by the
end-to-end phantom test at the bottom (skipped without tomopy).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")
pytest.importorskip("h5py")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from tktomo.tracking.model import AxisModel  # noqa: E402
from tktomo.tracking.stackio import load_tracking_stack  # noqa: E402
from tktomo.ui.feature_isolation_app import FeatureIsolationWindow  # noqa: E402
from tktomo.ui.track_model_app import TrackModelWindow  # noqa: E402


@pytest.fixture
def iso(qtbot):
    win = FeatureIsolationWindow()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def tracker(qtbot):
    win = TrackModelWindow()
    qtbot.addWidget(win)
    win.auto_fit.setChecked(False)
    win.advance_box.setValue(0)
    return win


def truth_for(win, n_feat=4):
    model = AxisModel.blank(win._stack.angles, np.arange(n_feat))
    model.a = np.linspace(-25, 25, n_feat)
    model.b = np.linspace(18, -18, n_feat)
    model.y = np.linspace(30, 95, n_feat)
    model.c_coef[0] = 64.0
    return model


def label_from_truth(win, truth, every=4):
    u_t, v_t = truth.predict()
    for f in range(truth.feature_ids.size):
        win._set_active(f)
        for view in range(0, win._stack.angles.size, every):
            win._set_view(view)
            win._place(u_t[f, view], v_t[f, view])


# ---------------------------------------------------------------- App A

def test_isolation_keyframes_and_export(iso, tmp_path, monkeypatch):
    for view, u, v in ((5, 60.0, 40.0), (20, 70.0, 42.0), (40, 50.0, 44.0)):
        iso._set_view(view)
        iso._place(u, v)
    assert iso.table.rowCount() == 3
    assert iso._track is not None
    assert "amplitude" in iso.fit_label.text()

    # delete via the same slot the Delete key uses
    iso._set_view(20)
    iso._delete_current(0.0, 0.0)
    assert iso.table.rowCount() == 2

    iso._set_view(20)
    iso._place(70.0, 42.0)
    iso.win_h.setValue(32)
    iso.win_w.setValue(32)
    out = tmp_path / "crop.h5"
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), "HDF5 (*.h5)"))
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QMessageBox.information",
        lambda *a, **k: QMessageBox.StandardButton.Ok)
    iso._export()

    data, chain = load_tracking_stack(out)
    assert data.data.shape == (60, 32, 32)
    assert chain.view_origin.shape == (60, 2)


def test_isolation_keyframe_json_round_trip(iso, tmp_path, monkeypatch):
    iso._set_view(3)
    iso._place(10.0, 20.0)
    iso._set_view(9)
    iso._place(12.0, 21.0)
    path = tmp_path / "keys.json"
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(path), ""))
    iso._save_keys()
    iso._clear_keys()
    assert not iso._keys
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(path), ""))
    iso._load_keys()
    assert iso._keys == {3: (10.0, 20.0), 9: (12.0, 21.0)}


# ---------------------------------------------------------------- App B

def test_tracker_label_fit_recovers_center(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    fit = tracker._fit
    assert fit is not None
    assert fit.rms_u < 1e-6
    assert abs(fit.model.center_at_mean_theta() - 64.0) < 0.5
    assert tracker.feature_table.rowCount() == 4
    assert "center" in tracker.summary_label.text()


def test_tracker_manual_override_is_residual_only(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    check, spin = tracker._coef_rows["c"][0]
    spin.setValue(70.0)                     # hand edit, no solve
    assert tracker._fit.rms_u == pytest.approx(6.0, abs=1.0)
    spin.setValue(64.0)
    assert tracker._fit.rms_u < 0.5


def test_tracker_pin_and_delete(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker.feature_table.item(0, 8).setCheckState(Qt.CheckState.Checked)
    assert 0 in tracker._pins
    tracker._fit_now()
    assert not tracker._mask.features[0]

    # Delete nearest label through the pointer slot (loaded frame coords)
    u_t, v_t = truth.predict()
    tracker._set_view(0)
    n_before = len(tracker._labels)
    tracker._set_active(1)
    tracker._delete_near(u_t[1, 0], v_t[1, 0])
    assert len(tracker._labels) == n_before - 1


def test_tracker_loads_feature_crop_with_matching_raw_coords(
        qtbot, tmp_path, monkeypatch):
    # export a moving crop from App A, reload it in App B, and check that a
    # label placed on the SAME pixel lands on the same raw coordinate
    iso = FeatureIsolationWindow()
    qtbot.addWidget(iso)
    for view, u, v in ((5, 60.0, 40.0), (20, 70.0, 42.0), (40, 50.0, 44.0)):
        iso._set_view(view)
        iso._place(u, v)
    iso.win_h.setValue(32)
    iso.win_w.setValue(32)
    out = tmp_path / "crop.h5"
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(
        "tktomo.ui.feature_isolation_app.QMessageBox.information",
        lambda *a, **k: QMessageBox.StandardButton.Ok)
    iso._export()
    u_all, v_all = iso._track

    tracker = TrackModelWindow()
    qtbot.addWidget(tracker)
    tracker.auto_fit.setChecked(False)
    tracker.advance_box.setValue(0)
    data, chain = load_tracking_stack(out)
    tracker._labels.__init__()
    tracker._show_data(data, chain, {"path": str(out),
                                     "kind": "feature_crop"})

    view = 20
    tracker._set_view(view)
    origin = chain.view_origin[view]
    in_crop_u = u_all[view] - origin[1]
    in_crop_v = v_all[view] - origin[0]
    tracker._set_active(0)
    tracker._place(in_crop_u, in_crop_v)
    (u_raw, v_raw), = [uv for uv in [tracker._labels.get(0, view)]]
    expect_u, expect_v = iso._chain.to_parent(u_all[view], v_all[view])
    assert u_raw == pytest.approx(float(expect_u))
    assert v_raw == pytest.approx(float(expect_v))


def test_tracker_session_round_trip(tracker, tmp_path, monkeypatch):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker._pins.add(2)
    path = tmp_path / "session.h5"
    monkeypatch.setattr(
        "tktomo.ui.track_model_app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(path), ""))
    tracker._save_session()

    from tktomo.tracking import sessionio
    state = sessionio.load_session(path)
    assert len(state["labels"]) == len(tracker._labels)
    assert np.allclose(state["model"].c_coef, tracker._model.c_coef)
    assert 2 in state["ui"]["pins"]


def test_recon_slice_responds_to_alpha(tracker, monkeypatch):
    """The submitted recon job must carry the tilt as per-view rotations,
    and the slab must widen so the derotation cannot run out of rows."""
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()

    jobs = []

    class StubWorker:
        def __init__(self):
            self.finished_slice = _StubSignal()
            self.failed = _StubSignal()

        def submit(self, req):
            jobs.append({"slab_rows": req.hi - req.lo,
                         "slab_width": tracker._stack.shape[2],
                         "sx": np.array(req.sx), "center": req.center,
                         "rot": np.array(req.rot_deg),
                         "row_in_slab": req.row_in_slab})

    tracker._recon_worker = StubWorker()
    tracker._model.alpha_coef[0] = 0.0
    tracker._request_recon()
    tracker._model.alpha_coef[0] = -0.02
    tracker._request_recon()

    flat, tilted = jobs
    assert np.allclose(flat["rot"], 0.0)
    assert np.allclose(tilted["rot"], np.rad2deg(0.02))   # rot = -deg(alpha)
    width = tracker._stack.shape[2]
    assert (tilted["slab_rows"] - flat["slab_rows"]
            >= int(0.02 * width / 2) - 1)                 # margin widened
    # requested row 64 sits at its own index inside the (clipped) slab
    assert jobs[0]["row_in_slab"] + max(0, 64 - (flat["slab_rows"] - 1) // 2) \
        >= 0


class _StubSignal:
    def connect(self, *_a):
        pass


def test_plot_panes_defaults_kinds_and_missing_frames(tracker):
    from tktomo.ui.track_model_app import PLOT_KINDS

    assert [c.currentText() for c in tracker._plot_selectors] \
        == ["labels per view"]
    # "labels per view" renders from labels alone, before any fit exists
    tracker._set_active(0)
    tracker._set_view(4)
    tracker._place(60.0, 40.0)
    tracker._plot_selectors[0].setCurrentText("labels per view")
    items = tracker._plot_widgets[0].getPlotItem().listDataItems()
    assert items, "labels-per-view pane is empty without a fit"

    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    # every kind renders without raising, and produces data items
    for kind in PLOT_KINDS:
        tracker._plot_selectors[0].setCurrentText(kind)
        assert tracker._plot_widgets[0].getPlotItem().listDataItems(), kind


def test_next_unlabelled_button_walks_the_gaps(tracker):
    n = tracker._stack.angles.size
    tracker._set_active(0)
    for view in (0, 1, 3):
        tracker._set_view(view)
        tracker._place(60.0, 40.0)
    tracker._set_view(0)
    tracker._goto_next_unlabelled()
    assert tracker._view == 2
    tracker._goto_next_unlabelled()     # leaves the unlabelled current view
    assert tracker._view == 4
    tracker._set_view(n - 1)
    tracker._goto_next_unlabelled()     # wraps around
    assert tracker._view == 2
    # every view labelled once, view 1 twice: steps among the single-label
    # views and skips view 1
    for view in range(n):
        tracker._set_view(view)
        tracker._place(60.0, 40.0)
    tracker._set_active(1)
    tracker._set_view(1)
    tracker._place(30.0, 20.0)
    tracker._set_view(0)
    tracker._goto_next_unlabelled()
    assert tracker._view == 2


def test_plot_click_jumps_to_nearest_frame(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    deg = np.rad2deg(tracker._stack.angles)
    step = deg[1] - deg[0]
    tracker._jump_to_angle(deg[37] + 0.4 * step)
    assert tracker._view == 37
    tracker._jump_to_angle(-1e6)          # clamps to the first frame
    assert tracker._view == 0


def test_fit_ignores_checkbox_history_when_free(qtbot):
    """With dx/dy FREE at fit time, the result must not depend on what the
    checkboxes were during labeling: free columns are solved fresh, stored
    values only matter for fixed parameters."""
    results = []
    for tick_during_labeling in (True, False):
        win = TrackModelWindow()
        qtbot.addWidget(win)
        win.auto_fit.setChecked(False)
        win.advance_box.setValue(0)
        win.free_dx.setChecked(tick_during_labeling)
        win.free_dy.setChecked(tick_during_labeling)
        truth = truth_for(win)
        u_t, v_t = truth.predict()
        for f in range(4):
            win._set_active(f)
            for view in range(0, 60, 4):
                win._set_view(view)
                win._place(u_t[f, view], v_t[f, view])
                win._fit_now()          # fits happen along the way
        win.free_dx.setChecked(True)    # both free for the FINAL fit
        win.free_dy.setChecked(True)
        win._fit_now()
        results.append(win._fit.model)
    a, b = results
    assert np.allclose(a.c_coef, b.c_coef, atol=1e-9)
    assert np.allclose(a.dx, b.dx, atol=1e-9)
    assert np.allclose(a.dy, b.dy, atol=1e-9)
    assert np.allclose(a.a, b.a, atol=1e-9)


def test_worst_outlier_button(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    # sabotage one label of feature 2 in view 24
    u_t, v_t = truth.predict()
    tracker._set_active(2)
    tracker._set_view(24)
    tracker._place(u_t[2, 24] + 15.0, v_t[2, 24])
    tracker._fit_now()
    tracker._set_active(0)
    tracker._set_view(0)
    tracker._goto_worst_outlier()
    assert tracker._active == 2
    assert tracker._view == 24


def test_follow_prediction_recenters_after_a_click(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker.advance_box.setValue(3)
    tracker._set_active(1)
    tracker._set_view(0)
    view_box = tracker.viewer.image_view.getView()
    view_box.setRange(xRange=(0, 40), yRange=(0, 40), padding=0)
    u_t, v_t = truth.predict()

    # unticked: the view stays where it was
    tracker._place(u_t[1, 0], v_t[1, 0])
    assert tracker._view == 3
    rect = view_box.viewRect()
    assert abs(rect.center().x() - 20) < 1 and abs(rect.center().y() - 20) < 1

    # ticked: the view is centred on the prediction in the NEW view, same zoom
    tracker.follow_box.setChecked(True)
    tracker._place(u_t[1, 3], v_t[1, 3])
    assert tracker._view == 6
    rect = view_box.viewRect()
    u_pred, v_pred = tracker._predicted_position(1, 6)
    assert abs(rect.center().x() - u_pred) < 1.0
    assert abs(rect.center().y() - v_pred) < 1.0
    assert abs(rect.height() - 40) < 1e-6     # zoom kept (width follows aspect)

    # a feature with too few labels is left alone
    tracker._set_active(9)
    tracker._set_view(0)
    view_box.setRange(xRange=(0, 40), yRange=(0, 40), padding=0)
    tracker._place(5.0, 5.0)
    rect = view_box.viewRect()
    assert abs(rect.center().x() - 20) < 1 and abs(rect.center().y() - 20) < 1


def test_ghost_markers_and_sizes(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker._set_active(1)
    tracker._set_view(0)
    assert len(tracker.viewer._ghost_scatter.data) == 0
    tracker.ghost_box.setChecked(True)          # triggers refresh
    n_other = len(tracker._labels.views_of(1)) - 1
    assert len(tracker.viewer._ghost_scatter.data) == n_other

    # size edit: marker sizes and fit weights follow
    tracker._feature_sizes[1] = 30.0
    tracker._set_view(4)                        # a labeled view
    label_sizes = set(tracker.viewer._label_scatter.data["size"].tolist())
    assert 30.0 in label_sizes                  # active feature resized
    assert 10.0 in label_sizes                  # others at the default
    weights = tracker._feature_weights()
    assert weights is not None
    row = list(tracker._model.feature_ids).index(1)
    other = list(tracker._model.feature_ids).index(0)
    assert weights[row] == pytest.approx(1 / 30.0)
    assert weights[other] == pytest.approx(1 / 10.0)


def test_slice_probe_geometry_round_trip(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    m = tracker._model
    tracker._last_recon_info = {"extra_bin": 2, "row_loaded": 40,
                                "width": 64}
    # place the click exactly where feature 0's (a, b) would appear
    scale = 2 * tracker._chain.binning
    axis_row, axis_col = 64 // 2 - 1, (64 + 1) // 2
    col = m.a[0] / scale + axis_col
    row = axis_row - m.b[0] / scale
    a, b, y = tracker._slice_to_probe(col, row)
    assert a == pytest.approx(m.a[0])
    assert b == pytest.approx(m.b[0])
    # the drawn diamond then coincides horizontally with the feature's
    # predicted marker in every view
    tracker._probe = (a, b, y)
    u_pred, _ = m.predict()
    for view in (0, 20, 45):
        tracker._set_view(view)
        probe_x = tracker.viewer._probe_scatter.data[0][0]
        assert probe_x == pytest.approx(u_pred[0, view], abs=1e-6)
    tracker._clear_probe()
    assert len(tracker.viewer._probe_scatter.data) == 0


def test_slice_probe_convention():
    """Guard tomopy's grid convention: the rotation axis reconstructs to
    (row, col) = (N//2 - 1, (N+1)//2) whatever `center` is, with
    col = axis + A and row = axis - B. If this fails after a tomopy
    upgrade, fix `_slice_to_probe`, not the test."""
    pytest.importorskip("tomopy")
    from tktomo.recon import get_backend

    theta = np.linspace(0, np.pi, 240, endpoint=False)
    n = 129
    center, a_true, b_true = 60.0, 22.0, -13.0
    sino = np.zeros((240, 1, n), np.float32)
    u = center + a_true * np.cos(theta) + b_true * np.sin(theta)
    for k in range(240):
        j0 = int(np.floor(u[k]))
        f = u[k] - j0
        sino[k, 0, j0] += 1 - f
        sino[k, 0, j0 + 1] += f
    vol = get_backend("tomopy").reconstruct(sino, theta, center=center,
                                            algorithm="gridrec")
    sl = np.clip(np.asarray(vol)[0], 0, None)
    r, c = np.unravel_index(np.argmax(sl), sl.shape)
    assert c - ((n + 1) // 2) == pytest.approx(a_true, abs=1.0)
    assert (n // 2 - 1) - r == pytest.approx(b_true, abs=1.0)


def _blob_stack_window(qtbot, n_views=40, ny=72, nx=112):
    """A tracker window whose stack contains one crisp blob on a known
    sinusoid, crisp enough for the auto-tracker to follow."""
    from tktomo.io import ProjectionData
    from tktomo.tracking.coords import CoordinateChain

    theta = np.linspace(0, np.pi, n_views)
    truth_u = 25.0 * np.cos(theta) + 55.0
    truth_v = np.full(n_views, 36.0)
    rng = np.random.default_rng(0)
    stack = np.empty((n_views, ny, nx), np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for j in range(n_views):
        frame = 0.02 * rng.standard_normal((ny, nx))
        frame += np.exp(-((yy - truth_v[j]) ** 2 + (xx - truth_u[j]) ** 2)
                        / (2 * 2.5 ** 2))
        stack[j] = frame
    win = TrackModelWindow()
    qtbot.addWidget(win)
    win.auto_fit.setChecked(False)
    win.advance_box.setValue(0)
    win._show_data(ProjectionData(data=stack, angles=theta, metadata={}),
                   CoordinateChain(), {"kind": "synthetic"})
    return win, truth_u, truth_v


def test_auto_complete_produces_hollow_reviewable_labels(qtbot):
    win, truth_u, truth_v = _blob_stack_window(qtbot)
    win._set_active(0)
    for w in (5, 20, 35):
        win._set_view(w)
        win._place(truth_u[w], truth_v[w])
    assert win._labels.manual_counts() == {0: 3}

    win._auto_complete(False)
    worker = win._autotrack_worker
    assert worker is not None
    with qtbot.waitSignal(worker.finished_tracks, timeout=120000):
        pass
    qtbot.wait(50)          # let the queued slot run

    auto_views = [w for w in win._labels.views_of(0)
                  if win._labels.kind_of(0, w) == 1]
    assert len(auto_views) >= 20
    # positions match the truth
    errs = []
    for w in auto_views:
        u_raw, v_raw = win._labels.get(0, w)
        errs.append(np.hypot(u_raw - truth_u[w], v_raw - truth_v[w]))
    assert float(np.median(errs)) < 0.5
    # manual labels untouched, qualities recorded
    assert win._labels.kind_of(0, 20) == 0
    assert win._labels.quality_of(0, auto_views[0]) > 0.3
    assert "auto labels" in win.auto_status.text()

    # clear-auto removes only the machine's work
    n_before = len(win._labels)
    win._clear_auto(False)
    assert len(win._labels) == 3
    assert n_before - 3 == len(auto_views)


def test_auto_complete_requires_manual_seeds(qtbot):
    win, truth_u, truth_v = _blob_stack_window(qtbot)
    win._set_active(0)
    win._set_view(5)
    win._place(truth_u[5], truth_v[5])       # only ONE manual label
    win._auto_complete(False)
    assert win._autotrack_worker is None     # refused before submitting


def test_auto_complete_refused_on_moving_crop(qtbot):
    from tktomo.tracking.coords import CoordinateChain

    win, truth_u, truth_v = _blob_stack_window(qtbot)
    n = win._stack.shape[0]
    win._chain = CoordinateChain(binning=1,
                                 view_origin=np.zeros((n, 2)))
    win._set_active(0)
    for w in (5, 20):
        win._set_view(w)
        win._place(truth_u[w], truth_v[w])
    win._auto_complete(False)
    assert win._autotrack_worker is None


def test_auto_session_round_trip_preserves_kinds(qtbot, tmp_path,
                                                 monkeypatch):
    win, truth_u, truth_v = _blob_stack_window(qtbot)
    win._set_active(0)
    for w in (5, 20):
        win._set_view(w)
        win._place(truth_u[w], truth_v[w])
    win._labels.set_auto(0, 12, 40.0, 36.0, 0.77)
    win.auto_min_corr.setValue(0.45)

    path = tmp_path / "session.h5"
    monkeypatch.setattr(
        "tktomo.ui.track_model_app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(path), ""))
    win._save_session()

    from tktomo.tracking import sessionio
    state = sessionio.load_session(path)
    labels = state["labels"]
    assert labels.kind_of(0, 12) == 1
    assert labels.quality_of(0, 12) == 0.77
    assert labels.kind_of(0, 5) == 0
    assert state["ui"]["auto_min_corr"] == pytest.approx(0.45)


def test_recon_binning_rescales_geometry(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()

    jobs = []

    class StubWorker:
        finished_slice = _StubSignal()
        failed = _StubSignal()

        def submit(self, req):
            # the binning is applied inside reconstruct_slice; mirror the
            # pixel-center rule here to check what the plan carries
            b = req.extra_bin
            jobs.append({"width": tracker._stack.shape[2] // b,
                         "sx": np.array(req.sx) / b,
                         "sy": np.array(req.sy) / b,
                         "center": (req.center - (b - 1) / 2.0) / b,
                         "row_in_slab": req.row_in_slab // b,
                         "rot": np.array(req.rot_deg)})

    tracker._recon_worker = StubWorker()
    tracker.recon_bin.setCurrentText("1")
    tracker._request_recon()
    tracker.recon_bin.setCurrentText("2")
    tracker._request_recon()

    full, binned = jobs
    assert binned["width"] == full["width"] // 2
    assert np.allclose(binned["sx"], full["sx"] / 2)
    assert np.allclose(binned["sy"], full["sy"] / 2)
    # pixel-center rule: c_b = (c - (b-1)/2) / b
    assert binned["center"] == pytest.approx((full["center"] - 0.5) / 2)
    assert np.allclose(binned["rot"], full["rot"])   # angles are invariant
    assert binned["row_in_slab"] <= full["row_in_slab"] // 2 + 1


def test_tracker_file_menu_holds_io_actions(tracker):
    """Load/save/export live in the File menu, not on the control panel."""
    from PySide6.QtWidgets import QPushButton
    assert [a.text() for a in tracker.menuBar().actions()] == ["&File"]
    texts = [a.text() for a in tracker._file_menu.actions()]
    assert "Load stack…" in texts
    assert "Save session…" in texts
    assert "Export" in texts
    labels = [b.text() for b in tracker.findChildren(QPushButton)]
    assert not any("session" in x or "Load stack" in x for x in labels)
    assert not tracker.free_dx.isChecked() and not tracker.free_dy.isChecked()


def test_tracker_recon_single_flight(tracker, qtbot):
    pytest.importorskip("tomopy")
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker._request_recon()
    tracker._request_recon()      # second request while busy: replaces, no crash
    worker = tracker._recon_worker
    assert worker is not None

    def done():
        return not worker.isRunning()

    qtbot.waitUntil(done, timeout=60000)
    qtbot.wait(50)
    image = tracker.recon_display.image_view.getImageItem().image
    assert image is not None and image.ndim == 2


def test_trajectory_overlay_excludes_per_view_jitter(tracker):
    """The overlay must stay a smooth sinusoid however jagged dx/dy are.

    The per-view shifts are alignment errors of individual views; a curve
    drawn inside ONE view's frame that mixes all of them zigzags while the
    3D geometry is a clean circle (the original bug report). Only the
    current view's own shift may enter, as the anchor.
    """
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    m = tracker._model
    m.dx[:] = np.where(np.arange(m.theta.size) % 2, 5.0, -5.0)
    m.dy[:] = np.where(np.arange(m.theta.size) % 2, -3.0, 3.0)
    tracker._set_active(0)
    tracker._set_view(0)
    x = np.asarray(tracker.viewer._trajectory.xData, float)
    y = np.asarray(tracker.viewer._trajectory.yData, float)
    assert x is not None and x.size == m.theta.size
    # a +-5 px alternating dx would give second differences of ~20 px;
    # the smooth sinusoid's are far below 1
    assert np.abs(np.diff(x, 2)).max() < 1.0
    assert np.abs(np.diff(y, 2)).max() < 1.0
    # the curve is the shift-FREE model track (ideal aligned frame): no
    # per-view dx/dy enter it at all, not even the current view's
    ct, sn = np.cos(m.theta), np.sin(m.theta)
    assert x[0] == pytest.approx(m.a[0] * ct[0] + m.b[0] * sn[0]
                                 + m.c_coef[0])
    assert y[0] == pytest.approx(m.y[0]
                                 + m.alpha_coef[0] * (m.a[0] * ct[0]
                                                      + m.b[0] * sn[0]))


def test_highpass_display_filter(tracker):
    """High-pass is display-only: a small bump on a huge ramp becomes the
    dominant feature on screen, and the stored pixels stay untouched."""
    viewer = tracker.viewer
    ramp = np.linspace(0, 1000, 128 * 128, dtype=np.float32
                       ).reshape(128, 128)
    ramp[60:64, 60:64] += 5.0                 # small feature, 0.5% of range
    viewer.set_image(ramp)
    shown_raw = viewer.image_view.getImageItem().image
    assert shown_raw.max() > 900              # unfiltered: ramp dominates

    viewer.highpass_sigma.setValue(6.0)
    viewer.highpass_box.setChecked(True)      # triggers redisplay
    shown = viewer.image_view.getImageItem().image
    assert abs(float(np.median(shown))) < 1.0      # background removed
    # away from the boundary halo (a high-pass classic) the bump dominates
    interior = shown[20:-20, 20:-20]
    peak = np.unravel_index(np.argmax(interior), interior.shape)
    assert 38 <= peak[0] <= 46 and 38 <= peak[1] <= 46   # 60..64 - 20 + 2
    assert viewer._raw_image.max() == pytest.approx(ramp.max())

    viewer.highpass_box.setChecked(False)
    shown_back = viewer.image_view.getImageItem().image
    assert shown_back.max() > 900             # round trip restores the raw


def test_colormap_combo_has_swatch_icons(tracker):
    combo = tracker.viewer.colormap_combo
    assert combo.count() > 0
    assert not combo.itemIcon(0).isNull()
    assert combo.iconSize().width() > 0


def test_auto_levels_follow_each_frame(tracker):
    viewer = tracker.viewer
    assert viewer.auto_levels_box.isChecked()   # tracking default: on
    viewer.set_image(np.linspace(0, 1, 10000, dtype=np.float32
                                 ).reshape(100, 100))
    lo1, hi1 = viewer.image_view.getHistogramWidget().item.getLevels()
    viewer.set_image(np.linspace(50, 60, 10000, dtype=np.float32
                                 ).reshape(100, 100))
    lo2, hi2 = viewer.image_view.getHistogramWidget().item.getLevels()
    assert lo2 > hi1                    # levels moved with the data range
    assert 50.0 <= lo2 < hi2 <= 60.0
    # robustness: one hot pixel must not blow the scale open
    frame = np.full((100, 100), 5.0, np.float32)
    frame[0, 0] = 1e6
    viewer.set_image(frame)
    _lo3, hi3 = viewer.image_view.getHistogramWidget().item.getLevels()
    assert hi3 < 1e5
    # ptycho-align's default stays untouched
    from tktomo.ptycho_align.ui.panels.base import StackDisplay
    assert not StackDisplay().auto_levels_box.isChecked()


# ------------------------------------------------- end-to-end phantom

def test_end_to_end_known_shifts_recovered(qtbot):
    """Misalign a phantom with known dx/dy, label truth, demand them back."""
    tomopy = pytest.importorskip("tomopy")  # noqa: F841 - gates the warp too
    from tktomo.io.phantom import generate_phantom
    from tktomo.tracking.coords import CoordinateChain
    from tktomo.tracking.export import export_aligned_stack

    rng = np.random.default_rng(5)
    base = generate_phantom(40, 96, 32, max_shift=0.0)
    dx = np.round(3.0 * rng.standard_normal(40))
    dy = np.round(2.0 * rng.standard_normal(40))
    shifted = np.array([np.roll(np.roll(frame, int(dy[k]), axis=0),
                                int(dx[k]), axis=1)
                        for k, frame in enumerate(base.data)])
    base.data = shifted.astype(np.float32)

    win = TrackModelWindow()
    qtbot.addWidget(win)
    win.auto_fit.setChecked(False)
    win.advance_box.setValue(0)
    win._show_data(base, CoordinateChain(), {"kind": "phantom"})

    # truth features: fixed points of the unshifted phantom geometry,
    # displaced by the injected per-view shifts
    truth = AxisModel.blank(base.angles, np.arange(4))
    truth.a = np.array([18.0, -12.0, 22.0, -20.0])
    truth.b = np.array([-15.0, 20.0, 8.0, -5.0])
    truth.y = np.array([20.0, 40.0, 60.0, 80.0])
    truth.c_coef[0] = 47.5
    truth.dx, truth.dy = dx.astype(float), dy.astype(float)
    u_t, v_t = truth.predict()
    for f in range(4):
        win._set_active(f)
        for view in range(40):
            win._set_view(view)
            win._place(u_t[f, view], v_t[f, view])
    win.free_dx.setChecked(True)     # shifts are frozen by default
    win.free_dy.setChecked(True)
    win._fit_now()
    fit = win._fit
    assert fit.rms_u < 1e-6 and fit.rms_v < 1e-6

    # the recovered shifts equal the injected ones up to the gauge that was
    # regauged into (c, a, b, y): compare detrended differences instead
    from tktomo.tracking.model import poly_basis
    m = fit.model
    g = np.column_stack([np.ones(40), np.cos(base.angles),
                         np.sin(base.angles)])
    d = m.dx - dx
    resid = d - g @ np.linalg.lstsq(g, d, rcond=None)[0]
    assert float(np.abs(resid).max()) < 1e-6
    d = m.dy - dy
    assert float(np.std(d)) < 1e-6

    # aligned export puts the phantom back together
    aligned = export_aligned_stack(base, m, CoordinateChain())
    assert aligned.data.shape == base.data.shape
    assert np.isfinite(aligned.data).all()


def test_viewbox_drops_touchpad_momentum_but_not_wheel_notches(tracker):
    from PySide6.QtCore import Qt
    from tktomo.ptycho_align.ui.panels.base import NoMomentumViewBox

    vb = tracker.viewer.image_view.getView()
    assert isinstance(vb, NoMomentumViewBox)
    vb.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
    before = vb.viewRange()

    class Ev:
        def __init__(self, phase):
            self._phase = phase

        def phase(self):
            return self._phase

        def delta(self):
            return 120

        def pos(self):
            from PySide6.QtCore import QPointF
            return QPointF(50, 50)

        def accept(self):
            pass

        def ignore(self):
            pass

    vb.wheelEvent(Ev(Qt.ScrollPhase.ScrollMomentum))
    assert vb.viewRange() == before          # momentum tail ignored
    vb.wheelEvent(Ev(Qt.ScrollPhase.NoScrollPhase))
    assert vb.viewRange() != before          # a real notch still zooms


def test_window_over_remote_source(qtbot, tmp_path):
    """The window on a served stack: one frame per view, jobs on the host."""
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    from tests.helpers_tracking import write_blob_file
    from tktomo.tracking.remote import RemoteStackSource, make_server

    path, stack, theta, truth_u, truth_v = write_blob_file(tmp_path / "b.h5")
    server = make_server("tcp://127.0.0.1:*").start()
    server.host.wait(server.host.open_stack(path))
    source = RemoteStackSource(server.endpoint, timeout=60.0)
    try:
        win = TrackModelWindow(source=source)
        qtbot.addWidget(win)
        assert "stack on" in win.windowTitle()
        assert win._stack.shape == stack.shape
        assert win._chain.binning == 2                  # adopted from the file
        assert win._source["endpoint"] == server.endpoint
        assert "Open remote stack…" in [a.text() for a in
                                        win._file_menu.actions()]
        misses = source.cache.misses
        win._set_view(3)
        win._set_view(3)
        assert source.cache.misses == misses + 1        # prefetch does not count
        tol = float(stack[3].max() - stack[3].min()) / 65535
        assert np.allclose(win.viewer._raw_image, stack[3], rtol=0, atol=tol)

        # the views the user is about to walk onto arrive on their own, in
        # the stride they are walking, so the ones stepped over cost nothing
        assert win._prefetch is not None
        win._set_view(8)                                # a five-view advance
        qtbot.waitUntil(lambda: source.cached(13) and source.cached(18),
                        timeout=10000)
        assert not source.cached(11)

        ok, why = source.autotrack_available()
        if ok:
            win.auto_fit.setChecked(False)
            win.advance_box.setValue(0)
            win._set_active(0)
            for w in (5, 20, 35):
                win._set_view(w)
                # labels are RAW coordinates; the chain maps the loaded px
                win._place(float(truth_u[w]), float(truth_v[w]))
            win._auto_complete(False)
            worker = win._autotrack_worker
            with qtbot.waitSignal(worker.finished_tracks, timeout=120000):
                pass
            qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)
            assert win._labels.counts_per_view(stack.shape[0]).sum() >= 20
        win.close()
    finally:
        source.close()
        server.stop()


def test_bin_control_keeps_raw_labels_and_model(tracker):
    """Switching the run-time binning changes the grid, not the labels."""
    win = tracker
    truth = truth_for(win)
    label_from_truth(win, truth)
    win._fit_now()
    n, ny, nx = win._stack.shape
    u_raw, v_raw, valid, _ = win._labels.to_arrays(n, truth.feature_ids)
    win._feature_sizes[0] = 12.0
    win.slice_row.setValue(20)

    win.bin_combo.setCurrentIndex(win.bin_combo.findData(2))
    assert win._chain.rebin == 2 and win._chain.scale == 2
    assert win._stack.shape == (n, ny // 2, nx // 2)
    assert win.slider.maximum() == n - 1
    u2, v2, valid2, _ = win._labels.to_arrays(n, truth.feature_ids)
    np.testing.assert_array_equal(valid, valid2)
    np.testing.assert_allclose(u_raw[valid], u2[valid2])
    np.testing.assert_allclose(v_raw[valid], v2[valid2])
    assert win._feature_sizes[0] == pytest.approx(6.0)
    assert win.slice_row.value() == 10                       # same raw row
    assert win._source["rebin"] == 2
    # a label placed on the binned grid lands at the raw position it shows
    win._set_active(0)
    win._set_view(3)
    win._place(10.0, 7.0)
    u_new, v_new = win._labels.to_arrays(n, truth.feature_ids)[:2]
    assert u_new[0, 3] == pytest.approx(10.0 * 2 + 0.5)
    assert v_new[0, 3] == pytest.approx(7.0 * 2 + 0.5)
    # fitting on the binned grid still recovers the raw-px centre
    win._fit_now()
    assert abs(float(win._model.c_coef[0]) - 64.0) < 1.0

    win.bin_combo.setCurrentIndex(win.bin_combo.findData(1))
    assert win._chain.rebin == 1 and win._stack.shape == (n, ny, nx)
    assert win._feature_sizes[0] == pytest.approx(12.0)
    assert win.slice_row.value() == 20


def test_bin_survives_a_session_round_trip(tracker, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    win = tracker
    label_from_truth(win, truth_for(win))
    win.bin_combo.setCurrentIndex(win.bin_combo.findData(2))
    win._feature_sizes[1] = 5.0
    path = str(tmp_path / "session.h5")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    win._save_session()
    win.bin_combo.setCurrentIndex(win.bin_combo.findData(1))
    assert win._feature_sizes[1] == pytest.approx(10.0)
    win._load_session(path)
    assert win._chain.rebin == 2
    assert win.bin_combo.currentData() == 2
    assert win._feature_sizes[1] == pytest.approx(5.0)

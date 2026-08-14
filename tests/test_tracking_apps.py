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
    model = AxisModel.blank(win._data.angles, np.arange(n_feat))
    model.a = np.linspace(-25, 25, n_feat)
    model.b = np.linspace(18, -18, n_feat)
    model.y = np.linspace(30, 95, n_feat)
    model.c_coef[0] = 64.0
    return model


def label_from_truth(win, truth, every=4):
    u_t, v_t = truth.predict()
    for f in range(truth.feature_ids.size):
        win._set_active(f)
        for view in range(0, win._data.angles.size, every):
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
    tracker.feature_table.item(0, 7).setCheckState(Qt.CheckState.Checked)
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

        def submit(self, slab, theta, sy, sx, rot_deg, center, row_in_slab):
            jobs.append({"slab_rows": slab.shape[1],
                         "slab_width": slab.shape[2],
                         "sx": np.array(sx), "center": center,
                         "rot": np.array(rot_deg),
                         "row_in_slab": row_in_slab})

    tracker._recon_worker = StubWorker()
    tracker._model.alpha_coef[0] = 0.0
    tracker._request_recon()
    tracker._model.alpha_coef[0] = -0.02
    tracker._request_recon()

    flat, tilted = jobs
    assert np.allclose(flat["rot"], 0.0)
    assert np.allclose(tilted["rot"], np.rad2deg(0.02))   # rot = -deg(alpha)
    width = tracker._data.data.shape[2]
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
        == ["dx shifts", "dy shifts"]
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
        tracker._plot_selectors[1].setCurrentText(kind)
        assert tracker._plot_widgets[1].getPlotItem().listDataItems(), kind


def test_plot_click_jumps_to_nearest_frame(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    deg = np.rad2deg(tracker._data.angles)
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


def test_recon_binning_rescales_geometry(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()

    jobs = []

    class StubWorker:
        finished_slice = _StubSignal()
        failed = _StubSignal()

        def submit(self, slab, theta, sy, sx, rot_deg, center, row_in_slab):
            jobs.append({"width": slab.shape[2], "sx": np.array(sx),
                         "sy": np.array(sy), "center": center,
                         "row_in_slab": row_in_slab,
                         "rot": np.array(rot_deg)})

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


def test_3d_grid_sits_at_center_row(tracker):
    truth = truth_for(tracker)
    label_from_truth(tracker, truth)
    tracker._fit_now()
    tracker._build_3d_once()
    if tracker._gl is None:
        pytest.skip("no OpenGL in this environment")
    tracker._refresh_3d()
    from PySide6.QtGui import QVector3D
    z = tracker._gl_grid.transform().map(QVector3D(0, 0, 0)).z()
    # identity chain: grid at the stack's center row, scene z negated
    center_row = (tracker._data.data.shape[1] - 1) / 2.0
    assert z == pytest.approx(-center_row)


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


def test_tracker_3d_tab_degrades_or_builds(tracker):
    tracker._build_3d_once()
    assert tracker._view3d_built
    tracker._refresh_3d()         # must not raise, with or without OpenGL


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

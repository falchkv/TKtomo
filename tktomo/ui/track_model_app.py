"""Track-model app: fit a tomography model from hand-labeled features.

Load a projection stack (or a feature-crop stack from
`tktomo-feature-isolation`), label features across views, and fit

    u = a cos(t) + b sin(t) + c(t) + dx      (per feature, per view)
    v = y + alpha(t)*s + beta(t)*t' + dy

with the rotation-axis position c, in-plane tilt alpha and out-of-plane
tilt beta as low-order polynomials in angle (constants by default), and
per-view shift groups dx/dy that can be held fixed or left free. Every
parameter can be edited by hand; edits re-evaluate the residuals without
re-fitting, so "what does the residual do if the center were HERE" is one
spinbox away.

Interaction: digits 0-9 pick the active feature, left click (or Space)
places it in the current view and advances N views, Delete removes the
nearest label in the view, arrows step views. Labels are stored in RAW
detector coordinates through the stack's provenance chain, so the same
labels are valid at any binning or crop of the same parent data.

Run with::

    python -m tktomo.ui.track_model_app [stack.h5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tktomo.io import ProjectionData, save_projections
from tktomo.io.phantom import generate_phantom
from tktomo.tracking import sessionio
from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.diagnostics import run_diagnostics
from tktomo.tracking.export import (
    export_aligned_stack,
    write_model_h5,
    write_slogger_shifts,
)
from tktomo.tracking.labels import LabelStore
from tktomo.tracking.model import (
    AxisModel,
    FreeMask,
    residuals,
    solve_model,
)
from tktomo.ui.common import run_app
from tktomo.ui.tracking_widgets import (
    MarkableStackView,
    feature_color,
    load_stack_interactive,
)

# plot kinds selectable in the two residual panes; the first two are the
# defaults, "labels per view" is the direct where-am-I-missing-data view
PLOT_KINDS = [
    "dx shifts",
    "dy shifts",
    "labels per view",
    "residual u",
    "residual v",
    "per-view spread",
    "axis center c",
    "tilts alpha/beta",
    "residual histogram",
]


class ReconWorker(QThread):
    """Runs tomopy gridrec off the GUI thread, one job at a time.

    TomoPy segfaults when reconstructing from two threads at once, and a
    stray call on the GUI thread freezes the app for seconds, so ALL
    reconstruction goes through this single persistent worker. The window
    keeps a dirty flag: a request arriving while the worker is busy
    replaces the pending one and runs when the current job finishes
    (single-flight; intermediate states are never queued up).
    """

    finished_slice = Signal(object)      # 2-D array
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._job = None

    def submit(self, slab, theta, sy, sx, rot_deg, center, row_in_slab):
        self._job = (np.array(slab, np.float32), np.asarray(theta, float),
                     np.asarray(sy, float), np.asarray(sx, float),
                     np.asarray(rot_deg, float), float(center),
                     int(row_in_slab))
        if not self.isRunning():
            self.start()

    def run(self):  # noqa: N802
        while self._job is not None:
            slab, theta, sy, sx, rot_deg, center, row_in_slab = self._job
            self._job = None
            try:
                from tktomo.align.transform import (  # noqa: PLC0415
                    Transform,
                    apply_transform,
                )
                from tktomo.recon import get_backend  # noqa: PLC0415

                # Full per-view 2D correction, same semantics as the
                # aligned-stack export: shift AND derotation by the
                # in-plane tilt alpha(theta), so the slice responds to the
                # tilt parameters. (The rotation is about the slab center;
                # versus the full-image center that is one constant
                # vertical offset, identical for every view.) beta needs
                # 3D geometry and stays out, as everywhere in 2D.
                for k in range(slab.shape[0]):
                    t = Transform(dx=sx[k], dy=sy[k], rotation=rot_deg[k])
                    if not t.is_identity():
                        slab[k] = apply_transform(slab[k], t, order=1)
                # the requested detector row's position inside the slab;
                # NOT the slab middle, which drifts when the slab is
                # clipped at the top or bottom of the stack
                sino = slab[:, row_in_slab:row_in_slab + 1, :]
                volume = get_backend("tomopy").reconstruct(
                    sino, theta, center=center, algorithm="gridrec")
                self.finished_slice.emit(np.asarray(volume)[0])
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                self.failed.emit(str(exc))


class AutoTrackWorker(QThread):
    """Runs seeded track completion off the GUI thread, one batch at a time.

    Same single-flight contract as ReconWorker: `submit` overwrites the
    pending job, `run()` loops until none is left, results leave only via
    signals. The 540 MB stack is NOT copied (read-only numpy access from
    this one thread); the highpassed copy is cached per (stack, sigma) so
    repeated runs skip the ~7 s preprocessing.
    """

    progress = Signal(int, int, object)     # done, total, fid (None = highpass)
    finished_tracks = Signal(object)        # [(fid, TrackResult)]
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._job = None
        self._cancelled = False
        self._hp_cache: dict = {}

    def submit(self, frames, theta, jobs, hp_sigma) -> None:
        self._cancelled = False
        self._job = (frames, np.asarray(theta, float), list(jobs),
                     float(hp_sigma))
        if not self.isRunning():
            self.start()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):  # noqa: N802
        while self._job is not None:
            frames, theta, jobs, hp_sigma = self._job
            self._job = None
            try:
                from tktomo.tracking.autotrack import (  # noqa: PLC0415
                    complete_track,
                    highpass2d,
                )

                key = (id(frames), hp_sigma)
                if key not in self._hp_cache:
                    self._hp_cache.clear()      # at most one stack cached
                    n = len(frames)
                    hp = []
                    for k in range(n):
                        if self._cancelled:
                            break
                        hp.append(highpass2d(frames[k], hp_sigma))
                        if k % 10 == 0:
                            self.progress.emit(k + 1, n, None)
                    else:
                        self._hp_cache[key] = hp
                if self._cancelled or key not in self._hp_cache:
                    self.finished_tracks.emit([])
                    continue
                hp = self._hp_cache[key]
                out = []
                for idx, (fid, seeds, params) in enumerate(jobs):
                    self.progress.emit(idx, len(jobs), fid)
                    result = complete_track(
                        hp, theta, seeds, params,
                        cancelled=lambda: self._cancelled)
                    if result.cancelled:
                        self.finished_tracks.emit([])
                        break
                    out.append((fid, result))
                else:
                    self.finished_tracks.emit(out)
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                self.failed.emit(str(exc))


class TrackModelWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo track model")

        self._data: ProjectionData | None = None
        self._chain = CoordinateChain()
        self._source: dict = {}
        self._labels = LabelStore()
        self._model: AxisModel | None = None
        self._mask: FreeMask | None = None
        self._fit = None
        self._diagnostics: dict | None = None
        self._pins: set[int] = set()
        self._feature_sizes: dict[int, float] = {}   # loaded px, default 10
        self._probe: tuple[float, float, float] | None = None  # (a, b, y) raw
        self._last_recon_info: dict | None = None
        self._active = 0
        self._view = 0
        self._updating_ui = False

        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(300)
        self._fit_timer.timeout.connect(self._fit_now)
        self._recon_timer = QTimer(self)
        self._recon_timer.setSingleShot(True)
        self._recon_timer.setInterval(500)
        self._recon_timer.timeout.connect(self._request_recon)
        self._recon_worker: ReconWorker | None = None
        self._autotrack_worker: AutoTrackWorker | None = None

        self.viewer = MarkableStackView()
        self.viewer.placeRequested.connect(self._place)
        self.viewer.deleteRequested.connect(self._delete_near)
        self.viewer.stepRequested.connect(self._step)
        self.viewer.digitPressed.connect(self._set_active)
        self.viewer.escapePressed.connect(self._clear_probe)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._set_view)
        self.view_box = QSpinBox()
        self.view_box.valueChanged.connect(self._set_view)
        self.angle_label = QLabel("")
        row = QHBoxLayout()
        row.addWidget(QLabel("View:"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.view_box)
        row.addWidget(self.angle_label)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.viewer, 1)
        row_w = QWidget()
        row_w.setLayout(row)
        left_layout.addWidget(row_w)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._build_tabs())
        splitter.addWidget(self._build_controls())
        splitter.setSizes([620, 460, 360])
        self.setCentralWidget(splitter)
        self.resize(1500, 900)

        if path:
            self._load(path)
        else:
            self._show_data(generate_phantom(60, 128, 48, max_shift=3.0),
                            CoordinateChain(), {"kind": "phantom"})

    # ------------------------------------------------------------------ UI

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plots_tab(), "Residuals")
        self._view3d_built = False
        self._view3d_holder = QWidget()
        QVBoxLayout(self._view3d_holder)
        self.tabs.addTab(self._view3d_holder, "3D")
        self.tabs.addTab(self._build_recon_tab(), "Recon slice")
        self.tabs.currentChanged.connect(self._tab_changed)
        return self.tabs

    def _build_plots_tab(self) -> QWidget:
        """Two panes, each with a plot-kind dropdown. Defaults dx and dy.

        Clicking in any angle-axis plot jumps to the nearest frame, which
        turns the plots into navigation: see a gap or a bad point, click
        it, and you are looking at that projection.
        """
        holder = QWidget()
        layout = QVBoxLayout(holder)
        self._plot_selectors: list[QComboBox] = []
        self._plot_widgets: list[pg.PlotWidget] = []
        for default in ("dx shifts", "dy shifts"):
            combo = QComboBox()
            combo.addItems(PLOT_KINDS)
            combo.setCurrentText(default)
            combo.currentTextChanged.connect(
                lambda _t: self._refresh_plots())
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.scene().sigMouseClicked.connect(
                lambda ev, c=combo: self._plot_clicked(ev, c))
            layout.addWidget(combo)
            layout.addWidget(plot, 1)
            self._plot_selectors.append(combo)
            self._plot_widgets.append(plot)
        hint = QLabel("click in a plot to jump to the nearest frame; "
                      "dots = labeled views (orange = only one label), "
                      "red base ticks = views with NO labels")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return holder

    def _plot_clicked(self, event, combo) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        if combo.currentText() == "residual histogram":
            return                      # x is residual px, not an angle
        index = self._plot_selectors.index(combo)
        plot = self._plot_widgets[index]
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self._jump_to_angle(float(vb.mapSceneToView(event.scenePos()).x()))
        event.accept()

    def _jump_to_angle(self, x_deg: float) -> None:
        if self._data is None:
            return
        deg = np.rad2deg(self._data.angles)
        self._set_view(int(np.argmin(np.abs(deg - x_deg))))

    def _build_recon_tab(self) -> QWidget:
        from tktomo.ptycho_align.ui.panels.base import (  # noqa: PLC0415
            StackDisplay,
        )
        holder = QWidget()
        layout = QVBoxLayout(holder)
        controls = QHBoxLayout()
        self.live_recon = QCheckBox("Live (recompute on change)")
        self.live_recon.setChecked(False)
        self.slice_row = QSpinBox()
        self.slice_row.valueChanged.connect(self._recon_maybe)
        self.recon_bin = QComboBox()
        self.recon_bin.addItems(["1", "2", "4", "8"])
        self.recon_bin.setToolTip(
            "Extra binning applied to the slab before reconstruction. "
            "Cost falls roughly as bin^3 (fewer pixels AND fewer rows per "
            "pixel), so 2 is much more than twice as fast; use it for "
            "live evaluation and go back to 1 for the final look.")
        self.recon_bin.currentTextChanged.connect(self._recon_maybe)
        recon_btn = QPushButton("Reconstruct now")
        recon_btn.clicked.connect(self._request_recon)
        controls.addWidget(self.live_recon)
        controls.addWidget(QLabel("detector row:"))
        controls.addWidget(self.slice_row)
        controls.addWidget(QLabel("bin:"))
        controls.addWidget(self.recon_bin)
        controls.addWidget(recon_btn)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.recon_status = QLabel("")
        layout.addWidget(self.recon_status)
        self.recon_display = StackDisplay()
        self.recon_display.auto_levels_box.setChecked(True)
        self.recon_display.image_view.scene.sigMouseClicked.connect(
            self._slice_clicked)
        self.recon_display.setToolTip(
            "Click a point in the slice to mark it as a magenta diamond "
            "in the projection view, following the point across frames. "
            "Escape (over the projection) clears it.")
        layout.addWidget(self.recon_display, 1)
        return holder

    def _slice_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        if self._last_recon_info is None or self._data is None:
            return
        view_box = self.recon_display.image_view.getView()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        self._probe = self._slice_to_probe(float(point.x()),
                                           float(point.y()))
        self._refresh_view()
        event.accept()

    def _slice_to_probe(self, col: float, row: float
                        ) -> tuple[float, float, float]:
        """Slice pixel -> object point (a, b, y) in raw px.

        tomopy's gridrec re-centers the grid on the rotation axis whatever
        `center` says; the axis lands at (row, col) = (N//2 - 1, (N+1)//2)
        with u = center + A*cos + B*sin mapping to col = axis + A,
        row = axis - B. Pinned empirically against tomopy in
        test_slice_probe_convention, because this is exactly the kind of
        convention that silently changes between library versions.
        """
        info = self._last_recon_info
        n = int(info["width"])
        axis_row = n // 2 - 1
        axis_col = (n + 1) // 2
        scale = info["extra_bin"] * self._chain.binning
        a_raw = (col - axis_col) * scale
        b_raw = -(row - axis_row) * scale
        y_raw = float(self._chain.to_parent(0.0, info["row_loaded"])[1])
        return (float(a_raw), float(b_raw), y_raw)

    def _clear_probe(self) -> None:
        if self._probe is not None:
            self._probe = None
            self._refresh_view()

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        top = QHBoxLayout()
        load_btn = QPushButton("Load stack…")
        load_btn.clicked.connect(lambda: self._load(None))
        top.addWidget(load_btn)
        session_save = QPushButton("Save session…")
        session_save.clicked.connect(self._save_session)
        session_load = QPushButton("Load session…")
        session_load.clicked.connect(self._load_session)
        top.addWidget(session_save)
        top.addWidget(session_load)
        layout.addLayout(top)

        feat_box = QGroupBox("Features (digit keys switch, click places)")
        feat_layout = QVBoxLayout(feat_box)
        self.feature_table = QTableWidget(0, 9)
        self.feature_table.setHorizontalHeaderLabels(
            ["id", "n", "rms u", "rms v", "a", "b", "y", "size", "pin"])
        self.feature_table.horizontalHeaderItem(7).setToolTip(
            "Marker diameter in image pixels. Match it to the feature: "
            "the fit weights each feature's labels by 1/size, because a "
            "click on a large diffuse feature localizes it less than one "
            "on a small sharp feature.")
        self.feature_table.verticalHeader().hide()
        self.feature_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.feature_table.itemSelectionChanged.connect(self._feature_selected)
        self.feature_table.cellChanged.connect(self._feature_cell_edited)
        feat_layout.addWidget(self.feature_table)
        feat_row = QHBoxLayout()
        new_btn = QPushButton("New feature")
        new_btn.clicked.connect(self._new_feature)
        drop_btn = QPushButton("Delete feature")
        drop_btn.clicked.connect(self._drop_feature)
        feat_row.addWidget(new_btn)
        feat_row.addWidget(drop_btn)
        feat_row.addWidget(QLabel("advance"))
        self.advance_box = QSpinBox()
        self.advance_box.setRange(0, 100)
        self.advance_box.setValue(5)
        feat_row.addWidget(self.advance_box)
        feat_row.addWidget(QLabel("views/click"))
        feat_layout.addLayout(feat_row)
        self.active_label = QLabel("active feature: 0")
        feat_layout.addWidget(self.active_label)
        self.ghost_box = QCheckBox("Show active feature's labels from "
                                   "other frames (small circles)")
        self.ghost_box.toggled.connect(lambda _c: self._refresh_view())
        feat_layout.addWidget(self.ghost_box)
        layout.addWidget(feat_box)

        auto_box = QGroupBox("Auto-track (manual labels are the anchors)")
        auto_layout = QVBoxLayout(auto_box)
        run_row = QHBoxLayout()
        self.auto_feature_btn = QPushButton("Auto-complete feature")
        self.auto_feature_btn.clicked.connect(
            lambda: self._auto_complete(False))
        self.auto_all_btn = QPushButton("Auto-complete all")
        self.auto_all_btn.clicked.connect(lambda: self._auto_complete(True))
        self.auto_all_btn.setToolTip(
            "Every feature with at least 2 manual labels. Templates are "
            "cut at your clicks and matched outward until the correlation "
            "drops; tracks STOP honestly instead of wandering.")
        run_row.addWidget(self.auto_feature_btn)
        run_row.addWidget(self.auto_all_btn)
        auto_layout.addLayout(run_row)
        param_row = QHBoxLayout()
        self.auto_min_corr = QDoubleSpinBox()
        self.auto_min_corr.setRange(0.05, 0.95)
        self.auto_min_corr.setSingleStep(0.05)
        self.auto_min_corr.setValue(0.30)
        self.auto_min_corr.setToolTip(
            "Matches below this correlation end the track. The template "
            "genuinely stops resembling the image ~10-20 views from a "
            "seed (measured 0.65 at 1 view, 0.23 at 10), so more manual "
            "labels = longer reach.")
        self.auto_radius = QDoubleSpinBox()
        self.auto_radius.setRange(2.0, 50.0)
        self.auto_radius.setValue(8.0)
        self.auto_radius.setSuffix(" px")
        self.auto_radius.setToolTip(
            "Search radius around the predicted position (grows slowly "
            "with distance from the seed).")
        self.auto_fb = QCheckBox("fwd-back")
        self.auto_fb.setChecked(True)
        self.auto_fb.setToolTip(
            "Track every accepted match BACK to its seed and reject it "
            "if the round trip misses or correlates poorly. Cheap, only "
            "ever removes labels.")
        param_row.addWidget(QLabel("min corr"))
        param_row.addWidget(self.auto_min_corr)
        param_row.addWidget(QLabel("search"))
        param_row.addWidget(self.auto_radius)
        param_row.addWidget(self.auto_fb)
        auto_layout.addLayout(param_row)
        clear_row = QHBoxLayout()
        clear_feat = QPushButton("Clear auto: feature")
        clear_feat.clicked.connect(lambda: self._clear_auto(False))
        clear_all = QPushButton("Clear auto: all")
        clear_all.clicked.connect(lambda: self._clear_auto(True))
        self.auto_cancel_btn = QPushButton("Cancel")
        self.auto_cancel_btn.setEnabled(False)
        self.auto_cancel_btn.clicked.connect(self._cancel_autotrack)
        clear_row.addWidget(clear_feat)
        clear_row.addWidget(clear_all)
        clear_row.addWidget(self.auto_cancel_btn)
        auto_layout.addLayout(clear_row)
        self.auto_status = QLabel("")
        self.auto_status.setWordWrap(True)
        auto_layout.addWidget(self.auto_status)
        layout.addWidget(auto_box)

        model_box = QGroupBox("Model (check = fixed at the shown value)")
        self._model_form = QFormLayout(model_box)
        deg_row = QHBoxLayout()
        self.deg_c = QSpinBox()
        self.deg_a = QSpinBox()
        self.deg_b = QSpinBox()
        for label, box in (("c", self.deg_c), ("α", self.deg_a),
                           ("β", self.deg_b)):
            box.setRange(0, 4)
            box.valueChanged.connect(self._degrees_changed)
            deg_row.addWidget(QLabel(f"deg {label}"))
            deg_row.addWidget(box)
        deg_holder = QWidget()
        deg_holder.setLayout(deg_row)
        self._model_form.addRow("Polynomial degrees:", deg_holder)
        self._coef_rows: dict[str, list[tuple[QCheckBox, QDoubleSpinBox]]] = {
            "c": [], "alpha": [], "beta": []}
        self._rebuild_coef_rows()
        self.free_dx = QCheckBox("dx free")
        self.free_dx.setChecked(True)
        self.free_dy = QCheckBox("dy free")
        self.free_dy.setChecked(True)
        self.free_dx.setToolTip(
            "dx: one HORIZONTAL displacement of the whole projection per "
            "view, the per-view alignment error the model corrects.\n"
            "Checked: the fit adjusts dx for every labeled view "
            "(unlabeled views are interpolated).\n"
            "Unchecked: dx is frozen at its current values (whatever the "
            "last fit or Zero dx left), and the axis polynomial plus "
            "feature positions must explain the labels alone.\n"
            "Beware: a free dx exactly absorbs any view that carries only "
            "one label, so label several features in the SAME views or "
            "the track geometry is unconstrained (warning W6).")
        self.free_dy.setToolTip(
            "dy: one VERTICAL displacement of the whole projection per "
            "view.\nChecked: fitted per labeled view (interpolated where "
            "unlabeled).\nUnchecked: frozen at the current values; height "
            "y and tilts must explain the labels alone.\n"
            "Single-label views are absorbed exactly by a free dy, same "
            "caveat as dx.")
        zero_dx = QPushButton("Zero dx")
        zero_dx.clicked.connect(lambda: self._zero_shift("dx"))
        zero_dy = QPushButton("Zero dy")
        zero_dy.clicked.connect(lambda: self._zero_shift("dy"))
        zero_dx.setToolTip(
            "Set every per-view horizontal shift to zero, i.e. assert the "
            "projections are already horizontally aligned. Residuals "
            "update immediately; combine with an UNCHECKED 'dx free' to "
            "keep it that way through the next fit, otherwise the fit "
            "writes new values.")
        zero_dy.setToolTip(
            "Set every per-view vertical shift to zero (projections "
            "assumed vertically aligned). Combine with an unchecked "
            "'dy free' to keep zeros through the next fit.")
        shift_row = QHBoxLayout()
        for w in (self.free_dx, zero_dx, self.free_dy, zero_dy):
            shift_row.addWidget(w)
            if isinstance(w, QCheckBox):
                w.toggled.connect(lambda _c: self._request_fit())
        shift_holder = QWidget()
        shift_holder.setLayout(shift_row)
        self._model_form.addRow("Per-view shifts:", shift_holder)
        # Frozen shifts are invisible state that changes what a fit means;
        # this line keeps them visible so "why does my fit depend on
        # earlier clicking" has an on-screen answer.
        self.shift_state_label = QLabel("dx: zero   dy: zero")
        self.shift_state_label.setToolTip(
            "Current content of the per-view shifts. 'free' is refitted "
            "from the labels every fit (history cannot matter); 'FROZEN' "
            "is held at the shown rms, which is whatever the last free "
            "fit left, so the fit DOES depend on that history until you "
            "press Zero.")
        self._model_form.addRow("", self.shift_state_label)
        self.huber = QDoubleSpinBox()
        self.huber.setRange(0.5, 20.0)
        self.huber.setValue(3.0)
        self.iters = QSpinBox()
        self.iters.setRange(1, 10)
        self.iters.setValue(4)
        robust_row = QHBoxLayout()
        robust_row.addWidget(QLabel("huber k"))
        robust_row.addWidget(self.huber)
        robust_row.addWidget(QLabel("IRLS iters"))
        robust_row.addWidget(self.iters)
        robust_holder = QWidget()
        robust_holder.setLayout(robust_row)
        self._model_form.addRow("Robustness:", robust_holder)
        layout.addWidget(model_box)

        fit_row = QHBoxLayout()
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._fit_now)
        self.auto_fit = QCheckBox("Auto-fit")
        self.auto_fit.setChecked(True)
        diag_btn = QPushButton("Run diagnostics")
        diag_btn.clicked.connect(self._run_diagnostics)
        outlier_btn = QPushButton("Worst outlier")
        outlier_btn.setToolTip(
            "Jump to the label with the largest residual and make its "
            "feature active, ready to inspect or fix.")
        outlier_btn.clicked.connect(self._goto_worst_outlier)
        fit_row.addWidget(fit_btn)
        fit_row.addWidget(self.auto_fit)
        fit_row.addWidget(diag_btn)
        fit_row.addWidget(outlier_btn)
        layout.addLayout(fit_row)

        self.summary_label = QLabel("no fit yet")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.warnings_box = QPlainTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(140)
        layout.addWidget(self.warnings_box)

        export_box = QGroupBox("Export")
        export_layout = QVBoxLayout(export_box)
        for text, slot in (
                ("Model + astra vectors (.h5)…", self._export_model),
                ("slogger shifts.h5…", self._export_shifts),
                ("Aligned projection stack…", self._export_aligned)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            export_layout.addWidget(btn)
        layout.addWidget(export_box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(340)
        return scroll

    # --------------------------------------------------------- model sync

    def _rebuild_coef_rows(self) -> None:
        """One (fixed-checkbox, value-spinbox) row per polynomial coefficient."""
        for group, rows in self._coef_rows.items():
            for check, spin in rows:
                self._model_form.removeRow(check.parentWidget())
            rows.clear()
        degrees = (self.deg_c.value(), self.deg_a.value(), self.deg_b.value())
        for group, symbol, degree, decimals, step in (
                ("c", "c", degrees[0], 2, 1.0),
                ("alpha", "α", degrees[1], 5, 0.001),
                ("beta", "β", degrees[2], 5, 0.001)):
            for k in range(degree + 1):
                check = QCheckBox("fix")
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setRange(-1e6, 1e6)
                spin.setSingleStep(step)
                spin.valueChanged.connect(self._coef_edited)
                check.toggled.connect(lambda _c: self._request_fit())
                row = QHBoxLayout()
                row.addWidget(check)
                row.addWidget(spin, 1)
                holder = QWidget()
                holder.setLayout(row)
                self._model_form.addRow(f"{symbol}[{k}]", holder)
                self._coef_rows[group].append((check, spin))

    def _degrees_changed(self) -> None:
        self._rebuild_coef_rows()
        if self._model is not None:
            self._model = self._model.with_degrees(
                self.deg_c.value(), self.deg_a.value(), self.deg_b.value())
            self._push_model_to_ui()
        self._request_fit()

    def _sync_model(self) -> None:
        """Make model/mask rows match the label store's feature ids."""
        if self._data is None:
            return
        ids = self._labels.feature_ids()
        if self._active not in ids:
            ids = sorted(set(ids) | {self._active})
        theta = self._data.angles
        degrees = (self.deg_c.value(), self.deg_a.value(), self.deg_b.value())
        old = self._model
        model = AxisModel.blank(theta, np.asarray(ids, int), degrees)
        if old is not None and old.theta.size == theta.size:
            model.c_coef[:] = np.resize(old.c_coef, model.c_coef.shape)
            model.alpha_coef[:] = np.resize(old.alpha_coef,
                                            model.alpha_coef.shape)
            model.beta_coef[:] = np.resize(old.beta_coef,
                                           model.beta_coef.shape)
            model.dx, model.dy = old.dx.copy(), old.dy.copy()
            lookup = {int(f): k for k, f in enumerate(old.feature_ids)}
            for row, fid in enumerate(ids):
                if fid in lookup:
                    src = lookup[fid]
                    model.a[row] = old.a[src]
                    model.b[row] = old.b[src]
                    model.y[row] = old.y[src]
        self._model = model
        mask = FreeMask.all_free(model)
        mask.dx = self.free_dx.isChecked()
        mask.dy = self.free_dy.isChecked()
        for group in ("c", "alpha", "beta"):
            flags = getattr(mask, group)
            for k, (check, _spin) in enumerate(self._coef_rows[group]):
                if k < flags.size:
                    flags[k] = not check.isChecked()
        mask.features = np.array([fid not in self._pins for fid in ids],
                                 bool)
        self._mask = mask

    def _push_model_to_ui(self) -> None:
        """Show the model's current values in the spinboxes, silently."""
        if self._model is None:
            return
        self._updating_ui = True
        try:
            for group, coef in (("c", self._model.c_coef),
                                ("alpha", self._model.alpha_coef),
                                ("beta", self._model.beta_coef)):
                for k, (_check, spin) in enumerate(self._coef_rows[group]):
                    if k < coef.size:
                        spin.setValue(float(coef[k]))
        finally:
            self._updating_ui = False

    def _coef_edited(self) -> None:
        """A hand-edited value: write into the model, re-evaluate, no solve."""
        if self._updating_ui or self._model is None:
            return
        for group, attr in (("c", "c_coef"), ("alpha", "alpha_coef"),
                            ("beta", "beta_coef")):
            coef = getattr(self._model, attr)
            for k, (_check, spin) in enumerate(self._coef_rows[group]):
                if k < coef.size:
                    coef[k] = spin.value()
        self._evaluate()

    def _zero_shift(self, which: str) -> None:
        if self._model is None:
            return
        getattr(self._model, which)[:] = 0.0
        self._evaluate()

    # ------------------------------------------------------------- labels

    def _place(self, u: float, v: float) -> None:
        if self._data is None:
            return
        u_raw, v_raw = self._chain.to_parent(u, v, view=self._view)
        self._labels.set(self._active, self._view, float(u_raw),
                         float(v_raw))
        advance = self.advance_box.value()
        if advance:
            self._set_view(self._view + advance)
        else:
            self._refresh_view()
        self._refresh_plots()      # label coverage updates even without a fit
        self._request_fit()

    def _delete_near(self, u: float, v: float) -> None:
        if self._data is None:
            return
        u_raw, v_raw = self._chain.to_parent(u, v, view=self._view)
        near = self._labels.nearest(self._view, float(u_raw), float(v_raw))
        if near is not None:
            self._labels.remove(near[0], self._view)
            self._refresh_view()
            self._refresh_plots()
            self._request_fit()

    def _set_active(self, fid: int) -> None:
        self._active = int(fid)
        self.active_label.setText(f"active feature: {self._active}")
        self._refresh_view()

    def _new_feature(self) -> None:
        ids = self._labels.feature_ids()
        self._set_active(max(ids) + 1 if ids else 0)

    def _drop_feature(self) -> None:
        n = self._labels.clear_feature(self._active)
        self._pins.discard(self._active)
        if n:
            self._request_fit()
        self._refresh_view()

    # ------------------------------------------------------------ fitting

    def _feature_weights(self) -> np.ndarray | None:
        """1/size per feature: a click's localization scales with the
        feature's extent, so big markers count for less. Only relative
        values matter to the solver."""
        if self._model is None:
            return None
        sizes = np.array([self._feature_sizes.get(int(f), 10.0)
                          for f in self._model.feature_ids], float)
        if np.allclose(sizes, sizes[0] if sizes.size else 1.0):
            return None                    # uniform sizes = unweighted
        return 1.0 / np.clip(sizes, 1e-3, None)

    def _goto_worst_outlier(self) -> None:
        if self._fit is None or self._fit.residual_u.size == 0:
            return
        i, j = self._fit.obs
        mag = np.hypot(self._fit.residual_u, self._fit.residual_v)
        k = int(np.argmax(mag))
        fid = int(self._fit.model.feature_ids[i[k]])
        self._set_active(fid)
        self._set_view(int(j[k]))
        self.statusBar().showMessage(
            f"worst outlier: feature {fid} in view {int(j[k])}, "
            f"residual {mag[k]:.2f} px "
            f"(u {self._fit.residual_u[k]:+.2f}, "
            f"v {self._fit.residual_v[k]:+.2f})", 8000)

    # ---------------------------------------------------------- auto-track

    def _auto_complete(self, all_features: bool) -> None:
        from tktomo.tracking.autotrack import (  # noqa: PLC0415
            AutoTrackParams,
            patch_size,
        )

        if self._data is None:
            return
        if self._chain.view_origin is not None:
            self.statusBar().showMessage(
                "auto-complete does not support per-view-cropped stacks "
                "(the crop already follows the feature)", 6000)
            return
        manual = self._labels.manual_counts()
        fids = (sorted(f for f, n in manual.items() if n >= 2)
                if all_features else
                [self._active] if manual.get(self._active, 0) >= 2 else [])
        if not fids:
            self.statusBar().showMessage(
                "auto-complete needs at least 2 MANUAL labels on the "
                "feature (they are the template anchors)", 6000)
            return

        jobs = []
        for fid in fids:
            seeds = []
            for w in self._labels.manual_views_of(fid):
                u_raw, v_raw = self._labels.get(fid, w)
                u, v = self._chain.from_parent(u_raw, v_raw)
                seeds.append((w, float(u), float(v)))
            params = AutoTrackParams(
                patch=patch_size(self._feature_sizes.get(fid, 10.0)),
                search_radius=float(self.auto_radius.value()),
                min_corr=float(self.auto_min_corr.value()),
                fb_check=self.auto_fb.isChecked())
            jobs.append((fid, seeds, params))

        if self._autotrack_worker is None:
            self._autotrack_worker = AutoTrackWorker(self)
            self._autotrack_worker.finished_tracks.connect(
                self._autotrack_finished)
            self._autotrack_worker.progress.connect(self._autotrack_progress)
            self._autotrack_worker.failed.connect(
                lambda msg: (self._autotrack_done_ui(),
                             self.auto_status.setText(f"failed: {msg}")))
        for btn in (self.auto_feature_btn, self.auto_all_btn):
            btn.setEnabled(False)
        self.auto_cancel_btn.setEnabled(True)
        self.auto_status.setText("preparing…")
        self._autotrack_worker.submit(self._data.data, self._data.angles,
                                      jobs, hp_sigma=12.0)

    def _autotrack_progress(self, done: int, total: int, fid) -> None:
        if fid is None:
            self.auto_status.setText(f"high-pass filtering {done}/{total}…")
        else:
            self.auto_status.setText(
                f"tracking feature {fid} ({done + 1}/{total})…")

    def _autotrack_done_ui(self) -> None:
        for btn in (self.auto_feature_btn, self.auto_all_btn):
            btn.setEnabled(True)
        self.auto_cancel_btn.setEnabled(False)

    def _autotrack_finished(self, results) -> None:
        self._autotrack_done_ui()
        if not results:
            self.auto_status.setText("cancelled, nothing applied")
            return
        lines = []
        for fid, res in results:
            self._labels.clear_auto(fid)
            added = 0
            for al in res.labels:
                u_raw, v_raw = self._chain.to_parent(al.u, al.v)
                if self._labels.set_auto(fid, al.view, float(u_raw),
                                         float(v_raw), al.quality):
                    added += 1
            qualities = [al.quality for al in res.labels]
            line = (f"feature {fid}: +{added} auto labels"
                    + (f", median corr "
                       f"{float(np.median(qualities)):.2f}"
                       if qualities else ""))
            refusals = [w for w in res.warnings if "refused" in w
                        or "not trackable" in w]
            if refusals:
                line += f" ({len(refusals)} seed(s) refused)"
            lines.append(line)
            for w in res.warnings:
                lines.append(f"  {w}")
        self.auto_status.setText("\n".join(lines))
        self._refresh_view()
        self._refresh_plots()
        self._request_fit()

    def _cancel_autotrack(self) -> None:
        if self._autotrack_worker is not None:
            self._autotrack_worker.cancel()

    def _clear_auto(self, all_features: bool) -> None:
        n = self._labels.clear_auto(None if all_features else self._active)
        self.auto_status.setText(
            f"removed {n} auto label(s)"
            + ("" if all_features else f" of feature {self._active}"))
        if n:
            self._refresh_view()
            self._refresh_plots()
            self._request_fit()

    # ------------------------------------------------------------- fitting

    def _request_fit(self) -> None:
        if self.auto_fit.isChecked():
            self._fit_timer.start()

    def _fit_now(self) -> None:
        if self._data is None or len(self._labels) == 0:
            return
        self._sync_model()
        u, v, valid, ids = self._labels.to_arrays(
            self._data.angles.size, self._model.feature_ids)
        if not valid.any():
            return
        try:
            self._fit = solve_model(u, v, valid, self._model, self._mask,
                                    iters=self.iters.value(),
                                    huber=self.huber.value(),
                                    feature_weight=self._feature_weights())
        except ValueError as exc:
            self.summary_label.setText(f"fit failed: {exc}")
            return
        self._model = self._fit.model
        self._diagnostics = None
        self._push_model_to_ui()
        self._after_evaluate()

    def _evaluate(self) -> None:
        """Residuals against the CURRENT model values, no solving."""
        if self._data is None or self._model is None:
            return
        if len(self._labels) == 0:
            return
        self._sync_model_dims_only()
        u, v, valid, _ids = self._labels.to_arrays(
            self._data.angles.size, self._model.feature_ids)
        if not valid.any():
            return
        self._fit = residuals(u, v, valid, self._model)
        self._after_evaluate()

    def _sync_model_dims_only(self) -> None:
        ids_now = self._labels.feature_ids()
        if self._active not in ids_now:
            ids_now = sorted(set(ids_now) | {self._active})
        if (self._model is None
                or list(self._model.feature_ids) != list(ids_now)):
            self._sync_model()

    def _after_evaluate(self) -> None:
        fit = self._fit
        self._update_shift_state_label()
        self._refresh_feature_table()
        self._refresh_plots()
        self._refresh_view()
        self._refresh_3d()
        self.summary_label.setText(
            f"rms u {fit.rms_u:.3f} px, v {fit.rms_v:.3f} px "
            f"({fit.residual_u.size} labels, "
            f"{int(fit.observed_views.sum())} views) | "
            f"center c(θ̄) = {fit.model.center_at_mean_theta():.2f} raw px | "
            f"α₀ = {fit.model.alpha_coef[0]:+.5f} rad")
        warn_lines = list(fit.warnings)
        if self._diagnostics is not None:
            warn_lines += [w for w in self._diagnostics["warnings"]
                           if w not in warn_lines]
        self.warnings_box.setPlainText("\n".join(warn_lines)
                                       if warn_lines else "no warnings")
        if self.live_recon.isChecked():
            self._recon_timer.start()

    def _update_shift_state_label(self) -> None:
        if self._model is None:
            return
        parts = []
        for name, values, free in (
                ("dx", self._model.dx, self.free_dx.isChecked()),
                ("dy", self._model.dy, self.free_dy.isChecked())):
            rms = float(np.sqrt(np.mean(values ** 2)))
            if free:
                state = f"free, fitted rms {rms:.2f} px"
            elif rms == 0.0:
                state = "fixed at zero"
            else:
                state = f"FROZEN at rms {rms:.2f} px"
            parts.append(f"{name}: {state}")
        self.shift_state_label.setText("   ".join(parts))

    def _run_diagnostics(self) -> None:
        if self._fit is None or self._data is None:
            return
        self._sync_model()
        u, v, valid, _ids = self._labels.to_arrays(
            self._data.angles.size, self._model.feature_ids)
        self._diagnostics = run_diagnostics(u, v, valid, self._model,
                                            self._mask, self._fit,
                                            self._feature_weights())
        d = self._diagnostics
        chain = self._chain
        center_grid = chain.parent_to_grid(d["center_estimate_raw_px"],
                                           chain.binning)
        QMessageBox.information(
            self, "Diagnostics",
            f"center c(θ̄): {d['center_estimate_raw_px']:.2f} raw px "
            f"= {float(center_grid):.2f} px on the loaded grid\n"
            f"center half-split: {d['center_split_px']:.2f} raw px "
            f"({'reliable' if d['center_reliable'] else 'NOT RELIABLE'})\n"
            f"alpha half-split: ±{d['holdout']['alpha_split']:.2e} rad, "
            f"beta half-split: ±{d['holdout']['beta_split']:.2e} rad\n"
            f"tilt significance: {d['tilt_significance_sigma']:.1f} sigma "
            f"(do not trust alone; the half-split wins)\n"
            f"held-out rms: u {d['holdout']['rms_u']:.3f}, "
            f"v {d['holdout']['rms_v']:.3f} px\n"
            f"shift split rms: dx {d['shift_split']['dx_rms']:.3f}, "
            f"dy {d['shift_split']['dy_rms']:.3f} px "
            f"(detrended {d['shift_split']['dx_detrended_rms']:.3f} / "
            f"{d['shift_split']['dy_detrended_rms']:.3f})")
        self._after_evaluate()

    # ------------------------------------------------------------ display

    def _step(self, delta: int) -> None:
        self._set_view(self._view + delta)

    def _set_view(self, view: int) -> None:
        if self._data is None:
            return
        view = int(np.clip(view, 0, self._data.data.shape[0] - 1))
        self._view = view
        for widget in (self.slider, self.view_box):
            widget.blockSignals(True)
            widget.setValue(view)
            widget.blockSignals(False)
        self._refresh_view()

    def _refresh_view(self) -> None:
        if self._data is None:
            return
        view = self._view
        self.angle_label.setText(
            f"{np.rad2deg(self._data.angles[view]):7.2f}°")
        self.viewer.set_image(self._data.data[view])

        sizes = {fid: self._feature_sizes.get(fid, 10.0)
                 for fid in self._labels.feature_ids()}
        marks = []
        for fid, u_raw, v_raw, kind, _q in self._labels.in_view_full(view):
            u, v = self._chain.from_parent(u_raw, v_raw, view=view)
            marks.append((fid, float(u), float(v), kind))
        self.viewer.show_labels(marks, active_id=self._active, sizes=sizes)

        ghosts = []
        if self.ghost_box.isChecked():
            for w in self._labels.views_of(self._active):
                if w == view:
                    continue
                u_raw, v_raw = self._labels.get(self._active, w)
                u, v = self._chain.from_parent(u_raw, v_raw, view=view)
                ghosts.append((self._active, float(u), float(v)))
        self.viewer.show_ghosts(ghosts, sizes=sizes)

        preds = []
        if self._fit is not None and self._model is not None \
                and self._model.theta.size == self._data.angles.size:
            u_pred, v_pred = self._model.predict()
            for row, fid in enumerate(self._model.feature_ids):
                if self._labels.counts().get(int(fid), 0) == 0:
                    continue
                u, v = self._chain.from_parent(u_pred[row, view],
                                               v_pred[row, view], view=view)
                preds.append((int(fid), float(u), float(v)))
            row = list(self._model.feature_ids).index(self._active) \
                if self._active in self._model.feature_ids else None
            if row is not None and self._chain.view_origin is None:
                # The trajectory overlay is the model track WITHOUT any
                # per-view shift: the feature's path in the ideal aligned
                # frame. The per-view dx/dy are alignment errors of
                # individual views; including them made the curve zigzag,
                # and anchoring with the current view's shift made the
                # whole curve jump with that one view's noise. Drawn this
                # way the curve is stable, and the gap between a marker
                # and the curve IS that view's misalignment.
                m = self._model
                ct, sn = np.cos(m.theta), np.sin(m.theta)
                s_row = m.a[row] * ct + m.b[row] * sn
                t_row = -m.a[row] * sn + m.b[row] * ct
                c_of, alpha_of, beta_of = m.axis_curves()
                u_tr = s_row + c_of
                v_tr = m.y[row] + alpha_of * s_row + beta_of * t_row
                u_all, v_all = self._chain.from_parent(u_tr, v_tr)
                self.viewer.show_trajectory(u_all, v_all)
            else:
                self.viewer.show_trajectory(None)
        self.viewer.show_predictions(preds, active_id=self._active)

        # the slice probe: an object point picked in the recon slice,
        # projected into THIS view through the model
        if self._probe is not None and self._model is not None:
            a, b, y = self._probe
            theta_v = float(self._data.angles[view])
            ct, sn = np.cos(theta_v), np.sin(theta_v)
            s = a * ct + b * sn
            t = -a * sn + b * ct
            c_of, alpha_of, beta_of = self._model.axis_curves()
            u_raw = s + c_of[view] + self._model.dx[view]
            v_raw = (y + alpha_of[view] * s + beta_of[view] * t
                     + self._model.dy[view])
            u, v = self._chain.from_parent(u_raw, v_raw, view=view)
            self.viewer.show_probe((float(u), float(v)))
        else:
            self.viewer.show_probe(None)

    def _refresh_feature_table(self) -> None:
        if self._model is None or self._fit is None:
            return
        table = self.feature_table
        rms_u, rms_v = self._fit.feature_rms()
        counts = self._labels.counts()
        ids = self._model.feature_ids
        self._updating_ui = True
        try:
            table.setRowCount(ids.size)
            for row, fid in enumerate(ids):
                fid = int(fid)
                id_item = QTableWidgetItem(str(fid))
                id_item.setFlags(Qt.ItemFlag.ItemIsEnabled
                                 | Qt.ItemFlag.ItemIsSelectable)
                id_item.setBackground(pg.mkBrush(*feature_color(fid), 120))
                table.setItem(row, 0, id_item)
                values = (counts.get(fid, 0),
                          _fmt(rms_u[row]), _fmt(rms_v[row]),
                          f"{self._model.a[row]:.1f}",
                          f"{self._model.b[row]:.1f}",
                          f"{self._model.y[row]:.1f}",
                          f"{self._feature_sizes.get(fid, 10.0):.1f}")
                for col, value in enumerate(values, start=1):
                    item = QTableWidgetItem(str(value))
                    if col in (4, 5, 6, 7):
                        item.setFlags(item.flags()
                                      | Qt.ItemFlag.ItemIsEditable)
                    else:
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled
                                      | Qt.ItemFlag.ItemIsSelectable)
                    table.setItem(row, col, item)
                pin = QTableWidgetItem()
                pin.setFlags(Qt.ItemFlag.ItemIsEnabled
                             | Qt.ItemFlag.ItemIsUserCheckable
                             | Qt.ItemFlag.ItemIsSelectable)
                pin.setCheckState(Qt.CheckState.Checked if fid in self._pins
                                  else Qt.CheckState.Unchecked)
                table.setItem(row, 8, pin)
        finally:
            self._updating_ui = False

    def _feature_selected(self) -> None:
        rows = {i.row() for i in self.feature_table.selectedIndexes()}
        if len(rows) == 1 and self._model is not None:
            row = next(iter(rows))
            if row < self._model.feature_ids.size:
                self._set_active(int(self._model.feature_ids[row]))

    def _feature_cell_edited(self, row: int, col: int) -> None:
        if self._updating_ui or self._model is None:
            return
        if row >= self._model.feature_ids.size:
            return
        fid = int(self._model.feature_ids[row])
        if col == 8:
            item = self.feature_table.item(row, 8)
            pinned = item.checkState() == Qt.CheckState.Checked
            (self._pins.add if pinned else self._pins.discard)(fid)
            self._request_fit()
            return
        if col == 7:
            item = self.feature_table.item(row, 7)
            try:
                size = float(item.text())
            except (TypeError, ValueError):
                return
            self._feature_sizes[fid] = max(size, 0.5)
            self._refresh_view()       # marker size updates immediately
            self._request_fit()        # and so does the 1/size weighting
            return
        if col in (4, 5, 6):
            item = self.feature_table.item(row, col)
            try:
                value = float(item.text())
            except (TypeError, ValueError):
                return
            attr = {4: "a", 5: "b", 6: "y"}[col]
            getattr(self._model, attr)[row] = value
            self._evaluate()

    def _refresh_plots(self) -> None:
        if self._data is None:
            return
        for combo, plot in zip(self._plot_selectors, self._plot_widgets):
            plot.clear()
            self._render_plot(combo.currentText(), plot)

    def _label_counts_per_view(self) -> np.ndarray:
        return self._labels.counts_per_view(self._data.angles.size)

    def _render_shift_plot(self, plot, values, observed, color,
                           name: str) -> None:
        """Dashed model curve, dots on labeled views, and the missing
        frames made explicit: orange dots where ONE label carries the view
        (its shift is that label, warning W4) and red base ticks where NO
        label exists (the shift is pure interpolation)."""
        deg = np.rad2deg(self._data.angles)
        counts = self._label_counts_per_view()
        plot.setLabel("left", f"{name} [raw px]")
        plot.setLabel("bottom", "angle [deg]")
        plot.plot(deg, values, pen=pg.mkPen(*color, 90, width=1,
                                            style=Qt.PenStyle.DashLine))
        solid = observed & (counts >= 2)
        thin = observed & (counts == 1)
        if solid.any():
            plot.plot(deg[solid], values[solid], pen=None, symbol="o",
                      symbolSize=5, symbolBrush=color, symbolPen=None)
        if thin.any():
            plot.plot(deg[thin], values[thin], pen=None, symbol="o",
                      symbolSize=5, symbolBrush=(255, 140, 0),
                      symbolPen=None)
        missing = counts == 0
        if missing.any() and values.size:
            base = float(values.min()) - 0.08 * (float(np.ptp(values)) or 1.0)
            plot.plot(deg[missing], np.full(int(missing.sum()), base),
                      pen=None, symbol="t1", symbolSize=6,
                      symbolBrush=(255, 60, 60), symbolPen=None)

    def _render_plot(self, kind: str, plot) -> None:
        deg = np.rad2deg(self._data.angles)
        fit = self._fit

        if kind == "labels per view":
            counts = self._label_counts_per_view()
            plot.setLabel("left", "labels in view")
            plot.setLabel("bottom", "angle [deg]")
            plot.plot(deg, counts, pen=pg.mkPen((200, 200, 200), width=1),
                      symbol="o", symbolSize=4,
                      symbolBrush=(200, 200, 200), symbolPen=None)
            missing = counts == 0
            if missing.any():
                plot.plot(deg[missing], np.zeros(int(missing.sum())),
                          pen=None, symbol="t1", symbolSize=7,
                          symbolBrush=(255, 60, 60), symbolPen=None)
            plot.addLine(y=2, pen=pg.mkPen((255, 140, 0, 120),
                                           style=Qt.PenStyle.DashLine))
            return

        if fit is None:
            return
        model = fit.model
        i, j = fit.obs

        if kind == "dx shifts":
            self._render_shift_plot(plot, model.dx, fit.observed_views,
                                    (255, 200, 0), "dx")
        elif kind == "dy shifts":
            self._render_shift_plot(plot, model.dy, fit.observed_views,
                                    (0, 200, 255), "dy")
        elif kind in ("residual u", "residual v"):
            res = fit.residual_u if kind == "residual u" else fit.residual_v
            plot.setLabel("left", f"{kind} [raw px]")
            plot.setLabel("bottom", "angle [deg]")
            brushes = [pg.mkBrush(*feature_color(int(model.feature_ids[fi])))
                       for fi in i]
            plot.plot(deg[j], res, pen=None, symbol="o", symbolSize=5,
                      symbolBrush=brushes, symbolPen=None)
            plot.addLine(y=0, pen=pg.mkPen((255, 255, 255, 60)))
        elif kind == "per-view spread":
            from tktomo.tracking.diagnostics import (  # noqa: PLC0415
                per_view_spread,
            )
            plot.setLabel("left", "MAD [raw px], u yellow / v cyan")
            plot.setLabel("bottom", "angle [deg]")
            for res, color in ((fit.residual_u, (255, 200, 0)),
                               (fit.residual_v, (0, 200, 255))):
                spread = per_view_spread(res, j, model.theta.size)
                good = np.isfinite(spread)
                plot.plot(deg[good], spread[good],
                          pen=pg.mkPen(color, width=1.5))
            plot.addLine(y=1.0, pen=pg.mkPen((255, 80, 80, 120),
                                             style=Qt.PenStyle.DashLine))
        elif kind == "axis center c":
            plot.setLabel("left", "c [raw px]")
            plot.setLabel("bottom", "angle [deg]")
            c_of, _, _ = model.axis_curves()
            plot.plot(deg, c_of, pen=pg.mkPen((255, 255, 255), width=1.5))
        elif kind == "tilts alpha/beta":
            plot.setLabel("left", "tilt [rad], alpha yellow / beta cyan")
            plot.setLabel("bottom", "angle [deg]")
            _, alpha_of, beta_of = model.axis_curves()
            plot.plot(deg, alpha_of, pen=pg.mkPen((255, 200, 0), width=1.5))
            plot.plot(deg, beta_of, pen=pg.mkPen((0, 200, 255), width=1.5))
        elif kind == "residual histogram":
            plot.setLabel("left", "count, u yellow / v cyan")
            plot.setLabel("bottom", "residual [raw px]")
            for res, color in ((fit.residual_u, (255, 200, 0, 140)),
                               (fit.residual_v, (0, 200, 255, 140))):
                if res.size:
                    # a perfect fit has (near-)zero range; bins must stay
                    # representable against the residuals' magnitude
                    span = max(float(np.ptp(res)),
                               1e-6 * max(1.0, float(np.abs(res).max())))
                    lo = float(res.min()) - 0.05 * span
                    hi = float(res.max()) + 0.05 * span
                    counts, edges = np.histogram(res, bins=60,
                                                 range=(lo, hi))
                    plot.plot(edges, counts, stepMode="center",
                              fillLevel=0, brush=pg.mkBrush(*color),
                              pen=pg.mkPen(*color))

    # ----------------------------------------------------------------- 3D

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self._view3d_holder:
            self._build_3d_once()
            self._refresh_3d()

    def _build_3d_once(self) -> None:
        if self._view3d_built:
            return
        self._view3d_built = True
        layout = self._view3d_holder.layout()
        try:
            import pyqtgraph.opengl as gl  # noqa: PLC0415
        except Exception as exc:  # ImportError or missing GL driver
            layout.addWidget(QLabel(
                f"3D view needs PyOpenGL (pip install PyOpenGL).\n{exc}"))
            self._gl = None
            return
        self._gl = gl
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=600)
        self._gl_grid = gl.GLGridItem()
        self._gl_grid.scale(20, 20, 1)
        self._gl_centered = False
        self.view3d.addItem(self._gl_grid)
        self._gl_axis = gl.GLLinePlotItem(width=2, antialias=True)
        self.view3d.addItem(self._gl_axis)
        self._gl_points = gl.GLScatterPlotItem(pxMode=True, size=9)
        self.view3d.addItem(self._gl_points)
        self._gl_cloud = gl.GLScatterPlotItem(pxMode=True, size=3)
        self.view3d.addItem(self._gl_cloud)
        self._gl_arcs: list = []
        layout.addWidget(self.view3d)

    def _refresh_3d(self) -> None:
        if not self._view3d_built or getattr(self, "_gl", None) is None:
            return
        if self._fit is None:
            return
        gl = self._gl
        model = self._fit.model
        i, j = self._fit.obs

        # The horizontal grid plane sits at the stack's CENTER ROW (in raw
        # coordinates, like everything in the scene), so height reads
        # relative to the middle of the field of view instead of raw zero,
        # which for a cropped stack is nowhere near the data.
        if self._data is not None:
            center_row = (self._data.data.shape[1] - 1) / 2.0
            view = self._view if self._chain.view_origin is not None else None
            _, v_center = self._chain.to_parent(0.0, center_row, view=view)
            z_grid = -float(v_center)
            self._gl_grid.resetTransform()
            self._gl_grid.scale(20, 20, 1)
            self._gl_grid.translate(0, 0, z_grid)
            if not self._gl_centered:
                self._gl_centered = True
                self.view3d.opts["center"] = pg.Vector(0, 0, z_grid)
                self.view3d.update()
        counts = self._labels.counts()
        keep = np.array([counts.get(int(f), 0) > 0
                         for f in model.feature_ids], bool)

        # fitted feature points in the rotating frame; z is the detector row
        # direction, drawn negated so "up in the image" is up on screen
        pos = np.column_stack([model.a[keep], model.b[keep],
                               -model.y[keep]])
        colors = np.array([(*feature_color(int(f)), 255)
                           for f in model.feature_ids[keep]],
                          float) / 255.0
        self._gl_points.setData(pos=pos, color=colors)

        # back-projected labels: feature position + residual along the view
        # direction (u residual on e_s, v residual vertical)
        ct, sn = np.cos(model.theta[j]), np.sin(model.theta[j])
        cloud = np.column_stack([
            model.a[i] + self._fit.residual_u * ct,
            model.b[i] + self._fit.residual_u * sn,
            -(model.y[i] + self._fit.residual_v)])
        cloud_colors = np.array(
            [(*feature_color(int(model.feature_ids[fi])), 110)
             for fi in i], float) / 255.0
        self._gl_cloud.setData(pos=cloud, color=cloud_colors)

        for item in self._gl_arcs:
            self.view3d.removeItem(item)
        self._gl_arcs.clear()
        for row in np.flatnonzero(keep):
            radius = float(np.hypot(model.a[row], model.b[row]))
            phase = float(np.arctan2(model.b[row], model.a[row]))
            arc_theta = np.linspace(0, 2 * np.pi, 90)
            pts = np.column_stack([
                radius * np.cos(phase + arc_theta),
                radius * np.sin(phase + arc_theta),
                np.full(arc_theta.size, -model.y[row])])
            color = (*(np.array(feature_color(
                int(model.feature_ids[row]))) / 255.0), 0.35)
            item = gl.GLLinePlotItem(pos=pts, color=color, width=1,
                                     antialias=True)
            self.view3d.addItem(item)
            self._gl_arcs.append(item)

        # the rotation axis: x = y = 0 in the rotating frame BY DEFINITION;
        # c(theta) drift and tilts live in the projection model, so the
        # interesting part is drawn as the axis plus its detector-frame
        # center curve is in the Residuals tab.
        if pos.size:
            z0, z1 = float(cloud[:, 2].min()) - 30, float(
                cloud[:, 2].max()) + 30
        else:
            z0, z1 = -100, 100
        self._gl_axis.setData(pos=np.array([[0, 0, z0], [0, 0, z1]]),
                              color=(1, 1, 1, 0.9))

    # -------------------------------------------------------------- recon

    def _recon_maybe(self) -> None:
        if self.live_recon.isChecked():
            self._recon_timer.start()

    def _request_recon(self) -> None:
        if self._data is None or self._model is None or self._fit is None:
            self.recon_status.setText("fit a model first")
            return
        if self._chain.view_origin is not None:
            self.recon_status.setText(
                "recon of a moving-crop stack is not meaningful")
            return
        if self._recon_worker is None:
            self._recon_worker = ReconWorker(self)
            self._recon_worker.finished_slice.connect(self._show_slice)
            self._recon_worker.failed.connect(
                lambda msg: self.recon_status.setText(f"recon failed: {msg}"))
        data = self._data
        n_rows = data.data.shape[1]
        self.slice_row.setMaximum(n_rows - 1)
        row = int(self.slice_row.value())
        model = self._model
        c_of, alpha_of, _ = model.axis_curves()
        c_ref = model.center_at_mean_theta()
        sx = -self._chain.shift_from_parent(model.dx + (c_of - c_ref))
        sy = -self._chain.shift_from_parent(model.dy)
        rot_deg = -np.rad2deg(alpha_of)
        # the slab must be tall enough that shifting AND derotating still
        # leaves the middle row valid: rotation moves rows by up to
        # |alpha| * width/2 at the image edges
        width = data.data.shape[2]
        margin = 4 + int(np.ceil(np.abs(alpha_of).max() * width / 2.0
                                 + np.abs(sy).max()))
        lo = max(0, row - margin)
        hi = min(n_rows, row + margin + 1)
        slab = data.data[:, lo:hi, :]
        center = float(self._chain.from_parent(c_ref, 0.0)[0])
        row_in_slab = row - lo
        extra_bin = int(self.recon_bin.currentText())
        if extra_bin > 1:
            # bin BEFORE warping: isotropic mean-pooling preserves the
            # rotation angle, and shifts/center rescale by the pixel-center
            # rule. The slice lands within half a binned pixel of the
            # requested row, which is what a speed preview is for.
            from tktomo.ptycho_align.core.preprocess import (  # noqa: PLC0415
                bin_stack,
            )
            slab = bin_stack(np.asarray(slab, np.float32), extra_bin)
            sy = sy / extra_bin
            sx = sx / extra_bin
            center = (center - (extra_bin - 1) / 2.0) / extra_bin
            row_in_slab = min(row_in_slab // extra_bin, slab.shape[1] - 1)
        self.recon_status.setText(
            f"reconstructing row {row} (bin {extra_bin})…")
        self._last_recon_info = {"extra_bin": extra_bin,
                                 "row_loaded": row,
                                 "width": slab.shape[2]}
        self._recon_worker.submit(slab, data.angles, sy, sx, rot_deg,
                                  center, row_in_slab)

    def _show_slice(self, image) -> None:
        self.recon_status.setText("")
        self.recon_display.set_image(np.asarray(image))

    # ------------------------------------------------------------ loading

    def _load(self, path: str | None) -> None:
        result = load_stack_interactive(self, path)
        if result is None:
            return
        data, chain, source = result
        if len(self._labels) and QMessageBox.question(
                self, "Discard labels?",
                "Loading a new stack discards labels and model. Continue?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self._labels = LabelStore()
        self._model = None
        self._mask = None
        self._fit = None
        self._pins.clear()
        self._show_data(data, chain, source)

    def _show_data(self, data: ProjectionData, chain: CoordinateChain,
                   source: dict) -> None:
        self._data, self._chain, self._source = data, chain, source
        n = data.data.shape[0]
        self.slider.setMaximum(n - 1)
        self.view_box.setMaximum(n - 1)
        self.slice_row.setMaximum(data.data.shape[1] - 1)
        self.slice_row.setValue(data.data.shape[1] // 2)
        self._view = min(self._view, n - 1)
        self.viewer.reset_levels(data.data[self._view])
        self._refresh_view()

    # ------------------------------------------------------------ session

    def _ui_state(self) -> dict:
        return {
            "active_feature": self._active,
            "advance": self.advance_box.value(),
            "degrees": [self.deg_c.value(), self.deg_a.value(),
                        self.deg_b.value()],
            "huber": self.huber.value(),
            "iters": self.iters.value(),
            "auto_fit": self.auto_fit.isChecked(),
            "slice_row": self.slice_row.value(),
            "pins": sorted(self._pins),
            "view": self._view,
            "feature_sizes": {str(k): float(size)
                              for k, size in self._feature_sizes.items()},
            "ghosts": self.ghost_box.isChecked(),
            "auto_min_corr": self.auto_min_corr.value(),
            "auto_search_radius": self.auto_radius.value(),
            "auto_fb_check": self.auto_fb.isChecked(),
        }

    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "track_session.h5", "HDF5 (*.h5)")
        if not path:
            return
        sessionio.save_session(path, labels=self._labels, model=self._model,
                               mask=self._mask, source=self._source,
                               ui_state=self._ui_state())

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", "", "HDF5 (*.h5)")
        if not path:
            return
        try:
            state = sessionio.load_session(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Could not load session", str(exc))
            return
        source = state["source"]
        stack_path = source.get("path")
        if stack_path and Path(str(stack_path)).exists():
            result = load_stack_interactive(self, str(stack_path))
            if result is None:
                return
            data, chain, new_source = result
            self._show_data(data, chain, new_source)
        elif self._data is None:
            QMessageBox.warning(
                self, "Stack missing",
                f"The session's stack ({stack_path}) is not readable; "
                f"load the stack first, then the session.")
            return
        self._labels = state["labels"]
        self._model = state["model"]
        self._mask = state["mask"]
        ui = state["ui"]
        self._pins = set(ui.get("pins", []))
        self._feature_sizes = {int(k): float(size) for k, size in
                               ui.get("feature_sizes", {}).items()}
        self.ghost_box.setChecked(bool(ui.get("ghosts", False)))
        self.auto_min_corr.setValue(float(ui.get("auto_min_corr", 0.30)))
        self.auto_radius.setValue(float(ui.get("auto_search_radius", 8.0)))
        self.auto_fb.setChecked(bool(ui.get("auto_fb_check", True)))
        self.advance_box.setValue(int(ui.get("advance", 5)))
        degrees = ui.get("degrees", [0, 0, 0])
        self.deg_c.setValue(int(degrees[0]))
        self.deg_a.setValue(int(degrees[1]))
        self.deg_b.setValue(int(degrees[2]))
        self.huber.setValue(float(ui.get("huber", 3.0)))
        self.iters.setValue(int(ui.get("iters", 4)))
        self.auto_fit.setChecked(bool(ui.get("auto_fit", True)))
        # live recon stays OFF on load regardless of what was saved
        self.live_recon.setChecked(False)
        self._set_active(int(ui.get("active_feature", 0)))
        self._set_view(int(ui.get("view", 0)))
        if self._model is not None:
            self._push_model_to_ui()
            self._evaluate()

    # ------------------------------------------------------------ exports

    def _export_model(self) -> None:
        if self._fit is None or self._mask is None:
            QMessageBox.information(self, "No fit", "Fit a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model", "track_model.h5", "HDF5 (*.h5)")
        if not path:
            return
        chain = self._chain
        det_h = chain.crop[1] - chain.crop[0]
        det_w = chain.crop[3] - chain.crop[2]
        if det_h <= 0 or det_w <= 0:
            det_h = self._data.data.shape[1] * chain.binning + chain.crop[0]
            det_w = self._data.data.shape[2] * chain.binning + chain.crop[2]
        write_model_h5(path, self._fit, self._mask, self._labels, chain,
                       source=self._source, diagnostics=self._diagnostics,
                       det_shape=(int(det_h), int(det_w)))
        QMessageBox.information(self, "Exported", f"Wrote {path}")

    def _export_shifts(self) -> None:
        if self._fit is None:
            QMessageBox.information(self, "No fit", "Fit a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export slogger shifts", "shifts.h5", "HDF5 (*.h5)")
        if not path:
            return
        split = (self._diagnostics or {}).get("center_split_px",
                                              float("nan"))
        write_slogger_shifts(path, self._fit, self._chain,
                             target_binning=self._chain.binning,
                             source=str(self._source.get("path", "")),
                             center_split_raw_px=float(split))
        if self._diagnostics is None:
            QMessageBox.information(
                self, "Exported without diagnostics",
                f"Wrote {path}. Run diagnostics before exporting to give "
                f"the file an honest center_split_px; without it the "
                f"center is marked unreliable.")
        else:
            QMessageBox.information(self, "Exported", f"Wrote {path}")

    def _export_aligned(self) -> None:
        if self._fit is None or self._data is None:
            QMessageBox.information(self, "No fit", "Fit a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export aligned stack", "aligned.h5", "HDF5 (*.h5)")
        if not path:
            return
        n = self._data.data.shape[0]
        progress = QProgressDialog("Warping projections…", "Cancel", 0, n,
                                   self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        def report(done, total):
            progress.setValue(done)
            return not progress.wasCanceled()

        try:
            aligned = export_aligned_stack(self._data, self._model,
                                           self._chain, progress=report)
        except (RuntimeError, ValueError) as exc:
            progress.close()
            QMessageBox.warning(self, "Export stopped", str(exc))
            return
        progress.close()
        save_projections(path, aligned)
        QMessageBox.information(
            self, "Exported",
            f"Wrote {path}\nfixed center on this grid: "
            f"{aligned.metadata['center_loaded_px']:.2f} px")


def _fmt(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="projection stack (.h5)")
    args = parser.parse_args()
    return run_app(lambda: TrackModelWindow(args.path))


if __name__ == "__main__":
    raise SystemExit(main())

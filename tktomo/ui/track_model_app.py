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
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
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

from tktomo.io import ProjectionData
from tktomo.io.phantom import generate_phantom
from tktomo.tracking import sessionio
from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.diagnostics import run_diagnostics
from tktomo.tracking.export import (
    aligned_metadata,
    aligned_view_transforms,
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
from tktomo.tracking.recon import plan_slice
from tktomo.tracking.stacksource import (
    AlignedExportRequest,
    LocalStackSource,
    StackSource,
    ViewPrefetcher,
)
from tktomo.ui.common import run_app
from tktomo.ui.tracking_widgets import (
    MarkableStackView,
    feature_color,
    open_stack_interactive,
    pick_stack_path,
    run_source_job,
)

# plot kinds selectable in the two residual panes; the first two are the
#: labels the active feature needs before "follow the prediction"
#: pans to the model's guess: fewer and the guess is not worth chasing
FOLLOW_MIN_LABELS = 4

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
    """Runs the live gridrec slice off the GUI thread, one job at a time.

    TomoPy segfaults when reconstructing from two threads at once, and a
    stray call on the GUI thread freezes the app for seconds, so ALL
    reconstruction goes through this single persistent worker. The window
    keeps a dirty flag: a request arriving while the worker is busy
    replaces the pending one and runs when the current job finishes
    (single-flight; intermediate states are never queued up).

    The numerics live in `tktomo.tracking.recon`; the worker only hands the
    `SliceRequest` to the stack source, which cuts the slab wherever the
    pixels are (in this process, or on a node) and returns the slice.
    """

    finished_slice = Signal(object)      # 2-D array
    failed = Signal(str)

    def __init__(self, source: StackSource, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._job = None

    def set_source(self, source: StackSource) -> None:
        self._source = source

    def submit(self, req) -> None:
        self._job = req
        if not self.isRunning():
            self.start()

    def run(self):  # noqa: N802
        while self._job is not None:
            req = self._job
            self._job = None
            try:
                self.finished_slice.emit(self._source.gridrec_slice(req))
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                self.failed.emit(str(exc))


class AutoTrackWorker(QThread):
    """Runs seeded track completion off the GUI thread, one batch at a time.

    Same single-flight contract as ReconWorker: `submit` overwrites the
    pending job, `run()` loops until none is left, results leave only via
    signals. The stack is never copied here: the source owns the pixels
    and the high-pass cache, and `tktomo.tracking.autotrack.run_autotrack`
    does the work wherever the source lives.
    """

    progress = Signal(int, int, object)     # done, total, fid (None = highpass)
    finished_tracks = Signal(object)        # [(fid, TrackResult)]
    failed = Signal(str)

    def __init__(self, source: StackSource, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._job = None
        self._cancelled = False

    def set_source(self, source: StackSource) -> None:
        self._source = source

    def submit(self, jobs, hp_sigma: float = 12.0) -> None:
        self._cancelled = False
        self._job = (list(jobs), float(hp_sigma))
        if not self.isRunning():
            self.start()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):  # noqa: N802
        while self._job is not None:
            jobs, hp_sigma = self._job
            self._job = None
            try:
                out = self._source.autotrack(
                    jobs, hp_sigma=hp_sigma, progress=self.progress.emit,
                    cancelled=lambda: self._cancelled)
                self.finished_tracks.emit(list(out))
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                self.failed.emit(str(exc))


class TrackModelWindow(QMainWindow):
    def __init__(self, path: str | None = None,
                 source: StackSource | None = None) -> None:
        super().__init__()
        self._stack: StackSource = (source if source is not None
                                    else LocalStackSource())
        title = "TKtomo track model"
        if self._stack.is_remote:
            title += f"  [stack on {self._stack.describe()}]"
        self.setWindowTitle(title)

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
        self._prefetch: ViewPrefetcher | None = None

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
        self.bin_combo = QComboBox()
        for k in (1, 2, 4, 8):
            self.bin_combo.addItem(str(k), k)
        self.bin_combo.setToolTip(
            "Mean-pool the projections by this factor before anything "
            "looks at them: the view, auto-track, the recon slice and the "
            "aligned export. Labels are kept in raw pixels, so switching "
            "back and forth loses nothing. On a remote stack a frame "
            "shrinks by the factor squared, which is what makes a slow "
            "link bearable.")
        self.bin_combo.currentIndexChanged.connect(self._bin_combo_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel("View:"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.view_box)
        row.addWidget(self.angle_label)
        row.addWidget(QLabel("bin:"))
        row.addWidget(self.bin_combo)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.viewer, 1)
        row_w = QWidget()
        row_w.setLayout(row)
        left_layout.addWidget(row_w)

        # top: projection viewer and recon slice as tabs; bottom: one
        # residual plot, 3:1 by default; controls on the right
        self.tabs = QTabWidget()
        self.tabs.addTab(left, "Projection")
        self.tabs.addTab(self._build_recon_panel(), "Recon slice")
        self.vsplitter = QSplitter(Qt.Orientation.Vertical)
        self.vsplitter.addWidget(self.tabs)
        self.vsplitter.addWidget(self._build_plots_panel())
        self.vsplitter.setStretchFactor(0, 3)
        self.vsplitter.setStretchFactor(1, 1)
        self.vsplitter.setSizes([660, 220])
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.vsplitter)
        splitter.addWidget(self._build_controls())
        splitter.setSizes([1040, 420])
        self.setCentralWidget(splitter)
        self._build_menu()
        self.resize(1500, 900)

        if path:
            self._load(path)
        elif self._stack.is_remote:
            info = self._stack.info()
            if info is not None:
                # the server was started with a stack: adopt it as-is, with
                # the provenance its file declares
                chain = CoordinateChain(binning=info.binning, crop=info.crop,
                                        view_origin=info.view_origin,
                                        rebin=info.rebin)
                self._adopt_stack(chain, {
                    "path": info.path, "kind": info.kind,
                    "binning": chain.binning, "crop": list(chain.crop),
                    "endpoint": self._stack.describe()})
        else:
            self._show_data(generate_phantom(60, 128, 48, max_shift=3.0),
                            CoordinateChain(), {"kind": "phantom"})

    # ------------------------------------------------------------------ UI

    def _build_menu(self) -> None:
        """File menu: everything that reads or writes a file lives here."""
        menu = self._file_menu = self.menuBar().addMenu("&File")

        def add(text, slot, shortcut=None, tip=None):
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if tip:
                action.setStatusTip(tip)
                action.setToolTip(tip)
            menu.addAction(action)
            return action

        add("Open remote stack…" if self._stack.is_remote else "Load stack…",
            lambda: self._load(None), "Ctrl+O",
            ("Open a projection stack by its path on "
             f"{self._stack.describe()}. Replaces the labels and model.")
            if self._stack.is_remote else
            "Open a projection stack (.h5). Replaces the labels and model.")
        menu.addSeparator()
        add("Load session…", lambda: self._load_session(None), "Ctrl+Shift+O",
            "Restore labels, model, masks and UI state from a session file.")
        add("Save session…", self._save_session, "Ctrl+S",
            "Write labels, model, masks and UI state to a session file. "
            "The stack itself is referenced by path, not copied.")
        menu.addSeparator()
        export = menu.addMenu("Export")
        for text, slot, tip in (
                ("Model + astra vectors (.h5)…", self._export_model,
                 "Fitted model in raw detector coordinates plus ASTRA "
                 "parallel3d_vec geometry."),
                ("slogger shifts.h5…", self._export_shifts,
                 "Per-view shifts in the slogger layout, on the loaded "
                 "(binned, cropped) grid."),
                ("Aligned projection stack…", self._export_aligned,
                 "The loaded stack re-warped by the fitted per-view shifts.")):
            action = QAction(text, self)
            action.triggered.connect(slot)
            action.setToolTip(tip)
            action.setStatusTip(tip)
            export.addAction(action)
        menu.addSeparator()
        add("Quit", self.close, "Ctrl+Q")

    def _build_plots_panel(self) -> QWidget:
        """One residual plot with a plot-kind dropdown. Default labels per view.

        Clicking in any angle-axis plot jumps to the nearest frame, which
        turns the plot into navigation: see a gap or a bad point, click
        it, and you are looking at that projection.
        """
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        panes = QHBoxLayout()
        layout.addLayout(panes, 1)
        self._plot_selectors: list[QComboBox] = []
        self._plot_widgets: list[pg.PlotWidget] = []
        for default in ("labels per view",):
            combo = QComboBox()
            combo.addItems(PLOT_KINDS)
            combo.setCurrentText(default)
            combo.currentTextChanged.connect(
                lambda _t: self._refresh_plots())
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.scene().sigMouseClicked.connect(
                lambda ev, c=combo: self._plot_clicked(ev, c))
            pane = QVBoxLayout()
            head = QHBoxLayout()
            head.addWidget(QLabel("Residuals:"))
            head.addWidget(combo)
            head.addStretch(1)
            pane.addLayout(head)
            pane.addWidget(plot, 1)
            panes.addLayout(pane, 1)
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
        if self._stack.info() is None:
            return
        deg = np.rad2deg(self._stack.angles)
        self._set_view(int(np.argmin(np.abs(deg - x_deg))))

    def _build_recon_panel(self) -> QWidget:
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
        if self._last_recon_info is None or self._stack.info() is None:
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
        scale = info["extra_bin"] * self._chain.scale
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
        feat_layout.addLayout(feat_row)
        adv_row = QHBoxLayout()
        self.active_label = QLabel("active feature: 0")
        adv_row.addWidget(self.active_label, 1)
        adv_row.addWidget(QLabel("advance"))
        self.advance_box = QSpinBox()
        self.advance_box.setRange(0, 100)
        self.advance_box.setValue(5)
        self.advance_box.setToolTip(
            "How many views the viewer steps forward after each placed "
            "label. 0 stays on the current view.")
        adv_row.addWidget(self.advance_box)
        adv_row.addWidget(QLabel("views/click"))
        feat_layout.addLayout(adv_row)
        self.ghost_box = QCheckBox("Ghost labels from other frames")
        self.ghost_box.setToolTip(
            "Draw the active feature's labels from OTHER frames as faint "
            "half-size circles, to judge where the next click belongs.")
        self.ghost_box.toggled.connect(lambda _c: self._refresh_view())
        feat_layout.addWidget(self.ghost_box)
        self.follow_box = QCheckBox("Follow the prediction after a click")
        self.follow_box.setToolTip(
            f"After a placed label (click or Space) pan the view so the "
            f"model's predicted position of the active feature in the view "
            f"shown next sits at the centre, at the current zoom. Only acts "
            f"once the active feature has more than {FOLLOW_MIN_LABELS} "
            f"labels and a fit exists, so the prediction means something. "
            f"Unchecked leaves the view where you put it.")
        feat_layout.addWidget(self.follow_box)
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
        self._learned_ok, self._learned_why = self._stack.autotrack_available()
        self.auto_thr_label = QLabel("min p")
        self.auto_min_corr = QDoubleSpinBox()
        self.auto_min_corr.setRange(0.05, 0.95)
        self.auto_min_corr.setSingleStep(0.05)
        self.auto_min_corr.setValue(0.20)
        self.auto_min_corr.setToolTip(
            "The 5 nearest manual labels vote on the position and a "
            "classifier rates the answer with the probability it is within "
            "4 raw px. Answers below this probability end the track. 0.20 "
            "keeps ~90% of answers from 10-view anchors with about 1% worse "
            "than 10 px; raise it when coverage is generous.")
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
        param_row.addWidget(self.auto_thr_label)
        param_row.addWidget(self.auto_min_corr)
        param_row.addWidget(QLabel("search"))
        param_row.addWidget(self.auto_radius)
        param_row.addWidget(self.auto_fb)
        auto_layout.addLayout(param_row)
        reject_row = QHBoxLayout()
        self.auto_reject = QCheckBox("reject auto >")
        self.auto_reject.setChecked(True)
        self.auto_reject.setToolTip(
            "After every fit, drop AUTO labels whose residual exceeds this "
            "multiple of the Huber scale and refit once. Manual labels are "
            "never touched. A Huber-weighted 15 px lock-on still drags a "
            "thinly covered view's shift by several px; removing it does "
            "not. Measured: dx rms 2.95 -> 2.43 raw px at 10-view anchors.")
        self.auto_reject_k = QDoubleSpinBox()
        self.auto_reject_k.setRange(1.0, 20.0)
        self.auto_reject_k.setSingleStep(0.5)
        self.auto_reject_k.setValue(3.0)
        self.auto_reject_k.setSuffix(" x Huber")
        self.auto_reject.toggled.connect(lambda _: self._request_fit())
        self.auto_reject_k.valueChanged.connect(lambda _: self._request_fit())
        reject_row.addWidget(self.auto_reject)
        reject_row.addWidget(self.auto_reject_k)
        reject_row.addStretch(1)
        auto_layout.addLayout(reject_row)
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
        deg_row.addStretch(1)
        deg_holder = QWidget()
        deg_holder.setLayout(deg_row)
        deg_holder.setToolTip(
            "Polynomial degree in theta of the axis center c, the in-plane "
            "tilt alpha and the out-of-plane tilt beta.")
        self._model_form.addRow("Degrees:", deg_holder)
        self._coef_rows: dict[str, list[tuple[QCheckBox, QDoubleSpinBox]]] = {
            "c": [], "alpha": [], "beta": []}
        self._rebuild_coef_rows()
        self.free_dx = QCheckBox("dx free")
        self.free_dx.setChecked(False)
        self.free_dy = QCheckBox("dy free")
        self.free_dy.setChecked(False)
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
        for label, check, zero in (("Shift dx:", self.free_dx, zero_dx),
                                   ("Shift dy:", self.free_dy, zero_dy)):
            check.toggled.connect(lambda _c: self._request_fit())
            shift_row = QHBoxLayout()
            shift_row.addWidget(check)
            shift_row.addWidget(zero)
            shift_row.addStretch(1)
            shift_holder = QWidget()
            shift_holder.setLayout(shift_row)
            self._model_form.addRow(label, shift_holder)
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
        robust_row.addWidget(QLabel("iters"))
        robust_row.addWidget(self.iters)
        robust_row.addStretch(1)
        robust_holder = QWidget()
        robust_holder.setLayout(robust_row)
        robust_holder.setToolTip(
            "Huber threshold k in units of the residual scale, and the "
            "number of reweighting (IRLS) passes per fit.")
        self._model_form.addRow("Robust:", robust_holder)
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
        layout.addLayout(fit_row)
        fit_row = QHBoxLayout()
        unlabeled_btn = QPushButton("Next unlabelled")
        unlabeled_btn.setToolTip(
            "Step forward (wrapping around) to the next projection that "
            "carries no label at all, always leaving the current one even "
            "if it is unlabelled itself. Once every projection has a "
            "label, steps to the next one with the fewest labels. Views "
            "without labels are where the per-view shifts are only "
            "interpolated, never measured.")
        unlabeled_btn.clicked.connect(self._goto_next_unlabelled)
        fit_row.addWidget(outlier_btn)
        fit_row.addWidget(unlabeled_btn)
        layout.addLayout(fit_row)

        self.summary_label = QLabel("no fit yet")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.warnings_box = QPlainTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(140)
        layout.addWidget(self.warnings_box)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(
            panel.minimumSizeHint().width() + scroll.frameWidth() * 2
            + scroll.verticalScrollBar().sizeHint().width())
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
                if group == "beta":
                    # out-of-plane tilt is fixed at zero until asked for:
                    # it is weakly constrained by few features and trades
                    # off against y and dy
                    check.setChecked(True)
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
        if self._stack.info() is None:
            return
        ids = self._labels.feature_ids()
        if self._active not in ids:
            ids = sorted(set(ids) | {self._active})
        theta = self._stack.angles
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
        if self._stack.info() is None:
            return
        u_raw, v_raw = self._chain.to_parent(u, v, view=self._view)
        self._labels.set(self._active, self._view, float(u_raw),
                         float(v_raw))
        advance = self.advance_box.value()
        if advance:
            self._set_view(self._view + advance)
        else:
            self._refresh_view()
        self._follow_prediction()
        self._refresh_plots()      # label coverage updates even without a fit
        self._request_fit()

    def _predicted_position(self, fid: int, view: int
                            ) -> tuple[float, float] | None:
        """The fitted model's position of feature `fid` in `view`, on the
        loaded grid, or None without a fit that knows the feature."""
        if self._fit is None or self._model is None \
                or self._model.theta.size != self._stack.angles.size:
            return None
        ids = list(self._model.feature_ids)
        if fid not in ids:
            return None
        row = ids.index(fid)
        u_pred, v_pred = self._model.predict()
        u, v = self._chain.from_parent(u_pred[row, view], v_pred[row, view],
                                       view=view)
        return float(u), float(v)

    def _follow_prediction(self) -> None:
        """Recentre the view on the active feature's predicted position,
        when the follow box is ticked and the feature has enough labels for
        the prediction to be trusted (see FOLLOW_MIN_LABELS)."""
        if not self.follow_box.isChecked():
            return
        if self._labels.counts().get(self._active, 0) <= FOLLOW_MIN_LABELS:
            return
        pos = self._predicted_position(self._active, self._view)
        if pos is not None:
            self.viewer.center_on(*pos)

    def _delete_near(self, u: float, v: float) -> None:
        if self._stack.info() is None:
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

    def _goto_next_unlabelled(self) -> None:
        """Forward (wrapping) to the next view with no labels, or, once
        every view has some, to the next view with the fewest labels."""
        if self._stack.info() is None:
            return
        counts = self._label_counts_per_view()
        targets = np.flatnonzero(counts == counts.min())
        targets = targets[targets != self._view]
        if targets.size == 0:
            return
        ahead = targets[targets > self._view]
        self._set_view(int(ahead[0] if ahead.size else targets[0]))

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
            AutoTrackJob,
            AutoTrackParams,
            patch_size,
        )

        if self._stack.info() is None:
            return
        if self._chain.view_origin is not None:
            self.statusBar().showMessage(
                "auto-complete does not support per-view-cropped stacks "
                "(the crop already follows the feature)", 6000)
            return
        if not self._learned_ok:
            self.statusBar().showMessage(
                f"auto-complete needs the learned matcher: "
                f"{self._learned_why}", 8000)
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
            jobs.append(AutoTrackJob(fid=fid, seeds=tuple(seeds),
                                     params=params))

        if self._autotrack_worker is None:
            self._autotrack_worker = AutoTrackWorker(self._stack, self)
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
        self._autotrack_worker.submit(jobs, hp_sigma=12.0)

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
                    + (f", median p "
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
        if self._stack.info() is None or len(self._labels) == 0:
            return
        self._sync_model()
        u, v, valid, ids = self._labels.to_arrays(
            self._stack.angles.size, self._model.feature_ids)
        if not valid.any():
            return
        try:
            self._fit = solve_model(u, v, valid, self._model, self._mask,
                                    iters=self.iters.value(),
                                    huber=self.huber.value(),
                                    feature_weight=self._feature_weights())
            n_rej = self._reject_auto_outliers(ids)
            if n_rej:
                u, v, valid, ids = self._labels.to_arrays(
                    self._stack.angles.size, self._model.feature_ids)
                self._fit = solve_model(u, v, valid, self._fit.model,
                                        self._mask, iters=self.iters.value(),
                                        huber=self.huber.value(),
                                        feature_weight=self._feature_weights())
        except ValueError as exc:
            self.summary_label.setText(f"fit failed: {exc}")
            return
        if n_rej:
            self.auto_status.setText(
                f"rejected {n_rej} auto label(s) with residual > "
                f"{self.auto_reject_k.value():g} x Huber, refitted")
            self._refresh_view()
        self._model = self._fit.model
        self._diagnostics = None
        self._push_model_to_ui()
        self._after_evaluate()

    def _reject_auto_outliers(self, ids) -> int:
        """One pass of `reject_auto_outliers` if the checkbox is on."""
        if not self.auto_reject.isChecked() or self._fit is None:
            return 0
        from tktomo.tracking.labels import reject_auto_outliers  # noqa: PLC0415

        limit = float(self.auto_reject_k.value()) * float(self.huber.value())
        return reject_auto_outliers(self._labels, self._fit, ids, limit)

    def _evaluate(self) -> None:
        """Residuals against the CURRENT model values, no solving."""
        if self._stack.info() is None or self._model is None:
            return
        if len(self._labels) == 0:
            return
        self._sync_model_dims_only()
        u, v, valid, _ids = self._labels.to_arrays(
            self._stack.angles.size, self._model.feature_ids)
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
        if self._fit is None or self._stack.info() is None:
            return
        self._sync_model()
        u, v, valid, _ids = self._labels.to_arrays(
            self._stack.angles.size, self._model.feature_ids)
        self._diagnostics = run_diagnostics(u, v, valid, self._model,
                                            self._mask, self._fit,
                                            self._feature_weights())
        d = self._diagnostics
        chain = self._chain
        center_grid = chain.parent_to_grid(d["center_estimate_raw_px"],
                                           chain.scale)
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
        if self._stack.info() is None:
            return
        view = int(np.clip(view, 0, self._stack.shape[0] - 1))
        step, self._view = view - self._view, view
        if self._prefetch is not None:
            self._prefetch.want(view, step)
        for widget in (self.slider, self.view_box):
            widget.blockSignals(True)
            widget.setValue(view)
            widget.blockSignals(False)
        self._refresh_view()

    def _refresh_view(self) -> None:
        if self._stack.info() is None:
            return
        view = self._view
        self.angle_label.setText(
            f"{np.rad2deg(self._stack.angles[view]):7.2f}°")
        self.viewer.set_image(self._stack.view(view))

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
                and self._model.theta.size == self._stack.angles.size:
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
            theta_v = float(self._stack.angles[view])
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
        if self._stack.info() is None:
            return
        for combo, plot in zip(self._plot_selectors, self._plot_widgets):
            plot.clear()
            self._render_plot(combo.currentText(), plot)

    def _label_counts_per_view(self) -> np.ndarray:
        return self._labels.counts_per_view(self._stack.angles.size)

    def _render_shift_plot(self, plot, values, observed, color,
                           name: str) -> None:
        """Dashed model curve, dots on labeled views, and the missing
        frames made explicit: orange dots where ONE label carries the view
        (its shift is that label, warning W4) and red base ticks where NO
        label exists (the shift is pure interpolation)."""
        deg = np.rad2deg(self._stack.angles)
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
        deg = np.rad2deg(self._stack.angles)
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

    # -------------------------------------------------------------- recon

    def _recon_maybe(self) -> None:
        if self.live_recon.isChecked():
            self._recon_timer.start()

    def _request_recon(self) -> None:
        if self._stack.info() is None or self._model is None or self._fit is None:
            self.recon_status.setText("fit a model first")
            return
        if self._chain.view_origin is not None:
            self.recon_status.setText(
                "recon of a moving-crop stack is not meaningful")
            return
        if self._recon_worker is None:
            self._recon_worker = ReconWorker(self._stack, self)
            self._recon_worker.finished_slice.connect(self._show_slice)
            self._recon_worker.failed.connect(
                lambda msg: self.recon_status.setText(f"recon failed: {msg}"))
        _, n_rows, width = self._stack.shape
        self.slice_row.setMaximum(n_rows - 1)
        row = int(self.slice_row.value())
        extra_bin = int(self.recon_bin.currentText())
        req = plan_slice(self._model, self._chain, n_rows, width, row,
                         extra_bin)
        self.recon_status.setText(
            f"reconstructing row {row} (bin {extra_bin})…")
        self._last_recon_info = {"extra_bin": extra_bin,
                                 "row_loaded": row,
                                 "width": width // extra_bin}
        self._recon_worker.submit(req)

    def _show_slice(self, image) -> None:
        self.recon_status.setText("")
        self.recon_display.set_image(np.asarray(image))

    # ------------------------------------------------------------ loading

    def _load(self, path: str | None) -> None:
        if not path:
            path = pick_stack_path(self, self._stack)
            if not path:
                return
        result = open_stack_interactive(self, self._stack, path)
        if result is None:
            return
        _info, chain, source = result
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
        self._adopt_stack(chain, source)

    def _show_data(self, data: ProjectionData, chain: CoordinateChain,
                   source: dict) -> None:
        """Show an in-memory stack (the phantom, tests)."""
        self._stack = LocalStackSource.from_projection_data(
            data, chain, kind=source.get("kind"), path=source.get("path"))
        self._adopt_stack(chain, source)

    def _adopt_stack(self, chain: CoordinateChain, source: dict) -> None:
        """The stack behind `self._stack` changed: resize the UI to it."""
        info = self._stack.info()
        if info is not None and info.rebin != chain.rebin:
            # a source opened fresh serves the file's grid; say so in the chain
            chain = chain.with_rebin(info.rebin)
        self._chain, self._source = chain, {**source, "rebin": chain.rebin}
        self._updating_ui = True
        try:
            idx = self.bin_combo.findData(chain.rebin)
            if idx < 0:
                self.bin_combo.addItem(str(chain.rebin), chain.rebin)
                idx = self.bin_combo.count() - 1
            self.bin_combo.setCurrentIndex(idx)
        finally:
            self._updating_ui = False
        for worker in (self._recon_worker, self._autotrack_worker):
            if worker is not None:
                worker.set_source(self._stack)
        self._sync_prefetch()
        n, n_rows, _ = self._stack.shape
        self.slider.setMaximum(n - 1)
        self.view_box.setMaximum(n - 1)
        self.slice_row.setMaximum(n_rows - 1)
        self.slice_row.setValue(n_rows // 2)
        self._view = min(self._view, n - 1)
        self.viewer.reset_levels(self._stack.view(self._view))
        self._refresh_view()

    def _bin_combo_changed(self, _index: int) -> None:
        if self._updating_ui:
            return
        self._set_binning(int(self.bin_combo.currentData()))

    def _set_binning(self, rebin: int) -> None:
        """Re-serve the stack mean-pooled by `rebin`, keeping labels and model.

        Labels live in raw px so nothing about them changes; the chain gets
        the new factor and everything in loaded px (marker sizes, the recon
        row) is rescaled to stay on the same physical spot.
        """
        if self._stack.info() is None:
            return
        old = self._chain.rebin
        if int(rebin) == old:
            return
        stack = self._stack
        info = run_source_job(
            self, f"Binning projections by {rebin}…",
            lambda progress, cancelled: stack.set_binning(
                int(rebin), progress=progress, cancelled=cancelled),
            error_title="Could not rebin")
        if info is None or info.rebin == old:
            self._adopt_stack(self._chain, self._source)   # combo back in step
            return
        ratio = old / info.rebin
        self._feature_sizes = {fid: max(size * ratio, 0.5)
                               for fid, size in self._feature_sizes.items()}
        row_raw = float(self._chain.to_parent(0.0, self.slice_row.value())[1])
        self._adopt_stack(self._chain.with_rebin(info.rebin), self._source)
        row = int(round(float(self._chain.from_parent(0.0, row_raw)[1])))
        self.slice_row.setValue(max(0, min(row, self.slice_row.maximum())))
        self._sync_model_dims_only()
        self._refresh_feature_table()
        self._refresh_plots()
        self._recon_maybe()

    def _sync_prefetch(self) -> None:
        """A prefetcher for a remote stack, and only while the stack is remote.

        Reading ahead is worth a background thread when a frame is a second
        away and pointless when the stack is in this process, so a local
        source gets none. `_show_data` can swap a remote source for a local
        one under us, hence the identity check rather than a flag.
        """
        if self._prefetch is not None and (self._prefetch.source is not self._stack
                                           or not self._stack.is_remote):
            self._prefetch.stop()
            self._prefetch = None
        if self._prefetch is None and self._stack.is_remote:
            self._prefetch = ViewPrefetcher(self._stack)
            self._prefetch.want(self._view, self.advance_box.value())

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch = None
        for worker in (self._recon_worker, self._autotrack_worker):
            if isinstance(worker, QThread) and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(2000)
        self._stack.close()
        super().closeEvent(event)

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
            "follow_prediction": self.follow_box.isChecked(),
            "auto_reject": self.auto_reject.isChecked(),
            "auto_reject_k": self.auto_reject_k.value(),
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

    def _load_session(self, path: str | None = None) -> None:
        """Load a session; without `path`, ask for one."""
        if not path:
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
        info = self._stack.info()
        if (stack_path and self._stack.is_remote and info is not None
                and info.path == str(stack_path)):
            # already open on the server: keep it, restore the provenance
            chain = CoordinateChain(
                binning=int(source.get("binning", info.binning)),
                crop=tuple(int(x) for x in source.get("crop", info.crop)),
                view_origin=info.view_origin, rebin=info.rebin)
            self._adopt_stack(chain, {**source,
                                      "endpoint": self._stack.describe()})
        elif stack_path and (self._stack.is_remote
                             or Path(str(stack_path)).exists()):
            result = open_stack_interactive(self, self._stack,
                                            str(stack_path))
            if result is None:
                return
            _info, chain, new_source = result
            self._adopt_stack(chain, new_source)
        elif info is None:
            where = source.get("endpoint")
            hint = (f" It was labelled against {where}; start the app with "
                    f"--connect {where} to reach it." if where else
                    " Load the stack first, then the session.")
            QMessageBox.warning(
                self, "Stack missing",
                f"The session's stack ({stack_path}) is not readable "
                f"here.{hint}")
            return
        # the grid the session was labelled on, so its marker sizes and
        # recon row (loaded px) mean what they meant
        self._set_binning(int(source.get("rebin", 1)))
        self._labels = state["labels"]
        self._model = state["model"]
        self._mask = state["mask"]
        ui = state["ui"]
        self._pins = set(ui.get("pins", []))
        self._feature_sizes = {int(k): float(size) for k, size in
                               ui.get("feature_sizes", {}).items()}
        self.ghost_box.setChecked(bool(ui.get("ghosts", False)))
        self.follow_box.setChecked(bool(ui.get("follow_prediction", False)))
        self.auto_reject.setChecked(bool(ui.get("auto_reject", True)))
        self.auto_reject_k.setValue(float(ui.get("auto_reject_k", 3.0)))
        self.auto_min_corr.setValue(float(ui.get("auto_min_corr", 0.20)))
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
            det_h = self._stack.shape[1] * chain.scale + chain.crop[0]
            det_w = self._stack.shape[2] * chain.scale + chain.crop[2]
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
        if self._fit is None or self._stack.info() is None:
            QMessageBox.information(self, "No fit", "Fit a model first.")
            return
        if self._stack.is_remote:
            # the warp runs where the pixels are and the result stays there:
            # shipping the aligned stack back is the whole stack over the wire
            path, ok = QInputDialog.getText(
                self, "Export aligned stack",
                f"Output path on {self._stack.describe()}:",
                text="aligned.h5")
            if not ok or not path:
                return
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export aligned stack", "aligned.h5", "HDF5 (*.h5)")
            if not path:
                return
        transforms = aligned_view_transforms(self._model, self._chain)
        metadata = aligned_metadata(self._stack.info().metadata, self._model,
                                    self._chain)
        req = AlignedExportRequest(
            dx=np.array([t.dx for t in transforms], float),
            dy=np.array([t.dy for t in transforms], float),
            rot_deg=np.array([t.rotation for t in transforms], float),
            metadata=metadata)
        stack = self._stack
        written = run_source_job(
            self, "Warping projections…",
            lambda progress, cancelled: stack.export_aligned(
                req, path, progress=progress, cancelled=cancelled),
            error_title="Export stopped")
        if written is None:
            return
        QMessageBox.information(
            self, "Exported",
            f"Wrote {written} on {stack.describe()}\nfixed center on this "
            f"grid: {metadata['center_loaded_px']:.2f} px")


def _fmt(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="projection stack (.h5)")
    parser.add_argument("--session", help="session file (.h5) to load; its "
                        "stack is loaded from the session's own provenance")
    parser.add_argument("--connect", metavar="ADDRESS",
                        help="use a stack served by tktomo-track-server at "
                        "this ZeroMQ address (e.g. tcp://127.0.0.1:5611 "
                        "through an SSH tunnel); PATH then names a file on "
                        "that machine")
    parser.add_argument("--exact-frames", action="store_true",
                        help="with --connect, send frames as raw float32 "
                        "instead of packing them to display precision. "
                        "Roughly twice the wait per view, and only the "
                        "displayed pixels differ (everything that computes "
                        "runs on the server either way)")
    args = parser.parse_args()

    def build():
        source = None
        if args.connect:
            from tktomo.tracking.remote import RemoteStackSource  # noqa: PLC0415
            source = RemoteStackSource(args.connect,
                                       quantise=not args.exact_frames)
        win = TrackModelWindow(args.path, source=source)
        if args.session:
            win._load_session(args.session)
        return win

    return run_app(build)


if __name__ == "__main__":
    raise SystemExit(main())

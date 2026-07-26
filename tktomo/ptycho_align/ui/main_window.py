"""The ptycho-align main window.

A thin shell over an :class:`~tktomo.ptycho_align.session.protocol.AlignmentSession`.
The window owns no engine and runs no compute: it asks the session for work, draws what
comes back, and never touches tomopy. Whether the session is driving an engine in this
process or a server on a cluster node is a connection detail it deliberately cannot see.

Two rules follow from that and run through the whole file.

**Nothing here blocks on compute.** Heavy verbs return a job handle and report through
events; where the window legitimately has to wait -- loading, preprocessing, COM -- it
does so behind a modal progress dialog via
:meth:`~tktomo.ptycho_align.ui.session_bridge.SessionBridge.run_job`, which is
cancellable and keeps the window painting.

**The session is the single source of truth.** The window keeps one cached
:class:`~tktomo.ptycho_align.session.types.SessionSummary` and draws from it, rather than
reaching into engine state. Pixels are fetched separately and only when displayed.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tktomo.ptycho_align.core import AlignConfig, Crop, center_is_plausible
from tktomo.ptycho_align.core.estimates import format_bytes as _gb
from tktomo.ptycho_align.core.estimates import format_duration as _duration
from tktomo.ptycho_align.core.preprocess import PreprocessOptions
from tktomo.ptycho_align.session import (
    STACK_ALIGNED,
    STACK_REPROJECTION,
    Busy,
    LocalSession,
    SessionSummary,
)
from tktomo.ptycho_align.session.protocol import JobFailed
from tktomo.ptycho_align.ui.session_bridge import JobCancelled, SessionBridge
from tktomo.ptycho_align.ui.panels import (
    ActionBar,
    AlignPanel,
    ComPanel,
    DataPanel,
    Hdf5BrowserDialog,
    PreprocessPanel,
    ProjectionView,
    ResourceView,
    ShiftView,
    SinogramView,
    TomogramView,
)
from tktomo.ptycho_align.ui.panels.base import MODE_RAW
from tktomo.ptycho_align.ui.panels.crop import CropDialog
from tktomo.ptycho_align.ui.panels.hdf5_browser import preview_text
from tktomo.ptycho_align.ui.planes import PlaneSource

logger = logging.getLogger("tktomo.ptycho_align")

_HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx5", ".hdf")

# Warn once a run's estimated peak is within this fraction of available RAM. Not 1.0:
# the estimate counts our own arrays, not tomopy's transient shared-memory buffers, and
# overshooting means the OOM killer takes the window down with no traceback at all.
_MEMORY_HEADROOM = 0.8

# Beyond this, a run is an "leave it overnight" proposition rather than something to sit
# and watch, and the user should be told before pressing Run, not after.
_SLOW_RUN_SECONDS = 30 * 60


def _is_hdf5(path: str) -> bool:
    return Path(path).suffix.lower() in _HDF5_SUFFIXES


class _LogDock(QObject, logging.Handler):
    """Mirror the engine's log into a dock, so a run is reconstructable after the fact.

    The engine logs one line per iteration from **inside** ``step()``, which runs on the
    worker thread. Qt widgets may only be touched from the GUI thread, so writing
    straight into the QPlainTextEdit from here corrupts Qt's internal state and
    segfaults the process later, inside an unrelated paint event -- nowhere near the
    actual culprit. Hop threads via a queued signal instead: ``message`` is emitted on
    whichever thread logged, and Qt delivers it to the widget on the GUI thread.
    """

    message = Signal(str)

    def __init__(self, widget: QPlainTextEdit) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self, level=logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        # AutoConnection: same-thread logs go direct, worker-thread logs get queued.
        self.message.connect(widget.appendPlainText)

    def emit(self, record: logging.LogRecord) -> None:
        self.message.emit(self.format(record))


def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap a panel so it can be scrolled instead of forcing the window taller."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    return area


class PtychoAlignWindow(QMainWindow):
    def __init__(self, path: str | None = None, session=None) -> None:
        super().__init__()
        self.setWindowTitle("ptycho-align -- reprojection alignment")
        self._fit_to_screen(1500, 950)

        # Defaults to an in-process session. Passing one in is how a cluster connection
        # arrives: nothing below this line knows or cares which it is.
        self.session = session if session is not None else LocalSession()
        self.bridge = SessionBridge(self.session, self)
        self._summary: SessionSummary = self.session.summary()
        # Every pixel the viewers draw comes through here, one displayed plane at a time.
        self._planes = PlaneSource(self.session)
        self._planes.update(self._summary)
        # The handle of the run in flight, so close and the tests can wait on it.
        self._run_handle = None
        # Measured seconds per unit of work, per algorithm (see `iteration_cost_units`).
        # Kept across engine rebuilds on purpose: it is a property of this machine and
        # this algorithm, not of the current grid, so one timed iteration at bin 2 can
        # predict what bin 1 would cost -- which is exactly when the user needs telling.
        self._seconds_per_unit: dict[str, float] = {}
        # (epoch, iteration) pairs already folded into the calibration, so re-reading the
        # history does not count the same iteration twice. Keyed on the session's epoch
        # rather than on engine identity: a rebuild restarts the numbering at 1, and with
        # a session that survives the rebuild those would collide with the keys already
        # recorded -- freezing the estimate at the old grid's cost exactly when the grid
        # got more expensive.
        self._calibrated: set[tuple[int, int]] = set()

        self.projection_view = ProjectionView()
        self.sinogram_view = SinogramView()
        self.tomogram_view = TomogramView()
        self.shift_view = ShiftView()

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.projection_view)
        top.addWidget(self.sinogram_view)
        top.setSizes([600, 600])
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.tomogram_view)
        bottom.addWidget(self.shift_view)
        bottom.setSizes([600, 600])
        grid = QSplitter(Qt.Vertical)
        grid.addWidget(top)
        grid.addWidget(bottom)
        # Without this the viewers collapse to their minimum hint and the top row ends
        # up a sliver; the four panels should start out equally sized.
        grid.setSizes([500, 500])

        self.action_bar = ActionBar()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(grid, 1)
        layout.addWidget(self.action_bar)
        self.setCentralWidget(central)

        self.data_panel = DataPanel()
        self.preprocess_panel = PreprocessPanel()
        self.com_panel = ComPanel()
        self.align_panel = AlignPanel()
        self.resource_view = ResourceView()
        self._add_left_docks()

        self.log_widget = QPlainTextEdit(readOnly=True)
        self.log_widget.setMaximumBlockCount(5000)
        self.log_widget.setMinimumHeight(60)
        self.log_widget.setMaximumHeight(150)
        log_dock = QDockWidget("Log", self)
        log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        # The engine's logger ("...core.engine") is a child of this one and propagates,
        # so a single handler catches both. Kept on self so closeEvent can detach it --
        # the logger is module-global, and a handler left pointing at a destroyed widget
        # would fire again the next time a window is opened.
        self._log_handler = _LogDock(self.log_widget)
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

        self._connect()
        self._build_menus()

        if path:
            self.load_path(path)

    # -- construction ------------------------------------------------------------------

    def _fit_to_screen(self, width: int, height: int) -> None:
        """Open at the requested size, but never larger than the screen actually is.

        A window taller than the display puts the action bar -- Step / Run / Stop --
        off the bottom edge, where it cannot be reached or dragged back.
        """
        screen = QGuiApplication.primaryScreen()
        if screen is None:  # offscreen platform in the tests
            self.resize(width, height)
            return

        available = screen.availableGeometry()
        self.resize(min(width, available.width()), min(height, available.height()))
        self.move(available.topLeft())

    def _add_left_docks(self) -> None:
        docks = []
        # Short titles: these become tab labels in a narrow dock, and anything longer
        # gets elided to "Alignme...".
        for title, widget in (
            ("Data", self.data_panel),
            ("Prep", self.preprocess_panel),
            ("COM", self.com_panel),
            ("Align", self.align_panel),
            ("Res", self.resource_view),
        ):
            dock = QDockWidget(title, self)
            # Each panel goes in a scroll area. Stacked, their natural minimum heights
            # add up to ~1000 px, which -- with the log dock and the action bar -- pushed
            # the window's minimum height past a 1080p screen, so the action bar could
            # not be reached at all. Scrolling lets a dock shrink below its content.
            dock.setWidget(_scrollable(widget))
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            docks.append(dock)

        # Tabbed rather than stacked: four panels sharing one column leaves each of them
        # a few scrollbar-ridden centimetres. Tabs give whichever one you are using the
        # full height. They are still ordinary docks, so any of them can be torn off and
        # placed side by side on a bigger screen.
        for previous, dock in zip(docks, docks[1:]):
            self.tabifyDockWidget(previous, dock)
        docks[0].raise_()
        self._left_docks = docks

    def _connect(self) -> None:
        self.data_panel.load_requested.connect(self._choose_dataset)
        self.data_panel.browse_requested.connect(self._browse_hdf5)
        self.data_panel.crop_requested.connect(self._adjust_crop)
        self.data_panel.session_load_requested.connect(self._open_session)

        self.preprocess_panel.apply_requested.connect(self._apply_preprocessing)
        self.preprocess_panel.reset_requested.connect(self._reset_preprocessing)
        self.projection_view.roi_changed.connect(self.preprocess_panel.show_roi)

        self.com_panel.prealign_requested.connect(self._run_com)
        self.com_panel.center_requested.connect(self._estimate_center)
        self.com_panel.center_overridden.connect(self._set_center)

        self.align_panel.config_changed.connect(self._config_changed)
        self.align_panel.bin_factor_changed.connect(self._bin_changed)

        self.action_bar.step_requested.connect(lambda: self._start_run(1))
        self.action_bar.run_requested.connect(self._start_run)
        # More iterations means more retained volumes, so the estimate moves with it.
        self.action_bar.run_spin.valueChanged.connect(lambda _: self._show_estimates())
        self.action_bar.stop_requested.connect(self._stop_run)
        self.action_bar.revert_requested.connect(self._revert)

        # The session narrates itself. These four carry the signatures the retired
        # AlignmentWorker had, so the slots below are unchanged.
        self.bridge.iteration_finished.connect(self._iteration_finished)
        self.bridge.progress.connect(self._progress)
        self.bridge.failed.connect(self._run_failed)
        self.bridge.run_finished.connect(self._run_finished)
        self.bridge.summary_changed.connect(self._summary_changed)
        self.bridge.telemetry.connect(self.resource_view.show_sample)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Load projections...", self._choose_dataset)
        file_menu.addAction("Browse HDF5 datasets...", self._browse_hdf5)
        file_menu.addAction("Adjust crop / component...", self._adjust_crop)
        file_menu.addAction("Open session...", self._open_session)
        file_menu.addAction("Save session...", self._save_session)
        file_menu.addSeparator()
        file_menu.addAction("Export aligned projections...", self._export_projections)
        file_menu.addAction("Export volume...", self._export_volume)
        file_menu.addAction("Export shifts (CSV)...", self._export_shifts)
        file_menu.addAction("Export convergence (CSV)...", self._export_convergence)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        align_menu = self.menuBar().addMenu("&Align")
        align_menu.addAction("Step once", lambda: self._start_run(1))
        align_menu.addAction("Run N iterations", lambda: self._start_run(self.action_bar.run_spin.value()))
        align_menu.addAction("Stop", self._stop_run)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About", self._about)

    # -- loading -----------------------------------------------------------------------

    def _choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load projections",
            "",
            "Projections (*.h5 *.hdf5 *.nxs *.npy *.npz);;All files (*)",
        )
        if path:
            self.load_path(path)

    def _browse_hdf5(self) -> None:
        """Pick the datasets by hand, for a file that follows no known layout."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Browse HDF5 datasets", "", "HDF5 (*.h5 *.hdf5 *.nxs *.hdf);;All files (*)"
        )
        if path:
            self._load_via_browser(path)

    def _load_via_browser(self, path: str) -> bool:
        """Open the dataset browser on ``path``; return whether a dataset was loaded."""
        try:
            # Listed once and shared: the dialog and the log line used to walk the file
            # separately, and this is the call that becomes an RPC once the file is on a
            # cluster.
            entries = self.session.list_hdf5(path)
            dialog = Hdf5BrowserDialog(path, self, entries=entries)
        except Exception as exc:
            self._error("Could not read the HDF5 file", str(exc))
            return False

        logger.info("Browsing %s: %s", path, preview_text(entries))
        if dialog.exec() != QDialog.Accepted:
            return False
        return self.load_path(path, **dialog.selection())

    def _adjust_crop(self) -> None:
        """Re-read the stack over a different detector region, or a different component.

        The crop is an HDF5 hyperslab, so widening it means going back to the file --
        the discarded region was never in memory to begin with. That resets the
        alignment, hence the confirmation.
        """
        source = self._source()
        dataset = self._summary.dataset
        if dataset is None or source is None or dataset.crop is None:
            self._error("No data", "Load a projection stack from an HDF5 file first.")
            return
        if self._busy():
            return

        current = Crop(*dataset.crop)
        dialog = CropDialog(
            tuple(dataset.full_shape),
            current,
            dataset.component or "phase",
            complex_source=self._source_is_complex(source),
            roi=self._roi_as_file_crop(),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        selection = dialog.selection()
        if selection["crop"] == current and selection["component"] == dataset.component:
            return  # nothing changed

        if self._summary.iteration > 0:
            answer = QMessageBox.question(
                self,
                "Re-read the data?",
                f"Re-reading discards the {self._summary.iteration} completed iteration(s) "
                "and their shifts.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.load_path(source["path"], **{**source["kwargs"], **selection})

    def _source(self) -> dict | None:
        """How the current stack was read, so a different crop can go back to the file."""
        return self._summary.metadata.get("source")

    def _source_is_complex(self, source: dict) -> bool:
        """Is the source dataset complex? (Only then is the component combo meaningful.)"""
        entries = {e.path: e for e in self.session.list_hdf5(source["path"])}
        entry = entries.get(source["kwargs"].get("data_path", ""))
        return bool(entry and entry.is_complex)

    def _roi_as_file_crop(self) -> Crop | None:
        """The projection view's ROI, re-expressed in the *file's* row/column numbers.

        Only meaningful while the displayed stack still has the shape it was loaded
        with: preprocessing pads, and the bin factor shrinks, so after either of those
        the ROI's numbers no longer index the file.
        """
        roi = self.projection_view.roi_bounds()
        dataset = self._summary.dataset
        if roi is None or dataset is None or dataset.crop is None or self._source() is None:
            return None
        displayed = self._planes.shape(MODE_RAW)
        raw_shape = self._summary.raw_shape
        if displayed is None or raw_shape is None or tuple(displayed) != tuple(raw_shape):
            return None
        return Crop(*roi).shifted_by(dataset.crop[0], dataset.crop[2])

    def load_path(self, path: str, **load_kwargs) -> bool:
        """Load a dataset. Extra keywords go straight to the session's loader.

        With no keywords this probes the conventional layouts; when that finds nothing
        in an HDF5 file, the dataset browser opens rather than the user being told
        "no projection dataset found" and left there.
        """
        try:
            result = self.bridge.run_job(
                self.session.open_dataset(path, load_kwargs),
                title="Loading",
                label="Reading projections...",
                parent=self,
            )
        except JobCancelled:
            self.action_bar.show_status("Loading cancelled.")
            return False
        except JobFailed as exc:
            if exc.exc_type == "KeyError" and not load_kwargs and _is_hdf5(path):
                logger.info("No conventional layout in %s; opening the browser", path)
                return self._load_via_browser(path)
            self._error("Could not load the dataset", str(exc))
            return False
        except Busy:
            self._refused()
            return False

        problems = result.get("problems", ())
        if problems:
            # These are the failures that would otherwise surface as a traceback from
            # deep inside tomopy, so say them plainly and let the user decide.
            answer = QMessageBox.warning(
                self,
                "Problems with this dataset",
                "\n\n".join(problems) + "\n\nLoad it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return False

        self._after_engine_rebuilt()
        self.action_bar.show_status("Loaded. Apply preprocessing, then run COM pre-alignment.")
        return True

    # -- preprocessing -----------------------------------------------------------------

    def _apply_preprocessing(self, options: PreprocessOptions) -> None:
        if not self._summary.has_engine:
            self._error("No data", "Load a projection stack first.")
            return

        try:
            report = self.bridge.run_job(
                self.session.apply_preprocessing(options, self.projection_view.roi_bounds()),
                title="Preprocessing",
                label="Preprocessing projections...",
                parent=self,
            )
        except JobCancelled:
            return
        except Busy:
            self._refused()
            return
        except JobFailed as exc:
            self._error("Preprocessing failed", str(exc))
            return

        if not report.mass_is_positive:
            QMessageBox.warning(
                self,
                "Negative mass",
                f"The projection integral is {report.mass_total:.4g}, i.e. negative. The "
                "reconstruction and the centre-of-mass both assume positive mass -- "
                "switch on 'Invert' and apply again.",
            )

        self._after_engine_rebuilt()
        self.action_bar.show_status("Preprocessing applied.")

    def _reset_preprocessing(self) -> None:
        if not self._summary.has_engine:
            return
        try:
            self.bridge.run_job(
                self.session.reset_preprocessing(),
                title="Preprocessing",
                label="Restoring the raw stack...",
                parent=self,
            )
        except (JobCancelled, Busy):
            return
        except JobFailed as exc:
            self._error("Could not reset", str(exc))
            return
        self._after_engine_rebuilt()
        self.action_bar.show_status("Reset to raw.")

    # -- COM ---------------------------------------------------------------------------

    def _refused(self) -> None:
        """Explain why an exclusive action was refused."""
        QMessageBox.information(
            self,
            "Alignment is running",
            "Stop the current run first. TomoPy cannot safely run two "
            "reconstructions at once.",
        )

    def _busy(self) -> bool:
        """True (with a nag) if the cached summary already says a run is in flight.

        Only a courtesy check, so a dialog does not open just to be thrown away. The
        authoritative answer is the session's :class:`Busy`, which every caller must
        still handle -- the cache can be a moment out of date, and the refusal is what
        actually keeps two reconstructions from running at once.
        """
        if self._summary.running:
            self._refused()
            return True
        return False

    def _run_com(self, reference: str) -> None:
        """Ask for COM pre-alignment and show what came back.

        The shifts, centre, history and volume that follow from the result are the
        session's business, not the window's -- this used to reach into ``engine.state``
        and rewrite six fields by hand.
        """
        if not self._summary.has_engine:
            self._error("No data", "Load a projection stack first.")
            return
        if self._busy():
            return

        try:
            result = self.bridge.run_job(
                self.session.run_com(reference),
                title="COM pre-alignment",
                label="Computing centroids...",
                parent=self,
            )
        except JobCancelled:
            return
        except Busy:
            self._refused()
            return
        except JobFailed as exc:
            self._error("COM pre-alignment failed", str(exc))
            return

        self.com_panel.show_result(result.center, result.fit_residual, result.amplitude)
        self.shift_view.update_com(
            self._summary.angles, result.com_u, result.fitted_u, result.fit_residual
        )
        self.sinogram_view.set_com(result.com_u, result.fitted_u, result.center)
        self._refresh_views()
        self.action_bar.show_status(
            f"COM pre-alignment done (centre {result.center:.2f} px). Ready to step."
        )

    def _estimate_center(self, method: str) -> None:
        if not self._summary.has_engine:
            return
        if self._busy():
            return
        try:
            center = self.bridge.run_job(
                self.session.estimate_center(method),
                title="Centre estimation",
                label=f"Finding the rotation centre ({method})...",
                parent=self,
            )
        except (JobCancelled, Busy):
            return
        except JobFailed as exc:
            self._error("Centre estimation failed", str(exc))
            return

        logger.info("Centre estimate (%s): %.3f px", method, center)

        # A failed centre-finder returns a number rather than an error, and a bad centre
        # quietly ruins a long run. find_center_vo in particular is unreliable on padded
        # phase data. Check it against the COM fit before letting it anywhere near the
        # engine.
        width = self._summary.original_shape[2]
        com = self._summary.com
        reference = com.center if com else None
        ok, reason = center_is_plausible(center, width, reference)
        if not ok:
            logger.warning("Rejected centre estimate (%s): %s", method, reason)
            answer = QMessageBox.warning(
                self,
                "Implausible centre estimate",
                f"The '{method}' estimator returned <b>{center:.3f} px</b>, but {reason}."
                "<br><br>TomoPy's Vo estimator is tuned for attenuation sinograms and is "
                "unreliable on zero-padded phase projections. The COM fit and the "
                "phase-correlation estimator are usually more trustworthy here."
                "<br><br>Use it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.com_panel.set_center(center)

    def _set_center(self, center: float) -> None:
        if not self._summary.has_engine:
            return
        self.session.set_center(center)
        self._summary = self.session.summary()
        com = self._summary.com
        self.sinogram_view.set_com(
            com.com_u if com else None, com.fitted_u if com else None, center
        )
        self.action_bar.show_status(f"Centre set to {center:.3f} px.")

    # -- engine ------------------------------------------------------------------------

    def _after_engine_rebuilt(self) -> None:
        """Re-read everything the window caches about a grid that has just changed."""
        self._summary = self.session.summary()
        dataset = self._summary.dataset
        if dataset is not None:
            self.data_panel.show_dataset(dataset)
        self.shift_view.clear()
        self._show_estimates()
        self._refresh_views()

    def _bin_changed(self, factor: int) -> None:
        if not self._summary.has_engine or factor == self._summary.bin_factor:
            return

        # Rebinning rebuilds the engine, and a run in flight is working on the old grid.
        # Rebuilding underneath it does not stop it: the iteration completes into an
        # engine nothing reads any more and is silently discarded -- minutes of compute,
        # no visible result, no error. The session refuses; put the spin box back.
        try:
            self.bridge.run_job(
                self.session.set_bin_factor(factor),
                title="Binning",
                label=f"Rebinning to {factor}x...",
                parent=self,
            )
        except Busy:
            self._refused()
            self.align_panel.set_bin_factor(self._summary.bin_factor)
            return
        except JobCancelled:
            self.align_panel.set_bin_factor(self._summary.bin_factor)
            return
        except JobFailed as exc:
            self._error("Could not change the bin factor", str(exc))
            self.align_panel.set_bin_factor(self._summary.bin_factor)
            return

        self._after_engine_rebuilt()
        # The COM result is in pixels too and outlives the rebuild -- it is the reference
        # new centre estimates are judged against -- so the session rescales it and the
        # panel re-reads it here.
        com = self._summary.com
        if com is not None:
            self.com_panel.show_result(com.center, com.fit_residual, com.amplitude)
        self.action_bar.show_status(
            f"Binning {factor}x. Shifts rescaled; history cleared (the volume grid changed)."
        )

    def _run_footprint(self, n_iterations: int) -> tuple[int, str]:
        """Peak RAM a run of ``n_iterations`` would need, and a one-line explanation.

        Priced by the session, because on a cluster the numbers that matter -- what is
        available, and what the cgroup will actually allow -- belong to the machine doing
        the work, not to the one drawing the window.
        """
        preflight = self.session.run_preflight(n_iterations)
        return preflight.footprint_bytes, preflight.footprint_text

    def _cost_units(self) -> float:
        return self.session.cost_units()

    def _calibrate(self) -> None:
        """Learn seconds-per-unit from the iterations that have actually run.

        Driven off the summary's history rather than the per-iteration signal: the signal
        is queued to the GUI thread, so anything that blocks the main thread (a modal
        dialog, a test waiting on a job) never delivers it, and the calibration would
        silently never happen. The history is the authoritative record either way.
        """
        if not self._summary.has_engine:
            return
        units = self._cost_units()
        if units <= 0:
            return
        algorithm = self._summary.config.get("recon_algorithm", "")
        for result in self._summary.history:
            key = (self._summary.epoch, result.iteration)
            if key in self._calibrated or result.wallclock_s <= 0:
                continue
            self._calibrated.add(key)
            measured = result.wallclock_s / units
            previous = self._seconds_per_unit.get(algorithm)
            # Average with what we knew: a single iteration can be an outlier (a cold
            # cache, the machine busy elsewhere), and a wild estimate is worse than none.
            self._seconds_per_unit[algorithm] = (
                measured if previous is None else 0.5 * (previous + measured)
            )

    def _run_duration(self, n_iterations: int) -> float | None:
        """Predicted wallclock for ``n_iterations``, or None if nothing is calibrated."""
        rate = self._seconds_per_unit.get(self._summary.config.get("recon_algorithm", ""))
        if rate is None:
            return None
        return rate * self._cost_units() * n_iterations

    def _show_estimates(self) -> None:
        """Refresh the "this is what Run will cost you" readouts."""
        if not self._summary.has_engine:
            return
        n = self.action_bar.run_spin.value()

        preflight = self.session.run_preflight(n)
        total, text = preflight.footprint_bytes, preflight.footprint_text
        available = preflight.ram_available
        tight = available is not None and total > _MEMORY_HEADROOM * available
        if available is not None:
            text += f" {_gb(available)} available."
        self.align_panel.show_memory_estimate(text, warn=tight)

        seconds = self._run_duration(n)
        if seconds is None:
            self.align_panel.show_time_estimate(
                "Unknown until one iteration has been timed.", warn=False
            )
            return
        per_iteration = seconds / max(n, 1)
        self.align_panel.show_time_estimate(
            f"~{_duration(per_iteration)} per iteration, {_duration(seconds)} for {n} "
            f"(measured on this machine; scales as angles x rows x width^2).",
            warn=seconds > _SLOW_RUN_SECONDS,
        )

    def _config_changed(self, config: AlignConfig) -> None:
        if not self._summary.has_engine:
            return
        config.pad = (0, 0)
        self.session.set_config(config.to_dict())
        self._summary = self.session.summary()
        # Algorithm and inner iterations both move the cost, so the estimates follow them.
        self._show_estimates()

    # -- running -----------------------------------------------------------------------

    def _start_run(self, n: int) -> None:
        if not self._summary.has_engine:
            self._error("No data", "Load a projection stack first.")
            return
        if self._summary.running:
            return

        # One round trip prices the whole run: the negative-data scan is a full pass over
        # the stack, and the memory numbers belong to the machine doing the work.
        preflight = self.session.run_preflight(n)

        # mlem/osem on phase data (which is ~20% negative) diverges explosively rather
        # than failing, so say so before burning a run on it.
        reason = preflight.negative_reason
        if reason:
            answer = QMessageBox.warning(
                self,
                "This algorithm will diverge",
                f"{reason}<br><br>Run it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        # Running out of memory does not raise -- the kernel kills the process, and the
        # window simply vanishes with no traceback and nothing in the log. Of every way
        # this app can fail, that is the one the user has no chance of diagnosing, so
        # spend a dialog on it.
        total, explanation = preflight.footprint_bytes, preflight.footprint_text
        available = preflight.ram_available
        if available is not None and total > _MEMORY_HEADROOM * available:
            logger.warning("Run needs %s but only %s is available", _gb(total), _gb(available))
            answer = QMessageBox.warning(
                self,
                "This run may exhaust memory",
                f"{explanation}<br><br>Only <b>{_gb(available)}</b> is available, so the "
                "process is likely to be killed by the kernel -- the window will vanish "
                "with no error message.<br><br>Raise the bin factor (2x cuts the volume "
                "eightfold), reduce the padding, or run fewer iterations."
                "<br><br>Run it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        # A slow run is not an error, but "Run 10" turning out to be a nine-hour job is
        # worth one click to confirm. Only asked once the cost is actually known.
        seconds = self._run_duration(n)
        if seconds is not None and seconds > _SLOW_RUN_SECONDS:
            answer = QMessageBox.question(
                self,
                "This will take a while",
                f"{n} iteration(s) at this setting is about <b>{_duration(seconds)}</b> "
                f"({_duration(seconds / max(n, 1))} each), based on what this machine has "
                "measured so far.<br><br>Raising the bin factor is the strongest lever -- "
                "the cost goes as the square of the detector width, so bin 2 is roughly "
                "eight times cheaper.<br><br>Start the run?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            self._run_handle = self.session.start_run(n)
        except Busy:
            return
        # Re-read rather than waiting for the pushed summary: Stop arriving before the
        # first event would otherwise consult a cache that still says nothing is running.
        self._summary = self.session.summary()
        self.action_bar.set_running(True)

    def _stop_run(self) -> None:
        # Unconditional: cancelling a run that has already finished is harmless, whereas
        # gating on a cached summary loses the cancel in the window before the first
        # event lands -- which is exactly when a user hits Stop by mistake.
        self.session.cancel_run()
        self.action_bar.show_status("Stopping at the next row chunk...")

    def _summary_changed(self, summary: SessionSummary) -> None:
        """The session pushes a fresh summary with every iteration and state change."""
        self._summary = summary

    def _progress(self, fraction: float, message: str) -> None:
        self.action_bar.progress.setValue(int(fraction * 100))
        self.action_bar.show_status(message)

    def _iteration_finished(self, result) -> None:
        self._calibrate()
        # The worker stops itself on a runaway (it cannot wait for us to call cancel --
        # that is a race it would lose under load), so there is nothing to do here but
        # say so. The dialog comes when the run finishes.
        if result.runaway:
            self.action_bar.show_status(
                f"Stopped: runaway shifts at iteration {result.iteration}."
            )
        if self.action_bar.live_check.isChecked():
            self._refresh_views()

    def _run_failed(self, message: str) -> None:
        logger.error("Run failed:\n%s", message)
        self._error("The alignment run failed", message.strip().splitlines()[-1])

    def _run_finished(self) -> None:
        self.action_bar.set_running(False)
        # Re-read rather than trusting the pushed summary: the per-iteration signal is
        # queued, so a blocked main thread may never have received the last one.
        self._summary = self.session.summary()
        self._calibrate()
        self._show_estimates()
        self._refresh_views()
        history = self._summary.history
        if history:
            last = history[-1]
            status = (
                f"Iteration {last.iteration}  |  shift RMS {last.error:.4f} px  |  "
                f"residual {last.residual:.4f}  |  {last.wallclock_s:.1f} s"
            )
            if last.runaway:
                status = "RUNAWAY SHIFTS -- " + status
            elif any(r.diverging for r in history):
                status = "DIVERGING -- " + status
            self.action_bar.show_status(status)
            self.action_bar.revert_spin.setMaximum(last.iteration)

            if last.runaway:
                good = next((r.iteration for r in reversed(history) if not r.runaway), 0)
                QMessageBox.warning(
                    self,
                    "Runaway shifts -- the run was stopped",
                    f"{last.runaway}<br><br>The run has been stopped. The usual cause is a "
                    "reconstruction too poor to reproject anything the registration can "
                    "lock onto: try <b>sirt</b> rather than art, raise the inner "
                    "iterations, and set <b>Max shift per iter</b> to clip outliers."
                    f"<br><br>Revert to iteration {good} to discard the bad shifts -- they "
                    "are cumulative, so they do not wash out on their own.",
                )
            elif last.diverging:
                QMessageBox.warning(
                    self,
                    "The alignment is diverging",
                    f"The residual has climbed to {last.residual:.4g}, well above the "
                    "best this run achieved. It is not converging.<br><br>"
                    "Usual causes: a residual phase ramp (enable 'Remove phase ramp'), "
                    "an emission algorithm (mlem/osem) on data with negative values, a "
                    "bad rotation centre, or too few inner reconstruction iterations."
                    "<br><br>Revert to a good iteration before continuing.",
                )

    def _revert(self, iteration: int) -> None:
        if not self._summary.has_engine:
            return
        try:
            self.bridge.run_job(
                self.session.revert(iteration),
                title="Revert",
                label=f"Reverting to iteration {iteration}...",
                parent=self,
                cancellable=False,
            )
        except Busy:
            self._refused()
            return
        except JobCancelled:
            return
        except JobFailed as exc:
            self._error("Cannot revert", str(exc))
            return
        self._summary = self.session.summary()
        self._refresh_views()
        self.action_bar.show_status(f"Reverted to iteration {iteration}.")

    # -- views -------------------------------------------------------------------------

    def _refresh_views(self) -> None:
        """Point every viewer at the current pixels. Each pulls the one plane it draws.

        This must NOT compute. Recomputing an aligned stack or a reprojection is a job
        for the session's compute thread -- doing it here would both freeze the window
        and, worse, call tomopy from a second thread, which segfaults the process. When
        the cached stacks are missing and nothing is running, ask the session to
        materialise them and come back.
        """
        summary = self._summary
        if not summary.has_engine:
            return

        available = {spec.key for spec in summary.stacks if spec.available}
        if STACK_ALIGNED not in available and not summary.running:
            try:
                self.bridge.run_job(
                    self.session.materialize([STACK_ALIGNED, STACK_REPROJECTION]),
                    title="Preparing views",
                    label="Applying shifts...",
                    parent=self,
                )
                self._summary = summary = self.session.summary()
            except (JobCancelled, Busy):
                return
            except JobFailed as exc:  # a missing backend must not break the viewers
                logger.warning("Could not prepare the views: %s", exc)

        # The epochs in the summary are what make the cached planes safe to keep: a plane
        # from before this iteration is filed under the old pixel_epoch and can never be
        # served in place of the current one.
        self._planes.update(summary)

        com = summary.com
        self.projection_view.set_source(self._planes, sx=summary.sx, sy=summary.sy)
        self.sinogram_view.set_source(self._planes)
        self.sinogram_view.set_com(
            com.com_u if com else None, com.fitted_u if com else None, summary.center
        )
        self.tomogram_view.set_source(
            self._planes, pixel_size_nm=summary.metadata.get("pixel_size_nm")
        )
        # Driven by the iteration numbers the policy kept, not by inspecting each result
        # for a volume: over a wire the volume is always stripped, so a `has_volume`
        # derived from its absence would report False for every iteration and the compare
        # dropdown would quietly empty itself.
        self.tomogram_view.set_comparison_choices(summary.volume_iterations)
        self.shift_view.update_history(summary.history, summary.angles)

    # -- session and exports -------------------------------------------------------------

    def _save_session(self) -> None:
        if not self._summary.has_engine:
            self._error("Nothing to save", "Load a projection stack first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.h5", "HDF5 (*.h5)")
        if not path:
            return
        self._run_export(
            self.session.save_session(path), path, title="Saving", label="Writing the session..."
        )

    def _open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open session", "", "HDF5 (*.h5)")
        if not path:
            return
        try:
            self.bridge.run_job(
                self.session.open_session(path),
                title="Opening",
                label="Restoring the session...",
                parent=self,
            )
        except (JobCancelled, Busy):
            return
        except JobFailed as exc:
            self._error("Could not open the session", str(exc))
            return

        self._after_engine_rebuilt()
        self._run_finished()

    def _export_projections(self) -> None:
        if not self._summary.has_engine:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export aligned projections", "aligned.h5", "HDF5 (*.h5);;TIFF (*.tif)"
        )
        if not path:
            return
        self._run_export(
            self.session.export("projections", path),
            path,
            title="Exporting",
            label="Applying shifts and writing...",
        )

    def _export_volume(self) -> None:
        if not self._summary.has_engine or self._summary.volume_shape is None:
            self._error("No volume", "Run at least one iteration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export volume", "volume.h5", "HDF5 (*.h5);;TIFF (*.tif)"
        )
        if not path:
            return
        self._run_export(
            self.session.export("volume", path), path, title="Exporting", label="Writing volume..."
        )

    def _export_shifts(self) -> None:
        if not self._summary.has_engine:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export shifts", "shifts.csv", "CSV (*.csv)")
        if path:
            self._write_table("shifts", path)

    def _export_convergence(self) -> None:
        if not self._summary.has_engine or not self._summary.history:
            self._error("No history", "Run at least one iteration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export convergence", "convergence.csv", "CSV (*.csv)"
        )
        if path:
            self._write_table("convergence", path)

    def _write_table(self, kind: str, path: str) -> None:
        """CSVs are kilobytes, so they come back as bytes and are written locally.

        The other exports are hundreds of megabytes and are written by whoever is running
        the engine -- shipping a volume across a tunnel to save it is not a plan.
        """
        try:
            Path(path).write_bytes(self.session.fetch_table(kind))
        except Exception as exc:
            self._error("Export failed", str(exc))
            return
        logger.info("Exported %s", path)
        self.action_bar.show_status(f"Exported {Path(path).name}.")

    def _run_export(self, handle, path: str, *, title: str, label: str) -> None:
        try:
            self.bridge.run_job(handle, title=title, label=label, parent=self)
        except JobCancelled:
            return
        except (JobFailed, Busy) as exc:
            self._error("Export failed", str(exc))
            return
        self.action_bar.show_status(f"Exported {Path(path).name}.")

    # -- misc --------------------------------------------------------------------------

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About ptycho-align",
            "<b>ptycho-align</b><br><br>"
            "Interactive iterative reprojection alignment for ptychographic "
            "tomography, after Gursoy et al., <i>Sci. Rep.</i> <b>7</b>, 11818 (2017)."
            "<br><br>The alignment loop is a re-implementation of TomoPy's "
            "<code>align_joint</code>, exposed one iteration at a time so it can be "
            "inspected between iterations.",
        )

    def _error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _wait_for_run_to_exit(self) -> None:
        """Block until the compute thread has exited, or the user chooses to quit anyway.

        Cancelling only takes effect *between* iterations -- a tomopy call cannot be
        interrupted once it is running -- so a cancelled run still has to finish the
        reconstruction in flight. That is not seconds: an observed SIRT iteration on a
        binned 410 x 33 x 1137 stack took 19 minutes.

        Three things could happen at that point, and only two of them are acceptable.
        Destroying the window on top of a live QThread (what the old fixed 30 s ``wait()``
        did once it expired) is undefined behaviour -- "QThread: Destroyed while thread is
        still running", a teardown racing tomopy's shared-memory cleanup, a segfault on
        the way out. Waiting it out is safe but can strand the user for 20 minutes.

        So offer the third: ``os._exit`` tears the whole process down at once. It is safe
        for exactly the reason destroying the QThread is not -- nothing survives to race
        anything, because there is no "after". Unsaved state is lost, so it is opt-in.
        """
        progress = QProgressDialog(
            "Finishing the band of detector rows already in flight before closing.\n"
            "A single tomopy call cannot be interrupted, so this takes as long as one\n"
            "band -- normally seconds, but longer on a big unbinned stack.\n\n"
            "'Quit now' ends the process immediately instead. Anything not exported\n"
            "(the volume, the shifts, the session) is lost.",
            "Quit now",
            0,
            0,
            self,
        )
        progress.setWindowTitle("Closing")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.show()
        try:
            while self.session.summary().running:
                QApplication.processEvents()
                if progress.wasCanceled():
                    logger.warning("Quit requested while a reconstruction was in flight")
                    os._exit(0)  # noqa: SLF001 - the point is to skip every teardown path
                time.sleep(0.1)
        finally:
            progress.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if self._summary.running or self.session.summary().running:
            logger.info("Closing: cancelling the run and waiting for the compute thread")
            self.session.cancel_run()
            self._wait_for_run_to_exit()
        self.bridge.close()
        self.session.close()
        logger.removeHandler(self._log_handler)
        super().closeEvent(event)


def main() -> int:
    import argparse

    from tktomo.ui.common import run_app

    parser = argparse.ArgumentParser(description="Interactive reprojection alignment.")
    parser.add_argument("path", nargs="?", help="projection stack to open (HDF5/npy/npz)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    return run_app(lambda: PtychoAlignWindow(args.path))


if __name__ == "__main__":
    raise SystemExit(main())

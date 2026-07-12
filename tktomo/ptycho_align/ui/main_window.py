"""The ptycho-align main window.

A thin shell over :mod:`tktomo.ptycho_align.core`. It owns the engine, drives it on a
worker thread, and keeps the four viewers fed.

One performance rule runs through the whole file: the aligned stack, the reprojection
and their difference are computed **once per iteration** and cached. Recomputing a
reprojection on every slider tick would make scrubbing unusable, and the viewers do
scrub.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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

from tktomo.io import ProjectionData
from tktomo.ptycho_align.core import (
    AlignConfig,
    AlignmentEngine,
    Crop,
    DatasetProblem,
    algorithm_rejects_negatives,
    center_is_plausible,
    com_prealign,
    find_center,
    inspect_dataset,
    list_hdf5_datasets,
    load_dataset,
)
from tktomo.ptycho_align.core import io as session_io
from tktomo.ptycho_align.core import preprocess as pp
from tktomo.ptycho_align.core.state import IterationResult
from tktomo.ptycho_align.ui.panels import (
    ActionBar,
    AlignPanel,
    ComPanel,
    DataPanel,
    Hdf5BrowserDialog,
    PreprocessOptions,
    PreprocessPanel,
    ProjectionView,
    ShiftView,
    SinogramView,
    TomogramView,
)
from tktomo.ptycho_align.ui.panels.base import (
    MODE_ALIGNED,
    MODE_DIFFERENCE,
    MODE_RAW,
    MODE_REPROJECTION,
)
from tktomo.ptycho_align.ui.panels.crop import CropDialog
from tktomo.ptycho_align.ui.panels.hdf5_browser import preview_text
from tktomo.ptycho_align.ui.worker import AlignmentRun

logger = logging.getLogger("tktomo.ptycho_align")

_HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx5", ".hdf")


def _is_hdf5(path: str) -> bool:
    return Path(path).suffix.lower() in _HDF5_SUFFIXES


class _Cancelled(Exception):
    """The user cancelled the load. Not an error, so it gets no dialog."""


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


def bin_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool the detector axes by ``factor`` (angles are left alone)."""
    if factor <= 1:
        return stack
    n, height, width = stack.shape
    height -= height % factor
    width -= width % factor
    trimmed = stack[:, :height, :width]
    return trimmed.reshape(n, height // factor, factor, width // factor, factor).mean(
        axis=(2, 4)
    )


class PtychoAlignWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ptycho-align -- reprojection alignment")
        self._fit_to_screen(1500, 950)

        self.raw: ProjectionData | None = None  # exactly as loaded
        self.preprocessed: ProjectionData | None = None  # engine input, before binning
        self.engine: AlignmentEngine | None = None
        self.com = None
        # How the current stack was read, so a different crop can go back to the file.
        self._source: dict | None = None
        self._run: AlignmentRun | None = None
        self._bin_factor = 1
        self._stacks: dict[str, np.ndarray | None] = {}

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
        self.action_bar.stop_requested.connect(self._stop_run)
        self.action_bar.revert_requested.connect(self._revert)

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
            dialog = Hdf5BrowserDialog(path, self)
        except Exception as exc:
            self._error("Could not read the HDF5 file", str(exc))
            return False

        logger.info("Browsing %s: %s", path, preview_text(path))
        if dialog.exec() != QDialog.Accepted:
            return False
        return self.load_path(path, **dialog.selection())

    def _read_with_progress(self, path: str, load_kwargs: dict) -> ProjectionData:
        """Load, showing a cancellable progress dialog for the big HDF5 reads.

        Reading 410 cropped projections off a 5 GB file takes long enough that a frozen
        window looks like a hang. ``load_hdf5`` calls back once per projection, which is
        also where cancellation is honoured.
        """
        if "data_path" not in load_kwargs:
            return load_dataset(path, **load_kwargs)

        progress = QProgressDialog("Reading projections...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)  # don't flash up for a small crop

        def tick(done: int, total: int) -> bool:
            progress.setMaximum(total)
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            return load_dataset(path, progress=tick, **load_kwargs)
        except DatasetProblem as exc:
            if progress.wasCanceled():
                raise _Cancelled from exc
            raise
        finally:
            progress.close()

    def _adjust_crop(self) -> None:
        """Re-read the stack over a different detector region, or a different component.

        The crop is an HDF5 hyperslab, so widening it means going back to the file --
        the discarded region was never in memory to begin with. That resets the
        alignment, hence the confirmation.
        """
        if self.raw is None or not self._source:
            self._error("No data", "Load a projection stack from an HDF5 file first.")
            return
        if self._busy():
            return

        meta = self.raw.metadata
        full_shape = tuple(meta["full_shape"])
        dialog = CropDialog(
            full_shape,
            Crop(*meta["crop"]),
            meta.get("component", "phase"),
            complex_source=self._source_is_complex(),
            roi=self._roi_as_file_crop(),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        selection = dialog.selection()
        if selection["crop"] == Crop(*meta["crop"]) and selection["component"] == meta.get(
            "component"
        ):
            return  # nothing changed

        if self.engine is not None and self.engine.iteration > 0:
            answer = QMessageBox.question(
                self,
                "Re-read the data?",
                f"Re-reading discards the {self.engine.iteration} completed iteration(s) "
                "and their shifts.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.load_path(self._source["path"], **{**self._source["kwargs"], **selection})

    def _source_is_complex(self) -> bool:
        """Is the source dataset complex? (Only then is the component combo meaningful.)"""
        entries = {e.path: e for e in list_hdf5_datasets(self._source["path"])}
        entry = entries.get(self._source["kwargs"].get("data_path", ""))
        return bool(entry and entry.is_complex)

    def _roi_as_file_crop(self) -> Crop | None:
        """The projection view's ROI, re-expressed in the *file's* row/column numbers.

        Only meaningful while the displayed stack still has the shape it was loaded
        with: preprocessing pads, and the bin factor shrinks, so after either of those
        the ROI's numbers no longer index the file.
        """
        roi = self.projection_view.roi_bounds()
        if roi is None or self.raw is None or not self._source:
            return None
        displayed = self._stacks.get(MODE_RAW)
        if displayed is None or displayed.shape != self.raw.data.shape:
            return None
        v0, u0 = self.raw.metadata["crop"][0], self.raw.metadata["crop"][2]
        return Crop(*roi).shifted_by(v0, u0)

    def load_path(self, path: str, **load_kwargs) -> bool:
        """Load a dataset. Extra keywords go straight to :func:`load_dataset`.

        With no keywords this probes the conventional layouts; when that finds nothing
        in an HDF5 file, the dataset browser opens rather than the user being told
        "no projection dataset found" and left there.
        """
        try:
            data = self._read_with_progress(path, load_kwargs)
        except KeyError as exc:
            if not load_kwargs and _is_hdf5(path):
                logger.info("No conventional layout in %s; opening the browser", path)
                return self._load_via_browser(path)
            self._error("Could not load the dataset", str(exc))
            return False
        except _Cancelled:
            self.action_bar.show_status("Loading cancelled.")
            return False
        except Exception as exc:
            self._error("Could not load the dataset", str(exc))
            return False

        # Only an explicitly-addressed HDF5 stack can be re-read with a different crop.
        self._source = (
            {"path": path, "kwargs": load_kwargs} if "data_path" in load_kwargs else None
        )

        problems = inspect_dataset(data)
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

        self.raw = data
        self.preprocessed = data
        self.com = None
        self.data_panel.show_dataset(data)
        logger.info("Loaded %s %s from %s", path, data.data.shape, data.metadata.get("data_path"))

        self._rebuild_engine()
        self.action_bar.show_status("Loaded. Apply preprocessing, then run COM pre-alignment.")
        return True

    # -- preprocessing -----------------------------------------------------------------

    def _apply_preprocessing(self, options: PreprocessOptions) -> None:
        if self.raw is None:
            self._error("No data", "Load a projection stack first.")
            return
        if self._busy():
            return

        array = np.asarray(self.raw.data, dtype=np.float32)
        roi = self.projection_view.roi_bounds()
        mask = None
        if roi is not None:
            v0, v1, u0, u1 = roi
            mask = np.zeros(array.shape[1:], dtype=bool)
            mask[v0:v1, u0:u1] = True

        try:
            if options.invert:
                array = pp.invert(array)
            if options.unwrap:
                array = pp.unwrap_phase(array)
            if options.remove_ramp:
                array = pp.remove_phase_ramp(array, mask=mask, border=options.border)
            if options.remove_offset:
                array = pp.remove_phase_offset(array, mask=mask, border=options.border)
        except Exception as exc:
            self._error("Preprocessing failed", str(exc))
            return

        ok, total = pp.check_mass_positive(array)
        if not ok:
            QMessageBox.warning(
                self,
                "Negative mass",
                f"The projection integral is {total:.4g}, i.e. negative. The "
                "reconstruction and the centre-of-mass both assume positive mass -- "
                "switch on 'Invert' and apply again.",
            )

        if options.pad_percent > 0:
            pad_v = int(round(array.shape[1] * options.pad_percent / 100.0))
            pad_u = int(round(array.shape[2] * options.pad_percent / 100.0))
            array = pp.pad(array, pad_u=pad_u, pad_v=pad_v)

        self.preprocessed = ProjectionData(
            data=array, angles=self.raw.angles, metadata=dict(self.raw.metadata)
        )
        self.com = None
        logger.info("Preprocessed -> %s", array.shape)
        self._rebuild_engine()
        self.action_bar.show_status("Preprocessing applied.")

    def _reset_preprocessing(self) -> None:
        if self.raw is None:
            return
        self.preprocessed = self.raw
        self.com = None
        self._rebuild_engine()
        self.action_bar.show_status("Reset to raw.")

    # -- COM ---------------------------------------------------------------------------

    def _busy(self) -> bool:
        """True (with a nag) if a run is in flight.

        Anything that calls into tomopy must refuse to start while the worker thread is
        already inside tomopy -- it is not thread-safe and the process segfaults.
        """
        if self._run is not None and self._run.running:
            QMessageBox.information(
                self,
                "Alignment is running",
                "Stop the current run first. TomoPy cannot safely run two "
                "reconstructions at once.",
            )
            return True
        return False

    def _run_com(self, reference: str) -> None:
        if self.engine is None:
            self._error("No data", "Load a projection stack first.")
            return
        if self._busy():
            return

        try:
            result = com_prealign(
                self.engine.state.original, self.engine.state.angles, vertical_reference=reference
            )
        except ValueError as exc:
            self._error("COM pre-alignment failed", str(exc))
            return

        self.com = result
        self.engine.state.sx = result.sx.copy()
        self.engine.state.sy = result.sy.copy()
        self.engine.state.center = result.center
        self.engine.state.history.clear()
        self.engine.state.volume = None
        self.engine.invalidate_cache()

        self.com_panel.show_result(result.center, result.fit_residual, result.amplitude)
        self.shift_view.update_com(
            self.engine.state.angles, result.com_u, result.fitted_u, result.fit_residual
        )
        self.sinogram_view.set_com(result.com_u, result.fitted_u, result.center)
        logger.info(
            "COM pre-alignment: centre %.3f px, fit residual %.3f px, amplitude %.2f px",
            result.center,
            result.fit_residual,
            result.amplitude,
        )
        self._refresh_views()
        self.action_bar.show_status(
            f"COM pre-alignment done (centre {result.center:.2f} px). Ready to step."
        )

    def _estimate_center(self, method: str) -> None:
        if self.engine is None:
            return
        if self._busy():
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            center = find_center(
                self.engine.aligned_projections(), self.engine.state.angles, method=method
            )
        except Exception as exc:
            self._error("Centre estimation failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        logger.info("Centre estimate (%s): %.3f px", method, center)

        # A failed centre-finder returns a number rather than an error, and a bad centre
        # quietly ruins a long run. find_center_vo in particular is unreliable on padded
        # phase data. Check it against the COM fit before letting it anywhere near the
        # engine.
        width = self.engine.state.original.shape[2]
        reference = self.com.center if self.com else None
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
        if self.engine is None:
            return
        self.engine.state.center = center
        self.engine.invalidate_cache()
        self.sinogram_view.set_com(
            self.com.com_u if self.com else None,
            self.com.fitted_u if self.com else None,
            center,
        )
        logger.info("Centre set to %.3f px", center)
        self.action_bar.show_status(f"Centre set to {center:.3f} px.")

    # -- engine ------------------------------------------------------------------------

    def _rebuild_engine(self, sx=None, sy=None, center=None) -> None:
        if self.preprocessed is None:
            return

        binned = bin_stack(
            np.asarray(self.preprocessed.data, dtype=np.float32), self._bin_factor
        )
        dataset = ProjectionData(
            data=binned,
            angles=self.preprocessed.angles,
            metadata=dict(self.preprocessed.metadata),
        )
        # The engine's own pad is left at (0,0): padding is applied in the preprocess
        # step so that what the viewers show is exactly what the engine aligns.
        config = self.align_panel.config()
        config.pad = (0, 0)

        self.engine = AlignmentEngine(
            dataset=dataset, config=config, sx0=sx, sy0=sy, center=center
        )
        self.shift_view.clear()
        self._refresh_views()

    def _bin_changed(self, factor: int) -> None:
        if self.engine is None or factor == self._bin_factor:
            self._bin_factor = factor
            return

        # Shifts are in pixels of the binned grid, so going 2x -> 1x doubles them.
        scale = self._bin_factor / factor
        sx = self.engine.state.sx * scale
        sy = self.engine.state.sy * scale
        center = self.engine.state.center * scale

        self._bin_factor = factor
        logger.info("Bin factor -> %dx; rescaled existing shifts by %.3g", factor, scale)
        self._rebuild_engine(sx=sx, sy=sy, center=center)
        self.action_bar.show_status(
            f"Binning {factor}x. Shifts rescaled; history cleared (the volume grid changed)."
        )

    def _config_changed(self, config: AlignConfig) -> None:
        if self.engine is None:
            return
        config.pad = (0, 0)
        self.engine.set_config(config)

    # -- running -----------------------------------------------------------------------

    def _start_run(self, n: int) -> None:
        if self.engine is None:
            self._error("No data", "Load a projection stack first.")
            return
        if self._run is not None and self._run.running:
            return

        # mlem/osem on phase data (which is ~20% negative) diverges explosively rather
        # than failing, so say so before burning a run on it.
        reason = algorithm_rejects_negatives(
            self.engine.config.recon_algorithm, self.engine.state.original
        )
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

        self.action_bar.set_running(True)
        self._run = AlignmentRun(self.engine, n)
        self._run.worker.iteration_finished.connect(self._iteration_finished)
        self._run.worker.progress.connect(self._progress)
        self._run.worker.failed.connect(self._run_failed)
        self._run.worker.run_finished.connect(self._run_finished)
        self._run.start()

    def _stop_run(self) -> None:
        if self._run is not None and self._run.running:
            self._run.cancel()
            self.action_bar.show_status("Stopping after the current iteration...")

    def _progress(self, fraction: float, message: str) -> None:
        self.action_bar.progress.setValue(int(fraction * 100))
        self.action_bar.show_status(message)

    def _iteration_finished(self, result: IterationResult) -> None:
        if self.action_bar.live_check.isChecked():
            self._refresh_views()

    def _run_failed(self, message: str) -> None:
        logger.error("Run failed:\n%s", message)
        self._error("The alignment run failed", message.strip().splitlines()[-1])

    def _run_finished(self) -> None:
        self.action_bar.set_running(False)
        self._refresh_views()
        if self.engine is not None and self.engine.history:
            last = self.engine.history[-1]
            status = (
                f"Iteration {last.iteration}  |  shift RMS {last.error:.4f} px  |  "
                f"residual {last.residual:.4f}  |  {last.wallclock_s:.1f} s"
            )
            if any(r.diverging for r in self.engine.history):
                status = "DIVERGING -- " + status
            self.action_bar.show_status(status)
            self.action_bar.revert_spin.setMaximum(last.iteration)

            if last.diverging:
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
        if self.engine is None:
            return
        try:
            self.engine.state.revert_to(iteration)
        except ValueError as exc:
            self._error("Cannot revert", str(exc))
            return
        self.engine.invalidate_cache()
        logger.info("Reverted to iteration %d", iteration)
        self._refresh_views()
        self.action_bar.show_status(f"Reverted to iteration {iteration}.")

    # -- views -------------------------------------------------------------------------

    def _refresh_views(self) -> None:
        """Feed every viewer from the stacks the last iteration already produced.

        This must NOT call tomopy. TomoPy is not thread-safe -- ``tomopy.util.mproc``
        holds its shared arrays in module-level globals -- so running ``project()``
        here on the GUI thread while the worker thread is inside ``recon()`` corrupts
        those globals and segfaults the process. With "live update views" on, that is
        exactly what would happen every iteration. The engine caches the aligned stack
        and the reprojection it already computed, so use those.
        """
        if self.engine is None:
            return

        state = self.engine.state
        running = self._run is not None and self._run.running

        aligned = self.engine.last_aligned
        simulated = self.engine.last_simulated

        if aligned is None:
            if running:
                return  # nothing cached yet and we must not compute; wait for the step
            # shift_images is pure skimage/numpy, so it is safe off the worker thread.
            aligned = self.engine.aligned_projections()

        if simulated is None and state.volume is not None and not running:
            try:
                simulated = self.engine.reproject()
            except Exception as exc:  # a missing backend must not break the viewers
                logger.warning("Could not reproject for display: %s", exc)

        self._stacks = {
            MODE_RAW: state.original,
            MODE_ALIGNED: aligned,
            MODE_REPROJECTION: simulated,
            MODE_DIFFERENCE: None if simulated is None else aligned - simulated,
        }

        self.projection_view.set_stacks(self._stacks, sx=state.sx, sy=state.sy)
        self.sinogram_view.set_stacks(self._stacks)
        self.sinogram_view.set_com(
            self.com.com_u if self.com else None,
            self.com.fitted_u if self.com else None,
            state.center,
        )
        self.tomogram_view.set_volume(
            state.volume, pixel_size_nm=state.metadata.get("pixel_size_nm")
        )
        self.tomogram_view.set_comparison_choices(
            {r.iteration: r.volume for r in state.history if r.has_volume}
        )
        self.shift_view.update_history(state.history, state.angles)

    # -- session and exports -------------------------------------------------------------

    def _save_session(self) -> None:
        if self.engine is None:
            self._error("Nothing to save", "Load a projection stack first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.h5", "HDF5 (*.h5)")
        if not path:
            return
        try:
            session_io.save_session(path, self.engine)
        except Exception as exc:
            self._error("Could not save the session", str(exc))
            return
        logger.info("Session saved to %s", path)
        self.action_bar.show_status(f"Session saved to {Path(path).name}.")

    def _open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open session", "", "HDF5 (*.h5)")
        if not path:
            return
        try:
            self.engine = session_io.load_session(path)
        except Exception as exc:
            self._error("Could not open the session", str(exc))
            return

        self.preprocessed = ProjectionData(
            data=self.engine.state.original,
            angles=self.engine.state.angles,
            metadata=dict(self.engine.state.metadata),
        )
        self.raw = self.preprocessed
        self.com = None
        self.data_panel.show_dataset(self.preprocessed)
        self._refresh_views()
        self._run_finished()
        logger.info("Session restored from %s (iteration %d)", path, self.engine.iteration)

    def _export_projections(self) -> None:
        if self.engine is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export aligned projections", "aligned.h5", "HDF5 (*.h5);;TIFF (*.tif)"
        )
        if not path:
            return
        data = ProjectionData(
            data=self.engine.aligned_projections(),
            angles=self.engine.state.angles,
            metadata={"source": "ptycho-align"},
        )
        self._export(lambda: session_io.export_projections(path, data), path)

    def _export_volume(self) -> None:
        if self.engine is None or self.engine.state.volume is None:
            self._error("No volume", "Run at least one iteration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export volume", "volume.h5", "HDF5 (*.h5);;TIFF (*.tif)"
        )
        if not path:
            return
        self._export(
            lambda: session_io.export_volume(
                path, self.engine.state.volume, self.engine.state.angles
            ),
            path,
        )

    def _export_shifts(self) -> None:
        if self.engine is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export shifts", "shifts.csv", "CSV (*.csv)")
        if not path:
            return
        state = self.engine.state
        self._export(
            lambda: session_io.export_shifts_csv(path, state.angles, state.sx, state.sy), path
        )

    def _export_convergence(self) -> None:
        if self.engine is None or not self.engine.history:
            self._error("No history", "Run at least one iteration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export convergence", "convergence.csv", "CSV (*.csv)"
        )
        if not path:
            return
        self._export(
            lambda: session_io.export_convergence_csv(path, self.engine.history), path
        )

    def _export(self, action, path: str) -> None:
        try:
            action()
        except Exception as exc:
            self._error("Export failed", str(exc))
            return
        logger.info("Exported %s", path)
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

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if self._run is not None and self._run.running:
            self._run.cancel()
            self._run.wait()
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

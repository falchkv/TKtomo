"""Shared widgets for the feature-isolation and track-model apps.

`MarkableStackView` extends the ptycho-align `StackDisplay` (which fixes
pyqtgraph's transposed axes) with click-to-place labeling. The interaction
follows `feature_alignment_app._MarkableImage` — hover tracking through the
scene, focus-follows-mouse, Delete removes nearest — but placement is a
left CLICK (pyqtgraph emits `sigMouseClicked` only when the press was not a
drag, so panning keeps working), with Space as the keyboard fallback for
pointing devices that cannot click without moving.

`load_stack_interactive` is the one "Load stack…" flow both apps share:
recognized tracking formats load directly with their coordinate chain,
foreign HDF5 goes through the ptycho-align dataset browser, TIFF gets an
angles dialog, and everything ends in a `ProvenanceDialog` where the
binning and raw-crop of the loaded grid are confirmed or typed. That
dialog is what makes "inferred values valid at full resolution" true for
pre-binned stacks the app knows nothing about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QWidget,
)

from tktomo.ptycho_align.ui.panels.base import StackDisplay
from tktomo.tracking.coords import CoordinateChain

# Qualitative palette (ColorBrewer Set1 + extras): distinct on both dark
# and light image content. Feature id -> color by modulo.
FEATURE_COLORS = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (0, 210, 210),
]


def feature_color(feature_id: int) -> tuple[int, int, int]:
    return FEATURE_COLORS[int(feature_id) % len(FEATURE_COLORS)]


class MarkableStackView(StackDisplay):
    """A StackDisplay that places labels and draws tracking overlays.

    Emits loaded-frame coordinates; the owning app converts through its
    `CoordinateChain`. The widget knows nothing about features or models,
    it only reports pointer events and draws what it is told.
    """

    placeRequested = Signal(float, float)     # left click / Space at hover
    deleteRequested = Signal(float, float)    # Delete/Backspace at hover
    stepRequested = Signal(int)               # arrow keys: +-1, +-10 (PgUp/Dn)
    digitPressed = Signal(int)                # 0..9
    hoverMoved = Signal(float, float)
    escapePressed = Signal()                  # clears transient overlays

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._mouse: tuple[float, float] | None = None
        self._raw_image: np.ndarray | None = None
        # tracking views step through hundreds of frames; per-frame robust
        # levels beat a histogram frozen on whatever frame came first
        self.auto_levels_box.setChecked(True)

        # Display-only high-pass: small features ride on a smooth phase
        # background orders of magnitude larger, and subtracting a Gaussian
        # blur is what makes them visible at all. Labels and coordinates
        # are untouched; only the pixels shown change.
        self.highpass_box = QCheckBox("High-pass")
        self.highpass_sigma = QDoubleSpinBox()
        self.highpass_sigma.setRange(0.5, 200.0)
        self.highpass_sigma.setValue(8.0)
        self.highpass_sigma.setSingleStep(1.0)
        self.highpass_sigma.setSuffix(" px")
        self.highpass_sigma.setToolTip(
            "sigma of the subtracted Gaussian blur; features larger than "
            "a few sigma fade out")
        self.highpass_box.toggled.connect(self._redisplay)
        self.highpass_sigma.valueChanged.connect(self._redisplay)
        stretch_at = self.controls_layout.count() - 1
        self.controls_layout.insertWidget(stretch_at, self.highpass_box)
        self.controls_layout.insertWidget(stretch_at + 1,
                                          self.highpass_sigma)

        view = self.image_view.getView()
        # labels are drawn in DATA pixels (pxMode=False) so their circles
        # can be sized to the physical feature and zoom with the image
        self._label_scatter = pg.ScatterPlotItem(pxMode=False)
        self._ghost_scatter = pg.ScatterPlotItem(pxMode=False)
        self._pred_scatter = pg.ScatterPlotItem(
            size=11, symbol="x", pen=pg.mkPen((255, 255, 255, 200), width=1.5),
            brush=None, pxMode=True)
        self._probe_scatter = pg.ScatterPlotItem(
            size=16, symbol="d", pen=pg.mkPen((255, 0, 255), width=2),
            brush=pg.mkBrush(255, 0, 255, 90), pxMode=True)
        self._trajectory = pg.PlotDataItem(
            pen=pg.mkPen((255, 255, 0, 160), width=1.5,
                         style=Qt.PenStyle.DashLine))
        self._crop_rect = pg.PlotDataItem(
            pen=pg.mkPen((0, 255, 255, 200), width=1.5))
        self._texts: list[pg.TextItem] = []
        for item in (self._trajectory, self._crop_rect, self._ghost_scatter,
                     self._pred_scatter, self._label_scatter,
                     self._probe_scatter):
            item.setZValue(10)
            view.addItem(item)

        # Keys land on THIS widget, wherever the pointer is.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.image_view.scene.sigMouseMoved.connect(self._on_mouse_moved)
        self.image_view.scene.sigMouseClicked.connect(self._on_mouse_clicked)

    # -- display filtering ------------------------------------------------

    def set_image(self, image, *, autorange: bool = False) -> None:
        image = np.asarray(image, np.float32)
        self._raw_image = image
        if self.highpass_box.isChecked():
            from scipy.ndimage import gaussian_filter  # noqa: PLC0415
            image = image - gaussian_filter(
                image, float(self.highpass_sigma.value()))
        super().set_image(image, autorange=autorange)

    def _redisplay(self) -> None:
        if self._raw_image is not None:
            self.set_image(self._raw_image)

    # -- pointer plumbing -------------------------------------------------

    def enterEvent(self, event):  # noqa: N802
        self.setFocus()
        super().enterEvent(event)

    def _view_coords(self, scene_pos) -> tuple[float, float] | None:
        view = self.image_view.getView()
        if not view.sceneBoundingRect().contains(scene_pos):
            return None
        point = view.mapSceneToView(scene_pos)
        return float(point.x()), float(point.y())

    def _on_mouse_moved(self, scene_pos) -> None:
        coords = self._view_coords(scene_pos)
        if coords is not None:
            self._mouse = coords
            self.hoverMoved.emit(*coords)

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        coords = self._view_coords(event.scenePos())
        if coords is not None:
            self.placeRequested.emit(*coords)
            event.accept()

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()
        text = event.text()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._mouse is not None:
                self.deleteRequested.emit(*self._mouse)
        elif key == Qt.Key.Key_Space:
            if self._mouse is not None:
                self.placeRequested.emit(*self._mouse)
        elif key == Qt.Key.Key_Left:
            self.stepRequested.emit(-1)
        elif key == Qt.Key.Key_Right:
            self.stepRequested.emit(1)
        elif key == Qt.Key.Key_PageDown:
            self.stepRequested.emit(-10)
        elif key == Qt.Key.Key_PageUp:
            self.stepRequested.emit(10)
        elif key == Qt.Key.Key_Escape:
            self.escapePressed.emit()
        elif text.isdigit():
            self.digitPressed.emit(int(text))
        else:
            super().keyPressEvent(event)

    # -- overlays ---------------------------------------------------------

    def show_labels(self, labels, active_id: int | None = None,
                    sizes: dict | None = None) -> None:
        """labels: (feature_id, u, v) or (feature_id, u, v, kind) tuples
        in the loaded frame; kind 1 (auto-placed) draws HOLLOW so the
        machine's work is distinguishable from the user's at a glance.

        `sizes` maps feature_id -> circle diameter in DATA pixels, so the
        marker can be matched to the physical feature it labels.
        """
        view = self.image_view.getView()
        for text in self._texts:
            view.removeItem(text)
        self._texts.clear()
        spots = []
        for entry in labels:
            fid, u, v = entry[0], entry[1], entry[2]
            kind = entry[3] if len(entry) > 3 else 0
            color = feature_color(fid)
            ring = 3 if (active_id is not None and fid == active_id) else 1.5
            spots.append({
                "pos": (u, v),
                "size": float((sizes or {}).get(fid, 10.0)),
                "pen": pg.mkPen(color, width=ring),
                "brush": None if kind else pg.mkBrush(*color, 70),
            })
            if not kind:      # auto labels stay untagged: less clutter
                text = pg.TextItem(str(fid), color=color, anchor=(0.5, 1.3))
                text.setPos(u, v)
                text.setZValue(11)
                view.addItem(text)
                self._texts.append(text)
        self._label_scatter.setData(spots)

    def show_ghosts(self, points, sizes: dict | None = None) -> None:
        """Faint markers of one feature's labels from OTHER frames:
        (feature_id, u, v) tuples, drawn at half size."""
        spots = []
        for fid, u, v in points:
            color = feature_color(fid)
            spots.append({
                "pos": (u, v),
                "size": float((sizes or {}).get(fid, 10.0)) / 2.0,
                "pen": pg.mkPen(*color, 110, width=1),
                "brush": None,
            })
        self._ghost_scatter.setData(spots)

    def show_predictions(self, points, active_id: int | None = None) -> None:
        """Model-predicted positions: (feature_id, u, v) in the loaded
        frame ((u, v) pairs also accepted). The ACTIVE feature's cross is
        drawn in its feature color and larger; the rest stay white.
        """
        spots = []
        for p in points:
            fid, u, v = (None, *p) if len(p) == 2 else p
            active = active_id is not None and fid == active_id
            color = feature_color(fid) if active else (255, 255, 255, 200)
            spots.append({
                "pos": (u, v),
                "symbol": "x",
                "size": 16 if active else 11,
                "pen": pg.mkPen(color, width=2.5 if active else 1.5),
                "brush": None,
            })
        self._pred_scatter.setData(spots)

    def show_probe(self, point=None) -> None:
        """A magenta diamond marking an object point picked in the recon
        slice, projected into the current frame. None clears it."""
        if point is None:
            self._probe_scatter.clear()
        else:
            self._probe_scatter.setData(x=[point[0]], y=[point[1]])

    def show_trajectory(self, u=None, v=None) -> None:
        if u is None or len(u) == 0:
            self._trajectory.clear()
        else:
            self._trajectory.setData(x=np.asarray(u), y=np.asarray(v))

    def show_crop_rect(self, origin_vu=None, window=None) -> None:
        """Rectangle at (v0, u0) of size (h, w) in the loaded frame."""
        if origin_vu is None or window is None:
            self._crop_rect.clear()
            return
        v0, u0 = float(origin_vu[0]), float(origin_vu[1])
        h, w = float(window[0]), float(window[1])
        self._crop_rect.setData(
            x=[u0, u0 + w, u0 + w, u0, u0],
            y=[v0, v0, v0 + h, v0 + h, v0])


class ProvenanceDialog(QDialog):
    """Confirm or type the loaded grid's relation to the raw detector grid.

    The center exported at "full resolution" is only as good as these two
    numbers, so they are always shown, prefilled when the file carried
    them, and editable when it did not.
    """

    def __init__(self, parent=None, *, binning: int = 1,
                 crop: tuple[int, int, int, int] = (0, 0, 0, 0),
                 from_file: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grid provenance")
        layout = QFormLayout(self)
        origin = ("read from the file's attrs" if from_file
                  else "not in the file: enter them")
        layout.addRow(QLabel(
            f"Loaded grid relative to the raw detector grid ({origin}).\n"
            f"Crop offsets are in RAW pixels, applied before the binning."))
        self.binning = QSpinBox()
        self.binning.setRange(1, 64)
        self.binning.setValue(int(binning))
        layout.addRow("Binning (loaded px per raw px):", self.binning)
        self._crop_boxes = []
        for label, value, limit in (("v0 (first raw row)", crop[0], 100000),
                                    ("v1 (past-end raw row)", crop[1], 100000),
                                    ("u0 (first raw column)", crop[2], 100000),
                                    ("u1 (past-end raw column)", crop[3],
                                     100000)):
            box = QSpinBox()
            box.setRange(0, limit)
            box.setValue(int(value))
            layout.addRow(f"Crop {label}:", box)
            self._crop_boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> tuple[int, tuple[int, int, int, int]]:
        return (int(self.binning.value()),
                tuple(int(b.value()) for b in self._crop_boxes))


class AnglesDialog(QDialog):
    """Angles for a stack that has none: uniform start/step in degrees."""

    def __init__(self, parent=None, *, n_views: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Projection angles")
        self._n = int(n_views)
        layout = QFormLayout(self)
        layout.addRow(QLabel(f"{n_views} projections, no angle metadata."))
        self.start = QSpinBox()
        self.start.setRange(-360, 360)
        self.start.setValue(0)
        self.stop = QSpinBox()
        self.stop.setRange(-360, 720)
        self.stop.setValue(180)
        layout.addRow("First angle [deg]:", self.start)
        layout.addRow("Last angle [deg]:", self.stop)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def angles_rad(self) -> np.ndarray:
        return np.deg2rad(np.linspace(float(self.start.value()),
                                      float(self.stop.value()), self._n))


class _SourceJobThread(QThread):
    """Runs one blocking StackSource call with progress/cancel callbacks."""

    progress = Signal(float, str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self.cancelled = False
        self.result = None
        self.error: BaseException | None = None

    def run(self):  # noqa: N802
        try:
            self.result = self._fn(
                lambda fraction, message: self.progress.emit(float(fraction),
                                                             str(message)),
                lambda: self.cancelled)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self.error = exc


def run_source_job(parent: QWidget, title: str, fn, *,
                   error_title: str = "Failed"):
    """Run `fn(progress, cancelled)` off the GUI thread behind a progress dialog.

    `fn` is a blocking StackSource verb (open a stack, export). Returns its
    result, or None if the user cancelled or it raised (a warning names the
    error). The dialog's Cancel flips the `cancelled()` flag the verb polls,
    so a remote job is cancelled on the host rather than abandoned.
    """
    from PySide6.QtWidgets import QApplication, QProgressDialog  # noqa: PLC0415

    dialog = QProgressDialog(title, "Cancel", 0, 1000, parent)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(300)
    dialog.setValue(0)
    thread = _SourceJobThread(fn, parent)

    def on_progress(fraction: float, message: str) -> None:
        dialog.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if message:
            dialog.setLabelText(f"{title}\n{message}")

    def on_cancel() -> None:
        thread.cancelled = True

    thread.progress.connect(on_progress)
    dialog.canceled.connect(on_cancel)
    thread.start()
    while thread.isRunning():
        QApplication.processEvents()
        thread.wait(20)
    # closing a QProgressDialog emits canceled(); that must not count as
    # the user cancelling a job that has just finished
    dialog.canceled.disconnect(on_cancel)
    dialog.close()
    if thread.error is not None:
        if not thread.cancelled:
            QMessageBox.warning(parent, error_title, str(thread.error))
        return None
    if thread.cancelled:
        return None
    return thread.result


def pick_stack_path(parent: QWidget, source=None) -> str | None:
    """Ask for a stack path: a file dialog here, a typed path on a remote host."""
    if source is not None and getattr(source, "is_remote", False):
        from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415

        path, ok = QInputDialog.getText(
            parent, "Open remote stack",
            f"Path of the stack on {source.describe()}:")
        return path.strip() if ok and path.strip() else None
    path, _ = QFileDialog.getOpenFileName(
        parent, "Load projection stack", "",
        "Stacks (*.h5 *.hdf5 *.nx *.nxs *.tif *.tiff *.npy *.npz);;"
        "All files (*)")
    return path or None


def open_stack_interactive(parent: QWidget, source, path: str):
    """Open `path` on `source` with the provenance dialogs.

    Returns (StackInfo, CoordinateChain, source_dict) or None if the user
    cancelled. `source_dict` is JSON-able and sufficient to re-load the same
    data (used by the session file); it records the endpoint when the
    source is remote. Format detection, directory listing and the read all
    happen on the source, i.e. where the file is.
    """
    from tktomo.ptycho_align.core.dataset import (  # noqa: PLC0415
        jsonable_load_kwargs,
    )
    from tktomo.ptycho_align.ui.panels.hdf5_browser import (  # noqa: PLC0415
        Hdf5BrowserDialog,
    )

    path = str(path)
    try:
        kind = source.detect_format(path)
        if kind is not None:
            info = run_source_job(
                parent, "Reading stack…",
                lambda progress, cancelled: source.open_stack(
                    path, progress=progress, cancelled=cancelled),
                error_title="Could not load stack")
            if info is None:
                return None
            dialog = ProvenanceDialog(parent, binning=info.binning,
                                      crop=info.crop, from_file=True)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            binning, crop = dialog.values()
            chain = CoordinateChain(binning=binning, crop=crop,
                                    view_origin=info.view_origin)
            src = {"path": path, "kind": kind}
        elif Path(path).suffix.lower() in (".h5", ".hdf5", ".nx", ".nxs"):
            browser = Hdf5BrowserDialog(path, parent,
                                        entries=source.list_hdf5(path))
            if browser.exec() != QDialog.DialogCode.Accepted:
                return None
            kwargs = browser.selection()
            load_kwargs = jsonable_load_kwargs(kwargs)
            info = run_source_job(
                parent, "Reading stack…",
                lambda progress, cancelled: source.open_stack(
                    path, load_kwargs=load_kwargs, progress=progress,
                    cancelled=cancelled),
                error_title="Could not load stack")
            if info is None:
                return None
            chain = _generic_chain(parent, extra_crop=kwargs.get("crop"))
            if chain is None:
                return None
            src = {"path": path, "kind": "generic_h5",
                   "load_kwargs": load_kwargs}
        else:
            info = run_source_job(
                parent, "Reading stack…",
                lambda progress, cancelled: source.open_stack(
                    path, progress=progress, cancelled=cancelled),
                error_title="Could not load stack")
            if info is None:
                return None
            if info.kind == "tiff":
                dialog = AnglesDialog(parent, n_views=info.shape[0])
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return None
                info = source.set_angles(dialog.angles_rad())
            chain = _generic_chain(parent, extra_crop=None)
            if chain is None:
                return None
            src = {"path": path, "kind": info.kind}
    except Exception as exc:  # a bad file must not take the app down
        QMessageBox.warning(parent, "Could not load stack", str(exc))
        return None

    src["binning"] = chain.binning
    src["crop"] = list(chain.crop)
    if getattr(source, "is_remote", False):
        src["endpoint"] = source.describe()
    return info, chain, src


def _generic_chain(parent, *, extra_crop) -> CoordinateChain | None:
    dialog = ProvenanceDialog(parent, binning=1, crop=(0, 0, 0, 0),
                              from_file=False)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    binning, crop = dialog.values()
    if extra_crop is not None and not isinstance(extra_crop, tuple):
        extra_crop = tuple(int(x) for x in
                           (extra_crop.as_tuple()
                            if hasattr(extra_crop, "as_tuple")
                            else extra_crop))
    return CoordinateChain(binning=binning, crop=crop, extra_crop=extra_crop)


def load_stack_interactive(parent: QWidget, path: str | None = None):
    """The in-process "Load stack…" flow (feature-isolation app).

    Returns (ProjectionData, CoordinateChain, source_dict) or None if the
    user cancelled. The same dialogs as `open_stack_interactive`, on a fresh
    `LocalStackSource`, handing back the in-memory stack.
    """
    from tktomo.tracking.stacksource import LocalStackSource  # noqa: PLC0415

    if path is None:
        path = pick_stack_path(parent)
        if not path:
            return None
    source = LocalStackSource()
    result = open_stack_interactive(parent, source, str(path))
    if result is None:
        return None
    _info, chain, src = result
    return source.data, chain, src

"""Feature isolation: track ONE feature by hand, export a moving crop.

Workflow: load a projection stack, click the feature in a handful of views
(keyframes), let the app interpolate the trajectory over angle (a
cos/sin/const sinusoid for u, which is what a rotating point does; linear
or PCHIP for v), and export a fixed-size window that follows the feature
through every view. The exported file carries per-view crop origins plus
the full binning/crop provenance, so labels placed on it in the
track-model app land in RAW detector coordinates, identical to labels
placed on the uncropped stack.

Interaction: left click (or Space) places the keyframe for the CURRENT
view, one keyframe per view; Delete removes it; Left/Right arrows step
views (PageUp/Down by 10). Focus follows the mouse.

Run with::

    python -m tktomo.ui.feature_isolation_app [stack.h5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tktomo.io import ProjectionData
from tktomo.io.phantom import generate_phantom
from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.labels import interpolate_track, sinusoid_fit_info
from tktomo.tracking.stackio import crop_track_windows, write_feature_crop
from tktomo.ui.common import run_app
from tktomo.ui.tracking_widgets import (
    MarkableStackView,
    load_stack_interactive,
)


class FeatureIsolationWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo feature isolation")

        self._data: ProjectionData | None = None
        self._chain = CoordinateChain()
        self._source: dict = {}
        self._keys: dict[int, tuple[float, float]] = {}   # view -> (u, v)
        self._view = 0
        self._track: tuple[np.ndarray, np.ndarray] | None = None

        self.viewer = MarkableStackView()
        self.viewer.placeRequested.connect(self._place)
        self.viewer.deleteRequested.connect(self._delete_current)
        self.viewer.stepRequested.connect(self._step)

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
        splitter.addWidget(self._build_controls())
        splitter.setSizes([900, 380])
        self.setCentralWidget(splitter)
        self.resize(1300, 800)

        if path:
            self._load(path)
        else:
            self._show_data(generate_phantom(60, 128, 48, max_shift=3.0),
                            CoordinateChain(), {"kind": "phantom"})

    # -- controls ---------------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        load_btn = QPushButton("Load stack…")
        load_btn.clicked.connect(lambda: self._load(None))
        layout.addWidget(load_btn)

        keys_box = QGroupBox("Keyframes (click in the image to add)")
        keys_layout = QVBoxLayout(keys_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["view", "angle°", "u", "v"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._jump_to_selected)
        keys_layout.addWidget(self.table)
        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_keys)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(clear_btn)
        keys_layout.addLayout(btn_row)
        layout.addWidget(keys_box, 1)

        interp_box = QGroupBox("Interpolation")
        form = QVBoxLayout(interp_box)
        self.u_mode = QComboBox()
        self.u_mode.addItems(["sinusoid", "linear"])
        self.v_mode = QComboBox()
        self.v_mode.addItems(["linear", "spline"])
        for label, combo in (("u over angle:", self.u_mode),
                             ("v over angle:", self.v_mode)):
            sub = QHBoxLayout()
            sub.addWidget(QLabel(label))
            sub.addWidget(combo, 1)
            form.addLayout(sub)
            combo.currentTextChanged.connect(lambda _t: self._refresh())
        self.fit_label = QLabel("sinusoid: (need 3 keyframes)")
        self.fit_label.setWordWrap(True)
        form.addWidget(self.fit_label)
        layout.addWidget(interp_box)

        crop_box = QGroupBox("Crop window")
        crop_layout = QHBoxLayout(crop_box)
        self.win_h = QSpinBox()
        self.win_h.setRange(8, 4096)
        self.win_h.setValue(128)
        self.win_w = QSpinBox()
        self.win_w.setRange(8, 4096)
        self.win_w.setValue(128)
        for label, box in (("height:", self.win_h), ("width:", self.win_w)):
            crop_layout.addWidget(QLabel(label))
            crop_layout.addWidget(box)
            box.valueChanged.connect(lambda _v: self._refresh())
        layout.addWidget(crop_box)

        save_btn = QPushButton("Save keyframes…")
        save_btn.clicked.connect(self._save_keys)
        load_keys_btn = QPushButton("Load keyframes…")
        load_keys_btn.clicked.connect(self._load_keys)
        export_btn = QPushButton("Export cropped stack…")
        export_btn.clicked.connect(self._export)
        layout.addWidget(save_btn)
        layout.addWidget(load_keys_btn)
        layout.addWidget(export_btn)
        layout.addStretch(1)
        return panel

    # -- data -------------------------------------------------------------

    def _load(self, path: str | None) -> None:
        result = load_stack_interactive(self, path)
        if result is None:
            return
        data, chain, source = result
        if self._keys and QMessageBox.question(
                self, "Discard keyframes?",
                "Loading a new stack discards the current keyframes. "
                "Continue?") != QMessageBox.StandardButton.Yes:
            return
        self._keys.clear()
        self._show_data(data, chain, source)

    def _show_data(self, data: ProjectionData, chain: CoordinateChain,
                   source: dict) -> None:
        self._data, self._chain, self._source = data, chain, source
        n = data.data.shape[0]
        self.slider.setMaximum(n - 1)
        self.view_box.setMaximum(n - 1)
        self._view = min(self._view, n - 1)
        self.viewer.reset_levels(data.data[self._view])
        self._refresh()

    # -- keyframes --------------------------------------------------------

    def _place(self, u: float, v: float) -> None:
        if self._data is None:
            return
        self._keys[self._view] = (u, v)
        self._refresh()

    def _delete_current(self, _u: float, _v: float) -> None:
        if self._keys.pop(self._view, None) is not None:
            self._refresh()

    def _remove_selected(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        views = sorted(self._keys)
        for row in rows:
            if row < len(views):
                self._keys.pop(views[row], None)
        self._refresh()

    def _clear_keys(self) -> None:
        self._keys.clear()
        self._refresh()

    def _jump_to_selected(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        views = sorted(self._keys)
        if len(rows) == 1:
            row = next(iter(rows))
            if row < len(views):
                self._set_view(views[row])

    # -- view stepping ----------------------------------------------------

    def _step(self, delta: int) -> None:
        self._set_view(self._view + delta)

    def _set_view(self, view: int) -> None:
        if self._data is None:
            return
        n = self._data.data.shape[0]
        view = int(np.clip(view, 0, n - 1))
        if view == self._view and self.slider.value() == view:
            pass
        self._view = view
        for widget in (self.slider, self.view_box):
            widget.blockSignals(True)
            widget.setValue(view)
            widget.blockSignals(False)
        self._refresh()

    # -- the single refresh path -----------------------------------------

    def _refresh(self) -> None:
        if self._data is None:
            return
        data = self._data
        view = self._view
        theta = data.angles
        self.angle_label.setText(f"{np.rad2deg(theta[view]):7.2f}°")
        self.viewer.set_image(data.data[view])

        # keyframe table
        views = sorted(self._keys)
        self.table.blockSignals(True)
        self.table.setRowCount(len(views))
        for row, w in enumerate(views):
            u, v = self._keys[w]
            for col, value in enumerate(
                    (w, f"{np.rad2deg(theta[w]):.2f}", f"{u:.1f}",
                     f"{v:.1f}")):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)

        # interpolated track + overlays
        self._track = None
        if views:
            key_u = [self._keys[w][0] for w in views]
            key_v = [self._keys[w][1] for w in views]
            u_all, v_all = interpolate_track(
                theta, views, key_u, key_v,
                u_mode=self.u_mode.currentText(),
                v_mode=self.v_mode.currentText())
            self._track = (u_all, v_all)
            info = sinusoid_fit_info(theta, views, key_u)
            if info is None:
                self.fit_label.setText("sinusoid: (need 3 keyframes)")
            else:
                self.fit_label.setText(
                    f"sinusoid: amplitude {info['amplitude']:.1f} px, "
                    f"center {info['c']:.1f} px, rms {info['rms']:.2f} px")
            self.viewer.show_trajectory(u_all, v_all)
            self.viewer.show_predictions([(u_all[view], v_all[view])])
            h, w = self.win_h.value(), self.win_w.value()
            full_h, full_w = data.data.shape[1:]
            v0 = float(np.clip(round(v_all[view] - h / 2), 0,
                               max(0, full_h - h)))
            u0 = float(np.clip(round(u_all[view] - w / 2), 0,
                               max(0, full_w - w)))
            self.viewer.show_crop_rect((v0, u0), (h, w))
        else:
            self.fit_label.setText("sinusoid: (need 3 keyframes)")
            self.viewer.show_trajectory(None)
            self.viewer.show_predictions([])
            self.viewer.show_crop_rect(None, None)

        key_here = self._keys.get(view)
        self.viewer.show_labels(
            [(0, key_here[0], key_here[1])] if key_here else [])

    # -- persistence ------------------------------------------------------

    def _save_keys(self) -> None:
        if not self._keys:
            QMessageBox.information(self, "Nothing to save",
                                    "No keyframes placed yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save keyframes", "keyframes.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(self._keyframe_state(), indent=2))

    def _keyframe_state(self) -> dict:
        return {
            "keyframes": [[w, *self._keys[w]] for w in sorted(self._keys)],
            "window": [self.win_h.value(), self.win_w.value()],
            "u_mode": self.u_mode.currentText(),
            "v_mode": self.v_mode.currentText(),
            "source": self._source,
        }

    def _load_keys(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load keyframes", "", "JSON (*.json)")
        if not path:
            return
        try:
            state = json.loads(Path(path).read_text())
            self._keys = {int(w): (float(u), float(v))
                          for w, u, v in state["keyframes"]}
            self.win_h.setValue(int(state["window"][0]))
            self.win_w.setValue(int(state["window"][1]))
            self.u_mode.setCurrentText(state.get("u_mode", "sinusoid"))
            self.v_mode.setCurrentText(state.get("v_mode", "linear"))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Could not load keyframes", str(exc))
            return
        self._refresh()

    # -- export -----------------------------------------------------------

    def _export(self) -> None:
        if self._data is None or self._track is None:
            QMessageBox.information(
                self, "Nothing to export",
                "Place at least one keyframe first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export cropped stack", "feature_crop.h5",
            "HDF5 (*.h5)")
        if not path:
            return
        window = (self.win_h.value(), self.win_w.value())
        u_all, v_all = self._track
        try:
            cropped, origin = crop_track_windows(
                self._data.data, u_all, v_all, window)
            write_feature_crop(
                path, cropped, self._data.angles, origin, self._chain,
                window=window, source=self._source,
                keyframes=self._keyframe_state()["keyframes"],
                u_mode=self.u_mode.currentText(),
                v_mode=self.v_mode.currentText())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Exported",
            f"Wrote {cropped.shape[0]} views of {window[0]}x{window[1]} px "
            f"to {path}.\nLoad it in tktomo-track-model; labels placed "
            f"there map back to raw detector coordinates.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="projection stack (.h5)")
    args = parser.parse_args()
    return run_app(lambda: FeatureIsolationWindow(args.path))


if __name__ == "__main__":
    raise SystemExit(main())

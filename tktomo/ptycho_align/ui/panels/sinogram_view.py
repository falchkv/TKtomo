"""Sinogram inspector.

Misalignment shows up here as a wobble in what ought to be a smooth sinusoidal band,
which is often the fastest way to see that something is wrong. The COM overlay makes
that comparison explicit: measured centroids as points, the fitted sinusoid as a
curve, and the rotation axis as a vertical line.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.ptycho_align.session import PlaneRef
from tktomo.ptycho_align.ui.panels.base import (
    MODE_ALIGNED,
    MODE_DIFFERENCE,
    MODE_RAW,
    MODE_REPROJECTION,
    MODE_SIDE_BY_SIDE,
    ModeSelector,
    StackDisplay,
    side_by_side,
)
from tktomo.ptycho_align.ui.planes import PlaneSource


class SinogramView(QWidget):
    """One detector row's sinogram: angle on y, u on x."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._source: PlaneSource | None = None
        self._sized = False
        self._com_u: np.ndarray | None = None
        self._fitted_u: np.ndarray | None = None
        self._center: float | None = None

        self.display = StackDisplay()
        self.mode_combo = ModeSelector(self._mode_changed)

        self.row_slider = QSlider(Qt.Horizontal)
        self.row_slider.valueChanged.connect(self._row_changed)
        self.row_spin = QSpinBox()
        self.row_spin.valueChanged.connect(self.row_slider.setValue)

        self.com_check = QCheckBox("COM overlay")
        self.com_check.toggled.connect(lambda _: self._refresh_overlays())
        self.center_check = QCheckBox("Rotation axis")
        self.center_check.setChecked(True)
        self.center_check.toggled.connect(lambda _: self._refresh_overlays())

        view = self.display.image_view.getView()
        self._com_points = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(255, 80, 80, 200), pen=None)
        self._com_curve = pg.PlotDataItem(pen=pg.mkPen("c", width=2))
        self._center_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", width=1, style=Qt.DashLine))
        for item in (self._com_points, self._com_curve, self._center_line):
            item.setVisible(False)
            view.addItem(item)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("Row:"))
        controls.addWidget(self.row_spin)
        controls.addWidget(self.row_slider, 1)
        controls.addWidget(self.com_check)
        controls.addWidget(self.center_check)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.display, 1)

    def set_source(self, source: PlaneSource) -> None:
        first = self._source is None
        self._source = source

        n_rows = source.extent(MODE_RAW, 1)
        if n_rows:
            self.row_slider.setMaximum(n_rows - 1)
            self.row_spin.setMaximum(n_rows - 1)
            if first:
                self.row_slider.setValue((n_rows - 1) // 2)

        self._refresh(autorange=first or not self._sized)

    def set_com(
        self, com_u: np.ndarray | None, fitted_u: np.ndarray | None, center: float | None
    ) -> None:
        self._com_u, self._fitted_u, self._center = com_u, fitted_u, center
        self._refresh_overlays()

    def _mode_changed(self, mode: str) -> None:
        self.display.set_symmetric(mode == MODE_DIFFERENCE)
        self._refresh(autorange=True)

    def _row_changed(self, value: int) -> None:
        if self.row_spin.value() != value:
            self.row_spin.setValue(value)
        self._refresh()

    def _refresh(self, *, autorange: bool = False) -> None:
        image = self._compose(self.mode_combo.currentText(), self.row_slider.value())
        if image is None:
            return
        self._sized = True
        self.display.set_image(image, autorange=autorange)
        self._refresh_overlays()

    def _compose(self, mode: str, row: int) -> np.ndarray | None:
        """One row's sinogram is axis 1 of the stack: ``stack[:, row, :]``, ``(angle, u)``."""
        if self._source is None:
            return None

        if mode == MODE_SIDE_BY_SIDE:
            keys = (MODE_ALIGNED, MODE_REPROJECTION, MODE_DIFFERENCE)
            parts = self._source.planes([PlaneRef(key, 1, row) for key in keys])
            if all(part is None for part in parts):
                return None
            return side_by_side(*parts)
        return self._source.plane(mode, 1, row)

    def _refresh_overlays(self) -> None:
        # The sinogram image is (angle, u): u is x, angle index is y.
        show_com = self.com_check.isChecked() and self._com_u is not None
        self._com_points.setVisible(show_com)
        self._com_curve.setVisible(show_com and self._fitted_u is not None)
        if show_com:
            angle_index = np.arange(len(self._com_u))
            self._com_points.setData(self._com_u, angle_index)
            if self._fitted_u is not None:
                self._com_curve.setData(self._fitted_u, angle_index)

        show_center = self.center_check.isChecked() and self._center is not None
        self._center_line.setVisible(show_center)
        if show_center:
            self._center_line.setPos(self._center)

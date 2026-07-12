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


class SinogramView(QWidget):
    """One detector row's sinogram: angle on y, u on x."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._stacks: dict[str, np.ndarray | None] = {}
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

    def set_stacks(self, stacks: dict[str, np.ndarray | None]) -> None:
        first = self._stacks == {}
        self._stacks = stacks

        reference = stacks.get(MODE_RAW)
        if reference is not None:
            last_row = reference.shape[1] - 1
            self.row_slider.setMaximum(max(last_row, 0))
            self.row_spin.setMaximum(max(last_row, 0))
            if first:
                self.row_slider.setValue(last_row // 2)

        self._refresh(autorange=first)

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
        self.display.set_image(image, autorange=autorange)
        self._refresh_overlays()

    def _compose(self, mode: str, row: int) -> np.ndarray | None:
        def sinogram(name: str) -> np.ndarray | None:
            stack = self._stacks.get(name)
            if stack is None or row >= stack.shape[1]:
                return None
            return stack[:, row, :]  # (n_angles, u)

        if mode == MODE_SIDE_BY_SIDE:
            parts = [
                sinogram(MODE_ALIGNED),
                sinogram(MODE_REPROJECTION),
                sinogram(MODE_DIFFERENCE),
            ]
            if all(part is None for part in parts):
                return None
            return side_by_side(*parts)
        return sinogram(mode)

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

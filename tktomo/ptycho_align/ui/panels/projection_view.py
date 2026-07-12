"""Projection inspector.

The Difference mode is the one that earns its keep: when the alignment has
converged, aligned - reprojection is structureless noise; when it has not, you see a
characteristic edge-outlined "embossed" residual, and you can see it long before the
convergence curve admits anything is wrong.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


class ProjectionView(QWidget):
    """Scrub through projections in any of the display modes."""

    roi_changed = Signal(object)  # (v0, v1, u0, u1) or None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._stacks: dict[str, np.ndarray | None] = {}
        self._sx: np.ndarray | None = None
        self._sy: np.ndarray | None = None

        self.display = StackDisplay()
        self.mode_combo = ModeSelector(self._mode_changed)

        self.index_slider = QSlider(Qt.Horizontal)
        self.index_slider.setMinimum(0)
        self.index_slider.valueChanged.connect(self._index_changed)

        self.index_spin = QSpinBox()
        self.index_spin.valueChanged.connect(self.index_slider.setValue)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)

        self.roi_check = QCheckBox("Background ROI")
        self.roi_check.setToolTip(
            "Draw a rectangle over vacuum, then use it as the region the phase ramp "
            "and offset are fitted over (Preprocess panel)."
        )
        self.roi_check.toggled.connect(self._toggle_roi)

        self.roi = pg.RectROI([10, 10], [30, 30], pen=pg.mkPen("y", width=2))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.setVisible(False)
        self.roi.sigRegionChangeFinished.connect(self._emit_roi)
        self.display.image_view.getView().addItem(self.roi)

        self.info_label = QLabel("-")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("Angle:"))
        controls.addWidget(self.index_spin)
        controls.addWidget(self.index_slider, 1)
        controls.addWidget(self.play_button)
        controls.addWidget(self.roi_check)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.display, 1)
        layout.addWidget(self.info_label)

    # -- data ------------------------------------------------------------------------

    def set_stacks(
        self,
        stacks: dict[str, np.ndarray | None],
        sx: np.ndarray | None = None,
        sy: np.ndarray | None = None,
    ) -> None:
        """Hand over the (cached) stacks for each mode. Recomputing a reprojection on
        every slider tick would be far too slow, so the main window computes them once
        per iteration and passes them in."""
        first = self._stacks == {}
        self._stacks = stacks
        self._sx, self._sy = sx, sy

        reference = stacks.get(MODE_RAW)
        if reference is not None:
            n = reference.shape[0] - 1
            self.index_slider.setMaximum(max(n, 0))
            self.index_spin.setMaximum(max(n, 0))

        self._refresh(autorange=first)

    def current_index(self) -> int:
        return self.index_slider.value()

    def roi_bounds(self) -> tuple[int, int, int, int] | None:
        """The ROI as ``(v0, v1, u0, u1)``, or None when it is not in use."""
        if not self.roi_check.isChecked():
            return None
        reference = self._stacks.get(MODE_RAW)
        if reference is None:
            return None

        n_v, n_u = reference.shape[1:]
        position, size = self.roi.pos(), self.roi.size()
        u0 = int(np.clip(round(position.x()), 0, n_u - 1))
        v0 = int(np.clip(round(position.y()), 0, n_v - 1))
        u1 = int(np.clip(round(position.x() + size.x()), u0 + 1, n_u))
        v1 = int(np.clip(round(position.y() + size.y()), v0 + 1, n_v))
        return v0, v1, u0, u1

    # -- internals -------------------------------------------------------------------

    def _mode_changed(self, mode: str) -> None:
        self.display.set_symmetric(mode == MODE_DIFFERENCE)
        self._refresh(autorange=True)

    def _index_changed(self, value: int) -> None:
        if self.index_spin.value() != value:
            self.index_spin.setValue(value)
        self._refresh()

    def _refresh(self, *, autorange: bool = False) -> None:
        image = self._compose(self.mode_combo.currentText(), self.index_slider.value())
        if image is None:
            self.info_label.setText("Nothing to show yet -- load data and run an iteration.")
            return

        self.display.set_image(image, autorange=autorange)
        self._update_info()

    def _compose(self, mode: str, index: int) -> np.ndarray | None:
        def frame(name: str) -> np.ndarray | None:
            stack = self._stacks.get(name)
            if stack is None or index >= stack.shape[0]:
                return None
            return stack[index]

        if mode == MODE_SIDE_BY_SIDE:
            parts = [frame(MODE_ALIGNED), frame(MODE_REPROJECTION), frame(MODE_DIFFERENCE)]
            if all(part is None for part in parts):
                return None
            return side_by_side(*parts)
        return frame(mode)

    def _update_info(self) -> None:
        index = self.index_slider.value()
        pieces = [f"angle index {index}"]
        if self._sx is not None and index < len(self._sx):
            pieces.append(f"sx = {self._sx[index]:+.3f} px")
        if self._sy is not None and index < len(self._sy):
            pieces.append(f"sy = {self._sy[index]:+.3f} px")
        self.info_label.setText("   |   ".join(pieces))

    def _toggle_play(self, playing: bool) -> None:
        self.play_button.setText("Stop" if playing else "Play")
        self._timer.start() if playing else self._timer.stop()

    def _advance(self) -> None:
        maximum = self.index_slider.maximum()
        if maximum > 0:
            self.index_slider.setValue((self.index_slider.value() + 1) % (maximum + 1))

    def _toggle_roi(self, enabled: bool) -> None:
        self.roi.setVisible(enabled)
        self._emit_roi()

    def _emit_roi(self) -> None:
        self.roi_changed.emit(self.roi_bounds())

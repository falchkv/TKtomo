"""Reconstructed-volume inspector.

The "compare to iteration N" control is how the user answers the only question that
matters after a run: did the last five iterations actually help? Differencing the
current volume against a stored earlier one answers it directly.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.ptycho_align.ui.panels.base import StackDisplay

# Volume axes are (z, y, x): z indexes reconstructed slices, y/x are in-plane.
_AXES = {"Axial (xy)": 0, "Coronal (xz)": 1, "Sagittal (yz)": 2}


class TomogramView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._volume: np.ndarray | None = None
        self._reference: np.ndarray | None = None  # the "compare to" volume
        self._pixel_size_nm: float | None = None

        self.display = StackDisplay()

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(list(_AXES))
        self.axis_combo.currentTextChanged.connect(lambda _: self._axis_changed())

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.slice_spin = QSpinBox()
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)

        self.compare_check = QCheckBox("Compare to iteration")
        self.compare_check.setToolTip(
            "Show the current volume minus a stored earlier one -- the direct way to "
            "see whether the last few iterations changed anything."
        )
        self.compare_check.toggled.connect(lambda _: self._refresh(autorange=True))

        self.compare_combo = QComboBox()
        self.compare_combo.setEnabled(False)
        self.compare_check.toggled.connect(self.compare_combo.setEnabled)
        self.compare_combo.currentTextChanged.connect(lambda _: self._compare_changed())

        self.info_label = QLabel("-")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plane:"))
        controls.addWidget(self.axis_combo)
        controls.addWidget(QLabel("Slice:"))
        controls.addWidget(self.slice_spin)
        controls.addWidget(self.slice_slider, 1)
        controls.addWidget(self.compare_check)
        controls.addWidget(self.compare_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.display, 1)
        layout.addWidget(self.info_label)

        self._history: dict[int, np.ndarray] = {}

    def set_volume(self, volume: np.ndarray | None, pixel_size_nm: float | None = None) -> None:
        first = self._volume is None
        self._volume = volume
        self._pixel_size_nm = pixel_size_nm
        if volume is None:
            return

        self._update_slice_range()
        self._refresh(autorange=first)

    def set_comparison_choices(self, history: dict[int, np.ndarray]) -> None:
        """Offer only the iterations whose volume the memory policy actually kept."""
        self._history = history
        current = self.compare_combo.currentText()
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.addItems([str(i) for i in sorted(history)])
        if current and current in [str(i) for i in history]:
            self.compare_combo.setCurrentText(current)
        self.compare_combo.blockSignals(False)
        self._compare_changed()

    # -- internals -------------------------------------------------------------------

    def _axis(self) -> int:
        return _AXES[self.axis_combo.currentText()]

    def _oriented(self, volume: np.ndarray) -> np.ndarray:
        return np.moveaxis(volume, self._axis(), 0)

    def _update_slice_range(self) -> None:
        if self._volume is None:
            return
        last = self._oriented(self._volume).shape[0] - 1
        was_centred = self.slice_slider.value() == 0
        self.slice_slider.setMaximum(max(last, 0))
        self.slice_spin.setMaximum(max(last, 0))
        if was_centred:
            self.slice_slider.setValue(last // 2)

    def _axis_changed(self) -> None:
        self._update_slice_range()
        self._refresh(autorange=True)

    def _slice_changed(self, value: int) -> None:
        if self.slice_spin.value() != value:
            self.slice_spin.setValue(value)
        self._refresh()

    def _compare_changed(self) -> None:
        text = self.compare_combo.currentText()
        self._reference = self._history.get(int(text)) if text else None
        self._refresh(autorange=True)

    def _refresh(self, *, autorange: bool = False) -> None:
        if self._volume is None:
            return

        comparing = (
            self.compare_check.isChecked()
            and self._reference is not None
            and self._reference.shape == self._volume.shape
        )
        volume = self._volume - self._reference if comparing else self._volume
        self.display.set_symmetric(comparing)

        index = min(self.slice_slider.value(), self._oriented(volume).shape[0] - 1)
        self.display.set_image(self._oriented(volume)[index], autorange=autorange)

        pieces = [f"{self.axis_combo.currentText()} slice {index}", f"volume {self._volume.shape}"]
        if self._pixel_size_nm:
            pieces.append(f"voxel {self._pixel_size_nm:g} nm")
        if comparing:
            pieces.append(f"difference vs iteration {self.compare_combo.currentText()}")
        self.info_label.setText("   |   ".join(pieces))

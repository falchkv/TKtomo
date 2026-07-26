"""Reconstructed-volume inspector.

The "compare to iteration N" control is how the user answers the only question that
matters after a run: did the last five iterations actually help? Differencing the
current volume against a stored earlier one answers it directly.
"""

from __future__ import annotations

from collections.abc import Sequence

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
from tktomo.ptycho_align.ui.planes import PlaneSource

# Volume axes are (z, y, x): z indexes reconstructed slices, y/x are in-plane.
_AXES = {"Axial (xy)": 0, "Coronal (xz)": 1, "Sagittal (yz)": 2}


class TomogramView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._source: PlaneSource | None = None
        self._shape: tuple[int, int, int] | None = None
        self._reference: int | None = None  # the "compare to" iteration, not its volume
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

        self._history: list[int] = []

    def set_source(self, source: PlaneSource, pixel_size_nm: float | None = None) -> None:
        first = self._shape is None
        self._source = source
        self._pixel_size_nm = pixel_size_nm
        self._shape = source.volume_shape()
        if self._shape is None:
            return

        self._update_slice_range()
        self._refresh(autorange=first)

    def set_comparison_choices(self, iterations: Sequence[int]) -> None:
        """Offer the iterations whose volume the memory policy actually kept.

        Iteration *numbers*, not volumes. Populating this used to mean fetching every
        retained volume -- three times 511 MiB to put three entries in a combo box, none
        of which is looked at unless the user ticks Compare. The difference is computed
        where the volumes live, one plane at a time.
        """
        self._history = list(iterations)
        current = self.compare_combo.currentText()
        labels = [str(i) for i in sorted(iterations)]
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.addItems(labels)
        if current and current in labels:
            self.compare_combo.setCurrentText(current)
        self.compare_combo.blockSignals(False)
        self._compare_changed()

    # -- internals -------------------------------------------------------------------

    def _axis(self) -> int:
        return _AXES[self.axis_combo.currentText()]

    def _extent(self) -> int:
        """How many slices the chosen plane has."""
        return 0 if self._shape is None else int(self._shape[self._axis()])

    def _update_slice_range(self) -> None:
        last = self._extent() - 1
        if last < 0:
            return
        was_centred = self.slice_slider.value() == 0
        self.slice_slider.setMaximum(last)
        self.slice_spin.setMaximum(last)
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
        self._reference = int(text) if text else None
        self._refresh(autorange=True)

    def _refresh(self, *, autorange: bool = False) -> None:
        if self._source is None or self._shape is None:
            return

        comparing = self.compare_check.isChecked() and self._reference is not None
        self.display.set_symmetric(comparing)

        index = min(self.slice_slider.value(), self._extent() - 1)
        image = self._source.volume_plane(
            self._axis(), index, against=self._reference if comparing else None
        )
        if image is None:
            return
        self.display.set_image(image, autorange=autorange)

        pieces = [f"{self.axis_combo.currentText()} slice {index}", f"volume {self._shape}"]
        if self._pixel_size_nm:
            pieces.append(f"voxel {self._pixel_size_nm:g} nm")
        if comparing:
            pieces.append(f"difference vs iteration {self.compare_combo.currentText()}")
        self.info_label.setText("   |   ".join(pieces))

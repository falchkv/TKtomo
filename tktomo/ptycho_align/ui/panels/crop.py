"""Choosing the detector region -- and the complex component -- to load.

A ptycho reconstruction is complex-valued and often far too large to align whole
(``obj`` in a real graphite scan: 410 x 733 x 1950 complex64, 4.7 GB on disk). Both
facts have to be settled *at load time*: the crop becomes an HDF5 hyperslab so the
discarded region never enters memory, and the component decides what real image the
alignment even works on.

:class:`CropBox` is shared by the two places that need it -- the dataset browser, for
the first load, and :class:`CropDialog`, for changing your mind afterwards.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.ptycho_align.core import COMPONENTS, Crop

# Loading more than this is usually a mistake, and the alignment holds several copies
# of the stack, so warn rather than let the machine start swapping.
_LARGE_BYTES = 2 * 1024**3


class CropBox(QGroupBox):
    """Row/column bounds for the detector region to load, with a size readout."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Crop (detector region to load)", parent)

        self._shape: tuple[int, int, int] = (1, 1, 1)

        self.v0 = QSpinBox()
        self.v1 = QSpinBox()
        self.u0 = QSpinBox()
        self.u1 = QSpinBox()
        for spin in (self.v0, self.v1, self.u0, self.u1):
            spin.setRange(0, 1)
            spin.setSingleStep(8)
            spin.valueChanged.connect(self._spin_changed)

        self.full_button = QPushButton("Full frame")
        self.full_button.clicked.connect(self.reset_to_full)

        self.size_label = QLabel()
        self.size_label.setWordWrap(True)

        rows = QHBoxLayout()
        rows.addWidget(QLabel("rows"))
        rows.addWidget(self.v0)
        rows.addWidget(QLabel("to"))
        rows.addWidget(self.v1)
        rows.addWidget(QLabel("  columns"))
        rows.addWidget(self.u0)
        rows.addWidget(QLabel("to"))
        rows.addWidget(self.u1)
        rows.addWidget(self.full_button)
        rows.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(rows)
        layout.addWidget(self.size_label)

    # -- state -----------------------------------------------------------------------

    def set_full_shape(self, shape: tuple[int, int, int], crop: Crop | None = None) -> None:
        """Point the box at a stack of ``(angle, v, u)`` shape.

        Defaults to the full frame: after a change of dataset or axis order, the old
        row/column numbers mean something different, so carrying them over would be a
        trap. Pass ``crop`` to restore a specific region instead.
        """
        self._shape = shape
        with _quiet(self.v0, self.v1, self.u0, self.u1):
            self.v0.setRange(0, shape[1] - 1)
            self.v1.setRange(1, shape[1])
            self.u0.setRange(0, shape[2] - 1)
            self.u1.setRange(1, shape[2])
            self._write((crop or Crop.full(shape)).clipped_to(shape))
        self._update_label()
        self.changed.emit()

    def reset_to_full(self) -> None:
        with _quiet(self.v0, self.v1, self.u0, self.u1):
            self._write(Crop.full(self._shape))
        self._update_label()
        self.changed.emit()

    def crop(self) -> Crop:
        v1 = max(self.v1.value(), self.v0.value() + 1)
        u1 = max(self.u1.value(), self.u0.value() + 1)
        return Crop(self.v0.value(), v1, self.u0.value(), u1)

    def set_crop(self, crop: Crop) -> None:
        with _quiet(self.v0, self.v1, self.u0, self.u1):
            self._write(crop.clipped_to(self._shape))
        self._update_label()
        self.changed.emit()

    def is_full_frame(self) -> bool:
        return self.crop() == Crop.full(self._shape)

    # -- internals -------------------------------------------------------------------

    def _write(self, crop: Crop) -> None:
        self.v0.setValue(crop.v0)
        self.v1.setValue(crop.v1)
        self.u0.setValue(crop.u0)
        self.u1.setValue(crop.u1)

    def _spin_changed(self) -> None:
        # Keep the upper bounds strictly above the lower ones, so the crop is never empty.
        with _quiet(self.v1, self.u1):
            self.v1.setMinimum(self.v0.value() + 1)
            self.u1.setMinimum(self.u0.value() + 1)
        self._update_label()
        self.changed.emit()

    def _update_label(self) -> None:
        crop = self.crop()
        n_angles = self._shape[0]
        megabytes = n_angles * crop.height * crop.width * 4 / 1024**2
        text = (
            f"{n_angles} x {crop.height} x {crop.width} float32 = {megabytes:,.0f} MB"
            f"  (full frame: {self._shape[1]} x {self._shape[2]})"
        )
        if megabytes * 1024**2 > _LARGE_BYTES:
            text += (
                "<br><b style='color:#d66'>That is a lot of memory, and the alignment "
                "keeps several copies. Crop tighter, or bin after loading.</b>"
            )
        self.size_label.setText(text)


class ComponentCombo(QComboBox):
    """Which real component of a complex stack to align."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.addItems(sorted(COMPONENTS))
        self.setCurrentText("phase")
        self.setToolTip(
            "A ptycho reconstruction is complex. 'phase' is the projected refractive "
            "index decrement -- the quantity this tool is built to align. 'amplitude' "
            "is the absorption channel; it is often cleaner to register against when "
            "the phase is badly wrapped."
        )


class CropDialog(QDialog):
    """Change the crop or the component of an already-loaded stack (it reloads from file)."""

    def __init__(
        self,
        full_shape: tuple[int, int, int],
        crop: Crop,
        component: str,
        *,
        complex_source: bool,
        roi: Crop | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust crop")

        self.crop_box = CropBox()
        self.crop_box.set_full_shape(full_shape, crop)

        self.component_combo = ComponentCombo()
        self.component_combo.setCurrentText(component)
        self.component_combo.setEnabled(complex_source)

        self.roi_button = QPushButton("Use the ROI drawn in the projection view")
        self.roi_button.setEnabled(roi is not None)
        if roi is not None:
            self.roi_button.clicked.connect(lambda: self.crop_box.set_crop(roi))
        else:
            self.roi_button.setToolTip("Draw a rectangular ROI in the projection view first.")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Component:", self.component_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("The stack is re-read from the file, so the alignment starts over.")
        )
        layout.addWidget(self.crop_box)
        layout.addWidget(self.roi_button)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def selection(self) -> dict:
        return {
            "crop": self.crop_box.crop(),
            "component": self.component_combo.currentText(),
        }


class _quiet:
    """Context manager that blocks signals on widgets, so a bulk edit emits once."""

    def __init__(self, *widgets: QWidget) -> None:
        self.widgets = widgets

    def __enter__(self) -> None:
        for widget in self.widgets:
            widget.blockSignals(True)

    def __exit__(self, *_exc) -> None:
        for widget in self.widgets:
            widget.blockSignals(False)


__all__ = ["ComponentCombo", "CropBox", "CropDialog"]

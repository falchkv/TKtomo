"""Left-dock panels: data, preprocessing and COM pre-alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.io import ProjectionData


@dataclass
class PreprocessOptions:
    remove_ramp: bool = True
    remove_offset: bool = True
    unwrap: bool = False
    invert: bool = False
    border: int = 8
    pad_percent: float = 15.0


class DataPanel(QWidget):
    """Load a dataset and show what came back."""

    load_requested = Signal()
    browse_requested = Signal()
    session_load_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.load_button = QPushButton("Load projections...")
        self.load_button.setToolTip(
            "Load a file, probing the conventional layouts (NXtomo, DXchange). If none "
            "of them fit, the HDF5 dataset browser opens instead."
        )
        self.load_button.clicked.connect(self.load_requested)

        self.browse_button = QPushButton("Browse HDF5...")
        self.browse_button.setToolTip(
            "Open an HDF5 file and navigate its tree to the projection stack yourself. "
            "Use this when the file holds several 3-D arrays, or when the automatic "
            "probe picked the wrong one."
        )
        self.browse_button.clicked.connect(self.browse_requested)

        self.session_button = QPushButton("Open session...")
        self.session_button.clicked.connect(self.session_load_requested)

        self.summary = QLabel("No data loaded.")
        self.summary.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.browse_button)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.session_button)
        layout.addWidget(self.summary)

    def show_dataset(self, data: ProjectionData) -> None:
        array = data.data
        angles = np.rad2deg(data.angles)
        step = float(np.mean(np.diff(angles))) if len(angles) > 1 else 0.0
        source = data.metadata.get("data_path")
        self.summary.setText(
            f"<b>{data.metadata.get('name', 'dataset')}</b><br>"
            + (f"<tt>{source}</tt><br>" if source else "")
            + f"shape {array.shape} (angles, v, u)<br>"
            f"angles {angles.min():.1f}..{angles.max():.1f} deg, step {step:.2f} deg<br>"
            f"min {array.min():.4g}, max {array.max():.4g}, mean {array.mean():.4g}"
        )


class PreprocessPanel(QWidget):
    """Toggles for each preprocessing step, plus apply/reset."""

    apply_requested = Signal(object)  # PreprocessOptions
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.ramp_check = QCheckBox("Remove phase ramp")
        self.ramp_check.setChecked(True)
        self.ramp_check.setToolTip(
            "Fit and subtract a plane over the background region. This matters more "
            "than anything else here: a residual linear ramp is mathematically "
            "indistinguishable from a lateral shift, so leaving it in poisons the "
            "alignment."
        )

        self.offset_check = QCheckBox("Remove phase offset")
        self.offset_check.setChecked(True)
        self.offset_check.setToolTip(
            "Subtract the background mean. The centre-of-mass code needs the vacuum "
            "to sit at zero."
        )

        self.unwrap_check = QCheckBox("Unwrap phase")
        self.unwrap_check.setToolTip("2-D phase unwrapping per projection. Slow; off by default.")

        self.invert_check = QCheckBox("Invert (make mass positive)")
        self.invert_check.setToolTip(
            "Phase is negative for most samples, but the reconstruction and the "
            "centre-of-mass both assume positive mass."
        )

        self.border_spin = QSpinBox()
        self.border_spin.setRange(1, 200)
        self.border_spin.setValue(8)
        self.border_spin.setToolTip(
            "Width of the border frame used as the background region, unless an ROI "
            "is drawn in the projection view."
        )

        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 100.0)
        self.pad_spin.setValue(15.0)
        self.pad_spin.setSuffix(" %")
        self.pad_spin.setToolTip(
            "Zero-pad the projections before aligning. Shifting moves the object "
            "toward the frame edge, and an object that walks out of frame cannot be "
            "registered back. 10-20% is a sensible default."
        )

        self.roi_label = QLabel("Background: border frame")
        self.roi_label.setWordWrap(True)

        self.apply_button = QPushButton("Apply preprocessing")
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.options()))
        self.reset_button = QPushButton("Reset to raw")
        self.reset_button.clicked.connect(self.reset_requested)

        form = QFormLayout()
        form.addRow(self.ramp_check)
        form.addRow(self.offset_check)
        form.addRow(self.unwrap_check)
        form.addRow(self.invert_check)
        form.addRow("Border width:", self.border_spin)
        form.addRow("Pad before align:", self.pad_spin)
        form.addRow(self.roi_label)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)

    def options(self) -> PreprocessOptions:
        return PreprocessOptions(
            remove_ramp=self.ramp_check.isChecked(),
            remove_offset=self.offset_check.isChecked(),
            unwrap=self.unwrap_check.isChecked(),
            invert=self.invert_check.isChecked(),
            border=self.border_spin.value(),
            pad_percent=self.pad_spin.value(),
        )

    def show_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi_label.setText(
            "Background: border frame"
            if roi is None
            else f"Background: ROI rows {roi[0]}-{roi[1]}, cols {roi[2]}-{roi[3]}"
        )


class ComPanel(QWidget):
    """Run COM pre-alignment and choose which centre estimate to trust."""

    prealign_requested = Signal(str)  # "mean" | "median"
    center_requested = Signal(str)  # "vo" | "pc"
    center_overridden = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["mean", "median"])
        self.reference_combo.setToolTip(
            "Reference for the vertical centroid. 'median' is more robust to a few "
            "bad projections."
        )

        self.run_button = QPushButton("Run COM pre-alignment")
        self.run_button.clicked.connect(
            lambda: self.prealign_requested.emit(self.reference_combo.currentText())
        )

        self.vo_button = QPushButton("Estimate centre (Vo)")
        self.vo_button.clicked.connect(lambda: self.center_requested.emit("vo"))
        self.pc_button = QPushButton("Estimate centre (phase corr.)")
        self.pc_button.clicked.connect(lambda: self.center_requested.emit("pc"))

        self.center_spin = QDoubleSpinBox()
        self.center_spin.setRange(0.0, 100_000.0)
        self.center_spin.setDecimals(3)
        self.center_spin.setToolTip("The rotation-axis position, in detector columns.")

        self.use_button = QPushButton("Use this centre")
        self.use_button.clicked.connect(
            lambda: self.center_overridden.emit(self.center_spin.value())
        )

        self.result_label = QLabel("Not run yet.")
        self.result_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Vertical reference:", self.reference_combo)
        form.addRow(self.run_button)
        form.addRow("Centre:", self.center_spin)
        form.addRow(self.use_button)

        centre_buttons = QHBoxLayout()
        centre_buttons.addWidget(self.vo_button)
        centre_buttons.addWidget(self.pc_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(centre_buttons)
        layout.addWidget(self.result_label)

    def show_result(self, center: float, fit_residual: float, amplitude: float) -> None:
        self.center_spin.setValue(center)
        warning = ""
        if amplitude < 0.5:
            # A flat sinusoid means the centroid barely moves with angle, which for a
            # real (asymmetric) object means the data or the offset removal is wrong.
            warning = (
                "<br><b style='color:#d66'>Sinusoid amplitude is nearly zero -- check "
                "the offset removal and the sign of the phase.</b>"
            )
        self.result_label.setText(
            f"centre = {center:.3f} px<br>"
            f"fit residual = {fit_residual:.3f} px<br>"
            f"sinusoid amplitude = {amplitude:.2f} px{warning}"
        )

    def set_center(self, center: float) -> None:
        self.center_spin.setValue(center)

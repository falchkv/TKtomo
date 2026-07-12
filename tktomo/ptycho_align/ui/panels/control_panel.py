"""The align-parameter panel and the bottom action bar.

The action bar is the heart of the UI: Step runs exactly one iteration and refreshes
every view, which is the whole reason this tool exists rather than a call to
``tomopy.prep.alignment.align_joint``.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)

from tktomo.ptycho_align.core.engine import AlignConfig

_ALGORITHMS = ("sirt", "mlem", "art", "gridrec", "fbp")
_UNSET = -1.0  # QDoubleSpinBox has no "None"; this stands in for max_shift_per_iter=None


class AlignPanel(QWidget):
    """Every field of :class:`AlignConfig`, with tooltips."""

    config_changed = Signal(object)  # AlignConfig
    bin_factor_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        from tktomo.recon import available_backends

        self.backend_combo = QComboBox()
        backends = available_backends() or ["tomopy"]
        self.backend_combo.addItems(backends)
        # tomopy is the default, not merely the first name in the registry -- anything
        # else registered (astra, or a test double) must be opted into deliberately.
        if "tomopy" in backends:
            self.backend_combo.setCurrentText("tomopy")
        self.backend_combo.setToolTip("Reconstruction backend. tomopy is the default.")

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(_ALGORITHMS)
        self.algorithm_combo.setToolTip(
            "TomoPy reconstruction algorithm. gridrec/fbp are direct (fast, no inner "
            "iterations and no warm start); the rest are iterative."
        )

        self.inner_spin = QSpinBox()
        self.inner_spin.setRange(1, 200)
        self.inner_spin.setValue(2)
        self.inner_spin.setToolTip(
            "TomoPy num_iter per OUTER iteration. Too few and the reprojection is too "
            "blurry to register against."
        )

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["joint", "sequential"])
        self.mode_combo.setToolTip(
            "joint: the volume is warm-started from the previous outer iteration "
            "(like align_joint). sequential: reconstructed from scratch each time "
            "(like align_seq) -- slower, but cannot inherit a bad volume."
        )

        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 200)
        self.upsample_spin.setValue(20)
        self.upsample_spin.setToolTip(
            "Subpixel registration precision is 1/upsample_factor pixels."
        )

        self.blur_check = QCheckBox("Blur edges before registering")
        self.blur_check.setChecked(True)
        self.blur_check.setToolTip(
            "Taper the projection borders before cross-correlation, to suppress edge "
            "artifacts. Applied to copies -- never to the data itself."
        )

        self.rin_spin = QDoubleSpinBox()
        self.rin_spin.setRange(0.0, 1.0)
        self.rin_spin.setSingleStep(0.05)
        self.rin_spin.setValue(0.5)
        self.rin_spin.setToolTip("Inner radius of the blur window; inside this, weight is 1.")

        self.rout_spin = QDoubleSpinBox()
        self.rout_spin.setRange(0.0, 1.0)
        self.rout_spin.setSingleStep(0.05)
        self.rout_spin.setValue(0.8)
        self.rout_spin.setToolTip("Outer radius of the blur window; outside this, weight is 0.")

        self.horizontal_check = QCheckBox("Align horizontally (sx)")
        self.horizontal_check.setChecked(True)
        self.vertical_check = QCheckBox("Align vertically (sy)")
        self.vertical_check.setChecked(True)

        self.damping_spin = QDoubleSpinBox()
        self.damping_spin.setRange(0.05, 2.0)
        self.damping_spin.setSingleStep(0.1)
        self.damping_spin.setValue(1.0)
        self.damping_spin.setToolTip(
            "Multiply every shift update by this. Below 1 stabilises a run that is "
            "oscillating instead of settling."
        )

        self.max_shift_spin = QDoubleSpinBox()
        self.max_shift_spin.setRange(_UNSET, 500.0)
        self.max_shift_spin.setSpecialValueText("none")
        self.max_shift_spin.setValue(_UNSET)
        self.max_shift_spin.setToolTip(
            "Clip each iteration's shift update to this many pixels, to stop one "
            "bad projection throwing the run off. 'none' disables the clip."
        )

        self.median_check = QCheckBox("Median-filter shifts across angle")
        self.median_check.setToolTip(
            "Smooth the per-angle shift updates. Helps when a few projections "
            "register badly; will flatten genuinely rapid stage motion."
        )

        self.refine_center_check = QCheckBox("Refine centre each iteration")
        self.refine_center_check.setToolTip(
            "Re-run TomoPy's centre finding every outer iteration. Slower, and it can "
            "wander -- leave off unless the centre is genuinely unknown."
        )

        self.ncore_spin = QSpinBox()
        self.ncore_spin.setRange(0, 256)
        self.ncore_spin.setSpecialValueText("auto")
        self.ncore_spin.setValue(0)

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 8)
        self.bin_spin.setValue(1)
        self.bin_spin.setToolTip(
            "Run on 2x or 4x binned data first: it is far faster and converges in "
            "fewer iterations. Changing this rescales the shifts you already have, so "
            "you can converge binned and then finish at full resolution."
        )
        self.bin_spin.valueChanged.connect(self.bin_factor_changed)

        form = QFormLayout(self)
        form.addRow("Backend:", self.backend_combo)
        form.addRow("Algorithm:", self.algorithm_combo)
        form.addRow("Inner iterations:", self.inner_spin)
        form.addRow("Mode:", self.mode_combo)
        form.addRow("Upsample factor:", self.upsample_spin)
        form.addRow(self.blur_check)
        form.addRow("Blur inner radius:", self.rin_spin)
        form.addRow("Blur outer radius:", self.rout_spin)
        form.addRow(self.horizontal_check)
        form.addRow(self.vertical_check)
        form.addRow("Shift damping:", self.damping_spin)
        form.addRow("Max shift / iter:", self.max_shift_spin)
        form.addRow(self.median_check)
        form.addRow(self.refine_center_check)
        form.addRow("Cores:", self.ncore_spin)
        form.addRow("Bin factor:", self.bin_spin)

        self.algorithm_combo.currentTextChanged.connect(self._algorithm_changed)
        self._algorithm_changed(self.algorithm_combo.currentText())

        for widget in (
            self.backend_combo,
            self.algorithm_combo,
            self.mode_combo,
        ):
            widget.currentTextChanged.connect(lambda _: self._emit())
        for widget in (self.inner_spin, self.upsample_spin, self.ncore_spin):
            widget.valueChanged.connect(lambda _: self._emit())
        for widget in (self.rin_spin, self.rout_spin, self.damping_spin, self.max_shift_spin):
            widget.valueChanged.connect(lambda _: self._emit())
        for widget in (
            self.blur_check,
            self.horizontal_check,
            self.vertical_check,
            self.median_check,
            self.refine_center_check,
        ):
            widget.toggled.connect(lambda _: self._emit())

    def _algorithm_changed(self, algorithm: str) -> None:
        # gridrec/fbp are direct: no inner iterations, no warm start to speak of.
        iterative = algorithm not in ("gridrec", "fbp")
        self.inner_spin.setEnabled(iterative)
        self.mode_combo.setEnabled(iterative)

    def _emit(self) -> None:
        self.config_changed.emit(self.config())

    def config(self) -> AlignConfig:
        max_shift = self.max_shift_spin.value()
        return AlignConfig(
            recon_algorithm=self.algorithm_combo.currentText(),
            recon_inner_iters=self.inner_spin.value(),
            mode=self.mode_combo.currentText(),
            backend=self.backend_combo.currentText(),
            upsample_factor=self.upsample_spin.value(),
            blur_edges=self.blur_check.isChecked(),
            rin=self.rin_spin.value(),
            rout=self.rout_spin.value(),
            refine_center=self.refine_center_check.isChecked(),
            align_vertical=self.vertical_check.isChecked(),
            align_horizontal=self.horizontal_check.isChecked(),
            shift_damping=self.damping_spin.value(),
            max_shift_per_iter=None if max_shift == _UNSET else max_shift,
            median_filter_shifts=self.median_check.isChecked(),
            ncore=self.ncore_spin.value() or None,
        )

    def bin_factor(self) -> int:
        return self.bin_spin.value()


class ActionBar(QWidget):
    """Step / Run N / Stop / Revert, the progress bar and the status readout."""

    step_requested = Signal()
    run_requested = Signal(int)
    stop_requested = Signal()
    revert_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.step_button = QPushButton("Step 1")
        self.step_button.clicked.connect(self.step_requested)

        self.run_spin = QSpinBox()
        self.run_spin.setRange(1, 500)
        self.run_spin.setValue(5)

        self.run_button = QPushButton("Run iterations")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.run_spin.value()))

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested)

        self.revert_spin = QSpinBox()
        self.revert_spin.setRange(0, 10_000)
        self.revert_button = QPushButton("Revert to iteration")
        self.revert_button.clicked.connect(
            lambda: self.revert_requested.emit(self.revert_spin.value())
        )

        self.live_check = QCheckBox("Live update views")
        self.live_check.setToolTip(
            "Refresh every view after each iteration instead of only at the end of a "
            "run. Slower, but you get to watch it converge."
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(220)

        self.status = QLabel("Load a dataset to begin.")

        layout = QHBoxLayout(self)
        layout.addWidget(self.step_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.run_spin)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.revert_button)
        layout.addWidget(self.revert_spin)
        layout.addWidget(self.live_check)
        layout.addStretch(1)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)

    def set_running(self, running: bool) -> None:
        self.step_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.revert_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if not running:
            self.progress.setValue(0)

    def show_status(self, text: str) -> None:
        self.status.setText(text)

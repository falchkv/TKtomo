"""Tomogram viewer: slice/scroll a reconstructed volume and reproject it.

Run with::

    python -m tktomo.ui.tomogram_app

The "Reproject" button forward-projects the current volume through the selected
reconstruction backend and shows the resulting projection stack.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from tktomo.io.phantom import generate_volume
from tktomo.recon import available_backends, get_backend
from tktomo.ui.common import SliceViewer, run_app


class TomogramWindow(QMainWindow):
    def __init__(self, volume: np.ndarray | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Tomogram")
        self.volume = generate_volume() if volume is None else np.asarray(volume)
        self.angles = np.linspace(0.0, np.pi, 180, endpoint=False)

        self.volume_viewer = SliceViewer(self.volume, slice_label="Z")
        self.reproj_viewer = SliceViewer(slice_label="Projection")

        # Axis selector for slicing plane.
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["axial (z)", "coronal (y)", "sagittal (x)"])
        self.axis_combo.currentIndexChanged.connect(self._change_axis)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(available_backends() or ["tomopy"])

        reproject_btn = QPushButton("Reproject →")
        reproject_btn.clicked.connect(self._reproject)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Slice plane:"))
        controls.addWidget(self.axis_combo)
        controls.addStretch(1)
        controls.addWidget(QLabel("Backend:"))
        controls.addWidget(self.backend_combo)
        controls.addWidget(reproject_btn)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.volume_viewer)
        split.addWidget(self.reproj_viewer)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(split)
        self.setCentralWidget(central)
        self.resize(1200, 700)

    def _change_axis(self, index: int) -> None:
        # Move the chosen spatial axis to the front for scrolling.
        vol = np.moveaxis(self.volume, index, 0)
        self.volume_viewer.set_volume(vol)

    def _reproject(self) -> None:
        try:
            backend = get_backend(self.backend_combo.currentText())
            projections = backend.reproject(self.volume, self.angles)
        except Exception as exc:  # backend missing or failed
            QMessageBox.warning(self, "Reprojection failed", str(exc))
            return
        self.reproj_viewer.set_volume(np.asarray(projections))


def main() -> int:
    return run_app(TomogramWindow)


if __name__ == "__main__":
    raise SystemExit(main())

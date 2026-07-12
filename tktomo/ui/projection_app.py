"""Projection viewer: scroll a projection stack, load HDF5, reconstruct.

Run with::

    python -m tktomo.ui.projection_app [file.h5]

"Load projections…" opens an HDF5 stack (NXtomo/DXchange layouts or the
blender-sim per-output groups, e.g. a ``proj.h5`` written by
``examples/headless_projections.py``). "Reconstruct →" runs the selected
reconstruction backend (algorithm/filter/center chosen from the dropdowns),
writes the tomogram to HDF5, and opens it in the tomogram viewer.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.io import list_projection_groups, load_projections, save_volume
from tktomo.io.phantom import generate_phantom
from tktomo.recon import available_backends, get_backend
from tktomo.ui.common import SliceViewer, run_app
from tktomo.ui.tomogram_app import TomogramWindow

# TomoPy algorithm names offered in the dropdown; the first two are the
# filtered analytic methods, the rest are iterative (use the iterations box).
_FILTERED_ALGORITHMS = ("gridrec", "fbp")
_ITERATIVE_ALGORITHMS = ("art", "bart", "sirt", "mlem", "osem", "tv", "grad")
_FILTERS = ("shepp", "ramlak", "cosine", "hann", "hamming", "parzen", "butterworth", "none")


class ProjectionWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Projections")
        self.data = load_projections(path) if path else generate_phantom()
        self._source_path = path
        self._tomogram_windows: list[TomogramWindow] = []

        # Scroll axis 0 = projection angle.
        self.viewer = SliceViewer(self.data.data, slice_label="Projection")

        load_btn = QPushButton("Load projections…")
        load_btn.clicked.connect(self._load)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(available_backends() or ["tomopy"])

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(_FILTERED_ALGORITHMS + _ITERATIVE_ALGORITHMS)
        self.algorithm_combo.currentTextChanged.connect(self._algorithm_changed)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(_FILTERS)
        self.filter_combo.setToolTip("Reconstruction filter (gridrec/fbp only)")

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1000)
        self.iterations_spin.setValue(50)
        self.iterations_spin.setToolTip("Iterations (iterative algorithms only)")

        self.center_edit = QLineEdit()
        self.center_edit.setPlaceholderText("auto")
        self.center_edit.setValidator(QDoubleValidator(bottom=0.0))
        self.center_edit.setMaximumWidth(80)
        self.center_edit.setToolTip("Rotation centre in pixels (blank = detector centre)")

        reconstruct_btn = QPushButton("Reconstruct →")
        reconstruct_btn.clicked.connect(self._reconstruct)

        controls = QHBoxLayout()
        controls.addWidget(load_btn)
        controls.addStretch(1)
        controls.addWidget(QLabel("Backend:"))
        controls.addWidget(self.backend_combo)
        controls.addWidget(QLabel("Algorithm:"))
        controls.addWidget(self.algorithm_combo)
        controls.addWidget(QLabel("Filter:"))
        controls.addWidget(self.filter_combo)
        controls.addWidget(QLabel("Iterations:"))
        controls.addWidget(self.iterations_spin)
        controls.addWidget(QLabel("Center:"))
        controls.addWidget(self.center_edit)
        controls.addWidget(reconstruct_btn)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(self.viewer)
        self.setCentralWidget(central)
        self.resize(1100, 700)

        self._algorithm_changed(self.algorithm_combo.currentText())
        self.statusBar().showMessage(self._source_message())

    def _source_message(self) -> str:
        n, h, w = self.data.shape
        source = self._source_path or "synthetic phantom"
        return f"{source} — {n} projections, {h}×{w} px"

    def _algorithm_changed(self, algorithm: str) -> None:
        filtered = algorithm in _FILTERED_ALGORITHMS
        self.filter_combo.setEnabled(filtered)
        self.iterations_spin.setEnabled(not filtered)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load projections", "", "HDF5 (*.h5 *.hdf5 *.nxs);;All files (*)"
        )
        if not path:
            return
        try:
            kwargs = {}
            groups = list_projection_groups(path)
            if len(groups) > 1:  # e.g. blender-sim attenuation + phase outputs
                group, ok = QInputDialog.getItem(
                    self, "Select output", "Projection group:", groups, 0, False
                )
                if not ok:
                    return
                kwargs = {
                    "data_path": f"/{group}/data",
                    "angle_path": f"/{group}/angles",
                }
            self.data = load_projections(path, **kwargs)
        except Exception as exc:  # noqa: BLE001  (any h5/format error → dialog)
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._source_path = path
        self.viewer.set_volume(self.data.data)
        self.statusBar().showMessage(self._source_message())

    def _recon_kwargs(self) -> dict:
        algorithm = self.algorithm_combo.currentText()
        kwargs: dict = {"algorithm": algorithm}
        if self.center_edit.text().strip():
            kwargs["center"] = float(self.center_edit.text())
        if algorithm in _FILTERED_ALGORITHMS:
            kwargs["filter_name"] = self.filter_combo.currentText()
        else:
            kwargs["num_iter"] = self.iterations_spin.value()
        return kwargs

    def _reconstruct(self) -> None:
        kwargs = self._recon_kwargs()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            backend = get_backend(self.backend_combo.currentText())
            volume = backend.reconstruct(self.data.data, self.data.angles, **kwargs)
        except Exception as exc:  # backend missing or reconstruction failed
            QMessageBox.warning(self, "Reconstruction failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        saved = self._save_tomogram(volume, kwargs)
        window = TomogramWindow(volume, angles=self.data.angles)
        window.show()
        self._tomogram_windows.append(window)
        message = f"Reconstructed {volume.shape} with {kwargs['algorithm']}"
        if saved:
            message += f" — saved to {saved}"
        self.statusBar().showMessage(message)

    def _save_tomogram(self, volume, kwargs: dict) -> str | None:
        """Prompt for an HDF5 path and write the tomogram; None if cancelled."""
        if self._source_path:
            stem = os.path.splitext(self._source_path)[0]
            suggested = f"{stem}_tomogram.h5"
        else:
            suggested = "tomogram.h5"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save tomogram", suggested, "HDF5 (*.h5 *.hdf5)"
        )
        if not path:
            return None
        try:
            save_volume(
                path,
                volume,
                angles=self.data.angles,
                metadata={
                    "backend": self.backend_combo.currentText(),
                    "source": self._source_path or "synthetic phantom",
                    **{k: v for k, v in kwargs.items() if v is not None},
                },
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
            return None
        return path


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    return run_app(lambda: ProjectionWindow(path))


if __name__ == "__main__":
    raise SystemExit(main())

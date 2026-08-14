"""Reusable slice viewer: a pyqtgraph ImageView + a colormap selector.

Wraps a 3D volume ``(n_slices, height, width)`` with scrolling between slices and
a colormap dropdown backed by :mod:`tktomo.colormaps`. Used by the sinogram,
projection and tomogram viewers.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from tktomo.colormaps import available_colormaps, get_colormap


class SliceViewer(QWidget):
    """Display a 3D array, scrolling along axis 0, with a colormap selector."""

    def __init__(
        self,
        volume: np.ndarray | None = None,
        *,
        default_colormap: str = "viridis",
        slice_label: str = "Slice",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._slice_label = slice_label

        self.image_view = pg.ImageView()
        # ImageView's built-in time-axis slider provides slice scrolling.

        self.colormap_combo = QComboBox()
        names = available_colormaps()
        self.colormap_combo.addItems(names)
        from tktomo.ui.common.colormap_icons import add_colormap_icons  # noqa: PLC0415
        add_colormap_icons(self.colormap_combo)
        if default_colormap in names:
            self.colormap_combo.setCurrentText(default_colormap)
        self.colormap_combo.currentTextChanged.connect(self._apply_colormap)

        top = QHBoxLayout()
        top.addWidget(QLabel("Colormap:"))
        top.addWidget(self.colormap_combo)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.image_view)

        self._apply_colormap(self.colormap_combo.currentText())
        if volume is not None:
            self.set_volume(volume)

    def set_volume(self, volume: np.ndarray) -> None:
        """Set/replace the displayed volume, preserving the current slice index."""
        volume = np.asarray(volume)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        idx = self.image_view.currentIndex if self.image_view.image is not None else 0
        # pyqtgraph indexes the first axis as the scroll axis when xvals given.
        self.image_view.setImage(
            volume,
            autoRange=self.image_view.image is None,
            autoLevels=self.image_view.image is None,
            xvals=np.arange(volume.shape[0]),
        )
        if self.image_view.image is not None:
            self.image_view.setCurrentIndex(min(idx, volume.shape[0] - 1))

    def _apply_colormap(self, name: str) -> None:
        try:
            self.image_view.setColorMap(get_colormap(name))
        except Exception:  # pragma: no cover - never let a bad map crash the UI
            pass

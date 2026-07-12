"""Shared display widget for the four viewers.

Wraps ``pyqtgraph.ImageView`` with the bits every panel here needs and the stock
:class:`tktomo.ui.common.SliceViewer` does not: an explicit axis mapping (so rows
stay vertical instead of pyqtgraph's default transpose), levels that stay symmetric
about zero for difference images, and a display-mode combo.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from tktomo.colormaps import available_colormaps, get_colormap

# Modes shared by the projection and sinogram viewers.
MODE_RAW = "Raw"
MODE_ALIGNED = "Aligned"
MODE_REPROJECTION = "Reprojection"
MODE_DIFFERENCE = "Difference"
MODE_SIDE_BY_SIDE = "Side by side"

# A diverging map, so a difference image reads as "signed" at a glance. Falls back to
# grayscale, which the colormap registry always provides.
_DIVERGING_PREFERENCES = ("cet-diverging-bwr", "cmo-balance", "grayscale")


def diverging_colormap_name() -> str:
    available = available_colormaps()
    return next(name for name in _DIVERGING_PREFERENCES if name in available)


class StackDisplay(QWidget):
    """An ImageView over a 3-D stack, scrolled externally (not by pyqtgraph's slider).

    The owning panel drives the index, because the projection and sinogram views need
    their own slider + spinbox + play button rather than pyqtgraph's timeline.
    """

    def __init__(self, *, default_colormap: str = "viridis", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Row 0 at the top, like every other image viewer in the world.
        self.image_view.getView().invertY(True)

        self.colormap_combo = QComboBox()
        names = available_colormaps()
        self.colormap_combo.addItems(names)
        self.colormap_combo.setCurrentText("viridis" if "viridis" in names else names[0])
        self.colormap_combo.currentTextChanged.connect(self._apply_colormap)

        top = QHBoxLayout()
        top.addWidget(QLabel("Colormap:"))
        top.addWidget(self.colormap_combo)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top_row = QWidget()
        top_row.setLayout(top)
        layout.addWidget(top_row)
        layout.addWidget(self.image_view)

        self._symmetric = False
        self._apply_colormap(self.colormap_combo.currentText())

    def set_symmetric(self, symmetric: bool) -> None:
        """Difference images get levels centred on zero and a diverging map."""
        if symmetric == self._symmetric:
            return
        self._symmetric = symmetric
        target = diverging_colormap_name() if symmetric else "viridis"
        if target in available_colormaps():
            self.colormap_combo.setCurrentText(target)

    def set_image(self, image: np.ndarray, *, autorange: bool = False) -> None:
        """Show a 2-D image (rows vertical, columns horizontal)."""
        image = np.asarray(image, dtype=np.float32)

        # axes= keeps rows on y and columns on x; without it pyqtgraph maps the first
        # axis to x and every image comes out transposed.
        self.image_view.setImage(
            image,
            axes={"y": 0, "x": 1},
            autoRange=autorange,
            autoLevels=False,
            autoHistogramRange=False,
        )

        if self._symmetric:
            extent = float(np.nanmax(np.abs(image))) if image.size else 1.0
            extent = extent or 1.0
            self.image_view.setLevels(-extent, extent)
        elif self.image_view.getHistogramWidget().item.getLevels() == (0.0, 1.0):
            # First real image: give it sensible levels once, then leave the user's
            # histogram alone on subsequent frames.
            self.image_view.autoLevels()

    def reset_levels(self, image: np.ndarray) -> None:
        if image.size:
            self.image_view.setLevels(float(np.nanmin(image)), float(np.nanmax(image)))

    def _apply_colormap(self, name: str) -> None:
        try:
            self.image_view.setColorMap(get_colormap(name))
        except Exception:  # never let a bad colormap take the window down
            pass


class ModeSelector(QComboBox):
    """The Raw / Aligned / Reprojection / Difference / Side-by-side combo."""

    def __init__(self, on_change: Callable[[str], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.addItems(
            [MODE_RAW, MODE_ALIGNED, MODE_REPROJECTION, MODE_DIFFERENCE, MODE_SIDE_BY_SIDE]
        )
        self.setCurrentText(MODE_ALIGNED)
        self.currentTextChanged.connect(on_change)


def side_by_side(*images: np.ndarray) -> np.ndarray:
    """Lay images out in one row, separated by a thin gap."""
    usable = [np.asarray(image) for image in images if image is not None]
    if not usable:
        return np.zeros((1, 1), dtype=np.float32)

    height = max(image.shape[0] for image in usable)
    gap = np.full((height, 2), np.nan, dtype=np.float32)

    columns: list[np.ndarray] = []
    for index, image in enumerate(usable):
        padded = np.full((height, image.shape[1]), np.nan, dtype=np.float32)
        padded[: image.shape[0]] = image
        if index:
            columns.append(gap)
        columns.append(padded)
    return np.hstack(columns)

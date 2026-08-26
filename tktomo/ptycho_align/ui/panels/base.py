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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from tktomo.colormaps import available_colormaps, get_colormap
from tktomo.ui.common.colormap_icons import add_colormap_icons

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


class NoMomentumViewBox(pg.ViewBox):
    """A ViewBox that ignores a touchpad's kinetic scroll tail.

    A mouse wheel sends one event per notch. A touchpad sends a stream of
    small events and, after the fingers lift, keeps sending "momentum"
    events that decay over a second or so. Fed into the exponential zoom,
    that tail keeps zooming after the gesture ended and overshoots. Qt
    tags those events with ``ScrollPhase.ScrollMomentum``, so they are
    dropped here; every other event (mouse wheels report ``NoScrollPhase``)
    reaches the stock handler untouched.
    """

    def wheelEvent(self, ev, axis=None):
        phase = getattr(ev, "phase", None)
        if phase is not None and phase() == Qt.ScrollPhase.ScrollMomentum:
            ev.accept()
            return
        super().wheelEvent(ev, axis)


class StackDisplay(QWidget):
    """An ImageView over a 3-D stack, scrolled externally (not by pyqtgraph's slider).

    The owning panel drives the index, because the projection and sinogram views need
    their own slider + spinbox + play button rather than pyqtgraph's timeline.
    """

    def __init__(self, *, default_colormap: str = "viridis", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.image_view = pg.ImageView(view=NoMomentumViewBox())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Row 0 at the top, like every other image viewer in the world.
        self.image_view.getView().invertY(True)

        self.colormap_combo = QComboBox()
        names = available_colormaps()
        self.colormap_combo.addItems(names)
        add_colormap_icons(self.colormap_combo)
        self.colormap_combo.setCurrentText("viridis" if "viridis" in names else names[0])
        self.colormap_combo.currentTextChanged.connect(self._apply_colormap)

        # Off by default: ptycho-align deliberately leaves the user's histogram
        # alone between frames. The tracking apps switch it on.
        self.auto_levels_box = QCheckBox("Auto levels")
        self.auto_levels_box.toggled.connect(self._auto_levels_toggled)

        # kept as an attribute so subclasses can add their own controls in
        # front of the stretch (e.g. the tracking views' high-pass filter)
        self.controls_layout = top = QHBoxLayout()
        top.addWidget(QLabel("Colormap:"))
        top.addWidget(self.colormap_combo)
        top.addWidget(self.auto_levels_box)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top_row = QWidget()
        top_row.setLayout(top)
        layout.addWidget(top_row)
        layout.addWidget(self.image_view)

        self._symmetric = False
        self._last_image: np.ndarray | None = None
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

        self._last_image = image
        if self._symmetric:
            extent = float(np.nanmax(np.abs(image))) if image.size else 1.0
            extent = extent or 1.0
            self.image_view.setLevels(-extent, extent)
        elif self.auto_levels_box.isChecked():
            self._apply_auto_levels(image)
        elif self.image_view.getHistogramWidget().item.getLevels() == (0.0, 1.0):
            # First real image: give it sensible levels once, then leave the user's
            # histogram alone on subsequent frames.
            self.image_view.autoLevels()

    def _apply_auto_levels(self, image: np.ndarray) -> None:
        """Robust per-frame levels: clamp to the 1-99 percentile range.

        Straight min/max hands the whole scale to one hot pixel; percentiles
        keep the display closed around where the signal actually lives.
        """
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return
        lo, hi = (float(x) for x in np.percentile(finite, (1.0, 99.0)))
        if lo == hi:
            # nearly-constant frame: widen around the bulk value instead of
            # falling back to min/max, which would re-admit the very
            # outliers the percentiles were there to exclude
            pad = max(0.5, 1e-3 * abs(lo))
            lo, hi = lo - pad, hi + pad
        self.image_view.setLevels(lo, hi)
        self.image_view.setHistogramRange(lo, hi)

    def _auto_levels_toggled(self, checked: bool) -> None:
        if checked and self._last_image is not None:
            self._apply_auto_levels(self._last_image)

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

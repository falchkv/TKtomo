"""Gradient swatch icons for colormap dropdowns.

A colormap name means nothing until you have used it a dozen times; the
swatch is the actual information. Built once per combo fill from the
registry's lookup tables.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QImage, QPixmap

ICON_SIZE = QSize(48, 12)


def colormap_icon(name: str) -> QIcon:
    """A horizontal gradient swatch of the named colormap."""
    from tktomo.colormaps import get_colormap  # noqa: PLC0415

    try:
        lut = get_colormap(name).getLookupTable(0.0, 1.0, ICON_SIZE.width())
    except Exception:  # a bad map gets a blank icon, never an error
        return QIcon()
    lut = np.ascontiguousarray(np.asarray(lut, np.uint8)[:, :3])
    strip = np.ascontiguousarray(
        np.repeat(lut[None, :, :], ICON_SIZE.height(), axis=0))
    image = QImage(strip.data, strip.shape[1], strip.shape[0],
                   strip.shape[1] * 3, QImage.Format.Format_RGB888)
    # .copy() detaches the QImage from the numpy buffer before it dies
    return QIcon(QPixmap.fromImage(image.copy()))


def add_colormap_icons(combo) -> None:
    """Give every entry of a colormap QComboBox its gradient swatch."""
    combo.setIconSize(ICON_SIZE)
    for index in range(combo.count()):
        combo.setItemIcon(index, colormap_icon(combo.itemText(index)))

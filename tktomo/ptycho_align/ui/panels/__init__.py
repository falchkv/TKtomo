"""Dock panels and viewers for the reprojection alignment window."""

from __future__ import annotations

from tktomo.ptycho_align.ui.panels.control_panel import ActionBar, AlignPanel
from tktomo.ptycho_align.ui.panels.data_panel import (
    ComPanel,
    DataPanel,
    PreprocessOptions,
    PreprocessPanel,
)
from tktomo.ptycho_align.ui.panels.crop import ComponentCombo, CropBox, CropDialog
from tktomo.ptycho_align.ui.panels.hdf5_browser import Hdf5BrowserDialog
from tktomo.ptycho_align.ui.panels.projection_view import ProjectionView
from tktomo.ptycho_align.ui.panels.shift_view import ShiftView
from tktomo.ptycho_align.ui.panels.sinogram_view import SinogramView
from tktomo.ptycho_align.ui.panels.tomogram_view import TomogramView

__all__ = [
    "ActionBar",
    "AlignPanel",
    "ComPanel",
    "ComponentCombo",
    "CropBox",
    "CropDialog",
    "DataPanel",
    "Hdf5BrowserDialog",
    "PreprocessOptions",
    "PreprocessPanel",
    "ProjectionView",
    "ShiftView",
    "SinogramView",
    "TomogramView",
]

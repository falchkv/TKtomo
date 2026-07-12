"""Headless core of the reprojection alignment app.

Importable and fully usable without PySide6 -- the whole alignment can be scripted
from a notebook. tomopy/skimage/scipy are imported lazily inside functions, so
``import tktomo.ptycho_align.core`` stays light.
"""

from __future__ import annotations

from tktomo.ptycho_align.core.com import ComResult, com_prealign, find_center
from tktomo.ptycho_align.core.dataset import (
    DatasetProblem,
    inspect_dataset,
    load_dataset,
)
from tktomo.ptycho_align.core.engine import AlignConfig, AlignmentEngine, apply_shifts
from tktomo.ptycho_align.core.state import AlignmentState, IterationResult, VolumePolicy

__all__ = [
    "AlignConfig",
    "AlignmentEngine",
    "AlignmentState",
    "ComResult",
    "DatasetProblem",
    "IterationResult",
    "VolumePolicy",
    "apply_shifts",
    "com_prealign",
    "find_center",
    "inspect_dataset",
    "load_dataset",
]

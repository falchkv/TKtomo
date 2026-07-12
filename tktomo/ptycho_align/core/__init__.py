"""Headless core of the reprojection alignment app.

Importable and fully usable without PySide6 -- the whole alignment can be scripted
from a notebook. tomopy/skimage/scipy are imported lazily inside functions, so
``import tktomo.ptycho_align.core`` stays light.
"""

from __future__ import annotations

from tktomo.ptycho_align.core.com import (
    ComResult,
    center_is_plausible,
    com_prealign,
    find_center,
)
from tktomo.ptycho_align.core.dataset import (
    COMPONENTS,
    Crop,
    DatasetProblem,
    Hdf5Entry,
    inspect_dataset,
    list_hdf5_datasets,
    load_dataset,
    load_hdf5,
    suggest_hdf5_paths,
    to_real,
)
from tktomo.ptycho_align.core.engine import (
    AlignConfig,
    AlignmentEngine,
    algorithm_rejects_negatives,
    apply_shifts,
)
from tktomo.ptycho_align.core.state import AlignmentState, IterationResult, VolumePolicy

__all__ = [
    "COMPONENTS",
    "AlignConfig",
    "AlignmentEngine",
    "AlignmentState",
    "ComResult",
    "Crop",
    "center_is_plausible",
    "DatasetProblem",
    "Hdf5Entry",
    "IterationResult",
    "VolumePolicy",
    "algorithm_rejects_negatives",
    "apply_shifts",
    "com_prealign",
    "find_center",
    "inspect_dataset",
    "list_hdf5_datasets",
    "load_dataset",
    "load_hdf5",
    "suggest_hdf5_paths",
    "to_real",
]

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
    coerce_load_kwargs,
    inspect_dataset,
    jsonable_load_kwargs,
    list_hdf5_datasets,
    load_dataset,
    load_hdf5,
    suggest_hdf5_paths,
    to_real,
)
from tktomo.ptycho_align.core.engine import (
    DIRECT_ALGORITHMS,
    AlignConfig,
    AlignmentEngine,
    Cancelled,
    algorithm_rejects_negatives,
    apply_shifts,
    row_chunk_size,
    shift_update_is_runaway,
)
from tktomo.ptycho_align.core.state import AlignmentState, IterationResult, VolumePolicy

__all__ = [
    "COMPONENTS",
    "DIRECT_ALGORITHMS",
    "AlignConfig",
    "AlignmentEngine",
    "AlignmentState",
    "Cancelled",
    "ComResult",
    "Crop",
    "center_is_plausible",
    "DatasetProblem",
    "Hdf5Entry",
    "IterationResult",
    "VolumePolicy",
    "algorithm_rejects_negatives",
    "apply_shifts",
    "coerce_load_kwargs",
    "com_prealign",
    "find_center",
    "inspect_dataset",
    "jsonable_load_kwargs",
    "list_hdf5_datasets",
    "load_dataset",
    "load_hdf5",
    "row_chunk_size",
    "shift_update_is_runaway",
    "suggest_hdf5_paths",
    "to_real",
]

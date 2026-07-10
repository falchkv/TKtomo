"""Apply an alignment transform to a projection within a stack.

Kept Qt-free and array-based so it is unit-testable and reusable by the sinogram
viewer (which owns the projection stack and re-derives sinograms from it).
"""

from __future__ import annotations

import numpy as np

from tktomo.align.transform import Transform, apply_transform


def apply_projection_transform(
    data: np.ndarray, index: int, transform: Transform, *, order: int = 1
) -> np.ndarray:
    """Return a copy of ``data`` with projection ``index`` warped by ``transform``.

    ``data`` has shape ``(n_angles, height, width)``. Only the given projection is
    modified; every sinogram row derived from that angle changes accordingly.
    """
    if not 0 <= index < data.shape[0]:
        raise IndexError(f"projection index {index} out of range [0, {data.shape[0]})")
    out = np.array(data, copy=True)
    if not transform.is_identity():
        out[index] = apply_transform(data[index], transform, order=order)
    return out

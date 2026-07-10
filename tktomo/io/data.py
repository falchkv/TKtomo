"""Core data container shared across the toolkit.

A :class:`ProjectionData` bundles a projection stack with its rotation angles and
arbitrary metadata. The same container is reused for sinogram / tomogram slices
so the UIs and the messaging layer all speak one type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProjectionData:
    """A stack of tomographic projections.

    Attributes
    ----------
    data:
        Array of shape ``(n_angles, height, width)``. May be real (amplitude or
        phase) or complex for raw ptycho output.
    angles:
        Rotation angles in **radians**, shape ``(n_angles,)``. This is the
        convention TomoPy uses, so no conversion is needed at the backend.
    metadata:
        Free-form dictionary (pixel size, energy, source path, channel, ...).
    """

    data: np.ndarray
    angles: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.angles = np.asarray(self.angles, dtype=np.float64)
        if self.data.ndim != 3:
            raise ValueError(
                f"data must be 3D (n_angles, height, width); got shape {self.data.shape}"
            )
        if self.angles.shape != (self.data.shape[0],):
            raise ValueError(
                f"angles shape {self.angles.shape} does not match "
                f"n_angles={self.data.shape[0]}"
            )

    @property
    def n_angles(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def sinogram(self, row: int) -> np.ndarray:
        """Return the sinogram for detector ``row``: shape ``(n_angles, width)``."""
        return self.data[:, row, :]

    def projection(self, index: int) -> np.ndarray:
        """Return a single projection image: shape ``(height, width)``."""
        return self.data[index]

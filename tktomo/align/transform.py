"""Rigid 2D transform, its application to an image, and an undo history.

The alignment UI edits a single :class:`Transform` (translation + rotation).
:class:`TransformHistory` provides the undo stack backing the UI's undo button
and hotkey.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Transform:
    """A rigid 2D transform: translation (pixels) + rotation (degrees).

    ``dx`` shifts along the last axis (columns / width), ``dy`` along the second
    axis (rows / height). Rotation is about the image centre, counter-clockwise.
    """

    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"dx": self.dx, "dy": self.dy, "rotation": self.rotation}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Transform":
        return cls(dx=d.get("dx", 0.0), dy=d.get("dy", 0.0), rotation=d.get("rotation", 0.0))

    def is_identity(self, tol: float = 1e-9) -> bool:
        return abs(self.dx) < tol and abs(self.dy) < tol and abs(self.rotation) < tol

    def with_shift(self, dx: float, dy: float) -> "Transform":
        return replace(self, dx=dx, dy=dy)


def apply_transform(image: np.ndarray, transform: Transform, *, order: int = 1) -> np.ndarray:
    """Apply ``transform`` to ``image`` (rotation about centre, then translation)."""
    from scipy.ndimage import affine_transform  # noqa: PLC0415

    if transform.is_identity():
        return image.copy()

    theta = np.deg2rad(transform.rotation)
    cos, sin = np.cos(theta), np.sin(theta)
    # Rotation matrix in (row, col) order for scipy's inverse mapping.
    rot = np.array([[cos, -sin], [sin, cos]])
    centre = 0.5 * (np.asarray(image.shape) - 1)
    offset = centre - rot @ centre - np.array([transform.dy, transform.dx])
    return affine_transform(image, rot, offset=offset, order=order, mode="nearest")


class TransformHistory:
    """Undo stack for the alignment UI.

    ``current`` is the active transform; :meth:`push` records a new state,
    :meth:`undo` reverts to the previous one.
    """

    def __init__(self, initial: Transform | None = None) -> None:
        self._stack: list[Transform] = [initial or Transform()]

    @property
    def current(self) -> Transform:
        return self._stack[-1]

    def push(self, transform: Transform) -> None:
        self._stack.append(transform)

    def undo(self) -> Transform:
        """Pop the most recent transform; keeps at least the initial state."""
        if len(self._stack) > 1:
            self._stack.pop()
        return self.current

    def can_undo(self) -> bool:
        return len(self._stack) > 1

    def reset(self, transform: Transform | None = None) -> None:
        self._stack = [transform or Transform()]

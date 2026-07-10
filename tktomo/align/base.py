"""Aligner protocol and the method registry.

An :class:`Aligner` estimates the :class:`~tktomo.align.transform.Transform` that
best maps a ``moving`` image onto a ``fixed`` image, optionally starting from an
initial guess (the alignment UI's current state). Methods register under a short
name so the UI dropdown can enumerate them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tktomo.align.transform import Transform


@runtime_checkable
class Aligner(Protocol):
    """Estimate a transform aligning ``moving`` to ``fixed``."""

    #: Human-readable name shown in the UI dropdown.
    name: str

    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial: Transform | None = None,
    ) -> Transform:
        """Return the transform that best maps ``moving`` onto ``fixed``."""
        ...


_REGISTRY: dict[str, Aligner] = {}


def register_aligner(aligner: Aligner) -> Aligner:
    """Register an aligner instance under ``aligner.name``. Returns it unchanged."""
    if not getattr(aligner, "name", None):
        raise ValueError("Aligner must define a non-empty 'name'.")
    _REGISTRY[aligner.name] = aligner
    return aligner


def available_aligners() -> list[str]:
    """Names of registered aligners, for populating the method dropdown."""
    return sorted(_REGISTRY)


def get_aligner(name: str) -> Aligner:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown aligner {name!r}. Available: {available_aligners()}"
        ) from None

"""ReconBackend protocol and registry.

A backend reconstructs a volume from projections and reprojects a volume back to
projections (used by the tomogram UI's "reproject" button and by joint alignment).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ReconBackend(Protocol):
    name: str

    def reconstruct(
        self, projections: np.ndarray, angles: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Reconstruct a volume from projections ``(n_angles, height, width)``."""
        ...

    def reproject(
        self, volume: np.ndarray, angles: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Forward-project a volume to projections ``(n_angles, height, width)``."""
        ...


_REGISTRY: dict[str, ReconBackend] = {}


def register_backend(backend: ReconBackend) -> ReconBackend:
    if not getattr(backend, "name", None):
        raise ValueError("Backend must define a non-empty 'name'.")
    _REGISTRY[backend.name] = backend
    return backend


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str = "tomopy") -> ReconBackend:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown backend {name!r}. Available: {available_backends()}"
        ) from None

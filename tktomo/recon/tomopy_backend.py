"""TomoPy reconstruction/reprojection backend.

TomoPy is imported lazily so the package works without it; a clear error is
raised on first use if it is missing. Install via conda-forge:
``conda install -c conda-forge tomopy``.
"""

from __future__ import annotations

import numpy as np

from tktomo.recon.backend import register_backend


def _float32(arr) -> np.ndarray:
    """What tomopy wants, cast here rather than inside tomopy.

    tomopy before 1.15 converts its inputs with ``np.array(x, copy=False)``,
    which NumPy 2 turns into an error whenever a cast is needed (float64
    angles are the usual case). Casting up front keeps every tomopy release
    working and costs nothing when the array already is float32.
    """
    return np.ascontiguousarray(arr, dtype=np.float32)


class TomoPyBackend:
    name = "tomopy"

    @staticmethod
    def _tomopy():
        try:
            import tomopy  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The TomoPy backend requires 'tomopy'. Install with: "
                "conda install -c conda-forge tomopy"
            ) from exc
        return tomopy

    def reconstruct(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        *,
        center: float | None = None,
        algorithm: str = "gridrec",
        **kwargs,
    ) -> np.ndarray:
        tomopy = self._tomopy()
        return tomopy.recon(
            _float32(projections), _float32(angles), center=center,
            algorithm=algorithm, **kwargs
        )

    def reproject(
        self,
        volume: np.ndarray,
        angles: np.ndarray,
        *,
        center: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        tomopy = self._tomopy()
        return tomopy.project(_float32(volume), _float32(angles), center=center,
                              pad=False, **kwargs)


register_backend(TomoPyBackend())

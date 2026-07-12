"""Preprocessing for ptychographic phase projections.

Every function takes and returns a ``(n_theta, n_v, n_u)`` float array and is
individually toggleable in the GUI.

:func:`remove_phase_ramp` is the one that matters. A residual linear phase ramp
across a projection is mathematically indistinguishable from a lateral shift of
that projection, so leaving one in poisons the alignment: the loop will happily
"correct" a shift that is really a ramp, and never converge.

scipy/skimage are imported lazily so importing this module stays light.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "background_mask",
    "check_mass_positive",
    "crop",
    "invert",
    "pad",
    "remove_phase_offset",
    "remove_phase_ramp",
    "scale",
    "unwrap_phase",
]


def background_mask(shape: tuple[int, int], border: int = 8) -> np.ndarray:
    """A boolean mask selecting a border frame of width ``border`` pixels.

    This is the default "vacuum" region for ramp/offset fitting: the frame around
    the object, where the phase should be flat.
    """
    n_v, n_u = shape
    if border <= 0:
        raise ValueError("border must be >= 1 pixel")
    if 2 * border >= min(n_v, n_u):
        raise ValueError(
            f"border={border} px leaves no interior for a {n_v}x{n_u} projection"
        )
    mask = np.zeros(shape, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def _resolve_mask(prj: np.ndarray, mask: np.ndarray | None, border: int) -> np.ndarray:
    if mask is None:
        return background_mask(prj.shape[1:], border)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != prj.shape[1:]:
        raise ValueError(
            f"mask shape {mask.shape} does not match projection shape {prj.shape[1:]}"
        )
    if not mask.any():
        raise ValueError("background mask selects no pixels")
    return mask


def remove_phase_ramp(
    prj: np.ndarray, mask: np.ndarray | None = None, *, border: int = 8
) -> np.ndarray:
    """Fit and subtract a plane ``a*u + b*v + c`` per projection.

    The plane is fitted by least squares over ``mask`` (default: a border frame of
    ``border`` px, i.e. presumed-vacuum) but subtracted over the *whole* frame.
    ``mask`` is typically the rectangular ROI the user drew in the projection view.
    """
    prj = np.asarray(prj, dtype=np.float32)
    mask = _resolve_mask(prj, mask, border)

    n_v, n_u = prj.shape[1:]
    v, u = np.mgrid[0:n_v, 0:n_u]
    # Design matrix over the background pixels only; the same basis is then
    # evaluated over the full frame to subtract.
    basis_bg = np.column_stack(
        [u[mask].ravel(), v[mask].ravel(), np.ones(int(mask.sum()))]
    ).astype(np.float64)
    basis_full = np.column_stack(
        [u.ravel(), v.ravel(), np.ones(u.size)]
    ).astype(np.float64)

    out = np.empty_like(prj)
    for i, frame in enumerate(prj):
        coeffs, *_ = np.linalg.lstsq(basis_bg, frame[mask].ravel().astype(np.float64), rcond=None)
        plane = (basis_full @ coeffs).reshape(n_v, n_u)
        out[i] = frame - plane.astype(np.float32)
    return out


def remove_phase_offset(
    prj: np.ndarray, mask: np.ndarray | None = None, *, border: int = 8
) -> np.ndarray:
    """Subtract the mean over the background region, per projection.

    Worth doing even when the ramp fit is switched off: the COM code needs the
    vacuum to sit at zero or the centroid is meaningless.
    """
    prj = np.asarray(prj, dtype=np.float32)
    mask = _resolve_mask(prj, mask, border)
    offsets = prj[:, mask].mean(axis=1, dtype=np.float64)
    return (prj - offsets[:, None, None].astype(np.float32)).astype(np.float32)


def unwrap_phase(prj: np.ndarray) -> np.ndarray:
    """Unwrap each projection's phase (2-D, per frame). Off by default."""
    from skimage.restoration import unwrap_phase as _unwrap  # noqa: PLC0415

    prj = np.asarray(prj, dtype=np.float32)
    return np.stack([_unwrap(frame).astype(np.float32) for frame in prj])


def crop(prj: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to ``roi = (v0, v1, u0, u1)`` (half-open, like slicing)."""
    v0, v1, u0, u1 = roi
    if not (0 <= v0 < v1 <= prj.shape[1] and 0 <= u0 < u1 <= prj.shape[2]):
        raise ValueError(f"roi {roi} out of bounds for projections {prj.shape}")
    return np.ascontiguousarray(prj[:, v0:v1, u0:u1])


def pad(prj: np.ndarray, pad_u: int, pad_v: int) -> np.ndarray:
    """Zero-pad by ``pad_u`` columns and ``pad_v`` rows on each side.

    Pad *before* the loop: shifting moves the object toward the frame edge, and an
    object that walks out of the frame cannot be registered back.
    """
    if pad_u < 0 or pad_v < 0:
        raise ValueError("padding must be non-negative")
    return np.pad(prj, ((0, 0), (pad_v, pad_v), (pad_u, pad_u)), mode="constant")


def invert(prj: np.ndarray) -> np.ndarray:
    """Negate. Phase is negative-valued for most samples, but the COM and the
    reconstruction both assume positive mass."""
    return -np.asarray(prj, dtype=np.float32)


def scale(prj: np.ndarray, factor: float) -> np.ndarray:
    return (np.asarray(prj, dtype=np.float32) * np.float32(factor)).astype(np.float32)


def check_mass_positive(prj: np.ndarray) -> tuple[bool, float]:
    """Report whether the projection integral is positive ("mass" points the right way).

    Returns ``(ok, total)``. The GUI warns and offers :func:`invert` when ``ok`` is
    False -- a negative total means the COM centroid and the reconstruction will
    both be nonsense.
    """
    total = float(np.sum(prj, dtype=np.float64))
    return total > 0.0, total

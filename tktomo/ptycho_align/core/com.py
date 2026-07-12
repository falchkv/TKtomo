"""Centre-of-mass pre-alignment.

Phase projections are line integrals of the density, so they conserve mass, which
makes the centroid a genuinely good initialiser for the reprojection loop:

* **Vertical** (``sy``, along the rotation axis): a rigid object does not travel up
  and down as it rotates, so the mass-weighted row centroid should be *constant*
  with angle. Whatever variation is present is misalignment.
* **Horizontal** (``sx``, perpendicular to the axis): the column centroid of a
  mass-conserving projection traces a sinusoid in angle,
  ``com_u(theta) = a*sin(theta) + b*cos(theta) + c``. Fit it; the offset ``c`` *is*
  the rotation-axis position, and the residual of the fit is the misalignment.

Sign convention -- read this before "fixing" the two formulas below. ``sx``/``sy``
are the **correction to apply**, i.e. what you hand to
:func:`~tktomo.ptycho_align.core.engine.apply_shifts`, *not* the displacement the
object currently has. Those differ by a minus sign. So a projection whose centroid
sits *below* the fitted curve needs a *positive* correction:

    sy = com_v - reference        (NOT reference - com_v)
    sx = com_u - fitted_com_u     (NOT fitted - measured)

which is the opposite of the formula in the build spec (docs, section 6). The spec
is internally inconsistent here: its section 6 states the displacement, while its
section 7 registration step accumulates a correction, and the two cannot both be
right. The engine follows TomoPy's (correction) convention, and the engine test is
what pins it down -- with the spec's signs, the COM initialiser actively fights the
loop it is supposed to be warm-starting.

A bad sinusoid fit is the earliest possible warning that something upstream is
broken (usually a phase ramp or a missed offset removal), which is why
:func:`com_prealign` hands the fitted curve back for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ComResult", "com_prealign", "find_center", "projection_centroids"]


@dataclass
class ComResult:
    sx: np.ndarray  # horizontal shift per angle (pixels)
    sy: np.ndarray  # vertical shift per angle (pixels)
    center: float  # rotation-axis position in u
    com_u: np.ndarray  # measured column centroids
    com_v: np.ndarray  # measured row centroids
    fitted_u: np.ndarray  # the fitted sinusoid, for overlaying on com_u
    fit_residual: float  # RMS of (fitted_u - com_u), in pixels
    amplitude: float  # sqrt(a^2 + b^2); ~0 means a suspiciously centred object


def projection_centroids(prj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass-weighted ``(com_v, com_u)`` centroids, one per projection.

    Raises if any projection has near-zero total mass -- that means the offset
    removal is wrong (or the data is inverted), and every downstream number would
    be garbage.
    """
    prj = np.asarray(prj, dtype=np.float64)
    # Clip negatives: the centroid of a signed field is not a centre of mass, and
    # phase noise in the vacuum routinely dips below zero.
    mass = np.clip(prj, 0.0, None)
    totals = mass.sum(axis=(1, 2))

    reference = np.max(np.abs(totals)) if totals.size else 0.0
    bad = totals <= 1e-9 * max(reference, 1.0)
    if np.any(bad):
        raise ValueError(
            f"{int(bad.sum())} of {len(totals)} projections have ~zero positive mass "
            f"(first: index {int(np.argmax(bad))}). Remove the phase offset first, "
            "and check the sign -- phase is negative for most samples, so you may "
            "need 'invert'."
        )

    n_v, n_u = prj.shape[1:]
    com_v = (mass.sum(axis=2) @ np.arange(n_v)) / totals
    com_u = (mass.sum(axis=1) @ np.arange(n_u)) / totals
    return com_v, com_u


def com_prealign(
    prj: np.ndarray, angles: np.ndarray, *, vertical_reference: str = "mean"
) -> ComResult:
    """Estimate initial shifts and the rotation-axis position from the centroids.

    ``vertical_reference`` is ``"mean"`` or ``"median"`` (the latter is more robust
    to a few bad projections).
    """
    angles = np.asarray(angles, dtype=np.float64)
    com_v, com_u = projection_centroids(prj)

    if vertical_reference == "mean":
        reference = float(np.mean(com_v))
    elif vertical_reference == "median":
        reference = float(np.median(com_v))
    else:
        raise ValueError(
            f"vertical_reference must be 'mean' or 'median', got {vertical_reference!r}"
        )
    # Corrections, not displacements -- see the module docstring.
    sy = com_v - reference

    # com_u(theta) = a*sin(theta) + b*cos(theta) + c
    basis = np.column_stack([np.sin(angles), np.cos(angles), np.ones_like(angles)])
    (a, b, c), *_ = np.linalg.lstsq(basis, com_u, rcond=None)
    fitted_u = basis @ np.array([a, b, c])

    sx = com_u - fitted_u
    residual = float(np.sqrt(np.mean((fitted_u - com_u) ** 2)))

    return ComResult(
        sx=sx,
        sy=sy,
        center=float(c),
        com_u=com_u,
        com_v=com_v,
        fitted_u=fitted_u,
        fit_residual=residual,
        amplitude=float(np.hypot(a, b)),
    )


def find_center(
    prj: np.ndarray, angles: np.ndarray, *, method: str = "vo", **kwargs
) -> float:
    """Rotation-axis position via TomoPy's centre finders.

    ``method`` is ``"vo"`` (:func:`tomopy.find_center_vo`) or ``"pc"``
    (:func:`tomopy.find_center_pc`, phase correlation of the 0/180 pair).
    """
    import tomopy  # noqa: PLC0415

    if method == "vo":
        return float(tomopy.find_center_vo(prj, **kwargs))
    if method == "pc":
        # find_center_pc compares two opposing projections, not the whole stack.
        return float(tomopy.find_center_pc(prj[0], prj[-1], **kwargs))
    raise ValueError(f"Unknown centre-finding method {method!r}; use 'vo' or 'pc'.")

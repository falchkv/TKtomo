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

    def scaled(self, factor: float) -> ComResult:
        """The same result expressed on a grid ``factor`` times finer.

        Every field is in pixels, so a change of binning rescales all of them. The
        caller must do this whenever the pixel grid changes underneath a stored
        result, or the COM's numbers silently start referring to a grid that no
        longer exists: the plausibility check in :func:`center_is_plausible` would
        reject a correct centre estimate for differing from a stale reference, and
        the sinogram overlay would be drawn at the wrong scale.
        """
        return ComResult(
            sx=self.sx * factor,
            sy=self.sy * factor,
            center=self.center * factor,
            com_u=self.com_u * factor,
            com_v=self.com_v * factor,
            fitted_u=self.fitted_u * factor,
            fit_residual=self.fit_residual * factor,
            amplitude=self.amplitude * factor,
        )


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

    Caveat, learned the hard way: ``find_center_vo`` is tuned for real attenuation
    sinograms and is **unreliable on zero-padded phase projections** -- on a 104 px
    wide padded phantom it has been observed to return 96.5 or even 0.0 where the true
    axis is 52.0. Always run the answer past :func:`center_is_plausible` before using
    it; the COM fit and ``"pc"`` are usually the more trustworthy estimates here.
    """
    import tomopy  # noqa: PLC0415

    if method == "vo":
        return float(tomopy.find_center_vo(prj, **kwargs))
    if method == "pc":
        # find_center_pc compares two opposing projections, not the whole stack.
        return float(tomopy.find_center_pc(prj[0], prj[-1], **kwargs))
    raise ValueError(f"Unknown centre-finding method {method!r}; use 'vo' or 'pc'.")


def center_is_plausible(
    center: float, width: int, reference: float | None = None, *, tolerance: float = 0.1
) -> tuple[bool, str]:
    """Sanity-check a rotation-axis estimate. Returns ``(ok, reason)``.

    A centre-finder that has failed does not say so -- it returns a number, and a bad
    number silently ruins a long run. Two cheap checks catch the failures actually
    seen: an axis outside the detector is impossible, and an axis far from an
    independent estimate (the COM fit, or failing that the detector midpoint) means
    the two disagree and at least one of them is wrong.

    ``tolerance`` is a fraction of the detector width.
    """
    if not np.isfinite(center):
        return False, "the estimate is not a finite number"
    if not 0.0 < center < width:
        return False, f"{center:.2f} px lies outside the detector (0..{width} px)"

    reference = width / 2.0 if reference is None else reference
    limit = tolerance * width
    if abs(center - reference) > limit:
        return False, (
            f"{center:.2f} px differs from the expected {reference:.2f} px by more than "
            f"{limit:.1f} px ({tolerance:.0%} of the detector width)"
        )
    return True, ""

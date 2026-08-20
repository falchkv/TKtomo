"""Vertical alignment from the vertical mass distribution (Odstrcil et al. 2019).

Stage 1 of the decoupled aligner in :mod:`~tktomo.ptycho_align.core.odstrcil`.
Row-sum every projection into a one-dimensional vertical profile, then register the
profiles against a common reference to sub-pixel precision. That is the whole
algorithm. Reference: M. Odstrcil et al., *Opt. Express* **27**, 36637 (2019).

**Why this is well posed, and the horizontal direction is not.** The rotation axis is
vertical, so a rotation about it maps every voxel to another voxel *in the same
detector row*. The line integral of a projection therefore has a vertical mass
distribution ``m(v) = sum_u p(v, u)`` that is **invariant under the rotation angle**:
every projection of a rigid object must produce the same ``m(v)``, up to noise and the
misalignment we are trying to measure. Two profiles that differ do so only because the
sample moved vertically, which makes "find the vertical shift" an ordinary,
well-conditioned 1-D registration problem with a unique answer.

Nothing of the sort is true horizontally. The horizontal mass distribution changes
with angle *by design* -- that variation is the tomographic signal -- so a horizontal
shift cannot be separated from a change of viewing angle by comparing two projections.
That is why horizontal alignment has to be solved against a reconstructed volume
(stage 2) and vertical does not.

**Cost.** No forward projection, no back projection, no reconstruction. The profiles
are computed **once** from the pristine stack, and the iteration then runs entirely in
1-D profile space -- a row-sum commutes with a vertical translation, so shifting a
profile is the same as shifting the projection and re-summing it, at
``n_angles x n_rows`` instead of ``n_angles x n_rows x n_cols`` cost. A typical
900 x 1500 x 1800 stack reduces to a 900 x 1500 array, and the whole loop is
milliseconds. Compared with
one outer iteration of the horizontal stage (a full tomographic reconstruction plus a
reprojection, minutes to hours) this step is free. Run it first, always.

Three things that are easy to get wrong here:

1. **Sign.** ``sy`` is the *correction to apply*, i.e. what you hand to
   :func:`~tktomo.ptycho_align.core.engine.apply_shifts`, not the displacement the
   object currently has. ``apply_shifts`` produces ``corrected(v) = measured(v + sy)``,
   so a projection whose sample sits at a *higher* row index than the reference needs a
   *positive* ``sy``. This matches :mod:`~tktomo.ptycho_align.core.com` exactly
   (``sy = com_v - reference``), and it is the opposite of the intuitive
   "displacement" convention.

2. **Never re-shift shifted data.** The cumulative ``sy`` is always applied to the
   pristine profiles, never to the output of the previous iteration -- the same rule
   the engine follows for the projections, and for the same reason: repeated
   resampling blurs the signal away. (Here the shift is done in Fourier space and is
   exact for a band-limited profile, but the rule still holds.)

3. **Truncation.** The invariance argument assumes the sample is *entirely inside* the
   field of view vertically. If it is cut off at the top or bottom, mass enters and
   leaves the frame as the sample rotates, ``m(v)`` is no longer angle-invariant, and
   the alignment is measuring an artefact. This module detects that and says so
   (:func:`truncation_flags`, ``VerticalResult.truncation_reason``) rather than
   returning a confident wrong answer.

Profile registration versus the centroid, measured rather than asserted
-----------------------------------------------------------------------
The obvious question is why register profiles at all when
:func:`~tktomo.ptycho_align.core.com.com_prealign` already tracks the row centroid, and
the honest answer is not uniformly flattering. RMS recovery error against known shifts,
same synthetic profiles fed to both estimators (``tests/test_vertical_alignment.py``
regenerates every row):

    regime                                  centroid    profile registration
    sample fully inside the FOV              0.000 px          0.002 px
    sample cut off at one edge               0.74  px          1.16  px
    sample cut off at both edges             0.97  px          1.67  px
    fully inside, artifact in the vacuum     1.45  px          0.002 px

Two conclusions, one of them uncomfortable.

*When the sample is fully inside the frame the centroid is exact*, and cannot be beaten:
translating a compactly supported mass distribution moves its centroid by exactly the
translation, so the only error is arithmetic. Profile registration merely matches it, to
within its own sub-pixel grid. This is the regime in which the roadmap calls CoM
"defensible for the vertical direction", and the roadmap is right.

*Under truncation both estimators fail, and the correlation fails **worse***. Once the
sample is cut off, the fixed detector window is itself a strong stationary feature, and
the correlation partly locks onto the window while the object slides past it; the
centroid at least responds smoothly to the mass it can still see. So truncation is not
something this method survives -- it is something it must **detect**, which is why
:func:`truncation_flags` is not optional and why the default is to warn loudly.

What registration does buy is robustness to *localised* corruption. A centroid is a
global first moment: an unremoved artifact sitting out in the vacuum, or a bright edge
the ramp fit missed, drags it in proportion to its lever arm -- 1.45 px in the table
above -- while a correlation peak does not move at all, because the artifact is simply
not where the object's structure is. Combined with a median reference across angle,
that is the practical reason to prefer it in a pipeline where a few projections
occasionally come back wrong, which is every real pipeline.

Pure numpy plus scipy: no skimage, no tomopy, no GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TruncationError",
    "VerticalConfig",
    "VerticalIteration",
    "VerticalResult",
    "align_profiles",
    "align_vertical",
    "register_profile",
    "truncation_flags",
    "vertical_profiles",
]


class TruncationError(ValueError):
    """The sample leaves the field of view vertically, so the profile is not invariant.

    Raised only when ``VerticalConfig.on_truncation == "raise"``. The default is to
    warn and flag the affected projections, because a little edge contact is common and
    usually survivable -- but it must never pass silently.
    """


@dataclass
class VerticalConfig:
    """Parameters for the vertical mass-distribution alignment."""

    #: Sub-pixel refinement: ``"upsample"`` (upsampled DFT, precision ~1/upsample px)
    #: or ``"parabolic"`` (three-point fit around the correlation peak, ~0.05 px and
    #: biased toward integers, but ~50x cheaper). ``"upsample"`` is the default because
    #: this stage is free anyway.
    subpixel: Literal["upsample", "parabolic"] = "upsample"
    upsample: int = 100
    #: Reference profile: mean or median over the (already aligned) stack.
    reference: Literal["mean", "median"] = "median"
    max_iterations: int = 15
    #: Stop when the RMS of one iteration's update falls below this, in pixels.
    tolerance: float = 0.005
    #: Clip negatives before summing, so the profile is a genuine mass distribution.
    #: Matches :func:`~tktomo.ptycho_align.core.com.projection_centroids`. Set False
    #: if the sample legitimately has both signs and you have removed the offset.
    clip_negative: bool = True
    #: Subtract a per-projection vacuum baseline estimated from ``baseline_rows`` rows
    #: at the top and bottom, so the profile is compactly supported and the zero-padded
    #: correlation is a true linear correlation.
    subtract_baseline: bool = True
    baseline_rows: int = 8
    #: Optional Gaussian smoothing of the profile before registration, in pixels.
    #: Only useful for very noisy data; 0 disables it.
    smooth_sigma: float = 0.0
    #: Reject an update larger than this many pixels (None = no limit).
    max_shift: float | None = None
    #: Rows counted as "border" when testing for truncation.
    truncation_border: int = 3
    #: Flag a projection whose profile is still above this fraction of its own PEAK in
    #: the border rows. 2% is loose enough not to fire on a normal vacuum halo.
    truncation_tolerance: float = 0.02
    on_truncation: Literal["warn", "raise", "ignore"] = "warn"
    #: Restrict the profile to ``(v0, v1)`` detector rows. The escape hatch for a scan
    #: where the sample is cut off: pick a band that stays inside the frame at every
    #: angle and align on that.
    row_range: tuple[int, int] | None = None


@dataclass
class VerticalIteration:
    """One pass of the profile registration loop."""

    iteration: int
    rms_update: float  # RMS of this iteration's dsy, in pixels
    max_update: float
    rms_residual: float  # ||aligned - reference|| / ||reference||, dimensionless
    sy: np.ndarray  # cumulative correction after this iteration


@dataclass
class VerticalResult:
    """What :func:`align_vertical` found."""

    sy: np.ndarray  # per-projection vertical correction, in pixels, zero-mean
    converged: bool
    history: list[VerticalIteration] = field(default_factory=list)
    profiles: np.ndarray | None = None  # pristine profiles (n_angles, n_rows)
    aligned_profiles: np.ndarray | None = None
    reference_profile: np.ndarray | None = None
    truncated: np.ndarray | None = None  # bool per projection
    truncation_reason: str | None = None  # None when the FOV assumption holds
    row_offset: int = 0  # first detector row the profiles cover

    @property
    def n_iterations(self) -> int:
        return len(self.history)

    @property
    def rms_shift(self) -> float:
        return float(np.sqrt(np.mean(self.sy**2))) if self.sy.size else 0.0


# -- profiles ----------------------------------------------------------------------


def vertical_profiles(
    prj: np.ndarray,
    *,
    clip_negative: bool = True,
    subtract_baseline: bool = True,
    baseline_rows: int = 8,
    row_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """Row-sum each projection into a 1-D vertical mass profile.

    Returns ``(profiles, row_offset)`` where ``profiles`` has shape
    ``(n_angles, n_rows)`` and ``row_offset`` is the first detector row covered (so a
    caller can map a profile index back to a detector row when ``row_range`` is used).

    The baseline subtraction matters more than it looks: a residual vacuum offset adds
    a constant ``n_cols * offset`` to every profile sample, i.e. a large pedestal under
    a small feature, and a pedestal dominates a cross-correlation. Subtracting the
    level measured at the frame edges, then clipping, leaves a compactly supported
    profile whose correlation peak is set by the sample and not by the frame.
    """
    prj = np.asarray(prj)
    if prj.ndim != 3:
        raise ValueError(f"projections must be 3-D (n_angles, n_rows, n_cols); got {prj.shape}")

    if row_range is not None:
        v0, v1 = int(row_range[0]), int(row_range[1])
        if not 0 <= v0 < v1 <= prj.shape[1]:
            raise ValueError(f"row_range {row_range} out of bounds for {prj.shape[1]} rows")
        prj = prj[:, v0:v1, :]
    else:
        v0 = 0

    field_ = np.asarray(prj, dtype=np.float64)
    if clip_negative:
        field_ = np.clip(field_, 0.0, None)

    profiles = field_.sum(axis=2)

    if subtract_baseline:
        rows = profiles.shape[1]
        border = max(1, min(int(baseline_rows), rows // 4))
        edge = np.concatenate([profiles[:, :border], profiles[:, -border:]], axis=1)
        profiles = profiles - np.median(edge, axis=1, keepdims=True)
        if clip_negative:
            profiles = np.clip(profiles, 0.0, None)

    return profiles, v0


def truncation_flags(
    profiles: np.ndarray,
    *,
    border: int = 3,
    tolerance: float = 0.02,
) -> tuple[np.ndarray, str | None]:
    """Which projections have sample mass pressed against the top/bottom frame edge.

    Returns ``(flags, reason)``: a boolean array per projection, and a human-readable
    explanation when any flag is set (``None`` when the field-of-view assumption holds).

    The test is the profile's **level at the frame edge relative to its own peak**: if
    ``m(v)`` is still appreciably non-zero in the outermost ``border`` rows, the sample
    is not contained and mass will cross the boundary as it rotates. The obvious
    alternative -- the *fraction of total mass* in the border rows -- was tried first and
    is not sensitive enough: a broad, slowly varying profile can be losing its whole tail
    off the edge while any individual border row still holds well under 2% of the total.
    The edge-level test catches that case at 41% where the mass-fraction test read 1.2%.

    The reason string also carries the mass-conservation number. A rigid object fully
    inside a parallel-beam field of view has an angle-independent total ``sum_v m(v)``;
    a total that drifts with angle is independent corroboration that mass is leaving the
    frame. It is reported rather than used as a trigger on its own, because dose
    fluctuation and normalisation errors move it too.

    This is the honest-failure gate for the whole module. The invariance of ``m(v)``
    under rotation only holds while the sample is fully inside the frame; if it is cut
    off, mass crosses the frame edge as the sample turns and the profiles differ for a
    reason that has nothing to do with misalignment. Registering them anyway produces a
    confident number that is wrong, and nothing downstream can tell.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    rows = profiles.shape[1]
    border = max(1, min(int(border), rows // 3))

    mass = np.clip(profiles, 0.0, None)
    peak = mass.max(axis=1)
    peak = np.where(peak > 0.0, peak, np.inf)  # a dead profile cannot be truncated

    top = mass[:, :border].mean(axis=1) / peak
    bottom = mass[:, -border:].mean(axis=1) / peak
    edge = np.maximum(top, bottom)

    flags = edge > tolerance
    if not flags.any():
        return flags, None

    total = mass.sum(axis=1)
    reference = float(np.median(total))
    drift = float(np.max(np.abs(total - reference)) / reference) if reference > 0 else float("nan")

    worst = int(np.argmax(edge))
    reason = (
        f"{int(flags.sum())} of {len(flags)} projections still carry "
        f"{edge[worst]:.0%} of their peak vertical mass in the outermost {border} "
        f"detector row(s) (worst: projection {worst}); the total projected mass varies "
        f"by {drift:.1%} across the scan. The sample is not fully inside the field of "
        "view vertically, so the vertical mass distribution is NOT invariant under "
        "rotation and this alignment is measuring truncation as well as misalignment. "
        "Either crop to a row band that stays inside the frame at every angle "
        "(VerticalConfig.row_range) or treat the result as an estimate only."
    )
    return flags, reason


# -- 1-D registration --------------------------------------------------------------


def _padded_spectrum(profiles: np.ndarray, m: int) -> np.ndarray:
    """FFT of zero-padded profiles, so the circular correlation is a linear one."""
    padded = np.zeros(profiles.shape[:-1] + (m,), dtype=np.float64)
    padded[..., : profiles.shape[-1]] = profiles
    return np.fft.fft(padded, axis=-1)


def _coarse_lag(cc: np.ndarray, m: int, max_shift: float | None) -> int:
    """Integer lag of the correlation peak, unwrapped to a signed value."""
    if max_shift is not None:
        limit = int(np.ceil(abs(max_shift)))
        allowed = np.zeros(m, dtype=bool)
        allowed[: limit + 1] = True
        allowed[m - limit :] = True
        cc = np.where(allowed, cc, -np.inf)
    index = int(np.argmax(cc))
    return index - m if index > m // 2 else index


def _refine_lag(
    spectrum: np.ndarray, m: int, lag: int, upsample: int, half_width: float = 1.0
) -> float:
    """Sub-pixel peak by evaluating the correlation on a fine grid of fractional lags.

    This is the 1-D case of the upsampled-DFT refinement of Guizar-Sicairos et al.,
    *Opt. Lett.* **33**, 156 (2008) -- the same algorithm skimage's
    ``phase_cross_correlation(upsample_factor=...)`` uses, written out here so the core
    keeps its numpy+scipy-only dependency budget. Because it evaluates the exact
    band-limited interpolant rather than fitting a parabola to three samples, it has no
    peak-locking bias toward integer lags.
    """
    if upsample <= 1:
        return float(lag)
    step = 1.0 / float(upsample)
    lags = lag + np.arange(-half_width, half_width + 0.5 * step, step)
    freqs = np.fft.fftfreq(m)
    kernel = np.exp(2j * np.pi * np.outer(lags, freqs))
    fine = (kernel @ spectrum).real / m
    return float(lags[int(np.argmax(fine))])


def _parabolic_lag(cc: np.ndarray, m: int, lag: int) -> float:
    """Three-point parabolic refinement of an integer correlation peak."""
    i = lag % m
    centre = cc[i]
    left = cc[(i - 1) % m]
    right = cc[(i + 1) % m]
    curvature = left - 2.0 * centre + right
    if curvature >= 0.0 or not np.isfinite(curvature):
        return float(lag)  # not a maximum; refuse to invent precision
    delta = 0.5 * (left - right) / curvature
    return float(lag + np.clip(delta, -0.5, 0.5))


def register_profile(
    reference: np.ndarray,
    profile: np.ndarray,
    *,
    upsample: int = 100,
    subpixel: Literal["upsample", "parabolic"] = "upsample",
    max_shift: float | None = None,
) -> float:
    """Sub-pixel lag ``d`` such that ``profile(v) ~ reference(v - d)``.

    Same sign convention as
    ``skimage.registration.phase_cross_correlation(reference, profile)`` and as the
    engine's registration step: hand ``(measured, simulated)`` and the returned value
    is the correction to *add* to the cumulative shift.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    profile = np.asarray(profile, dtype=np.float64).ravel()
    if reference.shape != profile.shape:
        raise ValueError(f"profile lengths differ: {profile.shape} vs {reference.shape}")

    n = reference.size
    m = 2 * n  # zero padding -> linear (not circular) correlation
    spectrum = _padded_spectrum(profile, m) * np.conj(_padded_spectrum(reference, m))
    spectrum[0] = 0.0  # kill the DC cross-term: it is a pedestal, not a peak
    cc = np.fft.ifft(spectrum).real / m

    lag = _coarse_lag(cc, m, max_shift)
    if subpixel == "parabolic":
        return _parabolic_lag(cc, m, lag)
    if subpixel == "upsample":
        return _refine_lag(spectrum, m, lag, upsample)
    raise ValueError(f"subpixel must be 'upsample' or 'parabolic', got {subpixel!r}")


def _shift_profiles(profiles: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """``out[i](v) = profiles[i](v + sy[i])``, by Fourier phase ramp on a padded copy.

    Applied to the *pristine* profiles every iteration (convention 3). Fourier shifting
    is exact for a band-limited signal, so repeating it does not accumulate
    interpolation error -- but the pristine-input rule is kept anyway, because it is
    what makes the cumulative ``sy`` the single source of truth.
    """
    n = profiles.shape[1]
    m = 2 * n
    padded = np.zeros((profiles.shape[0], m), dtype=np.float64)
    padded[:, :n] = profiles
    freqs = np.fft.rfftfreq(m)
    spectrum = np.fft.rfft(padded, axis=1)
    # out(v) = in(v - s) multiplies the spectrum by exp(-2i pi f s); we want s = -sy.
    spectrum *= np.exp(2j * np.pi * np.outer(np.asarray(sy, dtype=np.float64), freqs))
    return np.fft.irfft(spectrum, m, axis=1)[:, :n]


# -- the loop ----------------------------------------------------------------------


def align_profiles(
    profiles: np.ndarray, config: VerticalConfig | None = None
) -> VerticalResult:
    """Register 1-D vertical profiles to a common reference, iterating to convergence.

    Split out from :func:`align_vertical` so the expensive part (touching the full
    stack) happens exactly once and the loop can be re-run, plotted or unit-tested on
    profiles alone.
    """
    config = VerticalConfig() if config is None else config
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError(f"profiles must be 2-D (n_angles, n_rows); got {profiles.shape}")

    n_angles = profiles.shape[0]
    sy = np.zeros(n_angles, dtype=np.float64)
    history: list[VerticalIteration] = []
    converged = False

    if n_angles < 2:
        return VerticalResult(sy=sy, converged=True, profiles=profiles)

    work = profiles
    if config.smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter1d  # noqa: PLC0415

        work = gaussian_filter1d(profiles, config.smooth_sigma, axis=1, mode="nearest")

    if config.reference not in ("median", "mean"):
        raise ValueError(f"reference must be 'mean' or 'median', got {config.reference!r}")
    reduce = np.median if config.reference == "median" else np.mean

    aligned = work
    reference = np.asarray(reduce(aligned, axis=0))
    for iteration in range(1, int(config.max_iterations) + 1):
        aligned = _shift_profiles(work, sy)
        reference = np.asarray(reduce(aligned, axis=0))

        update = np.array(
            [
                register_profile(
                    reference,
                    aligned[i],
                    upsample=config.upsample,
                    subpixel=config.subpixel,
                    max_shift=config.max_shift,
                )
                for i in range(n_angles)
            ]
        )
        # A constant vertical shift translates the whole reconstructed volume along the
        # rotation axis; it is not a misalignment and, left alone, it random-walks.
        update -= update.mean()
        sy = sy + update

        norm = float(np.linalg.norm(reference)) * np.sqrt(n_angles)
        # np.subtract, not `aligned - reference`: see the note in odstrcil.step() --
        # numpy's temp elision is broken on CPython 3.14 and `norm(a - b)` can write the
        # difference into `a`.
        scatter = np.subtract(aligned, reference)
        residual = float(np.linalg.norm(scatter) / norm) if norm else float("nan")
        rms = float(np.sqrt(np.mean(update**2)))
        history.append(
            VerticalIteration(
                iteration=iteration,
                rms_update=rms,
                max_update=float(np.max(np.abs(update))) if update.size else 0.0,
                rms_residual=residual,
                sy=sy.copy(),
            )
        )
        if rms < config.tolerance:
            converged = True
            break

    sy = sy - sy.mean()
    aligned = _shift_profiles(work, sy)
    reference = np.asarray(reduce(aligned, axis=0))

    if not converged:
        logger.warning(
            "Vertical profile alignment did not converge in %d iterations "
            "(last update %.4f px, tolerance %.4f px).",
            len(history),
            history[-1].rms_update if history else float("nan"),
            config.tolerance,
        )

    return VerticalResult(
        sy=sy,
        converged=converged,
        history=history,
        profiles=profiles,
        aligned_profiles=aligned,
        reference_profile=reference,
    )


def align_vertical(
    prj: np.ndarray, config: VerticalConfig | None = None
) -> VerticalResult:
    """Stage 1: per-projection vertical correction from the vertical mass distribution.

    ``prj`` is the preprocessed stack ``(n_angles, n_rows, n_cols)`` -- ramp and offset
    already removed (roadmap step 0), because a residual offset puts a large pedestal
    under every profile and a residual ramp tilts it.

    Costs one pass over the stack to build the profiles, then runs entirely in 1-D.
    No reconstruction, no reprojection.
    """
    config = VerticalConfig() if config is None else config

    profiles, row_offset = vertical_profiles(
        prj,
        clip_negative=config.clip_negative,
        subtract_baseline=config.subtract_baseline,
        baseline_rows=config.baseline_rows,
        row_range=config.row_range,
    )

    if config.on_truncation not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"on_truncation must be 'warn', 'raise' or 'ignore', got {config.on_truncation!r}"
        )

    flags, reason = truncation_flags(
        profiles, border=config.truncation_border, tolerance=config.truncation_tolerance
    )
    if reason is not None:
        if config.on_truncation == "raise":
            raise TruncationError(reason)
        if config.on_truncation == "warn":
            logger.warning("%s", reason)

    result = align_profiles(profiles, config)
    result.truncated = flags
    result.truncation_reason = reason
    result.row_offset = row_offset

    logger.info(
        "vertical mass-distribution alignment: %d iteration(s), %s, "
        "RMS shift %.3f px, max |shift| %.3f px",
        result.n_iterations,
        "converged" if result.converged else "NOT converged",
        result.rms_shift,
        float(np.max(np.abs(result.sy))) if result.sy.size else 0.0,
    )
    return result

"""The probes: one executable test per row of the artifact-to-cause table.

Every probe is independently callable, returns a
:class:`~tktomo.diagnostics.artifacts.ProbeResult`, and is honest about its own
preconditions -- several need a reconstruction, a vacuum border, or a decent angular
span, and a probe that lacks one returns ``NOT_APPLICABLE`` with the reason rather
than a fabricated score. :func:`diagnose` runs them all; :func:`triage` runs them in
the roadmap's prescribed order and stops at the first thing that fires.

Four conventions, each of which is a bug if you get it wrong:

1. **Projections are ``(n_theta, n_v, n_u)``; v is the rotation-axis (vertical)
   direction, u the detector's horizontal direction.** Every shift and every centre in
   this module is in *pixels of that grid*, never in physical units.

2. **Mass points up.** Ptychographic phase is NEGATIVE inside material and near zero in
   air, while every moment here -- the centroid, the second moment, the vertical mass
   profile -- assumes positive mass. :func:`stack_moments` therefore checks the stack's
   total and flips the sign if it is negative, recording ``inverted`` in the result. It
   does not guess per projection, and it subtracts a noise floor before clipping;
   see its docstring, both halves are load-bearing.

3. **Angles: radians internally, and the units of the input are resolved once.** The
   repo's ``ProjectionData.angles`` is in radians; DXchange files store degrees. Pass
   ``theta_units="rad"`` or ``"deg"`` to be explicit; ``"auto"`` calls anything with a
   span above 2*pi degrees, and records the decision in the verdict context.

4. **Registration direction.** :func:`_register_1d` returns ``s`` such that
   ``moving(u) ~= reference(u - s)``: positive ``s`` means the moving profile sits
   further along ``u`` than the reference. That is the *displacement*, so the
   correction you would hand to the alignment engine is its negative -- the opposite
   sign to :mod:`tktomo.ptycho_align.core.com`, which reports corrections. This module
   reports what it measured, not what to do about it.

The reconstruction-based probes carry their own minimal filtered backprojection
(:func:`fbp_slice` / :func:`forward_project_slice`, numpy + ``scipy.ndimage.rotate``
only, imported lazily). That is deliberate: a diagnostic that only runs when
astra/tomopy is installed is a diagnostic nobody runs. Its geometry convention matches
``skimage.transform.radon`` (``p(theta, u) = sum_rows rotate(image, -theta)``), it is
its own adjoint pair, and it is meant for one small binned slice -- not for producing
a volume anybody keeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tktomo.diagnostics.artifacts import (
    DiagnosticConfig,
    FailureMode,
    Finding,
    ProbeResult,
    ProbeStatus,
    TriageStage,
    Verdict,
    apply_stage_discount,
    confidence_from_ratio,
    rank,
)

__all__ = [
    "PROBES",
    "StackMoments",
    "diagnose",
    "fbp_slice",
    "forward_project_slice",
    "probe_angle_readback",
    "probe_angular_coverage",
    "probe_axis_tilt",
    "probe_center_consistency",
    "probe_center_sweep",
    "probe_deformation",
    "probe_scale_drift",
    "probe_shift_jitter",
    "probe_truncation",
    "probe_vacuum_phase",
    "probe_vertical_drift",
    "stack_moments",
    "triage",
]


# --------------------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------------------


def _check_stack(projections: Any) -> tuple[int, int, int]:
    """Validate shape without materialising the array (h5py datasets pass through)."""
    shape = getattr(projections, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(
            f"projections must be a 3-D (n_theta, n_v, n_u) array, got shape {shape!r}"
        )
    if min(shape) < 2:
        raise ValueError(f"projections {shape} is degenerate; every axis needs >= 2 samples")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _theta_radians(theta: Any, n_theta: int, units: str = "auto") -> tuple[np.ndarray, str]:
    """Return ``(theta_rad, units_used)``.

    ``"auto"`` calls it degrees when the span exceeds 2*pi, which is unambiguous for
    anything but a scan whose *total* range is under 6.3 degrees. Such a scan is a
    missing-wedge catastrophe anyway, and the caller is told which way it was read.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.size != n_theta:
        raise ValueError(f"theta has {theta.size} entries but there are {n_theta} projections")
    if units == "auto":
        units = "deg" if float(np.ptp(theta)) > 2.0 * np.pi else "rad"
    if units == "deg":
        return np.deg2rad(theta), "deg"
    if units == "rad":
        return theta.copy(), "rad"
    raise ValueError(f"theta_units must be 'auto', 'rad' or 'deg', got {units!r}")


@dataclass(frozen=True)
class StackMoments:
    """Two chunked passes over the stack; everything the cheap probes need.

    Computing these once and handing them to the probes is what makes a full
    :func:`diagnose` cost one read of the data rather than eight. Each probe still
    computes them itself when called alone, so they stay independently usable.

    ``uprofile`` is the *signed* mean over v (truncation and ramp signatures live in
    the sign), while ``uprofile_mass``, ``vprofile`` and every moment are computed on
    the mass-clipped stack -- the centroid of a signed field is not a centre of mass,
    and phase noise in the vacuum routinely dips below zero.
    """

    n_theta: int
    n_v: int
    n_u: int
    inverted: bool
    mass: np.ndarray  # (n_theta,) total positive mass
    com_u: np.ndarray  # (n_theta,) column centroid, px
    com_v: np.ndarray  # (n_theta,) row centroid, px
    var_u: np.ndarray  # (n_theta,) second central moment along u, px^2
    uprofile: np.ndarray  # (n_theta, n_u) signed mean over v
    uprofile_mass: np.ndarray  # (n_theta, n_u) clipped mean over v
    vprofile: np.ndarray  # (n_theta, n_v) clipped mean over u
    band_uprofile: np.ndarray  # (n_theta, n_bands, n_u) clipped, for the arc test
    band_center_v: np.ndarray  # (n_bands,) geometric band centre row index
    band_z: np.ndarray  # (n_bands,) MASS-WEIGHTED mean row of each band
    band_rows: np.ndarray  # (n_bands, 2) half-open row ranges
    contrast: float  # robust 1-99 percentile spread of the interior, in phase units
    noise_floor: float  # threshold subtracted before clipping, in phase units


def stack_moments(
    projections: Any,
    *,
    chunk: int = 32,
    n_bands: int = 3,
    border: int = 8,
    noise_sigma: float = 3.0,
) -> StackMoments:
    """Marginals and moments of every projection, in two chunked passes.

    ``chunk`` projections are pulled into memory at a time, so this works on an
    ``h5py`` dataset or a memmap without loading the whole stack -- which matters, since
    a 900 x 1500 x 1800 float32 stack is around 10 GB.

    Two things happen before any mass is measured, and both are load-bearing:

    1. **Sign.** Ptychographic phase is negative inside material, and every moment here
       assumes positive mass, so the stack's total decides a global sign flip
       (recorded in ``inverted``). Global, never per projection.
    2. **Noise floor.** The mass is ``clip(projection - floor, 0)`` with
       ``floor = median(border) + noise_sigma * MAD(border)``. Clipping at zero instead
       -- the obvious thing -- turns detector noise into a uniform positive pedestal
       across the whole frame, and on a 64 px frame at 2% noise that pedestal carries
       enough "mass" to move the second moment toward the frame's own variance and
       manufacture a magnification drift at 0.31 confidence. Measured. The border also
       supplies the vacuum level, so this removes a constant phase offset in passing.
    """
    n_theta, n_v, n_u = _check_stack(projections)
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")

    uprofile = np.zeros((n_theta, n_u))
    uprofile_mass = np.zeros((n_theta, n_u))
    vprofile = np.zeros((n_theta, n_v))

    u = np.arange(n_u, dtype=np.float64)
    v = np.arange(n_v, dtype=np.float64)

    # Imported here, not at module scope: pulling in tktomo.ptycho_align.core costs the
    # whole alignment package, and `import tktomo.diagnostics` must stay numpy-only.
    from tktomo.ptycho_align.core.preprocess import background_mask  # noqa: PLC0415

    use_border = border > 0 and 2 * border < min(n_v, n_u)
    frame = background_mask((n_v, n_u), border) if use_border else None

    total = 0.0
    row_sum = np.zeros(n_v)
    border_samples: list[np.ndarray] = []
    for i0 in range(0, n_theta, chunk):
        i1 = min(i0 + chunk, n_theta)
        block = np.asarray(projections[i0:i1], dtype=np.float64)
        total += float(block.sum())
        row_sum += block.sum(axis=(0, 2))
        if frame is not None:
            border_samples.append(block[:, frame].ravel()[::13])
    inverted = total < 0.0
    sign = -1.0 if inverted else 1.0

    if border_samples:
        edge = sign * np.concatenate(border_samples)
        median = float(np.median(edge))
        mad = 1.4826 * float(np.median(np.abs(edge - median)))
        floor = median + noise_sigma * mad
    else:
        floor = 0.0

    # Bands are laid over the OBJECT's row support, not over the detector. A sample that
    # fills half the frame -- the normal case on a tall detector -- leaves the bottom of
    # three equal detector bands empty, and the arc test then correctly refuses to run,
    # which is honest and useless. Three bands across the support is the same test with
    # a shorter lever arm, and it runs.
    profile = sign * row_sum / (n_theta * n_u) - floor
    peak = float(profile.max())
    support = np.flatnonzero(profile > 0.05 * peak) if peak > 0 else np.arange(n_v)
    first, last = (
        (int(support[0]), int(support[-1]) + 1) if support.size >= n_bands else (0, n_v)
    )
    edges = np.linspace(first, last, n_bands + 1).round().astype(int)
    band_rows = np.stack([edges[:-1], edges[1:]], axis=1)
    if np.any(band_rows[:, 1] <= band_rows[:, 0]):
        edges = np.linspace(0, n_v, n_bands + 1).round().astype(int)
        band_rows = np.stack([edges[:-1], edges[1:]], axis=1)
    if np.any(band_rows[:, 1] <= band_rows[:, 0]):
        raise ValueError(f"{n_bands} bands do not fit into {n_v} detector rows")
    band_center_v = band_rows.mean(axis=1) - 0.5
    band_uprofile = np.zeros((n_theta, n_bands, n_u))

    interior = []
    for i0 in range(0, n_theta, chunk):
        i1 = min(i0 + chunk, n_theta)
        block = sign * np.asarray(projections[i0:i1], dtype=np.float64)
        clipped = np.clip(block - floor, 0.0, None)
        uprofile[i0:i1] = block.mean(axis=1)
        uprofile_mass[i0:i1] = clipped.mean(axis=1)
        vprofile[i0:i1] = clipped.mean(axis=2)
        for b, (r0, r1) in enumerate(band_rows):
            band_uprofile[i0:i1, b] = clipped[:, r0:r1].mean(axis=1)
        if use_border:
            interior.append(block[:, border:-border, border:-border].ravel()[::97])
        else:
            interior.append(block.ravel()[::97])

    mass = uprofile_mass.sum(axis=1)
    mass_v = vprofile.sum(axis=1)
    safe_u = np.where(mass > 0, mass, 1.0)
    safe_v = np.where(mass_v > 0, mass_v, 1.0)
    com_u = (uprofile_mass @ u) / safe_u
    com_v = (vprofile @ v) / safe_v
    var_u = (uprofile_mass @ (u**2)) / safe_u - com_u**2
    # A projection with no positive mass has no centroid and no second moment. Say so
    # with NaN rather than returning a plausible-looking zero: every probe here filters
    # on np.isfinite, so a NaN drops the projection instead of poisoning a fit.
    bad = mass <= 0
    com_u[bad] = np.nan
    var_u[bad] = np.nan
    com_v[mass_v <= 0] = np.nan

    # The arc test regresses against band HEIGHT, and the height that matters is where
    # the band's mass actually is, not the middle of the row range: an object that fills
    # only the inner half of the top band sits several pixels lower than the band centre,
    # and using the centre shrinks the fitted tilt by that ratio.
    mean_vprofile = vprofile.mean(axis=0)
    rows = np.arange(n_v, dtype=np.float64)
    band_z = np.empty(n_bands)
    for b, (r0, r1) in enumerate(band_rows):
        w = mean_vprofile[r0:r1]
        total_w = w.sum()
        band_z[b] = (w @ rows[r0:r1]) / total_w if total_w > 0 else band_center_v[b]

    sample = np.concatenate(interior) if interior else np.zeros(1)
    p_lo, p_hi = np.percentile(sample, [1.0, 99.0])
    contrast = float(p_hi - p_lo)

    return StackMoments(
        n_theta=n_theta,
        n_v=n_v,
        n_u=n_u,
        inverted=inverted,
        mass=mass,
        com_u=com_u,
        com_v=com_v,
        var_u=var_u,
        uprofile=uprofile,
        uprofile_mass=uprofile_mass,
        vprofile=vprofile,
        band_uprofile=band_uprofile,
        band_center_v=band_center_v,
        band_z=band_z,
        band_rows=band_rows,
        contrast=contrast,
        noise_floor=float(floor),
    )


def _moments(projections: Any, moments: StackMoments | None, **kwargs) -> StackMoments:
    if moments is not None:
        return moments
    return stack_moments(projections, **kwargs)


# --------------------------------------------------------------------------------------
# small numerical helpers
# --------------------------------------------------------------------------------------


def _register_1d(
    reference: np.ndarray, moving: np.ndarray, *, upsample: int = 20, gradient: bool = True
) -> float:
    """Shift ``s`` with ``moving(u) ~= reference(u - s)``, on the profiles' GRADIENTS.

    Registering the derivative rather than the profile is the roadmap's central trick in
    one dimension, and it is not a refinement -- it is what makes this function usable.
    Differentiating sends a constant offset to zero and a linear ramp to a constant, the
    two ambiguities ptychographic phase retrieval leaves behind; and a flat-topped mass
    profile, whose plain correlation peak is broad and flat, becomes a pair of sharp
    spikes whose correlation peak is not. Measured on the synthetic phantom's vertical
    mass profile, at shifts of 0.25 to 3 px:

        plain correlation   0.15 - 0.35 px biased LOW, and the bias varies with shift
        gradient            exact to the 1/upsample grid, and still exact at 2% noise

    A 0.3 px bias is the whole error budget of a 1/3-voxel alignment target, so the plain
    version is not an option. ``gradient=False`` is kept for the pathological case of a
    profile so noisy that differentiating destroys it; it has never been the right choice
    here.

    Sub-pixel resolution comes from zero-padding the cross-power spectrum (sinc
    interpolation of the correlation), never from a parabola through three samples: the
    parabola is biased on a sinc-interpolated peak and measurably made things worse.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    moving = np.asarray(moving, dtype=np.float64).ravel()
    if reference.shape != moving.shape:
        raise ValueError("profiles must have the same length")
    n = reference.size
    if gradient and n >= 5:
        kernel = np.ones(3) / 3.0
        reference = np.gradient(np.convolve(reference, kernel, mode="same"))
        moving = np.gradient(np.convolve(moving, kernel, mode="same"))
    ref = reference - reference.mean()
    mov = moving - moving.mean()
    if not np.any(ref) or not np.any(mov):
        return float("nan")
    pad = 2 * n
    spectrum = np.fft.rfft(mov, pad) * np.conj(np.fft.rfft(ref, pad))
    fine = np.fft.irfft(spectrum, pad * upsample)
    size = fine.size
    k = int(np.argmax(fine))
    if k > size // 2:
        k -= size
    return float(k) / upsample


def _fit_sinusoid(theta: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Least-squares ``y = a sin(theta) + b cos(theta) + c``; returns ``(a, b, c, fitted)``.

    Exact model, not an approximation: the column centroid of a parallel-beam
    projection of a *rigid* object traces precisely this curve, with ``c`` the
    rotation-axis position. Any residual is misalignment, truncation, or a wrong angle.
    """
    basis = np.column_stack([np.sin(theta), np.cos(theta), np.ones_like(theta)])
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), basis @ coeffs


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation in acquisition order: white ~ 0, smooth drift ~ 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x[:-1], x[1:]) / denom)


def _split_trend(x: np.ndarray, degree: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Split a series into a smooth polynomial trend (in index) and a white residual."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(x.size, dtype=np.float64)
    good = np.isfinite(x)
    trend = np.full_like(x, np.nan)
    if good.sum() <= degree + 1:
        return trend, np.full_like(x, np.nan)
    coeffs = np.polyfit(idx[good], x[good], degree)
    trend = np.polyval(coeffs, idx)
    return trend, x - trend


def _span_deg(theta_rad: np.ndarray) -> float:
    return float(np.rad2deg(np.ptp(theta_rad)))


def _nan_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def _clear(
    probe: str, stage: TriageStage, metrics: Mapping[str, float], detail: str = "", **kw
) -> ProbeResult:
    return ProbeResult(
        probe=probe, stage=stage, status=ProbeStatus.CLEAR, metrics=dict(metrics),
        detail=detail, **kw
    )


def _na(probe: str, stage: TriageStage, reason: str, **kw) -> ProbeResult:
    return ProbeResult(
        probe=probe, stage=stage, status=ProbeStatus.NOT_APPLICABLE, reason=reason, **kw
    )


# --------------------------------------------------------------------------------------
# a minimal, dependency-light single-slice reconstructor
# --------------------------------------------------------------------------------------


def _object_rim_px(moments: StackMoments, center: float) -> float:
    """Distance from the rotation axis to the object's outermost material, in pixels.

    Modes 7 and 8 both smear the reconstruction in proportion to this distance -- a
    tangential smear ``(g-1) theta r`` for an angle error, a radial one ``ds/s * r`` for a
    magnification drift -- so both probes must measure it the same way or their findings
    are not comparable, and they get compared. The radius of gyration will not do: it is
    half the radius for a disc, and it ignores an object that sits off the axis.
    """
    profile = moments.uprofile_mass.mean(axis=0)
    peak = float(profile.max())
    if peak <= 0:
        return float("nan")
    support = np.flatnonzero(profile > 0.02 * peak)
    if support.size == 0:
        return float("nan")
    return float(max(abs(support[0] - center), abs(support[-1] - center)))


def _reference_row(moments: StackMoments) -> int:
    """The detector row to reconstruct when only one slice is affordable.

    The object's mass-weighted centre row, NOT the heaviest row. With a tilted axis the
    apparent rotation centre depends on height, so an arbitrary "heaviest" row reports a
    centre offset that is real for that row and misleading for the volume; the mid-plane
    is the one height every other estimate here is referred to.
    """
    mean_com_v = float(np.nanmean(moments.com_v)) if np.isfinite(moments.com_v).any() else np.nan
    if not math.isfinite(mean_com_v):
        return int(np.argmax(np.nanmean(moments.vprofile, axis=0)))
    return int(np.clip(round(mean_com_v), 0, moments.n_v - 1))


def _ramp_filter(sino: np.ndarray, window: str = "hann") -> np.ndarray:
    """Ram-Lak filter along u, optionally apodised. ``sino`` is ``(n_theta, n_u)``."""
    n_u = sino.shape[1]
    size = int(2 ** math.ceil(math.log2(max(64, 2 * n_u))))
    freqs = np.fft.rfftfreq(size)
    ramp = 2.0 * freqs
    if window == "hann":
        ramp = ramp * (0.5 + 0.5 * np.cos(np.pi * freqs / freqs[-1]))
    elif window not in ("ramlak", "none"):
        raise ValueError(f"unknown filter window {window!r}; use 'hann', 'ramlak' or 'none'")
    spectrum = np.fft.rfft(sino, size, axis=1) * ramp
    return np.fft.irfft(spectrum, size, axis=1)[:, :n_u]


def forward_project_slice(
    image: np.ndarray, theta: np.ndarray, *, center: float | None = None, n_u: int | None = None
) -> np.ndarray:
    """Parallel-beam forward projection of one square slice. ``(n_theta, n_u)``.

    Convention matches ``skimage.transform.radon``: ``p(theta, u)`` is the column sum
    of the image rotated by ``-theta``. ``center`` is where the image's rotation centre
    lands on the detector (default: the detector midpoint).
    """
    from scipy.ndimage import rotate, shift as ndi_shift  # noqa: PLC0415

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2-D, got {image.shape}")
    size = image.shape[0]
    n_u = size if n_u is None else int(n_u)
    center = (n_u - 1) / 2.0 if center is None else float(center)
    theta_deg = np.rad2deg(np.asarray(theta, dtype=np.float64).ravel())

    out = np.zeros((theta_deg.size, n_u))
    lo = (n_u - size) // 2
    for i, angle in enumerate(theta_deg):
        line = rotate(image, -angle, reshape=False, order=1, prefilter=False).sum(axis=0)
        if n_u >= size:
            out[i, lo : lo + size] = line
        else:
            out[i] = line[-lo : -lo + n_u]
    delta = center - (n_u - 1) / 2.0
    if abs(delta) > 1e-9:
        out = ndi_shift(out, (0.0, delta), order=1, mode="nearest")
    return out


def fbp_slice(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    center: float | None = None,
    window: str = "hann",
) -> np.ndarray:
    """Filtered backprojection of one sinogram ``(n_theta, n_u)`` -> ``(n_u, n_u)``.

    The adjoint of :func:`forward_project_slice` by construction (broadcast along rows,
    rotate by ``+theta``, accumulate), so reprojection residuals computed with the pair
    mean something. Not a replacement for a real reconstruction engine: no
    regularisation, no negativity handling, one slice at a time.
    """
    from scipy.ndimage import rotate, shift as ndi_shift  # noqa: PLC0415

    sinogram = np.asarray(sinogram, dtype=np.float64)
    if sinogram.ndim != 2:
        raise ValueError(f"sinogram must be 2-D (n_theta, n_u), got {sinogram.shape}")
    n_theta, n_u = sinogram.shape
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.size != n_theta:
        raise ValueError(f"theta has {theta.size} entries for {n_theta} rows of sinogram")
    center = (n_u - 1) / 2.0 if center is None else float(center)

    delta = (n_u - 1) / 2.0 - center
    if abs(delta) > 1e-9:
        sinogram = ndi_shift(sinogram, (0.0, delta), order=1, mode="nearest")
    filtered = _ramp_filter(sinogram, window)

    recon = np.zeros((n_u, n_u))
    theta_deg = np.rad2deg(theta)
    for i, angle in enumerate(theta_deg):
        smeared = np.tile(filtered[i], (n_u, 1))
        recon += rotate(smeared, angle, reshape=False, order=1, prefilter=False)
    return recon * (np.pi / max(n_theta, 1))


def _circle_mask(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    r = (size - 1) / 2.0
    return ((x - r) ** 2 + (y - r) ** 2) <= (0.95 * r) ** 2


def _sharpness(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """``(entropy, gradient_energy)`` inside ``mask``. Entropy is minimised, gradient
    energy maximised, at the correct rotation centre."""
    values = image[mask]
    shifted = values - values.min()
    total = shifted.sum()
    if total <= 0:
        return float("inf"), 0.0
    p = shifted / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    gy, gx = np.gradient(image)
    grad = float(np.mean((gx**2 + gy**2)[mask]))
    return entropy, grad


def _prepare_slice_sinogram(
    projections: Any,
    theta: np.ndarray,
    moments: StackMoments,
    config: DiagnosticConfig,
    *,
    band: int = 3,
    rows: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """A small, binned, angle-subsampled sinogram from the heaviest detector rows.

    Returns ``(sinogram, theta_used, bin_factor)``. The bin factor is what the caller
    must divide pixel numbers by to get back to full-resolution pixels.
    """
    if rows is not None:
        r0, r1 = int(rows[0]), int(rows[1])
        if not 0 <= r0 < r1 <= moments.n_v:
            raise ValueError(f"rows {rows} out of range for {moments.n_v} detector rows")
    else:
        centre_row = _reference_row(moments)
        r0 = max(0, centre_row - band // 2)
        r1 = min(moments.n_v, r0 + band)
    sign = -1.0 if moments.inverted else 1.0

    step = max(1, int(math.ceil(moments.n_theta / config.max_recon_angles)))
    idx = np.arange(0, moments.n_theta, step)
    rows = np.stack(
        [sign * np.asarray(projections[int(i), r0:r1], dtype=np.float64).mean(axis=0) for i in idx]
    )

    factor = max(1, int(2 ** math.floor(math.log2(max(1, moments.n_u / config.max_recon_width)))))
    if factor > 1:
        width = (rows.shape[1] // factor) * factor
        rows = rows[:, :width].reshape(rows.shape[0], width // factor, factor).mean(axis=2)
    return rows, theta[idx], float(factor)


# --------------------------------------------------------------------------------------
# stage 1 -- data integrity (the ten-minute checks; run these first and most often)
# --------------------------------------------------------------------------------------


def probe_vacuum_phase(
    projections: Any,
    *,
    border: int | None = None,
    config: DiagnosticConfig | None = None,
    chunk: int = 32,
    moments: StackMoments | None = None,
) -> ProbeResult:
    """Mode 11: residual phase ramp / offset. The cheapest test, and the first to run.

    Fits offset + ramp on a presumed-vacuum border of every projection -- by *calling*
    :func:`tktomo.ptycho_align.core.preprocess.remove_phase_ramp` and taking the plane
    it removed, so there is exactly one ramp-fitting implementation in the repo -- and
    reports how big those coefficients are relative to the projection's own contrast,
    and how much they fluctuate from projection to projection.

    Why it comes first: a residual ramp is mathematically indistinguishable from a
    lateral shift, so it does not merely add artifacts, it poisons the alignment that
    is supposed to remove them. Needs no reconstruction and no angles.
    """
    from tktomo.ptycho_align.core.preprocess import (  # noqa: PLC0415
        background_mask,
        remove_phase_ramp,
    )

    name, stage = "vacuum_phase", TriageStage.DATA_INTEGRITY
    cfg = config or DiagnosticConfig()
    n_theta, n_v, n_u = _check_stack(projections)
    border = int(cfg.border if border is None else border)
    if border < 1 or 2 * border >= min(n_v, n_u):
        return _na(name, stage, f"border={border} px leaves no interior for a {n_v}x{n_u} frame")

    mask = background_mask((n_v, n_u), border)
    offset = np.zeros(n_theta)
    ramp_u = np.zeros(n_theta)
    ramp_v = np.zeros(n_theta)
    border_signal = np.zeros(n_theta)
    interior = []

    for i0 in range(0, n_theta, chunk):
        i1 = min(i0 + chunk, n_theta)
        block = np.asarray(projections[i0:i1], dtype=np.float32)
        flat = remove_phase_ramp(block, mask=mask)
        plane = block - flat
        offset[i0:i1] = plane.mean(axis=(1, 2))
        # A plane's peak-to-valley over the frame is |du| + |dv| (corner to corner).
        ramp_u[i0:i1] = plane[:, :, -1].mean(axis=1) - plane[:, :, 0].mean(axis=1)
        ramp_v[i0:i1] = plane[:, -1, :].mean(axis=1) - plane[:, 0, :].mean(axis=1)
        border_signal[i0:i1] = np.abs(flat[:, mask]).mean(axis=1)
        interior.append(flat[:, border:-border, border:-border].ravel()[::97])

    if moments is not None and moments.contrast > 0:
        contrast = moments.contrast
    else:
        sample = np.concatenate(interior)
        lo, hi = np.percentile(sample, [1.0, 99.0])
        contrast = float(hi - lo)
    if not np.isfinite(contrast) or contrast <= 0:
        return _na(name, stage, "the stack has no contrast to measure the ramp against")

    ramp_pv = np.abs(ramp_u) + np.abs(ramp_v)
    offset_spread = float(np.std(offset))
    offset_frac = offset_spread / contrast
    ramp_frac = float(np.median(ramp_pv)) / contrast
    border_frac = float(np.median(border_signal)) / contrast

    metrics = {
        "contrast": contrast,
        "vacuum_offset_mean": float(np.mean(offset)),
        "vacuum_offset_std": offset_spread,
        "vacuum_offset_ptp": float(np.ptp(offset)),
        "vacuum_offset_frac": offset_frac,
        "ramp_pv_median": float(np.median(ramp_pv)),
        "ramp_pv_p95": float(np.percentile(ramp_pv, 95)),
        "ramp_frac": ramp_frac,
        "ramp_u_median": float(np.median(ramp_u)),
        "ramp_v_median": float(np.median(ramp_v)),
        "border_signal_frac": border_frac,
        "border_px": float(border),
    }
    curves = {"vacuum_offset": offset, "ramp_pv": ramp_pv, "ramp_u": ramp_u, "ramp_v": ramp_v}

    caveat = ""
    if border_frac > 0.1:
        caveat = (
            f" NOTE the {border} px border carries {border_frac:.0%} of the contrast, so it "
            "may not be vacuum -- read probe_truncation before trusting these numbers."
        )

    conf = max(
        confidence_from_ratio(offset_frac, cfg.vacuum_mean_frac),
        confidence_from_ratio(ramp_frac, cfg.vacuum_ramp_frac),
    )
    if border_frac > 0.1:
        # The fit is only as good as the assumption that the border is vacuum. When the
        # object spills into it -- truncation, or a frame too tight -- the "ramp" is
        # partly the object, so the evidence is de-weighted in proportion to how much of
        # the contrast is sitting in the supposed vacuum, rather than reported at face
        # value. probe_truncation is what settles which it is, and runs first.
        conf *= 0.1 / border_frac
    if conf <= 0.0:
        return _clear(
            name,
            stage,
            metrics,
            detail=(
                f"vacuum offset varies by {offset_spread:.3g} ({offset_frac:.1%} of contrast) "
                f"and the fitted ramp spans {np.median(ramp_pv):.3g} ({ramp_frac:.1%})."
                + caveat
            ),
            curves=curves,
        )
    detail = (
        f"vacuum offset fluctuates by {offset_spread:.3g} rad ({offset_frac:.1%} of the "
        f"{contrast:.3g} rad contrast, tolerance {cfg.vacuum_mean_frac:.0%}) and the fitted "
        f"background ramp spans {np.median(ramp_pv):.3g} rad peak-to-valley "
        f"({ramp_frac:.1%}, tolerance {cfg.vacuum_ramp_frac:.0%})." + caveat
    )
    return ProbeResult(
        probe=name,
        stage=stage,
        status=ProbeStatus.FIRED,
        metrics=metrics,
        detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.PHASE_RAMP,
                confidence=conf,
                probe=name,
                detail=detail,
                evidence={
                    "vacuum_offset_frac": offset_frac,
                    "ramp_frac": ramp_frac,
                    "vacuum_offset_std": offset_spread,
                    "ramp_pv_median": float(np.median(ramp_pv)),
                },
            ),
        ),
    )


def probe_truncation(
    projections: Any,
    *,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    chunk: int = 32,
) -> ProbeResult:
    """Mode 12: local / interior tomography -- projections truncated by the field of view.

    Per projection the column profile is taken and its GRADIENT at the two frame edges
    compared with its own peak gradient. Gradient, not level: it needs no vacuum
    reference, so it survives exactly the phase offset and ramp that mode 11 is about,
    and it does not depend on where the profile's minimum happens to fall -- which is
    the trap, because a one-sidedly truncated profile has its minimum *at* the truncated
    edge, making every level-based measure read zero there. An enclosed object's profile
    is flat where the vacuum is; a truncated one is still on a slope where the field of
    view cut it off. Both edges sloping means the object exceeds the field of view
    (interior tomography); one edge means it is walking out of the frame. The same test
    on the row profile catches an object taller than the detector.

    Truncation matters far beyond its own cupping artifact: it invalidates *every*
    moment-based estimate here (centroid, second moment, vertical mass profile),
    because the mass that left the frame is missing from all of them.
    """
    name, stage = "truncation", TriageStage.DATA_INTEGRITY
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)

    def edges(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kernel = np.ones(3) / 3.0
        smooth = np.apply_along_axis(lambda p: np.convolve(p, kernel, mode="same"), 1, profiles)
        smooth = smooth[:, 1:-1]  # convolution corrupts the two outermost samples
        grad = np.gradient(smooth, axis=1)
        width = smooth.shape[1]
        # Two normalisations, whichever is larger. The 98th percentile of |grad| alone is
        # dominated by whatever internal feature is sharpest, which hides a real edge
        # slope; the amplitude over a quarter frame alone is tiny for a small object in a
        # big frame, which turns vacuum noise into a truncation alarm.
        amp = smooth.max(axis=1) - smooth.min(axis=1)
        scale = np.maximum(np.percentile(np.abs(grad), 98, axis=1), amp / (0.25 * width))
        scale = np.where(scale > 0, scale, np.nan)
        # SIGNED, so that noise averages out across projections while a real truncation --
        # which always slopes the same way at a given edge -- survives the median.
        left = grad[:, 1:4].mean(axis=1) / scale
        right = grad[:, -4:-1].mean(axis=1) / scale
        return left, right, scale

    u_left, u_right, _ = edges(mom.uprofile)
    v_left, v_right, _ = edges(mom.vprofile)

    left = abs(float(np.nanmedian(u_left)))
    right = abs(float(np.nanmedian(u_right)))
    vertical = min(abs(float(np.nanmedian(v_left))), abs(float(np.nanmedian(v_right))))
    per_projection = np.minimum(np.abs(u_left), np.abs(u_right))

    metrics = {
        "u_edge_both_median": min(left, right),
        "u_edge_worst_median": max(left, right),
        "u_edge_left_median": left,
        "u_edge_right_median": right,
        "v_edge_both_median": vertical,
        "fraction_projections_truncated": float(np.nanmean(per_projection > cfg.truncation_frac)),
    }
    curves = {
        "u_edge_left": u_left,
        "u_edge_right": u_right,
        "v_edge_both": np.minimum(np.abs(v_left), np.abs(v_right)),
    }

    two_sided = max(metrics["u_edge_both_median"], metrics["v_edge_both_median"])
    conf = confidence_from_ratio(two_sided, cfg.truncation_frac)
    one_sided = confidence_from_ratio(metrics["u_edge_worst_median"], 2 * cfg.truncation_frac)

    if conf > 0.0:
        axis = "horizontally" if metrics["u_edge_both_median"] >= metrics["v_edge_both_median"] else "vertically"
        detail = (
            f"the profile is still sloping at both frame edges ({two_sided:.0%} of its peak "
            f"gradient, tolerance {cfg.truncation_frac:.0%}): the object is truncated {axis} in "
            f"{metrics['fraction_projections_truncated']:.0%} of projections. Every "
            "moment-based estimate in this report is biased as a result, and the vacuum "
            "border the ramp fit uses is not vacuum."
        )
    elif one_sided > 0.0:
        detail = (
            f"one edge is sloping at {metrics['u_edge_worst_median']:.0%} of the peak gradient "
            f"while the other is at {metrics['u_edge_both_median']:.0%}: the object leaves the "
            "frame on one side only."
        )
        conf = 0.5 * one_sided
    else:
        return _clear(
            name,
            stage,
            metrics,
            detail=(
                f"profiles are flat at both frame edges "
                f"(worst edge gradient {metrics['u_edge_worst_median']:.1%} of the peak gradient)."
            ),
            curves=curves,
        )

    return ProbeResult(
        probe=name,
        stage=stage,
        status=ProbeStatus.FIRED,
        metrics=metrics,
        detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.LOCAL_TOMOGRAPHY,
                confidence=conf,
                probe=name,
                detail=detail,
                evidence={k: metrics[k] for k in ("u_edge_both_median", "u_edge_worst_median", "v_edge_both_median")},
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# stage 2 -- angular coverage (pure geometry: theta only)
# --------------------------------------------------------------------------------------


def probe_angular_coverage(
    theta: Any, *, config: DiagnosticConfig | None = None, theta_units: str = "auto"
) -> ProbeResult:
    """Mode 10: missing wedge. Angles only -- no projections, no reconstruction.

    Parallel-beam views are redundant modulo 180 deg, so coverage is measured on
    ``theta mod 180``: sort, difference, and include the wrap-around gap. The largest
    gap is the missing wedge.

    This probe cannot be "fixed" by alignment, which is exactly why it runs early: a
    40 deg wedge changes how every later number should be read (and it is the one
    condition under which GridRec fails outright rather than degrading).
    """
    name, stage = "angular_coverage", TriageStage.COVERAGE
    cfg = config or DiagnosticConfig()
    theta_arr = np.asarray(theta, dtype=np.float64).ravel()
    if theta_arr.size < 2:
        return _na(name, stage, "need at least 2 angles")
    theta_rad, _ = _theta_radians(theta_arr, theta_arr.size, theta_units)

    folded = np.sort(np.mod(np.rad2deg(theta_rad), 180.0))
    gaps = np.diff(folded)
    wrap = (folded[0] + 180.0) - folded[-1]
    all_gaps = np.append(gaps, wrap)
    max_gap = float(all_gaps.max())
    steps = np.diff(np.sort(np.rad2deg(theta_rad)))

    metrics = {
        "n_angles": float(theta_arr.size),
        "span_deg": _span_deg(theta_rad),
        "covered_deg": 180.0 - max_gap,
        "max_gap_deg": max_gap,
        "median_step_deg": float(np.median(steps)) if steps.size else float("nan"),
        "n_gaps_over_1deg": float(np.sum(all_gaps > 1.0)),
    }
    curves = {"folded_theta_deg": folded, "gaps_deg": all_gaps}

    conf = confidence_from_ratio(max_gap, cfg.wedge_gap_deg)
    if conf <= 0.0:
        return _clear(
            name,
            stage,
            metrics,
            detail=(
                f"{theta_arr.size} views cover {180.0 - max_gap:.1f} of 180 deg; largest gap "
                f"{max_gap:.2f} deg (tolerance {cfg.wedge_gap_deg:.0f})."
            ),
            curves=curves,
        )
    detail = (
        f"largest gap in theta mod 180 is {max_gap:.1f} deg, leaving {180.0 - max_gap:.1f} deg "
        f"covered by {theta_arr.size} views. Expect elongation perpendicular to the missing "
        "direction; do not use GridRec."
    )
    return ProbeResult(
        probe=name,
        stage=stage,
        status=ProbeStatus.FIRED,
        metrics=metrics,
        detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.MISSING_WEDGE,
                confidence=conf,
                probe=name,
                detail=detail,
                evidence={"max_gap_deg": max_gap, "covered_deg": 180.0 - max_gap},
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# stage 3 -- rotation centre and axis geometry
# --------------------------------------------------------------------------------------


def probe_center_consistency(
    projections: Any,
    theta: Any,
    *,
    center: float | None = None,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    pair_tol_deg: float = 2.0,
    max_pairs: int = 24,
    chunk: int = 32,
) -> ProbeResult:
    """Mode 1 / 5: is the assumed rotation centre the actual one? No reconstruction.

    Two independent estimators, because either alone can be fooled:

    * the constant term of the centroid sinusoid ``com_u = a sin + b cos + c`` -- ``c``
      *is* the axis position for a mass-conserving, untruncated object;
    * 0-vs-180 mirror registration, which uses no model of the object at all:
      ``p(theta+180, u) = p(theta, 2c - u)``, so registering a projection against the
      left-right flip of its opposite partner gives ``c = ((n_u - 1) - s) / 2``.

    When both are available and they disagree, the confidence is halved and the
    disagreement is reported -- that pattern (a good mirror estimate, a bad sinusoid)
    is itself diagnostic of truncation or a residual ramp.
    """
    name, stage = "center_consistency", TriageStage.ROTATION_CENTRE
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)
    assumed = (mom.n_u - 1) / 2.0 if center is None else float(center)

    metrics: dict[str, float] = {"assumed_center_px": assumed, "n_u": float(mom.n_u)}
    estimates: dict[str, float] = {}

    span = _span_deg(theta_rad)
    if span >= 60.0 and np.isfinite(mom.com_u).sum() >= 4:
        good = np.isfinite(mom.com_u)
        a, b, c, fitted = _fit_sinusoid(theta_rad[good], mom.com_u[good])
        estimates["sinusoid"] = c
        metrics["center_sinusoid_px"] = c
        metrics["sinusoid_amplitude_px"] = float(math.hypot(a, b))
        metrics["sinusoid_residual_px"] = _nan_rms(mom.com_u[good] - fitted)
    else:
        metrics["center_sinusoid_px"] = float("nan")

    # 0-vs-180 mirror pairs
    deg = np.rad2deg(theta_rad)
    pair_estimates = []
    used = set()
    for i in range(mom.n_theta):
        if len(pair_estimates) >= max_pairs:
            break
        if i in used:
            continue
        delta = np.abs(np.mod(deg - deg[i], 360.0) - 180.0)
        j = int(np.argmin(delta))
        if delta[j] > pair_tol_deg or j in used or j == i:
            continue
        used.update({i, j})
        shift = _register_1d(mom.uprofile_mass[i], mom.uprofile_mass[j][::-1])
        if math.isfinite(shift):
            pair_estimates.append(((mom.n_u - 1) - shift) / 2.0)

    if pair_estimates:
        arr = np.asarray(pair_estimates)
        # A 0-180 scan yields exactly one theta/theta+180 pair (its two endpoints), and
        # one pair has no spread: the reported 0.00 px scatter means "not measured".

        estimates["mirror"] = float(np.median(arr))
        metrics["center_mirror_px"] = estimates["mirror"]
        metrics["center_mirror_spread_px"] = float(np.std(arr))
        metrics["n_mirror_pairs"] = float(arr.size)
    else:
        metrics["center_mirror_px"] = float("nan")
        metrics["n_mirror_pairs"] = 0.0

    if not estimates:
        return _na(
            name,
            stage,
            f"no usable estimator: span is {span:.0f} deg (need >= 60 for the sinusoid fit) "
            f"and no theta/theta+180 pair within {pair_tol_deg} deg was found",
            metrics=metrics,
        )

    best_key = "mirror" if "mirror" in estimates else "sinusoid"
    estimate = estimates[best_key]
    offset = estimate - assumed
    metrics["center_estimate_px"] = estimate
    metrics["center_offset_px"] = offset
    metrics["estimator"] = 1.0 if best_key == "mirror" else 0.0

    disagreement = float("nan")
    if len(estimates) == 2:
        disagreement = abs(estimates["mirror"] - estimates["sinusoid"])
        metrics["estimator_disagreement_px"] = disagreement

    conf = confidence_from_ratio(abs(offset), cfg.center_tol_px)
    if math.isfinite(disagreement) and disagreement > 2 * cfg.center_tol_px:
        conf *= 0.5

    detail = (
        f"{best_key} estimate puts the axis at {estimate:.2f} px, {offset:+.2f} px from the "
        f"assumed {assumed:.2f} px (tolerance {cfg.center_tol_px} px)"
    )
    if math.isfinite(disagreement):
        detail += f"; the two estimators disagree by {disagreement:.2f} px"
    if best_key == "mirror" and metrics["n_mirror_pairs"] < 2:
        detail += (
            " (from a single theta/theta+180 pair -- the scan spans 180 deg, so its two "
            "endpoints are the only pair and their scatter is unmeasured)"
        )
    detail += "."

    if conf <= 0.0:
        return _clear(name, stage, metrics, detail=detail)

    findings = [
        Finding(
            mode=FailureMode.WRONG_CENTER,
            confidence=conf,
            probe=name,
            detail=detail,
            evidence={"center_estimate_px": estimate, "center_offset_px": offset},
        )
    ]
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics,
        detail=detail, findings=tuple(findings),
    )


def probe_axis_tilt(
    projections: Any,
    theta: Any,
    *,
    center: float | None = None,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
) -> ProbeResult:
    """The three-slice arc test: modes 4 vs 5 vs 6, discriminated rather than guessed.

    Split the detector rows into bands (three by default -- top, middle, bottom), fit the
    centroid sinusoid ``com_u = a sin(theta) + b cos(theta) + c`` in each band, and
    regress the coefficients against the band's mass-weighted height ``z``. The three
    signatures are algebraically distinct, which is what makes the test work:

    * **tilt-axis ANGLE error (4)** -- the axis is tilted by ``alpha`` *within* the
      detector plane, so material at height z rotates about ``c(z) = c0 + z tan(alpha)``.
      The CONSTANT term walks with z. This one is unconfounded: the constant term of a
      band's centroid sinusoid is the rotation-axis position for that band whatever the
      sample looks like, because the sample's own shape lives in ``a`` and ``b``. In
      slices it is arcing that curves in OPPOSITE directions above and below the mid-plane.
    * **tilt-axis LATERAL shift (5)** -- ``c`` flat in z but displaced from the assumed
      centre: arcing in the SAME direction in every slice. The geometric twin of mode 1.
    * **out-of-plane tilt (6)** -- the axis leans toward the beam by ``beta``, so a point
      at height z lands at ``u += z sin(beta) sin(theta)``: the sinusoid AMPLITUDE walks
      with z while ``c`` stays put.

    The trap, and the reason this probe is more than three regressions: **mode 6's
    amplitude signature is exactly what a tilted or sheared SAMPLE produces too**. A slice
    whose centre of mass sits at ``(X_z, Y_z)`` has ``b = X_z``, ``a = -Y_z``, so any
    sample whose centroid walks linearly with height mimics an out-of-plane tilt. On the
    synthetic phantom here that false signal is 1.25 px -- two and a half times the
    firing threshold.

    The discriminator is the vertical channel. A rigid sample rotating about a truly
    vertical axis has an **exactly** rotation-invariant vertical mass profile (the row sum
    is the density integrated over the whole x-y plane), so its centroid ``com_v`` is
    constant in theta no matter how tilted the sample is. A non-vertical axis breaks that:
    to first order ``com_v`` picks up ``-X sin(beta) sin(theta)`` from an out-of-plane
    tilt. So the out-of-plane finding fires only when the amplitude walks with z AND
    ``com_v`` is modulated -- and the modulation gives an independent estimate
    ``sin(beta) = -V_sin / b`` that does not depend on the arc regression at all.

    Only the ``sin(theta)`` component of the modulation is used: over a monotone 0-180 deg
    sweep, ``cos(theta)`` is nearly collinear with a vertical drift in acquisition order
    (mode 3), and attributing a drift to a tilt would be worse than missing the tilt.

    Known bias: under tilt, material leaks between row bands as theta turns, which dilutes
    the fitted ``c(z)`` slope. On the synthetic phantom the recovered tilt angle is about
    70% of the injected one, so read ``tilt_angle_deg`` as a lower bound.
    """
    name, stage = "axis_tilt", TriageStage.ROTATION_CENTRE
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)
    assumed = (mom.n_u - 1) / 2.0 if center is None else float(center)

    n_bands = mom.band_uprofile.shape[1]
    if n_bands < 3:
        return _na(name, stage, f"the arc test needs >= 3 row bands, moments carry {n_bands}")
    span = _span_deg(theta_rad)
    if span < 60.0:
        return _na(name, stage, f"span is {span:.0f} deg; the sinusoid fit needs >= 60 deg")

    u = np.arange(mom.n_u, dtype=np.float64)
    band_mass = mom.band_uprofile.sum(axis=2)  # (n_theta, n_bands)
    mean_mass = band_mass.mean(axis=0)
    weak = mean_mass < 0.05 * mean_mass.max()
    if np.any(weak):
        return _na(
            name,
            stage,
            f"row band(s) {list(np.flatnonzero(weak))} carry under 5% of the mass of the "
            "heaviest band, so their centroid sinusoid is noise. The object does not span "
            "enough of the detector height for a three-slice arc test",
        )

    coeff_a = np.zeros(n_bands)
    coeff_b = np.zeros(n_bands)
    coeff_c = np.zeros(n_bands)
    resid = np.zeros(n_bands)
    for b in range(n_bands):
        com = (mom.band_uprofile[:, b] @ u) / band_mass[:, b]
        a, bb, c, fitted = _fit_sinusoid(theta_rad, com)
        coeff_a[b], coeff_b[b], coeff_c[b] = a, bb, c
        resid[b] = _nan_rms(com - fitted)

    z = mom.band_z - mom.band_z.mean()
    lever = float(np.ptp(mom.band_z)) / 2.0  # half the object's own height, in px
    slope_c, intercept_c = np.polyfit(z, coeff_c, 1)
    slope_a = np.polyfit(z, coeff_a, 1)[0]
    slope_b = np.polyfit(z, coeff_b, 1)[0]
    slope_sin = float(math.hypot(slope_a, slope_b))

    travel_c = abs(float(slope_c)) * lever
    travel_sin = slope_sin * lever
    lateral = float(intercept_c) - assumed

    # The vertical channel: com_v = p sin + q cos + (smooth trend in acquisition index).
    # The trend absorbs a vertical drift, which would otherwise leak into q -- and q is
    # discarded anyway, for the same reason.
    idx = np.arange(mom.n_theta, dtype=np.float64) / max(mom.n_theta - 1, 1)
    good_v = np.isfinite(mom.com_v)
    modulation = float("nan")
    modulation_se = float("nan")
    if good_v.sum() >= 6:
        design = np.column_stack(
            [
                np.sin(theta_rad[good_v]),
                np.cos(theta_rad[good_v]),
                np.ones(int(good_v.sum())),
                idx[good_v],
                idx[good_v] ** 2,
            ]
        )
        coeffs, *_ = np.linalg.lstsq(design, mom.com_v[good_v], rcond=None)
        modulation = float(coeffs[0])
        # ... and its standard error, because per-projection jitter of sigma px puts
        # roughly sigma*sqrt(2/n) of spurious modulation into this coefficient. Without
        # this gate, 1.2 px of vertical jitter on 60 views manufactures an out-of-plane
        # tilt at 0.65 confidence. Measured; it is the false positive this gate exists for.
        residual_v = mom.com_v[good_v] - design @ coeffs
        dof = max(1, int(good_v.sum()) - design.shape[1])
        try:
            cov = float(np.linalg.inv(design.T @ design)[0, 0])
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate angle set
            cov = float("nan")
        modulation_se = float(np.sqrt(max(cov, 0.0) * float(residual_v @ residual_v) / dof))

    # sin(beta) = -V_sin / b, with b the cos coefficient of the whole-stack centroid
    # sinusoid (the x-coordinate of the object's centre of mass in the rotating frame).
    # An IN-PLANE tilt modulates com_v too, by -a sin(alpha), so its share is predicted
    # from the already-measured c(z) slope and subtracted before the out-of-plane test.
    # The subtraction is conservative: the c(z) slope is itself biased low (band leakage),
    # so a large mode-4 tilt can still leak into mode 6. Fix mode 4 first, then re-run.
    good_u = np.isfinite(mom.com_u)
    beta_deg = float("nan")
    modulation_inplane = float("nan")
    modulation_residual = modulation
    if good_u.sum() >= 4:
        a_all, b_all, _, _ = _fit_sinusoid(theta_rad[good_u], mom.com_u[good_u])
        if math.isfinite(modulation):
            modulation_inplane = float(-a_all * slope_c)
            modulation_residual = modulation - modulation_inplane
            if abs(b_all) > 1.0:
                beta_deg = float(
                    np.rad2deg(np.arcsin(np.clip(-modulation_residual / b_all, -1.0, 1.0)))
                )

    metrics = {
        "slope_c_px_per_row": float(slope_c),
        "tilt_angle_deg": float(np.rad2deg(np.arctan(slope_c))),
        "axis_travel_px": travel_c,
        "slope_sin_px_per_row": slope_sin,
        "amplitude_travel_px": travel_sin,
        "lateral_offset_px": lateral,
        "vertical_modulation_px": modulation,
        "vertical_modulation_se_px": modulation_se,
        "vertical_modulation_from_inplane_px": modulation_inplane,
        "vertical_modulation_residual_px": modulation_residual,
        "out_of_plane_deg": beta_deg,
        "band_residual_rms_px": float(np.mean(resid)),
        "object_half_height_px": lever,
        "n_bands": float(n_bands),
    }
    curves = {
        "band_z": mom.band_z, "band_c": coeff_c, "band_a": coeff_a, "band_b": coeff_b,
        "band_residual": resid,
    }

    # How much of the model actually fits. See DiagnosticConfig.arc_residual_ratio: on a
    # stack whose centroids do not follow the rigid sinusoid, the fitted constant term --
    # and therefore the whole c(z) slope -- is not measuring the axis.
    effect = abs(float(slope_c)) * max(float(np.ptp(mom.band_z)), 1e-9)
    strain = float(np.mean(resid)) / max(effect, 1e-9)
    metrics["band_residual_over_effect"] = strain
    strained = strain > cfg.arc_residual_ratio
    caveat = (
        f" CAVEAT: the per-band sinusoid residual ({np.mean(resid):.2f} px rms) is "
        f"{strain:.1f}x the axis walk this is built on, so the rigid-object model is not "
        "fitting and the slope may be leakage rather than geometry. Confirm with an "
        "independent estimator before acting: probe_center_sweep(rows=(r0, r1)) on a top "
        "and a bottom band uses a reconstruction and shares none of these assumptions."
        if strained
        else ""
    )

    findings: list[Finding] = []
    conf_angle = confidence_from_ratio(travel_c, cfg.tilt_tol_px)
    if strained:
        conf_angle *= 0.5
    if conf_angle > 0.0:
        detail = (
            f"the sinusoid offset walks {slope_c:+.4f} px per detector row -- {travel_c:.2f} px "
            f"over half the object's height (tolerance {cfg.tilt_tol_px} px), i.e. an in-plane "
            f"axis tilt of at least {metrics['tilt_angle_deg']:+.3f} deg. Top and bottom slices "
            "will arc in OPPOSITE directions." + caveat
        )
        findings.append(
            Finding(
                mode=FailureMode.TILT_AXIS_ANGLE, confidence=conf_angle, probe=name, detail=detail,
                evidence={"axis_travel_px": travel_c, "tilt_angle_deg": metrics["tilt_angle_deg"]},
            )
        )

    walks = confidence_from_ratio(travel_sin, cfg.tilt_tol_px)
    if strained:
        walks *= 0.5
    modulation_floor = cfg.vertical_modulation_tol_px
    if math.isfinite(modulation_se):
        modulation_floor = max(modulation_floor, 3.0 * modulation_se)
    modulated = confidence_from_ratio(abs(modulation_residual), modulation_floor)
    if walks > 0.0 and modulated > 0.0:
        detail = (
            f"the sinusoid AMPLITUDE grows {slope_sin:.4f} px per detector row "
            f"({travel_sin:.2f} px over half the object's height) AND the vertical mass "
            f"centroid is modulated by {modulation_residual:+.2f} px sin(theta) beyond what the "
            f"in-plane tilt explains, which a vertical axis cannot produce: the axis leans out "
            f"of the detector plane by {beta_deg:.2f} deg." + caveat
        )
        findings.append(
            Finding(
                mode=FailureMode.OUT_OF_PLANE_TILT, confidence=min(walks, modulated), probe=name,
                detail=detail,
                evidence={"amplitude_travel_px": travel_sin, "vertical_modulation_px": modulation,
                          "out_of_plane_deg": beta_deg},
            )
        )

    conf_lat = confidence_from_ratio(abs(lateral), cfg.center_tol_px)
    if conf_lat > 0.0 and conf_angle <= 0.0:
        detail = (
            f"the axis sits {lateral:+.2f} px from the assumed {assumed:.2f} px at every height "
            f"(slope {slope_c:+.4f} px/row is within tolerance): arcing in the SAME direction in "
            "all slices. This is the geometric twin of a wrong rotation centre (mode 1)."
        )
        findings.append(
            Finding(
                mode=FailureMode.TILT_AXIS_LATERAL, confidence=conf_lat, probe=name, detail=detail,
                evidence={"lateral_offset_px": lateral},
            )
        )

    if not findings:
        detail = (
            f"axis position varies {travel_c:.2f} px and sinusoid amplitude {travel_sin:.2f} px "
            f"over half the object's height; lateral offset {lateral:+.2f} px; vertical "
            f"modulation {modulation:+.2f} px."
        )
        if walks > 0.0:
            detail += (
                " The amplitude walk exceeds tolerance but the vertical mass profile is "
                "rotation-invariant, so this is a tilted or sheared SAMPLE, not a tilted axis "
                "-- no alignment correction applies."
            )
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics,
        detail="; ".join(f.detail.split(".")[0] for f in findings), findings=rank(findings),
        curves=curves,
    )


def probe_center_sweep(
    projections: Any,
    theta: Any,
    *,
    center: float | None = None,
    rows: tuple[int, int] | None = None,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
) -> ProbeResult:
    """Mode 1, confirmed the way the roadmap says to confirm it: sweep and minimise.

    Reconstructs one binned slice at a grid of assumed centres with the module's own
    filtered backprojection and reports where slice entropy is minimised (and, as a
    cross-check, where gradient energy is maximised). Both are computed inside an
    inscribed circle so the reconstruction's corner artifacts cannot drive the metric.

    The sweep's resolution is ``center_sweep_step * bin_factor`` full-resolution pixels,
    and the firing threshold is widened to that when binning is coarse -- a sweep that
    cannot resolve half a pixel is not allowed to claim half a pixel.

    ``rows=(r0, r1)`` reconstructs a chosen band of detector rows instead of the object's
    mid-plane. That is the independent cross-check for the arc test's mode-4 verdict: run
    it on a top and a bottom band and see whether the entropy-optimal centre really walks
    with height. It shares no assumption with the centroid sinusoid -- which is the point,
    because on real data the two have been seen to disagree in sign.
    """
    name, stage = "center_sweep", TriageStage.ROTATION_CENTRE
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)
    assumed = (mom.n_u - 1) / 2.0 if center is None else float(center)

    span = _span_deg(theta_rad)
    if span < 150.0:
        return _na(
            name, stage,
            f"span is {span:.0f} deg; filtered backprojection needs >= 150 deg before its "
            "streaks stop dominating the sharpness metric",
        )
    if mom.n_theta < 20:
        return _na(name, stage, f"{mom.n_theta} angles is too few to reconstruct a slice")

    sino, theta_used, factor = _prepare_slice_sinogram(projections, theta_rad, mom, cfg, rows=rows)
    width = sino.shape[1]
    assumed_binned = (assumed + 0.5) / factor - 0.5
    step = cfg.center_sweep_step
    half = max(step, cfg.center_sweep_px / factor)
    grid = np.arange(assumed_binned - half, assumed_binned + half + 0.5 * step, step)
    grid = grid[(grid > 0.15 * width) & (grid < 0.85 * width)]
    if grid.size < 5:
        return _na(name, stage, "the centre sweep grid does not fit inside the detector")

    mask = _circle_mask(width)
    entropy = np.zeros(grid.size)
    gradient = np.zeros(grid.size)
    for i, c in enumerate(grid):
        recon = fbp_slice(sino, theta_used, center=float(c))
        entropy[i], gradient[i] = _sharpness(recon, mask)

    def refine(values: np.ndarray, minimise: bool) -> float:
        k = int(np.argmin(values) if minimise else np.argmax(values))
        if 0 < k < values.size - 1:
            y0, y1, y2 = values[k - 1], values[k], values[k + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                return float(grid[k] + 0.5 * step * (y0 - y2) / denom)
        return float(grid[k])

    best_entropy = refine(entropy, minimise=True)
    best_gradient = refine(gradient, minimise=False)
    at_edge = int(np.argmin(entropy)) in (0, entropy.size - 1)
    best_px = (best_entropy + 0.5) * factor - 0.5
    grad_px = (best_gradient + 0.5) * factor - 0.5
    offset = best_px - assumed
    resolution = step * factor
    tolerance = max(cfg.center_tol_px, resolution)

    metrics = {
        "center_entropy_px": best_px,
        "center_gradient_px": grad_px,
        "center_offset_px": offset,
        "bin_factor": factor,
        "sweep_resolution_px": resolution,
        "n_angles_used": float(theta_used.size),
        "entropy_contrast": float(np.ptp(entropy)),
    }
    curves = {"sweep_center_px": (grid + 0.5) * factor - 0.5, "entropy": entropy, "gradient": gradient}

    detail = (
        f"entropy is minimised at {best_px:.2f} px, {offset:+.2f} px from the assumed "
        f"{assumed:.2f} px; gradient energy peaks at {grad_px:.2f} px "
        f"(sweep resolution {resolution:.2f} px after binning {factor:.0f}x)."
    )
    metrics["optimum_at_sweep_edge"] = float(at_edge)
    conf = confidence_from_ratio(abs(offset), tolerance)
    if abs(best_px - grad_px) > 3 * resolution:
        conf *= 0.5
        detail += " The two sharpness metrics disagree, so treat this as weak evidence."
    if at_edge:
        # The minimum is wherever the grid stopped, which is not a measurement. Seen on
        # real data: a 10 px sweep on a stack whose centre was further out than that.
        conf *= 0.5
        detail += (
            f" The minimum sits ON the edge of the swept range, so the true centre is "
            f"outside +/-{cfg.center_sweep_px:.0f} px and this is a lower bound, not an estimate."
        )
    if conf <= 0.0:
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.WRONG_CENTER, confidence=conf, probe=name, detail=detail,
                evidence={"center_entropy_px": best_px, "center_offset_px": offset},
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# stage 4 -- vertical (the easy, decoupled direction)
# --------------------------------------------------------------------------------------


def _vertical_shifts(mom: StackMoments) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-projection vertical displacement, split into trend and white residual.

    The vertical mass profile (row sum of a projection) is *exactly* invariant under
    rotation about a vertical axis in parallel-beam geometry -- it is the density
    integrated over the whole (x, y) plane at height z. So every variation in it is
    misalignment (or truncation), which is why the vertical direction needs neither a
    forward nor a back projection: register the 1D profiles and you are done
    (Odstrcil et al., Opt. Express 27:36637, 2019).
    """
    reference = np.median(mom.vprofile, axis=0)
    shifts = np.array([_register_1d(reference, prof) for prof in mom.vprofile])
    trend, white = _split_trend(shifts, degree=2)
    return shifts, trend, white


def probe_vertical_drift(
    projections: Any,
    *,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    chunk: int = 32,
) -> ProbeResult:
    """Mode 3: vertical drift. Row-sum, register 1D, split trend from noise.

    Fires on the *smooth* part only: the white part is jitter (mode 2) and belongs to
    :func:`probe_shift_jitter`, which reads the same numbers.
    """
    name, stage = "vertical_drift", TriageStage.VERTICAL
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    shifts, trend, white = _vertical_shifts(mom)
    if not np.isfinite(shifts).any():
        return _na(name, stage, "vertical mass profiles are empty; is the stack all zeros?")

    drift_ptp = float(np.ptp(trend[np.isfinite(trend)])) if np.isfinite(trend).any() else float("nan")
    finite = shifts[np.isfinite(shifts)]
    # A raw peak-to-peak is not a description of this series: on real data a handful of
    # projections mis-register by tens of pixels (a truncated top edge, a dropped frame)
    # and dominate it. Report the robust range and count the outliers separately -- they
    # are themselves a finding, just not this one.
    spread = np.abs(white[np.isfinite(white)] - np.nanmedian(white))
    scale = 1.4826 * float(np.median(spread)) if spread.size else float("nan")
    outliers = int(np.sum(spread > 5.0 * scale)) if scale > 0 else 0
    metrics = {
        "vertical_ptp_px": float(np.ptp(finite)) if finite.size else float("nan"),
        "vertical_range_p5_p95_px": float(np.subtract(*np.percentile(finite, [95, 5])))
        if finite.size
        else float("nan"),
        "n_outlier_projections": float(outliers),
        "drift_ptp_px": drift_ptp,
        "vertical_jitter_rms_px": _nan_rms(white),
        "lag1_autocorr": _lag1_autocorr(shifts),
        "drift_start_px": float(trend[0]) if np.isfinite(trend[0]) else float("nan"),
        "drift_end_px": float(trend[-1]) if np.isfinite(trend[-1]) else float("nan"),
    }
    curves = {"vertical_shift_px": shifts, "vertical_trend_px": trend, "vertical_white_px": white}

    conf = confidence_from_ratio(drift_ptp, cfg.vertical_drift_tol_px)
    detail = (
        f"vertical mass profile moves {metrics['vertical_range_p5_p95_px']:.2f} px over its 5-95 "
        f"percentile range, of which {drift_ptp:.2f} px is a smooth trend (tolerance "
        f"{cfg.vertical_drift_tol_px} px) and {metrics['vertical_jitter_rms_px']:.2f} px rms is "
        f"white; {outliers} projection(s) failed to register (> 5 MAD)."
    )
    if conf <= 0.0:
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.VERTICAL_DRIFT, confidence=conf, probe=name, detail=detail,
                evidence={"drift_ptp_px": drift_ptp, "vertical_ptp_px": metrics["vertical_ptp_px"]},
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# stage 5 -- horizontal (entangled with the rotation angle; hardest direction)
# --------------------------------------------------------------------------------------


def probe_shift_jitter(
    projections: Any,
    theta: Any,
    *,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
) -> ProbeResult:
    """Mode 2: random per-projection jitter -- and the test that says it is *random*.

    Blur alone does not identify jitter; a wrong centre blurs too (into doubled edges,
    mode 1) and so does a drift (mode 3). The discriminator is the residual's character
    in *acquisition order*: horizontal residual about the fitted centroid sinusoid and
    the white part of the vertical shift series are both examined, and jitter is
    declared only when a residual is both large and white (lag-1 autocorrelation below
    the configured bound). A large but smooth residual is reported as such and pointed
    at drift or an angle error instead.
    """
    name, stage = "shift_jitter", TriageStage.HORIZONTAL
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)

    metrics: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    span = _span_deg(theta_rad)
    good = np.isfinite(mom.com_u)
    horizontal = np.full(mom.n_theta, np.nan)
    if span >= 60.0 and good.sum() >= 4:
        _, _, _, fitted = _fit_sinusoid(theta_rad[good], mom.com_u[good])
        horizontal[good] = mom.com_u[good] - fitted
        metrics["horizontal_rms_px"] = _nan_rms(horizontal)
        metrics["horizontal_lag1"] = _lag1_autocorr(horizontal)
        curves["horizontal_residual_px"] = horizontal
    else:
        metrics["horizontal_rms_px"] = float("nan")
        metrics["horizontal_lag1"] = float("nan")

    _, _, white = _vertical_shifts(mom)
    metrics["vertical_rms_px"] = _nan_rms(white)
    metrics["vertical_lag1"] = _lag1_autocorr(white)
    curves["vertical_residual_px"] = white

    if not np.isfinite(metrics["horizontal_rms_px"]) and not np.isfinite(metrics["vertical_rms_px"]):
        return _na(name, stage, f"no usable residual series (span {span:.0f} deg, no mass?)")

    candidates = []
    for axis in ("horizontal", "vertical"):
        rms = metrics[f"{axis}_rms_px"]
        lag1 = metrics[f"{axis}_lag1"]
        if not math.isfinite(rms):
            continue
        conf = confidence_from_ratio(rms, cfg.jitter_tol_px)
        white_enough = math.isfinite(lag1) and abs(lag1) < cfg.jitter_autocorr_max
        candidates.append((axis, rms, lag1, conf, white_enough))

    firing = [c for c in candidates if c[3] > 0.0 and c[4]]
    if not firing:
        smooth = [c for c in candidates if c[3] > 0.0 and not c[4]]
        detail = "; ".join(
            f"{axis} residual {rms:.2f} px rms, lag-1 {lag1:+.2f}" for axis, rms, lag1, _, _ in candidates
        )
        if smooth:
            detail += (
                f" -- large but SMOOTH (lag-1 >= {cfg.jitter_autocorr_max}), so this is drift or an "
                "angle error, not jitter; see probe_vertical_drift and probe_angle_readback."
            )
        return _clear(name, stage, metrics, detail=detail, curves=curves)

    axis, rms, lag1, conf, _ = max(firing, key=lambda c: c[3])
    detail = (
        f"{axis} residual is {rms:.2f} px rms (tolerance {cfg.jitter_tol_px} px) and white in "
        f"acquisition order (lag-1 autocorrelation {lag1:+.2f}): random per-projection jitter. "
        "Expect uniform blurring, not doubled edges."
    )
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.JITTER, confidence=conf, probe=name, detail=detail,
                evidence={f"{axis}_rms_px": rms, f"{axis}_lag1": lag1},
            ),
        ),
    )


def probe_angle_readback(
    projections: Any,
    theta: Any,
    *,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
    gain_range: float = 0.2,
) -> ProbeResult:
    """Mode 7: systematic angle readback error, found by refitting with a free gain.

    If the recorded angles are wrong by a *scale* factor (a stage calibration error, a
    span that is not the span you think), the centroid sinusoid does not close over the
    reported range. Refitting ``com_u = a sin(g theta) + b cos(g theta) + c`` over a
    grid of gains and finding a minimum well away from ``g = 1`` that also *halves* the
    residual is strong evidence -- and the fitted gain is the correction.

    Honest limitation: RANDOM readback noise is indistinguishable from lateral jitter
    (mode 2) at the centroid level, because both enter as ``amplitude * delta``. Only
    the systematic part is separable this cheaply.
    """
    name, stage = "angle_readback", TriageStage.HORIZONTAL
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)

    span = _span_deg(theta_rad)
    good = np.isfinite(mom.com_u)
    if mom.n_theta < 20 or good.sum() < 20:
        return _na(name, stage, f"{int(good.sum())} usable projections; the gain fit needs >= 20")
    if span < 90.0:
        return _na(name, stage, f"span is {span:.0f} deg; the gain fit needs >= 90 deg of lever arm")

    th = theta_rad[good]
    com = mom.com_u[good]
    a, b, _, fitted = _fit_sinusoid(th, com)
    amplitude = float(math.hypot(a, b))
    base_resid = _nan_rms(com - fitted)
    if amplitude < 2.0:
        return _na(
            name, stage,
            f"the centroid sinusoid has amplitude {amplitude:.2f} px: the object sits on the "
            "rotation axis, so the projections carry almost no angle information",
        )

    def residual_at(gain: float) -> float:
        basis = np.column_stack([np.sin(gain * th), np.cos(gain * th), np.ones_like(th)])
        coeffs, *_ = np.linalg.lstsq(basis, com, rcond=None)
        return float(np.sqrt(np.mean((com - basis @ coeffs) ** 2)))

    coarse = np.linspace(1.0 - gain_range, 1.0 + gain_range, 81)
    values = np.array([residual_at(g) for g in coarse])
    k = int(np.argmin(values))
    at_edge = k in (0, coarse.size - 1)
    fine = np.linspace(coarse[max(k - 1, 0)], coarse[min(k + 1, coarse.size - 1)], 41)
    fine_values = np.array([residual_at(g) for g in fine])
    best_gain = float(fine[int(np.argmin(fine_values))])
    best_resid = float(fine_values.min())

    ratio = best_resid / base_resid if base_resid > 0 else 1.0
    span_error = (best_gain - 1.0) * span
    # What the error costs in the reconstruction, which is the number that decides
    # whether it matters: an angular error of (g-1)*theta smears a feature at radius r
    # tangentially by (g-1)*theta*r, worst at the end of the scan, with r the distance
    # from the axis to the object's outermost material (see :func:`_object_rim_px`).
    radius = _object_rim_px(mom, (mom.n_u - 1) / 2.0)
    tangential = abs(best_gain - 1.0) * np.deg2rad(span) * radius

    metrics = {
        "angle_gain": best_gain,
        "residual_px_at_gain1": base_resid,
        "residual_px_best": best_resid,
        "residual_ratio": ratio,
        "implied_span_error_deg": span_error,
        "implied_rim_smear_px": float(tangential),
        "object_radius_px": radius,
        "sinusoid_amplitude_px": amplitude,
        "gain_at_grid_edge": float(at_edge),
    }
    curves = {"gain_grid": coarse, "gain_residual_px": values}

    detail = (
        f"free-gain refit prefers g = {best_gain:.4f} (implied span error {span_error:+.2f} deg), "
        f"cutting the centroid residual from {base_resid:.3f} to {best_resid:.3f} px and implying "
        f"{tangential:.2f} px of tangential smear at the rim of a {radius:.0f} px object."
    )
    # The centroid residual is a poor firing criterion and the tangential smear is a
    # good one: over a half turn a sinusoid absorbs almost all of a gain error, so a 6%
    # error hides in 0.05 px of centroid residual while smearing the rim by 3 px.
    if at_edge:
        # The best gain is wherever the grid stopped, which means the sinusoid residual
        # is being driven by something that is not an angle error at all (on our own
        # scans, by 13 px rms of genuine misalignment). Seen on real data; the reported
        # smear would otherwise be hundreds of pixels of nonsense.
        detail += (
            f" The optimum sits on the edge of the searched range (g = {best_gain:.2f}), so the "
            "residual is not being driven by an angle error -- clear the earlier stages first."
        )
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    fires = (
        ratio < cfg.angle_gain_gain_ratio
        and float(tangential) > cfg.angle_tangential_tol_px
        and abs(span_error) > 0.5
    )
    if not fires:
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    conf = confidence_from_ratio(float(tangential), cfg.angle_tangential_tol_px)
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.ANGLE_READBACK, confidence=conf, probe=name,
                detail=detail + " Expect azimuthal (tangential) smearing, not radial.",
                evidence={"angle_gain": best_gain, "residual_ratio": ratio,
                          "implied_span_error_deg": span_error},
            ),
        ),
    )


def probe_scale_drift(
    projections: Any,
    theta: Any,
    *,
    center: float | None = None,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
) -> ProbeResult:
    """Mode 8: magnification / scale drift, from the projected second moment.

    For a rigid object the second central moment of a parallel-beam projection along u
    is *exactly* harmonic in 2 theta::

        sigma_u^2(theta) = A + B cos(2 theta) + C sin(2 theta)

    (it is the (u,u) element of the rotated inertia tensor), and a magnification s(i)
    multiplies it by s(i)^2. Both are fitted **at once**, with the drift as two extra
    columns in the same least-squares problem::

        sigma_u^2 = A + B cos(2 theta) + C sin(2 theta) + D i + E i^2

    Fitting the harmonic first and looking for a trend in the residual does not work:
    over a 0-180 deg sweep theta is monotone in acquisition index, so a drift partly
    projects onto sin(2 theta) and the harmonic eats it. Measured on the synthetic
    phantom, an injected 2% drift came back as 0.9% that way and 2.9% this way.

    What fires the probe is the **displacement at the object's rim**, not the fractional
    size change: 1% on a 10 px object is nothing and 1% on a 1000 px object is 10 px of
    blur, and mode 8's whole signature is blur growing with distance from the axis. The
    drift is also gated against its own standard error, because a least-squares trend
    through a noisy series is never exactly zero -- 2% detector noise on the phantom
    otherwise manufactures a 1% drift out of nothing.

    Truncation invalidates this probe completely (the mass outside the frame is missing
    from every moment), so the fit residual is reported alongside as a warning flag: it
    also catches a deformation (mode 9) or a wrong angle axis (mode 7) leaking in, both
    of which break the rigid-object model rather than scaling it.
    """
    name, stage = "scale_drift", TriageStage.HORIZONTAL
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)
    assumed = (mom.n_u - 1) / 2.0 if center is None else float(center)

    span = _span_deg(theta_rad)
    good = np.isfinite(mom.var_u) & (mom.var_u > 0)
    if good.sum() < 10:
        return _na(name, stage, f"{int(good.sum())} projections have a usable second moment")
    if span < 60.0:
        return _na(
            name, stage,
            f"span is {span:.0f} deg; the cos(2 theta) model needs >= 60 deg to be identifiable",
        )

    th = theta_rad[good]
    var = mom.var_u[good]
    index = np.arange(mom.n_theta, dtype=np.float64)[good] / max(mom.n_theta - 1, 1)
    # ONE joint linear fit, harmonic and trend together:
    #     sigma_u^2 = A + B cos(2 theta) + C sin(2 theta) + D i + E i^2
    # Fitting the harmonic first and looking for a trend in what is left does not work:
    # over a 0-180 deg sweep theta is monotone in acquisition index, so a drift partly
    # projects onto sin(2 theta) and the harmonic eats it. Measured on the synthetic
    # phantom, an injected 2% drift came back as 0.9% that way and as 2.9% this way.
    basis = np.column_stack(
        [np.ones_like(th), np.cos(2 * th), np.sin(2 * th), index, index**2]
    )
    coeffs, *_ = np.linalg.lstsq(basis, var, rcond=None)
    harmonic = basis[:, :3] @ coeffs[:3]
    trend = basis[:, 3:] @ coeffs[3:]
    if np.any(harmonic <= 0):
        return _na(
            name, stage,
            "the harmonic second-moment fit went non-positive; the data is not "
            "rigid-object-like and no scale can be read off it",
        )

    level = float(np.mean(harmonic))
    # sigma scales with s, so a relative change in sigma^2 is twice the change in scale.
    total = 0.5 * float(trend[-1] - trend[0]) / level
    residual = var - basis @ coeffs
    fit_residual = float(np.sqrt(np.mean(residual**2)) / level)

    # ... and the standard error of that drift, because a least-squares trend through a
    # noisy series is never exactly zero. Without this gate, 2% detector noise on the
    # synthetic phantom manufactures a 1% magnification drift -- five times the firing
    # threshold -- out of nothing. The contrast vector picks out trend(1) - trend(0).
    contrast_vector = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dof = max(1, var.size - basis.shape[1])
    sigma2 = float(residual @ residual) / dof
    try:
        cov = float(contrast_vector @ np.linalg.inv(basis.T @ basis) @ contrast_vector)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate angle set
        cov = float("nan")
    total_se = 0.5 * float(np.sqrt(max(cov, 0.0) * sigma2)) / level
    non_smooth = fit_residual / max(abs(total), 1e-12)
    scale = np.sqrt(np.clip(var / harmonic, 1e-12, None))
    radius = _object_rim_px(mom, assumed)
    rim_shift = abs(total) * radius
    rim_shift_se = abs(total_se) * radius

    metrics = {
        "scale_change_frac": total,
        "scale_change_se": total_se,
        "scale_ppm_per_projection": total / max(mom.n_theta - 1, 1) * 1e6,
        "fit_residual_frac": fit_residual,
        "non_smooth_frac": non_smooth,
        "object_rim_px": radius,
        "implied_rim_shift_px": rim_shift,
        "implied_rim_shift_se_px": rim_shift_se,
    }
    curves = {"scale_series": scale, "var_u_px2": mom.var_u}

    detail = (
        f"projected size drifts {total * 100:+.3f}% across the scan "
        f"({metrics['scale_ppm_per_projection']:+.1f} ppm per projection), which displaces material "
        f"at the object's rim, {radius:.0f} px from the axis, by {rim_shift:.2f} px "
        f"(standard error {rim_shift_se:.2f} px). Rigid-object fit residual {fit_residual:.2%} "
        "of the mean second moment."
    )
    floor = cfg.scale_rim_tol_px
    if math.isfinite(rim_shift_se):
        floor = max(floor, 3.0 * rim_shift_se)
    conf = confidence_from_ratio(rim_shift, floor)
    if non_smooth > 0.25:
        conf *= 0.5
        detail += (
            f" The fit residual is {non_smooth:.0%} of the drift it implies, so the size series is "
            "not a smooth magnification: suspect a deformation (mode 9, check the projected mass) "
            "or a wrong angle axis (mode 7)."
        )
    if conf <= 0.0:
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(
                mode=FailureMode.SCALE_DRIFT, confidence=conf, probe=name, detail=detail,
                evidence={"scale_change_frac": total, "implied_rim_shift_px": rim_shift,
                          "object_rim_px": radius},
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# stage 6 -- non-rigid (only meaningful once every rigid error above is cleared)
# --------------------------------------------------------------------------------------


def _residual_stats(measured: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-angle reprojection residual AFTER the best rigid shift, gain and offset.

    Removing the shift is the whole point: what is left is what no rigid alignment can
    fix. The gain and offset go too, so a global intensity mismatch between a
    quick-and-dirty reconstruction and the measurement does not masquerade as
    deformation.
    """
    n_theta, n_u = measured.shape
    u = np.arange(n_u, dtype=np.float64)
    fractions = np.zeros(n_theta)
    energy = np.zeros(n_u)
    shifts = np.zeros(n_theta)
    for i in range(n_theta):
        meas = measured[i]
        sim = simulated[i]
        shift = _register_1d(sim, meas)
        if not math.isfinite(shift):
            fractions[i] = np.nan
            continue
        shifts[i] = shift
        aligned = np.interp(u - shift, u, sim, left=sim[0], right=sim[-1])
        design = np.column_stack([aligned, np.ones_like(aligned)])
        coeffs, *_ = np.linalg.lstsq(design, meas, rcond=None)
        residual = meas - design @ coeffs
        scale = float(np.std(meas))
        fractions[i] = float(np.sqrt(np.mean(residual**2))) / scale if scale > 0 else np.nan
        energy += residual**2
    return fractions, float(np.nanmedian(fractions)), energy


def probe_deformation(
    projections: Any,
    theta: Any,
    *,
    center: float | None = None,
    volume: np.ndarray | None = None,
    config: DiagnosticConfig | None = None,
    moments: StackMoments | None = None,
    theta_units: str = "auto",
    chunk: int = 32,
    band: int = 3,
    allow_reconstruction: bool = True,
) -> ProbeResult:
    """Mode 9: sample deformation or radiation damage -- no rigid alignment works.

    Two independent statistics, because the roadmap's own test is surprisingly blunt:

    1. **Projected mass conservation.** A parallel-beam projection of an *enclosed*
       object has a line integral whose total is the object's mass -- exactly, at every
       angle, independent of alignment, rotation centre, axis tilt and angle readback.
       So the non-smooth part of the projected mass (a smooth trend is removed first,
       since that is intensity or scale drift, not deformation) is a rigorous rigidity
       test that costs one pass over the data. On the synthetic phantom it separates a
       clean stack from an injected deformation by a factor of 128, where the
       reprojection residual below separates them by a factor of 1.1.
    2. **Reprojection residual**, as the roadmap prescribes: reconstruct, forward
       project, remove the best rigid shift + gain + offset per angle, and ask whether
       what is left is both large and *localised* (deformation is localised to
       sub-regions; noise, a missing wedge and a mediocre reconstruction are spread
       across the detector). Honest about its own weakness: filtered backprojection
       partially absorbs inconsistent data into streaks that reproject back onto the
       measurement, so this statistic is much less sensitive than it looks.

    Either can fire. Both are last in the triage order for the same reason: a phase
    offset (mode 11) and truncation (mode 12) also destroy mass conservation, and every
    uncorrected rigid error also leaves a high reprojection residual, so a firing here
    before the earlier stages are cleared means nothing.

    The internal reconstruction is a plain filtered backprojection and needs a near
    complete 180 deg span; pass ``volume`` to use a real reconstruction instead. Without
    either, statistic 1 is still reported.
    """
    name, stage = "deformation", TriageStage.NON_RIGID
    cfg = config or DiagnosticConfig()
    mom = _moments(projections, moments, chunk=chunk)
    theta_rad, _ = _theta_radians(theta, mom.n_theta, theta_units)
    assumed = (mom.n_u - 1) / 2.0 if center is None else float(center)
    span = _span_deg(theta_rad)
    sign = -1.0 if mom.inverted else 1.0

    # (1) mass conservation
    mean_mass = float(np.mean(mom.mass))
    if mean_mass <= 0:
        return _na(name, stage, "the stack has no positive mass")
    normalised = mom.mass / mean_mass
    mass_trend, mass_white = _split_trend(normalised, degree=2)
    mass_white_frac = float(np.nanstd(mass_white))
    metrics: dict[str, float] = {
        "mass_white_frac": mass_white_frac,
        "mass_trend_frac": float(np.ptp(mass_trend[np.isfinite(mass_trend)]))
        if np.isfinite(mass_trend).any()
        else float("nan"),
    }
    curves: dict[str, np.ndarray] = {"projected_mass": normalised, "mass_white": mass_white}

    # (2) reprojection residual
    reproj_detail = ""
    conf_reproj = 0.0
    row = _reference_row(mom)
    step = max(1, int(math.ceil(mom.n_theta / cfg.max_recon_angles)))
    idx = np.arange(0, mom.n_theta, step)
    simulated: np.ndarray | None = None
    measured: np.ndarray | None = None
    source = ""

    if volume is not None:
        vol = np.asarray(volume)
        if vol.ndim != 3 or vol.shape[1] != vol.shape[2]:
            reproj_detail = f" Reprojection skipped: volume must be (n_slices, N, N), got {vol.shape}."
        elif vol.shape[0] != mom.n_v or vol.shape[1] != mom.n_u:
            reproj_detail = (
                f" Reprojection skipped: volume {vol.shape} does not match projections "
                f"(n_v={mom.n_v}, n_u={mom.n_u}); crop or pad it to the projection grid first."
            )
        else:
            measured = np.stack(
                [sign * np.asarray(projections[int(i), row], dtype=np.float64) for i in idx]
            )
            simulated = forward_project_slice(vol[row], theta_rad[idx], center=assumed)
            source = f"supplied volume slice {row}"
            metrics["bin_factor"] = 1.0
    elif not allow_reconstruction:
        reproj_detail = (
            " Reprojection skipped: reconstruction-based probes are switched off "
            "(allow_reconstruction=False) and no volume was supplied."
        )
    elif span < 150.0:
        reproj_detail = (
            f" Reprojection skipped: span is {span:.0f} deg and the internal filtered "
            "backprojection needs >= 150 deg; pass volume= to use a real reconstruction."
        )
    else:
        measured, theta_used, factor = _prepare_slice_sinogram(
            projections, theta_rad, mom, cfg, band=band
        )
        binned_center = (assumed + 0.5) / factor - 0.5
        # No need to try both senses of theta: reconstruct-then-reproject with a
        # consistently negated angle axis reconstructs the mirror image and reprojects
        # it back to the same sinogram, so the residual is identical. The residual sees
        # data inconsistency, never the handedness of the geometry.
        recon = fbp_slice(measured, theta_used, center=binned_center)
        simulated = forward_project_slice(
            recon, theta_used, center=binned_center, n_u=measured.shape[1]
        )
        source = f"internal FBP of detector row {row} (binned {factor:.0f}x)"
        metrics["bin_factor"] = factor

    if simulated is not None and measured is not None:
        fractions, median_frac, energy = _residual_stats(measured, simulated)
        order = np.sort(energy)[::-1]
        top = max(1, int(round(0.1 * energy.size)))
        locality = float(order[:top].sum() / order.sum()) if order.sum() > 0 else float("nan")
        metrics.update(
            {
                "residual_frac_median": median_frac,
                "residual_frac_p90": float(np.nanpercentile(fractions, 90)),
                "locality": locality,
                "detector_row": float(row),
                "n_angles_used": float(measured.shape[0]),
            }
        )
        curves.update({"residual_frac": fractions, "residual_energy": energy})
        conf_reproj = min(
            confidence_from_ratio(median_frac, cfg.deformation_residual_frac),
            confidence_from_ratio(locality, cfg.deformation_locality),
        )
        reproj_detail = (
            f" After the best rigid shift, gain and offset the reprojection residual is "
            f"{median_frac:.1%} of the projection's own rms ({source}), with {locality:.0%} of its "
            "energy in the busiest 10% of detector columns (uniform noise would give 10%)."
        )

    conf_mass = confidence_from_ratio(mass_white_frac, cfg.deformation_mass_frac)
    detail = (
        f"projected mass varies by {mass_white_frac:.2%} rms about its smooth trend "
        f"(tolerance {cfg.deformation_mass_frac:.1%}), which parallel-beam geometry says should be "
        "zero for a rigid, enclosed object." + reproj_detail
    )
    conf = max(conf_mass, conf_reproj)
    if conf <= 0.0:
        return _clear(name, stage, metrics, detail=detail, curves=curves)
    detail += (
        " Trust this only once modes 11, 12, 1, 3, 4 and 5 are cleared: a phase offset and "
        "truncation also break mass conservation, and every uncorrected rigid error also "
        "leaves a high reprojection residual."
    )
    evidence = {"mass_white_frac": mass_white_frac}
    if "residual_frac_median" in metrics:
        evidence["residual_frac_median"] = metrics["residual_frac_median"]
        evidence["locality"] = metrics["locality"]
    return ProbeResult(
        probe=name, stage=stage, status=ProbeStatus.FIRED, metrics=metrics, detail=detail,
        curves=curves,
        findings=(
            Finding(mode=FailureMode.DEFORMATION, confidence=conf, probe=name, detail=detail,
                    evidence=evidence),
        ),
    )


# --------------------------------------------------------------------------------------
# the runner: every probe, in the roadmap's order
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    projections: Any
    theta: np.ndarray  # radians
    moments: StackMoments
    config: DiagnosticConfig
    center: float
    volume: np.ndarray | None
    allow_reconstruction: bool


def _needs_recon(fn: Callable[[_Context], ProbeResult], name: str) -> Callable[[_Context], ProbeResult]:
    def wrapped(ctx: _Context) -> ProbeResult:
        if not ctx.allow_reconstruction and ctx.volume is None:
            return _na(
                name,
                _PROBE_STAGES[name],
                "reconstruction-based probes are switched off (allow_reconstruction=False) "
                "and no volume was supplied",
            )
        return fn(ctx)

    return wrapped


_PROBE_STAGES = {
    "vacuum_phase": TriageStage.DATA_INTEGRITY,
    "truncation": TriageStage.DATA_INTEGRITY,
    "angular_coverage": TriageStage.COVERAGE,
    "center_consistency": TriageStage.ROTATION_CENTRE,
    "axis_tilt": TriageStage.ROTATION_CENTRE,
    "center_sweep": TriageStage.ROTATION_CENTRE,
    "vertical_drift": TriageStage.VERTICAL,
    "shift_jitter": TriageStage.HORIZONTAL,
    "angle_readback": TriageStage.HORIZONTAL,
    "scale_drift": TriageStage.HORIZONTAL,
    "deformation": TriageStage.NON_RIGID,
}


#: Every probe, in the order the roadmap prescribes. ``triage`` walks this list and
#: stops at the first probe that fires; ``diagnose`` runs all of them.
PROBES: tuple[tuple[str, TriageStage, Callable[[_Context], ProbeResult]], ...] = (
    # Truncation comes before the ramp fit, not after: if the object runs off the frame
    # then the "presumed-vacuum border" the ramp is fitted on is object, and mode 11
    # will fire on a frame that has no vacuum in it at all.
    (
        "truncation",
        TriageStage.DATA_INTEGRITY,
        lambda c: probe_truncation(c.projections, config=c.config, moments=c.moments),
    ),
    (
        "vacuum_phase",
        TriageStage.DATA_INTEGRITY,
        lambda c: probe_vacuum_phase(c.projections, config=c.config, moments=c.moments),
    ),
    (
        "angular_coverage",
        TriageStage.COVERAGE,
        lambda c: probe_angular_coverage(c.theta, config=c.config, theta_units="rad"),
    ),
    (
        "center_consistency",
        TriageStage.ROTATION_CENTRE,
        lambda c: probe_center_consistency(
            c.projections, c.theta, center=c.center, config=c.config, moments=c.moments,
            theta_units="rad",
        ),
    ),
    (
        "axis_tilt",
        TriageStage.ROTATION_CENTRE,
        lambda c: probe_axis_tilt(
            c.projections, c.theta, center=c.center, config=c.config, moments=c.moments,
            theta_units="rad",
        ),
    ),
    (
        "center_sweep",
        TriageStage.ROTATION_CENTRE,
        _needs_recon(
            lambda c: probe_center_sweep(
                c.projections, c.theta, center=c.center, config=c.config, moments=c.moments,
                theta_units="rad",
            ),
            "center_sweep",
        ),
    ),
    (
        "vertical_drift",
        TriageStage.VERTICAL,
        lambda c: probe_vertical_drift(c.projections, config=c.config, moments=c.moments),
    ),
    (
        "shift_jitter",
        TriageStage.HORIZONTAL,
        lambda c: probe_shift_jitter(
            c.projections, c.theta, config=c.config, moments=c.moments, theta_units="rad"
        ),
    ),
    (
        "angle_readback",
        TriageStage.HORIZONTAL,
        lambda c: probe_angle_readback(
            c.projections, c.theta, config=c.config, moments=c.moments, theta_units="rad"
        ),
    ),
    (
        "scale_drift",
        TriageStage.HORIZONTAL,
        lambda c: probe_scale_drift(
            c.projections, c.theta, center=c.center, config=c.config, moments=c.moments,
            theta_units="rad",
        ),
    ),
    (
        "deformation",
        TriageStage.NON_RIGID,
        lambda c: probe_deformation(
            c.projections, c.theta, center=c.center, volume=c.volume, config=c.config,
            moments=c.moments, theta_units="rad",
            allow_reconstruction=c.allow_reconstruction,
        ),
    ),
)


def diagnose(
    projections: Any,
    theta: Any,
    *,
    volume: np.ndarray | None = None,
    center: float | None = None,
    config: DiagnosticConfig | None = None,
    theta_units: str = "auto",
    allow_reconstruction: bool = True,
    stop_at_first: bool = False,
    only: Sequence[str] | None = None,
    chunk: int = 32,
    moments: StackMoments | None = None,
) -> Verdict:
    """Run the probes and return a ranked, JSON-serialisable :class:`Verdict`.

    Parameters
    ----------
    projections:
        ``(n_theta, n_v, n_u)`` array-like of *phase* projections. An ``h5py`` dataset
        or memmap is fine -- the moment pass reads it in chunks. The sign is handled
        for you (see :func:`stack_moments`).
    theta:
        Rotation angles, one per projection. See ``theta_units``.
    volume:
        An existing reconstruction, ``(n_v, n_u, n_u)``. Used by the reprojection
        probes in place of the module's own filtered backprojection; supplying one both
        speeds them up and makes them trust *your* geometry.
    center:
        The rotation-axis position you intend to reconstruct with, in pixels along u.
        Defaults to the detector midpoint ``(n_u - 1) / 2``, which is what "I have not
        thought about it yet" looks like -- and mode 1 is exactly the failure of that
        assumption.
    stop_at_first:
        Stop at the first probe that fires (see :func:`triage`), leaving the rest
        ``SKIPPED``. Off here, on there.
    only:
        Restrict to these probe names, for a quick re-check after a fix.

    Notes
    -----
    A probe that raises is recorded as ``ProbeStatus.ERROR`` with its message and the
    run continues; a probe whose preconditions are missing is ``NOT_APPLICABLE`` with
    the reason. Read :attr:`Verdict.coverage` before reading an empty finding list as
    good news.
    """
    cfg = config or DiagnosticConfig()
    n_theta, n_v, n_u = _check_stack(projections)
    theta_rad, units_used = _theta_radians(theta, n_theta, theta_units)
    mom = _moments(projections, moments, chunk=chunk, border=cfg.border)
    assumed = (n_u - 1) / 2.0 if center is None else float(center)

    ctx = _Context(
        projections=projections,
        theta=theta_rad,
        moments=mom,
        config=cfg,
        center=assumed,
        volume=volume,
        allow_reconstruction=allow_reconstruction,
    )

    selected = set(only) if only is not None else None
    if selected is not None:
        unknown = selected - {name for name, _, _ in PROBES}
        if unknown:
            raise ValueError(f"unknown probe(s) {sorted(unknown)}; available: {[p[0] for p in PROBES]}")

    results: list[ProbeResult] = []
    findings: list[Finding] = []
    stopped_at: str | None = None

    for name, stage, run in PROBES:
        if selected is not None and name not in selected:
            continue
        if stopped_at is not None:
            results.append(
                ProbeResult(
                    probe=name, stage=stage, status=ProbeStatus.SKIPPED,
                    reason=f"triage stopped at {stopped_at}",
                )
            )
            continue
        try:
            result = run(ctx)
        except Exception as exc:  # noqa: BLE001 - a broken probe must not hide the others
            result = ProbeResult(
                probe=name, stage=stage, status=ProbeStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        findings.extend(result.findings)
        if stop_at_first and result.fired:
            stopped_at = name

    context = {
        "n_theta": n_theta,
        "n_v": n_v,
        "n_u": n_u,
        "theta_units": units_used,
        "span_deg": _span_deg(theta_rad),
        "assumed_center_px": assumed,
        "sign_inverted": mom.inverted,
        "contrast": mom.contrast,
        "volume_supplied": volume is not None,
        "allow_reconstruction": allow_reconstruction,
        "config": cfg.to_dict(),
    }
    return Verdict(
        findings=rank(apply_stage_discount(findings)),
        probes=tuple(results),
        stopped_at=stopped_at,
        context=context,
    )


def triage(projections: Any, theta: Any, **kwargs) -> Verdict:
    """:func:`diagnose` in the roadmap's order, stopping at the first thing that fires.

    Order of operations, never to be violated: ramp/offset removal, angular coverage,
    rotation centre, vertical, horizontal, and only then non-rigid. Each stage's fix
    invalidates every measurement after it -- correcting a phase ramp moves every
    centroid, and correcting the centre changes every reprojection residual -- so
    reporting the whole list at once invites the caller to chase the wrong number.
    Fix what fired, run again.
    """
    kwargs.setdefault("stop_at_first", True)
    return diagnose(projections, theta, **kwargs)

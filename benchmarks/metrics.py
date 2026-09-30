"""Scoring for alignment benchmarks: shift recovery first, FSC and residuals second.

The primary metric is :func:`score_shifts` -- recovered shift versus known injected
truth. The secondaries (:func:`fourier_shell_correlation`,
:func:`reprojection_residual`) exist because a single number never explains a
failure, but they are secondary for a reason spelled out in
:func:`fourier_shell_correlation`'s docstring and pinned by a test: **FSC cannot see
a systematic geometric bias at all.**

The gauge is the part everyone gets wrong, so it is the first thing in the file.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DX_GAUGE",
    "DY_GAUGE",
    "FscResult",
    "Plateau",
    "ResidualMap",
    "ShiftRecovery",
    "Timer",
    "Timing",
    "fourier_shell_correlation",
    "gauge_basis",
    "remove_gauge",
    "reprojection_residual",
    "residual_plateau",
    "score_shifts",
    "split_half_indices",
]


# --------------------------------------------------------------------------------
# The gauge: which parts of a shift estimate are physically unobservable
# --------------------------------------------------------------------------------
#
# Getting this wrong makes a perfect aligner look broken. Two shift vectors that
# differ by a gauge mode describe the *same* object, reconstructed in a different
# place, and no algorithm operating on the projections could ever prefer one over the
# other -- the data are identical.
#
#   Vertical (dy): translating the object along the rotation axis by t shifts every
#       projection by the same t. One mode: {1}.
#
#   Horizontal (dx): translating the object in the rotation plane by (X, Y) shifts
#       projection i by X cos(theta_i) + Y sin(theta_i) -- a two-parameter family --
#       and on top of that a constant dx is exactly the rotation-axis position, which
#       the reconstruction estimates separately. Three modes: {1, sin, cos}.
#
# Removing only the mean, which is the intuitive thing to do, leaves two of the three
# horizontal modes in. TKtomo's own tests/test_ptycho_engine.py measured a perfectly
# correct alignment scoring ~0.2 px under mean-only removal. So :func:`score_shifts`
# reports the gauge-removed number as primary AND the mean-only and raw numbers
# beside it, and states the removed amplitude, so nothing is hidden by the choice.

DY_GAUGE: tuple[str, ...] = ("const",)
DX_GAUGE: tuple[str, ...] = ("const", "sin", "cos")


def gauge_basis(angles: np.ndarray, modes: Sequence[str]) -> np.ndarray:
    """Design matrix ``(n_angles, n_modes)`` for the named unobservable modes."""
    angles = np.asarray(angles, dtype=np.float64)
    columns = []
    for mode in modes:
        if mode == "const":
            columns.append(np.ones_like(angles))
        elif mode == "sin":
            columns.append(np.sin(angles))
        elif mode == "cos":
            columns.append(np.cos(angles))
        elif mode == "linear":
            # Not a physical gauge mode -- offered only for deliberate experiments,
            # e.g. asking how much of an error is a slow drift. Never in DY/DX_GAUGE.
            columns.append(np.linspace(-1.0, 1.0, angles.size))
        else:
            raise ValueError(f"Unknown gauge mode {mode!r}")
    if not columns:
        return np.zeros((angles.size, 0))
    return np.column_stack(columns)


def remove_gauge(
    values: np.ndarray, angles: np.ndarray, modes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``values`` onto the complement of the gauge modes.

    Returns ``(observable_part, coefficients)``. ``coefficients`` is what was removed,
    in the basis ``modes`` -- report it, because "we forgave 4.2 px of constant
    offset" is a materially different statement from "we forgave 0.01 px".
    """
    values = np.asarray(values, dtype=np.float64)
    basis = gauge_basis(angles, modes)
    if basis.shape[1] == 0:
        return values.copy(), np.zeros(0)
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return values - basis @ coefficients, coefficients


# --------------------------------------------------------------------------------
# PRIMARY METRIC: shift recovery against known truth
# --------------------------------------------------------------------------------


@dataclass
class ShiftRecovery:
    """How well an aligner recovered the injected truth. All lengths in pixels."""

    n_angles: int
    # Primary: after removing the unobservable modes (DY_GAUGE / DX_GAUGE).
    rms_dy: float
    rms_dx: float
    max_dy: float
    max_dx: float
    # Stricter: only the constant removed from each axis.
    rms_dy_mean_only: float
    rms_dx_mean_only: float
    # Rawest: nothing removed. Large values here with small primary values mean the
    # aligner found a different-but-equally-valid placement of the volume.
    rms_dy_raw: float
    rms_dx_raw: float
    # What the gauge removal forgave, in px RMS.
    gauge_amplitude_dy: float
    gauge_amplitude_dx: float
    gauge_coefficients_dy: list[float]
    gauge_coefficients_dx: list[float]
    # Physical units and the target.
    pixel_size_nm: float
    voxel_nm: float
    target_fraction: float
    target_px: float
    rms_dy_nm: float
    rms_dx_nm: float
    rms_dy_voxels: float
    rms_dx_voxels: float
    meets_target_dy: bool
    meets_target_dx: bool
    # For reference: the misalignment that was there to begin with.
    injected_rms_dy: float
    injected_rms_dx: float

    @property
    def meets_target(self) -> bool:
        return self.meets_target_dy and self.meets_target_dx

    @property
    def improvement_dy(self) -> float:
        """Injected RMS divided by residual RMS. 1.0 means the aligner did nothing."""
        return self.injected_rms_dy / self.rms_dy if self.rms_dy else math.inf

    @property
    def improvement_dx(self) -> float:
        return self.injected_rms_dx / self.rms_dx if self.rms_dx else math.inf

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["meets_target"] = self.meets_target
        out["improvement_dy"] = self.improvement_dy
        out["improvement_dx"] = self.improvement_dx
        return out


def score_shifts(
    sy: np.ndarray,
    sx: np.ndarray,
    truth_dy: np.ndarray,
    truth_dx: np.ndarray,
    angles: np.ndarray,
    *,
    pixel_size_nm: float = 1.0,
    voxel_nm: float | None = None,
    target_fraction: float = 1.0 / 3.0,
) -> ShiftRecovery:
    """Score recovered shifts ``(sy, sx)`` against the injected truth.

    Sign convention (see :mod:`benchmarks.phantom`): ``truth_dy``/``truth_dx`` are the
    content displacement that was injected, and a correct aligner reports the **same
    sign**, because TKtomo's ``apply_shifts`` moves content by ``-s``. If this call
    returns an RMS of very nearly twice the injected RMS, the aligner's sign is
    flipped -- that is the signature, and it is a much more common bug than a
    genuinely 2x-worse algorithm.

    ``voxel_nm`` defaults to ``pixel_size_nm``: the roadmap's target is "residual
    alignment error at or below 1/3 of the *target voxel*", and when you are
    reconstructing on the detector grid the target voxel is the detector pixel, so
    1/3 voxel is 0.333 px. Pass ``voxel_nm`` explicitly when the intended
    reconstruction grid is coarser or finer than the detector -- for our own scan,
    74.50973137 nm, one third is 0.333 px.
    """
    sy = np.asarray(sy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    truth_dy = np.asarray(truth_dy, dtype=np.float64)
    truth_dx = np.asarray(truth_dx, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    for label, array in (("sy", sy), ("sx", sx), ("truth_dy", truth_dy), ("truth_dx", truth_dx)):
        if array.shape != angles.shape:
            raise ValueError(
                f"{label} has shape {array.shape} but there are {angles.size} angles"
            )
    if not (np.all(np.isfinite(sy)) and np.all(np.isfinite(sx))):
        raise ValueError(
            "the estimated shifts contain non-finite values; the aligner diverged and "
            "there is no meaningful score to report"
        )

    error_dy = sy - truth_dy
    error_dx = sx - truth_dx

    obs_dy, coeff_dy = remove_gauge(error_dy, angles, DY_GAUGE)
    obs_dx, coeff_dx = remove_gauge(error_dx, angles, DX_GAUGE)
    mean_dy = error_dy - error_dy.mean()
    mean_dx = error_dx - error_dx.mean()

    voxel_nm = pixel_size_nm if voxel_nm is None else voxel_nm
    # The target is a fraction of a voxel; express it in detector pixels, which is what
    # the shifts are measured in.
    target_px = target_fraction * voxel_nm / pixel_size_nm

    rms = lambda a: float(np.sqrt(np.mean(a**2)))  # noqa: E731
    rms_dy, rms_dx = rms(obs_dy), rms(obs_dx)

    return ShiftRecovery(
        n_angles=int(angles.size),
        rms_dy=rms_dy,
        rms_dx=rms_dx,
        max_dy=float(np.max(np.abs(obs_dy))),
        max_dx=float(np.max(np.abs(obs_dx))),
        rms_dy_mean_only=rms(mean_dy),
        rms_dx_mean_only=rms(mean_dx),
        rms_dy_raw=rms(error_dy),
        rms_dx_raw=rms(error_dx),
        gauge_amplitude_dy=rms(error_dy - obs_dy),
        gauge_amplitude_dx=rms(error_dx - obs_dx),
        gauge_coefficients_dy=[float(c) for c in coeff_dy],
        gauge_coefficients_dx=[float(c) for c in coeff_dx],
        pixel_size_nm=float(pixel_size_nm),
        voxel_nm=float(voxel_nm),
        target_fraction=float(target_fraction),
        target_px=float(target_px),
        rms_dy_nm=rms_dy * pixel_size_nm,
        rms_dx_nm=rms_dx * pixel_size_nm,
        rms_dy_voxels=rms_dy * pixel_size_nm / voxel_nm,
        rms_dx_voxels=rms_dx * pixel_size_nm / voxel_nm,
        meets_target_dy=bool(rms_dy <= target_px),
        meets_target_dx=bool(rms_dx <= target_px),
        injected_rms_dy=rms(truth_dy - truth_dy.mean()),
        injected_rms_dx=rms(remove_gauge(truth_dx, angles, DX_GAUGE)[0]),
    )


# --------------------------------------------------------------------------------
# SECONDARY: split-half FSC
# --------------------------------------------------------------------------------


def split_half_indices(n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Even / odd angle subsets, the standard split-data partition.

    Interleaving rather than splitting the scan in half is what makes the two
    half-sets cover the same angular range; a first-half/second-half split gives two
    limited-angle reconstructions whose FSC measures the missing wedge, not the
    resolution.
    """
    index = np.arange(int(n_angles))
    return index[0::2], index[1::2]


@dataclass
class FscResult:
    frequency: np.ndarray  # cycles per pixel, 0 .. 0.5
    fsc: np.ndarray
    threshold: np.ndarray
    n_voxels: np.ndarray  # samples per shell, for the half-bit curve
    resolution_px: float  # period at the first crossing; inf if never crossed
    resolution_nm: float
    criterion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "resolution_px": self.resolution_px,
            "resolution_nm": self.resolution_nm,
            "frequency": self.frequency.tolist(),
            "fsc": self.fsc.tolist(),
            "threshold": self.threshold.tolist(),
        }


def fourier_shell_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    pixel_size_nm: float = 1.0,
    criterion: str = "half-bit",
    n_shells: int | None = None,
    mask: np.ndarray | None = None,
) -> FscResult:
    """Shell-averaged correlation between two half-set reconstructions (2D or 3D).

    **This metric is blind to systematic geometric bias, and that is not a subtlety
    but the main thing to know about it.** A common-mode translation ``d`` applied to
    both half-sets multiplies their transforms by ``exp(-2 pi i k.d)`` and
    ``exp(+2 pi i k.d)``; the two cancel exactly in the cross term and leave every
    shell's correlation *bit-identical*. The same argument covers any rigid geometry
    error applied identically to both halves -- a wrong rotation centre, a wrong
    tilt, a wrong magnification. We measured this directly on a phantom: the half-bit
    FRC read 508.6 nm at centring errors of 0, 4, 8, 16, 32 and 64 px while the true
    edge blur grew to 128 px. ``tests/test_benchmark.py`` pins the invariance to
    machine precision.

    So never report an FSC on its own. Pair it with the shift-recovery error against
    known truth (:func:`score_shifts`) and with a reprojection-residual map
    (:func:`reprojection_residual`), which *do* see common-mode error.

    ``criterion`` is ``"half-bit"`` (van Heel & Schatz), ``"0.143"`` or ``"0.5"``.
    ``mask`` (a boolean array the shape of the inputs) is applied to both volumes
    first; masking to the sample and away from vacuum is what stops an empty border
    from inflating the correlation.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"half-set shapes differ: {a.shape} vs {b.shape}")
    if a.ndim not in (2, 3):
        raise ValueError(f"FSC needs a 2D or 3D array; got {a.ndim}D")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask shape {mask.shape} does not match data {a.shape}")
        a = a * mask
        b = b * mask

    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)

    grids = [np.fft.fftfreq(n) for n in a.shape]
    radius = np.sqrt(sum(g.reshape([-1 if i == j else 1 for j in range(a.ndim)]) ** 2
                         for i, g in enumerate(grids)))

    n_shells = n_shells or int(min(a.shape) // 2)
    edges = np.linspace(0.0, 0.5, n_shells + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, n_shells - 1)
    inside = radius.ravel() <= 0.5

    cross = np.bincount(index[inside], weights=(fa * np.conj(fb)).ravel()[inside].real, minlength=n_shells)
    power_a = np.bincount(index[inside], weights=(np.abs(fa) ** 2).ravel()[inside], minlength=n_shells)
    power_b = np.bincount(index[inside], weights=(np.abs(fb) ** 2).ravel()[inside], minlength=n_shells)
    counts = np.bincount(index[inside], minlength=n_shells).astype(np.float64)

    denominator = np.sqrt(power_a * power_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsc = np.where(denominator > 0, cross / denominator, 0.0)

    frequency = 0.5 * (edges[:-1] + edges[1:])
    threshold = _fsc_threshold(criterion, counts)
    resolution_px = _first_crossing(frequency, fsc, threshold)

    return FscResult(
        frequency=frequency,
        fsc=fsc,
        threshold=threshold,
        n_voxels=counts,
        resolution_px=resolution_px,
        resolution_nm=resolution_px * pixel_size_nm,
        criterion=criterion,
    )


def _fsc_threshold(criterion: str, counts: np.ndarray) -> np.ndarray:
    if criterion == "0.143":
        return np.full(counts.shape, 0.143)
    if criterion == "0.5":
        return np.full(counts.shape, 0.5)
    if criterion != "half-bit":
        raise ValueError(f"criterion must be 'half-bit', '0.143' or '0.5'; got {criterion!r}")
    n = np.maximum(counts, 1.0)
    root = np.sqrt(n)
    return (0.2071 + 1.9102 / root) / (1.2071 + 0.9102 / root)


def _first_crossing(frequency: np.ndarray, fsc: np.ndarray, threshold: np.ndarray) -> float:
    """Period (px) where the FSC first falls below its threshold, linearly interpolated."""
    below = fsc < threshold
    # Skip the DC shell: it is 1.0 by construction and carries no information.
    for i in range(1, below.size):
        if below[i]:
            if i == 0:
                return math.inf
            f0, f1 = frequency[i - 1], frequency[i]
            d0 = fsc[i - 1] - threshold[i - 1]
            d1 = fsc[i] - threshold[i]
            frac = d0 / (d0 - d1) if d0 != d1 else 0.0
            crossing = f0 + frac * (f1 - f0)
            return float(1.0 / crossing) if crossing > 0 else math.inf
    return math.inf  # never crossed: resolution-limited by the sampling, not the data


# --------------------------------------------------------------------------------
# SECONDARY: reprojection residual versus angle
# --------------------------------------------------------------------------------


@dataclass
class ResidualMap:
    """Per-angle reprojection residual -- the diagnostic FSC cannot give you.

    A rigid-alignment failure and a genuine non-rigid deformation both leave the
    residual on a plateau, but they look different across angle: a rigid error is
    *structured* in angle (it is a smooth function of the geometry, so neighbouring
    projections fail together, and :attr:`lag1` stays high), whereas a deformation
    that varies projection to projection leaves a residual that is high but
    angularly *uncorrelated*. :attr:`lag1` and :attr:`peakiness` are the two numbers
    that separate them; neither is a proof, both are evidence.
    """

    per_angle: np.ndarray
    total: float
    mean: float
    std: float
    worst_angle_index: int
    lag1: float  # lag-1 autocorrelation of per_angle across the scan
    peakiness: float  # (max - median) / median

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "mean": self.mean,
            "std": self.std,
            "worst_angle_index": self.worst_angle_index,
            "lag1": self.lag1,
            "peakiness": self.peakiness,
            "per_angle": self.per_angle.tolist(),
        }


def reprojection_residual(
    measured: np.ndarray,
    simulated: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    normalize: bool = True,
) -> ResidualMap:
    """Relative residual ``||measured - simulated|| / ||measured||``, per angle and total.

    ``mask`` is a 2D boolean frame mask, applied to every projection -- use it to
    exclude a border that alignment shifts have pulled zeros into, which otherwise
    dominates the residual and hides the real signal.
    """
    measured = np.array(measured, dtype=np.float64, copy=True)
    simulated = np.array(simulated, dtype=np.float64, copy=True)
    if measured.shape != simulated.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {simulated.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != measured.shape[1:]:
            raise ValueError(f"mask shape {mask.shape} does not match frames {measured.shape[1:]}")
        measured = measured[:, mask]
        simulated = simulated[:, mask]

    # Flatten every frame to one row and reduce along a single contiguous axis. The
    # obvious `np.sum(..., axis=(1, 2))` form is avoided deliberately: the totals it
    # produced here disagreed with `np.linalg.norm` of the very same difference array
    # by two orders of magnitude (per-frame norms came back exactly equal to the
    # per-frame norms of `measured`, as if `simulated` were zero, while the global norm
    # of the same expression was right). Whatever the cause -- this runs on a very new
    # Python/NumPy pair -- a metric that reports a converging reconstruction as
    # diverging is worse than no metric, so the reduction is done once, on an owned
    # C-contiguous 2-D copy, and `total` is derived FROM the per-frame values rather
    # than computed separately. The two can then never disagree.
    n_frames = measured.shape[0]
    flat_measured = np.ascontiguousarray(measured.reshape(n_frames, -1))
    flat_difference = np.ascontiguousarray((measured - simulated).reshape(n_frames, -1))

    difference = np.linalg.norm(flat_difference, axis=1)
    scale = (
        np.linalg.norm(flat_measured, axis=1) if normalize else np.ones_like(difference)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        per_angle = np.where(scale > 0, difference / scale, np.nan)

    total_scale = float(np.sqrt(np.sum(scale**2))) if normalize else 1.0
    total = (
        float(np.sqrt(np.sum(difference**2)) / total_scale) if total_scale else math.nan
    )

    finite = per_angle[np.isfinite(per_angle)]
    if finite.size < 2:
        lag1 = math.nan
    else:
        centred = finite - finite.mean()
        denominator = float(np.sum(centred**2))
        lag1 = float(np.sum(centred[:-1] * centred[1:]) / denominator) if denominator else math.nan
    median = float(np.median(finite)) if finite.size else math.nan

    return ResidualMap(
        per_angle=per_angle,
        total=total,
        mean=float(np.nanmean(per_angle)),
        std=float(np.nanstd(per_angle)),
        worst_angle_index=int(np.nanargmax(per_angle)) if finite.size else -1,
        lag1=lag1,
        peakiness=float((finite.max() - median) / median) if finite.size and median else math.nan,
    )


@dataclass
class Plateau:
    """Where a residual history stopped improving, and by how little."""

    plateaued: bool
    iteration: int | None  # 1-based index of the first iteration on the plateau
    value: float
    tail_relative_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def residual_plateau(
    history: Sequence[float], *, rel_tol: float = 0.01, patience: int = 3
) -> Plateau:
    """First iteration after which the residual improves by less than ``rel_tol`` for
    ``patience`` consecutive iterations.

    A plateau is not a success. It says the loop has extracted everything its *model*
    can explain; if the plateau value is far above the noise floor, what remains is
    something the model cannot represent -- a tilt, a magnification drift, a genuine
    deformation -- and more iterations will not touch it.
    """
    values = [float(v) for v in history]
    if len(values) < patience + 1:
        return Plateau(False, None, values[-1] if values else math.nan, math.nan)

    for start in range(1, len(values) - patience + 1):
        window = values[start - 1 : start + patience]
        improvements = [
            (window[i] - window[i + 1]) / window[i] if window[i] else 0.0
            for i in range(len(window) - 1)
        ]
        if all(imp < rel_tol for imp in improvements):
            return Plateau(True, start, values[start - 1], float(np.mean(improvements)))

    tail = values[-patience - 1 :]
    improvement = (tail[0] - tail[-1]) / tail[0] if tail[0] else math.nan
    return Plateau(False, None, values[-1], float(improvement))


# --------------------------------------------------------------------------------
# Wall clock
# --------------------------------------------------------------------------------


@dataclass
class Timing:
    wallclock_s: float
    iterations: int
    per_iteration_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.per_iteration_s = self.wallclock_s / self.iterations if self.iterations else math.nan

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Timer:
    """``with Timer() as t: ...`` then ``t.seconds``. Uses ``perf_counter``."""

    def __init__(self) -> None:
        self.seconds = math.nan
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._start

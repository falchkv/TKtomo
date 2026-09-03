"""Gradient-domain registration: the trick that makes ptycho-tomo alignment work.

Ptychographic phase retrieval determines the object's phase only up to a **constant
offset** and a **linear ramp**. Both are exact gauge freedoms of the reconstruction --
different projections in the same scan come back with different, unknowable values of
them -- and a linear phase ramp across a projection is *mathematically identical to a
lateral translation of that projection*. So a registration performed on the phase
itself cannot distinguish "the sample moved" from "the phase retrieval picked a
different gauge", and it will happily spend iterations correcting a shift that is not
there. Ramp removal (:func:`~tktomo.ptycho_align.core.preprocess.remove_phase_ramp`)
attacks this from the other end, but it only ever removes the ramp it can *see* in a
presumed-vacuum border, and any residual goes straight into the alignment estimate.

Differentiating removes both ambiguities structurally rather than by estimation:

    d/dx [ phi(x, y) + a*x + b*y + c ]  =  d/dx phi(x, y)  +  a

The constant ``c`` differentiates to **exactly zero**, and the ramp differentiates to
the **constant** ``a``, which the registration then removes by subtracting the mean of
the gradient image before correlating. What is left is invariant to the gauge, to
machine precision -- not approximately, not after tuning a border width, but exactly.

Be precise about which ambiguity each step buys, because the two are not equally hard.
A *constant* offset is already harmless to a cross-correlation: it lands entirely in
the DC bin, and both this module and skimage's ``phase_cross_correlation`` remove it
(mean subtraction here, and it does not move the peak there). The *ramp* is the one
that actually bites, and only differentiation removes it. Measured on a 96 x 96
synthetic projection with a known -2.70 px shift and a ramp of ``A`` radians across
the frame (``tests/test_gradient_registration.py`` regenerates this table):

    ramp A      value-domain error      gradient-x error
    0                +4e-16 px               +4e-16 px
    0.1              -0.06  px               +4e-16 px
    1                -0.59  px               +4e-16 px
    5                -3.25  px               +4e-16 px
    10              -25.7   px               +4e-16 px
    100             -28.3   px               +4e-16 px

The value-domain column is not a straw man: it is this module's own ``domain="value"``
path, which reproduces ``skimage.registration.phase_cross_correlation(..., 
normalization=None)`` -- what the engine actually calls -- to the last digit
(-0.59 px and -25.65 px at A = 1 and 10). At A = 10 it has stopped tracking the object
at all and locked onto the ramp. The gradient column does not move. That is the whole
argument for Odstrcil et al., *Opt. Express* **27**, 36637 (2019) being the reference
method for ptychographic tomography rather than the generic Gursoy reprojection loop.

Design decisions, with the reasoning rather than the assertion:

* **Which derivative: both, by default -- which is not what the reasoning predicted.**
  The a-priori argument says use ``d/dx`` alone: stage 2 estimates a *horizontal* shift,
  ``d/dx`` is the direction that displacement acts along, and it is the cheaper of the
  two. So that was the first default. Then it was measured on the synthetic reprojection
  loop in ``tests/test_odstrcil.py`` (RMS horizontal recovery error against known shifts,
  12 outer iterations of SIRT):

      registration domain        clean    ramp 1 rad   ramp 4 rad
      value                      0.081 px   0.171 px     0.548 px
      gradient-x                 0.256 px   0.277 px     0.389 px
      gradient-both              0.167 px   0.177 px     0.263 px

  ``"gradient-both"`` -- correlate ``d/dx`` and ``d/dy`` and sum the two correlation
  surfaces -- wins at every ramp amplitude, so it is the default. The extra cost is one
  more FFT pair per projection, invisible next to the reconstruction that produced the
  reprojection in the first place, and it yields a ``dy`` estimate the decoupled loop
  uses as a free cross-check on stage 1.

  The reason x-only loses is the same mechanism that makes the derivative useful.
  Differentiation is a high-pass filter (it multiplies the spectrum by ``2*pi*i*f``),
  which is what whitens the strongly low-frequency-dominated phase and sharpens the
  correlation peak -- but it also puts the comparison's weight in exactly the band where
  a half-converged SIRT reprojection is *least* faithful. Keeping both derivatives keeps
  more of the reliable mid-band. It shows in the "clean" column above, where the value
  domain is the most accurate of the three: with no gauge ambiguity at all there is
  nothing to buy and the derivative only costs bandwidth. The gradient earns its keep the
  moment a ramp exists, and the last column is the point -- once a residual ramp is
  present, only ``"gradient-both"`` still meets the 1/3-voxel accuracy target. Note that
  ``"gradient-x"`` misses it too, at 0.389 px: differentiating is necessary but choosing
  the cheaper single derivative is not sufficient.
* **Not the gradient magnitude.** ``|grad(phi + ramp)| != |grad phi| + const``: taking
  the magnitude is nonlinear and *destroys* the invariance the derivative just bought.
  ``"gradient-magnitude"`` exists only for the case of two images with opposite contrast
  sign, and using it forfeits the entire point of this module. The test suite documents
  the damage.
* **No smoothing by default.** Differentiation amplifies high-frequency noise by
  ``|2*pi*f|``, so a Gaussian pre-filter (``sigma`` ~1 px) is the usual remedy and is
  available. It is off by default because it *breaks the exactness*: a Gaussian filter
  with any finite-support boundary rule does not return a constant for a linear ramp
  within ``~3*sigma`` of the frame edge, so the ramp is no longer annihilated there.
  The tests measure exactly how much invariance each sigma costs. Turn it on for noisy
  data with a clean vacuum border. In practice the default taper hides the damage for
  a while, because it zeroes the frame border where the smoothing is inexact: measured
  on the table above with ``taper=0.1``, sigma up to 4 px still moves the estimate by
  exactly 0, and only at sigma = 8 px does the roll-off region start to leak, at
  0.39 px per 10 rad of ramp. Treat "sigma smaller than the taper width" as the
  rule. In the loop above, sigma = 1 px bought about 15% on the clean column
  (0.167 -> 0.143 px) and cost about 7% on the ramp-4 column: worth trying, not worth
  defaulting to.
* **Central differences, via ``np.gradient``.** One-sided at the two border rows and
  columns, central in the interior; *both* are exact for a linear function, which is
  precisely what preserves the invariance at the frame edge. A Sobel or a Gaussian
  derivative kernel would not be, and would leak the ramp back in through the border.
* **Order trap: subtract the mean BEFORE tapering.** The taper that suppresses the
  FFT's periodic-wrap discontinuity is a *multiplicative* window. Applying it first
  turns the ramp's constant ``a`` into ``a * w(x, y)``, which is no longer a constant
  and no longer removable. Mean first, window second. Getting this backwards silently
  reintroduces the ramp bias the module exists to eliminate, which is why
  :func:`prepare_field` does both steps itself and does not let the caller interleave.
* **A taper is safe on a gradient and not on a value.** Windowing is only
  translation-equivariant for a field that vanishes inside the roll-off. A phase
  *gradient* does vanish there -- it is zero wherever the phase is flat, i.e. in the
  vacuum -- but the phase *value* does not: a bright object reaching toward the frame
  edge is scaled differently in the two images being compared, and the peak moves.
  Measured on the same test projection, ``taper=0.1`` costs the value domain 0.6 px
  and the gradient domain nothing at all. One more reason the gradient is the better
  field to correlate, quite apart from the gauge argument.

Sign convention, identical to skimage and to the engine: ``register_translation(
reference, moving)`` returns ``(dy, dx)`` such that ``moving(v) = reference(v + d)``,
i.e. the shift you would apply to ``moving`` to bring it onto ``reference``. Passing
``(measured, simulated)`` therefore yields the correction to *add* to the cumulative
shifts, exactly as :meth:`~tktomo.ptycho_align.core.engine.AlignmentEngine.step` does.
``tests/test_gradient_registration.py`` pins this against skimage when it is installed.

Pure numpy plus scipy (scipy only for the optional Gaussian pre-filter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DOMAINS",
    "GradientConfig",
    "RAMP_INVARIANT_DOMAINS",
    "RegistrationResult",
    "image_gradient",
    "prepare_field",
    "register_stack",
    "register_translation",
    "taper_window",
]

Domain = Literal["gradient-x", "gradient-y", "gradient-both", "gradient-magnitude", "value"]

#: Every domain :func:`prepare_field` understands.
DOMAINS: tuple[str, ...] = (
    "gradient-x",
    "gradient-y",
    "gradient-both",
    "gradient-magnitude",
    "value",
)

#: The domains that annihilate a linear phase ramp, which is the ambiguity that matters.
#: A constant offset is removed by the mean subtraction in every domain including
#: ``"value"``, so it is not what distinguishes them. ``"gradient-magnitude"`` is listed
#: out because taking the magnitude is nonlinear and gives the ramp invariance back up.
RAMP_INVARIANT_DOMAINS: frozenset[str] = frozenset({"gradient-x", "gradient-y", "gradient-both"})


@dataclass(frozen=True)
class GradientConfig:
    """Parameters for gradient-domain registration."""

    domain: Domain = "gradient-both"
    #: Gaussian pre-smoothing in pixels, applied *before* differentiating. 0 keeps the
    #: ramp invariance exact; see the module docstring.
    sigma: float = 0.0
    #: Fraction of each axis given over to a raised-cosine taper (Tukey alpha), to
    #: suppress the wrap-around discontinuity the FFT sees. 0 disables it.
    taper: float = 0.1
    #: Sub-pixel precision is 1/upsample px. Matches ``AlignConfig.upsample_factor``.
    upsample: int = 20
    #: ``None`` = plain cross-correlation (what the engine uses); ``"phase"`` = phase
    #: correlation, which whitens the amplitude spectrum. Phase correlation on top of a
    #: derivative is usually over-whitening: the derivative has already flattened the
    #: spectrum, and normalising again amplifies the noise floor.
    normalization: Literal["phase"] | None = None
    #: Reject a correlation peak further than this many pixels from zero. A guard
    #: against a half-converged reprojection producing a wild lock-on; ``None`` = off.
    max_shift: float | None = None

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        if self.normalization not in (None, "phase"):
            raise ValueError(
                f"normalization must be None or 'phase', got {self.normalization!r}"
            )
        if not 0.0 <= self.taper < 1.0:
            raise ValueError(f"taper must be in [0, 1), got {self.taper}")
        if self.upsample < 1:
            raise ValueError(f"upsample must be >= 1, got {self.upsample}")

    @property
    def is_ramp_invariant(self) -> bool:
        """Does this configuration actually deliver the gauge invariance?

        False for the value domain and for the gradient magnitude, and False whenever
        smoothing is on -- smoothing degrades exactness near the frame edge, so the
        honest answer is "not exactly".
        """
        return self.domain in RAMP_INVARIANT_DOMAINS and self.sigma == 0.0


@dataclass
class RegistrationResult:
    """One pairwise registration."""

    dy: float
    dx: float
    peak: float  # correlation value at the refined peak
    quality: float  # peak / RMS of the correlation surface: >~5 is a real lock
    coarse: tuple[int, int]  # the integer-pixel peak, before refinement

    @property
    def shift(self) -> tuple[float, float]:
        return (self.dy, self.dx)


# -- field preparation --------------------------------------------------------------


def taper_window(shape: tuple[int, int], alpha: float) -> np.ndarray:
    """Separable raised-cosine (Tukey) window, 1 in the middle, 0 at the frame edge.

    The FFT treats an image as periodic, so the jump from the right edge to the left
    edge is a discontinuity that spreads a cross-shaped artefact through the whole
    correlation surface. Tapering removes it. ``alpha`` is the fraction of each axis
    spent in the cosine roll-off.
    """
    if alpha <= 0.0:
        return np.ones(shape, dtype=np.float64)

    def axis_window(n: int) -> np.ndarray:
        w = np.ones(n, dtype=np.float64)
        width = int(np.floor(alpha * (n - 1) / 2.0))
        if width < 1:
            return w
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(width + 1) / width))
        w[: width + 1] = ramp
        w[n - width - 1 :] = ramp[::-1]
        return w

    return np.outer(axis_window(shape[0]), axis_window(shape[1]))


def _smoothed(image: np.ndarray, sigma: float) -> np.ndarray:
    """Optional Gaussian pre-filter. scipy is imported lazily, per the repo's layering."""
    if sigma <= 0.0:
        return np.asarray(image, dtype=np.float64)
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    return gaussian_filter(np.asarray(image, dtype=np.float64), sigma, mode="nearest")


def image_gradient(image: np.ndarray, axis: int, sigma: float = 0.0) -> np.ndarray:
    """Partial derivative along ``axis`` (0 = rows/y, 1 = columns/x).

    ``np.gradient``'s central differences (one-sided at the borders) are exact for a
    linear function everywhere including the frame edge, which is what keeps the ramp
    invariance exact. ``sigma > 0`` applies a Gaussian pre-filter first and gives that
    exactness up in exchange for noise suppression -- see the module docstring.
    """
    return np.gradient(_smoothed(image, sigma), axis=axis)


_warned_magnitude = False


def _warn_magnitude_forfeits_invariance() -> None:
    """Say once, loudly, that this domain gives up the property the module exists for.

    ``|grad(phi + a*x + b*y + c)| != |grad phi| + const``. The magnitude is a nonlinear
    function of the derivatives, so the ramp is no longer a removable constant and the
    estimate is ramp-biased again -- by 43 px at a 100 rad ramp, measured. It is kept
    only for the case of two images with opposite contrast sign, where nothing else
    correlates at all.
    """
    global _warned_magnitude
    if not _warned_magnitude:
        _warned_magnitude = True
        logger.warning(
            "domain='gradient-magnitude' is NOT invariant to a linear phase ramp: the "
            "magnitude is a nonlinear function of the derivatives, so the ramp stops "
            "being a removable constant. Use 'gradient-both' unless the two images have "
            "opposite contrast sign."
        )


def prepare_field(image: np.ndarray, config: GradientConfig) -> list[np.ndarray]:
    """Turn one projection into the field(s) that will be correlated.

    Returns a list because ``"gradient-both"`` produces two. Each field is
    mean-subtracted (this is the step that annihilates the ramp's constant) and *then*
    tapered -- never the other way round, see the module docstring.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D projection, got shape {image.shape}")

    domain = config.domain
    if domain == "value":
        fields = [_smoothed(image, config.sigma)]
    elif domain == "gradient-x":
        fields = [image_gradient(image, axis=1, sigma=config.sigma)]
    elif domain == "gradient-y":
        fields = [image_gradient(image, axis=0, sigma=config.sigma)]
    elif domain == "gradient-both":
        fields = [
            image_gradient(image, axis=0, sigma=config.sigma),
            image_gradient(image, axis=1, sigma=config.sigma),
        ]
    elif domain == "gradient-magnitude":
        _warn_magnitude_forfeits_invariance()
        gy = image_gradient(image, axis=0, sigma=config.sigma)
        gx = image_gradient(image, axis=1, sigma=config.sigma)
        fields = [np.hypot(gy, gx)]
    else:  # pragma: no cover - GradientConfig validates
        raise ValueError(f"unknown domain {domain!r}")

    window = taper_window(image.shape, config.taper)
    # Mean FIRST (removes the ramp's constant exactly), window SECOND.
    return [(f - f.mean()) * window for f in fields]


# -- correlation --------------------------------------------------------------------


def _cross_power(a: np.ndarray, b: np.ndarray, normalization: str | None) -> np.ndarray:
    spectrum = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    if normalization == "phase":
        magnitude = np.abs(spectrum)
        floor = 1e-12 * max(float(magnitude.max()), 1e-30)
        spectrum = spectrum / np.maximum(magnitude, floor)
    # The DC bin is a pedestal under the whole correlation surface, not a peak. It is
    # already ~0 because both fields were mean-subtracted, but the taper reintroduces a
    # little; zeroing it costs nothing and removes any dependence on the window.
    spectrum[0, 0] = 0.0
    return spectrum


def _coarse_peak(
    cc: np.ndarray, max_shift: float | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Integer peak as ``((index_y, index_x), (lag_y, lag_x))`` with signed lags."""
    ny, nx = cc.shape
    if max_shift is not None:
        limit = int(np.ceil(abs(max_shift)))
        ly = np.abs(((np.arange(ny) + ny // 2) % ny) - ny // 2)
        lx = np.abs(((np.arange(nx) + nx // 2) % nx) - nx // 2)
        allowed = (ly[:, None] <= limit) & (lx[None, :] <= limit)
        cc = np.where(allowed, cc, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)
    return (int(iy), int(ix)), (
        int(iy) - ny if iy > ny // 2 else int(iy),
        int(ix) - nx if ix > nx // 2 else int(ix),
    )


def _refine_peak(
    spectrum: np.ndarray,
    lag: tuple[int, int],
    upsample: int,
    half_width: float = 1.0,
) -> tuple[float, float, float]:
    """Upsampled-DFT sub-pixel refinement (Guizar-Sicairos et al. 2008).

    Evaluates the exact band-limited interpolant of the correlation surface on a grid
    of fractional lags within ``+/- half_width`` px of the integer peak, rather than
    fitting a parabola -- so it carries no peak-locking bias toward integer shifts.
    Cost is ``O(G * ny * nx)`` with ``G = 2 * upsample + 1`` grid points per axis, the
    same order as skimage's implementation, and negligible beside a reconstruction.
    """
    ny, nx = spectrum.shape
    if upsample <= 1:
        cc = np.fft.ifft2(spectrum).real
        return float(lag[0]), float(lag[1]), float(cc[lag[0] % ny, lag[1] % nx])

    step = 1.0 / float(upsample)
    grid_y = lag[0] + np.arange(-half_width, half_width + 0.5 * step, step)
    grid_x = lag[1] + np.arange(-half_width, half_width + 0.5 * step, step)
    kernel_y = np.exp(2j * np.pi * np.outer(grid_y, np.fft.fftfreq(ny)))
    kernel_x = np.exp(2j * np.pi * np.outer(grid_x, np.fft.fftfreq(nx)))
    fine = (kernel_y @ spectrum @ kernel_x.T).real / (ny * nx)

    jy, jx = np.unravel_index(int(np.argmax(fine)), fine.shape)
    return float(grid_y[jy]), float(grid_x[jx]), float(fine[jy, jx])


def register_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    config: GradientConfig | None = None,
) -> RegistrationResult:
    """Sub-pixel translation registering ``moving`` onto ``reference``.

    Returns ``(dy, dx)`` with ``moving(v) = reference(v + d)`` -- the same convention as
    ``skimage.registration.phase_cross_correlation(reference, moving)``. In the
    alignment loop, call it as ``register_translation(measured, simulated)`` and add the
    result to the cumulative shifts.

    With the default ``domain="gradient-x"`` the estimate is invariant to a constant
    phase offset and to a linear phase ramp on either input, to machine precision.
    """
    config = GradientConfig() if config is None else config

    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, moving {moving.shape}")

    fields_a = prepare_field(reference, config)
    fields_b = prepare_field(moving, config)

    # "gradient-both" sums the two correlation surfaces rather than concatenating the
    # fields: correlation is linear in the cross-power spectrum, so summing there is
    # the same thing and costs one inverse FFT instead of two.
    spectrum = sum(
        _cross_power(a, b, config.normalization) for a, b in zip(fields_a, fields_b)
    )
    cc = np.fft.ifft2(spectrum).real

    index, lag = _coarse_peak(cc, config.max_shift)
    dy, dx, peak = _refine_peak(spectrum, lag, config.upsample)

    rms = float(np.sqrt(np.mean(cc**2)))
    quality = float(cc[index] / rms) if rms > 0.0 else float("inf")
    return RegistrationResult(dy=dy, dx=dx, peak=peak, quality=quality, coarse=lag)


def register_stack(
    measured: np.ndarray,
    simulated: np.ndarray,
    config: GradientConfig | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[RegistrationResult]]:
    """Register a whole projection stack against its reprojection, angle by angle.

    Returns ``(dsy, dsx, results)``: the per-angle shift updates in the engine's
    convention (add them to the cumulative ``sy``/``sx``), plus the per-angle
    :class:`RegistrationResult` for diagnostics -- ``quality`` in particular is what
    tells you whether a given angle actually locked on or matched noise.
    """
    config = GradientConfig() if config is None else config
    measured = np.asarray(measured)
    simulated = np.asarray(simulated)
    if measured.shape != simulated.shape:
        raise ValueError(
            f"stack shape mismatch: measured {measured.shape}, simulated {simulated.shape}"
        )
    if measured.ndim != 3:
        raise ValueError(f"expected (n_angles, n_rows, n_cols); got {measured.shape}")

    n = measured.shape[0]
    dsy = np.zeros(n, dtype=np.float64)
    dsx = np.zeros(n, dtype=np.float64)
    results: list[RegistrationResult] = []
    for i in range(n):
        if progress is not None:
            progress(i / n, f"registering projection {i + 1}/{n}")
        result = register_translation(measured[i], simulated[i], config)
        dsy[i], dsx[i] = result.dy, result.dx
        results.append(result)
    return dsy, dsx, results

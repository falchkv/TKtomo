"""The decision gate: is a non-rigid model *warranted* by this data, or not?

Non-rigid alignment is the last stage of the roadmap and the only one with enough
freedom to be dangerous. A deformation vector field has of order a thousand times more
parameters than a per-projection rigid shift, so it can absorb an unfixed phase ramp, a
wrong rotation centre, an unconverged rigid alignment, or plain detector noise, and
return a *sharp, plausible, wrong* volume. Nothing downstream can detect that. The
roadmap's answer is that the decision must be evidential, never speculative, and this
module is that decision written as code so it can be recorded in a report rather than
asserted in a meeting.

THE THREE OUTCOMES THAT ARE NOT "RUN IT"
----------------------------------------
Most of this file exists to reach one of the outcomes that says *no*, because those are
the ones a keen user talks themselves out of:

``FIX_UPSTREAM``
    The leftover residual is explained by something the roadmap puts *earlier*: a
    per-projection phase ramp or offset, a global gain mismatch, or a rotation-centre
    error. Running non-rigid here paves over an error that a single upstream fix would
    remove exactly, and hides it forever.

``MORE_RIGID_ITERATIONS``
    The residual has not plateaued, or it has but the leftover is still a *per-angle
    rigid shift that is smooth in acquisition time* -- i.e. drift the rigid stage has
    not finished removing. More rigid iterations are free; a deformation field is not.

``ACCEPT_RIGID``
    The residual has plateaued, nothing upstream explains it, and what is left is
    spread over the frame and uncorrelated between angles. That is jitter and noise.
    It is the confusable alternative to deformation and the one that costs you a paper:
    a deformation field fitted to angle-random residual will fit it, will improve the
    fitted residual, and will be pure fabrication. The honest move is to report the
    rigid result and its residual floor.

Only when all three are excluded does the gate return ``RUN_NONRIGID``.

WHAT IS MEASURED, AND WHY EACH MEASUREMENT IS NOT THE OBVIOUS ONE
-----------------------------------------------------------------
1. **The plateau is measured, not asserted** (:func:`measure_plateau`). "The residual
   stopped falling" is a statement about a trend against the scatter of that trend, so
   both are estimated: a log-linear slope over the trailing window *and* the
   iteration-to-iteration scatter about it. A curve still falling at 0.3% per iteration
   with 3% scatter has not plateaued in any useful sense and neither has one whose last
   two points happened to tie. The projected further gain from continuing is reported
   in the same units the user cares about, so "run five more rigid iterations" is a
   number and not a hunch.

2. **The statistic is computed on the residual no rigid model can reach**
   (:func:`rigid_reduced_residual`). The raw reprojection residual of a real dataset is
   dominated by things rigid alignment handles perfectly well: on the benchmark phantom
   89% of its energy is per-angle offset and gain -- a consequence of reconstructing
   inconsistent data, present with no normalisation error anywhere in it. Measure
   concentration on that and you measure the offset. So the best per-angle offset, gain
   and rigid shift are fitted out first, and what remains is by construction *what no
   rigid alignment can fix*, which is exactly the quantity the roadmap's criterion is
   about. When almost nothing survives that reduction the verdict is ``ACCEPT_RIGID`` on
   those grounds alone, before localisation is consulted at all.

3. **Localisation is measured against a null, not against a threshold**
   (:func:`measure_localisation`). The natural statistic -- the fraction of residual
   energy in the hottest 10% of blocks -- has no fixed meaning: for a uniform spread it
   is 0.10 only if the residual is Gaussian and the blocks are large. Heavy-tailed
   residual (a few hot pixels, cosmic rays, a dead column) concentrates *by
   construction* and sails past any fixed threshold while being the opposite of
   deformation. So the null is realised explicitly: the residual pixels are permuted
   within each frame, which destroys spatial organisation while preserving the
   amplitude distribution exactly, and the observed concentration is reported as a
   z-score and an empirical p-value against that null. A test on Laplace-distributed
   noise, which fools a fixed threshold, pins this.

4. **Localisation is measured inside the object's own footprint.** A frame that is
   mostly vacuum makes every residual look localised: all the energy is on the object,
   which is a tenth of the frame, and the statistic reports triumph. The support mask
   is derived from the *measured* projections (never from the residual, which would be
   circular) and the concentration is computed among support blocks only, so
   "localised" means localised *within the sample* -- the thing Odstrcil et al. actually
   saw.

5. **The gauge is removed before any leftover shift is called an error.** A leftover
   per-angle horizontal shift of the form ``X cos t + Y sin t`` is a translation of the
   reconstructed object: unobservable, harmless, and *not* a misalignment. A leftover
   *constant* horizontal shift is exactly the rotation-centre error. Conflating them is
   how a perfectly aligned dataset gets sent back for more alignment (this repo measured
   a correct alignment scoring 0.2 px under mean-only gauge removal; see
   ``benchmarks/metrics.py``). :func:`measure_upstream` splits the two and reports them
   separately.

WHAT THIS GATE CANNOT DO
------------------------
* It cannot see a **global** deformation. Uniform swelling or bulk thermal expansion
  produces a residual spread over the whole frame, which this scores as "not localised"
  and vetoes. That is the roadmap's criterion implemented faithfully, and it is a real
  limitation, not a bug in the code: say so, and use :attr:`GateVerdict.notes`, which
  records when the veto came from the localisation test alone.
* It cannot prove there is no deformation. ``ACCEPT_RIGID`` says "the evidence does not
  support a deformation model", which is a statement about the evidence.
* It is a gate, not a validator. After running non-rigid, the deformation still has to
  be validated out of sample -- the held-out residual of
  :class:`~tktomo.ptycho_align.core.nonrigid.NonRigidAligner` -- and the volume by
  split-data FSC paired with a residual map, because FSC is blind to common-mode
  geometric error.

RELATION TO THE REST OF THE TREE
--------------------------------
* :mod:`tktomo.ptycho_align.core.nonrigid` carries a lighter-weight version of the same
  idea (``residual_localisation`` / ``nonrigid_is_warranted``) which the aligner uses as
  an internal precondition. This module is the *reportable* version: it adds the null
  model, the support mask, the upstream decomposition and the confusable alternatives,
  and it never raises -- it returns a verdict you can serialise.
  :func:`GateVerdict.as_rigid_evidence` converts back, so the two never disagree about
  a dataset.
* :mod:`tktomo.diagnostics`, when present, owns the twelve-mode artifact catalogue.
  :meth:`GateVerdict.to_finding` and :meth:`GateVerdict.to_probe_result` emit that
  package's types so a gate verdict can be dropped into a diagnostics report; both
  return ``None`` when the package is not importable, and every other part of this
  module works unchanged without it.

numpy only (``scipy`` is not needed at all here). Nothing in this file reads a file,
knows a path, or requires a reconstruction backend.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Alternative",
    "GateConfig",
    "GateVerdict",
    "LocalisationStat",
    "PlateauReport",
    "TemporalReport",
    "Recommendation",
    "UpstreamReport",
    "evaluate_gate",
    "format_gate",
    "gate_from_engine",
    "measure_localisation",
    "measure_plateau",
    "measure_temporal_change",
    "measure_upstream",
    "rigid_reduced_residual",
    "angle_subset",
]


# ---------------------------------------------------------------------------------
# Optional integration with tktomo.diagnostics
# ---------------------------------------------------------------------------------

try:  # pragma: no cover - exercised by whichever branch the environment provides
    from tktomo.diagnostics.artifacts import (
        FailureMode as _FailureMode,
        Finding as _Finding,
        ProbeResult as _ProbeResult,
        ProbeStatus as _ProbeStatus,
        TriageStage as _TriageStage,
    )

    DIAGNOSTICS_AVAILABLE = True
except ImportError:  # pragma: no cover - the fallback path
    _FailureMode = _Finding = _ProbeResult = _ProbeStatus = _TriageStage = None  # type: ignore[assignment]
    DIAGNOSTICS_AVAILABLE = False


class Recommendation(str, Enum):
    """What to do next. ``str`` mixin so ``json.dumps`` needs no encoder.

    Ordered by the roadmap's stages: the earlier the outcome, the earlier the stage it
    sends you back to.
    """

    FIX_UPSTREAM = "fix_upstream"
    MORE_RIGID_ITERATIONS = "more_rigid_iterations"
    ACCEPT_RIGID = "accept_rigid"
    RUN_NONRIGID = "run_nonrigid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class GateConfig:
    """Every threshold the gate uses, in one place, each with the reason for its value.

    The defaults are deliberately conservative in one direction: they make ``no`` easier
    to reach than ``yes``. A false ``no`` costs a rerun; a false ``yes`` costs a
    fabricated result that nothing downstream can catch.
    """

    # -- plateau ---------------------------------------------------------------------
    #: Iterations at the tail of the rigid history used to judge the trend. Four is the
    #: smallest window in which a slope can be told from a two-point coincidence.
    plateau_window: int = 4
    #: Relative improvement over that window below which the residual counts as flat.
    plateau_tolerance: float = 0.02
    #: The trend must also be small compared with its own scatter: a curve falling by
    #: less than this many standard deviations over the window is noise, not a trend.
    plateau_snr: float = 1.0
    #: ...and a clean trend only overrides the tolerance if continuing is worth it. Another
    #: whole window of rigid iterations must buy at least this much residual, or a decay
    #: that is real but negligible (0.02%/iteration, perfectly monotone -- what a
    #: deterministic loop looks like once it has converged) would be read as "still
    #: converging" forever and the gate would never open.
    min_projected_gain: float = 0.005
    #: Rigid shift update (px RMS) below which the rigid loop counts as converged.
    shift_tolerance_px: float = 0.1

    # -- localisation ----------------------------------------------------------------
    #: Side of the square blocks the residual energy is pooled into, in pixels; ``0``
    #: chooses ``clip(min(n_v, n_u) // 16, 4, 32)``, which gives of order 16 blocks across
    #: the short axis. Should be of order the correlation length of the deformation you
    #: are looking for: too small and every block holds one noisy pixel, too large and a
    #: localised hot spot is averaged into the background -- and with only a handful of
    #: blocks the "hottest 10%" is one block and the statistic means nothing, which is
    #: what the automatic choice exists to prevent.
    block: int = 0
    #: Fraction of (support) blocks counted as "the hottest".
    top_fraction: float = 0.1
    #: How many permutation realisations build the null. 32 gives the null mean to about
    #: 1/6 of its own standard deviation, which is plenty for a z of 4.
    n_null: int = 32
    #: Concentration must beat the null by this many null standard deviations.
    localisation_z: float = 4.0
    #: ...and by this ratio, so a huge z on a trivial excess does not qualify. A null of
    #: 0.12 and an observation of 0.13 is significant and meaningless.
    min_concentration_ratio: float = 1.4
    #: Angle-to-angle consistency of the block energy maps, as a z against its own null.
    consistency_z: float = 4.0
    #: ...and an absolute floor, for the same reason as above.
    min_angle_consistency: float = 0.1
    #: Blocks whose measured signal is below this fraction of the frame's strongest
    #: block are vacuum and are excluded from the concentration.
    support_fraction: float = 0.05
    #: Acquisition-time blocks the residual is grouped into to ask whether it EVOLVES.
    #: Four is enough to see a drift and few enough that each block has angles in it.
    n_time_blocks: int = 4
    #: How many null standard deviations the time-blocked residual maps must be less alike
    #: than randomly grouped ones before the residual counts as time-varying.
    temporal_z: float = 3.0
    #: Angles subsampled for the localisation and upstream measurements. The statistics
    #: converge long before a 900-projection scan is exhausted and the permutation null
    #: is O(n_null * n_angles * n_pixels).
    max_angles: int = 64

    # -- upstream --------------------------------------------------------------------
    #: Fraction of residual energy uniquely explained by a per-projection LINEAR ramp,
    #: above which the ramp-removal stage is judged unfinished. The per-angle offset and
    #: gain are nuisances and have no threshold -- see :class:`UpstreamReport` for why.
    ramp_fraction_threshold: float = 0.10
    #: Constant leftover horizontal shift (px) above which the rotation centre is wrong.
    #: A tenth of a pixel of centre error is visible as a ring; 0.25 px is generous.
    center_tolerance_px: float = 0.25
    #: Leftover per-angle shift (px RMS, gauge removed) above which the rigid stage has
    #: not converged. The roadmap's accuracy target is a third of a voxel.
    shift_residual_tolerance_px: float = 0.33
    #: Fraction of residual energy explained by per-angle shifts above which the
    #: leftover is "shift-like" at all.
    shift_fraction_threshold: float = 0.10
    #: Fraction of the residual that the whole rigid + nuisance basis CANNOT explain,
    #: below which there is simply nothing left for a deformation model. A pure per-angle
    #: gain error leaves 0% and would otherwise reach the localisation test with a residual
    #: made of floating-point dust, which is beautifully concentrated and means nothing.
    min_residual_beyond_rigid: float = 0.05
    #: Lag-1 autocorrelation (in ACQUISITION order) of the leftover shifts above which
    #: they are a smooth drift the rigid stage should have caught; below it they are
    #: jitter, which no rigid model can remove and no non-rigid model should try to.
    drift_lag1_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GateConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown GateConfig field(s): {', '.join(unknown)}")
        return cls(**dict(raw))


# ---------------------------------------------------------------------------------
# (a) the plateau
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateauReport:
    """Has the rigid residual stopped falling, and how confidently do we know that?

    ``tail_improvement`` is the plain relative improvement over the trailing window --
    the number everyone quotes. ``slope_per_iter`` is the fitted log-linear rate, and
    ``scatter`` the RMS of the window about that fit; ``snr`` is their ratio and is what
    stops a noisy-but-flat history being read as a trend, or vice versa.
    ``projected_gain`` is what another whole window of rigid iterations would buy at the
    current rate, which is the number that actually answers "should I run more?".
    """

    plateaued: bool
    n_iterations: int
    window: int
    value: float
    tail_improvement: float
    slope_per_iter: float
    scatter: float
    snr: float
    projected_gain: float
    last_shift_px: float
    shift_converged: bool
    at_noise_floor: bool
    floor_ratio: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}


def measure_plateau(
    residuals: Sequence[float],
    *,
    shift_rms: Sequence[float] | None = None,
    noise_floor: float | None = None,
    config: GateConfig | None = None,
) -> PlateauReport:
    """Decide whether a rigid residual history has flattened, and by how much it has not.

    ``residuals`` is ``[IterationResult.residual, ...]`` from the rigid run, in order;
    ``shift_rms`` the matching ``.error`` (RMS shift update, px). ``noise_floor``, when
    known -- from a repeat scan, a dark/flat pair, or the reprojection residual of a
    noiseless simulation of the same geometry -- turns "it plateaued" into the far more
    useful "it plateaued *at the noise floor*, so there is nothing left to explain".

    The plateau test is deliberately two-sided. A history is on a plateau when the
    trailing window has improved by less than ``plateau_tolerance`` **and** the fitted
    downward slope is smaller than ``plateau_snr`` times the scatter about it. Requiring
    both means a run that is genuinely still improving at 0.5% per iteration with a
    clean, monotone curve is *not* called a plateau even though 0.5% is under the
    tolerance -- and that is the case where more rigid iterations are the right answer.
    """
    cfg = config or GateConfig()
    values = np.asarray([float(v) for v in residuals], dtype=np.float64)
    values = values[np.isfinite(values)]
    window = int(cfg.plateau_window)

    last_shift = math.nan
    if shift_rms is not None:
        shifts = np.asarray([float(v) for v in shift_rms], dtype=np.float64)
        shifts = shifts[np.isfinite(shifts)]
        if shifts.size:
            last_shift = float(shifts[-1])
    shift_converged = bool(np.isfinite(last_shift) and last_shift <= cfg.shift_tolerance_px)

    if values.size < window:
        return PlateauReport(
            plateaued=False, n_iterations=int(values.size), window=window,
            value=float(values[-1]) if values.size else math.nan,
            tail_improvement=math.nan, slope_per_iter=math.nan, scatter=math.nan,
            snr=math.nan, projected_gain=math.nan, last_shift_px=last_shift,
            shift_converged=shift_converged, at_noise_floor=False, floor_ratio=math.nan,
            reason=(
                f"only {values.size} rigid iteration(s) recorded; at least {window} are "
                "needed before a plateau can be told from a still-falling residual"
            ),
        )

    tail = values[-window:]
    tail_improvement = float((tail[0] - tail[-1]) / abs(tail[0])) if tail[0] else math.nan

    # Log-linear fit: the residual of a converging iteration decays geometrically, so a
    # rate per iteration is the natural parameter and the fit is linear in log space.
    # Guard the log: a residual that has genuinely reached zero is not a plateau problem.
    positive = tail > 0
    if positive.all():
        x = np.arange(window, dtype=np.float64)
        logs = np.log(tail)
        slope, intercept = np.polyfit(x, logs, 1)
        fitted = slope * x + intercept
        scatter = float(np.sqrt(np.mean((logs - fitted) ** 2)))
        slope_per_iter = float(-slope)  # positive == still improving
        # A perfectly clean decay has zero scatter, which makes the trend infinitely
        # well determined -- the opposite of "the trend is noise". `math.isfinite` must
        # NOT gate this: doing so read a flawless geometric convergence as a plateau.
        snr = slope_per_iter / scatter if scatter > 0 else math.copysign(math.inf, slope_per_iter)
        projected_gain = float(1.0 - math.exp(-slope_per_iter * window))
    else:
        slope_per_iter = scatter = snr = projected_gain = math.nan

    flat_enough = bool(np.isfinite(tail_improvement) and tail_improvement < cfg.plateau_tolerance)
    falling = bool(snr > cfg.plateau_snr)  # NaN compares False, which is the safe way here
    rising = bool(snr < -cfg.plateau_snr)
    worth_continuing = bool(falling and projected_gain >= cfg.min_projected_gain)
    plateaued = bool(flat_enough and not rising and not worth_continuing)

    floor_ratio = math.nan
    at_floor = False
    if noise_floor is not None and noise_floor > 0:
        floor_ratio = float(tail[-1] / noise_floor)
        at_floor = floor_ratio <= 1.1

    if plateaued:
        reason = (
            f"the rigid residual is flat: {tail_improvement:.2%} over the last {window} "
            f"iterations (tolerance {cfg.plateau_tolerance:.0%}), and the fitted decay "
            f"{slope_per_iter:+.2%}/iter is within {snr:.1f} sigma of its own scatter"
        )
    elif rising:
        reason = (
            f"the rigid residual is RISING at {-slope_per_iter:.2%}/iter over the last "
            f"{window} iterations. That is a diverging alignment, not a plateau: check the "
            "rotation centre and the shift-update guard before anything else"
        )
    elif not flat_enough:
        reason = (
            f"the rigid residual is still falling: {tail_improvement:.2%} over the last "
            f"{window} iterations (tolerance {cfg.plateau_tolerance:.0%}); another "
            f"{window} would buy about {projected_gain:.1%} more"
        )
    else:
        reason = (
            f"the rigid residual improved only {tail_improvement:.2%} over the last "
            f"{window} iterations, but the trend is clean ({snr:.1f} sigma above its own "
            f"scatter) and still downward at {slope_per_iter:.2%}/iter -- another {window} "
            f"would buy {projected_gain:.2%}, above the {cfg.min_projected_gain:.1%} that "
            "makes continuing worth it. This is slow convergence, not a plateau"
        )

    return PlateauReport(
        plateaued=plateaued, n_iterations=int(values.size), window=window,
        value=float(tail[-1]), tail_improvement=tail_improvement,
        slope_per_iter=slope_per_iter, scatter=scatter, snr=snr,
        projected_gain=projected_gain, last_shift_px=last_shift,
        shift_converged=shift_converged, at_noise_floor=at_floor, floor_ratio=floor_ratio,
        reason=reason,
    )


# ---------------------------------------------------------------------------------
# (b) localisation, against a null of uniform spread
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalisationStat:
    """Is the leftover residual concentrated in sub-regions, or spread over the frame?

    ``concentration`` is the share of residual energy carried by the hottest
    ``top_fraction`` of blocks *within the object's support*. Its value alone means
    nothing (see the module docstring), so it is always accompanied by
    ``null_mean``/``null_std``, obtained by permuting the residual pixels within each
    frame, and by the resulting ``z`` and ``p_value``.

    ``angle_consistency`` is the mean pairwise correlation between per-angle block-energy
    maps, with its own null from independently permuting the blocks of each frame.
    Deformation is a property of the sample, so the same places stay hot from angle to
    angle; jitter and noise are angle-random. Both numbers are needed: a heavy-tailed
    noise field is concentrated *and* inconsistent, and a hot dead column is consistent
    *and* uninteresting -- only the conjunction points at deformation.

    ``support_from`` records where the mask came from, because a concentration measured
    over a frame full of vacuum is meaningless and the reader must be able to tell.
    """

    concentration: float
    null_mean: float
    null_std: float
    z: float
    p_value: float
    gini: float
    angle_consistency: float
    consistency_null_mean: float
    consistency_null_std: float
    consistency_z: float
    block: int
    n_blocks: int
    n_support_blocks: int
    n_angles_used: int
    support_from: str
    is_localised: bool
    is_angle_consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}


def angle_subset(n_angles: int, max_angles: int) -> np.ndarray:
    """Evenly spaced angle indices, at most ``max_angles`` of them.

    The localisation and upstream statistics saturate long before a 900-projection scan is
    exhausted, and both are O(n_angles x n_pixels) with a 32-fold permutation loop on top.
    Evenly spaced rather than random so the subset still spans the full angular range --
    a random subset of a scan with a missing wedge can miss the wedge.
    """
    if n_angles <= max_angles:
        return np.arange(n_angles)
    return np.unique(np.linspace(0, n_angles - 1, max_angles).astype(int))


def _block_sums(frames: np.ndarray, block: int) -> np.ndarray:
    """Pool ``(n, n_v, n_u)`` into ``(n, bv, bu)`` block sums, trimming the remainder."""
    n, n_v, n_u = frames.shape
    bv, bu = n_v // block, n_u // block
    trimmed = frames[:, : bv * block, : bu * block]
    return trimmed.reshape(n, bv, block, bu, block).sum(axis=(2, 4))


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector: 0 = perfectly even, 1 = all in one."""
    x = np.sort(np.asarray(values, dtype=np.float64).ravel())
    total = x.sum()
    if total <= 0 or x.size < 2:
        return math.nan
    index = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * (index * x).sum()) / (x.size * total) - (x.size + 1.0) / x.size)


def _mean_pairwise_correlation(maps: np.ndarray) -> float:
    """Mean off-diagonal correlation between rows of ``(n_angles, n_blocks)``."""
    if maps.shape[0] < 2:
        return math.nan
    centred = maps - maps.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    unit = centred / norms[:, None]
    gram = unit @ unit.T
    off = gram[~np.eye(gram.shape[0], dtype=bool)]
    return float(off.mean()) if off.size else math.nan


def measure_localisation(
    residual: np.ndarray,
    *,
    measured: np.ndarray | None = None,
    support: np.ndarray | None = None,
    config: GateConfig | None = None,
    seed: int = 0,
) -> LocalisationStat:
    """Quantify localisation of a residual stack ``(n_angles, n_v, n_u)`` against a null.

    ``measured`` is the measured projection stack, used *only* to find the object's
    footprint -- deriving the support from the residual itself would be circular, since
    the residual is exactly what we are asking about. ``support`` overrides it with an
    explicit 2D boolean frame mask when you know the geometry.

    Raises ``ValueError`` when the frame cannot be cut into at least 2x2 blocks (a
    concentration over three blocks is not a measurement) or when the residual has no
    energy at all. It never returns a silently degraded answer.
    """
    cfg = config or GateConfig()
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 3:
        raise ValueError(f"residual must be (n_angles, n_v, n_u), got {residual.shape}")
    n_angles, n_v, n_u = residual.shape

    block = int(cfg.block) or int(np.clip(min(n_v, n_u) // 16, 4, 32))
    block = max(1, min(block, n_v, n_u))
    bv, bu = n_v // block, n_u // block
    if bv < 2 or bu < 2:
        raise ValueError(
            f"block={block} px leaves only {bv}x{bu} blocks in a {n_v}x{n_u} frame; "
            "localisation cannot be measured on fewer than 2x2. Reduce GateConfig.block."
        )

    # Subsample angles: the statistics saturate quickly and the permutation null costs
    # n_null passes over whatever is kept. A caller who has already subsampled (as
    # evaluate_gate does, so the expensive rigid reduction runs on the subset only) passes
    # a stack no longer than max_angles and this is a no-op.
    pick = angle_subset(n_angles, cfg.max_angles)
    residual = residual[pick]
    measured_sub = None if measured is None else np.asarray(measured)[pick]
    n_used = residual.shape[0]

    energy = _block_sums(np.square(residual), block)  # (n_used, bv, bu)
    flat = energy.reshape(n_used, -1)
    n_blocks = flat.shape[1]

    # -- the support mask, in block coordinates
    if support is not None:
        mask2d = np.asarray(support, dtype=bool)
        if mask2d.shape != (n_v, n_u):
            raise ValueError(f"support {mask2d.shape} does not match frames {(n_v, n_u)}")
        cover = _block_sums(mask2d[None].astype(np.float64), block)[0]
        keep = (cover > 0.5 * block * block).ravel()
        support_from = "caller-supplied mask"
    elif measured_sub is not None:
        frames = np.asarray(measured_sub, dtype=np.float64)
        if frames.shape[1:] != (n_v, n_u):
            raise ValueError(
                f"measured frames {frames.shape[1:]} do not match residual {(n_v, n_u)}"
            )
        # Signal relative to each frame's own background, so a constant phase offset --
        # which is a nuisance, not the object -- does not define the support.
        background = np.median(frames.reshape(frames.shape[0], -1), axis=1)
        strength = _block_sums(np.abs(frames - background[:, None, None]), block).mean(axis=0)
        peak = float(strength.max())
        keep = (
            (strength >= cfg.support_fraction * peak).ravel()
            if peak > 0
            else np.ones(n_blocks, dtype=bool)
        )
        support_from = f"measured stack, blocks above {cfg.support_fraction:.0%} of peak signal"
    else:
        keep = np.ones(n_blocks, dtype=bool)
        support_from = (
            "ALL blocks -- no measured stack or mask was supplied, so vacuum counts as "
            "sample and the concentration is inflated by however much of the frame is air"
        )

    if keep.sum() < 4:
        raise ValueError(
            f"the support mask keeps only {int(keep.sum())} of {n_blocks} blocks; "
            "localisation among fewer than 4 blocks is not a measurement. Lower "
            "GateConfig.block or GateConfig.support_fraction."
        )

    def concentration_of(block_energy: np.ndarray) -> float:
        """Mean over angles of the energy share held by the hottest blocks."""
        kept = block_energy[:, keep]
        totals = kept.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        share = kept / totals
        top = max(1, int(round(cfg.top_fraction * share.shape[1])))
        return float(np.sort(share, axis=1)[:, -top:].sum(axis=1).mean())

    observed = concentration_of(flat)
    if not np.isfinite(observed):
        raise ValueError("the residual has no finite energy; nothing to localise")

    # -- the null: permute pixels within each frame's support region. Spatial structure
    #    is destroyed; the amplitude distribution, including its tails, is preserved
    #    exactly. That is the difference between "is this concentrated?" (always yes for
    #    heavy tails) and "is this concentrated MORE THAN ITS OWN PIXEL VALUES FORCE?".
    rng = np.random.default_rng(seed)
    keep2d = keep.reshape(bv, bu)
    pixel_mask = np.repeat(np.repeat(keep2d, block, axis=0), block, axis=1)
    padded = np.zeros((n_v, n_u), dtype=bool)
    padded[: bv * block, : bu * block] = pixel_mask
    values = np.square(residual[:, padded])  # (n_used, n_support_px)

    nulls = np.empty(cfg.n_null, dtype=np.float64)
    scratch = np.zeros((n_used, n_v, n_u), dtype=np.float64)
    for k in range(cfg.n_null):
        scratch[:, padded] = rng.permuted(values, axis=1)
        nulls[k] = concentration_of(_block_sums(scratch, block).reshape(n_used, -1))

    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1)) if nulls.size > 1 else 0.0
    z = float((observed - null_mean) / null_std) if null_std > 0 else math.inf
    # Empirical one-sided p, with the +1 correction so it is never exactly zero: with
    # 32 draws the smallest honest statement is p <= 1/33.
    p_value = float((1 + int((nulls >= observed).sum())) / (cfg.n_null + 1))

    # -- angle consistency, with its own null from permuting BLOCKS within each frame
    kept_energy = flat[:, keep]
    consistency = _mean_pairwise_correlation(kept_energy)
    cons_nulls = np.empty(cfg.n_null, dtype=np.float64)
    for k in range(cfg.n_null):
        cons_nulls[k] = _mean_pairwise_correlation(rng.permuted(kept_energy, axis=1))
    cons_mean = float(cons_nulls.mean())
    cons_std = float(cons_nulls.std(ddof=1)) if cons_nulls.size > 1 else 0.0
    cons_z = (
        float((consistency - cons_mean) / cons_std)
        if cons_std > 0 and np.isfinite(consistency)
        else math.nan
    )

    is_localised = bool(
        z >= cfg.localisation_z and observed >= cfg.min_concentration_ratio * null_mean
    )
    is_consistent = bool(
        np.isfinite(cons_z)
        and cons_z >= cfg.consistency_z
        and consistency >= cfg.min_angle_consistency
    )

    return LocalisationStat(
        concentration=observed, null_mean=null_mean, null_std=null_std, z=z,
        p_value=p_value, gini=_gini(kept_energy.mean(axis=0)),
        angle_consistency=consistency, consistency_null_mean=cons_mean,
        consistency_null_std=cons_std, consistency_z=cons_z, block=block,
        n_blocks=n_blocks, n_support_blocks=int(keep.sum()), n_angles_used=n_used,
        support_from=support_from, is_localised=is_localised,
        is_angle_consistent=is_consistent,
    )


# ---------------------------------------------------------------------------------
# (b2) does the residual CHANGE during the scan? The criterion that separates a
#      deformation from a reconstruction artefact.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalReport:
    """Does the leftover residual pattern evolve over acquisition time, or sit still?

    This is the criterion that the localisation test cannot supply and that the benchmark
    forced into existence. A reconstruction artefact -- an FBP streak, an edge the
    band-limited projector cannot reproduce, an unmodelled static feature -- is fixed in
    the object frame, so it is *localised* and *angle-consistent* and passes both of the
    roadmap's tests while having nothing whatever to do with the sample moving. Measured on
    the benchmark phantom with **no deformation at all**: concentration z = 108,
    angle-consistency z = 305, and the gate said "go non-rigid" until this test was added.

    Deformation is, by definition, something that happens *during* the scan. So the
    projections are grouped into acquisition-time blocks, each block's mean residual-energy
    map is formed, and ``evolution`` is one minus the mean pairwise correlation between
    those maps. On its own that number means nothing -- with few angles per block the maps
    are noisy and decorrelate anyway -- so the null is the same statistic with the
    acquisition labels **shuffled**, which preserves the group sizes and the noise exactly
    and destroys only the time ordering. ``z`` is how many null standard deviations the
    time-blocked maps are *less* alike than randomly grouped ones.

    Requires ``acquisition_index``. Angle order will not do: for an interlaced scan or a
    repeated series it is a different ordering, and shuffling one to test the other is
    exactly the null hypothesis.
    """

    evolution: float
    null_evolution: float
    null_std: float
    z: float
    n_time_blocks: int
    n_angles_used: int
    evolves: bool

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}


def measure_temporal_change(
    residual: np.ndarray,
    acquisition_index: np.ndarray,
    *,
    measured: np.ndarray | None = None,
    support: np.ndarray | None = None,
    config: GateConfig | None = None,
    seed: int = 0,
) -> TemporalReport:
    """Does the residual-energy map differ between acquisition-time blocks beyond chance?

    ``residual`` should be the rigid-reduced residual (see
    :func:`rigid_reduced_residual`), the same stack :func:`measure_localisation` is given,
    and ``acquisition_index`` must match it angle for angle.
    """
    cfg = config or GateConfig()
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 3:
        raise ValueError(f"residual must be (n_angles, n_v, n_u), got {residual.shape}")
    acquisition = np.asarray(acquisition_index).ravel()
    if acquisition.size != residual.shape[0]:
        raise ValueError(
            f"{acquisition.size} acquisition indices for {residual.shape[0]} residual maps"
        )

    n_angles, n_v, n_u = residual.shape
    pick = angle_subset(n_angles, cfg.max_angles)
    residual = residual[pick]
    acquisition = acquisition[pick]
    n_used = residual.shape[0]

    n_blocks = int(cfg.n_time_blocks)
    if n_blocks < 2 or n_used < 2 * n_blocks:
        raise ValueError(
            f"{n_used} projections cannot be split into {n_blocks} acquisition-time blocks "
            "with at least two each; lower GateConfig.n_time_blocks"
        )

    block = int(cfg.block) or int(np.clip(min(n_v, n_u) // 16, 4, 32))
    block = max(1, min(block, n_v, n_u))
    bv, bu = n_v // block, n_u // block
    if bv < 2 or bu < 2:
        raise ValueError(f"block={block} px leaves only {bv}x{bu} blocks; need at least 2x2")

    # np.square, never ``residual**2``: on CPython 3.14 + NumPy 2.2.6 the
    # temporary-elision optimisation squares a live local IN PLACE, so the later
    # ``values`` line would silently square an already-squared array and the
    # permutation null would be built from 4th-power values (measured: null
    # 0.302 instead of 0.202 on the deformation fixture, flipping the verdict
    # to ACCEPT_RIGID). Same defect class as deformation._horn_schunck_level.
    energy = _block_sums(np.square(residual), block).reshape(n_used, -1)
    if support is not None:
        mask2d = np.asarray(support, dtype=bool)
        cover = _block_sums(mask2d[None].astype(np.float64), block)[0]
        keep = (cover > 0.5 * block * block).ravel()
    elif measured is not None:
        frames = np.asarray(measured, dtype=np.float64)[pick]
        background = np.median(frames.reshape(frames.shape[0], -1), axis=1)
        strength = _block_sums(np.abs(frames - background[:, None, None]), block).mean(axis=0)
        peak = float(strength.max())
        keep = (
            (strength >= cfg.support_fraction * peak).ravel()
            if peak > 0
            else np.ones(energy.shape[1], dtype=bool)
        )
    else:
        keep = np.ones(energy.shape[1], dtype=bool)
    if keep.sum() < 4:
        raise ValueError("the support mask keeps fewer than 4 blocks; nothing to compare")

    shares = energy[:, keep]
    totals = shares.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    shares = shares / totals

    order = np.argsort(acquisition, kind="stable")

    def evolution_of(grouping: list[np.ndarray]) -> float:
        maps = np.stack([shares[g].mean(axis=0) for g in grouping])
        return 1.0 - _mean_pairwise_correlation(maps)

    observed = evolution_of([np.sort(part) for part in np.array_split(order, n_blocks)])

    rng = np.random.default_rng(seed)
    nulls = np.empty(cfg.n_null, dtype=np.float64)
    for k in range(cfg.n_null):
        shuffled = rng.permutation(n_used)
        nulls[k] = evolution_of(np.array_split(shuffled, n_blocks))
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1)) if nulls.size > 1 else 0.0
    z = float((observed - null_mean) / null_std) if null_std > 0 else math.inf

    return TemporalReport(
        evolution=float(observed), null_evolution=null_mean, null_std=null_std, z=z,
        n_time_blocks=n_blocks, n_angles_used=n_used,
        evolves=bool(z >= cfg.temporal_z),
    )


# ---------------------------------------------------------------------------------
# (c) is the residual really an upstream error?
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamReport:
    """How much of the leftover residual an *earlier* roadmap stage would have removed.

    The residual of every angle is regressed onto a small physical basis::

        R_i  ~  c_i + g_i*S_i                <- offset and gain: NUISANCES, see below
             +  e_i*u + f_i*v                <- a per-projection phase ramp  (stage 0)
             +  a_i*dS_i/du + b_i*dS_i/dv    <- a leftover rigid shift       (stages 2-4)

    **The per-angle offset and gain are fitted and removed, never diagnosed.** They are
    the one part of this basis that a *deformation* also produces: a sample that deforms
    makes the reconstruction inconsistent, the reprojection of that inconsistent volume
    mismatches the measurement in overall scale and level, and on the phantom here that
    lands 15% of the residual energy on the gain column with no normalisation error
    anywhere in the data. Diagnosing it would send every genuine deformation off to fix
    an imaginary flat-field. It is also the harmless part: a constant and a global scale
    do not move anything, so neither can be confused with a misalignment. This is the
    same treatment ``tktomo.diagnostics``' own residual statistic gives them, and
    ``nuisance_fraction`` reports how much was absorbed so it is not hidden.

    The *linear* ramp is diagnosed, and it is the one that matters: a ramp across the
    frame is mathematically indistinguishable from a lateral shift, so it does not merely
    add artifacts, it poisons the alignment meant to remove them.

    ``ramp_fraction`` and ``shift_fraction`` are each group's
    **unique** contribution: the energy the full model explains minus the energy it still
    explains with that group deleted. They are not a partition -- what several groups
    could explain equally well belongs to none of them and is reported separately as
    ``shared_fraction``, with ``unexplained_fraction`` for what none of them reaches::

        ramp + shift + nuisance + shared + unexplained = 1

    ``unexplained_fraction`` is the one to look at first: it is the share of the residual
    that the whole rigid-plus-nuisance model cannot reach, and it is the only part a
    deformation could possibly explain. When it is near zero there is nothing here for
    non-rigid alignment to do, whatever the localisation statistic says about the rest.

    The obvious alternative, entering the groups in the roadmap's order and crediting each
    with what it explains over the earlier ones, was tried first and is wrong here for a
    concrete reason: over a compact object a small lateral shift produces a residual that
    projects onto a linear ramp almost as well as onto ``dS/du``. Measured on a phantom,
    a *pure* 0.8 px shift was credited with 16% "ramp" under sequential attribution --
    enough to fire the ramp test and send someone off to fix a phase ramp that was never
    there. The unique contribution of the ramp in that same case is essentially zero,
    because the shift group explains everything the ramp could. When the two really are
    indistinguishable in a given dataset, ``shared_fraction`` is large and says so
    instead of resolving the ambiguity silently in either direction.

    The shift coefficients are converted to pixels (``du = -a``, since a measurement
    shifted by ``du`` reads ``S(u - du) ~ S - du dS/du``) and then split by the gauge,
    which is the part that is usually got wrong:

    * ``center_offset_px`` -- the *constant* part of the horizontal shift. A constant
      column offset in a sinogram is exactly the rotation-axis position, so this is a
      centre error and it is upstream. It comes with ``center_offset_se_px``, the
      standard error of that constant, and the two must be read together: the mean of a
      jittery leftover shift is itself a random number of order
      ``du_rms / sqrt(n_angles)``, so on a short scan a "centre error" of a quarter pixel
      can be nothing but the noise in one's own estimate of it. The gate requires the
      offset to beat both the physical tolerance *and* twice its own standard error.
    * ``gauge_amplitude_px`` -- the ``sin``/``cos`` part. This is a translation of the
      object in the rotation plane: unobservable, harmless, and explicitly NOT counted
      as an error.
    * ``du_residual_rms_px`` / ``dv_residual_rms_px`` -- what is left after removing the
      gauge. This is genuine leftover misalignment, and ``du_lag1``/``dv_lag1`` (in
      acquisition order, when it is known) say whether it is a smooth drift the rigid
      stage should still be chasing or angle-random jitter that no rigid model can ever
      remove.
    """

    ramp_fraction: float
    shift_fraction: float
    nuisance_fraction: float
    shared_fraction: float
    unexplained_fraction: float
    center_offset_px: float
    center_offset_se_px: float
    gauge_amplitude_px: float
    du_residual_rms_px: float
    dv_residual_rms_px: float
    du_lag1: float
    dv_lag1: float
    ramp_ptv_frac: float
    n_angles_used: int
    time_order_known: bool
    dominant: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}


def rigid_reduced_residual(measured: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """``measured - simulated`` after removing, per angle, the best offset, gain and shift.

    **This, and not the raw difference, is what the localisation test must look at.** The
    raw reprojection residual of any real dataset is dominated by things a rigid model
    handles perfectly well: on the benchmark phantom, 89% of its energy is per-angle offset
    and gain -- a consequence of reconstructing inconsistent data, present even with no
    normalisation error anywhere. Measuring concentration on that measures the offset, not
    the deformation, and the answer is meaningless in both directions.

    What is left after the removal is, by construction, *what no rigid alignment can fix*,
    which is exactly the quantity the roadmap's localisation criterion is about. The same
    reduction is what ``tktomo.diagnostics``' own deformation probe performs before it
    measures locality.

    The linear ramp is deliberately **not** removed: a per-projection ramp is a real data
    error with its own place in the roadmap, and hiding it from the localisation map would
    hide it from the reader too. (It is spread over the frame, so it does not make a
    residual look localised; and :func:`evaluate_gate` tests for it *before* localisation
    matters.)
    """
    measured = np.asarray(measured, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    if measured.shape != simulated.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {simulated.shape}")
    out = np.empty_like(measured)
    ones = np.ones(measured.shape[1:]).ravel()
    for i in range(measured.shape[0]):
        sim = simulated[i]
        g_v, g_u = np.gradient(sim)
        design = np.column_stack([ones, sim.ravel(), g_u.ravel(), g_v.ravel()])
        target = (measured[i] - sim).ravel()
        coefficients = np.linalg.pinv(design.T @ design) @ (design.T @ target)
        out[i] = (target - design @ coefficients).reshape(sim.shape)
    return out


def _lag1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return math.nan
    x = x - x.mean()
    denominator = float(np.sum(x * x))
    if denominator <= 0:
        return math.nan
    return float(np.sum(x[:-1] * x[1:]) / denominator)


def measure_upstream(
    measured: np.ndarray,
    simulated: np.ndarray,
    *,
    angles: np.ndarray | None = None,
    acquisition_index: np.ndarray | None = None,
    config: GateConfig | None = None,
) -> UpstreamReport:
    """Regress the residual onto ramp / gain / shift and report what each explains.

    ``angles`` (radians) is needed to separate the rotation-centre error from the
    unobservable in-plane translation; without it both land in ``center_offset_px`` and
    the report says so by returning ``gauge_amplitude_px`` as NaN.
    ``acquisition_index`` orders the leftover shifts in time so drift can be told from
    jitter; without it, angle order is used, which is the same thing only for a single
    monotonic sweep.
    """
    cfg = config or GateConfig()
    measured = np.asarray(measured, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    if measured.shape != simulated.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {simulated.shape}")
    if measured.ndim != 3:
        raise ValueError(f"stacks must be (n_angles, n_v, n_u), got {measured.shape}")

    n_angles, n_v, n_u = measured.shape
    order = np.arange(n_angles)
    if acquisition_index is not None:
        acq = np.asarray(acquisition_index).ravel()
        if acq.size != n_angles:
            raise ValueError(f"{acq.size} acquisition indices for {n_angles} projections")
        order = np.argsort(acq, kind="stable")
        time_known = True
    else:
        time_known = False

    pick = order[angle_subset(n_angles, cfg.max_angles)]
    n_used = int(pick.size)

    v = np.linspace(-1.0, 1.0, n_v)[:, None] * np.ones((1, n_u))
    u = np.ones((n_v, 1)) * np.linspace(-1.0, 1.0, n_u)[None, :]
    ones = np.ones((n_v, n_u))

    # Column layout of the design matrix, and which columns belong to which group.
    #   0:1, 3:S            offset and gain -- fitted, reported, NOT diagnosed
    #   1:u, 2:v            a per-projection linear phase ramp
    #   4:dS/du, 5:dS/dv    a leftover rigid shift
    groups = {"nuisance": (0, 3), "ramp": (1, 2), "shift": (4, 5)}
    n_columns = 6

    du = np.zeros(n_used)
    dv = np.zeros(n_used)
    total_energy = np.zeros(n_used)
    full_energy = np.zeros(n_used)
    unique_energy = {name: np.zeros(n_used) for name in groups}
    ramp_ptv = np.zeros(n_used)
    contrast = np.zeros(n_used)

    def explained(gram: np.ndarray, cross: np.ndarray, columns: Sequence[int]) -> float:
        """Energy a sub-model explains: ``c^T G^-1 c`` on the selected columns.

        Via the Gram matrix rather than a fresh least squares, so all four fits (full
        plus three leave-one-group-out) cost one pass over the pixels instead of four.
        ``pinv`` because a constant simulated frame makes the gain and offset columns
        collinear, and a singular Gram there means "these two explain the same thing",
        not "crash".
        """
        index = np.asarray(columns, dtype=int)
        sub_g = gram[np.ix_(index, index)]
        sub_c = cross[index]
        return float(sub_c @ (np.linalg.pinv(sub_g) @ sub_c))

    for k, i in enumerate(pick):
        sim = simulated[i]
        res = measured[i] - sim
        g_v, g_u = np.gradient(sim)
        design = np.column_stack(
            [ones.ravel(), u.ravel(), v.ravel(), sim.ravel(), g_u.ravel(), g_v.ravel()]
        )
        target = res.ravel()
        gram = design.T @ design
        cross = design.T @ target
        total_energy[k] = float(target @ target)

        all_columns = tuple(range(n_columns))
        full = explained(gram, cross, all_columns)
        full_energy[k] = full
        for name, columns in groups.items():
            without = tuple(c for c in all_columns if c not in columns)
            unique_energy[name][k] = max(0.0, full - explained(gram, cross, without))

        # The physical shift is the FULL-model coefficient: the shift you would fit with
        # the ramp and the gain as nuisances, which is exactly what the joint solve gives.
        coefficients = np.linalg.pinv(gram) @ cross
        du[k] = -coefficients[4]
        dv[k] = -coefficients[5]
        contrast[k] = float(np.percentile(sim, 99) - np.percentile(sim, 1))
        ramp_ptv[k] = float(
            2.0 * (abs(coefficients[1]) + abs(coefficients[2])) + abs(coefficients[0])
        )

    total = float(total_energy.sum())
    if total <= 0:
        raise ValueError("the residual is identically zero; there is nothing to explain")
    ramp_fraction = float(unique_energy["ramp"].sum() / total)
    nuisance_fraction = float(unique_energy["nuisance"].sum() / total)
    shift_fraction = float(unique_energy["shift"].sum() / total)
    explained_fraction = float(full_energy.sum() / total)
    shared = float(
        max(0.0, explained_fraction - ramp_fraction - nuisance_fraction - shift_fraction)
    )
    unexplained = float(max(0.0, 1.0 - explained_fraction))

    # -- the gauge split of the horizontal shift
    if angles is not None:
        theta = np.asarray(angles, dtype=np.float64).ravel()
        if theta.size != n_angles:
            raise ValueError(f"{theta.size} angles for {n_angles} projections")
        theta = theta[pick]
        design = np.column_stack([np.ones(n_used), np.sin(theta), np.cos(theta)])
        gauge_coefficients, *_ = np.linalg.lstsq(design, du, rcond=None)
        center_offset = float(gauge_coefficients[0])
        gauge_amplitude = float(math.hypot(gauge_coefficients[1], gauge_coefficients[2]))
        du_residual = du - design @ gauge_coefficients
    else:
        center_offset = float(du.mean())
        gauge_amplitude = math.nan
        du_residual = du - du.mean()
    dv_residual = dv - dv.mean()

    du_rms = float(np.sqrt(np.mean(np.square(du_residual))))
    dv_rms = float(np.sqrt(np.mean(np.square(dv_residual))))
    center_se = float(du_rms / math.sqrt(max(n_used, 1)))

    scores = {"ramp": ramp_fraction, "nuisance": nuisance_fraction, "shift": shift_fraction}
    dominant = max(scores, key=lambda k: scores[k])
    if scores[dominant] < 0.05:
        dominant = "none"

    mean_contrast = float(np.mean(contrast))
    return UpstreamReport(
        ramp_fraction=ramp_fraction, shift_fraction=shift_fraction,
        nuisance_fraction=nuisance_fraction, shared_fraction=shared,
        unexplained_fraction=unexplained,
        center_offset_px=center_offset, center_offset_se_px=center_se,
        gauge_amplitude_px=gauge_amplitude,
        du_residual_rms_px=du_rms, dv_residual_rms_px=dv_rms,
        du_lag1=_lag1(du_residual), dv_lag1=_lag1(dv_residual),
        ramp_ptv_frac=float(np.mean(ramp_ptv) / mean_contrast) if mean_contrast > 0 else math.nan,
        n_angles_used=n_used, time_order_known=time_known, dominant=dominant,
    )


# ---------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Alternative:
    """One confusable explanation of the residual, and whether it was ruled out.

    Every ``no`` this gate can give corresponds to an alternative that was *not*
    excluded, and a ``yes`` is only worth as much as the list of alternatives that were.
    Keeping them in the verdict rather than in the prose is the point: it survives into
    a JSON report and into a methods section.
    """

    name: str
    excluded: bool
    statistic: float
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "excluded": bool(self.excluded),
            "statistic": _jsonable(self.statistic),
            "threshold": _jsonable(self.threshold),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateVerdict:
    """The answer, the numbers behind it, and what else it could have been.

    JSON-serialisable all the way down (:meth:`to_json`), so a decision to run a method
    with this much freedom leaves a record that can be checked later by someone who was
    not in the room.
    """

    recommendation: Recommendation
    confidence: float
    headline: str
    plateau: PlateauReport | None
    localisation: LocalisationStat | None
    upstream: UpstreamReport | None
    temporal: TemporalReport | None = None
    alternatives: tuple[Alternative, ...] = ()
    notes: tuple[str, ...] = ()
    config: GateConfig = field(default_factory=GateConfig)
    context: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """``True`` only for :attr:`Recommendation.RUN_NONRIGID`. Every other outcome --
        including ``INSUFFICIENT_EVIDENCE`` -- is a ``no``, because the default for a
        model that can fabricate structure has to be the conservative one."""
        return self.recommendation is Recommendation.RUN_NONRIGID

    @property
    def unexcluded(self) -> tuple[Alternative, ...]:
        """The alternatives that were *not* ruled out. Empty is what a ``yes`` needs."""
        return tuple(a for a in self.alternatives if not a.excluded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation.value,
            "confidence": round(float(self.confidence), 4),
            "headline": self.headline,
            "plateau": self.plateau.to_dict() if self.plateau else None,
            "localisation": self.localisation.to_dict() if self.localisation else None,
            "upstream": self.upstream.to_dict() if self.upstream else None,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "notes": list(self.notes),
            "config": self.config.to_dict(),
            "context": {k: _jsonable(v) for k, v in self.context.items()},
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # -- interoperability -------------------------------------------------------------

    def as_rigid_evidence(self, residuals: Sequence[float], shift_rms: Sequence[float]):
        """Convert to :class:`~tktomo.ptycho_align.core.nonrigid.RigidEvidence`.

        So the aligner's internal precondition and this gate cannot disagree about a
        dataset: hand the same evidence to both. The
        :class:`~tktomo.ptycho_align.core.nonrigid.LocalisationReport` is filled from
        this verdict's own localisation measurement, with ``is_localised`` set to the
        conjunction this module requires (localised *and* angle-consistent), which is
        stricter than the lighter test in that module.
        """
        from tktomo.ptycho_align.core.nonrigid import (  # noqa: PLC0415
            LocalisationReport,
            RigidEvidence,
        )

        report = None
        if self.localisation is not None:
            report = LocalisationReport(
                concentration=self.localisation.concentration,
                angle_consistency=self.localisation.angle_consistency,
                block=self.localisation.block,
                is_localised=bool(
                    self.localisation.is_localised and self.localisation.is_angle_consistent
                ),
            )
        return RigidEvidence(
            residuals=np.asarray(residuals, dtype=np.float64),
            shift_rms=np.asarray(shift_rms, dtype=np.float64),
            localisation=report,
        )

    def to_finding(self):
        """A :class:`tktomo.diagnostics.Finding`, or ``None`` when that package is absent.

        Emitted only for ``RUN_NONRIGID``: a finding means "this failure mode is present",
        and the other recommendations are statements about *this* stage's evidence, not
        about the earlier modes -- those have their own probes and their own findings, and
        duplicating them here with a different threshold would be worse than useless.
        """
        if not DIAGNOSTICS_AVAILABLE or self.recommendation is not Recommendation.RUN_NONRIGID:
            return None
        evidence: dict[str, float] = {}
        if self.localisation is not None:
            evidence["concentration"] = self.localisation.concentration
            evidence["localisation_z"] = self.localisation.z
            evidence["angle_consistency"] = self.localisation.angle_consistency
        if self.plateau is not None:
            evidence["plateau_tail_improvement"] = self.plateau.tail_improvement
        return _Finding(  # type: ignore[misc]
            mode=_FailureMode.DEFORMATION,
            confidence=float(self.confidence),
            probe="nonrigid_gate",
            detail=self.headline,
            evidence=evidence,
        )

    def to_probe_result(self):
        """A :class:`tktomo.diagnostics.ProbeResult`, or ``None`` when absent.

        ``FIRED`` for ``RUN_NONRIGID``, ``CLEAR`` for the outcomes that say the rigid
        result stands, and ``NOT_APPLICABLE`` for ``INSUFFICIENT_EVIDENCE`` and for the
        outcomes that send you back to an earlier stage -- because the non-rigid question
        genuinely cannot be answered until that stage is fixed, and reporting it as
        "clear" would claim knowledge the measurement does not have.
        """
        if not DIAGNOSTICS_AVAILABLE:
            return None
        status = {
            Recommendation.RUN_NONRIGID: _ProbeStatus.FIRED,
            Recommendation.ACCEPT_RIGID: _ProbeStatus.CLEAR,
            Recommendation.FIX_UPSTREAM: _ProbeStatus.NOT_APPLICABLE,
            Recommendation.MORE_RIGID_ITERATIONS: _ProbeStatus.NOT_APPLICABLE,
            Recommendation.INSUFFICIENT_EVIDENCE: _ProbeStatus.NOT_APPLICABLE,
        }[self.recommendation]
        metrics: dict[str, float] = {}
        for source in (self.plateau, self.localisation, self.upstream, self.temporal):
            if source is None:
                continue
            for key, value in source.to_dict().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = float(value)
        finding = self.to_finding()
        return _ProbeResult(  # type: ignore[misc]
            probe="nonrigid_gate",
            stage=_TriageStage.NON_RIGID,
            status=status,
            reason="" if status is not _ProbeStatus.NOT_APPLICABLE else self.headline,
            detail=self.headline,
            metrics=metrics,
            findings=() if finding is None else (finding,),
        )


# ---------------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------------


def evaluate_gate(
    *,
    residual_history: Sequence[float] | None = None,
    shift_history: Sequence[float] | None = None,
    measured: np.ndarray | None = None,
    simulated: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    acquisition_index: np.ndarray | None = None,
    support: np.ndarray | None = None,
    noise_floor: float | None = None,
    config: GateConfig | None = None,
    seed: int = 0,
) -> GateVerdict:
    """Should this dataset get a non-rigid model? Returns a verdict, never raises on data.

    Pass what you have. The two inputs that carry the decision are:

    * ``residual_history`` (and ``shift_history``) -- the rigid engine's per-iteration
      ``IterationResult.residual`` and ``.error``. Without it the plateau cannot be
      measured and the verdict is ``INSUFFICIENT_EVIDENCE``, because "the residual has
      stopped falling" is a statement about a *history*, and no single frame contains it.
    * ``measured`` and ``simulated`` -- the rigidly aligned stack and its reprojection
      from the final rigid iteration (``engine.last_aligned`` / ``engine.last_simulated``).
      From these the residual maps, the localisation and the upstream decomposition all
      follow. ``residual`` may be given instead, but then the upstream decomposition and
      the object support cannot be computed and the verdict says so.

    ``noise_floor`` is the relative reprojection residual attributable to counting noise
    alone. Supplying it is what turns "plateaued" into "plateaued at the floor, so there
    is nothing left for any model to explain" -- the cleanest possible ``ACCEPT_RIGID``.

    The order of the tests is the roadmap's order of operations and is not negotiable:
    upstream first, then convergence, then localisation. A localised residual on top of
    an unfixed phase ramp is still a phase ramp.
    """
    cfg = config or GateConfig()
    notes: list[str] = []
    alternatives: list[Alternative] = []

    if residual is None and measured is not None and simulated is not None:
        # The RIGID-REDUCED residual: what no rigid alignment can fix. See
        # rigid_reduced_residual() for why the raw difference is the wrong thing to
        # measure concentration on. Subsample FIRST: the reduction is a least-squares fit
        # per angle and allocates a float64 copy, which on a 907 x 1488 x 1816 stack would
        # be 39 GB and minutes of work for a statistic that saturates at ~64 angles.
        keep = angle_subset(np.asarray(measured).shape[0], cfg.max_angles)
        measured = np.asarray(measured)[keep]
        simulated = np.asarray(simulated)[keep]
        if angles is not None:
            angles = np.asarray(angles).ravel()[keep]
        if acquisition_index is not None:
            acquisition_index = np.asarray(acquisition_index).ravel()[keep]
        residual_maps: np.ndarray | None = rigid_reduced_residual(measured, simulated)
    elif residual is not None:
        residual_maps = np.asarray(residual, dtype=np.float64)
        notes.append(
            "residual maps were supplied directly, so (a) the upstream decomposition "
            "(ramp, leftover shift, rotation centre) could NOT be run, and (b) the "
            "localisation was measured on the RAW residual rather than the "
            "rigid-reduced one, so a per-angle offset, gain or leftover shift is "
            "contributing to the concentration. Pass measured= and simulated= to close "
            "both holes."
        )
    else:
        residual_maps = None

    # -- (a) plateau
    plateau = None
    if residual_history is not None:
        plateau = measure_plateau(
            residual_history, shift_rms=shift_history, noise_floor=noise_floor, config=cfg
        )

    # -- (b) localisation
    localisation = None
    if residual_maps is not None:
        try:
            localisation = measure_localisation(
                residual_maps, measured=measured, support=support, config=cfg, seed=seed
            )
        except ValueError as exc:
            notes.append(f"localisation could not be measured: {exc}")

    # -- (b2) does it change with acquisition time?
    temporal = None
    if residual_maps is not None and acquisition_index is not None:
        try:
            temporal = measure_temporal_change(
                residual_maps, acquisition_index, measured=measured, support=support,
                config=cfg, seed=seed,
            )
        except ValueError as exc:
            notes.append(f"the temporal-change test could not be run: {exc}")

    # -- (c) upstream
    upstream = None
    if measured is not None and simulated is not None:
        try:
            upstream = measure_upstream(
                measured, simulated, angles=angles,
                acquisition_index=acquisition_index, config=cfg,
            )
        except ValueError as exc:
            notes.append(f"the upstream decomposition could not be run: {exc}")

    if localisation is not None and localisation.support_from.startswith("ALL blocks"):
        notes.append(
            "no object support was available, so the concentration counts vacuum blocks "
            "as sample. On a frame that is mostly air this inflates it and can turn a "
            "flat residual into a 'localised' one."
        )
    if upstream is not None and upstream.shared_fraction > 0.25:
        notes.append(
            f"{upstream.shared_fraction:.0%} of the residual energy is explained equally well "
            "by more than one of ramp / gain / shift, so which upstream stage owns it is not "
            "identifiable from this data. Treat every upstream 'excluded' below as provisional."
        )
    if upstream is not None and not upstream.time_order_known:
        notes.append(
            "no acquisition_index was given, so drift and jitter were told apart in ANGLE "
            "order. For an interlaced scan or a repeated series those are different "
            "orderings and this discrimination is then meaningless."
        )

    # ------------------------------------------------------------------ the decision
    if plateau is None or (localisation is None and upstream is None):
        missing = []
        if plateau is None:
            missing.append("a rigid residual history (residual_history=)")
        if localisation is None and upstream is None:
            missing.append("the final residual maps (measured= and simulated=)")
        return GateVerdict(
            recommendation=Recommendation.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            headline="Cannot decide: missing " + " and ".join(missing) + ".",
            plateau=plateau, localisation=localisation, upstream=upstream,
            temporal=temporal, alternatives=(), notes=tuple(notes), config=cfg,
            context=_context(measured, simulated, residual_maps),
        )

    # (c) upstream alternatives, considered FIRST because the roadmap's order says so
    upstream_hits: list[str] = []
    if upstream is not None:
        ramp_ok = upstream.ramp_fraction < cfg.ramp_fraction_threshold
        alternatives.append(
            Alternative(
                name="phase_ramp_or_offset",
                excluded=ramp_ok,
                statistic=upstream.ramp_fraction,
                threshold=cfg.ramp_fraction_threshold,
                reason=(
                    f"a per-projection LINEAR ramp uniquely explains "
                    f"{upstream.ramp_fraction:.1%} of the residual energy "
                    f"({upstream.shared_fraction:.1%} is shared with the other terms and is "
                    f"credited to none of them; a further {upstream.nuisance_fraction:.1%} is "
                    "per-angle offset and gain, which are fitted out as nuisances because a "
                    "deforming sample produces them too)"
                    + (
                        "; below threshold, so stage 0 is done"
                        if ramp_ok
                        else ". Remove the ramp first (preprocess.remove_phase_ramp): it is "
                        "mathematically indistinguishable from a lateral shift, so it does not "
                        "merely add artifacts, it poisons the alignment meant to remove them"
                    )
                ),
            )
        )
        if not ramp_ok:
            upstream_hits.append("phase ramp")

        center_bar = max(cfg.center_tolerance_px, 2.0 * upstream.center_offset_se_px)
        center_ok = abs(upstream.center_offset_px) <= center_bar
        alternatives.append(
            Alternative(
                name="rotation_centre",
                excluded=center_ok,
                statistic=abs(upstream.center_offset_px),
                threshold=center_bar,
                reason=(
                    f"the constant part of the leftover horizontal shift is "
                    f"{upstream.center_offset_px:+.3f} +/- {upstream.center_offset_se_px:.3f} px, "
                    "which IS the rotation-axis error (the sin/cos part, "
                    f"{upstream.gauge_amplitude_px:.3f} px, is a translation of the object and is "
                    "unobservable -- it is not counted)"
                    + (
                        f"; within the bar of {center_bar:.3f} px"
                        if center_ok
                        else ". Refine the centre first"
                    )
                ),
            )
        )
        if not center_ok:
            upstream_hits.append("rotation centre")

        # Leftover rigid shift: drift (rigid stage unfinished) vs jitter (irreducible).
        shift_like = upstream.shift_fraction >= cfg.shift_fraction_threshold
        drifting = bool(np.isfinite(upstream.du_lag1) and upstream.du_lag1 >= cfg.drift_lag1_threshold)
        big = max(upstream.du_residual_rms_px, upstream.dv_residual_rms_px)
        drift_excluded = not (shift_like and drifting and big > cfg.shift_residual_tolerance_px)
        alternatives.append(
            Alternative(
                name="unconverged_rigid_drift",
                excluded=drift_excluded,
                statistic=big,
                threshold=cfg.shift_residual_tolerance_px,
                reason=(
                    f"per-angle shifts uniquely explain {upstream.shift_fraction:.1%} of the "
                    "residual; "
                    f"gauge-removed they are {upstream.du_residual_rms_px:.2f} px (du) and "
                    f"{upstream.dv_residual_rms_px:.2f} px (dv) RMS with lag-1 autocorrelation "
                    f"{upstream.du_lag1:+.2f} in "
                    + ("acquisition" if upstream.time_order_known else "angle")
                    + " order"
                    + (
                        "; not a smooth drift above tolerance"
                        if drift_excluded
                        else ". A smooth, correlated leftover shift is drift the rigid stage has "
                        "not finished removing -- run more rigid iterations, do not model it as "
                        "deformation"
                    )
                ),
            )
        )
        if not drift_excluded:
            upstream_hits.append("unconverged rigid drift")

        # Jitter is a *competing explanation*, so it has to explain more than the
        # alternative does: the leftover must be better described as per-angle shifts
        # than as anything the rigid basis cannot represent. A localised deformation
        # also leaks a little into the gradient basis (a bump on the object partly
        # looks like a small local shift), and without this comparison that leak alone
        # would veto every genuine deformation.
        jitter_like = bool(
            shift_like and not drifting and upstream.shift_fraction >= upstream.unexplained_fraction
        )
    else:
        jitter_like = False

    beyond_rigid = upstream.unexplained_fraction if upstream is not None else math.nan
    if upstream is not None:
        enough_left = beyond_rigid >= cfg.min_residual_beyond_rigid
        alternatives.append(
            Alternative(
                name="nothing_beyond_rigid",
                excluded=bool(enough_left),
                statistic=beyond_rigid,
                threshold=cfg.min_residual_beyond_rigid,
                reason=(
                    f"{beyond_rigid:.1%} of the residual energy survives the best per-angle "
                    "offset, gain, ramp and rigid shift"
                    + (
                        "; there is something here a rigid model genuinely cannot reach"
                        if enough_left
                        else ". The rigid model already explains essentially all of it, so "
                        "there is nothing left for a deformation field to describe -- "
                        "whatever it fits will be the numerical dust of that fit"
                    )
                ),
            )
        )

    # (b) the localisation alternatives
    if localisation is not None:
        alternatives.append(
            Alternative(
                name="spread_residual_noise",
                excluded=bool(localisation.is_localised),
                statistic=localisation.z,
                threshold=cfg.localisation_z,
                reason=(
                    f"residual energy in the hottest {cfg.top_fraction:.0%} of support blocks is "
                    f"{localisation.concentration:.3f} against a permutation null of "
                    f"{localisation.null_mean:.3f} +/- {localisation.null_std:.3f} "
                    f"(z = {localisation.z:.1f}, p = {localisation.p_value:.3f})"
                    + (
                        "; concentrated well beyond what its own pixel values force"
                        if localisation.is_localised
                        else ". A residual no more concentrated than its own permuted pixels is "
                        "spread noise, and a deformation field fitted to it fabricates structure"
                    )
                ),
            )
        )
        jitter_reason = (
            f"the per-angle residual maps correlate at {localisation.angle_consistency:+.3f} "
            f"(null {localisation.consistency_null_mean:+.3f} +/- "
            f"{localisation.consistency_null_std:.3f}, z = {localisation.consistency_z:.1f})"
        )
        if not localisation.is_angle_consistent:
            jitter_reason += (
                ". A residual that does not stay in the same places from angle to angle is "
                "jitter or noise: no geometric model, rigid or not, can remove it, and a "
                "deformation field fitted to it will fabricate structure"
            )
        elif jitter_like:
            jitter_reason += (
                " -- consistent enough -- BUT it is still mostly a per-angle rigid shift "
                f"({upstream.shift_fraction:.0%} of the energy, uniquely) with lag-1 "
                f"{upstream.du_lag1:+.2f}, i.e. random from projection to projection. That is "
                "jitter wearing the costume of a localised residual: it concentrates on the "
                "object edges, where a shift shows up, at every angle"
            )
        else:
            jitter_reason += (
                "; the same regions stay hot from angle to angle, which is a property of the "
                "sample and not of the stage"
            )
        alternatives.append(
            Alternative(
                name="angle_random_jitter",
                excluded=bool(localisation.is_angle_consistent and not jitter_like),
                statistic=localisation.consistency_z,
                threshold=cfg.consistency_z,
                reason=jitter_reason,
            )
        )

    if temporal is not None:
        alternatives.append(
            Alternative(
                name="static_reconstruction_artefact",
                excluded=bool(temporal.evolves),
                statistic=temporal.z,
                threshold=cfg.temporal_z,
                reason=(
                    f"grouped into {temporal.n_time_blocks} acquisition-time blocks, the "
                    f"residual maps decorrelate by {temporal.evolution:.3f} against "
                    f"{temporal.null_evolution:.3f} +/- {temporal.null_std:.3f} for randomly "
                    f"grouped projections (z = {temporal.z:.1f})"
                    + (
                        "; the pattern CHANGES during the scan, which is what a deforming "
                        "sample does and a reconstruction artefact does not"
                        if temporal.evolves
                        else ". The pattern does not change with time. A residual that is "
                        "localised and angle-consistent but STATIC is fixed in the object "
                        "frame: an FBP streak, an edge the projector cannot reproduce, an "
                        "unmodelled feature. Deformation is by definition something that "
                        "happens during the scan"
                    )
                ),
            )
        )

    # (a) the plateau alternative
    alternatives.append(
        Alternative(
            name="rigid_not_yet_plateaued",
            excluded=bool(plateau.plateaued),
            statistic=plateau.tail_improvement,
            threshold=cfg.plateau_tolerance,
            reason=plateau.reason,
        )
    )
    if plateau.at_noise_floor:
        alternatives.append(
            Alternative(
                name="plateau_is_the_noise_floor",
                excluded=False,
                statistic=plateau.floor_ratio,
                threshold=1.1,
                reason=(
                    f"the plateau sits at {plateau.floor_ratio:.2f} x the supplied noise floor, "
                    "i.e. essentially all of the leftover residual is counting noise. There is "
                    "nothing for a deformation model to explain"
                ),
            )
        )

    # -- pick the outcome, in the roadmap's order
    if upstream_hits:
        recommendation = Recommendation.FIX_UPSTREAM
        headline = (
            "Do NOT go non-rigid: the leftover residual is dominated by an unfixed upstream "
            "stage (" + ", ".join(upstream_hits) + "). Non-rigid alignment is the last stage "
            "and would absorb this error into a plausible deformation field, hiding it "
            "permanently."
        )
        confidence = _confidence_from_alternatives(
            alternatives,
            ("phase_ramp_or_offset", "rotation_centre", "unconverged_rigid_drift"),
        )
    elif not plateau.plateaued:
        recommendation = Recommendation.MORE_RIGID_ITERATIONS
        headline = "Not yet: " + plateau.reason + ". Finish the rigid alignment first."
        confidence = _monotone(abs(plateau.tail_improvement), cfg.plateau_tolerance)
    elif not plateau.shift_converged and np.isfinite(plateau.last_shift_px):
        recommendation = Recommendation.MORE_RIGID_ITERATIONS
        headline = (
            f"Not yet: the residual is flat but the rigid shift update is still "
            f"{plateau.last_shift_px:.3g} px RMS (tolerance {cfg.shift_tolerance_px:g} px), so "
            "the rigid solution is wandering rather than converged."
        )
        confidence = _monotone(plateau.last_shift_px, cfg.shift_tolerance_px)
    elif upstream is not None and beyond_rigid < cfg.min_residual_beyond_rigid:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: only {beyond_rigid:.1%} of the reprojection residual "
            "survives the best per-angle offset, gain, ramp and rigid shift. A rigid model "
            "already explains it; there is nothing for a deformation field to describe."
        )
        confidence = _monotone(cfg.min_residual_beyond_rigid, max(beyond_rigid, 1e-6))
    elif plateau.at_noise_floor:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: the residual plateaued at {plateau.floor_ratio:.2f} x the "
            "noise floor. What is left is counting noise, and a deformation field fitted to "
            "noise will fit it."
        )
        confidence = _monotone(1.0 / max(plateau.floor_ratio, 1e-6), 1.0 / 1.1)
    elif localisation is None:
        recommendation = Recommendation.INSUFFICIENT_EVIDENCE
        headline = (
            "Cannot decide: the residual plateaued, but its localisation could not be "
            "measured, and 'the residual is high' is not evidence for deformation -- it is "
            "the symptom every unfixed error shares."
        )
        confidence = 0.0
    elif not localisation.is_localised:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: the residual has plateaued but it is SPREAD, not localised "
            f"(concentration {localisation.concentration:.3f} against a null of "
            f"{localisation.null_mean:.3f}, z = {localisation.z:.1f}). That is jitter or noise; "
            "a deformation field fitted to it would overfit."
        )
        confidence = _monotone(cfg.localisation_z, max(localisation.z, 1e-6))
    elif not localisation.is_angle_consistent:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: the residual is concentrated but ANGLE-RANDOM "
            f"(consistency {localisation.angle_consistency:+.3f}, z = "
            f"{localisation.consistency_z:.1f}). Deformation is a property of the sample and "
            "stays in the same places from angle to angle; this does not, so it is jitter, "
            "noise or a detector defect."
        )
        confidence = _monotone(cfg.consistency_z, max(localisation.consistency_z, 1e-6))
    elif temporal is None:
        recommendation = Recommendation.INSUFFICIENT_EVIDENCE
        headline = (
            "Cannot decide: the residual is localised and angle-consistent, but without "
            "acquisition_index there is no way to ask whether it CHANGES during the scan -- "
            "and a static localised residual is a reconstruction artefact, not deformation. "
            "Measured on an undeformed phantom, localisation z = 108 and angle-consistency "
            "z = 305 with nothing moving at all. Supply the acquisition order (the "
            "non-rigid aligner requires it too, for the same reason)."
        )
        confidence = 0.0
    elif not temporal.evolves:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: the residual is localised and angle-consistent but it does "
            f"not CHANGE with acquisition time (z = {temporal.z:.1f}, threshold "
            f"{cfg.temporal_z:g}). A pattern that sits still in the object frame for the whole "
            "scan is a reconstruction artefact or an unmodelled static feature. Deformation is "
            "by definition something that happens during the scan."
        )
        confidence = _monotone(cfg.temporal_z, max(temporal.z, 1e-6))
    elif temporal is None:
        recommendation = Recommendation.INSUFFICIENT_EVIDENCE
        headline = (
            "Cannot decide: the residual is localised and angle-consistent, but without "
            "acquisition_index there is no way to ask whether it CHANGES during the scan -- "
            "and a static localised residual is a reconstruction artefact, not deformation. "
            "Measured on an undeformed phantom, localisation z = 108 and angle-consistency "
            "z = 305, with nothing moving at all. Supply the acquisition order; the non-rigid "
            "aligner requires it too, for the same reason."
        )
        confidence = 0.0
    elif not temporal.evolves:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: the residual is localised and angle-consistent but it does "
            f"not CHANGE with acquisition time (z = {temporal.z:.1f}, threshold "
            f"{cfg.temporal_z:g}). A pattern that sits still in the object frame for the whole "
            "scan is a reconstruction artefact or an unmodelled static feature. Deformation is "
            "by definition something that happens during the scan."
        )
        confidence = _monotone(cfg.temporal_z, max(temporal.z, 1e-6))
    elif jitter_like:
        recommendation = Recommendation.ACCEPT_RIGID
        headline = (
            f"Do NOT go non-rigid: what is left is still largely a per-angle rigid shift "
            f"({upstream.shift_fraction:.0%} of the residual energy) and it is angle-random "
            f"(lag-1 {upstream.du_lag1:+.2f}). That is jitter: irreducible by any geometric "
            "model, and exactly what a deformation field will happily absorb."
        )
        confidence = _monotone(upstream.shift_fraction, cfg.shift_fraction_threshold)
    else:
        recommendation = Recommendation.RUN_NONRIGID
        headline = (
            f"Go non-rigid: the rigid residual has plateaued ({plateau.tail_improvement:.2%} "
            f"over {plateau.window} iterations, shift update {plateau.last_shift_px:.3g} px); "
            f"{beyond_rigid:.0%} of what is left is beyond any rigid model, and it is localised "
            f"(z = {localisation.z:.1f}), angle-consistent (z = {localisation.consistency_z:.1f}) "
            f"and CHANGES over acquisition time (z = {temporal.z:.1f}), with no upstream stage "
            "explaining it."
        )
        confidence = min(
            _monotone(localisation.z, cfg.localisation_z),
            _monotone(localisation.consistency_z, cfg.consistency_z),
            _monotone(temporal.z, cfg.temporal_z),
        )
        notes.append(
            "A 'yes' here licenses running the method, not believing its output. Validate the "
            "recovered deformation on held-out projections (NonRigidAligner does this) and the "
            "volume by split-data FSC paired with a residual map -- FSC cannot see a "
            "common-mode geometric error."
        )

    if recommendation is not Recommendation.RUN_NONRIGID and localisation is not None:
        if localisation.is_localised and localisation.is_angle_consistent:
            notes.append(
                "The localisation test itself PASSED; the 'no' came from an earlier stage. Fix "
                "that and re-run this gate rather than concluding there is no deformation."
            )
    if recommendation is Recommendation.ACCEPT_RIGID and localisation is not None:
        if not localisation.is_localised:
            notes.append(
                "This criterion vetoes a GLOBAL deformation (uniform swelling, bulk thermal "
                "expansion), which is genuinely non-rigid but not localised. If you have "
                "independent reason to expect one -- a long scan of a beam-sensitive sample -- "
                "the localisation test is the wrong instrument and this 'no' is not evidence of "
                "rigidity."
            )

    return GateVerdict(
        recommendation=recommendation,
        confidence=float(confidence),
        headline=headline,
        plateau=plateau,
        localisation=localisation,
        upstream=upstream,
        temporal=temporal,
        alternatives=tuple(alternatives),
        notes=tuple(notes),
        config=cfg,
        context=_context(measured, simulated, residual_maps),
    )


def gate_from_engine(
    engine: Any,
    *,
    acquisition_index: np.ndarray | None = None,
    noise_floor: float | None = None,
    config: GateConfig | None = None,
    support: np.ndarray | None = None,
    seed: int = 0,
) -> GateVerdict:
    """Run the gate straight off a converged :class:`~...core.engine.AlignmentEngine`.

    Reads the residual/shift history from ``engine.history`` and the final residual maps
    from the engine's cached ``last_aligned`` / ``last_simulated`` -- the cache, not a
    fresh ``reproject()``, because TomoPy is not thread-safe and a GUI may be mid-run
    (see :attr:`AlignmentEngine.last_simulated`). Duck-typed on purpose: anything with
    ``history``, ``last_aligned``, ``last_simulated`` and ``state.angles`` works, so this
    is testable without a reconstruction backend.
    """
    history = list(getattr(engine, "history", []) or [])
    residuals = [float(r.residual) for r in history]
    shifts = [float(r.error) for r in history]
    measured = getattr(engine, "last_aligned", None)
    simulated = getattr(engine, "last_simulated", None)
    state = getattr(engine, "state", None)
    angles = getattr(state, "angles", None) if state is not None else None
    if measured is None or simulated is None:
        logger.warning(
            "The engine has no cached aligned/simulated stack (it has not stepped since the "
            "last invalidate_cache). The gate can only measure the plateau; run one more "
            "rigid iteration to refill the cache."
        )
    return evaluate_gate(
        residual_history=residuals or None,
        shift_history=shifts or None,
        measured=measured,
        simulated=simulated,
        angles=angles,
        acquisition_index=acquisition_index,
        support=support,
        noise_floor=noise_floor,
        config=config,
        seed=seed,
    )


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------


def format_gate(verdict: GateVerdict, *, width: int = 88) -> str:
    """A plain-text block for a log or a report. The JSON is :meth:`GateVerdict.to_json`."""
    import textwrap  # noqa: PLC0415

    def wrap(text: str, indent: int = 2) -> str:
        pad = " " * indent
        return textwrap.fill(text, width=width, initial_indent=pad, subsequent_indent=pad)

    lines = [
        "=" * width,
        f"NON-RIGID GATE: {verdict.recommendation.value.upper()}  "
        f"(confidence {verdict.confidence:.2f})",
        "=" * width,
        wrap(verdict.headline),
        "",
    ]
    if verdict.plateau is not None:
        p = verdict.plateau
        lines.append("PLATEAU")
        lines.append(
            f"  {p.n_iterations} rigid iterations, residual {p.value:.5g}, "
            f"tail improvement {p.tail_improvement:.2%} over {p.window}, "
            f"slope {p.slope_per_iter:+.2%}/iter (snr {p.snr:.1f}), "
            f"shift update {p.last_shift_px:.3g} px"
        )
    if verdict.localisation is not None:
        s = verdict.localisation
        lines.append("LOCALISATION")
        lines.append(
            f"  concentration {s.concentration:.3f} vs null {s.null_mean:.3f}"
            f" +/- {s.null_std:.3f}  (z {s.z:.1f}, p {s.p_value:.3f}, gini {s.gini:.2f})"
        )
        lines.append(
            f"  angle consistency {s.angle_consistency:+.3f} vs null "
            f"{s.consistency_null_mean:+.3f} (z {s.consistency_z:.1f}); "
            f"{s.n_support_blocks}/{s.n_blocks} blocks in support, {s.n_angles_used} angles"
        )
        lines.append(f"  support: {s.support_from}")
    if verdict.temporal is not None:
        t = verdict.temporal
        lines.append("TEMPORAL CHANGE")
        lines.append(
            f"  residual maps over {t.n_time_blocks} acquisition-time blocks decorrelate by "
            f"{t.evolution:.3f} vs {t.null_evolution:.3f} +/- {t.null_std:.3f} for random "
            f"groups (z {t.z:.1f}) -> {'EVOLVES' if t.evolves else 'static'}"
        )
    if verdict.upstream is not None:
        u = verdict.upstream
        lines.append("UPSTREAM DECOMPOSITION (unique contributions)")
        lines.append(
            f"  ramp {u.ramp_fraction:.1%} | shift {u.shift_fraction:.1%} | "
            f"offset+gain (nuisance) {u.nuisance_fraction:.1%} | "
            f"shared {u.shared_fraction:.1%} | unexplained {u.unexplained_fraction:.1%}"
        )
        lines.append(
            f"  centre error {u.center_offset_px:+.3f} +/- {u.center_offset_se_px:.3f} px "
            f"(gauge, not counted: "
            f"{u.gauge_amplitude_px:.3f} px); leftover shift {u.du_residual_rms_px:.2f} px du / "
            f"{u.dv_residual_rms_px:.2f} px dv, lag-1 {u.du_lag1:+.2f}"
        )
    if verdict.alternatives:
        lines.append("ALTERNATIVES CONSIDERED")
        for alt in verdict.alternatives:
            mark = "excluded" if alt.excluded else "NOT EXCLUDED"
            lines.append(f"  [{mark}] {alt.name}")
            lines.append(wrap(alt.reason, indent=6))
    if verdict.notes:
        lines.append("NOTES")
        for note in verdict.notes:
            lines.append(wrap("- " + note, indent=2))
    lines.append("=" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


def _monotone(value: float, threshold: float) -> float:
    """``0`` at the threshold, ``0.5`` at twice it, ``0.9`` at ten times it.

    The same shape as :func:`tktomo.diagnostics.confidence_from_ratio`, reimplemented so
    this module does not depend on an optional package for a four-line function. It is
    deliberately not a probability: it orders verdicts, it does not price them.
    """
    if threshold <= 0 or not math.isfinite(value) or value <= threshold:
        return 0.0
    return float(1.0 - threshold / value)


def _confidence_from_alternatives(
    alternatives: Sequence[Alternative], names: Sequence[str]
) -> float:
    """Strongest monotone confidence among the named alternatives that were not excluded."""
    best = 0.0
    for alt in alternatives:
        if alt.name in names and not alt.excluded:
            best = max(best, _monotone(alt.statistic, alt.threshold))
    return best


def _context(
    measured: np.ndarray | None, simulated: np.ndarray | None, residual: np.ndarray | None
) -> dict[str, Any]:
    shape = None
    for candidate in (measured, simulated, residual):
        if candidate is not None:
            shape = list(np.asarray(candidate).shape)
            break
    return {
        "stack_shape": shape,
        "has_simulated": simulated is not None,
        "diagnostics_available": DIAGNOSTICS_AVAILABLE,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        # JSON has no NaN or Infinity. Emitting them produces a file that json.load
        # accepts and every other parser rejects, so they become null here.
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)

"""Non-rigid (deformation-field) alignment: the last stage of the roadmap.

An implementation of Odstrcil et al., *Ab initio nonrigid X-ray nanotomography*,
Nat. Commun. **10**, 4778 (2019), in the practical form that can be bolted onto an
existing projector:

1. Split the projections into subsets **ordered by acquisition time**.
2. Reconstruct each subset -> a sequence of "partial" volumes, each showing the
   sample as it was during one stretch of the scan.
3. Estimate a deformation vector field (DVF) between each partial and a common
   reference volume, by 3D optical flow. The reference is the temporally central
   partial by default (``reference_mode``), so both volumes in every registration are
   single deformation states with the same angular sampling and the same streak
   character.
4. Smooth and interpolate those fields in **acquisition time** to get a deformation
   for every projection.
5. Warp the volume by that projection's DVF *before* projecting it. Warping the
   volume is mathematically equivalent to curving the rays and is far easier to
   implement on top of a projector that only knows about straight ones.
6. Iterate: estimate what deformation is *left* after the current field, compose the
   two coarse fields, and rebuild the volume reference by carrying every partial back
   into the common frame through the inverse of its own field. Refining rather than
   re-estimating matters because a regularised flow estimate is systematically too
   small; composing successive increments is a Richardson iteration that recovers some
   of what one estimate leaves behind.

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
Measured on a synthetic phantom (32 x 64 x 64 voxels, six sub-tomograms of 60 angles,
a known deformation that drifts and shears over the scan, 1.78 px rms after gauge
fixing), at the defaults and six iterations:

* recovered DVF **0.75 px rms** from the truth inside the object support, **1.46 px**
  over the whole coarse grid including the air where the flow is unconstrained;
* **correlation 0.97** with the true field, but only **0.59** of its amplitude;
* reprojection residual 0.1718 -> 0.1393 on the fitted projections (+18.9%) and
  0.1750 -> 0.1424 on the held-out ones (+18.6%) -- the two move together, which is
  what says the deformation is real;
* on the same phantom with **no** deformation the run stops at the first iteration
  with **0.13 px rms / 0.33 px max** of invented field, flagged.

The amplitude shortfall is not a bug to be tuned away: ``flow_alpha`` moves the
recovered amplitude and the deformation invented from rigid data together, and the
measured trade on the small version of that phantom (1.24 px rms truth) is::

    flow_alpha   error vs truth   amplitude   correlation   invented on rigid data
       0.3          0.62 px         0.64          0.68           0.40 px rms
       0.5          0.51 px         0.63          0.78           0.26 px rms
       1.0          0.46 px         0.59          0.84           0.12 px rms
       2.0          0.51 px         0.50          0.78           0.05 px rms  <- default
       4.0          0.60 px         0.37          0.69           0.03 px rms
       8.0          0.67 px         0.25          0.58           0.01 px rms

The default sits where the method still finds most of the deformation while inventing
under a tenth of a pixel from nothing. Read a recovered DVF as "where, when and in
which direction the sample moved"; do not quote it as a calibrated strain.

WHEN NOT TO RUN THIS, which matters more than how to run it
-----------------------------------------------------------
Non-rigid alignment is the **last** stage. The roadmap's order is: ramp/offset
removal, rotation centre, vertical, horizontal, geometry refinement, and only then
this. A deformation field has enough freedom to absorb an unfixed rigid error and
will happily do so, producing a sharp volume built on a wrong geometry, which no
amount of downstream analysis can detect. So:

* :class:`NonRigidAligner` **refuses to start** on data that does not look rigidly
  aligned. The check is data-driven (a centre-of-mass consistency test on the
  projections themselves, see :func:`rigid_alignment_is_plausible`), not a promise
  from the caller, because the caller is exactly who gets this wrong.
* The decision to go non-rigid at all is *evidential*, and
  :func:`nonrigid_is_warranted` implements the roadmap's two tests: the reprojection
  residual must have **plateaued** after the best rigid alignment, and the remaining
  residual must be **localised** to sub-regions rather than spread over the frame. A
  residual that is spread out and uncorrelated between angles is jitter or noise;
  fitting a deformation field to it only overfits.

The overfitting guards, in the order they bind
----------------------------------------------
1. **A coarse DVF grid** (``grid_spacing``). A hard restriction of the model space:
   of order ``3*(N/spacing)^3`` parameters per subset against ``N^3`` voxels.
2. **Temporal smoothing and interpolation** (``time_sigma``). K fields describe N
   projections; deformation that is not smooth in time cannot be represented.
3. **A magnitude cap** (``max_dvf_px``). Optical flow that has locked onto a streak
   returns tens of voxels; a warp that large makes anything match anything.
4. **A monitored data-consistency residual, fitted and held out.** A fraction of the
   projections
   (``holdout_fraction``) is excluded from every partial reconstruction and from the
   reference volume, so they are predicted, never fitted. The fitted residual always
   improves -- more parameters always fit better. Only the held-out residual can tell
   you whether the deformation is real. If the fitted residual improves and the
   held-out one does not, :attr:`NonRigidResult.overfitting` says so and
   :meth:`NonRigidAligner.run` stops, exactly as the rigid engine stops on a runaway
   shift update.

Verify the finished result by split-data FSC, never by looking at slices -- and pair
the FSC with reprojection-residual maps, because FSC is blind to common-mode
geometric error (both halves share it, so it does not show up as a disagreement).

The seam with the rigid engine
------------------------------
This module implements **no projector and no reconstructor**. It fetches a
:class:`~tktomo.recon.backend.ReconBackend` from the same registry
``AlignmentEngine`` uses (``tktomo.recon.get_backend(config.backend)``), calls the
same ``reconstruct``/``reproject`` methods with the same ``center``, and reuses
:func:`~tktomo.ptycho_align.core.engine.row_chunk_size` for interruptible chunking.
The entire non-rigid addition is one composition in the forward model::

    simulated = backend.reproject(warp_volume(volume, dvf), angles, center=center)

Everything else -- geometry, centre, algorithm, chunking, cancellation -- is the
engine's, unchanged. :meth:`NonRigidAligner.step` runs exactly one outer iteration
and returns, matching :meth:`AlignmentEngine.step`, so a GUI or a benchmark can step
the rigid and non-rigid stages identically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tktomo.ptycho_align.core.deformation import (
    DeformationField,
    DeformationSequence,
    compose,
    estimate_flow,
    invert,
    warp_volume,
)
from tktomo.ptycho_align.core.state import VolumePolicy

logger = logging.getLogger(__name__)

__all__ = [
    "LocalisationReport",
    "NonRigidAligner",
    "NonRigidConfig",
    "NonRigidResult",
    "NonRigidVerdict",
    "RigidAlignmentRequired",
    "RigidEvidence",
    "angular_gap_deg",
    "nonrigid_is_warranted",
    "residual_localisation",
    "rigid_alignment_is_plausible",
    "time_subsets",
]


class RigidAlignmentRequired(RuntimeError):
    """Raised when non-rigid alignment is asked to run before rigid alignment is done.

    Deliberately not a warning. Running non-rigid on top of an unfixed rigid error is
    the single failure mode the roadmap's order-of-operations rule exists to prevent,
    and it produces a *plausible* result, so it cannot be caught downstream.
    """


# ---------------------------------------------------------------------------------
# The evidential gate: is non-rigid warranted at all?
# ---------------------------------------------------------------------------------


@dataclass
class LocalisationReport:
    """Where the leftover reprojection residual sits in the frame.

    ``concentration`` is the fraction of the total residual energy carried by the
    hottest 10% of image blocks, averaged over angles. A residual spread uniformly
    over the frame scores 0.10; one confined to a tenth of the frame scores 1.0.

    ``angle_consistency`` is the mean pairwise correlation between the per-angle block
    energy maps. Deformation is a property of the *sample*, so the same regions stay
    hot from angle to angle and this is high. Jitter, noise and residual ramp are
    angle-random and score near zero -- and a non-rigid model fitted to those will
    fabricate structure, which is why both numbers are needed, not just the first.

    Two honest limitations, because this is a heuristic gate and not a proof:

    * It is a test for **localised** deformation, the kind Odstrcil et al. saw in a
      beetle. A sample that deforms *globally* -- uniform thermal drift, bulk swelling --
      produces a residual spread over the whole frame and will be judged "not
      localised", i.e. this will veto a case the method could actually have handled.
      Measured on a synthetic scan with a genuine global deformation: concentration
      0.24, consistency 0.01, verdict False.
    * ``angle_consistency`` compares residual maps across the *whole* scan. When the
      deformation evolves in time, projections from different parts of the scan see
      genuinely different states, so the consistency is low even though the cause is
      real deformation. Compute it within one acquisition-time block to see the effect
      cleanly.
    """

    concentration: float
    angle_consistency: float
    block: int
    is_localised: bool


def residual_localisation(
    residual: np.ndarray, *, block: int = 32, top_fraction: float = 0.1,
    concentration_threshold: float = 0.30, consistency_threshold: float = 0.3,
) -> LocalisationReport:
    """Measure whether a stack of residual maps ``(n_angles, n_v, n_u)`` is localised.

    Pass ``measured - simulated`` from the best rigid alignment. See
    :class:`LocalisationReport` for what the two numbers mean and why one is not
    enough.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 3:
        raise ValueError(f"residual must be (n_angles, n_v, n_u), got {residual.shape}")
    n_angles, n_v, n_u = residual.shape
    block = max(1, min(int(block), n_v, n_u))
    bv, bu = n_v // block, n_u // block
    if bv < 2 or bu < 2:
        raise ValueError(
            f"block={block} px leaves only {bv}x{bu} blocks in a {n_v}x{n_u} frame; "
            "localisation cannot be measured on fewer than 2x2."
        )

    trimmed = residual[:, : bv * block, : bu * block] ** 2
    energy = trimmed.reshape(n_angles, bv, block, bu, block).sum(axis=(2, 4))
    flat = energy.reshape(n_angles, -1)
    totals = flat.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    share = flat / totals

    keep = max(1, int(round(top_fraction * share.shape[1])))
    hottest = np.sort(share, axis=1)[:, -keep:].sum(axis=1)
    concentration = float(hottest.mean())

    centred = share - share.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    norms[norms == 0] = 1.0
    unit = centred / norms[:, None]
    correlations = unit @ unit.T
    off_diagonal = correlations[~np.eye(n_angles, dtype=bool)]
    consistency = float(off_diagonal.mean()) if off_diagonal.size else 0.0

    return LocalisationReport(
        concentration=concentration,
        angle_consistency=consistency,
        block=block,
        is_localised=(
            concentration >= concentration_threshold and consistency >= consistency_threshold
        ),
    )


@dataclass
class RigidEvidence:
    """What the rigid stage produced, as evidence for or against going non-rigid.

    ``residuals`` and ``shift_rms`` are the per-outer-iteration
    ``IterationResult.residual`` and ``.error`` of the rigid run, in order.
    ``localisation`` is an optional :class:`LocalisationReport` on the final residual
    maps.
    """

    residuals: np.ndarray
    shift_rms: np.ndarray
    localisation: LocalisationReport | None = None

    def __post_init__(self) -> None:
        self.residuals = np.asarray(self.residuals, dtype=np.float64).ravel()
        self.shift_rms = np.asarray(self.shift_rms, dtype=np.float64).ravel()

    @classmethod
    def from_engine(cls, engine: Any, localisation: LocalisationReport | None = None):
        """Pull the residual/shift history straight out of an :class:`AlignmentEngine`."""
        history = list(getattr(engine, "history", []))
        if not history:
            raise RigidAlignmentRequired(
                "the rigid engine has run no iterations, so there is no rigid solution "
                "to start from. Run the rigid alignment to convergence first."
            )
        return cls(
            residuals=np.array([r.residual for r in history]),
            shift_rms=np.array([r.error for r in history]),
            localisation=localisation,
        )


@dataclass
class NonRigidVerdict:
    warranted: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.warranted


def nonrigid_is_warranted(
    evidence: RigidEvidence,
    *,
    window: int = 4,
    plateau_tolerance: float = 0.02,
    shift_tolerance: float = 0.1,
) -> NonRigidVerdict:
    """The roadmap's evidential test, as a function you can put in a report.

    Two conditions, both necessary:

    * **Plateau.** Over the last ``window`` rigid iterations the residual improved by
      less than ``plateau_tolerance`` (relative), and the shift updates have fallen
      below ``shift_tolerance`` px. A residual still falling means the rigid stage is
      not finished, and the honest move is to run more rigid iterations.
    * **Localisation.** The leftover residual is concentrated in sub-regions *and*
      consistent between angles (see :func:`residual_localisation`). Without a
      :class:`LocalisationReport` this cannot be judged, and the verdict is "no" --
      absence of evidence is not evidence, and the default has to be the conservative
      one for a method with this much freedom.
    """
    reasons: list[str] = []
    residuals = evidence.residuals
    if residuals.size < window:
        return NonRigidVerdict(
            False,
            [
                f"only {residuals.size} rigid iteration(s) recorded; at least {window} are "
                "needed to tell a plateau from a still-falling residual"
            ],
        )

    recent = residuals[-window:]
    improvement = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-12)
    plateaued = improvement < plateau_tolerance
    if not plateaued:
        reasons.append(
            f"the rigid residual is still falling ({improvement:.1%} over the last "
            f"{window} iterations, tolerance {plateau_tolerance:.1%}). Finish the rigid "
            "alignment before adding a deformation model."
        )
    last_shift = float(evidence.shift_rms[-1]) if evidence.shift_rms.size else float("nan")
    if not np.isfinite(last_shift) or last_shift > shift_tolerance:
        reasons.append(
            f"the rigid shift update is still {last_shift:.3g} px (tolerance "
            f"{shift_tolerance:g} px); the rigid solution has not converged."
        )

    report = evidence.localisation
    if report is None:
        reasons.append(
            "no residual-localisation report was supplied, so it is unknown whether the "
            "leftover residual is localised (deformation) or spread and angle-random "
            "(jitter). Compute one with residual_localisation() on the final "
            "measured-minus-reprojected maps."
        )
    elif not report.is_localised:
        reasons.append(
            f"the leftover residual is not localised: {report.concentration:.0%} of its "
            f"energy is in the hottest 10% of blocks and the angle-to-angle consistency "
            f"is {report.angle_consistency:.2f}. That pattern is jitter or noise, not "
            "sample deformation, and a deformation field fitted to it will overfit."
        )

    return NonRigidVerdict(not reasons, reasons or ["residual has plateaued and is localised"])


def rigid_alignment_is_plausible(
    projections: np.ndarray, angles: np.ndarray, *, tolerance_px: float = 2.0
) -> str | None:
    """Does this stack look like it has already been rigidly aligned? Reason, or None.

    A mass-conserving projection stack of a rigid object has a **constant** vertical
    centroid and a **sinusoidal** horizontal one (see
    :mod:`~tktomo.ptycho_align.core.com`). After a good rigid alignment both hold to a
    fraction of a pixel; before one, they do not. That makes this a check on the
    *data*, not on the caller's word, which is the point -- the caller believing the
    data is aligned is exactly the situation this has to catch.

    ``tolerance_px`` is deliberately loose (2 px by default): real deformation itself
    perturbs the centroid, so a tight threshold would reject the very datasets this
    method exists for. It is a guard against gross misalignment, not a convergence
    test; :func:`nonrigid_is_warranted` is the convergence test.

    Returns None if the check cannot be run at all (for instance the projections have
    no positive mass because the phase is negative and has not been inverted), after
    logging a warning -- a check that cannot run must not masquerade as a check that
    passed, but neither should it block a legitimate run.
    """
    from tktomo.ptycho_align.core.com import com_prealign  # noqa: PLC0415

    try:
        result = com_prealign(projections, angles)
    except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
        logger.warning(
            "Could not run the centre-of-mass rigid-alignment check (%s). Proceeding "
            "WITHOUT it: verify the rigid alignment yourself before trusting any "
            "deformation field this produces.",
            exc,
        )
        return None

    vertical = float(np.sqrt(np.mean(result.sy**2)))
    horizontal = float(result.fit_residual)
    worst = max(vertical, horizontal)
    if worst <= tolerance_px:
        return None
    return (
        f"the projections do not look rigidly aligned: the vertical centroid varies by "
        f"{vertical:.2f} px RMS and the horizontal centroid departs from its sinusoid by "
        f"{horizontal:.2f} px RMS, against a tolerance of {tolerance_px:g} px. Non-rigid "
        "alignment is the LAST stage -- run the rigid alignment (and check the rotation "
        "centre and the phase-ramp removal) first. A deformation field will absorb a "
        "rigid error and hide it."
    )


# ---------------------------------------------------------------------------------
# Subsets in acquisition time
# ---------------------------------------------------------------------------------


def angular_gap_deg(angles: np.ndarray) -> float:
    """Largest gap in angular coverage, in degrees, modulo 180 (parallel beam).

    A subset with a large gap has a missing wedge, and its reconstruction is elongated
    along the missing direction. Optical flow between such a partial and the reference
    measures the missing wedge, not the deformation, and it does so with a smooth,
    entirely plausible-looking field. This is the single most dangerous failure mode of
    the whole method, which is why it is checked rather than assumed.
    """
    a = np.sort(np.mod(np.degrees(np.asarray(angles, dtype=np.float64)), 180.0))
    if a.size < 2:
        return 180.0
    return float(np.max(np.diff(np.concatenate([a, [a[0] + 180.0]]))))


def time_subsets(
    acquisition_index: np.ndarray, n_subsets: int, mode: str = "time_blocks"
) -> list[np.ndarray]:
    """Split projection indices into ``n_subsets`` groups ordered by acquisition time.

    ``mode`` is the trade this method lives or dies by, and neither option is right
    for every scan:

    * ``"time_blocks"`` (default) -- contiguous blocks of acquisition time. Each subset
      is **one deformation state**, which is what makes a DVF between it and the
      reference mean anything. It is correct only if each block is *angularly
      complete*, i.e. if the scan interleaves angles (golden-ratio / interlaced
      schemes) or repeats whole tomograms in a series. That is the geometry Odstrcil
      et al. used, and it is the geometry of a P06 series where the sample is scanned
      through 0-180 several times over many hours.
    * ``"interleaved"`` -- every ``n_subsets``-th projection in time order. Each subset
      spans the full angular range *and* the full time range, so every partial shows
      the same time-averaged sample and the deformation contrast between them is
      close to zero. Offered because for a single sequential 0->180 sweep it is the
      only split that reconstructs at all, but it recovers almost nothing, and that is
      an honest statement about the data rather than a fixable defect: **a single
      sequential sweep does not identify a time-varying deformation.** If that is your
      scan, say so in the paper instead of running this.

    :class:`NonRigidAligner` checks the angular coverage of every subset it builds and
    refuses (by default) if a subset has a missing wedge, so the failure above is
    caught rather than absorbed into a plausible field.
    """
    acquisition_index = np.asarray(acquisition_index)
    if acquisition_index.ndim != 1:
        raise ValueError(f"acquisition_index must be 1D, got shape {acquisition_index.shape}")
    n = acquisition_index.size
    if n_subsets < 2:
        raise ValueError(f"n_subsets must be >= 2, got {n_subsets}")
    if n_subsets > n:
        raise ValueError(f"n_subsets={n_subsets} exceeds the {n} projections")

    order = np.argsort(acquisition_index, kind="stable")
    if mode == "time_blocks":
        return [np.sort(part) for part in np.array_split(order, n_subsets)]
    if mode == "interleaved":
        return [np.sort(order[k::n_subsets]) for k in range(n_subsets)]
    raise ValueError(f"unknown subset mode {mode!r}; expected 'time_blocks' or 'interleaved'")


# ---------------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------------


@dataclass
class NonRigidConfig:
    """Parameters of the non-rigid stage. Mirrors :class:`AlignConfig` where they overlap."""

    # -- the deformation model
    n_subsets: int = 6
    subset_mode: str = "time_blocks"  # see time_subsets() for the trade
    # Which volume the optical flow registers each partial against:
    #   "central" -- the temporally central partial. Both volumes are then single
    #                deformation states with identical angular sampling and identical
    #                streak character, so the flow is not damped by registering a sharp
    #                volume against a blurred one, and the common-mode artefacts cancel.
    #                The gauge is fixed by construction: zero deformation at the middle
    #                of the scan.
    #   "average"  -- the deformation-corrected mean of all partials. Less noisy, but it
    #                is an average over the deformation that has not been modelled yet,
    #                so it is blurred, and flow to a blurred target under-estimates.
    reference_mode: str = "central"
    # DVF node spacing in VOXELS -- the main regulariser. Aim for 8-20 nodes per axis:
    # on a 1488^3 tomogram that is a spacing of 75-190 voxels, not 16. The default suits
    # the few-hundred-voxel volumes this is usually prototyped on; _prepare() logs the
    # resulting parameter count against the voxel count, and warns when it gets rich.
    grid_spacing: float = 16.0
    time_sigma: float = 0.75  # Gaussian smoothing of the fields across subsets
    max_dvf_px: float = 8.0  # magnitude cap per node
    n_time_bins: int = 0  # volume warps per residual pass; 0 -> 2 * n_subsets

    # -- optical flow
    flow_method: str = "horn_schunck"  # scipy-only; "tvl1" needs scikit-image
    flow_alpha: float = 2.0  # smoothness weight (dimensionless, see estimate_flow)
    flow_levels: int = 3
    flow_iterations: int = 40
    flow_warps: int = 3
    # Low-pass applied to both volumes before the flow is estimated; negative means
    # "0.25 x grid_spacing". OFF by default: measured on the synthetic phantom it buys
    # a further 25% cut in spurious deformation (0.090 -> 0.069 px rms on rigid data)
    # but costs more than that in accuracy (0.51 -> 0.59 px rms against the truth),
    # because flow_alpha is already doing that job better. Reach for it when the
    # partial reconstructions are visibly streaky and raising flow_alpha has started to
    # flatten the deformation you are trying to measure.
    flow_prefilter_sigma: float = 0.0
    warp_order: int = 1  # interpolation order for warp_volume

    # -- validation
    holdout_fraction: float = 0.15  # projections never fitted, used to detect overfitting
    holdout_seed: int = 0
    overfit_tolerance: float = 0.01  # held-out residual may not worsen by more than this
    min_holdout_share: float = 0.2  # held-out must capture >= this share of the fitted gain
    min_fitted_gain: float = 0.002  # below this the model is not describing anything
    # How far the held-out gain may fall back from the previous iteration before the
    # refinement is called finished. 0.0 means "any fall back stops it", which errs
    # towards stopping one iteration early rather than one late -- the right direction
    # for a model whose failure mode is inventing structure.
    holdout_stall_tolerance: float = 0.0

    # -- preconditions
    require_rigid: bool = True
    max_rigid_residual_px: float = 2.0
    require_angular_coverage: bool = True
    max_angular_gap_deg: float = 20.0

    # -- reconstruction (handed straight to the ReconBackend, as AlignConfig does)
    recon_algorithm: str = "gridrec"
    recon_inner_iters: int = 2
    backend: str = "tomopy"
    ncore: int | None = None
    row_chunk: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NonRigidConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"Unknown NonRigidConfig field(s): {', '.join(unknown)}. "
                "The config was probably written by a newer version of tktomo."
            )
        return cls(**dict(raw))

    @classmethod
    def from_align_config(cls, align: Any, **overrides: Any) -> "NonRigidConfig":
        """Inherit the reconstruction settings from the rigid stage's config.

        The non-rigid stage must reconstruct with the same backend, algorithm and core
        count as the rigid stage it follows, or its partial volumes are not comparable
        with the rigid solution's and the residuals cannot be read against each other.
        """
        base = dict(
            recon_algorithm=getattr(align, "recon_algorithm", cls.recon_algorithm),
            recon_inner_iters=getattr(align, "recon_inner_iters", cls.recon_inner_iters),
            backend=getattr(align, "backend", cls.backend),
            ncore=getattr(align, "ncore", cls.ncore),
            row_chunk=getattr(align, "row_chunk", cls.row_chunk),
        )
        base.update(overrides)
        return cls(**base)


@dataclass
class NonRigidResult:
    """Everything one outer non-rigid iteration produced. Mirrors ``IterationResult``."""

    iteration: int
    sequence: DeformationSequence
    residual: float  # data consistency on the FITTED projections
    holdout_residual: float  # ... and on the projections never fitted
    baseline_residual: float  # the same, with no deformation at all (rigid only)
    baseline_holdout_residual: float
    dvf_rms_px: float
    dvf_max_px: float
    wallclock_s: float
    volume: np.ndarray | None = None
    overfitting: str | None = None  # why this iteration looks like overfitting, or None

    @property
    def fitted_gain(self) -> float:
        """Relative improvement of the fitted residual over the rigid-only baseline."""
        return _relative_gain(self.baseline_residual, self.residual)

    @property
    def holdout_gain(self) -> float:
        """The number that decides whether the deformation is real."""
        return _relative_gain(self.baseline_holdout_residual, self.holdout_residual)


def _relative_gain(baseline: float, value: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return float((baseline - value) / baseline)


# ---------------------------------------------------------------------------------
# The aligner
# ---------------------------------------------------------------------------------


@dataclass
class NonRigidAligner:
    """Non-rigid alignment, one outer iteration per :meth:`step`.

    ``projections`` must be the **rigidly aligned** stack (the engine's
    ``aligned_projections()``), not the raw one: this stage operates on the residual
    the rigid stage could not remove. ``acquisition_index`` is required and must be
    the order in which the projections were *acquired*, which is not the order of
    ``angles`` unless the scan happened to be a single monotonic sweep. Getting it
    from the angles would be silently wrong for interlaced schemes and for a repeated
    series -- precisely the schemes this method needs -- so there is no default.

    Qt-free and backend-agnostic, like :class:`AlignmentEngine`.
    """

    projections: np.ndarray
    angles: np.ndarray
    acquisition_index: np.ndarray
    center: float | None = None
    config: NonRigidConfig = field(default_factory=NonRigidConfig)
    backend: Any | None = None
    rigid_evidence: RigidEvidence | None = None
    # Same memory policy as the rigid engine: reconstructed volumes are large and a
    # long run would otherwise pin one per iteration.
    policy: VolumePolicy = field(default_factory=VolumePolicy)

    def __post_init__(self) -> None:
        self.projections = np.asarray(self.projections, dtype=np.float32)
        if self.projections.ndim != 3:
            raise ValueError(
                f"projections must be (n_angles, n_v, n_u), got {self.projections.shape}"
            )
        n = self.projections.shape[0]
        self.angles = np.asarray(self.angles, dtype=np.float64).ravel()
        if self.angles.size != n:
            raise ValueError(f"{self.angles.size} angles for {n} projections")
        self.acquisition_index = np.asarray(self.acquisition_index).ravel()
        if self.acquisition_index.size != n:
            raise ValueError(
                f"{self.acquisition_index.size} acquisition indices for {n} projections. "
                "Acquisition order is required and is NOT angle order -- pass the scan "
                "number, the file index, or the timestamp of each projection."
            )
        if self.center is None:
            self.center = self.projections.shape[2] / 2.0
        self.center = float(self.center)

        cfg = self.config
        if cfg.require_rigid:
            reason = rigid_alignment_is_plausible(
                self.projections, self.angles, tolerance_px=cfg.max_rigid_residual_px
            )
            if reason:
                raise RigidAlignmentRequired(reason)
            if self.rigid_evidence is not None:
                verdict = nonrigid_is_warranted(self.rigid_evidence)
                if not verdict:
                    raise RigidAlignmentRequired(
                        "the rigid stage does not warrant going non-rigid: "
                        + " ".join(verdict.reasons)
                    )

        self._holdout, self._fitted = self._split_holdout(n)
        self._subsets = self._build_subsets()
        self._times = np.array(
            [float(np.mean(self.acquisition_index[s])) for s in self._subsets]
        )
        self._check_angular_coverage()

        self._partials: list[np.ndarray] | None = None
        self._reference: np.ndarray | None = None
        self._flow_reference: np.ndarray | None = None
        self._sequence: DeformationSequence | None = None
        self._baseline: tuple[float, float] | None = None
        self._history: list[NonRigidResult] = []

    # -- accessors ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        return self._history[-1].iteration if self._history else 0

    @property
    def history(self) -> list[NonRigidResult]:
        return self._history

    @property
    def sequence(self) -> DeformationSequence | None:
        """The current deformation sequence, or None before the first step."""
        return self._sequence

    @property
    def reference_volume(self) -> np.ndarray | None:
        return self._reference

    @property
    def subsets(self) -> list[np.ndarray]:
        return self._subsets

    @property
    def holdout(self) -> np.ndarray:
        """Indices of the projections that are never fitted."""
        return self._holdout

    def deformation_at(self, acquisition_time: float) -> DeformationField:
        if self._sequence is None:
            raise ValueError("no deformation yet -- run at least one iteration")
        return self._sequence.at(acquisition_time)

    def revert_to(self, iteration: int) -> NonRigidResult:
        """Roll the deformation model back to the end of ``iteration``.

        The counterpart to :meth:`run` stopping on an overfitting flag: the flagged
        iteration is *recorded*, not discarded, so the caller can see what went wrong
        and then step back to the last one that was still earning its parameters.
        Mirrors :meth:`AlignmentState.revert_to`, including that it can fail if the
        volume policy has already dropped that iteration's volume -- the deformation
        sequence is always kept, being a few kilobytes.
        """
        kept = [r for r in self._history if r.iteration <= iteration]
        if not kept or kept[-1].iteration != iteration:
            raise ValueError(
                f"cannot revert to iteration {iteration}; history holds "
                f"{[r.iteration for r in self._history]}"
            )
        target = kept[-1]
        if target.volume is None:
            raise ValueError(
                f"iteration {iteration}'s volume was dropped by the memory policy; its "
                "deformation sequence is still available as history[...].sequence, but "
                "the reference volume would have to be rebuilt."
            )
        self._history = kept
        self._sequence = target.sequence
        self._reference = target.volume
        return target

    def warped_volume(self, acquisition_time: float) -> np.ndarray:
        """The reference volume deformed into the state it was in at that time."""
        if self._reference is None:
            raise ValueError("no volume yet -- run at least one iteration")
        return warp_volume(
            self._reference, self.deformation_at(acquisition_time), order=self.config.warp_order
        )

    # -- construction from the rigid stage --------------------------------------------

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        acquisition_index: np.ndarray,
        *,
        config: NonRigidConfig | None = None,
        localisation: LocalisationReport | None = None,
        **kwargs: Any,
    ) -> "NonRigidAligner":
        """Start from a converged :class:`AlignmentEngine`, taking its rigid solution.

        Uses ``engine.aligned_projections()``, which applies the cumulative shifts to
        the *pristine* stack exactly once -- the engine's "never re-shift shifted data"
        rule, which matters here because this stage then warps that stack's
        reconstruction, and stacked interpolations are what destroy resolution.
        """
        evidence = RigidEvidence.from_engine(engine, localisation)
        if config is None:
            config = NonRigidConfig.from_align_config(engine.config)
        return cls(
            projections=engine.aligned_projections(),
            angles=engine.state.angles,
            acquisition_index=acquisition_index,
            center=engine.state.center,
            config=config,
            rigid_evidence=evidence,
            **kwargs,
        )

    # -- one outer iteration ----------------------------------------------------------

    def step(
        self,
        cancel: Any | None = None,
        report: Callable[[float, str], None] | None = None,
    ) -> NonRigidResult:
        """Run exactly ONE outer iteration, and return what it produced.

        Same contract as :meth:`AlignmentEngine.step`: ``cancel`` is an event checked
        between the expensive sub-steps and raises
        :class:`~tktomo.ptycho_align.core.engine.Cancelled`, which records nothing;
        ``report(fraction, message)`` is called as the iteration proceeds.
        """
        started = time.perf_counter()
        cfg = self.config

        if self._partials is None:
            self._prepare(cancel, report)
        assert self._partials is not None and self._reference is not None  # noqa: S101
        assert self._baseline is not None  # noqa: S101

        # 1. Refine the field for each subset: estimate what deformation is LEFT over
        #    after the field we already have, then compose the two.
        #
        #    Why not simply re-estimate the whole field each iteration: the smoothness
        #    prior that keeps this method honest also makes every single flow estimate
        #    systematically *too small* -- measured at 30-40% of the truth on a
        #    synthetic phantom. Re-estimating from scratch reproduces that same
        #    shrunken answer forever. Refining instead makes the iteration a Richardson
        #    scheme: each pass recovers a fraction of what is left, so the total
        #    converges towards the true amplitude while every individual estimate stays
        #    as conservative as before.
        #
        #    The composition is done on the coarse fields (deformation.py convention 3).
        #    The volume handed to the flow is warped ONCE, from the reference that was
        #    itself rebuilt from pristine partials this iteration -- never a warp of a
        #    warp -- so no interpolation blur accumulates across iterations.
        prefilter = (
            0.25 * cfg.grid_spacing
            if cfg.flow_prefilter_sigma < 0
            else cfg.flow_prefilter_sigma
        )
        fields = []
        for k, partial in enumerate(self._partials):
            _check(cancel, report, 0.05 + 0.45 * k / len(self._partials), f"optical flow {k}")
            current = None if self._sequence is None else self._sequence.fields[k]
            base = self._reference if self._flow_reference is None else self._flow_reference
            moving = (
                base if current is None else warp_volume(base, current, order=cfg.warp_order)
            )
            increment = estimate_flow(
                partial,
                moving,
                spacing=cfg.grid_spacing,
                method=cfg.flow_method,
                alpha=cfg.flow_alpha,
                levels=cfg.flow_levels,
                iterations=cfg.flow_iterations,
                warps=cfg.flow_warps,
                prefilter_sigma=prefilter,
            )
            fields.append(increment if current is None else compose(increment, current))

        # 2. Regularise: smooth in time, fix the gauge, cap the magnitude. Order
        #    matters -- clipping last means the cap is the last word, and zeroing the
        #    mean before clipping stops a large common mode from eating the budget.
        sequence = DeformationSequence(tuple(fields), self._times)
        sequence = sequence.smoothed_in_time(cfg.time_sigma)
        if self._flow_reference is None:
            # With an averaged reference the frame drifts, so the unobservable common
            # mode has to be removed explicitly. With a central reference the gauge is
            # already fixed -- zeroing the mean there would move the fields out of the
            # frame the flow was estimated in and break the next iteration's refinement.
            sequence = sequence.zero_mean()
        sequence = sequence.clipped(cfg.max_dvf_px)

        # 3. Rebuild the reference by carrying each partial back into the common frame.
        #    Into a local, not onto self: like AlignmentEngine.step, an iteration
        #    abandoned part-way must record nothing, so nothing is published until
        #    every cancellable sub-step has passed.
        _check(cancel, report, 0.55, "rebuilding the reference volume")
        reference = self._back_warped_mean(sequence)

        # 4. Data consistency, fitted and held out.
        residual, holdout_residual = self._data_residual(
            reference, sequence, cancel, report, 0.6, 0.98
        )
        baseline, baseline_holdout = self._baseline

        result = NonRigidResult(
            iteration=self.iteration + 1,
            sequence=sequence,
            residual=residual,
            holdout_residual=holdout_residual,
            baseline_residual=baseline,
            baseline_holdout_residual=baseline_holdout,
            dvf_rms_px=sequence.rms_magnitude,
            dvf_max_px=sequence.max_magnitude,
            wallclock_s=time.perf_counter() - started,
            volume=reference,
        )
        result.overfitting = self._overfitting_reason(result)
        if result.overfitting:
            logger.warning("Iteration %d looks like OVERFITTING. %s", result.iteration,
                           result.overfitting)
        logger.info(
            "nonrigid iter %d: DVF %.2f px rms / %.2f px max, residual %.4f (rigid %.4f), "
            "held-out %.4f (rigid %.4f), %.1f s",
            result.iteration, result.dvf_rms_px, result.dvf_max_px, residual, baseline,
            holdout_residual, baseline_holdout, result.wallclock_s,
        )
        self._reference = reference
        self._sequence = sequence
        self._history.append(result)
        self.policy.apply(self._history)
        return result

    def run(
        self,
        n: int,
        cancel_event: Any | None = None,
        callback: Callable[[NonRigidResult], None] | None = None,
    ) -> list[NonRigidResult]:
        """Call :meth:`step` ``n`` times, stopping early if it starts overfitting.

        Stops rather than leaving it to the caller, for the same reason
        :meth:`AlignmentEngine.run` stops on a runaway shift: continuing makes the
        model fit the noise harder, and a caller who only inspects the last result
        would find a confident, wrong deformation field.
        """
        from tktomo.ptycho_align.core.engine import Cancelled  # noqa: PLC0415

        results: list[NonRigidResult] = []
        for _ in range(n):
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                result = self.step(cancel=cancel_event)
            except Cancelled:
                logger.info("non-rigid run cancelled after %d complete", len(results))
                break
            results.append(result)
            if callback is not None:
                callback(result)
            if result.overfitting:
                logger.warning(
                    "Stopping the non-rigid run at iteration %d: %s Use revert_to(%d) to "
                    "go back to the last iteration that was still earning its parameters.",
                    result.iteration, result.overfitting, max(1, result.iteration - 1),
                )
                break
        return results

    # -- internals --------------------------------------------------------------------

    def _split_holdout(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        if not 0.0 <= cfg.holdout_fraction < 0.5:
            raise ValueError(
                f"holdout_fraction must be in [0, 0.5), got {cfg.holdout_fraction}"
            )
        count = int(round(cfg.holdout_fraction * n))
        if cfg.holdout_fraction > 0 and count < 2:
            raise ValueError(
                f"holdout_fraction={cfg.holdout_fraction} leaves {count} held-out "
                f"projection(s) out of {n}; that cannot measure anything. Raise the "
                "fraction or disable the check by setting it to 0 -- and then say in "
                "the write-up that the deformation was never validated out of sample."
            )
        if count == 0:
            logger.warning(
                "holdout_fraction=0: the non-rigid fit will NOT be validated on held-out "
                "projections. The fitted residual always improves; it is not evidence."
            )
            return np.zeros(0, dtype=int), np.arange(n)
        rng = np.random.default_rng(cfg.holdout_seed)
        holdout = np.sort(rng.choice(n, size=count, replace=False))
        fitted = np.setdiff1d(np.arange(n), holdout)
        return holdout, fitted

    def _build_subsets(self) -> list[np.ndarray]:
        subsets = time_subsets(
            self.acquisition_index[self._fitted], self.config.n_subsets, self.config.subset_mode
        )
        return [self._fitted[s] for s in subsets]

    def _check_angular_coverage(self) -> None:
        gaps = [angular_gap_deg(self.angles[s]) for s in self._subsets]
        worst = max(gaps)
        if worst <= self.config.max_angular_gap_deg:
            return
        message = (
            f"subset {int(np.argmax(gaps))} of {len(self._subsets)} has a {worst:.0f} deg "
            f"angular gap (limit {self.config.max_angular_gap_deg:g} deg), i.e. a missing "
            "wedge. Its partial reconstruction is elongated along the missing direction, "
            "and the optical flow will measure that elongation and call it deformation. "
            "Either the scan interleaves angles (use subset_mode='time_blocks' with fewer "
            "subsets) or it is a single sequential sweep, in which case a time-varying "
            "deformation is simply not identifiable from this data."
        )
        if self.config.require_angular_coverage:
            raise RigidAlignmentRequired(message)
        logger.warning("Angular coverage check DISABLED: %s", message)

    def _get_backend(self) -> Any:
        if self.backend is not None:
            return self.backend
        from tktomo.recon import get_backend  # noqa: PLC0415

        return get_backend(self.config.backend)

    def _reconstruct(self, indices: np.ndarray, cancel: Any, report: Any, base: float) -> np.ndarray:
        """Reconstruct from a subset of projections, in interruptible row chunks.

        Chunking, kwargs and the degrade-on-TypeError behaviour are the engine's; the
        chunk size comes from ``engine.row_chunk_size`` so both stages make the same
        wallclock/cancellation trade on the same machine.
        """
        from tktomo.ptycho_align.core.engine import (  # noqa: PLC0415
            DIRECT_ALGORITHMS,
            row_chunk_size,
        )

        cfg = self.config
        backend = self._get_backend()
        prj = np.ascontiguousarray(self.projections[indices])
        angles = self.angles[indices]
        rows = prj.shape[1]
        chunk = row_chunk_size(rows, cfg.ncore, cfg.row_chunk)
        direct = cfg.recon_algorithm in DIRECT_ALGORITHMS

        slabs = []
        for r0 in range(0, rows, chunk):
            r1 = min(r0 + chunk, rows)
            _check(cancel, report, base, f"reconstructing rows {r0}-{r1}")
            kwargs: dict[str, Any] = {
                "algorithm": cfg.recon_algorithm,
                "center": self.center,
            }
            if cfg.ncore is not None:
                kwargs["ncore"] = cfg.ncore
            if not direct:
                kwargs["num_iter"] = cfg.recon_inner_iters
            try:
                slabs.append(backend.reconstruct(prj[:, r0:r1, :], angles, **kwargs))
            except (TypeError, ValueError):
                if direct:
                    raise
                slabs.append(
                    backend.reconstruct(
                        prj[:, r0:r1, :],
                        angles,
                        algorithm=cfg.recon_algorithm,
                        center=self.center,
                    )
                )
        return np.ascontiguousarray(np.concatenate(slabs, axis=0), dtype=np.float32)

    def _prepare(self, cancel: Any, report: Any) -> None:
        """Reconstruct the partials and the rigid-only reference, once.

        The partials are built from the pristine (rigidly aligned) projections and are
        never rebuilt: they are the *data*, in volume form. Rebuilding them each
        iteration would cost K reconstructions per iteration and change nothing.
        """
        _check(cancel, report, 0.0, "reconstructing the reference volume")
        self._reference = self._reconstruct(self._fitted, cancel, report, 0.0)
        nz, nu = self.projections.shape[1], self.projections.shape[2]
        grid = DeformationField.grid_for((nz, nu, nu), self.config.grid_spacing)
        parameters = 3 * int(np.prod(grid)) * len(self._subsets)
        voxels = nz * nu * nu
        logger.info(
            "deformation model: %s nodes x 3 components x %d subsets = %d free parameters "
            "against %d voxels (1 per %.0f voxels)",
            grid, len(self._subsets), parameters, voxels, voxels / max(parameters, 1),
        )
        if parameters > 0.01 * voxels:
            logger.warning(
                "the deformation model has %d free parameters for %d voxels (%.1f%%). That "
                "is enough freedom to absorb genuine structure. Increase grid_spacing "
                "until there are of order 8-20 nodes per axis, and check the held-out "
                "residual carefully.",
                parameters, voxels, 100 * parameters / voxels,
            )
        gigabytes = (len(self._subsets) + 1) * nz * nu * nu * 4 / 2**30
        if gigabytes > 8.0:
            logger.warning(
                "the %d partial volumes plus the reference need about %.0f GB of RAM. "
                "Reduce n_subsets, or bin the projections first -- the deformation is "
                "smooth by construction and does not need full resolution to estimate.",
                len(self._subsets), gigabytes,
            )
        partials = []
        for k, subset in enumerate(self._subsets):
            _check(cancel, report, 0.01 * k, f"reconstructing subset {k}")
            partials.append(self._reconstruct(subset, cancel, report, 0.0))
        self._partials = partials
        if self.config.reference_mode == "central":
            # The temporally central subset: the state the sample was in halfway
            # through the scan. Every field is then referred to that instant.
            self._flow_reference = partials[len(partials) // 2]
        elif self.config.reference_mode == "average":
            self._flow_reference = None  # follows the rebuilt volume reference
        else:
            raise ValueError(
                f"unknown reference_mode {self.config.reference_mode!r}; "
                "expected 'central' or 'average'"
            )
        # The rigid-only baseline: the same volume, projected with NO deformation.
        # Everything this stage claims is measured against these two numbers.
        self._baseline = self._data_residual(self._reference, None, cancel, report, 0.0, 0.05)
        logger.info(
            "non-rigid baseline (no deformation): residual %.4f fitted, %.4f held out "
            "(%d subsets, %d held-out projections)",
            self._baseline[0], self._baseline[1], len(self._subsets), self._holdout.size,
        )

    def _back_warped_mean(self, sequence: DeformationSequence) -> np.ndarray:
        """Average the partials after carrying each back into the common frame.

        ``warp_volume(reference, u_k) ~= partial_k``, so ``warp_volume(partial_k,
        invert(u_k)) ~= reference``: every partial contributes its own angular sampling
        to one deformation-corrected volume. Each partial is warped **once**, from its
        own pristine reconstruction -- never a warp of a warp (deformation.py,
        convention 3).
        """
        total = None
        for k, partial in enumerate(self._partials or []):
            try:
                inverse = invert(sequence.fields[k])
            except ValueError as exc:
                raise ValueError(
                    f"the deformation field for subset {k} could not be inverted: {exc}"
                ) from exc
            warped = warp_volume(partial, inverse, order=self.config.warp_order)
            total = warped if total is None else total + warped
        if total is None:
            raise ValueError("no partial reconstructions to average")
        return (total / len(self._partials or [1])).astype(np.float32)

    def simulate(
        self, volume: np.ndarray | None = None, sequence: DeformationSequence | None = None
    ) -> np.ndarray:
        """The full warp-then-project forward model, materialised.

        Allocates a copy of the whole projection stack, which for a real dataset is
        gigabytes -- :meth:`step` therefore never calls this, it accumulates the
        residual bin by bin instead. Use it when you actually want the maps, e.g. to
        feed :func:`residual_localisation` with ``measured - simulated``.
        """
        volume = self._reference if volume is None else volume
        if volume is None:
            raise ValueError("no volume yet -- run at least one iteration")
        sequence = self._sequence if sequence is None else sequence
        simulated = np.empty_like(self.projections)
        for indices, block in self._forward(volume, sequence, None, None, 0.0, 0.0):
            simulated[indices] = block
        return simulated

    def _forward(
        self,
        volume: np.ndarray,
        sequence: DeformationSequence | None,
        cancel: Any,
        report: Any,
        base: float,
        span: float,
    ):
        """Yield ``(indices, simulated_block)`` one time bin at a time.

        The forward model is warp-then-project, evaluated in time bins: within a bin
        the deformation is treated as constant, so the volume is warped **once** and
        all the bin's angles are projected from it. ``n_time_bins`` is purely a cost
        knob -- one warp per projection would be ideal and is unaffordable at real
        volume sizes. This is the only place the non-rigid model touches the geometry;
        everything else is the backend's, unchanged.
        """
        backend = self._get_backend()
        bins = self._time_bins()
        for i, indices in enumerate(bins):
            _check(cancel, report, base + span * i / max(1, len(bins)), f"reprojecting bin {i}")
            if sequence is None:
                warped = volume
            else:
                mean_time = float(np.mean(self.acquisition_index[indices]))
                warped = warp_volume(
                    volume, sequence.at(mean_time), order=self.config.warp_order
                )
            kwargs: dict[str, Any] = {"center": self.center}
            if self.config.ncore is not None:
                kwargs["ncore"] = self.config.ncore
            block = np.asarray(
                backend.reproject(warped, self.angles[indices], **kwargs), dtype=np.float32
            )
            expected = (indices.size, *self.projections.shape[1:])
            if block.shape != expected:
                raise ValueError(
                    f"the backend returned reprojections of shape {block.shape}, expected "
                    f"{expected}. A silent shape mismatch here would corrupt every "
                    "residual, so this is fatal."
                )
            yield indices, block

    def _data_residual(
        self,
        volume: np.ndarray,
        sequence: DeformationSequence | None,
        cancel: Any,
        report: Any,
        base: float,
        span: float,
    ) -> tuple[float, float]:
        """||measured - reprojected|| / ||measured||, on the fitted and held-out sets.

        Accumulated bin by bin rather than by building the whole simulated stack: at
        907 x 1488 x 1816 float32 that stack is 9.8 GB, and the residual only needs two
        running sums.
        """
        in_holdout = np.zeros(self.projections.shape[0], dtype=bool)
        in_holdout[self._holdout] = True
        sums = np.zeros(2)  # [fitted, held out]
        norms = np.zeros(2)
        for indices, block in self._forward(volume, sequence, cancel, report, base, span):
            measured = self.projections[indices]
            difference = (measured - block).astype(np.float64)
            measured = measured.astype(np.float64)
            held = in_holdout[indices]
            for slot, selection in enumerate((~held, held)):
                if not selection.any():
                    continue
                sums[slot] += float((difference[selection] ** 2).sum())
                norms[slot] += float((measured[selection] ** 2).sum())

        def score(slot: int) -> float:
            if norms[slot] <= 0:
                return float("nan")
            return float(np.sqrt(sums[slot] / norms[slot]))

        return score(0), score(1)

    def _time_bins(self) -> list[np.ndarray]:
        count = self.config.n_time_bins or 2 * self.config.n_subsets
        count = int(max(1, min(count, self.projections.shape[0])))
        order = np.argsort(self.acquisition_index, kind="stable")
        return [np.sort(part) for part in np.array_split(order, count) if part.size]

    def _overfitting_reason(self, result: NonRigidResult) -> str | None:
        """Is this iteration measuring deformation, or inventing it?

        Three tests, in increasing subtlety:

        * The **fitted** residual must improve at all. A deformation field has far more
          freedom than the rigid solution, so if it cannot beat that solution even on
          the projections it was handed, there is nothing here to describe. This is the
          test that catches rigid data: the field then wanders, driven by the streak
          differences between subsets, and because :meth:`step` *refines* the field
          rather than re-estimating it, that wander accumulates -- 0.19 px rms after one
          iteration and 0.60 px after six, measured on an undeformed phantom. Stopping
          at the first iteration that buys nothing is what keeps it at the first number.
        * The held-out residual must keep improving **from one iteration to the next**.
          This is the test that bounds the refinement: on rigid data the held-out gain
          peaks at the first iteration and falls away from the second while the field
          keeps growing, so the run stops with a 0.25 px artefact instead of a 0.86 px
          one.
        * The **held-out** residual must not get worse than the rigid-only baseline.
        * The held-out residual must improve by at least ``min_holdout_share`` of the
          share the fitted residual improved by. A model that improves what it fitted
          by 20% and what it did not by 1% has learnt the noise in the fitted
          projections, not a property of the sample.

        This is why :meth:`run`, not :meth:`step`, is the entry point for an unattended
        run: it stops on the first flagged iteration.
        """
        cfg = self.config
        if np.isfinite(result.fitted_gain) and result.fitted_gain < cfg.min_fitted_gain:
            return (
                f"the deformation model does not improve even the projections it was "
                f"fitted to ({result.fitted_gain:+.1%} against the rigid-only baseline, "
                f"needs {cfg.min_fitted_gain:+.1%}). More parameters always fit better, so "
                "a model that cannot beat the rigid solution on its own training data is "
                "describing nothing -- there is no deformation here to find, and the "
                f"{result.dvf_rms_px:.2f} px rms field it produced is an artefact of the "
                "differing streaks between subset reconstructions."
            )
        previous = self._history[-1] if self._history else None
        if (
            previous is not None
            and np.isfinite(result.holdout_gain)
            and np.isfinite(previous.holdout_gain)
            and result.holdout_gain < previous.holdout_gain - cfg.holdout_stall_tolerance
        ):
            return (
                f"this iteration predicts the held-out projections *worse* than the last "
                f"one did ({result.holdout_gain:+.2%} against {previous.holdout_gain:+.2%}) "
                f"while the field grew from {previous.dvf_rms_px:.2f} to "
                f"{result.dvf_rms_px:.2f} px rms. The refinement has stopped extracting "
                "deformation and started accumulating whatever differs between the subset "
                f"reconstructions. Revert to iteration {previous.iteration}."
            )
        if self._holdout.size == 0 or not np.isfinite(result.holdout_residual):
            return (
                "no held-out projections, so overfitting cannot be detected at all "
                "(holdout_fraction=0)"
            )
        if result.holdout_residual > result.baseline_holdout_residual * (1 + cfg.overfit_tolerance):
            return (
                f"the held-out residual got WORSE: {result.holdout_residual:.4f} against a "
                f"rigid-only {result.baseline_holdout_residual:.4f}. The deformation field "
                "is fitting the projections it was given and mispredicting the ones it was "
                "not, which is the definition of overfitting here."
            )
        fitted_gain, holdout_gain = result.fitted_gain, result.holdout_gain
        if fitted_gain > 0.01 and holdout_gain < cfg.min_holdout_share * fitted_gain:
            return (
                f"the fitted residual improved by {fitted_gain:.1%} but the held-out one by "
                f"only {holdout_gain:.1%} (needs {cfg.min_holdout_share:.0%} of it). The "
                "model is absorbing what it was fitted to rather than describing the "
                "sample; coarsen grid_spacing, raise time_sigma, or lower max_dvf_px."
            )
        return None


def _check(cancel: Any, report: Any, fraction: float, message: str) -> None:
    """Honour a pending cancel and report progress. Mirrors ``engine._Progress.check``."""
    if cancel is not None and cancel.is_set():
        from tktomo.ptycho_align.core.engine import Cancelled  # noqa: PLC0415

        raise Cancelled(message)
    if report is not None:
        report(float(fraction), message)

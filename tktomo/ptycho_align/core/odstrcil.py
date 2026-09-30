"""The decoupled two-stage aligner: vertical by mass, horizontal by gradient.

This is the ptycho-tomography reference method (M. Odstrcil et al., *Opt. Express*
**27**, 36637, 2019), assembled from the two modules beside it. It exists because the
two transverse directions are *not* the same problem and should not be solved by the
same machinery:

**Stage 1 -- vertical.** The rotation axis is vertical, so rotating the sample moves
every voxel within its own detector row and the vertical mass distribution of a
projection is invariant under angle. Aligning it is a 1-D registration with a unique
answer and needs no forward projection, no back projection and no volume. See
:mod:`~tktomo.ptycho_align.core.vertical`. It is essentially free, so it runs first and
runs to convergence before anything expensive starts.

**Stage 2 -- horizontal.** No such luck. A horizontal shift is entangled with the
rotation angle: the horizontal mass distribution is *supposed* to change from view to
view, and that change is the tomographic signal, so no pairwise comparison of two
projections can separate "the sample moved sideways" from "the sample turned". The
only reference that resolves it is the object itself, so the horizontal shift has to be
solved against a reconstructed volume: reconstruct, reproject, compare each measured
projection with its own reprojection, update that angle's shift, repeat. This is why
centre-of-mass pre-alignment is defensible vertically and structurally wrong
horizontally, and why the horizontal stage costs a full reconstruction per iteration
while the vertical stage costs a row-sum.

**The comparison is made on the phase gradient, not the phase.** Ptychographic phase
retrieval leaves a constant offset and a linear ramp undetermined per projection, and a
linear ramp is mathematically identical to a lateral shift -- exactly the quantity being
estimated. Differentiating annihilates both. See
:mod:`~tktomo.ptycho_align.core.gradient`, which measures the difference: a 10 rad ramp
across the frame moves the value-domain estimate by 25.7 px and the gradient-domain
estimate by 4e-16 px. This single substitution is what separates this loop from the
generic Gursoy reprojection loop in :mod:`~tktomo.ptycho_align.core.engine`.

Order of operations, from the roadmap, never to be violated:
ramp/offset removal (:func:`~tktomo.ptycho_align.core.preprocess.remove_phase_ramp`)
-> rotation centre -> **vertical** -> **horizontal** -> non-rigid. This module owns the
middle two and assumes the first two have been done.

The seam with :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine`
---------------------------------------------------------------------
:class:`OdstrcilEngine` **subclasses** ``AlignmentEngine`` and overrides exactly one
thing: :meth:`~OdstrcilEngine.step`, which replaces step 5 of the parent's iteration
(the ``phase_cross_correlation`` on the phase) with the gradient-domain estimator, and
freezes the vertical axis because stage 1 already solved it. Everything else is the
parent's and is called, not copied: ``_reconstruct`` (with its row chunking, warm start,
cancellation points and progress reporting), ``_reproject``, ``apply_shifts``,
``_condition_update`` (damping, clipping, median filtering), ``_is_diverging``,
``shift_update_is_runaway``, ``AlignmentState.record`` with its volume-memory policy,
and the inherited ``run``, ``revert_to`` and history. A GUI or session host that can
drive an ``AlignmentEngine`` can drive this without knowing it exists.

The clean design would be a registration *estimator* hook on ``AlignmentEngine`` that
both loops configure; ``step`` is currently monolithic, so overriding it is the least
invasive way in. If that hook is ever added, this class should shrink to supplying an
estimator plus the stage-1 pre-pass, and nothing here should need to change otherwise.
The parent's three conventions (registration direction, ``(dy, dx)`` axis order, never
re-shift shifted data) are followed exactly -- read ``engine.py``'s module docstring.

What it achieves, measured
--------------------------
On a synthetic phantom with injected known shifts (``tests/test_odstrcil.py``, 48 views,
64 px detector, SIRT, 12 outer iterations), RMS recovery error against truth, after
removing the physically unobservable modes:

    vertical   (stage 1)   0.015 px
    horizontal (stage 2)   0.167 px
    target                 0.333 px  (1/3 of a voxel)

The vertical number is an order of magnitude better than the horizontal one, and that
gap is the decoupling working: stage 1 solves a well-posed 1-D problem against the data
itself, while stage 2 can only ever be as good as the volume it compares against. With a
4 rad residual phase ramp on top, the horizontal figure moves to 0.263 px and still
meets the target, where the same loop registering on the phase *value* goes to 0.548 px
and misses it.

Two caveats worth carrying into any write-up. First, the accuracy target is *residual
alignment error*, and the honest way to confirm it on real data is split-data FSC paired
with reprojection-residual maps -- FSC alone cannot see a systematic geometric bias,
because a rigid-but-wrong geometry applied identically to both half-sets gives a
deceptively good curve. Second, "vertical is the easy direction" is a statement about
how well-posed the *problem* is, not a promise that vertical shifts will be small: real
scans have been measured with vertical drift several times larger than horizontal, and
with worse vertical than horizontal resolution in the aligned result. Easy to solve is
not the same as already solved.

What this module refuses to do
------------------------------
The roadmap documents two reconstruction choices as failure modes, and they are
encoded here rather than left in a README: **GridRec fails outright with a limited
angular range** (it interpolates onto a Cartesian Fourier grid, so an unsampled wedge
gets filled with interpolation noise instead of being left empty), and **initialising
from FBP converges more slowly**, because FBP's streak artifacts are reprojected into
the simulated stack and the registration then measures the streaks along with the
sample. :func:`check_reconstruction_choice` raises on both by default. SIRT (or MLEM on
non-negative data) is the default and what the loop is tuned for.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from tktomo.ptycho_align.core.engine import (
    DIRECT_ALGORITHMS,
    AlignConfig,
    AlignmentEngine,
    algorithm_rejects_negatives,
    apply_shifts,
    shift_update_is_runaway,
)

# Package-internal helpers, deliberately reused rather than re-implemented: _Progress is
# the parent's cancel/progress carrier (so a Stop lands between row chunks exactly as it
# does in the parent loop), and _match_shape is its reprojection shape guard.
from tktomo.ptycho_align.core.engine import _match_shape, _Progress
from tktomo.ptycho_align.core.gradient import GradientConfig, register_stack
from tktomo.ptycho_align.core.state import IterationResult
from tktomo.ptycho_align.core.vertical import VerticalConfig, VerticalResult, align_vertical

logger = logging.getLogger(__name__)

__all__ = [
    "AngularCoverage",
    "LimitedAngleError",
    "OdstrcilConfig",
    "OdstrcilEngine",
    "SeriesAligner",
    "SeriesAlignerEntry",
    "angular_coverage",
    "available_series_aligners",
    "check_reconstruction_choice",
    "default_odstrcil_config",
    "get_series_aligner",
    "register_series_aligner",
]


class LimitedAngleError(ValueError):
    """A reconstruction algorithm was chosen that cannot cope with this angular range."""


# -- angular coverage ---------------------------------------------------------------


@dataclass(frozen=True)
class AngularCoverage:
    """What the scan's angles actually cover. All angles in degrees.

    ``span`` is what decides whether an algorithm is usable; ``max_gap`` catches the
    other way a scan can be incomplete -- a full 180 deg range with a block of views
    missing in the middle is a missing wedge just as much as a truncated range is.
    """

    n_views: int
    span: float
    median_step: float
    max_gap: float
    n_gaps_over_1deg: int
    limited: bool
    reason: str | None  # why `limited` is True, or None

    def summary(self) -> str:
        return (
            f"{self.n_views} views over {self.span:.3f} deg "
            f"(median step {self.median_step:.3f} deg, largest gap {self.max_gap:.3f} deg, "
            f"{self.n_gaps_over_1deg} gap(s) > 1 deg)"
        )


def angular_coverage(
    angles: np.ndarray,
    *,
    min_span_deg: float = 170.0,
    max_gap_deg: float = 10.0,
) -> AngularCoverage:
    """Summarise the angular sampling of a scan. ``angles`` are radians, TomoPy style.

    Parallel-beam tomography needs 180 deg; ``min_span_deg`` is set a little below that
    so an ordinary ``linspace(0, pi, n, endpoint=False)`` scan (which spans
    ``180*(n-1)/n``) is not flagged. ``max_gap_deg`` catches a missing block of views
    within an otherwise full range.
    """
    theta = np.sort(np.rad2deg(np.asarray(angles, dtype=np.float64)))
    n = theta.size
    if n < 2:
        return AngularCoverage(
            n_views=n,
            span=0.0,
            median_step=0.0,
            max_gap=0.0,
            n_gaps_over_1deg=0,
            limited=True,
            reason=f"{n} view(s) is not a tomographic scan",
        )

    steps = np.diff(theta)
    span = float(theta[-1] - theta[0])
    median_step = float(np.median(steps))
    max_gap = float(np.max(steps))
    n_big = int((steps > 1.0).sum())

    reasons = []
    if span < min_span_deg:
        reasons.append(
            f"the angular span is {span:.2f} deg, below the {min_span_deg:.0f} deg a "
            "parallel-beam reconstruction needs"
        )
    if max_gap > max_gap_deg:
        reasons.append(
            f"the largest gap between consecutive views is {max_gap:.2f} deg, above the "
            f"{max_gap_deg:.0f} deg limit -- there is a missing wedge"
        )

    return AngularCoverage(
        n_views=n,
        span=span,
        median_step=median_step,
        max_gap=max_gap,
        n_gaps_over_1deg=n_big,
        limited=bool(reasons),
        reason=" and ".join(reasons) if reasons else None,
    )


# -- what we refuse to reconstruct with ---------------------------------------------

_FBP_REASON = (
    "initialising the alignment loop from a direct reconstruction is a documented "
    "failure mode: FBP/GridRec streak artifacts are reprojected into the simulated "
    "stack, so the registration measures the streaks as well as the sample and every "
    "shift update carries an artifact-driven component. The loop still converges, but "
    "more slowly than from a plain zero start, and the early iterations are the ones "
    "that decide whether it converges at all. Direct algorithms also accept no "
    "'init_recon', so mode='joint' silently degrades to 'sequential'."
)

_GRIDREC_LIMITED_REASON = (
    "GridRec resamples the measured projections onto a Cartesian grid in Fourier space. "
    "With an incomplete angular range the unsampled wedge is not left empty -- it is "
    "filled by interpolation from the sampled region -- so the reconstruction does not "
    "degrade gracefully, it fails outright, and reprojecting it produces a simulated "
    "stack the registration cannot lock onto."
)


def check_reconstruction_choice(
    algorithm: str,
    coverage: AngularCoverage,
    *,
    policy: Literal["refuse", "warn", "allow"] = "refuse",
) -> str | None:
    """Vet the reconstruction algorithm against the angular coverage.

    Returns a reason string when the choice is questionable (``None`` when it is fine),
    and raises :class:`LimitedAngleError` when it is one this loop will not run:

    * ``policy="refuse"`` (default) -- any direct algorithm (``gridrec``, ``fbp``) is
      refused, because neither is a sensible thing to drive an alignment loop with.
    * ``policy="warn"`` -- direct algorithms are allowed with a loud warning, *except*
      GridRec on a limited angular range, which is still refused. That combination does
      not converge slowly, it produces garbage.
    * ``policy="allow"`` -- nothing is refused. The warning is still logged, because the
      log is the record of what the run was actually told to do.
    """
    if policy not in ("refuse", "warn", "allow"):
        raise ValueError(f"policy must be 'refuse', 'warn' or 'allow', got {policy!r}")

    algorithm = str(algorithm).lower()
    if algorithm not in DIRECT_ALGORITHMS:
        return None

    fatal = algorithm == "gridrec" and coverage.limited
    if fatal:
        reason = (
            f"'gridrec' cannot be used here: {coverage.reason}. {_GRIDREC_LIMITED_REASON} "
            f"Scan: {coverage.summary()}. Use 'sirt'."
        )
    else:
        reason = f"'{algorithm}' is a direct algorithm and {_FBP_REASON} Use 'sirt'."

    if policy == "refuse" or (fatal and policy == "warn"):
        raise LimitedAngleError(
            reason
            + " Set OdstrcilConfig.direct_recon_policy='allow' if you intend this anyway."
        )
    logger.warning("%s", reason)
    return reason


# -- configuration ------------------------------------------------------------------


@dataclass
class OdstrcilConfig:
    """The two-stage aligner's own settings. Reconstruction lives in :class:`AlignConfig`."""

    vertical: VerticalConfig = field(default_factory=VerticalConfig)
    #: Registration domain for stage 2. The default is the point of the whole module;
    #: ``domain="value"`` turns it back into the engine's (ramp-sensitive) behaviour and
    #: exists so the two can be compared on identical footing.
    gradient: GradientConfig = field(default_factory=GradientConfig)
    #: Run stage 1 at all. Off means "the vertical shifts I passed in are final".
    run_vertical: bool = True
    #: Re-run stage 1 every outer iteration. It costs a row-sum, so this is cheap, but
    #: it is off by default: the roadmap's order of operations solves vertical first and
    #: then leaves it alone, and re-solving it from data that stage 2 has since warped
    #: horizontally buys nothing (a horizontal shift does not change a row-sum).
    refine_vertical_each_iteration: bool = False
    #: Also take the vertical component of the stage-2 gradient registration. Off by
    #: default: stage 1's answer comes from a measurement that involves no reconstruction
    #: and so cannot be contaminated by reconstruction artifacts, which is strictly
    #: better. The stage-2 ``dy`` is still computed and reported as a *cross-check* --
    #: if it is not small, one of the two stages is wrong.
    use_gradient_vertical: bool = False
    direct_recon_policy: Literal["refuse", "warn", "allow"] = "refuse"
    min_angular_span_deg: float = 170.0
    max_angular_gap_deg: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form, matching :meth:`AlignConfig.to_dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OdstrcilConfig":
        """Rebuild from :meth:`to_dict` output that has been through JSON or msgpack.

        The nested :class:`VerticalConfig` and :class:`GradientConfig` come back as plain
        dicts and have to be reconstructed, or the engine ends up holding dicts where it
        expects dataclasses and fails somewhere unrelated. Unknown fields are rejected
        rather than dropped, for the same reason ``AlignConfig.from_dict`` rejects them:
        a config written by a newer tktomo must fail loudly, not silently lose settings.
        """
        nested = {"vertical": VerticalConfig, "gradient": GradientConfig}
        values = dict(raw)
        for name, factory in nested.items():
            value = values.get(name)
            if isinstance(value, Mapping):
                known = {f.name for f in fields(factory)}
                unknown = sorted(set(value) - known)
                if unknown:
                    raise ValueError(
                        f"Unknown {factory.__name__} field(s): {', '.join(unknown)}."
                    )
                values[name] = factory(**value)

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown OdstrcilConfig field(s): {', '.join(unknown)}. The config was "
                "probably written by a newer version of tktomo."
            )
        return cls(**values)


# -- the engine ---------------------------------------------------------------------


@dataclass
class OdstrcilEngine(AlignmentEngine):
    """Decoupled two-stage alignment, one outer iteration per :meth:`step`.

    Drop-in for :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine`: same
    constructor arguments plus ``odstrcil`` (and ``projector``, a test/backend seam),
    same ``step``/``run``/``history``/``state`` surface, same
    :class:`~tktomo.ptycho_align.core.state.IterationResult`.

    ``sy0`` is *ignored* when ``odstrcil.run_vertical`` is True, because stage 1 solves
    the vertical axis outright rather than refining a guess. Pass
    ``run_vertical=False`` to supply your own vertical shifts.
    """

    odstrcil: OdstrcilConfig = field(default_factory=OdstrcilConfig)
    #: Reconstruction backend override. ``None`` uses ``AlignConfig.backend`` through
    #: :func:`tktomo.recon.get_backend`. Anything with ``reconstruct``/``reproject``
    #: works, which is how the tests drive the loop without tomopy installed.
    projector: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        self.coverage = angular_coverage(
            self.state.angles,
            min_span_deg=self.odstrcil.min_angular_span_deg,
            max_gap_deg=self.odstrcil.max_angular_gap_deg,
        )
        # Refuse a hopeless reconstruction choice at construction, not five minutes into
        # the first iteration.
        self.recon_warning = check_reconstruction_choice(
            self.config.recon_algorithm,
            self.coverage,
            policy=self.odstrcil.direct_recon_policy,
        )
        reason = algorithm_rejects_negatives(self.config.recon_algorithm, self.state.original)
        if reason:
            logger.warning("%s", reason)

        self.vertical: VerticalResult | None = None
        self.last_registration: list[Any] = []
        #: RMS of the stage-2 gradient registration's *vertical* component. A
        #: cross-check on stage 1, not an update (unless ``use_gradient_vertical``).
        self.vertical_cross_check: float = float("nan")

        if self.odstrcil.run_vertical and self.sy0 is not None and np.any(self.sy0):
            logger.info(
                "sy0 was supplied but stage 1 solves the vertical axis from the data; "
                "it will be replaced. Set OdstrcilConfig.run_vertical=False to keep it."
            )

    # -- stage 1 --------------------------------------------------------------------

    def align_vertical_stage(self) -> VerticalResult:
        """Run (or re-run) stage 1 and write its answer into the state.

        Public because it is worth running and looking at on its own: it is fast enough
        to be interactive, and its convergence history and truncation report are the
        first honest read on whether the scan's geometry is what you think it is. Uses
        no reconstruction backend at all -- it will run happily with ``projector=None``
        and no tomopy installed.
        """
        result = align_vertical(self.state.original, self.odstrcil.vertical)
        self.vertical = result
        self.state.sy = result.sy.copy()
        self.invalidate_cache()
        return result

    # -- stage 2 --------------------------------------------------------------------

    def _backend(self):
        if self.projector is not None:
            return self.projector
        return super()._backend()

    def step(
        self,
        cancel: Any | None = None,
        report: Callable[[float, str], None] | None = None,
    ) -> IterationResult:
        """Run exactly ONE outer iteration of the decoupled loop.

        The first call also runs stage 1 (vertical), which happens before any
        reconstruction and costs a row-sum. Subsequent calls are pure stage 2.
        Cancelling behaves exactly as in the parent: :class:`Cancelled` is raised from a
        row-chunk boundary, and nothing is recorded.
        """
        cfg = self.config
        ocfg = self.odstrcil
        started = time.perf_counter()
        ctx = _Progress(cancel, report)

        # -- Stage 1: vertical, from the mass distribution. No projection of any kind.
        needs_vertical = (
            ocfg.run_vertical
            and cfg.align_vertical
            and (self.vertical is None or ocfg.refine_vertical_each_iteration)
        )
        if needs_vertical:
            ctx.phase(recon=True).check(0.0, "stage 1: vertical mass distribution")
            vertical = self.align_vertical_stage()
            logger.info(
                "stage 1: %d iteration(s), %s, RMS sy %.3f px%s",
                vertical.n_iterations,
                "converged" if vertical.converged else "NOT converged",
                vertical.rms_shift,
                "" if vertical.truncation_reason is None else " [TRUNCATION WARNING]",
            )

        # -- Stage 2: horizontal, by tomographic consistency on the gradient.
        # Convention 3: always rebuild the aligned stack from the pristine original.
        prj_aligned = apply_shifts(self.state.original.copy(), self.state.sy, self.state.sx)

        volume = self._reconstruct(prj_aligned, ctx.phase(recon=True))
        sim = _match_shape(self._reproject(volume, ctx.phase(recon=False)), prj_aligned.shape)

        # Convention 1 and 2: reference = measured, and the result is (dy, dx).
        dsy_raw, dsx_raw, results = register_stack(prj_aligned, sim, ocfg.gradient)
        self.last_registration = results
        self.vertical_cross_check = float(np.sqrt(np.mean(dsy_raw**2)))

        dsy_in = dsy_raw if ocfg.use_gradient_vertical else np.zeros_like(dsy_raw)
        dsx, dsy = self._condition_update(dsx_raw, dsy_in)

        sx = self.state.sx + dsx
        sy = self.state.sy + dsy
        # Degenerate global modes: a constant shift in either axis is a pure translation
        # of the volume, not a misalignment. The rotation axis stays owned by `center`.
        sx = sx - sx.mean()
        sy = sy - sy.mean()

        center = self.state.center
        if cfg.refine_center:
            center = self._refine_center(prj_aligned, center)

        error = float(np.sqrt(np.mean(dsx**2 + dsy**2)))
        denominator = float(np.linalg.norm(prj_aligned))
        # np.subtract rather than `prj_aligned - sim`: numpy's temporary-elision
        # optimisation is broken on CPython 3.14 (LOAD_FAST_BORROW leaves the left
        # operand at refcount 1, so numpy believes it is a temporary and writes the
        # difference into it). `np.linalg.norm(a - b)` then silently destroys `a` --
        # which here is the aligned stack we are about to cache for the GUI. Fixed in
        # numpy >= 2.3; the explicit call is correct on every version.
        difference = np.subtract(prj_aligned, sim)
        residual = float(np.linalg.norm(difference) / denominator) if denominator else float("nan")
        runaway = shift_update_is_runaway(
            error, self.state.original.shape[2], self.com_amplitude
        )

        result = IterationResult(
            iteration=self.state.iteration + 1,
            sx=sx,
            sy=sy,
            dsx=dsx,
            dsy=dsy,
            error=error,
            residual=residual,
            volume=volume,
            center=center,
            wallclock_s=time.perf_counter() - started,
            config_changed=self._pending_config_change,
            diverging=self._is_diverging(residual),
            runaway=runaway,
        )
        if runaway:
            logger.warning("Iteration %d: RUNAWAY SHIFTS. %s", result.iteration, runaway)
        if result.diverging:
            logger.warning(
                "Iteration %d is DIVERGING: residual %.4g. With the gradient estimator a "
                "residual phase ramp is NOT a plausible cause -- look at the rotation "
                "centre, the reconstruction algorithm, and the stage-1 vertical result.",
                result.iteration,
                residual,
            )
        self._pending_config_change = False
        self._last_aligned = prj_aligned
        self._last_simulated = sim

        quality = float(np.median([r.quality for r in results])) if results else float("nan")
        logger.info(
            "iter %d: dx RMS %.4f px, residual %.4f, centre %.2f, %.1f s "
            "(%s/%s, domain=%s, peak quality %.1f, vertical cross-check %.3f px)",
            result.iteration,
            error,
            residual,
            center,
            result.wallclock_s,
            cfg.recon_algorithm,
            cfg.mode,
            ocfg.gradient.domain,
            quality,
            self.vertical_cross_check,
        )
        return self.state.record(result)

    # -- reporting ------------------------------------------------------------------

    def describe(self) -> str:
        """One-paragraph account of what this engine is set up to do, for a UI or a log."""
        lines = [
            f"Odstrcil decoupled aligner -- {self.coverage.summary()}",
            f"  stage 1 vertical: {'on' if self.odstrcil.run_vertical else 'off'} "
            f"(mass distribution, no reconstruction)",
            f"  stage 2 horizontal: {self.config.recon_algorithm} x "
            f"{self.config.recon_inner_iters} inner, {self.config.mode}, "
            f"registration domain '{self.odstrcil.gradient.domain}'"
            f"{'' if self.odstrcil.gradient.is_ramp_invariant else ' (NOT ramp-invariant)'}",
        ]
        if self.coverage.limited:
            lines.append(f"  WARNING: {self.coverage.reason}")
        if self.recon_warning:
            lines.append(f"  WARNING: {self.recon_warning}")
        if self.vertical is not None and self.vertical.truncation_reason:
            lines.append(f"  WARNING: {self.vertical.truncation_reason}")
        return "\n".join(lines)


# -- series-aligner registry ---------------------------------------------------------
#
# The repo selects interchangeable methods by name through registries, and the UIs build
# their dropdowns by enumerating them (tktomo.align.base for pairwise 2-D aligners,
# tktomo.recon.backend for reconstruction backends). Series-level tomographic alignment
# needs its own registry: `tktomo.align.base.Aligner` is a *pairwise* protocol
# (`estimate(fixed, moving) -> Transform`) and a tomographic series aligner is not a
# pairwise problem -- it consumes a whole stack plus its angles and produces one shift
# per angle. Registering an engine factory here keeps the discovery mechanism the repo
# already uses. If a canonical series registry is later added to the package root, this
# should move there unchanged; nothing but the import path would differ.


@runtime_checkable
class SeriesAligner(Protocol):
    """Factory producing an engine that aligns a whole projection series."""

    def __call__(self, dataset: Any, **kwargs: Any) -> AlignmentEngine: ...


@dataclass(frozen=True)
class SeriesAlignerEntry:
    name: str
    factory: SeriesAligner
    description: str


_SERIES_REGISTRY: dict[str, SeriesAlignerEntry] = {}


def register_series_aligner(
    name: str, factory: SeriesAligner, description: str = ""
) -> SeriesAlignerEntry:
    """Register a series-alignment engine factory under ``name``."""
    if not name:
        raise ValueError("A series aligner must have a non-empty name.")
    entry = SeriesAlignerEntry(name=name, factory=factory, description=description)
    _SERIES_REGISTRY[name] = entry
    return entry


def available_series_aligners() -> list[str]:
    """Names of registered series aligners, for a method dropdown."""
    return sorted(_SERIES_REGISTRY)


def get_series_aligner(name: str) -> SeriesAlignerEntry:
    try:
        return _SERIES_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown series aligner {name!r}. Available: {available_series_aligners()}"
        ) from None


register_series_aligner(
    "reprojection-joint",
    AlignmentEngine,
    "Gursoy et al. 2017 joint iterative reprojection. Registers the phase itself, in "
    "both axes at once. Sensitive to a residual phase ramp.",
)
register_series_aligner(
    "odstrcil-decoupled",
    OdstrcilEngine,
    "Odstrcil et al. 2019. Vertical from the vertical mass distribution (no "
    "reconstruction), then horizontal by tomographic consistency on the phase "
    "GRADIENT, which is invariant to the constant phase offset and the linear phase "
    "ramp that ptychographic phase retrieval leaves undetermined.",
)


def default_odstrcil_config(dataset: Any) -> tuple[AlignConfig, OdstrcilConfig]:
    """Sensible starting configs for a dataset: SIRT, joint, gradient-both.

    A convenience for the UI and for scripts, so the defaults live in one place rather
    than in every call site.
    """
    coverage = angular_coverage(getattr(dataset, "angles", np.zeros(2)))
    align = AlignConfig(recon_algorithm="sirt", recon_inner_iters=2, mode="joint")
    if coverage.limited:
        # A limited-angle problem needs more inner iterations to fill the wedge with
        # something the registration can lock onto than a full scan does.
        align.recon_inner_iters = 5
    return align, OdstrcilConfig()

"""Joint gradient-descent alignment: solve for the volume and the shifts together.

Where :mod:`~tktomo.ptycho_align.core.engine` alternates *reconstruct -> reproject ->
cross-correlate* (Gursoy et al. 2017), this module treats alignment as one smooth
optimisation problem and walks downhill in both unknowns at once::

    min_{v, s}  sum_i  w_i || P_{theta_i} v  -  T_{s_i} d_i ||^2

``v`` is the 3-D volume, ``s_i = (dy, dx)`` the per-projection shift, ``d_i`` the
measured phase projection, ``T`` a translation, ``P`` the parallel-beam projector and
``w_i`` a per-projection quality weight. The objective is Odstrcil et al.,
*Opt. Express* **27**, 36637 (2019); the optimiser machinery -- analytic adjoints
instead of autograd, Nesterov momentum, coarse-to-fine multi-resolution, a low-pass on
the early volume updates -- is carried over from the ASRM/"Dora" work,
*Opt. Express* **32**, 10801 (2024).

Both gradients are written by hand, which is why no autograd framework appears in the
dependency list:

* **volume**  ``dL/dv = P^T (P v - T_s d)`` -- a backprojection of the residual,
  preconditioned into a SIRT step by the ray-length and voxel-count images
  ``R = P(1)`` and ``C = P^T(1)``;
* **shift**   ``dL/ds_i = <res_i, grad(T_{s_i} d_i)>`` -- an image-gradient inner
  product, turned into a Gauss-Newton step by dividing by ``<grad, grad>``.

Origin. This is a port of ``joint_align_gd.py`` from the P06 ptycho-tomography
pipeline (beamtime 11023330), where it ran as the top-end alignment method on 907-918
projection lens-1/lens-2 stacks. The original was configured entirely through
environment variables (``JOINT_SMOKE``, ``JOINT_LONG``, ``JOINT_SCAP``,
``JOINT_REFINE``) and ran its whole schedule inside one ``main()``; here the schedule
is a :class:`JointGDConfig` and the loop is exposed **one iteration at a time**
through :meth:`JointGDAligner.step`, so the same driver can run this, the reprojection
engine and a vertical-mass-fluctuation pass identically. See ``docs/joint_gd.md``.


Five conventions, each of which is a bug if you get it wrong
------------------------------------------------------------

1. **Shifts are ``(dy, dx)``, row first, in
   :func:`~tktomo.ptycho_align.core.engine.apply_shifts`'s sign.** The order is
   ``(vertical rows, horizontal columns)``, matching ``com_prealign``'s ``(sy, sx)``
   and ``phase_cross_correlation``'s ``(d_row, d_col)``; storing them ``(dx, dy)``
   transposes every shift and the loop quietly optimises the wrong thing.

   The **sign** is the one trap this port had to be dragged through. TKtomo's
   ``apply_shifts(prj, sy, sx)`` is ``scipy.ndimage.shift(prj, (-sy, -sx))``: a feature
   at row 10 given ``sy=+3`` lands on row 7, which ``tests/test_ptycho_engine.py`` pins
   deliberately. The original P06 script's ``shifts_joint.tsv`` used the opposite --
   ndimage's -- sense, and a faithful transliteration therefore produced a *perfectly
   converged, sign-inverted* answer: the benchmark harness scored it 4.49 px while
   reporting that negating it would score 0.009 px. So everything public here (the
   ``shifts`` accessor, ``initial_shifts``, :class:`FinalizedShifts`) is in TKtomo's
   sense, and :meth:`JointGDAligner._shift_stack` is the single place the negation
   happens. If you export a TSV to compare against the original script, negate it.

2. **Shifts inside a stage are in that stage's BINNED pixels; the public API is always
   in full-resolution pixels.** Every stage boundary rescales by ``prev_binning /
   binning``. :attr:`JointGDAligner.shifts` and every :class:`JointGDIteration` convert
   back to full resolution for you. Mixing the two silently scales the answer by 2, 4
   or 16 -- large enough to look like a real misalignment, small enough not to crash.

3. **Never re-shift shifted data.** Each iteration re-shifts the *pristine* binned
   stack by the current cumulative shift. Warping an already-warped array compounds the
   interpolation and blurs the data away; this is the same rule as
   ``engine.py``'s convention 3.

4. **Sinogram layout is ``(det_row, angle, det_col)``, not ``(angle, row, col)``.**
   That is ASTRA's ``parallel3d`` layout and it is what :class:`Projector3D`
   implementations take and return. The measured stack is ``(angle, row, col)``, so a
   transpose sits between them. Getting it wrong does not raise -- it produces a
   plausible-looking, completely wrong reconstruction whenever ``n_angles`` happens to
   be compatible with ``n_rows``.

5. **Median-centre the shifts before you use them.** See :meth:`JointGDAligner.finalize`.
   The *mean* shift of the solution is degenerate with the rotation-axis position (in
   ``dx``) and with the ``z`` origin (in ``dy``): the objective cannot see it, so it
   random-walks. Skipping the centring makes a long run drift, and makes two runs of
   the same data disagree by a constant that is not an error.


Load-bearing numerics (do not "clean these up")
-----------------------------------------------

These were learned the hard way on real data and the method fails without them:

* **Median centring** -- convention 5 above. Applied in :meth:`~JointGDAligner.finalize`,
  which is the only supported way to read the answer out.
* **MAD outlier fallback** -- degraded projections do not fail loudly; they escape by
  sliding out of frame, because a featureless projection has no gradient to hold it.
  20 of 918 did exactly that on the first real run. Any projection whose shift is more
  than ``outlier_mad`` MADs from the median, or further than ``outlier_abs_px`` in
  absolute terms, is reset to the fallback shift and reported. Without this the bad
  projections are not merely useless, they poison the reconstruction they are fed into.
* **Shift-step damping (``lr_shift``) and the per-iteration cap (``shift_cap_px``)** --
  before these were added, this method diverged: loss to NaN and shifts to 1e4 px. The
  Gauss-Newton denominator ``<grad, grad>`` is tiny wherever a projection is flat, so
  an undamped, uncapped step is unbounded exactly where it is least trustworthy.
* **Volume warm-up (``warmup_iters``)** -- shifts stay frozen for the first iterations
  of *every* stage. Registering against a volume that has not formed yet is registering
  against noise, which is the same failure ``engine.shift_update_is_runaway`` exists to
  catch.
* **SIRT preconditioners** ``R = max(P(1), 1)`` and ``C = max(P^T(1), 1)`` -- the raw
  backprojection is wildly non-uniform (edges of the field of view see a handful of
  rays, the centre sees all of them) and a plain gradient step on it is unusable.
* **Loss back-off** -- if the loss rises above ``loss_backoff_factor`` times the best
  seen, the volume learning rate is halved and the momentum buffer cleared, in place,
  mid-stage. Cheaper and more reliable than picking a safe learning rate up front.

Deliberate deviations from the original script, all behaviour-preserving on the
default schedule: shifts accumulate in float64 rather than float32; the separable
three-axis ``gaussian_filter1d`` loop is one :func:`scipy.ndimage.gaussian_filter`
call (identical result); the volume is *not* carried across stage boundaries, matching
the original's explicit ``vol0=None``; non-finite losses raise
:class:`JointGDDivergence` instead of being logged and continued; and the reported
shifts are negated into TKtomo's sign convention (convention 1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from tktomo.ptycho_align.core.preprocess import bin_stack

logger = logging.getLogger(__name__)

__all__ = [
    "STAGES_LONG_JITTER",
    "STAGES_REFINE",
    "STAGES_SMOKE",
    "STAGES_STANDARD",
    "AstraProjector3D",
    "FinalizedShifts",
    "GDStage",
    "JointGDAligner",
    "JointGDConfig",
    "JointGDDivergence",
    "JointGDIteration",
    "NumpyProjector3D",
    "Projector3D",
    "available_projectors",
    "clean_shifts",
    "make_projector",
    "quality_weights",
    "register_projector",
]


class JointGDDivergence(RuntimeError):
    """The optimisation has left the region where its result means anything.

    Raised when the loss stops being finite, when the volume acquires NaNs, or when a
    shift exceeds :attr:`JointGDConfig.runaway_shift_px`. Continuing past any of those
    produces numbers, not answers, so this is deliberately fatal rather than a flag:
    the aligner's state is left exactly as it was for inspection, but the run is over.
    """


# --------------------------------------------------------------------------------
# Projector layer
# --------------------------------------------------------------------------------


@runtime_checkable
class Projector3D(Protocol):
    """A parallel-beam forward/back projector pair over a fixed geometry.

    Parallel-beam tomography about a vertical axis is **slice-independent**: detector
    row ``z`` sees volume slice ``z`` and nothing else. So the geometry is fully
    described by ``(n_slices, n_cols, angles)`` and both directions are a stack of
    independent 2-D problems -- which is what makes the numpy fallback tractable and
    what lets ASTRA do the whole volume in one kernel launch.

    Shapes, and they are not the obvious ones (convention 4 in the module docstring):

    * volume ``(n_slices, n_cols, n_cols)`` -- ``(z, y, x)``
    * sinogram ``(n_slices, n_angles, n_cols)`` -- ``(det_row, angle, det_col)``
    """

    name: str
    n_slices: int
    n_angles: int
    n_cols: int

    def forward(self, volume: np.ndarray) -> np.ndarray:
        """Project ``(n_slices, n_cols, n_cols)`` to ``(n_slices, n_angles, n_cols)``."""
        ...

    def backward(self, sino: np.ndarray) -> np.ndarray:
        """Backproject ``(n_slices, n_angles, n_cols)`` to ``(n_slices, n_cols, n_cols)``."""
        ...


class AstraProjector3D:
    """ASTRA ``parallel3d`` FP/BP on the GPU. The production backend.

    ASTRA is an *optional* dependency (``pip install -e ".[gpu]"``, or better
    ``conda install -c conda-forge astra-toolbox``) and is imported inside
    :meth:`__init__`, so importing this module never requires it. Constructing one
    without ASTRA, or without a CUDA device, raises immediately with a message that
    names the alternative rather than falling back silently -- a run that quietly
    switched to the numpy projector would take days and nobody would know why.
    """

    name = "astra"

    def __init__(self, n_slices: int, n_cols: int, angles: np.ndarray) -> None:
        try:
            import astra  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The 'astra' projector needs the astra-toolbox, which is not installed "
                "(conda install -c conda-forge astra-toolbox). Use "
                "projector='numpy' for a CPU parallel-beam projector -- correct, but "
                "orders of magnitude slower and only sane for small phantoms."
            ) from exc

        self._astra = astra
        self.n_slices = int(n_slices)
        self.n_cols = int(n_cols)
        self.angles = np.ascontiguousarray(angles, dtype=np.float64)
        self.n_angles = int(self.angles.size)
        # create_vol_geom(rows, cols, slices) -> arrays indexed (slice, row, col).
        self._vg = astra.create_vol_geom(self.n_cols, self.n_cols, self.n_slices)
        self._pg = astra.create_proj_geom(
            "parallel3d", 1.0, 1.0, self.n_slices, self.n_cols, self.angles
        )

    def _run(self, algorithm: str, vol_id: int, sino_id: int, out_id: int) -> np.ndarray:
        astra = self._astra
        cfg = astra.astra_dict(algorithm)
        if algorithm == "FP3D_CUDA":
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id
        else:
            cfg["ReconstructionDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg_id)
            out = astra.data3d.get(out_id)
        finally:
            astra.algorithm.delete(alg_id)
            astra.data3d.delete([vol_id, sino_id])
        return np.asarray(out, dtype=np.float32)

    def forward(self, volume: np.ndarray) -> np.ndarray:
        astra = self._astra
        vol_id = astra.data3d.create("-vol", self._vg, np.ascontiguousarray(volume, np.float32))
        sino_id = astra.data3d.create("-sino", self._pg)
        return self._run("FP3D_CUDA", vol_id, sino_id, sino_id)

    def backward(self, sino: np.ndarray) -> np.ndarray:
        astra = self._astra
        sino_id = astra.data3d.create("-sino", self._pg, np.ascontiguousarray(sino, np.float32))
        vol_id = astra.data3d.create("-vol", self._vg)
        return self._run("BP3D_CUDA", vol_id, sino_id, vol_id)


class NumpyProjector3D:
    """Rotate-and-sum parallel-beam projector in pure numpy/scipy. No GPU, no ASTRA.

    Exists so the unit tests -- and anyone evaluating the method before installing a
    CUDA stack -- can run the whole algorithm on a small phantom. It is ``O(n_angles)``
    array rotations per projection, so it is genuinely slow: usable up to roughly
    ``128**3``, hopeless on real data. :meth:`backward` is the numerical adjoint of
    :meth:`forward` only up to interpolation asymmetry (a few percent), which SIRT's
    preconditioners absorb without complaint -- where ASTRA's pair is exact to 0.000%.

    It agrees with :class:`AstraProjector3D` on geometry, not merely on shape:
    forward-projecting the same phantom through both differs by 0.5% relative (measured
    on a 20x48x48 phantom, 32 angles, an A100 node), so the angle handedness matches and
    a volume from one is not mirrored with respect to the other. That is worth
    re-checking if either side's geometry is ever touched -- note though that the
    *shifts* would survive a handedness flip anyway, since reversing the angle
    convention mirrors the volume and leaves every reprojection, and therefore every
    residual, identical.
    """

    name = "numpy"

    def __init__(self, n_slices: int, n_cols: int, angles: np.ndarray) -> None:
        self.n_slices = int(n_slices)
        self.n_cols = int(n_cols)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.n_angles = int(self.angles.size)
        self._degrees = np.degrees(self.angles)

    def forward(self, volume: np.ndarray) -> np.ndarray:
        from scipy.ndimage import rotate  # noqa: PLC0415

        volume = np.ascontiguousarray(volume, dtype=np.float32)
        out = np.empty((self.n_slices, self.n_angles, self.n_cols), dtype=np.float32)
        for i, degrees in enumerate(self._degrees):
            rotated = rotate(
                volume, degrees, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0
            )
            out[:, i, :] = rotated.sum(axis=1)
        return out

    def backward(self, sino: np.ndarray) -> np.ndarray:
        from scipy.ndimage import rotate  # noqa: PLC0415

        sino = np.ascontiguousarray(sino, dtype=np.float32)
        volume = np.zeros((self.n_slices, self.n_cols, self.n_cols), dtype=np.float32)
        for i, degrees in enumerate(self._degrees):
            smeared = np.repeat(sino[:, i, :][:, None, :], self.n_cols, axis=1)
            volume += rotate(
                smeared, -degrees, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0
            )
        return volume


_PROJECTORS: dict[str, Callable[[int, int, np.ndarray], Projector3D]] = {
    "astra": AstraProjector3D,
    "numpy": NumpyProjector3D,
}


def register_projector(name: str, factory: Callable[[int, int, np.ndarray], Projector3D]) -> None:
    """Add a projector under ``name``, callable as ``factory(n_slices, n_cols, angles)``.

    The door left open for a native TKtomo projector. Note that
    :class:`~tktomo.recon.backend.ReconBackend` is *not* directly usable here: it
    offers ``reconstruct``/``reproject``, and this optimiser needs the raw adjoint pair
    ``P``/``P^T`` -- a reconstruction is a whole inner solve, not a backprojection.
    """
    if not name:
        raise ValueError("projector name must be non-empty")
    _PROJECTORS[name] = factory


def available_projectors() -> list[str]:
    return sorted(_PROJECTORS)


def make_projector(
    name: str, n_slices: int, n_cols: int, angles: np.ndarray
) -> Projector3D:
    """Build a projector by name. ``"auto"`` prefers ASTRA and says which it chose.

    ``"auto"`` exists for benchmark drivers that may or may not land on a GPU node. It
    logs the choice at WARNING when it falls back, because the fallback is a ~1000x
    slowdown and a silent one would look like a hang.
    """
    if name == "auto":
        try:
            projector = AstraProjector3D(n_slices, n_cols, angles)
        except ImportError as exc:
            logger.warning(
                "projector='auto': ASTRA unavailable (%s); falling back to the numpy "
                "projector, which is orders of magnitude slower.",
                exc,
            )
            return NumpyProjector3D(n_slices, n_cols, angles)
        logger.info("projector='auto': using ASTRA")
        return projector

    try:
        factory = _PROJECTORS[name]
    except KeyError:
        raise KeyError(
            f"Unknown projector {name!r}. Available: {available_projectors()} (plus 'auto')."
        ) from None
    return factory(n_slices, n_cols, angles)


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class GDStage:
    """One multi-resolution stage: bin the data, run ``iterations``, rescale, repeat.

    ``smooth_sigma`` is a Gaussian low-pass (in binned voxels) applied to the *volume
    update*, never to the data. Coarse stages need it because a coarse volume is where
    high-frequency junk gets baked in; the finest stage runs with 0.
    """

    binning: int
    iterations: int
    smooth_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.binning < 1:
            raise ValueError(f"binning must be >= 1, got {self.binning}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.smooth_sigma < 0:
            raise ValueError(f"smooth_sigma must be >= 0, got {self.smooth_sigma}")


#: The schedule the P06 lens-1/lens-2 stacks were aligned with (the original default).
STAGES_STANDARD: tuple[GDStage, ...] = (
    GDStage(16, 150, 2.0),
    GDStage(8, 150, 1.0),
    GDStage(4, 100, 0.0),
)
#: One coarse stage. A numeric smoke test, not an alignment (``JOINT_SMOKE=1``).
STAGES_SMOKE: tuple[GDStage, ...] = (GDStage(16, 60, 2.0),)
#: For large-jitter datasets -- lens-2 ran ~120 px rms and needed the extra iterations
#: at every scale to walk in from that far out (``JOINT_LONG=1``).
STAGES_LONG_JITTER: tuple[GDStage, ...] = (
    GDStage(16, 250, 2.0),
    GDStage(8, 200, 1.0),
    GDStage(4, 150, 0.0),
)
#: Refinement of an already-aligned, normalised stack: residual shifts are a few px, so
#: skip the coarse stages entirely and add the binning-2 stage the standard schedule
#: lacks -- that is where sub-4-px sinogram edge jitter lives (``JOINT_REFINE=1``).
STAGES_REFINE: tuple[GDStage, ...] = (GDStage(4, 100, 0.5), GDStage(2, 80, 0.0))


@dataclass
class JointGDConfig:
    """Everything the joint gradient-descent aligner needs, in the repo's config style.

    The original script read all of this from environment variables; the four named
    schedules above are the four ``JOINT_*`` presets, and :attr:`shift_cap_px` is
    ``JOINT_SCAP``.
    """

    stages: tuple[GDStage, ...] = STAGES_STANDARD

    # -- optimiser -----------------------------------------------------------------
    lr_volume: float = 1.0
    """SIRT relaxation. Stable for ``0 < lr_volume < 2`` on the preconditioned step."""
    lr_shift: float = 0.5
    """Damping on the Gauss-Newton shift step. Load-bearing -- see the module docstring."""
    shift_cap_px: float = 0.5
    """Max ``|shift change|`` per iteration, in *binned* px. Load-bearing."""
    warmup_iters: int = 15
    """Volume-only iterations at the start of **each** stage, before shifts engage."""
    momentum: float = 0.9
    """Nesterov momentum on the volume update only; the shifts have no momentum."""
    loss_backoff_factor: float = 3.0
    """Halve ``lr_volume`` and clear momentum when the loss exceeds this x its best."""

    # -- which directions to solve for ---------------------------------------------
    align_vertical: bool = True
    align_horizontal: bool = True
    """Solving one direction at a time is not the original's behaviour but is what the
    roadmap argues for: vertical is decoupled from the rotation angle and can be fixed
    far more cheaply elsewhere, leaving this loop the horizontal problem it is actually
    needed for. Turning ``align_vertical`` off freezes ``dy`` at its initial value."""

    # -- backend --------------------------------------------------------------------
    projector: str = "astra"
    """``"astra"`` | ``"numpy"`` | ``"auto"`` | any name given to
    :func:`register_projector`. The default is the production one and raises a clear
    error when ASTRA is missing rather than degrading to the CPU projector."""

    # -- honest failure --------------------------------------------------------------
    runaway_shift_px: float = 1000.0
    """A shift beyond this (full-resolution px) raises :class:`JointGDDivergence`."""

    # -- finalisation (see JointGDAligner.finalize) ----------------------------------
    median_center: bool = True
    outlier_mad: float = 5.0
    outlier_abs_px: float = 150.0
    outlier_fallback: str = "zero"
    """``"zero"`` (no net correction -- keep whatever pre-alignment the input carries,
    the original's behaviour) or ``"initial"`` (revert to ``initial_shifts``)."""

    def __post_init__(self) -> None:
        self.stages = tuple(
            s if isinstance(s, GDStage) else GDStage(**dict(s)) for s in self.stages
        )
        if not self.stages:
            raise ValueError("at least one stage is required")
        if not 0.0 < self.lr_volume < 2.0:
            raise ValueError(f"lr_volume must lie in (0, 2), got {self.lr_volume}")
        if self.lr_shift <= 0.0:
            raise ValueError(f"lr_shift must be > 0, got {self.lr_shift}")
        if self.shift_cap_px <= 0.0:
            raise ValueError(f"shift_cap_px must be > 0, got {self.shift_cap_px}")
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {self.momentum}")
        if self.outlier_fallback not in ("zero", "initial"):
            raise ValueError(
                f"outlier_fallback must be 'zero' or 'initial', got {self.outlier_fallback!r}"
            )
        if not (self.align_vertical or self.align_horizontal):
            raise ValueError(
                "both align_vertical and align_horizontal are off, so there is nothing "
                "to solve for. Use the projector directly if you only want a volume."
            )

    @property
    def total_iterations(self) -> int:
        return sum(stage.iterations for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly. ``stages`` becomes a list of dicts."""
        raw = asdict(self)
        raw["stages"] = [asdict(stage) for stage in self.stages]
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> JointGDConfig:
        """Rebuild from :meth:`to_dict` output that has been through JSON.

        JSON has neither tuples nor dataclasses, so ``stages`` comes back as a list of
        dicts and has to be put back on every round trip. Doing it here rather than at
        each call site is what stops a session file and a socket from disagreeing --
        the same reasoning as :meth:`AlignConfig.from_dict`.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"Unknown JointGDConfig field(s): {', '.join(unknown)}. "
                "The config was probably written by a newer version of tktomo."
            )
        values = dict(raw)
        stages = values.get("stages")
        if stages is not None:
            values["stages"] = tuple(
                s if isinstance(s, GDStage) else GDStage(**dict(s)) for s in stages
            )
        return cls(**values)


def quality_weights(
    steepness: np.ndarray, *, scale: float = 0.01, low: float = 0.2, high: float = 1.0
) -> np.ndarray:
    """Per-projection weights from a per-projection degradation metric.

    ``w_i = clip(scale / max(steepness_i, 1e-3), low, high)``: the worse the metric,
    the less the projection contributes to the volume. The P06 pipeline's metric is the
    fitted steepness of the phase ramp left after ptychographic reconstruction, which
    tracks the projections that failed to converge.

    Note that the weights enter the **volume** update only, exactly as in the original.
    They deliberately do *not* damp the shift update: a bad projection still needs its
    own shift estimated, and if that estimate runs away, the MAD rule in
    :meth:`JointGDAligner.finalize` is what catches it.
    """
    steepness = np.asarray(steepness, dtype=np.float64)
    return np.clip(scale / np.maximum(steepness, 1e-3), low, high).astype(np.float32)


# --------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------


@dataclass
class JointGDIteration:
    """What one call to :meth:`JointGDAligner.step` produced."""

    iteration: int  # 1-based, counted across the whole schedule
    stage: int  # 0-based index into config.stages
    stage_iteration: int  # 0-based index within the stage
    binning: int
    loss: float  # mean squared residual, in this stage's binned units
    shifts: np.ndarray  # (n, 2) cumulative (dy, dx), FULL-RESOLUTION px
    update_rms: float  # RMS of this iteration's shift step, full-resolution px
    max_abs_shift: float  # max |shift|, full-resolution px
    lr_volume: float  # current volume learning rate (halved by back-offs)
    backed_off: bool  # the loss rose and the learning rate was cut this iteration
    shifts_engaged: bool  # False during a stage's warm-up
    wallclock_s: float

    @property
    def is_warmup(self) -> bool:
        return not self.shifts_engaged


@dataclass
class FinalizedShifts:
    """The cleaned answer. Produced only by :meth:`JointGDAligner.finalize`."""

    shifts: np.ndarray  # (n, 2) (dy, dx), full-resolution px, ready for apply_shifts
    outliers: np.ndarray  # (n,) bool -- which projections were reset to the fallback
    median_offset: np.ndarray  # (2,) the degenerate global shift that was removed
    mad: np.ndarray  # (2,) median absolute deviation used for the outlier rule
    raw: np.ndarray  # (n, 2) the shifts before centring and outlier rejection

    @property
    def n_outliers(self) -> int:
        return int(self.outliers.sum())

    @property
    def rms(self) -> tuple[float, float]:
        """``(rms dy, rms dx)`` in full-resolution px."""
        return (float(self.shifts[:, 0].std()), float(self.shifts[:, 1].std()))

    @property
    def max_abs(self) -> float:
        return float(np.abs(self.shifts).max()) if self.shifts.size else 0.0

    def summary(self) -> str:
        return (
            f"joint-GD shifts: rms dy {self.rms[0]:.2f} px, dx {self.rms[1]:.2f} px, "
            f"max |s| {self.max_abs:.1f} px, median offset removed "
            f"({self.median_offset[0]:+.2f}, {self.median_offset[1]:+.2f}) px, "
            f"{self.n_outliers} MAD outlier(s) reset"
        )


def clean_shifts(
    shifts: np.ndarray,
    *,
    median_center: bool = True,
    outlier_mad: float = 5.0,
    outlier_abs_px: float = 150.0,
    fallback: np.ndarray | float = 0.0,
) -> FinalizedShifts:
    """Median-centre a shift solution and reject its outliers. Pure, no state.

    Split out of :meth:`JointGDAligner.finalize` because these two rules are the part
    of the method that has to be *right*, and a pure function is the part that can be
    tested without running an optimisation. The docstring of :meth:`finalize` explains
    why each one exists; this is the arithmetic.

    ``shifts`` is ``(n, 2)`` of ``(dy, dx)``. ``fallback`` is either a scalar/2-vector
    applied to every outlier or an ``(n, 2)`` array of per-projection fallbacks, and is
    interpreted in the same (centred, if ``median_center``) frame as the output.
    """
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError(f"shifts must be (n, 2) of (dy, dx), got shape {shifts.shape}")
    if not np.isfinite(shifts).all():
        raise JointGDDivergence(
            "the shift solution contains non-finite values; the run diverged and there "
            "is nothing to finalize."
        )

    raw = shifts
    shifts = shifts.copy()
    # The MEDIAN, not the mean: one projection that has slid out of frame moves a mean
    # by enough to shift every other projection, which is exactly the failure the
    # outlier rule below is trying to contain.
    median_offset = np.median(shifts, axis=0)
    if median_center:
        shifts -= median_offset

    # The MAD test is on the deviation from the median, the absolute test on the shift
    # itself (a 150 px correction is implausible wherever the median sits). With
    # median_center on -- the default and the original's behaviour -- the two framings
    # coincide exactly, because the median of a median-centred vector is zero.
    centre = np.median(shifts, axis=0)
    deviation = shifts - centre
    mad = np.median(np.abs(deviation), axis=0) + 1e-3
    outliers = (np.abs(deviation) > outlier_mad * mad).any(axis=1) | (
        np.hypot(shifts[:, 0], shifts[:, 1]) > outlier_abs_px
    )
    fallback = np.asarray(fallback, dtype=np.float64)
    shifts[outliers] = fallback[outliers] if fallback.ndim == 2 else fallback
    return FinalizedShifts(
        shifts=shifts, outliers=outliers, median_offset=median_offset, mad=mad, raw=raw
    )


# --------------------------------------------------------------------------------
# The aligner
# --------------------------------------------------------------------------------


@dataclass
class JointGDAligner:
    """Joint volume/shift gradient descent, one iteration per :meth:`step`.

    Qt-free and headless by design, like :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine`:
    a GUI worker thread, a notebook and the benchmark driver all use the same two calls.

    ``projections`` is ``(n_angles, n_rows, n_cols)`` and is **never modified**;
    ``angles`` is in **radians** (the repo-wide convention -- the original script's
    ``np.deg2rad`` happens at the caller now). ``initial_shifts`` is ``(n_angles, 2)``
    of ``(dy, dx)`` in full-resolution px, e.g. straight from
    :func:`~tktomo.ptycho_align.core.com.com_prealign` as
    ``np.column_stack([result.sy, result.sx])``.

    Typical use::

        aligner = JointGDAligner(prj, angles, JointGDConfig(projector="astra"))
        for result in aligner.run():          # or: while not aligner.done: aligner.step()
            print(result.iteration, result.loss)
        answer = aligner.finalize()           # median-centred, outliers rejected
        aligned = aligner.aligned_projections(answer.shifts)

    Memory: the full-resolution stack is held in RAM and re-binned at every stage
    boundary, so peak use is roughly ``stack + stack/binning^2 + 3 volumes``. A
    907x1488x1816 float32 stack is 9.8 GB before anything else, which is why the real
    runs are batch jobs.
    """

    projections: np.ndarray
    angles: np.ndarray
    config: JointGDConfig = field(default_factory=JointGDConfig)
    weights: np.ndarray | None = None
    initial_shifts: np.ndarray | None = None

    def __post_init__(self) -> None:
        prj = np.asarray(self.projections)
        if prj.ndim != 3:
            raise ValueError(
                f"projections must be (n_angles, n_rows, n_cols), got shape {prj.shape}"
            )
        self.projections = np.ascontiguousarray(prj, dtype=np.float32)
        n = self.projections.shape[0]

        self.angles = np.asarray(self.angles, dtype=np.float64).ravel()
        if self.angles.size != n:
            raise ValueError(
                f"{self.angles.size} angles for {n} projections. Angles are in RADIANS "
                "here; if yours span 0..180 you are passing degrees."
            )
        if np.ptp(self.angles) > 4.0 * np.pi:
            raise ValueError(
                f"angles span {np.ptp(self.angles):.1f}, which is not radians. "
                "Convert with np.deg2rad()."
            )

        if self.weights is None:
            self._weights = np.ones(n, dtype=np.float32)
        else:
            self._weights = np.asarray(self.weights, dtype=np.float32).ravel()
            if self._weights.size != n:
                raise ValueError(f"{self._weights.size} weights for {n} projections")
            if np.any(self._weights < 0):
                raise ValueError("weights must be non-negative")

        if self.initial_shifts is None:
            self._initial = np.zeros((n, 2), dtype=np.float64)
        else:
            self._initial = np.asarray(self.initial_shifts, dtype=np.float64)
            if self._initial.shape != (n, 2):
                raise ValueError(
                    f"initial_shifts must be ({n}, 2) of (dy, dx), got "
                    f"{self._initial.shape}"
                )

        # Working state. `_shifts` is in the CURRENT stage's binned px (convention 2).
        self._stage_index = -1
        self._stage_iteration = 0
        self._iteration = 0
        self._binning = 0
        self._shifts = self._initial.copy()  # rescaled by _begin_stage
        self._data: np.ndarray | None = None
        self._projector: Projector3D | None = None
        self._volume: np.ndarray | None = None
        self._velocity: np.ndarray | float = 0.0
        self._R: np.ndarray | None = None
        self._C: np.ndarray | None = None
        self._lr_volume = self.config.lr_volume
        self._best_loss = np.inf
        self._sino_weights = self._weights[None, :, None]
        self.history: list[JointGDIteration] = []

    # -- accessors ------------------------------------------------------------------

    @property
    def n_angles(self) -> int:
        return int(self.projections.shape[0])

    @property
    def iteration(self) -> int:
        """Completed iterations across the whole schedule."""
        return self._iteration

    @property
    def total_iterations(self) -> int:
        return self.config.total_iterations

    @property
    def done(self) -> bool:
        return self._iteration >= self.total_iterations

    @property
    def binning(self) -> int:
        """The binning factor of the stage currently running (0 before the first step)."""
        return self._binning

    @property
    def volume(self) -> np.ndarray | None:
        """The current stage's volume, on that stage's **binned** voxel grid.

        Not a deliverable: the volume is a means to the shifts here, it is reset at
        every stage boundary, and the final one lives on the finest *stage's* grid,
        which is usually still coarser than the data. Reconstruct properly from
        :meth:`aligned_projections` when you want a volume.
        """
        return self._volume

    @property
    def shifts(self) -> np.ndarray:
        """Cumulative ``(n, 2)`` ``(dy, dx)`` shifts in **full-resolution** px.

        Raw: neither median-centred nor outlier-filtered. Use :meth:`finalize` before
        applying these to anything (module docstring, convention 5).
        """
        scale = float(self._binning) if self._binning else 1.0
        return self._shifts * scale

    # -- the loop -------------------------------------------------------------------

    def step(self) -> JointGDIteration:
        """Run exactly ONE gradient-descent iteration, advancing stages as needed.

        The first call, and the first call after a stage's iteration budget runs out,
        also does that stage's setup: bin the stack, rescale the shifts, build the
        projector and compute the SIRT preconditioners ``R = P(1)``, ``C = P^T(1)``.
        That setup is two full projections and is charged to the iteration that
        triggered it, so the first iteration of a stage is always the slow one.
        """
        if self.done:
            raise RuntimeError(
                f"the schedule is exhausted ({self.total_iterations} iterations over "
                f"{len(self.config.stages)} stage(s)). Call finalize(), or build a new "
                "aligner with a longer schedule."
            )
        started = time.perf_counter()

        if self._stage_index < 0 or self._stage_iteration >= self._stage().iterations:
            self._begin_stage(self._stage_index + 1)

        cfg = self.config
        stage = self._stage()
        data = self._data
        projector = self._projector
        assert data is not None and projector is not None  # set by _begin_stage

        # 1. Re-shift the PRISTINE binned stack by the cumulative shift (convention 3).
        shifted = self._shift_stack(data, self._shifts)

        # 2. Residual in ASTRA sinogram layout (convention 4).
        sino = np.ascontiguousarray(np.transpose(shifted, (1, 0, 2)))
        residual = projector.forward(self._volume) - sino
        # float32 square (one temporary the size of the sinogram, as in the original)
        # but a float64 accumulator: summing 1e8 float32 terms in float32 loses the
        # last digits of the loss, which is what the back-off threshold compares.
        loss = float(np.mean(np.square(residual), dtype=np.float64))
        if not np.isfinite(loss):
            raise JointGDDivergence(
                f"the loss became {loss} at iteration {self._iteration + 1} "
                f"(stage {self._stage_index}, binning {stage.binning}). The volume step "
                "has diverged; lower lr_volume, or raise warmup_iters so the shifts "
                "engage against a formed volume."
            )

        # 3. Loss back-off. Compared against the BEST loss so far, not the previous
        #    one -- a single bad iteration should not raise the bar for the next.
        backed_off = loss > cfg.loss_backoff_factor * self._best_loss
        if backed_off:
            self._lr_volume *= 0.5
            self._velocity = 0.0
            logger.warning(
                "stage %d it %d: loss %.4e jumped above %gx the best %.4e -> lr_volume=%.4f",
                self._stage_index,
                self._stage_iteration,
                loss,
                cfg.loss_backoff_factor,
                self._best_loss,
                self._lr_volume,
            )
        self._best_loss = min(self._best_loss, loss)

        # 4. Volume: preconditioned SIRT step, smoothed, with Nesterov momentum.
        update = self._lr_volume * (
            projector.backward(residual * self._sino_weights / self._R) / self._C
        )
        if stage.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter  # noqa: PLC0415

            update = gaussian_filter(update, stage.smooth_sigma)
        self._velocity = cfg.momentum * self._velocity + update
        # Sutskever's form of Nesterov: look ahead by one momentum step.
        self._volume = self._volume - (cfg.momentum * self._velocity + update)
        if not np.isfinite(self._volume).all():
            raise JointGDDivergence(
                f"the volume acquired non-finite values at iteration {self._iteration + 1}. "
                "lr_volume is too large for this data."
            )

        # 5. Shifts: damped, capped Gauss-Newton, after the stage's volume warm-up.
        engaged = self._stage_iteration >= cfg.warmup_iters
        update_shift = np.zeros_like(self._shifts)
        if engaged:
            update_shift = self._shift_step(residual, shifted)
            # PLUS, not minus -- see _shift_step for the derivation. It is minus in the
            # original script, which works in the opposite sign convention.
            self._shifts = self._shifts + update_shift
            self._guard_runaway()

        self._stage_iteration += 1
        self._iteration += 1
        scale = float(stage.binning)
        result = JointGDIteration(
            iteration=self._iteration,
            stage=self._stage_index,
            stage_iteration=self._stage_iteration - 1,
            binning=stage.binning,
            loss=loss,
            shifts=self._shifts * scale,
            update_rms=float(np.sqrt(np.mean(update_shift**2))) * scale,
            max_abs_shift=float(np.abs(self._shifts).max()) * scale,
            lr_volume=self._lr_volume,
            backed_off=backed_off,
            shifts_engaged=engaged,
            wallclock_s=time.perf_counter() - started,
        )
        self.history.append(result)
        if result.stage_iteration % 20 == 0 or self._stage_iteration == stage.iterations:
            logger.info(
                "[bin %d] it %4d  loss %.5e  |s|max %.2f px (full-res)",
                stage.binning,
                result.stage_iteration,
                loss,
                result.max_abs_shift,
            )
        return result

    def run(
        self,
        n: int | None = None,
        cancel_event: Any | None = None,
        callback: Callable[[JointGDIteration], None] | None = None,
    ) -> list[JointGDIteration]:
        """Call :meth:`step` ``n`` times, or to the end of the schedule when ``n`` is None.

        Aborts before the next iteration if ``cancel_event`` is set, so a cancelled run
        always leaves a complete, valid state -- the same contract as
        :meth:`AlignmentEngine.run`. :class:`JointGDDivergence` is *not* swallowed:
        there is nothing useful to return from a diverged run.
        """
        remaining = self.total_iterations - self._iteration if n is None else int(n)
        results: list[JointGDIteration] = []
        for _ in range(max(0, remaining)):
            if self.done:
                break
            if cancel_event is not None and cancel_event.is_set():
                logger.info("run cancelled after %d iteration(s)", len(results))
                break
            result = self.step()
            results.append(result)
            if callback is not None:
                callback(result)
        return results

    # -- reading the answer out -----------------------------------------------------

    def finalize(self) -> FinalizedShifts:
        """Median-centre the shifts and reject MAD outliers. **The only supported exit.**

        Two things happen here, both load-bearing (module docstring):

        1. **Median centring.** The global mean shift is invisible to the objective --
           in ``dx`` it is the rotation-axis position, in ``dy`` the ``z`` origin -- so
           the optimiser lets it drift. Pinning it to the median (not the mean: one
           runaway projection moves a mean) makes the answer reproducible and keeps the
           rotation centre where the caller put it.
        2. **MAD outlier rejection.** A projection too degraded to register does not
           stall, it slides out of frame; on the first real 918-projection run, 20 did.
           Anything more than ``outlier_mad`` MADs from the median in either axis, or
           further than ``outlier_abs_px`` in total, is reset to the fallback
           (``"zero"``: no net correction, i.e. whatever pre-alignment the input stack
           already carries -- which after centring is also the consensus of the good
           projections) and flagged in :attr:`FinalizedShifts.outliers`.

        The arithmetic lives in :func:`clean_shifts`; this only chooses the fallback.
        Safe to call at any point, including mid-run.
        """
        cfg = self.config
        raw = self.shifts
        if cfg.outlier_fallback == "zero":
            fallback: np.ndarray | float = 0.0
        else:
            # "initial" is expressed in the same centred frame as the output, so the
            # same median offset has to come off it.
            fallback = self._initial - (
                np.median(raw, axis=0) if cfg.median_center else 0.0
            )

        result = clean_shifts(
            raw,
            median_center=cfg.median_center,
            outlier_mad=cfg.outlier_mad,
            outlier_abs_px=cfg.outlier_abs_px,
            fallback=fallback,
        )
        if result.outliers.any():
            index = np.flatnonzero(result.outliers)
            logger.warning(
                "%d of %d projections exceeded %g MAD or %g px and were reset to the "
                "%s fallback: %s%s",
                index.size,
                self.n_angles,
                cfg.outlier_mad,
                cfg.outlier_abs_px,
                cfg.outlier_fallback,
                index[:12].tolist(),
                "..." if index.size > 12 else "",
            )
        return result

    def aligned_projections(self, shifts: np.ndarray | None = None) -> np.ndarray:
        """The pristine full-resolution stack with ``shifts`` applied, once.

        Pass ``FinalizedShifts.shifts``. Defaults to the raw shifts, which is almost
        never what you want -- see :meth:`finalize`.
        """
        shifts = self.shifts if shifts is None else np.asarray(shifts, dtype=np.float64)
        if shifts.shape != (self.n_angles, 2):
            raise ValueError(
                f"shifts must be ({self.n_angles}, 2) of (dy, dx), got {shifts.shape}"
            )
        return self._shift_stack(self.projections, shifts)

    # -- internals ------------------------------------------------------------------

    def _stage(self) -> GDStage:
        return self.config.stages[self._stage_index]

    def _begin_stage(self, index: int) -> None:
        """Bin the data, rescale the shifts, rebuild the projector and the preconditioners."""
        stage = self.config.stages[index]
        previous = self._binning

        # At binning 1 both calls are no-ops and `data` IS `self.projections`. That
        # aliasing is deliberate -- a copy would double the peak memory of a 10 GB
        # stack -- and safe only because nothing here writes into `data`: _shift_stack
        # allocates its output. Keep it that way.
        data = np.ascontiguousarray(
            bin_stack(self.projections, stage.binning), dtype=np.float32
        )
        # Convention 2: shifts live in the current stage's binned px. Coming in from
        # full resolution (the first stage) the scale is 1/binning; between stages it
        # is previous/current.
        scale = (previous if previous else 1.0) / stage.binning
        self._shifts = self._shifts * scale

        n_slices, n_cols = data.shape[1], data.shape[2]
        if n_slices < 4 or n_cols < 4:
            raise ValueError(
                f"stage {index} bins the projections to {n_slices}x{n_cols} px, which is "
                "too small to align. Drop the coarse stages (see STAGES_REFINE)."
            )
        projector = make_projector(self.config.projector, n_slices, n_cols, self.angles)

        ones_volume = np.ones((n_slices, n_cols, n_cols), dtype=np.float32)
        ones_sino = np.ones((n_slices, self.n_angles, n_cols), dtype=np.float32)
        # SIRT preconditioners: ray lengths and voxel hit counts. Floored at 1 so the
        # divisions cannot explode where the field of view is barely sampled.
        self._R = np.maximum(projector.forward(ones_volume), 1.0)
        self._C = np.maximum(projector.backward(ones_sino), 1.0)

        self._stage_index = index
        self._stage_iteration = 0
        self._binning = stage.binning
        self._data = data
        self._projector = projector
        # A fresh volume per stage, matching the original's explicit vol0=None. Carrying
        # an upsampled volume across the boundary would be the obvious improvement, and
        # would change the numbers, so it is not done here.
        self._volume = np.zeros((n_slices, n_cols, n_cols), dtype=np.float32)
        self._velocity = 0.0
        self._lr_volume = self.config.lr_volume
        self._best_loss = np.inf
        self._sino_weights = self._weights[None, :, None]

        logger.info(
            "=== stage %d: binning %d, data %s, %d iterations, smooth %.1f, "
            "projector %r ===",
            index,
            stage.binning,
            data.shape,
            stage.iterations,
            stage.smooth_sigma,
            getattr(projector, "name", self.config.projector),
        )

    @staticmethod
    def _shift_stack(stack: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """Translate every frame. ``order=1``/``mode='nearest'`` is not decoration.

        Linear interpolation because this runs once per iteration on the whole stack and
        a spline would dominate the cost; ``mode='nearest'`` because zero-filling the
        edge of a *phase* projection injects a hard step that the image gradient then
        registers against -- the object's own edge, not the sample's.
        """
        from scipy.ndimage import shift as ndshift  # noqa: PLC0415

        # THE negation (convention 1). scipy shifts by +s, apply_shifts by -s; this is
        # the only place in the module that knows, so the whole public surface reads in
        # TKtomo's sense and the optimiser below reads in this one.
        out = np.empty_like(stack)
        for i in range(stack.shape[0]):
            out[i] = ndshift(stack[i], (-shifts[i, 0], -shifts[i, 1]), order=1, mode="nearest")
        return out

    def _shift_step(self, residual: np.ndarray, shifted: np.ndarray) -> np.ndarray:
        """Damped, capped Gauss-Newton step for every projection's ``(dy, dx)``.

        In TKtomo's sign convention the aligned projection is ``T_s d(x) = d(x + s)``
        (convention 1), so ``d(T_s d)/ds = +grad(T_s d)`` and the gradient of
        ``0.5||P v - T_s d||^2`` with respect to ``s`` is ``-<res, grad(T_s d)>``, with
        Gauss-Newton curvature ``<grad, grad>``. The step is therefore *added*.
        Both are per-projection scalars per axis -- the step is two inner products and a
        division, which is why the shift half of this optimiser is free next to the
        projections.

        The ``1e-6`` in the denominator and the clip to ``+-shift_cap_px`` are the two
        things standing between this and the divergence in the module docstring: a flat
        projection has ``<grad, grad> ~ 0`` and would otherwise take an infinite step.
        """
        cfg = self.config
        residual_t = np.transpose(residual, (1, 0, 2))  # -> (angle, row, col)
        step = np.zeros_like(self._shifts)
        for i in range(shifted.shape[0]):
            gy, gx = np.gradient(shifted[i])
            numerator = np.array(
                [float((residual_t[i] * gy).sum()), float((residual_t[i] * gx).sum())],
                dtype=np.float64,
            )
            denominator = np.array(
                [float((gy * gy).sum()), float((gx * gx).sum())], dtype=np.float64
            ) + 1e-6
            step[i] = np.clip(
                cfg.lr_shift * numerator / denominator, -cfg.shift_cap_px, cfg.shift_cap_px
            )
        if not cfg.align_vertical:
            step[:, 0] = 0.0
        if not cfg.align_horizontal:
            step[:, 1] = 0.0
        return step

    def _guard_runaway(self) -> None:
        full_res = np.abs(self._shifts) * self._binning
        if not np.isfinite(full_res).all() or full_res.max() > self.config.runaway_shift_px:
            worst = int(np.argmax(full_res.max(axis=1)))
            raise JointGDDivergence(
                f"projection {worst} has run to {full_res[worst].round(1).tolist()} px "
                f"(full-resolution), past runaway_shift_px="
                f"{self.config.runaway_shift_px:g}. Either the data is not the geometry "
                "the angles describe, or lr_shift/shift_cap_px are too large for it."
            )

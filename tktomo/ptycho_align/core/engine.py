"""The step-wise iterative reprojection alignment loop.

This is a re-implementation of the Gursoy et al. (2017) joint re-projection
algorithm as found in ``tomopy.prep.alignment.align_joint`` / ``align_seq``,
exposed **one outer iteration at a time**.

Why re-implement rather than call TomoPy: ``align_joint``/``align_seq`` run their
whole iteration loop internally, with no callback, no generator and no way to
pause. The entire point of this tool is that the user can run one iteration, look
at the tomogram, then run five more. So :meth:`AlignmentEngine.step` is a plain
function that does exactly one iteration and returns control to the caller.

Three conventions, each of which is a bug if you get it wrong:

1. **Registration direction.** ``phase_cross_correlation(reference, moving)``
   returns the shift that registers *moving* onto *reference*. We pass
   ``(measured, simulated)``, matching TomoPy. Passing them the other way round
   negates every update and the alignment diverges instead of converging.

2. **Axis order.** ``phase_cross_correlation`` returns ``(d_row, d_col)``, i.e.
   ``(dy, dx)``: row is vertical, column is horizontal. Note that TomoPy's
   ``shift_images(prj, sx, sy)`` has *misleading parameter names* -- it builds a
   ``SimilarityTransform(translation=(sy, sx))`` and skimage's ``translation`` is
   ``(column, row)``, so its **first** argument is really the row shift. We wrap it
   as :func:`apply_shifts` with honest names so no caller has to know that.

3. **Never re-shift shifted data.** The cumulative shift is always applied to the
   pristine ``original``. Repeatedly warping an already-warped array compounds the
   interpolation and blurs the data away. (``shift_images`` also rescales its input
   *in place*, so it always gets a copy.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from tktomo.ptycho_align.core.state import AlignmentState, IterationResult, VolumePolicy

logger = logging.getLogger(__name__)

__all__ = [
    "AlignConfig",
    "AlignmentEngine",
    "algorithm_rejects_negatives",
    "apply_shifts",
]

# Algorithms that take neither `num_iter` nor a warm-start `init_recon`.
_DIRECT_ALGORITHMS = frozenset({"gridrec", "fbp"})

# Emission algorithms: they model photon counts with a *multiplicative* update, so they
# assume the data is non-negative. Phase projections routinely are not (~20% of voxels
# are negative after ramp/offset removal), and feeding them negatives makes the update
# flip sign and diverge explosively -- residual 0.05 -> 62 within five iterations.
_EMISSION_ALGORITHMS = frozenset({"mlem", "osem"})

# How far above its own best residual a run may climb before we call it diverging.
_DIVERGENCE_FACTOR = 2.0


def algorithm_rejects_negatives(algorithm: str, projections: np.ndarray) -> str | None:
    """Return a reason string if this algorithm cannot cope with this data, else None."""
    if algorithm not in _EMISSION_ALGORITHMS:
        return None
    negative = int((projections < 0).sum())
    if not negative:
        return None
    fraction = negative / projections.size
    return (
        f"'{algorithm}' is an emission algorithm and assumes non-negative data, but "
        f"{fraction:.0%} of the projection values are negative. It will diverge. Use "
        f"'sirt', 'art' or 'gridrec', or enable 'Invert' and re-check the preprocessing."
    )


def apply_shifts(prj: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Shift each projection by ``(sy, sx)`` = (vertical rows, horizontal columns).

    Wraps ``tomopy.prep.alignment.shift_images``, whose own parameter names are
    the wrong way round (see this module's docstring). Uses TomoPy's implementation
    rather than a fresh one so results stay comparable with ``align_joint``.

    The input is **mutated** by TomoPy's internal rescaling, so pass a copy.
    """
    sy = np.asarray(sy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)

    if not sy.any() and not sx.any():
        # An identity warp, but TomoPy would still run a 5th-order spline resample over
        # the whole stack to compute it -- 30 s for a 410 x 300 x 600 ptycho stack, on
        # the GUI thread, every time the views refresh before the first iteration.
        return prj

    from tomopy.prep.alignment import shift_images  # noqa: PLC0415

    # shift_images' first positional arg is applied as the row translation.
    return shift_images(prj, sy, sx)


@dataclass
class AlignConfig:
    recon_algorithm: str = "sirt"  # sirt | mlem | gridrec | art | fbp (tomopy names)
    recon_inner_iters: int = 2  # tomopy `num_iter` per OUTER iteration
    mode: str = "joint"  # "joint" (warm-start volume) | "sequential" (fresh)
    backend: str = "tomopy"
    upsample_factor: int = 20  # subpixel precision = 1/upsample px
    blur_edges: bool = True
    rin: float = 0.5
    rout: float = 0.8
    pad: tuple[int, int] = (0, 0)
    refine_center: bool = False  # re-run centre finding each iteration
    align_vertical: bool = True
    align_horizontal: bool = True
    shift_damping: float = 1.0  # multiply shift updates by this (< 1 stabilises)
    max_shift_per_iter: float | None = None  # clip outlier updates, in px
    median_filter_shifts: bool = False  # smooth shifts across angle
    median_filter_size: int = 3
    ncore: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentEngine:
    """Runs the reprojection alignment loop, one outer iteration per :meth:`step`.

    Qt-free by design: the GUI's worker thread drives this, and it is equally
    usable from a notebook.
    """

    dataset: Any  # tktomo.io.ProjectionData
    config: AlignConfig = field(default_factory=AlignConfig)
    sx0: np.ndarray | None = None
    sy0: np.ndarray | None = None
    center: float | None = None
    policy: VolumePolicy = field(default_factory=VolumePolicy)

    def __post_init__(self) -> None:
        original = np.asarray(self.dataset.data, dtype=np.float32)
        center = float(self.center) if self.center is not None else None

        pad_u, pad_v = self.config.pad
        if (pad_u, pad_v) != (0, 0):
            from tktomo.ptycho_align.core.preprocess import pad as pad_fn  # noqa: PLC0415

            original = pad_fn(original, pad_u, pad_v)
            # Padding inserts pad_u columns on the left, so every u coordinate --
            # including the rotation axis the caller measured on the unpadded data --
            # moves right by pad_u. The shifts are relative and need no adjustment.
            if center is not None:
                center += pad_u

        angles = np.asarray(self.dataset.angles, dtype=np.float64)
        n = original.shape[0]
        if center is None:
            center = original.shape[2] / 2.0

        self.state = AlignmentState(
            original=original,
            angles=angles,
            sx=np.zeros(n) if self.sx0 is None else np.asarray(self.sx0, dtype=np.float64).copy(),
            sy=np.zeros(n) if self.sy0 is None else np.asarray(self.sy0, dtype=np.float64).copy(),
            center=center,
            policy=self.policy,
            metadata=dict(getattr(self.dataset, "metadata", {})),
        )
        self._pending_config_change = False
        # step() already computes both of these; cache them so a display never has to
        # re-run tomopy. See `last_simulated` for why that matters beyond speed.
        self._last_aligned: np.ndarray | None = None
        self._last_simulated: np.ndarray | None = None
        self._best_residual = float("inf")
        self._warned_algorithm = False

    # -- accessors the GUI leans on -------------------------------------------------

    @property
    def last_aligned(self) -> np.ndarray | None:
        """The aligned stack from the most recent :meth:`step`, or None."""
        return self._last_aligned

    @property
    def last_simulated(self) -> np.ndarray | None:
        """The reprojection from the most recent :meth:`step`, or None.

        Callers displaying the reprojection must prefer this over calling
        :meth:`reproject` again. **TomoPy is not thread-safe**: ``tomopy.util.mproc``
        keeps its shared arrays in module-level globals, so a ``project()`` on one
        thread while ``recon()`` runs on another corrupts memory and segfaults. The
        GUI refreshes its views while the worker thread is mid-run, so it must read
        this cache rather than re-derive it.
        """
        return self._last_simulated

    def invalidate_cache(self) -> None:
        """Drop the cached aligned/simulated stacks (after a revert, or new shifts)."""
        self._last_aligned = None
        self._last_simulated = None

    @property
    def iteration(self) -> int:
        return self.state.iteration

    @property
    def history(self) -> list[IterationResult]:
        return self.state.history

    def set_config(self, config: AlignConfig) -> None:
        """Swap the config mid-run. Takes effect from the next iteration; the state
        is deliberately *not* reset, but the change is flagged so the history plot
        can mark the iteration where it happened."""
        self.config = config
        self._pending_config_change = True

    def aligned_projections(self) -> np.ndarray:
        """The measured stack with the current cumulative shift applied."""
        return apply_shifts(self.state.original.copy(), self.state.sy, self.state.sx)

    def reproject(self, volume: np.ndarray | None = None) -> np.ndarray:
        """Forward-project the current volume at the measured angles."""
        volume = self.state.volume if volume is None else volume
        if volume is None:
            raise ValueError("No volume yet -- run at least one iteration.")
        return self._reproject(volume)

    # -- the loop -------------------------------------------------------------------

    def step(self) -> IterationResult:
        """Run exactly ONE outer iteration."""
        from skimage.registration import phase_cross_correlation  # noqa: PLC0415

        cfg = self.config
        started = time.perf_counter()

        # 1. Aligned sinogram, always rebuilt from the pristine original.
        prj_aligned = apply_shifts(self.state.original.copy(), self.state.sy, self.state.sx)

        # 2. Reconstruct (warm-started in joint mode).
        volume = self._reconstruct(prj_aligned)

        # 3. Reproject at the measured angles.
        sim = self._reproject(volume)
        sim = _match_shape(sim, prj_aligned.shape)

        # 4. Taper the borders -- on copies used ONLY for registration. Blurring the
        #    data itself would destroy it.
        if cfg.blur_edges:
            from tomopy.prep.alignment import blur_edges  # noqa: PLC0415

            prj_reg = blur_edges(prj_aligned.copy(), cfg.rin, cfg.rout)
            sim_reg = blur_edges(sim.copy(), cfg.rin, cfg.rout)
        else:
            prj_reg, sim_reg = prj_aligned, sim

        # 5. Register measured against simulated. Reference = measured (see docstring):
        #    the returned shift is (d_row, d_col) = (dy, dx).
        dsy = np.zeros(prj_aligned.shape[0])
        dsx = np.zeros(prj_aligned.shape[0])
        for i in range(prj_aligned.shape[0]):
            shift, _error, _phase = phase_cross_correlation(
                prj_reg[i],
                sim_reg[i],
                upsample_factor=cfg.upsample_factor,
                normalization=None,
            )
            dsy[i], dsx[i] = float(shift[0]), float(shift[1])

        # 6. Condition the update.
        dsx, dsy = self._condition_update(dsx, dsy)

        # 7. Accumulate onto the cumulative shifts.
        sx = self.state.sx + dsx
        sy = self.state.sy + dsy

        # 8. Remove the degenerate global modes. A constant shift in either axis is a
        #    pure translation of the reconstructed volume, not a misalignment, and if
        #    left alone it random-walks over dozens of iterations. The rotation axis
        #    is NOT folded into sx -- it is owned by `center` and only `refine_center`
        #    touches it, so `sx` stays a pure zero-mean misalignment.
        sx = sx - sx.mean()
        sy = sy - sy.mean()

        # 9. Optionally re-estimate the rotation axis.
        center = self.state.center
        if cfg.refine_center:
            center = self._refine_center(prj_aligned, center)

        # 10. Metrics.
        error = float(np.sqrt(np.mean(dsx**2 + dsy**2)))
        denominator = float(np.linalg.norm(prj_aligned))
        residual = (
            float(np.linalg.norm(prj_aligned - sim) / denominator) if denominator else float("nan")
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
        )
        if result.diverging:
            logger.warning(
                "Iteration %d is DIVERGING: residual %.4g is far above the best so far "
                "(%.4g). Common causes: a residual phase ramp, an emission algorithm "
                "(mlem/osem) on data with negative values, or a bad rotation centre. "
                "Stop, revert to a good iteration, and fix the cause.",
                result.iteration,
                residual,
                self._best_residual,
            )
        self._pending_config_change = False
        self._last_aligned = prj_aligned
        self._last_simulated = sim

        logger.info(
            "iter %d: shift RMS %.4f px, residual %.4f, centre %.2f, %.1f s (%s/%s)",
            result.iteration,
            error,
            residual,
            center,
            result.wallclock_s,
            cfg.recon_algorithm,
            cfg.mode,
        )
        return self.state.record(result)

    def run(
        self,
        n: int,
        cancel_event: Any | None = None,
        callback: Callable[[IterationResult], None] | None = None,
    ) -> list[IterationResult]:
        """Call :meth:`step` ``n`` times, invoking ``callback`` after each.

        Aborts before the next iteration if ``cancel_event`` is set, so a cancelled
        run always leaves a complete, valid state.
        """
        results: list[IterationResult] = []
        for _ in range(n):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("run cancelled after %d iteration(s)", len(results))
                break
            result = self.step()
            results.append(result)
            if callback is not None:
                callback(result)
        return results

    # -- internals ------------------------------------------------------------------

    def _is_diverging(self, residual: float) -> bool:
        """A run whose residual climbs well past its own best is not converging.

        Iteration 1 always has a poor residual (the volume has barely formed), so only
        judge against the best seen so far, and only once there is a history to compare
        against.
        """
        if not np.isfinite(residual):
            return True
        diverging = self.state.history and residual > _DIVERGENCE_FACTOR * self._best_residual
        self._best_residual = min(self._best_residual, residual)
        return bool(diverging)

    def _backend(self):
        from tktomo.recon import get_backend  # noqa: PLC0415

        return get_backend(self.config.backend)

    def _reconstruct(self, prj_aligned: np.ndarray) -> np.ndarray:
        cfg = self.config
        backend = self._backend()

        kwargs: dict[str, Any] = {"algorithm": cfg.recon_algorithm, "center": self.state.center}
        if cfg.ncore is not None:
            kwargs["ncore"] = cfg.ncore

        reason = algorithm_rejects_negatives(cfg.recon_algorithm, prj_aligned)
        if reason and not self._warned_algorithm:
            logger.warning("%s", reason)
            self._warned_algorithm = True

        direct = cfg.recon_algorithm in _DIRECT_ALGORITHMS
        if not direct:
            kwargs["num_iter"] = cfg.recon_inner_iters
            if cfg.mode == "joint" and self.state.volume is not None:
                kwargs["init_recon"] = self.state.volume

        try:
            return _owned(backend.reconstruct(prj_aligned, self.state.angles, **kwargs))
        except (TypeError, ValueError) as exc:
            # e.g. gridrec rejects num_iter/init_recon. Degrade to a fresh
            # reconstruction rather than killing a long run.
            if direct:
                raise
            logger.warning(
                "%r rejected the iterative/warm-start arguments (%s); "
                "falling back to a fresh reconstruction.",
                cfg.recon_algorithm,
                exc,
            )
            return _owned(
                backend.reconstruct(
                    prj_aligned,
                    self.state.angles,
                    algorithm=cfg.recon_algorithm,
                    center=self.state.center,
                )
            )

    def _reproject(self, volume: np.ndarray) -> np.ndarray:
        kwargs: dict[str, Any] = {"center": self.state.center}
        if self.config.ncore is not None:
            kwargs["ncore"] = self.config.ncore
        simulated = self._backend().reproject(volume, self.state.angles, **kwargs)

        return _owned(simulated)

    def _condition_update(
        self, dsx: np.ndarray, dsy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        if not cfg.align_horizontal:
            dsx = np.zeros_like(dsx)
        if not cfg.align_vertical:
            dsy = np.zeros_like(dsy)

        if cfg.shift_damping != 1.0:
            dsx = dsx * cfg.shift_damping
            dsy = dsy * cfg.shift_damping

        if cfg.max_shift_per_iter is not None:
            limit = abs(cfg.max_shift_per_iter)
            dsx = np.clip(dsx, -limit, limit)
            dsy = np.clip(dsy, -limit, limit)

        if cfg.median_filter_shifts:
            from scipy.ndimage import median_filter  # noqa: PLC0415

            size = cfg.median_filter_size
            dsx = median_filter(dsx, size=size, mode="nearest")
            dsy = median_filter(dsy, size=size, mode="nearest")

        return dsx, dsy

    def _refine_center(self, prj_aligned: np.ndarray, fallback: float) -> float:
        from tktomo.ptycho_align.core.com import find_center  # noqa: PLC0415

        try:
            return find_center(prj_aligned, self.state.angles, method="vo")
        except Exception as exc:  # centre finding is a nicety, never fatal
            logger.warning("Centre refinement failed (%s); keeping centre %.2f", exc, fallback)
            return fallback


def _owned(array: np.ndarray) -> np.ndarray:
    """Return an array that owns its memory.

    ``tomopy.project`` hands back a **view** onto tomopy's multiprocessing
    shared-memory buffer (``owndata`` is False), and tomopy recycles that buffer on its
    next call. Anything still holding the view is then reading freed memory. The GUI
    caches these arrays and pyqtgraph paints them lazily, so the process segfaults
    inside a paint event -- intermittently, and nowhere near the real culprit. Copy on
    the way out of the engine and the whole class of bug disappears.
    """
    return array if array.flags.owndata else np.array(array, copy=True)


def _match_shape(sim: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Crop/pad the reprojection to the measured stack's shape.

    ``tomopy.project(pad=False)`` preserves the width today, but the guard is cheap
    and a silent one-pixel mismatch here would corrupt every registration.
    """
    if sim.shape == shape:
        return sim

    out = np.zeros(shape, dtype=sim.dtype)
    slices = []
    for extent, target in zip(sim.shape, shape):
        keep = min(extent, target)
        start = (extent - keep) // 2
        slices.append((start, start + keep, (target - keep) // 2))

    (a0, a1, d0), (b0, b1, d1), (c0, c1, d2) = slices
    out[d0 : d0 + (a1 - a0), d1 : d1 + (b1 - b0), d2 : d2 + (c1 - c0)] = sim[
        a0:a1, b0:b1, c0:c1
    ]
    logger.warning("Reprojection shape %s did not match %s; cropped/padded.", sim.shape, shape)
    return out

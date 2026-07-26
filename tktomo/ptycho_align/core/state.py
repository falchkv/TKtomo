"""Iteration bookkeeping for the reprojection alignment loop.

:class:`AlignmentState` owns the immutable preprocessed projections, the current
cumulative shifts, the current volume/centre, and the per-iteration history.

Memory policy: reconstructed volumes are large and a 50-iteration run at 1024**3
would exhaust RAM, so :class:`VolumePolicy` drops the volume from older
:class:`IterationResult` entries. Shift arrays are kept for *every* iteration --
they are a few kilobytes and are what "revert to iteration N" actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass
class IterationResult:
    """Everything one outer iteration produced."""

    iteration: int
    sx: np.ndarray  # cumulative horizontal shift per angle (pixels)
    sy: np.ndarray  # cumulative vertical shift per angle (pixels)
    dsx: np.ndarray  # shift *update* applied this iteration
    dsy: np.ndarray
    error: float  # RMS of the shift updates -> convergence metric
    residual: float  # ||measured - reprojected|| / ||measured||
    center: float
    wallclock_s: float
    volume: np.ndarray | None = None  # dropped for old iterations, see VolumePolicy
    config_changed: bool = False  # marks iterations where the user edited AlignConfig
    diverging: bool = False  # residual climbed well above the best seen so far
    runaway: str | None = None  # why this iteration's shift update is implausibly large

    @property
    def has_volume(self) -> bool:
        return self.volume is not None


@dataclass
class VolumePolicy:
    """Which iterations keep their reconstructed volume in memory.

    Keeps the last ``keep_last`` iterations plus every ``keep_every``-th one.
    ``keep_every=0`` disables the periodic keeps.
    """

    keep_last: int = 3
    keep_every: int = 10

    def keeps(self, iteration: int, latest: int) -> bool:
        if iteration > latest - self.keep_last:
            return True
        return self.keep_every > 0 and iteration % self.keep_every == 0

    def apply(self, history: list[IterationResult]) -> None:
        """Drop volumes from history entries the policy no longer keeps."""
        if not history:
            return
        latest = history[-1].iteration
        for result in history:
            if result.volume is not None and not self.keeps(result.iteration, latest):
                result.volume = None

    def estimate_bytes(self, volume_shape: tuple[int, ...], n_iterations: int) -> int:
        """Peak volume memory for a run of ``n_iterations``, for the GUI to show."""
        per_volume = int(np.prod(volume_shape)) * 4  # float32
        periodic = n_iterations // self.keep_every if self.keep_every > 0 else 0
        # +1 for the engine's own live (warm-start) volume.
        return per_volume * (self.keep_last + periodic + 1)


@dataclass
class AlignmentState:
    """The alignment's mutable state plus its full history."""

    original: np.ndarray  # preprocessed projections; NEVER modified in place
    angles: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    center: float
    volume: np.ndarray | None = None
    history: list[IterationResult] = field(default_factory=list)
    policy: VolumePolicy = field(default_factory=VolumePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def iteration(self) -> int:
        """Number of completed outer iterations (0 before the first step)."""
        return self.history[-1].iteration if self.history else 0

    def record(self, result: IterationResult) -> IterationResult:
        self.history.append(result)
        self.sx = result.sx.copy()
        self.sy = result.sy.copy()
        self.center = result.center
        if result.volume is not None:
            self.volume = result.volume
        self.policy.apply(self.history)
        return result

    def revert_to(self, iteration: int) -> None:
        """Roll the state back to the end of ``iteration`` (0 = before any step).

        Truncates the history. The volume is restored only if that iteration still
        holds one (see :class:`VolumePolicy`); otherwise it is cleared and the next
        step reconstructs from scratch, which is correct -- just not warm-started.
        """
        if iteration == 0:
            self.history = []
            self.sx = np.zeros_like(self.sx)
            self.sy = np.zeros_like(self.sy)
            self.volume = None
            return

        kept = [r for r in self.history if r.iteration <= iteration]
        if not kept or kept[-1].iteration != iteration:
            available = [r.iteration for r in self.history]
            raise ValueError(
                f"Cannot revert to iteration {iteration}; history holds {available}"
            )
        self.history = kept
        last = kept[-1]
        self.sx = last.sx.copy()
        self.sy = last.sy.copy()
        self.center = last.center
        self.volume = last.volume  # may be None if the policy dropped it

    def rescale_shifts(self, factor: float) -> None:
        """Scale the shifts when the binning factor changes.

        Going from bin 2 to bin 1 doubles every shift (they are in pixels of the
        binned grid). ``factor`` is old_bin / new_bin.
        """
        self.sx = self.sx * factor
        self.sy = self.sy * factor
        self.center = self.center * factor

    def snapshot(self) -> IterationResult | None:
        """The most recent result, with its volume detached (for compare views)."""
        if not self.history:
            return None
        return replace(self.history[-1])

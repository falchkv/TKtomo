"""Predicting what a run will cost, before committing to it.

An outer iteration can take tens of minutes, so "start 20 iterations" is a decision
worth pricing first -- both in wallclock and in RAM. None of this is Qt-aware: the
numbers are computed wherever the engine runs, which on a cluster is not where the
window is.
"""

from __future__ import annotations

from tktomo.ptycho_align.core.engine import DIRECT_ALGORITHMS

__all__ = ["format_bytes", "format_duration", "iteration_cost_units"]


def iteration_cost_units(
    n_angles: int, rows: int, width: int, algorithm: str, inner_iters: int
) -> float:
    """A relative measure of the work in one outer iteration.

    Both halves of an iteration -- reconstruct and reproject -- sweep every ray of every
    angle, so they scale as ``angles x rows x width^2``, with the iterative algorithms
    repeating the reconstruction ``inner_iters`` times (the direct ones ignore it). The
    **square** on width is the whole point: it is why a 30% pad quadruples the cost, and
    why bin 2 is worth roughly eightfold rather than twofold.

    The absolute constant depends on the machine and the algorithm, so it is not modelled
    -- it is measured, from the iterations this session has actually run.
    """
    repeats = 1 if algorithm in DIRECT_ALGORITHMS else max(1, inner_iters)
    return float(n_angles) * rows * width * width * repeats


def format_bytes(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
